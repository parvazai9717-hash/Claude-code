"""The agent runtime: an explicit PLAN → OBSERVE → ACT → VERIFY → FINISH machine.

The runtime is the only component that may execute anything. It receives
proposals from the model, validates them, asks the security layer, asks a human
where required, executes, gathers verification evidence, and decides whether to
continue, replan or stop.

It contains no provider-specific logic whatsoever: swapping Gemini for Ollama
changes nothing in this file.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Config
from .errors import (
    ErrorCategory,
    ProviderError,
)
from .events import Event, EventBus, EventType
from .memory.conversations import ConversationStore
from .memory.facts import FactStore
from .memory.summaries import SummaryStore, compact_conversation
from .memory.tasks import TaskStore
from .messages import Message, ModelResponse, ToolDefinition, ToolResult
from .prompts import (
    build_planning_hint,
    build_replan_hint,
    build_system_prompt,
    build_verification_hint,
)
from .providers.base import ModelProvider
from .security.approvals import Approver
from .security.limits import LimitTracker
from .security.paths import PathPolicy
from .security.permissions import PermissionChecker
from .security.redaction import Redactor
from .skills.registry import SkillRegistry
from .task_state import (
    ActionRecord,
    Phase,
    TaskState,
    TaskStatus,
    VerificationRecord,
)
from .tools.base import ToolContext
from .tools.registry import ToolRegistry

#: Tool names whose results count as observations rather than actions.
READ_ONLY_OBSERVATION_TOOLS = frozenset(
    {"get_current_time", "list_files", "read_file", "search_files"}
)

#: How many consecutive model turns with neither text nor tool calls are tolerated.
_MAX_EMPTY_TURNS = 2


@dataclass
class RunResult:
    """What a completed (or halted) run produced."""

    task: TaskState
    #: `completed`, `partial`, `failed`, `denied`, `cancelled`, `paused`, `unverified`.
    outcome: str
    text: str
    messages: list[Message] = field(default_factory=list)
    #: Side-effecting actions that succeeded but produced no verification evidence.
    #: Read-only actions are excluded: there is nothing to verify about a read.
    unverified: list[ActionRecord] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.outcome == "completed"

    def summary(self) -> str:
        """A human-readable close-out that distinguishes evidence from claims."""
        lines = [self.text.strip() or "(no final message)"]
        task = self.task
        verified = [v for v in task.verifications if v.verified]
        if verified:
            lines.append("\nVerified:")
            lines.extend(f"  - {v.evidence}" for v in verified[-5:])
        if self.unverified:
            lines.append("\nCompleted but NOT verified:")
            lines.extend(f"  - {a.tool_name}: {a.summary}" for a in self.unverified[-5:])
        denied = [a for a in task.actions if a.error_category is ErrorCategory.APPROVAL_DENIED]
        if denied:
            lines.append("\nDenied by the human:")
            lines.extend(f"  - {a.tool_name}" for a in denied)
        if task.failures:
            lines.append("\nFailures:")
            lines.extend(f"  - [{f.category.value}] {f.message}" for f in task.failures[-5:])
        return "\n".join(lines)


class AgentRunner:
    """Executes one task through the lifecycle state machine."""

    def __init__(
        self,
        *,
        config: Config,
        provider: ModelProvider,
        tools: ToolRegistry,
        events: EventBus | None = None,
        approver: Approver | None = None,
        conversations: ConversationStore | None = None,
        task_store: TaskStore | None = None,
        fact_store: FactStore | None = None,
        summary_store: SummaryStore | None = None,
        skills: SkillRegistry | None = None,
        redactor: Redactor | None = None,
    ) -> None:
        self.config = config
        self.provider = provider
        self.tools = tools
        self.events = events or EventBus()
        self.redactor = redactor or Redactor()
        self.conversations = conversations
        self.task_store = task_store
        self.fact_store = fact_store
        self.summary_store = summary_store
        self.skills = skills
        if approver is not None:
            self.tools.approver = approver

        workspace = Path(config.workspace).expanduser()
        self.paths = PathPolicy.for_workspace(
            workspace, max_file_bytes=config.limits.max_file_bytes
        )
        self.permissions = PermissionChecker(config)
        self.limits = LimitTracker(limits=config.limits)
        self._cancel_event = asyncio.Event()
        self._pause_event = asyncio.Event()
        #: Working memory for the current run.
        self.messages: list[Message] = []

    # -- control ------------------------------------------------------------
    def request_cancel(self, reason: str = "cancelled by the user") -> None:
        """Ask the run to stop. Honoured before the next consequential action."""
        self._cancel_reason = reason
        self._cancel_event.set()

    def request_pause(self) -> None:
        """Ask the run to pause. Honoured before the next consequential action."""
        self._pause_event.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel_event.is_set()

    @property
    def paused(self) -> bool:
        return self._pause_event.is_set()

    # -- setup --------------------------------------------------------------
    def tool_definitions(self) -> list[ToolDefinition]:
        return self.tools.definitions()

    def build_system_message(self) -> Message:
        """Compose the system prompt from live configuration, facts and skills."""
        definitions = self.tool_definitions()
        facts_block = self.fact_store.prompt_block() if self.fact_store else ""
        skills_block = ""
        if self.skills is not None:
            skills_block = self.skills.prompt_block(
                available_tools=self.tools.names(), config=self.config
            )
        return Message.system(
            build_system_prompt(
                self.config,
                definitions,
                facts_block=facts_block,
                skills_block=skills_block,
            )
        )

    def _context(self, task: TaskState) -> ToolContext:
        return ToolContext(
            config=self.config,
            paths=self.paths,
            redactor=self.redactor,
            task_id=task.id,
            step=task.current_step,
            store=self.fact_store,
        )

    # -- main loop ----------------------------------------------------------
    async def run(self, task: TaskState, *, history: list[Message] | None = None) -> RunResult:
        """Drive `task` to a terminal state (or to a clean pause).

        The loop is the state machine: each iteration is one step, and each step
        moves through OBSERVE → ACT → VERIFY before deciding whether to continue.
        """
        task.provider = getattr(self.provider, "name", self.config.provider)
        task.model = getattr(self.provider, "model", self.config.active_model)
        task.workspace = str(self.paths.root)

        self.messages = list(history) if history else []
        if not any(m.role == "system" for m in self.messages):
            self.messages.insert(0, self.build_system_message())
        if not history:
            self.messages.append(Message.user(build_planning_hint(task.goal)))

        if task.status is TaskStatus.PAUSED:
            # Resuming: clear the stale flags that caused the pause.
            task.pause_requested = False
            self._pause_event.clear()
        task.transition_to(TaskStatus.RUNNING)
        self._persist(task)
        self._emit(EventType.TASK_STARTED, task, message=f"task started: {task.goal[:120]}")

        task.phase = Phase.PLAN
        self._emit(EventType.PHASE_ENTERED, task, phase=Phase.PLAN.value)

        empty_turns = 0
        try:
            while True:
                stop = self._check_control(task)
                if stop is not None:
                    return stop

                limit_check = self.limits.start_step(task.current_step + 1)
                if not limit_check.ok:
                    return self._halt(
                        task,
                        "partial",
                        f"Stopped: {limit_check.message}. "
                        "The work so far is described above; raise `max_steps` to continue.",
                        EventType.LIMIT_REACHED,
                    )
                task.current_step += 1
                self._emit(EventType.STEP_STARTED, task, message=f"step {task.current_step}")

                # --- OBSERVE / ACT: ask the model what to do next ------------
                task.phase = Phase.OBSERVE
                self._compact_if_needed(task)
                try:
                    response = await self._generate()
                except ProviderError as exc:
                    handled = self._handle_provider_error(task, exc)
                    if handled is not None:
                        return handled
                    continue

                self.messages.append(Message.assistant(response.text or "", response.tool_calls))

                if not response.has_tool_calls:
                    if not (response.text or "").strip():
                        empty_turns += 1
                        if empty_turns >= _MAX_EMPTY_TURNS:
                            return self._halt(
                                task,
                                "failed",
                                "The model returned no content and no action twice in a row.",
                                EventType.ERROR,
                            )
                        self.messages.append(
                            Message.user(
                                "You returned nothing. Either take an action with a tool "
                                "or give your final answer."
                            )
                        )
                        continue
                    empty_turns = 0
                    return self._finish(task, response.text or "")

                empty_turns = 0
                task.phase = Phase.ACT
                self._emit(EventType.PHASE_ENTERED, task, phase=Phase.ACT.value)
                should_stop = await self._execute_calls(task, response)
                if should_stop is not None:
                    return should_stop
        except asyncio.CancelledError:
            return self._halt(task, "cancelled", "The run was cancelled.", EventType.TASK_FINISHED)
        finally:
            self._persist(task)

    # -- steps --------------------------------------------------------------
    async def _generate(self) -> ModelResponse:
        """One provider call, with the configured timeout applied by the adapter."""
        self._emit_raw(
            EventType.PROVIDER_REQUEST,
            message=f"asking {getattr(self.provider, 'name', '?')}",
            data={"messages": len(self.messages)},
        )
        response = await self.provider.generate(
            self.messages,
            self.tool_definitions(),
            temperature=0.2,
            max_output_tokens=self.config.limits.max_model_output_tokens,
        )
        self._emit_raw(
            EventType.PROVIDER_RESPONSE,
            message=f"finish reason: {response.finish_reason.value}",
            data={
                "tool_calls": [c.name for c in response.tool_calls],
                "has_text": bool(response.text),
            },
        )
        return response

    async def _execute_calls(self, task: TaskState, response: ModelResponse) -> RunResult | None:
        """Run every tool call in a model turn. Returns a result only if the run ends."""
        for call in response.tool_calls:
            stop = self._check_control(task, before_action=True)
            if stop is not None:
                return stop

            budget = self.limits.check_tool_call()
            if not budget.ok:
                self._emit(EventType.LIMIT_REACHED, task, message=budget.message)
                return self._halt(
                    task,
                    "partial",
                    f"Stopped: {budget.message}.",
                    EventType.LIMIT_REACHED,
                )

            definition = self.tools.get(call.name).definition() if call.name in self.tools else None

            # A repeatedly failing identical action is stopped rather than retried.
            signature = f"{call.name}:{sorted(call.arguments.items())}"
            attempts = task.retries_for(signature)
            retry_check = self.limits.check_retry(attempts)
            if not retry_check.ok:
                result = ToolResult(
                    call_id=call.id,
                    tool_name=call.name,
                    ok=False,
                    error_category=ErrorCategory.LIMIT_EXCEEDED,
                    error_message=(f"{retry_check.message}; try a different approach or stop"),
                )
            else:
                self.limits.consume_tool_call()
                result = await self.tools.execute(call, self._context(task))

            if not result.ok:
                task.note_retry(signature)

            action = TaskStore.action_from_result(
                result, task.current_step, call.arguments, approved=None
            )
            task.record_action(action)
            self.messages.append(Message.from_tool_result(result))

            if self.task_store is not None:
                self.task_store.record_tool_result(
                    result,
                    task_id=task.id,
                    session_id=task.session_id,
                    arguments=call.arguments,
                )

            if result.ok and call.name in READ_ONLY_OBSERVATION_TOOLS:
                task.record_observation(call.name, self._observation_summary(call.name, result))

            # --- VERIFY -------------------------------------------------
            if call.name == "verify_result":
                self._record_verification(task, result)
            elif result.ok and definition is not None and definition.requires_verification:
                task.phase = Phase.VERIFY
                self._emit(EventType.PHASE_ENTERED, task, phase=Phase.VERIFY.value)
                self.messages.append(
                    Message.user(
                        build_verification_hint(f"{call.name} on step {task.current_step}")
                    )
                )

            # --- REPLAN -------------------------------------------------
            if not result.ok:
                stop = self._handle_failure(task, result)
                if stop is not None:
                    return stop
        self._persist(task)
        return None

    def _record_verification(self, task: TaskState, result: ToolResult) -> None:
        """Turn a `verify_result` call into a recorded verification."""
        output = result.output if result.ok else {}
        verified = bool(output.get("verified", False))
        evidence = str(output.get("evidence", "") or "")
        unavailable = str(output.get("unavailable_reason", "") or "")
        # Attribute the evidence to the most recent successful consequential action.
        target = next(
            (a for a in reversed(task.actions) if a.ok and a.tool_name != "verify_result"),
            None,
        )
        record = VerificationRecord(
            step=task.current_step,
            action_id=target.id if target else None,
            method=str(output.get("method", "unknown")),
            verified=verified and result.ok,
            evidence=evidence,
            detail=unavailable or (result.error_message or ""),
        )
        task.record_verification(record)
        if self.task_store is not None:
            self.task_store.record_verification(record, task.id)
        self._emit(
            EventType.VERIFICATION_RESULT,
            task,
            message=("verified: " + evidence) if record.verified else "verification failed",
            data={"verified": record.verified, "method": record.method},
        )
        if not record.verified:
            task.record_failure(
                ErrorCategory.VERIFICATION_FAILED
                if not unavailable
                else ErrorCategory.VERIFICATION_UNAVAILABLE,
                evidence or unavailable or "verification did not produce evidence",
            )
            task.phase = Phase.REPLAN
            task.replan_count += 1
            self._emit(EventType.REPLAN, task, message="verification failed; replanning")
            self.messages.append(
                Message.user(
                    build_replan_hint(
                        f"verification failed — {evidence or unavailable}",
                        max(0, self.config.limits.max_retries_per_action),
                    )
                )
            )

    def _handle_failure(self, task: TaskState, result: ToolResult) -> RunResult | None:
        """Record a failed tool call and prompt a replan, or stop if bounded out."""
        category = result.error_category or ErrorCategory.TOOL_FAILED
        message = result.error_message or "the tool failed"
        task.record_failure(category, message, tool_name=result.tool_name)

        if category is ErrorCategory.CANCELLED:
            return self._halt(
                task,
                "cancelled",
                "The run was cancelled at an approval prompt.",
                EventType.TASK_FINISHED,
            )

        replan_check = self.limits.check_replan(task.replan_count)
        if not replan_check.ok:
            return self._halt(
                task,
                "partial",
                f"Stopped: {replan_check.message}. The last failure was: {message}",
                EventType.LIMIT_REACHED,
            )

        task.phase = Phase.REPLAN
        task.replan_count += 1
        self._emit(
            EventType.REPLAN,
            task,
            message=f"{result.tool_name} failed ({category.value}); replanning",
            data={"error_category": category.value},
        )
        attempts_left = max(
            0,
            self.config.limits.max_retries_per_action
            - task.retries_for(f"{result.tool_name}:{sorted(())}"),
        )
        self.messages.append(
            Message.user(build_replan_hint(f"{result.tool_name} — {message}", attempts_left or 1))
        )
        return None

    def _handle_provider_error(self, task: TaskState, exc: ProviderError) -> RunResult | None:
        """A provider failure ends the run: the model is the one component with no fallback."""
        task.record_failure(exc.category, exc.message)
        self._emit(
            EventType.PROVIDER_ERROR,
            task,
            message=exc.message,
            data={"category": exc.category.value},
        )
        return self._halt(
            task,
            "failed",
            f"The model provider failed and the run cannot continue: {exc.message}",
            EventType.ERROR,
        )

    # -- control checks -----------------------------------------------------
    def _check_control(self, task: TaskState, *, before_action: bool = False) -> RunResult | None:
        """Honour pause and cancel requests at a safe point."""
        if self._cancel_event.is_set() or task.cancel_requested:
            reason = getattr(self, "_cancel_reason", "") or task.cancel_reason
            return self._halt(
                task,
                "cancelled",
                f"The run was cancelled{f': {reason}' if reason else '.'}",
                EventType.TASK_FINISHED,
                status=TaskStatus.CANCELLED,
            )
        if self._pause_event.is_set() or task.pause_requested:
            # Pausing stops *before* the next consequential action, never mid-write.
            task.pause_requested = True
            return self._halt(
                task,
                "paused",
                "The run is paused before the next action. Resume it with "
                f"`local-agent task resume {task.id}`.",
                EventType.TASK_STATUS_CHANGED,
                status=TaskStatus.PAUSED,
            )
        return None

    # -- termination --------------------------------------------------------
    def _finish(self, task: TaskState, text: str) -> RunResult:
        """The model gave a final answer. Decide how honest a claim it can make."""
        task.phase = Phase.FINISH
        consequential_unverified = self._consequential_unverified(task)
        if consequential_unverified:
            outcome = "unverified"
        elif task.failures:
            outcome = "partial"
        else:
            outcome = "completed"

        task.result = text
        task.outcome = outcome
        task.transition_to(TaskStatus.COMPLETED)
        self._persist(task)
        self._emit(
            EventType.FINAL_ANSWER,
            task,
            message="final answer produced",
            data={"outcome": outcome},
        )
        self._emit(EventType.TASK_FINISHED, task, message=f"task finished: {outcome}")
        return RunResult(
            task=task,
            outcome=outcome,
            text=text,
            messages=list(self.messages),
            unverified=consequential_unverified,
        )

    def _halt(
        self,
        task: TaskState,
        outcome: str,
        text: str,
        event_type: EventType,
        status: TaskStatus | None = None,
    ) -> RunResult:
        """Stop the run without a model-produced final answer."""
        task.result = text
        task.outcome = outcome
        target = status or {
            "cancelled": TaskStatus.CANCELLED,
            "paused": TaskStatus.PAUSED,
            "failed": TaskStatus.FAILED,
        }.get(outcome, TaskStatus.COMPLETED)
        if task.can_transition_to(target):
            task.transition_to(target)
        self._persist(task)
        self._emit(event_type, task, message=text[:200], data={"outcome": outcome})
        self._emit(EventType.TASK_FINISHED, task, message=f"task finished: {outcome}")
        return RunResult(
            task=task,
            outcome=outcome,
            text=text,
            messages=list(self.messages),
            unverified=self._consequential_unverified(task),
        )

    # -- support ------------------------------------------------------------
    def _compact_if_needed(self, task: TaskState) -> None:
        """Fold old turns into a summary when the conversation grows too long."""
        limit = self.config.limits.max_conversation_messages
        if len(self.messages) <= limit:
            return
        compacted, summary = compact_conversation(self.messages, limit)
        if summary is None:
            return
        self.messages = compacted
        if self.summary_store is not None:
            self.summary_store.add(task.session_id, summary, len(self.messages))
        self._emit(
            EventType.MESSAGE,
            task,
            message="compacted the conversation to stay within the context limit",
        )

    def _consequential_unverified(self, task: TaskState) -> list[ActionRecord]:
        """Successful side-effecting actions with no verification evidence.

        A read is not a claim about the world, so read-only tools are excluded:
        listing them would dilute the one signal that matters here.
        """
        return [
            action
            for action in task.unverified_actions()
            if action.tool_name in self.tools
            and not self.tools.get(action.tool_name).definition().read_only
        ]

    @staticmethod
    def _observation_summary(tool_name: str, result: ToolResult) -> str:
        output = result.output
        if tool_name == "list_files":
            return f"listed {output.get('count', 0)} entries under {output.get('root', '.')}"
        if tool_name == "read_file":
            return (
                f"read {output.get('path', '?')} "
                f"({output.get('returned_lines', 0)} of {output.get('total_lines', 0)} lines)"
            )
        if tool_name == "search_files":
            return (
                f"searched for {output.get('query', '')!r}: "
                f"{output.get('match_count', 0)} match(es)"
            )
        if tool_name == "get_current_time":
            return f"current time is {output.get('local', '?')}"
        return f"{tool_name} returned a result"

    def _persist(self, task: TaskState) -> None:
        if self.task_store is not None:
            self.task_store.save(task)

    def _emit(
        self,
        event_type: EventType,
        task: TaskState,
        *,
        message: str = "",
        data: dict[str, Any] | None = None,
        phase: str | None = None,
    ) -> Event:
        return self.events.emit_event(
            event_type,
            task_id=task.id,
            session_id=task.session_id,
            step=task.current_step,
            phase=phase or task.phase.value,
            provider=task.provider,
            model=task.model,
            message=message,
            data=data or {},
        )

    def _emit_raw(
        self, event_type: EventType, *, message: str = "", data: dict[str, Any] | None = None
    ) -> Event:
        return self.events.emit_event(event_type, message=message, data=data or {})


def build_runner(
    config: Config,
    *,
    provider: ModelProvider,
    approver: Approver,
    events: EventBus | None = None,
    database: Any = None,
    include_shell: bool | None = None,
) -> AgentRunner:
    """Assemble a fully wired runner.

    This is the composition root: it is the one place that knows how the config,
    the security layer, the tools, the stores and the provider fit together.
    """
    from .memory.database import Database
    from .tools import build_default_tools, initialize_workspace

    redactor = Redactor()
    bus = events or EventBus(redactor=redactor.redact)
    initialize_workspace(config.workspace)

    conversations = task_store = fact_store = summary_store = None
    if database is not None:
        if not isinstance(database, Database):
            database = Database(database)
        conversations = ConversationStore(database)
        task_store = TaskStore(database)
        fact_store = FactStore(database, max_facts=config.limits.max_persisted_facts)
        summary_store = SummaryStore(database)
        bus.subscribe(task_store.event_writer())

    registry = ToolRegistry(permissions=PermissionChecker(config), approver=approver, events=bus)
    registry.register_all(
        build_default_tools(
            include_shell=config.shell_enabled if include_shell is None else include_shell,
            include_memory=fact_store is not None,
        )
    )

    skills = SkillRegistry.from_directory(config.skills_dir)

    return AgentRunner(
        config=config,
        provider=provider,
        tools=registry,
        events=bus,
        approver=approver,
        conversations=conversations,
        task_store=task_store,
        fact_store=fact_store,
        summary_store=summary_store,
        skills=skills,
        redactor=redactor,
    )
