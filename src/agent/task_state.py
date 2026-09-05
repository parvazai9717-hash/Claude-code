"""Explicit task state and its transition rules.

The runtime is a state machine, not a free-running loop. `TaskState` is the
serialisable record of where a task is, what it has done, and what evidence it
has for those claims. It is persisted to SQLite so a task survives a restart.

It deliberately stores **no hidden chain-of-thought** — only concise plans,
observations, actions, verification evidence, errors and outcomes.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from .errors import AgentError, ErrorCategory
from .messages import new_id


def _now() -> datetime:
    return datetime.now(UTC)


class TaskStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


#: Terminal statuses: a task in one of these never runs again.
TERMINAL_STATUSES: frozenset[TaskStatus] = frozenset(
    {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}
)

#: Allowed transitions. Anything not listed here is rejected.
ALLOWED_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.CREATED: frozenset({TaskStatus.RUNNING, TaskStatus.CANCELLED, TaskStatus.FAILED}),
    TaskStatus.RUNNING: frozenset(
        {
            TaskStatus.WAITING_FOR_APPROVAL,
            TaskStatus.PAUSED,
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }
    ),
    TaskStatus.WAITING_FOR_APPROVAL: frozenset(
        {TaskStatus.RUNNING, TaskStatus.PAUSED, TaskStatus.CANCELLED, TaskStatus.FAILED}
    ),
    TaskStatus.PAUSED: frozenset({TaskStatus.RUNNING, TaskStatus.CANCELLED, TaskStatus.FAILED}),
    TaskStatus.COMPLETED: frozenset(),
    TaskStatus.FAILED: frozenset(),
    TaskStatus.CANCELLED: frozenset(),
}


class InvalidTransitionError(AgentError):
    """A status change that the state machine forbids."""

    category = ErrorCategory.UNKNOWN


class Phase(StrEnum):
    """The lifecycle phase the runtime is currently in."""

    PLAN = "plan"
    OBSERVE = "observe"
    ACT = "act"
    VERIFY = "verify"
    REPLAN = "replan"
    FINISH = "finish"


class PlanStep(BaseModel):
    """One concise checklist entry. Not reasoning — an intended outcome."""

    id: str = Field(default_factory=lambda: new_id("step"))
    description: str
    done: bool = False
    note: str = ""


class Plan(BaseModel):
    steps: list[PlanStep] = Field(default_factory=list)
    revision: int = 0
    updated_at: datetime = Field(default_factory=_now)

    @property
    def open_steps(self) -> list[PlanStep]:
        return [s for s in self.steps if not s.done]

    def summary(self) -> str:
        if not self.steps:
            return "(no plan)"
        return "\n".join(
            f"{i + 1}. [{'x' if s.done else ' '}] {s.description}" for i, s in enumerate(self.steps)
        )


class ObservationRecord(BaseModel):
    """Something the agent learned with a read-only action."""

    id: str = Field(default_factory=lambda: new_id("obs"))
    step: int
    source: str
    summary: str
    created_at: datetime = Field(default_factory=_now)


class ActionRecord(BaseModel):
    """A tool call the runtime actually attempted, and how it went."""

    id: str = Field(default_factory=lambda: new_id("act"))
    step: int
    call_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    ok: bool = False
    approved: bool | None = None
    error_category: ErrorCategory | None = None
    summary: str = ""
    duration_ms: int = 0
    created_at: datetime = Field(default_factory=_now)

    def signature(self) -> str:
        """A stable key used to detect an identical repeated failing action."""
        import json

        return f"{self.tool_name}:{json.dumps(self.arguments, sort_keys=True, default=str)}"


class VerificationRecord(BaseModel):
    """Evidence — or a clear statement that evidence was unavailable."""

    id: str = Field(default_factory=lambda: new_id("ver"))
    step: int
    action_id: str | None = None
    method: str
    verified: bool
    evidence: str = ""
    detail: str = ""
    created_at: datetime = Field(default_factory=_now)


class FailureRecord(BaseModel):
    id: str = Field(default_factory=lambda: new_id("fail"))
    step: int
    category: ErrorCategory
    message: str
    tool_name: str | None = None
    created_at: datetime = Field(default_factory=_now)


class TaskState(BaseModel):
    """The full, serialisable state of one task."""

    id: str = Field(default_factory=lambda: new_id("task"))
    session_id: str = Field(default_factory=lambda: new_id("sess"))
    goal: str
    status: TaskStatus = TaskStatus.CREATED
    phase: Phase = Phase.PLAN

    plan: Plan = Field(default_factory=Plan)
    current_step: int = 0
    completed_steps: list[str] = Field(default_factory=list)

    observations: list[ObservationRecord] = Field(default_factory=list)
    actions: list[ActionRecord] = Field(default_factory=list)
    verifications: list[VerificationRecord] = Field(default_factory=list)
    failures: list[FailureRecord] = Field(default_factory=list)

    provider: str = ""
    model: str = ""
    workspace: str = ""

    total_tool_calls: int = 0
    replan_count: int = 0
    #: Per-action-signature retry counters, used to bound repeated failures.
    retry_counts: dict[str, int] = Field(default_factory=dict)

    cancel_requested: bool = False
    pause_requested: bool = False
    cancel_reason: str = ""

    result: str | None = None
    #: `completed`, `partial`, `failed`, `denied`, `cancelled`, `unverified`.
    outcome: str | None = None

    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None

    # -- transitions --------------------------------------------------------
    def can_transition_to(self, status: TaskStatus) -> bool:
        if status == self.status:
            return True
        return status in ALLOWED_TRANSITIONS[self.status]

    def transition_to(self, status: TaskStatus) -> None:
        """Move to `status`, or raise :class:`InvalidTransitionError`."""
        if status == self.status:
            return
        if not self.can_transition_to(status):
            raise InvalidTransitionError(
                f"cannot move task from {self.status.value} to {status.value}",
                details={"from": self.status.value, "to": status.value},
            )
        self.status = status
        self.updated_at = _now()
        if status == TaskStatus.RUNNING and self.started_at is None:
            self.started_at = self.updated_at
        if status in TERMINAL_STATUSES:
            self.finished_at = self.updated_at

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def is_resumable(self) -> bool:
        return self.status in {TaskStatus.PAUSED, TaskStatus.CREATED, TaskStatus.RUNNING}

    # -- recording ----------------------------------------------------------
    def record_observation(self, source: str, summary: str) -> ObservationRecord:
        record = ObservationRecord(step=self.current_step, source=source, summary=summary)
        self.observations.append(record)
        self.updated_at = _now()
        return record

    def record_action(self, record: ActionRecord) -> ActionRecord:
        self.actions.append(record)
        self.total_tool_calls += 1
        self.updated_at = _now()
        return record

    def record_verification(self, record: VerificationRecord) -> VerificationRecord:
        self.verifications.append(record)
        self.updated_at = _now()
        return record

    def record_failure(
        self, category: ErrorCategory, message: str, tool_name: str | None = None
    ) -> FailureRecord:
        record = FailureRecord(
            step=self.current_step, category=category, message=message, tool_name=tool_name
        )
        self.failures.append(record)
        self.updated_at = _now()
        return record

    def note_retry(self, signature: str) -> int:
        """Increment and return the retry count for an action signature."""
        count = self.retry_counts.get(signature, 0) + 1
        self.retry_counts[signature] = count
        self.updated_at = _now()
        return count

    def retries_for(self, signature: str) -> int:
        return self.retry_counts.get(signature, 0)

    def set_plan(self, descriptions: list[str]) -> Plan:
        self.plan = Plan(
            steps=[PlanStep(description=d) for d in descriptions],
            revision=self.plan.revision + 1,
        )
        self.updated_at = _now()
        return self.plan

    # -- reporting ----------------------------------------------------------
    def unverified_actions(self) -> list[ActionRecord]:
        """Successful, consequential actions with no verification evidence."""
        verified_ids = {v.action_id for v in self.verifications if v.verified}
        return [a for a in self.actions if a.ok and a.id not in verified_ids]

    def context_summary(self) -> str:
        """A compact, model-safe status block appended to the prompt."""
        lines = [
            f"Task status: {self.status.value} "
            f"(phase: {self.phase.value}, step {self.current_step})",
            f"Plan:\n{self.plan.summary()}",
        ]
        if self.observations:
            recent = self.observations[-3:]
            lines.append("Recent observations:\n" + "\n".join(f"- {o.summary}" for o in recent))
        if self.failures:
            recent_failures = self.failures[-3:]
            lines.append(
                "Recent failures:\n"
                + "\n".join(f"- [{f.category.value}] {f.message}" for f in recent_failures)
            )
        if self.verifications:
            last = self.verifications[-1]
            lines.append(
                f"Last verification: {'passed' if last.verified else 'FAILED'} "
                f"via {last.method} — {last.evidence or last.detail}"
            )
        return "\n\n".join(lines)
