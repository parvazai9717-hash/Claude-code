"""Command-line interface.

Every command builds the same composition root (`build_runner`), so the CLI has
no privileged path into the runtime: what `chat` can do, `task run` can do, and
neither can bypass the security layer.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import signal
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .config import ApprovalMode, Config, load_config
from .errors import AgentError
from .events import Event, EventBus, EventType, JsonlEventWriter, configure_logging
from .manage import AgentManager
from .memory.conversations import ConversationStore
from .memory.database import Database
from .memory.facts import FactStore
from .memory.tasks import TaskStore
from .messages import Message
from .providers.factory import KNOWN_PROVIDERS, create_provider
from .runtime import AgentRunner, RunResult, build_runner
from .security.approvals import ConsoleApprover, UnsafeAutoApprover
from .security.redaction import Redactor
from .skills.registry import SkillRegistry
from .task_state import TaskState, TaskStatus
from .tools import initialize_workspace

console = Console()

app = typer.Typer(
    name="local-agent",
    help="A local-first autonomous agent with a secure tool layer.",
    no_args_is_help=True,
    add_completion=False,
)
task_app = typer.Typer(name="task", help="Create, run and control tasks.", no_args_is_help=True)
memory_app = typer.Typer(name="memory", help="Inspect durable memory.", no_args_is_help=True)
sessions_app = typer.Typer(
    name="sessions", help="Inspect conversation history.", no_args_is_help=True
)
skills_app = typer.Typer(name="skills", help="Inspect available skills.", no_args_is_help=True)
connectors_app = typer.Typer(
    name="connectors",
    help="Add and manage MCP servers and other tool connectors.",
    no_args_is_help=True,
)
app.add_typer(task_app)
app.add_typer(memory_app)
app.add_typer(sessions_app)
app.add_typer(skills_app)
app.add_typer(connectors_app)


# --------------------------------------------------------------------------
# Shared plumbing
# --------------------------------------------------------------------------
class Context:
    """Everything a command needs, built once from global options."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.database = Database(config.database_path)
        self.redactor = Redactor()
        self.events = EventBus(redactor=self.redactor.redact)
        self.events.subscribe(JsonlEventWriter(config.log_path))

    def close(self) -> None:
        self.database.close()


#: Populated by the root callback and consumed by every subcommand.
_state: dict[str, Any] = {}


def get_context() -> Context:
    context = _state.get("context")
    if context is None:
        raise typer.Exit(code=2)
    return context


def _manager(context: Context) -> AgentManager:
    """The same management API a UI would use. The CLI is only a renderer over it."""
    return AgentManager(config=context.config, database=context.database)


def _fail(message: str, code: int = 1) -> None:
    console.print(f"[red]Error:[/red] {message}")
    raise typer.Exit(code=code)


def _handle(exc: Exception) -> None:
    if isinstance(exc, AgentError):
        console.print(f"[red]{exc.category.value}:[/red] {exc.message}")
        raise typer.Exit(code=1)
    raise exc


def _progress_handler(verbose: bool) -> Any:
    """Render lifecycle events without exposing private reasoning."""
    icons = {
        EventType.STEP_STARTED: ("·", "dim"),
        EventType.TOOL_REQUESTED: ("→", "cyan"),
        EventType.TOOL_RESULT: ("←", "cyan"),
        EventType.APPROVAL_REQUESTED: ("?", "yellow"),
        EventType.VERIFICATION_RESULT: ("✓", "green"),
        EventType.REPLAN: ("↻", "yellow"),
        EventType.LIMIT_REACHED: ("!", "yellow"),
        EventType.PROVIDER_ERROR: ("✗", "red"),
        EventType.ERROR: ("✗", "red"),
    }
    shown = (
        set(icons)
        if verbose
        else {
            EventType.TOOL_REQUESTED,
            EventType.TOOL_RESULT,
            EventType.VERIFICATION_RESULT,
            EventType.REPLAN,
            EventType.LIMIT_REACHED,
            EventType.PROVIDER_ERROR,
            EventType.ERROR,
        }
    )

    def handle(event: Event) -> None:
        if event.type not in shown:
            return
        icon, style = icons.get(event.type, ("·", "dim"))
        console.print(f"[{style}]{icon}[/{style}] {event.message}")

    return handle


def _build_runner(context: Context, *, unsafe: bool = False) -> AgentRunner:
    provider = create_provider(context.config)
    if context.config.unsafe_disable_approvals or unsafe:
        console.print(
            "[bold red]WARNING:[/bold red] approvals are disabled. Every action will run "
            "without confirmation. This is unsafe and is intended for testing only."
        )
        approver: Any = UnsafeAutoApprover()
    else:
        approver = ConsoleApprover(console=console, redactor=context.redactor)
    return build_runner(
        context.config,
        provider=provider,
        approver=approver,
        events=context.events,
        database=context.database,
    )


# --------------------------------------------------------------------------
# Root
# --------------------------------------------------------------------------
def _version_callback(value: bool) -> None:
    if value:
        console.print(f"local-agent {__version__}")
        raise typer.Exit()


@app.callback()
def main_callback(
    config_file: Path | None = typer.Option(
        None, "--config", "-c", help="Path to config.yaml or config.toml."
    ),
    provider: str | None = typer.Option(
        None, "--provider", "-p", help=f"Override the provider: {', '.join(KNOWN_PROVIDERS)}."
    ),
    model: str | None = typer.Option(None, "--model", "-m", help="Override the model name."),
    workspace: Path | None = typer.Option(
        None, "--workspace", "-w", help="Override the workspace."
    ),
    approval_mode: str | None = typer.Option(
        None, "--approval", "-a", help="Override approval mode: always, risky, automatic."
    ),
    log_level: str | None = typer.Option(None, "--log-level", help="DEBUG, INFO, WARNING, ERROR."),
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, help="Show the version."
    ),
) -> None:
    """Set up configuration shared by every subcommand."""
    try:
        config = load_config(
            config_file,
            provider=provider,
            model=model,
            workspace=workspace,
            approval_mode=ApprovalMode(approval_mode) if approval_mode else None,
            log_level=log_level,
        )
    except ValueError as exc:
        _fail(str(exc))
        return
    except AgentError as exc:
        _fail(exc.message)
        return

    configure_logging(config.log_level)
    config.ensure_directories()
    _state["context"] = Context(config)


# --------------------------------------------------------------------------
# doctor / config
# --------------------------------------------------------------------------
@app.command()
def doctor() -> None:
    """Check configuration, workspace, database, provider and model availability."""
    context = get_context()
    config = context.config
    table = Table(title="local-agent doctor", show_lines=False)
    table.add_column("Check", style="bold")
    table.add_column("Status")
    table.add_column("Detail", overflow="fold")

    def row(name: str, ok: bool | None, detail: str) -> None:
        mark = (
            "[green]ok[/green]"
            if ok
            else ("[red]fail[/red]" if ok is False else "[yellow]warn[/yellow]")
        )
        table.add_row(name, mark, detail)

    row("configuration", True, f"provider={config.provider} model={config.active_model}")

    workspace = config.workspace.expanduser()
    try:
        initialize_workspace(workspace)
        writable = workspace.exists()
        probe = workspace / "temp" / ".doctor-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        row("workspace", writable, f"{workspace} (writable)")
    except OSError as exc:
        row("workspace", False, f"{workspace}: {exc}")

    try:
        stats = context.database.stats()
        row(
            "database",
            True,
            f"{config.database_path} (schema v{context.database.version}, "
            f"{stats['tasks']} tasks, {stats['facts']} facts)",
        )
    except AgentError as exc:
        row("database", False, exc.message)

    credential_var = config.credential_env_var()
    if credential_var is None:
        row("credential", True, "this provider needs no API key")
    else:
        present = config.has_credential()
        row(
            "credential",
            present,
            f"{credential_var} is set" if present else f"{credential_var} is not set",
        )

    try:
        provider = create_provider(config)
        status = asyncio.run(provider.health_check())
        row(f"provider ({status.provider})", status.ok, status.detail)
        if status.available_models:
            row("models", True, ", ".join(status.available_models[:8]))
        asyncio.run(provider.aclose())
    except AgentError as exc:
        row("provider", False, exc.message)

    manager = _manager(context)
    media = manager.media_support()
    accepted = [kind for kind, ok in media["accepted"].items() if ok]
    row(
        "media input",
        bool(accepted),
        (", ".join(accepted) if accepted else "text only")
        + ("" if media["video_enabled"] else "; video disabled"),
    )

    connectors = manager.list_connectors()
    if not config.connectors_enabled:
        row("connectors", None, "disabled by configuration")
    elif not connectors:
        row("connectors", True, "none configured")
    else:
        enabled = [c for c in connectors if c["enabled"]]
        row("connectors", True, f"{len(enabled)} enabled of {len(connectors)} configured")
        for entry in connectors:
            missing = entry.get("missing_credentials") or []
            row(
                f"  {entry['name']}",
                None if not entry["enabled"] else not missing,
                ("disabled" if not entry["enabled"] else "enabled")
                + (f"; unset: {', '.join(missing)}" if missing else ""),
            )

    registry = SkillRegistry.from_directory(config.skills_dir)
    row("skills", True, f"{len(registry)} loaded from {config.skills_dir}")
    for name, reason in registry.problems:
        row(f"  skill {name}", False, reason)

    approval_ok = not config.unsafe_disable_approvals
    row(
        "approvals",
        approval_ok,
        f"mode={config.approval_mode.value}"
        + ("" if approval_ok else " — UNSAFE: approvals are disabled"),
    )
    row(
        "shell",
        True,
        f"{'enabled' if config.shell_enabled else 'disabled'}; allowlist: "
        + (", ".join(sorted(config.shell_allowed_commands)) or "(empty)"),
    )
    row("browser", None, "not enabled in this release (interface only)")

    console.print(table)


config_app = typer.Typer(name="config", help="Inspect configuration.", no_args_is_help=True)
app.add_typer(config_app)


@config_app.command("show")
def config_show(
    as_json: bool = typer.Option(False, "--json", help="Print raw JSON instead of a table."),
) -> None:
    """Print the effective configuration. Secrets are shown as presence flags only."""
    context = get_context()
    data = context.config.safe_dump()
    if as_json:
        console.print_json(json.dumps(data, indent=2, default=str))
        return
    table = Table(title="Effective configuration")
    table.add_column("Setting", style="bold")
    table.add_column("Value", overflow="fold")
    for key, value in data.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                table.add_row(f"{key}.{sub_key}", str(sub_value))
        else:
            table.add_row(key, str(value))
    console.print(table)


# --------------------------------------------------------------------------
# chat
# --------------------------------------------------------------------------
SLASH_HELP = """\
[bold]Commands[/bold]
  /help     show this help
  /status   current provider, model, workspace and limits
  /tools    the tools available in this run
  /model    the active provider and model
  /memory   durable facts you have approved
  /connectors  configured MCP servers and other connectors
  /tasks    recent tasks
  /clear    clear the conversation in this session
  /quit     exit
"""


@app.command()
def chat(
    goal: str | None = typer.Argument(None, help="Run one goal and exit, instead of chatting."),
    session: str | None = typer.Option(None, "--session", "-s", help="Continue a saved session."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show every lifecycle event."),
    unsafe_no_approvals: bool = typer.Option(
        False, "--unsafe-no-approvals", help="TESTING ONLY: run every action without approval."
    ),
) -> None:
    """Start an interactive session, or run a single goal."""
    context = get_context()
    unsubscribe = context.events.subscribe(_progress_handler(verbose))
    try:
        runner = _build_runner(context, unsafe=unsafe_no_approvals)
    except AgentError as exc:
        _handle(exc)
        return

    conversations = ConversationStore(context.database)
    session_id = session or None
    history: list[Message] = []
    if session_id:
        resolved = conversations.find_session(session_id)
        if resolved is None:
            _fail(f"no session named {session_id}")
            return
        session_id = resolved["id"]
        history = conversations.get_messages(session_id)
        console.print(f"[dim]Resumed session {session_id} ({len(history)} messages)[/dim]")

    if goal:
        _run_once(context, runner, goal, session_id, history)
        unsubscribe()
        return

    console.print(
        Panel(
            f"[bold]local-agent[/bold] {__version__} — {context.config.provider}"
            f"/{context.config.active_model}\n"
            f"workspace: {context.config.workspace.expanduser()}\n"
            f"approval mode: {context.config.approval_mode.value}\n\n"
            "Type a goal, or /help for commands. Ctrl-C interrupts a run; Ctrl-D exits.",
            title="ready",
            border_style="cyan",
        )
    )

    while True:
        try:
            line = console.input("\n[bold cyan]>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            break
        if not line:
            continue
        if line.startswith("/"):
            if _slash_command(context, runner, line, history):
                break
            continue
        session_id = _run_once(context, runner, line, session_id, history)

    unsubscribe()


def _run_once(
    context: Context,
    runner: AgentRunner,
    goal: str,
    session_id: str | None,
    history: list[Message],
) -> str:
    """Run one goal, persisting the conversation and honouring Ctrl-C."""
    task = TaskState(goal=goal)
    if session_id:
        task.session_id = session_id
    conversations = ConversationStore(context.database)
    conversations.create_session(
        session_id=task.session_id,
        title=goal[:80],
        provider=context.config.provider,
        model=context.config.active_model,
    )

    async def _go() -> RunResult:
        loaded = await runner.load_connector_tools()
        if loaded:
            console.print(f"[dim]connector tools loaded: {', '.join(loaded)}[/dim]")
        loop = asyncio.get_running_loop()
        # Ctrl-C requests a cancellation rather than killing the process, so the
        # runtime can stop cleanly before its next consequential action. Not every
        # platform supports signal handlers on the loop; there it stays a KeyboardInterrupt.
        with contextlib.suppress(NotImplementedError, RuntimeError):
            loop.add_signal_handler(signal.SIGINT, runner.request_cancel, "interrupted by Ctrl-C")
        return await runner.run(task, history=history or None)

    try:
        result = asyncio.run(_go())
    except AgentError as exc:
        _handle(exc)
        raise
    except KeyboardInterrupt:
        console.print("[yellow]interrupted[/yellow]")
        return task.session_id

    history.clear()
    history.extend(result.messages)
    conversations.add_messages(task.session_id, result.messages, task_id=task.id)

    style = {"completed": "green", "unverified": "yellow", "partial": "yellow"}.get(
        result.outcome, "red"
    )
    console.print(
        Panel(
            result.summary(),
            title=f"[{style}]{result.outcome}[/{style}] · task {task.id}",
            border_style=style,
        )
    )
    return task.session_id


def _slash_command(
    context: Context, runner: AgentRunner, line: str, history: list[Message]
) -> bool:
    """Handle a `/command`. Returns True when the session should end."""
    command = line.split()[0].lower()
    if command in {"/quit", "/exit", "/q"}:
        console.print("[dim]bye[/dim]")
        return True
    if command == "/help":
        console.print(SLASH_HELP)
    elif command == "/status":
        config = context.config
        console.print(
            f"provider: {config.provider}/{config.active_model}\n"
            f"workspace: {config.workspace.expanduser()}\n"
            f"approval mode: {config.approval_mode.value}\n"
            f"messages in context: {len(history)}\n"
            f"limits: max_steps={config.limits.max_steps}, "
            f"max_total_tool_calls={config.limits.max_total_tool_calls}"
        )
    elif command == "/tools":
        table = Table(title="Registered tools")
        table.add_column("Tool", style="bold")
        table.add_column("Risk")
        table.add_column("Approval")
        table.add_column("Description", overflow="fold")
        for definition in runner.tool_definitions():
            table.add_row(
                definition.name,
                definition.risk.value,
                "required" if definition.requires_approval else "no",
                definition.description,
            )
        console.print(table)
    elif command == "/model":
        console.print(f"{context.config.provider} / {context.config.active_model}")
    elif command == "/connectors":
        _print_connectors(_manager(context).list_connectors())
    elif command == "/memory":
        _print_facts(FactStore(context.database).list_facts(approved_only=True))
    elif command == "/tasks":
        _print_tasks(TaskStore(context.database).list_tasks(limit=10))
    elif command == "/clear":
        history.clear()
        console.print("[dim]conversation cleared (stored history is untouched)[/dim]")
    else:
        console.print(f"[yellow]unknown command {command}; try /help[/yellow]")
    return False


# --------------------------------------------------------------------------
# tasks
# --------------------------------------------------------------------------
def _print_tasks(rows: list[dict[str, Any]]) -> None:
    if not rows:
        console.print("[dim]no tasks[/dim]")
        return
    table = Table(title="Tasks")
    table.add_column("ID", style="bold")
    table.add_column("Status")
    table.add_column("Goal", overflow="fold")
    table.add_column("Updated")
    for row in rows:
        table.add_row(row["id"], row["status"], row["goal"][:70], row["updated_at"][:19])
    console.print(table)


@task_app.command("create")
def task_create(
    goal: str = typer.Argument(..., help="What the task should accomplish."),
) -> None:
    """Create a task without running it."""
    context = get_context()
    store = TaskStore(context.database)
    task = TaskState(
        goal=goal,
        provider=context.config.provider,
        model=context.config.active_model,
        workspace=str(context.config.workspace.expanduser()),
    )
    store.save(task)
    console.print(f"created task [bold]{task.id}[/bold]")
    console.print(f"run it with: local-agent task run {task.id}")


@task_app.command("list")
def task_list(
    status: str | None = typer.Option(None, "--status", help="Filter by status."),
    limit: int = typer.Option(20, "--limit"),
) -> None:
    """List recent tasks."""
    context = get_context()
    store = TaskStore(context.database)
    filter_status = TaskStatus(status) if status else None
    _print_tasks(store.list_tasks(status=filter_status, limit=limit))


@task_app.command("run")
def task_run(
    task_id: str = typer.Argument(..., help="Task ID or unique prefix."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run (or resume) a stored task in the foreground."""
    context = get_context()
    store = TaskStore(context.database)
    task = store.find(task_id)
    if task is None:
        _fail(f"no task matching {task_id!r}")
        return
    if task.is_terminal:
        _fail(f"task {task.id} is already {task.status.value} and cannot be run again")
        return

    context.events.subscribe(_progress_handler(verbose))
    runner = _build_runner(context)
    # Resuming revalidates configuration: the provider and workspace may have
    # changed since the task was created, and silently switching would be wrong.
    if task.provider and task.provider != context.config.provider:
        console.print(
            f"[yellow]note:[/yellow] this task was created with provider "
            f"{task.provider!r}; it will now run with {context.config.provider!r}"
        )
    task.cancel_requested = False
    task.pause_requested = False

    async def _go() -> RunResult:
        await runner.load_connector_tools()
        return await runner.run(task)

    result = asyncio.run(_go())
    style = {"completed": "green", "unverified": "yellow", "partial": "yellow"}.get(
        result.outcome, "red"
    )
    console.print(
        Panel(result.summary(), title=f"[{style}]{result.outcome}[/{style}]", border_style=style)
    )


@task_app.command("status")
def task_status(task_id: str = typer.Argument(...)) -> None:
    """Show a task's state, plan and evidence."""
    context = get_context()
    task = TaskStore(context.database).find(task_id)
    if task is None:
        _fail(f"no task matching {task_id!r}")
        return
    table = Table(show_header=False, box=None)
    table.add_column(style="bold")
    table.add_column(overflow="fold")
    table.add_row("id", task.id)
    table.add_row("goal", task.goal)
    table.add_row("status", task.status.value)
    table.add_row("phase", task.phase.value)
    table.add_row("outcome", task.outcome or "-")
    table.add_row("provider", f"{task.provider}/{task.model}")
    table.add_row("steps", str(task.current_step))
    table.add_row("tool calls", str(task.total_tool_calls))
    table.add_row("replans", str(task.replan_count))
    table.add_row("created", task.created_at.isoformat(timespec="seconds"))
    console.print(Panel(table, title=f"task {task.id}"))

    if task.plan.steps:
        console.print("\n[bold]Plan[/bold]\n" + task.plan.summary())
    if task.verifications:
        console.print("\n[bold]Verification evidence[/bold]")
        for record in task.verifications:
            mark = "[green]✓[/green]" if record.verified else "[red]✗[/red]"
            console.print(f"  {mark} {record.method}: {record.evidence or record.detail}")
    if task.failures:
        console.print("\n[bold]Failures[/bold]")
        for failure in task.failures:
            console.print(f"  [red]•[/red] [{failure.category.value}] {failure.message}")
    if task.result:
        console.print(f"\n[bold]Result[/bold]\n{task.result}")


@task_app.command("pause")
def task_pause(task_id: str = typer.Argument(...)) -> None:
    """Ask a task to pause before its next consequential action."""
    context = get_context()
    store = TaskStore(context.database)
    task = store.find(task_id)
    if task is None:
        _fail(f"no task matching {task_id!r}")
        return
    if task.is_terminal:
        _fail(f"task {task.id} is already {task.status.value}")
        return
    store.request_pause(task.id)
    console.print(f"pause requested for {task.id}; it will stop before its next action")


@task_app.command("resume")
def task_resume(
    task_id: str = typer.Argument(...), verbose: bool = typer.Option(False, "--verbose", "-v")
) -> None:
    """Resume a paused task after revalidating configuration and permissions."""
    context = get_context()
    store = TaskStore(context.database)
    task = store.find(task_id)
    if task is None:
        _fail(f"no task matching {task_id!r}")
        return
    if task.status is not TaskStatus.PAUSED:
        _fail(f"task {task.id} is {task.status.value}, not paused")
        return
    console.print(f"[dim]resuming {task.id} from step {task.current_step}[/dim]")
    task_run(task.id, verbose)


@task_app.command("cancel")
def task_cancel(
    task_id: str = typer.Argument(...),
    reason: str = typer.Option("cancelled by the user", "--reason"),
) -> None:
    """Cancel a task. Future actions stop; a running step is interrupted."""
    context = get_context()
    store = TaskStore(context.database)
    task = store.find(task_id)
    if task is None:
        _fail(f"no task matching {task_id!r}")
        return
    if task.is_terminal:
        _fail(f"task {task.id} is already {task.status.value}")
        return
    store.request_cancel(task.id, reason)
    reloaded = store.load(task.id)
    if reloaded is not None and reloaded.can_transition_to(TaskStatus.CANCELLED):
        reloaded.transition_to(TaskStatus.CANCELLED)
        reloaded.outcome = "cancelled"
        store.save(reloaded)
    console.print(f"cancelled {task.id}")


@task_app.command("events")
def task_events(
    task_id: str = typer.Argument(...), limit: int = typer.Option(100, "--limit")
) -> None:
    """Show a task's recorded lifecycle events."""
    context = get_context()
    store = TaskStore(context.database)
    task = store.find(task_id)
    if task is None:
        _fail(f"no task matching {task_id!r}")
        return
    rows = store.events_for(task.id, limit=limit)
    if not rows:
        console.print("[dim]no events recorded[/dim]")
        return
    table = Table(title=f"events for {task.id}")
    table.add_column("Time")
    table.add_column("Step")
    table.add_column("Type", style="bold")
    table.add_column("Message", overflow="fold")
    for row in rows:
        table.add_row(
            str(row["created_at"])[11:19],
            str(row["step"] or "-"),
            str(row["type"]),
            str(row["message"])[:100],
        )
    console.print(table)


@task_app.command("recover")
def task_recover() -> None:
    """Mark tasks left running by a crashed process as paused, so they can resume."""
    context = get_context()
    recovered = TaskStore(context.database).recover_interrupted()
    if not recovered:
        console.print("[dim]nothing to recover[/dim]")
        return
    console.print(f"recovered {len(recovered)} task(s): {', '.join(recovered)}")
    console.print("resume one with: local-agent task resume <id>")


# --------------------------------------------------------------------------
# memory / sessions / skills
# --------------------------------------------------------------------------
def _print_connectors(entries: list[dict[str, Any]]) -> None:
    if not entries:
        console.print("[dim]no connectors configured[/dim]")
        return
    for entry in entries:
        state = "[green]enabled[/green]" if entry["enabled"] else "[dim]disabled[/dim]"
        missing = entry.get("missing_credentials") or []
        warning = f" [yellow](unset: {', '.join(missing)})[/yellow]" if missing else ""
        console.print(f"  {entry['name']} — {state}{warning}")


def _print_facts(rows: list[dict[str, Any]]) -> None:
    if not rows:
        console.print("[dim]no durable facts stored[/dim]")
        return
    table = Table(title="Durable memory")
    table.add_column("ID", style="bold")
    table.add_column("Category")
    table.add_column("Fact", overflow="fold")
    table.add_column("Stored")
    for row in rows:
        table.add_row(str(row["id"]), row["category"], row["fact"], str(row["created_at"])[:19])
    console.print(table)


@memory_app.command("list")
def memory_list(
    all_facts: bool = typer.Option(False, "--all", help="Include unapproved proposals."),
) -> None:
    """List durable facts."""
    context = get_context()
    _print_facts(FactStore(context.database).list_facts(approved_only=not all_facts))


@memory_app.command("remove")
def memory_remove(
    fact_id: int = typer.Argument(..., help="The fact ID from `memory list`."),
) -> None:
    """Delete one durable fact."""
    context = get_context()
    store = FactStore(context.database)
    fact = store.get(fact_id)
    if fact is None:
        _fail(f"no fact with id {fact_id}")
        return
    console.print(f"about to delete: [italic]{fact['fact']}[/italic]")
    if not typer.confirm("delete it?"):
        console.print("[dim]cancelled[/dim]")
        return
    store.remove(fact_id)
    console.print(f"deleted fact {fact_id}")


@sessions_app.command("list")
def sessions_list(limit: int = typer.Option(20, "--limit")) -> None:
    """List saved conversation sessions."""
    context = get_context()
    rows = ConversationStore(context.database).list_sessions(limit=limit)
    if not rows:
        console.print("[dim]no sessions[/dim]")
        return
    table = Table(title="Sessions")
    table.add_column("ID", style="bold")
    table.add_column("Title", overflow="fold")
    table.add_column("Provider")
    table.add_column("Messages")
    table.add_column("Updated")
    for row in rows:
        table.add_row(
            row["id"],
            row["title"][:60],
            f"{row['provider']}/{row['model']}" if row["provider"] else "-",
            str(row["message_count"]),
            str(row["updated_at"])[:19],
        )
    console.print(table)


@sessions_app.command("show")
def sessions_show(
    session_id: str = typer.Argument(...), limit: int = typer.Option(50, "--limit")
) -> None:
    """Print the messages in a session."""
    context = get_context()
    store = ConversationStore(context.database)
    session = store.find_session(session_id)
    if session is None:
        _fail(f"no session named {session_id}")
        return
    session_id = session["id"]
    for message in store.get_messages(session_id, limit=limit):
        colour = {"user": "cyan", "assistant": "white", "tool": "dim", "system": "dim"}[
            message.role
        ]
        header: str = message.role
        if message.tool_calls:
            header += f" → {', '.join(c.name for c in message.tool_calls)}"
        console.print(f"[{colour}][bold]{header}[/bold]: {message.content[:800]}[/{colour}]")


@sessions_app.command("delete")
def sessions_delete(session_id: str = typer.Argument(...)) -> None:
    """Delete a session and its messages."""
    context = get_context()
    store = ConversationStore(context.database)
    session = store.find_session(session_id)
    if session is None:
        _fail(f"no session named {session_id}")
        return
    session_id = session["id"]
    count = store.count_messages(session_id)
    if not typer.confirm(f"delete session {session_id} and its {count} message(s)?"):
        console.print("[dim]cancelled[/dim]")
        return
    store.delete_session(session_id)
    console.print(f"deleted session {session_id}")


@skills_app.command("list")
def skills_list() -> None:
    """List discovered skills and say why any are unavailable."""
    context = get_context()
    registry = SkillRegistry.from_directory(context.config.skills_dir)
    runner_tools = [
        "get_current_time",
        "list_files",
        "read_file",
        "search_files",
        "write_file",
        "verify_result",
        "run_shell",
        "remember_fact",
    ]
    if not len(registry) and not registry.problems:
        console.print(f"[dim]no skills in {context.config.skills_dir}[/dim]")
        return
    table = Table(title=f"Skills in {context.config.skills_dir}")
    table.add_column("Name", style="bold")
    table.add_column("Available")
    table.add_column("Description", overflow="fold")
    for name in registry.names():
        check = registry.check(name, available_tools=runner_tools, config=context.config)
        table.add_row(
            name,
            "[green]yes[/green]" if check.available else f"[red]no[/red] — {check.reason}",
            check.skill.description,
        )
    for name, reason in registry.problems:
        table.add_row(name, f"[red]invalid[/red] — {reason}", "")
    console.print(table)


@skills_app.command("show")
def skills_show(name: str = typer.Argument(...)) -> None:
    """Print a skill's metadata and workflow."""
    context = get_context()
    registry = SkillRegistry.from_directory(context.config.skills_dir)
    try:
        skill = registry.get(name)
    except AgentError as exc:
        _fail(exc.message)
        return
    console.print(Panel(skill.prompt_block(), title=skill.name, border_style="cyan"))
    if skill.body:
        console.print(skill.body)


# --------------------------------------------------------------------------
# connectors
# --------------------------------------------------------------------------
@connectors_app.command("list")
def connectors_list(
    as_json: bool = typer.Option(False, "--json", help="Print raw JSON, for scripting or a UI."),
) -> None:
    """List configured connectors and whether their credentials are present."""
    context = get_context()
    entries = _manager(context).list_connectors()
    if as_json:
        console.print_json(json.dumps(entries, indent=2, default=str))
        return
    if not entries:
        console.print("[dim]no connectors configured[/dim]")
        console.print("add one with: local-agent connectors add NAME --command npx --arg ...")
        return
    table = Table(title="Connectors")
    table.add_column("Name", style="bold")
    table.add_column("Kind")
    table.add_column("Enabled")
    table.add_column("Target", overflow="fold")
    table.add_column("Credentials")
    for entry in entries:
        target = entry.get("url") or " ".join([entry.get("command", ""), *entry.get("args", [])])
        missing = entry.get("missing_credentials") or []
        credentials = (
            f"[red]missing: {', '.join(missing)}[/red]"
            if missing
            else (
                "[green]present[/green]" if entry["credentials"]["env_vars"] else "[dim]none[/dim]"
            )
        )
        table.add_row(
            entry["name"],
            entry["kind"],
            "[green]yes[/green]" if entry["enabled"] else "[dim]no[/dim]",
            target.strip() or "-",
            credentials,
        )
    console.print(table)
    console.print(
        "[dim]Connector tools are namespaced `mcp__<connector>__<tool>` and require "
        "approval unless declared read-only.[/dim]"
    )


@connectors_app.command("add")
def connectors_add(
    name: str = typer.Argument(..., help="Short name; becomes the tool prefix."),
    command: str = typer.Option("", "--command", help="Program for a stdio MCP server."),
    arg: list[str] = typer.Option([], "--arg", help="Argument for the command. Repeatable."),
    url: str = typer.Option("", "--url", help="URL for an HTTP MCP server."),
    env: list[str] = typer.Option(
        [], "--env", help="Environment variable to forward. Repeatable. Names only."
    ),
    header_env: list[str] = typer.Option(
        [], "--header-env", help="HEADER=ENV_VAR for an HTTP connector. Repeatable."
    ),
    description: str = typer.Option("", "--description"),
    enable: bool = typer.Option(False, "--enable", help="Enable it immediately."),
    allow: list[str] = typer.Option(
        [], "--allow", help="Only expose these remote tools. Repeatable."
    ),
    read_only: list[str] = typer.Option(
        [], "--read-only", help="Remote tools you have verified are read-only. Repeatable."
    ),
    timeout: float = typer.Option(30.0, "--timeout"),
    replace: bool = typer.Option(False, "--replace", help="Overwrite an existing connector."),
) -> None:
    """Add an MCP server. It starts disabled unless you pass --enable."""
    context = get_context()
    if bool(command) == bool(url):
        _fail("give exactly one of --command (stdio) or --url (http)")
        return
    headers: dict[str, str] = {}
    for pair in header_env:
        if "=" not in pair:
            _fail(f"--header-env expects HEADER=ENV_VAR, got {pair!r}")
            return
        header, variable = pair.split("=", 1)
        headers[header.strip()] = variable.strip()

    try:
        entry = _manager(context).add_connector(
            name=name,
            kind="mcp_stdio" if command else "mcp_http",
            command=command,
            args=list(arg),
            url=url,
            env=list(env),
            header_env=headers,
            description=description,
            enabled=enable,
            tool_allowlist=list(allow),
            read_only_tools=list(read_only),
            timeout_seconds=timeout,
            replace=replace,
        )
    except AgentError as exc:
        _fail(exc.message)
        return

    console.print(f"added connector [bold]{entry['name']}[/bold]")
    missing = entry.get("credentials", {}).get("env_vars", [])
    if missing:
        console.print(f"[dim]reads these environment variables: {', '.join(missing)}[/dim]")
    if not enable:
        console.print(f"enable it with: local-agent connectors enable {name}")
    console.print(f"test it with:   local-agent connectors test {name}")


@connectors_app.command("remove")
def connectors_remove(
    name: str = typer.Argument(...),
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation."),
) -> None:
    """Remove a connector definition."""
    context = get_context()
    manager = _manager(context)
    if not any(c["name"] == name for c in manager.list_connectors()):
        _fail(f"no connector named {name!r}")
        return
    if not yes and not typer.confirm(f"remove connector {name}?"):
        console.print("[dim]cancelled[/dim]")
        return
    manager.remove_connector(name)
    console.print(f"removed connector {name}")


@connectors_app.command("enable")
def connectors_enable(name: str = typer.Argument(...)) -> None:
    """Enable a connector so its tools load on the next run."""
    context = get_context()
    entry = _manager(context).set_connector_enabled(name, True)
    if entry is None:
        _fail(f"no connector named {name!r}")
        return
    console.print(f"enabled [bold]{name}[/bold]")
    console.print(
        "[yellow]Its tools will be offered to the model on the next run. They require "
        "approval unless you declared them read-only.[/yellow]"
    )


@connectors_app.command("disable")
def connectors_disable(name: str = typer.Argument(...)) -> None:
    """Disable a connector without deleting its definition."""
    context = get_context()
    entry = _manager(context).set_connector_enabled(name, False)
    if entry is None:
        _fail(f"no connector named {name!r}")
        return
    console.print(f"disabled {name}")


@connectors_app.command("test")
def connectors_test(
    name: str | None = typer.Argument(None, help="Test one connector, or all of them."),
) -> None:
    """Contact connectors and list the tools each one offers."""
    context = get_context()
    try:
        statuses = asyncio.run(_manager(context).test_connectors(name))
    except AgentError as exc:
        _handle(exc)
        return
    if not statuses:
        console.print("[dim]no connectors configured[/dim]")
        return
    for status in statuses:
        mark = "[green]ok[/green]" if status["ok"] else "[red]unavailable[/red]"
        console.print(f"\n[bold]{status['name']}[/bold] {mark} — {status['detail']}")
        if status["missing_credentials"]:
            console.print(
                f"  [yellow]unset variables: {', '.join(status['missing_credentials'])}[/yellow]"
            )
        for tool in status["tools"]:
            console.print(f"  · {tool}")


# --------------------------------------------------------------------------
# data management
# --------------------------------------------------------------------------
@app.command("clear-data")
def clear_data(
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation prompt."),
    connectors: bool = typer.Option(
        False,
        "--connectors",
        help="Also delete connector definitions. They are configuration, so they are kept by default.",
    ),
) -> None:
    """Delete stored data: sessions, messages, tasks, facts and events.

    Connector definitions are configuration rather than data, so they survive
    unless you pass --connectors. Workspace files are never touched.
    """
    context = get_context()
    manager = _manager(context)
    stats = context.database.stats()
    total = sum(stats.values())
    configured = manager.list_connectors()

    table = Table(title="About to delete")
    table.add_column("Item", style="bold")
    table.add_column("Count", justify="right")
    for name, count in stats.items():
        table.add_row(name, str(count))
    if configured:
        table.add_row(
            "connectors",
            f"[red]{len(configured)}[/red]"
            if connectors
            else f"[dim]{len(configured)} (kept)[/dim]",
        )
    console.print(table)

    if total == 0 and not (connectors and configured):
        console.print("[dim]nothing to delete[/dim]")
        # Say so rather than leaving the user to wonder why `doctor` still
        # reports connectors after a command that claimed to clear everything.
        if configured:
            console.print(
                f"[dim]{len(configured)} connector definition(s) are configuration and were "
                "kept. Remove them with `local-agent connectors remove NAME`, or re-run "
                "this with --connectors.[/dim]"
            )
        return

    scope = f"{total} rows from {context.config.database_path}"
    if connectors and configured:
        scope += f", and {len(configured)} connector definition(s)"
    console.print(
        f"[yellow]This permanently deletes {scope}. Workspace files are not touched.[/yellow]"
    )

    if not yes and not typer.confirm("Delete it?"):
        console.print("[dim]cancelled[/dim]")
        return

    deleted = context.database.clear_all()
    console.print(f"deleted {sum(deleted.values())} rows")

    if connectors:
        for entry in configured:
            manager.remove_connector(entry["name"])
        if configured:
            console.print(f"deleted {len(configured)} connector definition(s)")
    elif configured:
        console.print(
            f"[dim]kept {len(configured)} connector definition(s): "
            f"{', '.join(c['name'] for c in configured)}. "
            "They are configuration, not data. Use --connectors to remove them too.[/dim]"
        )


def main() -> None:
    """Console-script entry point."""
    try:
        app()
    finally:
        context = _state.get("context")
        if context is not None:
            context.close()
