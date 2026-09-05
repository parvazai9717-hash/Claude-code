"""Human approval.

An approver is asked *after* the permission layer has already allowed an action.
Approvers are deliberately simple and injectable so tests never block on a
terminal prompt, and so the model can never reach one: only the runtime calls an
approver, and the model has no tool that does so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from ..messages import RiskCategory, RiskLevel, ToolDefinition
from .redaction import Redactor


class ApprovalDecision(StrEnum):
    """What the human chose."""

    APPROVE_ONCE = "approve_once"
    APPROVE_FOR_RUN = "approve_for_run"
    DENY = "deny"
    CANCEL = "cancel"

    @property
    def approved(self) -> bool:
        return self in {ApprovalDecision.APPROVE_ONCE, ApprovalDecision.APPROVE_FOR_RUN}


@dataclass(frozen=True)
class ApprovalRequest:
    """Everything a human needs in order to decide."""

    tool: ToolDefinition
    arguments: dict[str, Any]
    task_id: str | None = None
    step: int = 0
    #: A one-line, concrete description of what will happen.
    summary: str = ""
    #: The concrete target — a path, a command, a fact.
    target: str = ""
    reason: str = ""

    @property
    def risk(self) -> RiskLevel:
        return self.tool.risk

    @property
    def reversible(self) -> bool:
        return self.tool.reversible

    def redacted_arguments(self, redactor: Redactor) -> dict[str, Any]:
        result = redactor.redact(self.arguments)
        return result if isinstance(result, dict) else {}


@dataclass(frozen=True)
class ApprovalResponse:
    decision: ApprovalDecision
    note: str = ""

    @property
    def approved(self) -> bool:
        return self.decision.approved


class Approver(Protocol):
    """Anything that can answer an approval request."""

    def request(self, request: ApprovalRequest) -> ApprovalResponse:  # pragma: no cover
        ...


@dataclass
class PolicyApprover:
    """Answers from a fixed script. Used by tests and by non-interactive runs.

    Attributes:
        default: The answer used when no rule matches.
        by_tool: Per-tool answers, checked first.
        responses: A queue of answers consumed in order, checked before `by_tool`.
    """

    default: ApprovalDecision = ApprovalDecision.DENY
    by_tool: dict[str, ApprovalDecision] = field(default_factory=dict)
    responses: list[ApprovalDecision] = field(default_factory=list)
    #: Requests seen, in order — useful for assertions.
    seen: list[ApprovalRequest] = field(default_factory=list)

    def request(self, request: ApprovalRequest) -> ApprovalResponse:
        self.seen.append(request)
        if self.responses:
            return ApprovalResponse(self.responses.pop(0), note="scripted")
        if request.tool.name in self.by_tool:
            return ApprovalResponse(self.by_tool[request.tool.name], note="per-tool policy")
        return ApprovalResponse(self.default, note="default policy")


class AutoDenyApprover:
    """Denies everything. The safe default whenever no human is present."""

    def __init__(self, reason: str = "no interactive approver is available") -> None:
        self.reason = reason
        self.seen: list[ApprovalRequest] = []

    def request(self, request: ApprovalRequest) -> ApprovalResponse:
        self.seen.append(request)
        return ApprovalResponse(ApprovalDecision.DENY, note=self.reason)


class UnsafeAutoApprover:
    """Approves everything.

    UNSAFE. Exists only for tests and for the explicitly-flagged
    `unsafe_disable_approvals` setting. It is never selected by default and the
    CLI prints a warning whenever it is active.
    """

    def __init__(self) -> None:
        self.seen: list[ApprovalRequest] = []

    def request(self, request: ApprovalRequest) -> ApprovalResponse:
        self.seen.append(request)
        return ApprovalResponse(ApprovalDecision.APPROVE_ONCE, note="UNSAFE auto-approval")


_RISK_STYLES = {
    RiskLevel.READ_ONLY: "green",
    RiskLevel.LOW: "cyan",
    RiskLevel.MEDIUM: "yellow",
    RiskLevel.HIGH: "bold red",
}


class ConsoleApprover:
    """Interactive terminal approval with a redacted, explicit prompt.

    The prompt always shows: the tool, the redacted arguments, the risk level and
    category, the concrete target, and whether the action is reversible.
    """

    def __init__(self, console: Any = None, redactor: Redactor | None = None) -> None:
        from rich.console import Console

        self.console = console or Console()
        self.redactor = redactor or Redactor()
        #: Tools approved for the remainder of this run.
        self.session_approved: set[str] = set()

    def request(self, request: ApprovalRequest) -> ApprovalResponse:
        if request.tool.name in self.session_approved:
            return ApprovalResponse(
                ApprovalDecision.APPROVE_FOR_RUN, note="already approved for this run"
            )

        from rich.panel import Panel
        from rich.prompt import Prompt
        from rich.table import Table

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style="bold")
        table.add_column(overflow="fold")
        table.add_row("Tool", request.tool.name)
        table.add_row("Action", request.summary or request.tool.description)
        if request.target:
            table.add_row("Target", request.target)
        style = _RISK_STYLES.get(request.risk, "yellow")
        table.add_row("Risk", f"[{style}]{request.risk.value}[/{style}]")
        table.add_row("Category", request.tool.risk_category.value)
        table.add_row(
            "Reversible",
            "[green]yes[/green]" if request.reversible else "[red]no — this cannot be undone[/red]",
        )
        arguments = request.redacted_arguments(self.redactor)
        for key, value in arguments.items():
            rendered = str(value)
            if len(rendered) > 500:
                rendered = rendered[:500] + " …"
            table.add_row(f"arg: {key}", rendered)

        self.console.print(Panel(table, title="[bold]Approval required[/bold]", border_style=style))
        choice = Prompt.ask(
            "[bold]Allow?[/bold]",
            choices=["y", "a", "n", "c"],
            default="n",
            console=self.console,
        )
        mapping = {
            "y": ApprovalDecision.APPROVE_ONCE,
            "a": ApprovalDecision.APPROVE_FOR_RUN,
            "n": ApprovalDecision.DENY,
            "c": ApprovalDecision.CANCEL,
        }
        decision = mapping[choice]
        if decision is ApprovalDecision.APPROVE_FOR_RUN:
            self.session_approved.add(request.tool.name)
        return ApprovalResponse(decision)


def describe_request(tool: ToolDefinition, arguments: dict[str, Any]) -> tuple[str, str]:
    """Build a `(summary, target)` pair for an approval prompt.

    Keeping this here means the description a human sees is derived from the same
    validated arguments the tool will actually receive.
    """
    if tool.risk_category is RiskCategory.SHELL:
        command = arguments.get("command", "")
        rendered = " ".join(command) if isinstance(command, list) else str(command)
        return "Run a shell command in the workspace", rendered
    if tool.risk_category is RiskCategory.WRITE:
        path = str(arguments.get("path", "?"))
        content = arguments.get("content", "")
        size = len(content) if isinstance(content, str) else 0
        mode = arguments.get("mode", "overwrite")
        return f"Write {size} characters to a workspace file ({mode})", path
    if tool.risk_category is RiskCategory.MEMORY:
        return "Save a durable fact to local memory", str(arguments.get("fact", ""))[:200]
    if tool.risk_category is RiskCategory.DELETE:
        return "Delete from the workspace", str(arguments.get("path", "?"))
    return tool.description, str(next(iter(arguments.values()), "")) if arguments else ""
