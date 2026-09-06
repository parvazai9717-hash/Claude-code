"""The interactive approval prompt.

`ConsoleApprover` is what a human actually sees before a consequential action
runs, so its behaviour is worth pinning down: what the panel discloses, how each
keypress maps to a decision, and that "approve for this run" is remembered for
exactly one tool and not the others.

The prompt needs no real terminal: the console writes to a buffer and the
keypress is fed through `sys.stdin`.
"""

from __future__ import annotations

import io
from collections.abc import Iterator

import pytest
from rich.console import Console

from agent.messages import RiskCategory, RiskLevel, ToolDefinition
from agent.security.approvals import (
    ApprovalDecision,
    ApprovalRequest,
    ConsoleApprover,
    describe_request,
)
from agent.security.redaction import Redactor

WRITE_TOOL = ToolDefinition(
    name="write_file",
    description="Write a file.",
    read_only=False,
    requires_approval=True,
    risk=RiskLevel.MEDIUM,
    risk_category=RiskCategory.WRITE,
    reversible=False,
)
SHELL_TOOL = ToolDefinition(
    name="run_shell",
    description="Run a command.",
    read_only=False,
    requires_approval=True,
    risk=RiskLevel.HIGH,
    risk_category=RiskCategory.SHELL,
    reversible=False,
)


@pytest.fixture
def buffer() -> io.StringIO:
    return io.StringIO()


@pytest.fixture
def console(buffer: io.StringIO) -> Console:
    # `force_terminal=False` keeps the output plain so assertions read cleanly.
    return Console(file=buffer, width=100, force_terminal=False)


@pytest.fixture
def keypress(monkeypatch: pytest.MonkeyPatch) -> Iterator[callable]:
    """Feed a single keypress to the prompt."""

    def press(key: str) -> None:
        monkeypatch.setattr("sys.stdin", io.StringIO(f"{key}\n"))

    yield press


def _request(tool: ToolDefinition = WRITE_TOOL, **arguments: object) -> ApprovalRequest:
    args = arguments or {"path": "outputs/report.md", "content": "hello"}
    summary, target = describe_request(tool, args)
    return ApprovalRequest(tool=tool, arguments=args, summary=summary, target=target, step=3)


# -- decisions --------------------------------------------------------------
@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("y", ApprovalDecision.APPROVE_ONCE),
        ("a", ApprovalDecision.APPROVE_FOR_RUN),
        ("n", ApprovalDecision.DENY),
        ("c", ApprovalDecision.CANCEL),
    ],
)
def test_each_key_maps_to_its_decision(
    console: Console, keypress, key: str, expected: ApprovalDecision
) -> None:
    keypress(key)
    response = ConsoleApprover(console=console).request(_request())
    assert response.decision is expected
    assert response.approved is (
        expected in {ApprovalDecision.APPROVE_ONCE, ApprovalDecision.APPROVE_FOR_RUN}
    )


def test_the_default_is_to_deny(console: Console, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pressing enter without choosing must not approve anything."""
    monkeypatch.setattr("sys.stdin", io.StringIO("\n"))
    response = ConsoleApprover(console=console).request(_request())
    assert response.decision is ApprovalDecision.DENY


# -- what the panel discloses ----------------------------------------------
def test_the_panel_shows_everything_needed_to_decide(
    console: Console, buffer: io.StringIO, keypress
) -> None:
    keypress("n")
    ConsoleApprover(console=console).request(_request())
    output = buffer.getvalue()
    assert "Approval required" in output
    assert "write_file" in output
    assert "outputs/report.md" in output  # the concrete target
    assert "medium" in output  # risk level
    assert "write" in output  # risk category
    assert "cannot be undone" in output  # reversibility
    assert "y/a/n/c" in output  # the available choices


def test_a_reversible_action_is_labelled_as_such(
    console: Console, buffer: io.StringIO, keypress
) -> None:
    reversible = ToolDefinition(
        name="remember_fact",
        description="Save a fact.",
        read_only=False,
        requires_approval=True,
        risk_category=RiskCategory.MEMORY,
        reversible=True,
    )
    keypress("n")
    ConsoleApprover(console=console).request(_request(reversible, fact="the user likes tea"))
    output = buffer.getvalue()
    assert "yes" in output and "cannot be undone" not in output


def test_the_shell_command_is_shown_verbatim(
    console: Console, buffer: io.StringIO, keypress
) -> None:
    keypress("n")
    ConsoleApprover(console=console).request(_request(SHELL_TOOL, command=["ls", "-la", "files"]))
    assert "ls -la files" in buffer.getvalue()


def test_arguments_are_redacted_in_the_prompt(
    console: Console, buffer: io.StringIO, keypress
) -> None:
    """A human must never be shown a credential in order to approve an action."""
    redactor = Redactor(environ={"SOME_API_KEY": "sk-prompt-secret-abcdef123"})
    keypress("n")
    ConsoleApprover(console=console, redactor=redactor).request(
        _request(WRITE_TOOL, path="outputs/a.md", content="token=sk-prompt-secret-abcdef123")
    )
    output = buffer.getvalue()
    assert "sk-prompt-secret-abcdef123" not in output
    assert "REDACTED" in output


def test_very_long_arguments_are_clipped(console: Console, buffer: io.StringIO, keypress) -> None:
    keypress("n")
    ConsoleApprover(console=console).request(
        _request(WRITE_TOOL, path="outputs/a.md", content="x" * 5000)
    )
    assert len(buffer.getvalue()) < 4000


# -- "approve for this run" -------------------------------------------------
def test_approve_for_run_is_remembered_and_not_asked_again(
    console: Console, buffer: io.StringIO, keypress
) -> None:
    approver = ConsoleApprover(console=console)
    keypress("a")
    assert approver.request(_request()).decision is ApprovalDecision.APPROVE_FOR_RUN
    assert approver.session_approved == {"write_file"}

    # A second request must be answered from memory, with no new prompt drawn.
    buffer.truncate(0)
    buffer.seek(0)
    second = approver.request(_request())
    assert second.approved is True
    assert "already approved" in second.note
    assert buffer.getvalue() == "", "the human must not be asked twice"


def test_approve_for_run_does_not_leak_to_other_tools(console: Console, keypress) -> None:
    """Approving writes must not silently approve shell commands."""
    approver = ConsoleApprover(console=console)
    keypress("a")
    approver.request(_request())
    assert approver.session_approved == {"write_file"}

    keypress("n")
    assert approver.request(_request(SHELL_TOOL, command=["ls"])).approved is False


def test_approve_once_is_not_remembered(console: Console, keypress) -> None:
    approver = ConsoleApprover(console=console)
    keypress("y")
    approver.request(_request())
    assert approver.session_approved == set(), "approve-once must not persist"
