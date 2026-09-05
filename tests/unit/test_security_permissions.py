"""Permission decisions and shell policy."""

from __future__ import annotations

import pytest

from agent.config import ApprovalMode, Config
from agent.errors import ErrorCategory
from agent.messages import RiskCategory, RiskLevel, ToolDefinition
from agent.security.approvals import (
    ApprovalDecision,
    ApprovalRequest,
    AutoDenyApprover,
    PolicyApprover,
    UnsafeAutoApprover,
    describe_request,
)
from agent.security.permissions import PermissionChecker


def _tool(**kwargs: object) -> ToolDefinition:
    defaults: dict[str, object] = {"name": "t", "description": "d"}
    return ToolDefinition(**{**defaults, **kwargs})  # type: ignore[arg-type]


READ_TOOL = _tool(name="read_file", read_only=True, risk_category=RiskCategory.READ)
WRITE_TOOL = _tool(
    name="write_file",
    read_only=False,
    requires_approval=True,
    risk=RiskLevel.MEDIUM,
    risk_category=RiskCategory.WRITE,
)
SHELL_TOOL = _tool(
    name="run_shell",
    read_only=False,
    requires_approval=True,
    risk=RiskLevel.HIGH,
    risk_category=RiskCategory.SHELL,
)


def test_read_only_tools_never_need_approval(config: Config) -> None:
    decision = PermissionChecker(config).check_tool(READ_TOOL)
    assert decision.allowed and not decision.requires_approval


def test_risky_mode_gates_writes(config: Config) -> None:
    decision = PermissionChecker(config).check_tool(WRITE_TOOL)
    assert decision.allowed and decision.requires_approval


def test_always_mode_gates_every_non_read_only_tool(config: Config) -> None:
    config.approval_mode = ApprovalMode.ALWAYS
    checker = PermissionChecker(config)
    assert checker.check_tool(WRITE_TOOL).requires_approval
    # A read-only tool still runs unattended: there is nothing to approve.
    assert not checker.check_tool(READ_TOOL).requires_approval


def test_automatic_mode_denies_anything_not_allowlisted(config: Config) -> None:
    config.approval_mode = ApprovalMode.AUTOMATIC
    decision = PermissionChecker(config).check_tool(WRITE_TOOL)
    assert decision.allowed is False
    assert decision.error_category is ErrorCategory.PERMISSION_DENIED


def test_automatic_mode_allows_the_allowlist(config: Config) -> None:
    config.approval_mode = ApprovalMode.AUTOMATIC
    config.auto_approve_tools = ["write_file"]
    decision = PermissionChecker(config).check_tool(WRITE_TOOL)
    assert decision.allowed and not decision.requires_approval


def test_shell_can_be_disabled_entirely(config: Config) -> None:
    config.shell_enabled = False
    decision = PermissionChecker(config).check_tool(SHELL_TOOL, {"command": ["ls"]})
    assert decision.allowed is False
    assert "disabled" in decision.reason


def test_allowlisted_command_is_permitted(config: Config) -> None:
    decision = PermissionChecker(config).check_shell_command(["ls", "-la"])
    assert decision.allowed and decision.requires_approval


def test_command_outside_the_allowlist_is_denied(config: Config) -> None:
    decision = PermissionChecker(config).check_shell_command(["nmap", "-p", "80"])
    assert decision.allowed is False
    assert "not in the shell allowlist" in decision.reason


def test_permanently_forbidden_commands(config: Config) -> None:
    config.shell_allowed_commands = ["ls"]
    for command in (["rm", "-rf", "/"], ["sudo", "ls"], ["curl", "http://x"], ["bash", "-c", "x"]):
        decision = PermissionChecker(config).check_shell_command(command)
        assert decision.allowed is False


@pytest.mark.parametrize(
    "command", ["ls | grep x", "ls; rm -rf /", "echo $(whoami)", "cat a > b", "ls && ls"]
)
def test_shell_metacharacters_are_refused(config: Config, command: str) -> None:
    decision = PermissionChecker(config).check_shell_command(command)
    assert decision.allowed is False
    assert "metacharacters" in decision.reason


def test_path_qualified_program_is_refused(config: Config) -> None:
    decision = PermissionChecker(config).check_shell_command(["/bin/ls"])
    assert decision.allowed is False
    assert "bare command name" in decision.reason


def test_empty_and_wrong_type_commands(config: Config) -> None:
    checker = PermissionChecker(config)
    assert checker.check_shell_command([]).allowed is False
    assert checker.check_shell_command("").allowed is False
    assert checker.check_shell_command(42).allowed is False


def test_parse_command_handles_quotes(config: Config) -> None:
    argv = PermissionChecker.parse_command("echo 'hello world'")
    assert argv == ["echo", "hello world"]


# -- approvers -------------------------------------------------------------
def _request(tool: ToolDefinition = WRITE_TOOL) -> ApprovalRequest:
    return ApprovalRequest(tool=tool, arguments={"path": "a.txt", "content": "x"})


def test_auto_deny_approver_denies() -> None:
    response = AutoDenyApprover().request(_request())
    assert response.approved is False


def test_unsafe_approver_approves() -> None:
    assert UnsafeAutoApprover().request(_request()).approved is True


def test_policy_approver_consumes_the_queue() -> None:
    approver = PolicyApprover(
        responses=[ApprovalDecision.APPROVE_ONCE, ApprovalDecision.DENY],
        default=ApprovalDecision.CANCEL,
    )
    assert approver.request(_request()).approved is True
    assert approver.request(_request()).approved is False
    assert approver.request(_request()).decision is ApprovalDecision.CANCEL
    assert len(approver.seen) == 3


def test_policy_approver_per_tool_rules() -> None:
    approver = PolicyApprover(
        by_tool={"write_file": ApprovalDecision.DENY}, default=ApprovalDecision.APPROVE_ONCE
    )
    assert approver.request(_request(WRITE_TOOL)).approved is False
    assert approver.request(_request(SHELL_TOOL)).approved is True


def test_approval_request_redacts_arguments(redactor) -> None:  # type: ignore[no-untyped-def]
    request = ApprovalRequest(
        tool=WRITE_TOOL, arguments={"path": "a", "api_key": "sk-test-secret-value-1234"}
    )
    assert "sk-test-secret-value-1234" not in str(request.redacted_arguments(redactor))


def test_describe_request_is_concrete() -> None:
    summary, target = describe_request(SHELL_TOOL, {"command": ["ls", "-la"]})
    assert target == "ls -la"
    summary, target = describe_request(WRITE_TOOL, {"path": "out.md", "content": "abc"})
    assert target == "out.md" and "3 characters" in summary
