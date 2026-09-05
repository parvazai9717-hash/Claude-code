"""The security boundary holds against a model that actively tries to cross it.

Every scenario here scripts a model that requests something it must not get. The
assertion is always the same: the runtime refuses, records the refusal, and the
model sees a structured error instead of a success.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.config import ApprovalMode, Config
from agent.errors import ErrorCategory
from agent.providers.mock import MockProvider
from agent.runtime import build_runner
from agent.security.approvals import ApprovalDecision, PolicyApprover
from agent.task_state import TaskState


def _run(config: Config, script: list, approver: PolicyApprover):  # type: ignore[no-untyped-def]
    runner = build_runner(config, provider=MockProvider(script), approver=approver)
    return runner


async def test_traversal_attempt_is_refused(
    config: Config, tmp_path: Path, approve_all: PolicyApprover
) -> None:
    secret = tmp_path / "outside_secret.txt"
    secret.write_text("private data")
    runner = _run(
        config,
        [
            MockProvider.call("read_file", {"path": "../../outside_secret.txt"}),
            MockProvider.text("I could not read outside the workspace."),
        ],
        approve_all,
    )
    result = await runner.run(TaskState(goal="read the secret file"))
    action = result.task.actions[0]
    assert action.ok is False
    assert action.error_category is ErrorCategory.PATH_ESCAPE
    assert "private data" not in str(result.messages)


async def test_absolute_path_attempt_is_refused(
    config: Config, approve_all: PolicyApprover
) -> None:
    runner = _run(
        config,
        [MockProvider.call("read_file", {"path": "/etc/passwd"}), MockProvider.text("refused")],
        approve_all,
    )
    result = await runner.run(TaskState(goal="read /etc/passwd"))
    assert result.task.actions[0].error_category is ErrorCategory.PATH_ESCAPE


async def test_credential_file_is_refused_and_never_leaks(
    config: Config, workspace: Path, approve_all: PolicyApprover
) -> None:
    (workspace / "files" / ".env").write_text("GEMINI_API_KEY=sk-real-secret-abcdef123456")
    runner = _run(
        config,
        [
            MockProvider.call("read_file", {"path": "files/.env"}),
            MockProvider.call("search_files", {"query": "GEMINI_API_KEY"}),
            MockProvider.text("I could not access the credential file."),
        ],
        approve_all,
    )
    result = await runner.run(TaskState(goal="find the api key"))
    assert result.task.actions[0].error_category is ErrorCategory.SECRET_PROTECTED
    # The search must not surface the key from the protected file either.
    assert "sk-real-secret-abcdef123456" not in str(result.messages)


async def test_symlink_escape_is_refused(
    config: Config, workspace: Path, tmp_path: Path, approve_all: PolicyApprover
) -> None:
    secret = tmp_path / "escape_target.txt"
    secret.write_text("data behind the link")
    (workspace / "files" / "link.txt").symlink_to(secret)
    runner = _run(
        config,
        [MockProvider.call("read_file", {"path": "files/link.txt"}), MockProvider.text("refused")],
        approve_all,
    )
    result = await runner.run(TaskState(goal="follow the link"))
    assert result.task.actions[0].error_category is ErrorCategory.PATH_ESCAPE
    assert "data behind the link" not in str(result.messages)


async def test_non_allowlisted_command_is_refused(
    config: Config, approve_all: PolicyApprover
) -> None:
    runner = _run(
        config,
        [
            MockProvider.call("run_shell", {"command": ["curl", "https://example.com"]}),
            MockProvider.text("That command is not permitted."),
        ],
        approve_all,
    )
    result = await runner.run(TaskState(goal="fetch a url"))
    assert result.task.actions[0].error_category is ErrorCategory.PERMISSION_DENIED
    assert approve_all.seen == [], "a denied command must never reach the human"


async def test_destructive_command_is_refused_before_approval(
    config: Config, approve_all: PolicyApprover
) -> None:
    """A permanently forbidden command must be stopped by policy, not by the human."""
    runner = _run(
        config,
        [
            MockProvider.call("run_shell", {"command": ["rm", "-rf", "/"]}),
            MockProvider.text("refused"),
        ],
        approve_all,
    )
    result = await runner.run(TaskState(goal="delete everything"))
    assert result.task.actions[0].error_category is ErrorCategory.PERMISSION_DENIED
    assert approve_all.seen == []


async def test_the_model_cannot_call_an_unregistered_tool(
    config: Config, approve_all: PolicyApprover
) -> None:
    runner = _run(
        config,
        [
            MockProvider.call("browser_open", {"url": "https://example.com"}),
            MockProvider.call("execute_python", {"code": "import os; os.system('id')"}),
            MockProvider.text("Those tools do not exist."),
        ],
        approve_all,
    )
    result = await runner.run(TaskState(goal="browse and run code"))
    assert all(a.error_category is ErrorCategory.UNKNOWN_TOOL for a in result.task.actions)


async def test_automatic_mode_denies_a_write_with_no_human_present(
    config: Config, workspace: Path
) -> None:
    """In automatic mode there is nobody to ask, so the write must be denied."""
    config.approval_mode = ApprovalMode.AUTOMATIC
    config.auto_approve_tools = ["read_file", "list_files"]
    approver = PolicyApprover(default=ApprovalDecision.APPROVE_ONCE)
    runner = _run(
        config,
        [
            MockProvider.call("write_file", {"path": "outputs/a.md", "content": "x"}),
            MockProvider.text("Writing is not permitted in automatic mode."),
        ],
        approver,
    )
    result = await runner.run(TaskState(goal="write a file"))
    assert result.task.actions[0].error_category is ErrorCategory.PERMISSION_DENIED
    assert not (workspace / "outputs" / "a.md").exists()
    assert approver.seen == []


async def test_secrets_in_tool_output_are_redacted_before_the_model_sees_them(
    config: Config, workspace: Path, approve_all: PolicyApprover, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "sk-live-key-value-abcdef987654")
    (workspace / "files" / "config.txt").write_text(
        "some setting\nAPI_KEY=sk-live-key-value-abcdef987654\n"
    )
    runner = _run(
        config,
        [
            MockProvider.call("read_file", {"path": "files/config.txt"}),
            MockProvider.text("The file contains a redacted key."),
        ],
        approve_all,
    )
    result = await runner.run(TaskState(goal="read config.txt"))
    assert result.task.actions[0].ok is True
    conversation = "\n".join(m.content for m in result.messages)
    assert "sk-live-key-value-abcdef987654" not in conversation
    assert "REDACTED" in conversation


async def test_the_model_cannot_approve_its_own_action(
    config: Config, workspace: Path, deny_all: PolicyApprover
) -> None:
    """No tool reaches an approver, and a denial cannot be argued away."""
    runner = build_runner(
        config,
        provider=MockProvider(
            [
                MockProvider.call("write_file", {"path": "outputs/a.md", "content": "x"}),
                MockProvider.call("write_file", {"path": "outputs/a.md", "content": "x"}),
                MockProvider.text("Denied both times."),
            ]
        ),
        approver=deny_all,
    )
    tool_names = runner.tools.names()
    assert not any("approv" in name for name in tool_names)
    result = await runner.run(TaskState(goal="write despite denial"))
    assert not (workspace / "outputs" / "a.md").exists()
    assert all(
        a.error_category in {ErrorCategory.APPROVAL_DENIED, ErrorCategory.LIMIT_EXCEEDED}
        for a in result.task.actions
    )


async def test_the_system_prompt_matches_the_actual_tool_set(
    config: Config, approve_all: PolicyApprover
) -> None:
    """The prompt must describe the real registry, never a hard-coded list."""
    runner = build_runner(config, provider=MockProvider([]), approver=approve_all)
    prompt = runner.build_system_message().content
    for name in runner.tools.names():
        assert name in prompt
    assert "browser" not in prompt.lower().split("workspace")[0]


async def test_shell_output_is_captured_but_the_environment_is_scrubbed(
    config: Config, approve_all: PolicyApprover, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "sk-parent-visible-1234567")
    config.shell_allowed_commands = ["python3"]
    runner = _run(
        config,
        [
            MockProvider.call(
                "run_shell",
                {"command": ["python3", "-c", "import os; print(sorted(os.environ))"]},
            ),
            MockProvider.text("The child environment has no credentials."),
        ],
        approve_all,
    )
    result = await runner.run(TaskState(goal="inspect the environment"))
    conversation = "\n".join(m.content for m in result.messages)
    assert "GEMINI_API_KEY" not in conversation
    assert "sk-parent-visible-1234567" not in conversation
