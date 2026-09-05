"""Shell execution: policy, timeout, output limits, environment scrubbing."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.config import Config
from agent.errors import PathEscapeError, PermissionDeniedError
from agent.tools.base import ToolContext
from agent.tools.shell import RunShellTool, build_child_env


def test_child_environment_drops_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "sk-secret-value-9999")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "another-secret")
    env = build_child_env("/tmp/ws")
    assert "GEMINI_API_KEY" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert "sk-secret-value-9999" not in str(env)
    assert env["HOME"] == "/tmp/ws", "HOME points at the workspace, not the real home"
    assert env["LOCAL_AGENT_SANDBOX"] == "1"


async def test_allowlisted_command_runs(context: ToolContext, workspace: Path) -> None:
    (workspace / "files" / "a.txt").write_text("hello")
    output = await RunShellTool().run({"command": ["ls", "files"], "cwd": "."}, context)
    assert output["exit_code"] == 0 and output["success"] is True
    assert "a.txt" in output["stdout"]


async def test_non_zero_exit_is_reported_not_hidden(context: ToolContext) -> None:
    output = await RunShellTool().run({"command": ["ls", "definitely-missing"]}, context)
    assert output["exit_code"] != 0
    assert output["success"] is False


async def test_command_outside_allowlist_is_refused(context: ToolContext) -> None:
    with pytest.raises(PermissionDeniedError, match="allowlist"):
        await RunShellTool().run({"command": ["nmap", "-p80", "localhost"]}, context)


async def test_forbidden_command_is_refused_even_if_allowlisted(context: ToolContext) -> None:
    context.config.shell_allowed_commands = ["ls"]
    with pytest.raises(PermissionDeniedError):
        await RunShellTool().run({"command": ["rm", "-rf", "/"]}, context)


async def test_metacharacters_are_refused(context: ToolContext) -> None:
    with pytest.raises(PermissionDeniedError, match="metacharacters"):
        await RunShellTool().run({"command": "ls | grep x"}, context)


async def test_uninstalled_program_is_reported(context: ToolContext) -> None:
    context.config.shell_allowed_commands = ["definitely_not_installed_xyz"]
    with pytest.raises(Exception, match="not installed"):
        await RunShellTool().run({"command": ["definitely_not_installed_xyz"]}, context)


async def test_cwd_must_stay_inside_the_workspace(context: ToolContext) -> None:
    with pytest.raises(PathEscapeError):
        await RunShellTool().run({"command": ["ls"], "cwd": "../.."}, context)


async def test_timeout_kills_the_process(context: ToolContext, config: Config) -> None:
    config.limits.tool_timeout_seconds = 0.5
    config.shell_allowed_commands = ["python3"]
    output = await RunShellTool().run(
        {"command": ["python3", "-c", "import time; time.sleep(30)"]}, context
    )
    assert output["timed_out"] is True and output["success"] is False


async def test_output_is_truncated(context: ToolContext, config: Config) -> None:
    config.limits.max_tool_output_chars = 200
    config.shell_allowed_commands = ["python3"]
    output = await RunShellTool().run({"command": ["python3", "-c", "print('x' * 5000)"]}, context)
    assert output["output_truncated"] is True
    assert len(output["stdout"]) < 500


async def test_process_runs_in_its_own_session(context: ToolContext, config: Config) -> None:
    """A separate process group is what makes killing a timed-out tree possible."""
    config.shell_allowed_commands = ["python3"]
    output = await RunShellTool().run(
        {"command": ["python3", "-c", "import os; print(os.getpid() == os.getpgid(0))"]},
        context,
    )
    assert output["stdout"].strip() == "True"


async def test_child_cannot_see_a_parent_secret(
    context: ToolContext, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "sk-parent-secret-1234")
    config.shell_allowed_commands = ["python3"]
    output = await RunShellTool().run(
        {
            "command": [
                "python3",
                "-c",
                "import os; print(os.environ.get('GEMINI_API_KEY', 'ABSENT'))",
            ]
        },
        context,
    )
    assert output["stdout"].strip() == "ABSENT"


def test_shell_tool_classification() -> None:
    definition = RunShellTool().definition()
    assert definition.requires_approval is True
    assert definition.requires_verification is True
    assert definition.read_only is False
    assert definition.reversible is False
