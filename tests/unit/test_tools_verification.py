"""Verification: real evidence, or an explicit statement that it is unavailable."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.config import Config
from agent.errors import InvalidArgumentsError, PathEscapeError
from agent.tools.base import ToolContext
from agent.tools.verification import VerifyResultTool


async def test_file_exists_passes(context: ToolContext, workspace: Path) -> None:
    (workspace / "outputs" / "a.md").write_text("data")
    output = await VerifyResultTool().run(
        {"method": "file_exists", "path": "outputs/a.md"}, context
    )
    assert output["verified"] is True and "exists" in output["evidence"]


async def test_file_exists_fails_for_missing(context: ToolContext) -> None:
    output = await VerifyResultTool().run(
        {"method": "file_exists", "path": "outputs/missing.md"}, context
    )
    assert output["verified"] is False


async def test_file_absent(context: ToolContext, workspace: Path) -> None:
    assert (await VerifyResultTool().run({"method": "file_absent", "path": "outputs/x"}, context))[
        "verified"
    ] is True
    (workspace / "outputs" / "x").write_text("here")
    assert (await VerifyResultTool().run({"method": "file_absent", "path": "outputs/x"}, context))[
        "verified"
    ] is False


async def test_file_contains_reports_the_line(context: ToolContext, workspace: Path) -> None:
    (workspace / "outputs" / "a.md").write_text("first\nneedle here\n")
    output = await VerifyResultTool().run(
        {"method": "file_contains", "path": "outputs/a.md", "expected": "needle"}, context
    )
    assert output["verified"] is True and output["line"] == 2


async def test_file_contains_fails_when_absent(context: ToolContext, workspace: Path) -> None:
    (workspace / "outputs" / "a.md").write_text("nothing here")
    output = await VerifyResultTool().run(
        {"method": "file_contains", "path": "outputs/a.md", "expected": "needle"}, context
    )
    assert output["verified"] is False
    assert "does not contain" in output["evidence"]


async def test_file_contains_requires_expected(context: ToolContext) -> None:
    with pytest.raises(InvalidArgumentsError):
        await VerifyResultTool().run({"method": "file_contains", "path": "outputs/a.md"}, context)


async def test_binary_file_reports_evidence_unavailable(
    context: ToolContext, workspace: Path
) -> None:
    (workspace / "outputs" / "b.bin").write_bytes(b"\x00needle")
    output = await VerifyResultTool().run(
        {"method": "file_contains", "path": "outputs/b.bin", "expected": "needle"}, context
    )
    assert output["verified"] is False
    assert output["unavailable_reason"] == "binary file"


async def test_command_succeeds(context: ToolContext, config: Config) -> None:
    config.shell_allowed_commands = ["python3"]
    output = await VerifyResultTool().run(
        {"method": "command_succeeds", "command": ["python3", "-c", "print(1)"]}, context
    )
    assert output["verified"] is True and output["exit_code"] == 0


async def test_command_failure_is_not_verified(context: ToolContext, config: Config) -> None:
    config.shell_allowed_commands = ["python3"]
    output = await VerifyResultTool().run(
        {"method": "command_succeeds", "command": ["python3", "-c", "raise SystemExit(3)"]},
        context,
    )
    assert output["verified"] is False and output["exit_code"] == 3


async def test_disallowed_verification_command_is_unavailable_not_verified(
    context: ToolContext,
) -> None:
    output = await VerifyResultTool().run(
        {"method": "command_succeeds", "command": ["curl", "http://example.com"]}, context
    )
    assert output["verified"] is False
    assert "cannot verify by command" in output["unavailable_reason"]


async def test_verification_cannot_escape_the_workspace(context: ToolContext) -> None:
    with pytest.raises(PathEscapeError):
        await VerifyResultTool().run({"method": "file_exists", "path": "../../etc/passwd"}, context)


def test_verification_is_read_only_and_needs_no_approval() -> None:
    definition = VerifyResultTool().definition()
    assert definition.read_only is True
    assert definition.requires_approval is False
