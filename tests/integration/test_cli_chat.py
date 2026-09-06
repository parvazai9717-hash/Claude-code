"""The `chat` command end to end.

`chat` is the primary entry point, so it is exercised here through Typer's runner
with a scripted provider: a real run, real tools, real security layer, real
SQLite — only the model is fake. The interactive loop and its slash commands are
driven through stdin.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent.cli import app
from agent.providers.mock import MockProvider

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    for key in list(os.environ):
        if key.startswith("LOCAL_AGENT_"):
            monkeypatch.delenv(key, raising=False)
    yield home
    from agent import cli

    cli._state.clear()


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """Replace the provider factory with a scripted mock, keeping everything else real."""

    def install(*responses: object) -> MockProvider:
        provider = MockProvider(list(responses))
        monkeypatch.setattr("agent.cli.create_provider", lambda config, **kw: provider)
        return provider

    return install


def test_a_one_shot_goal_runs_the_whole_loop(
    tmp_path: Path,
    scripted,  # type: ignore[no-untyped-def]
) -> None:
    workspace = tmp_path / "workspace" / "files"
    workspace.mkdir(parents=True)
    (workspace / "notes.txt").write_text("the answer is 42")

    scripted(
        MockProvider.call("read_file", {"path": "files/notes.txt"}),
        MockProvider.text("The notes say the answer is 42."),
    )
    result = runner.invoke(app, ["-p", "mock", "chat", "what do the notes say?"])
    assert result.exit_code == 0
    assert "read_file" in result.output  # progress events are shown
    assert "answer is 42" in result.output  # the final answer is shown
    assert "completed" in result.output  # the outcome is stated


def test_a_write_is_approved_through_the_prompt(
    tmp_path: Path,
    scripted,  # type: ignore[no-untyped-def]
) -> None:
    """The human is asked, and answering 'y' lets the write through."""
    scripted(
        MockProvider.call("write_file", {"path": "outputs/r.md", "content": "written"}),
        MockProvider.call(
            "verify_result",
            {"method": "file_contains", "path": "outputs/r.md", "expected": "written"},
        ),
        MockProvider.text("Wrote and verified outputs/r.md."),
    )
    result = runner.invoke(app, ["-p", "mock", "chat", "write a report"], input="y\n")
    assert result.exit_code == 0
    assert "Approval required" in result.output
    assert (tmp_path / "workspace" / "outputs" / "r.md").read_text() == "written"


def test_declining_the_prompt_prevents_the_write(
    tmp_path: Path,
    scripted,  # type: ignore[no-untyped-def]
) -> None:
    scripted(
        MockProvider.call("write_file", {"path": "outputs/r.md", "content": "written"}),
        MockProvider.text("The write was denied."),
    )
    result = runner.invoke(app, ["-p", "mock", "chat", "write a report"], input="n\n")
    assert result.exit_code == 0
    assert not (tmp_path / "workspace" / "outputs" / "r.md").exists()
    assert "Denied by the human" in result.output


def test_a_run_is_persisted_and_can_be_inspected(
    tmp_path: Path,
    scripted,  # type: ignore[no-untyped-def]
) -> None:
    scripted(MockProvider.text("Hello."))
    assert runner.invoke(app, ["-p", "mock", "chat", "say hello"]).exit_code == 0

    sessions = runner.invoke(app, ["-p", "mock", "sessions", "list"])
    assert "say hello" in sessions.output

    tasks = runner.invoke(app, ["-p", "mock", "task", "list"])
    assert "completed" in tasks.output


def test_a_provider_failure_is_reported_not_crashed(
    scripted,  # type: ignore[no-untyped-def]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "agent.cli.create_provider", lambda config, **kw: MockProvider(healthy=False)
    )
    result = runner.invoke(app, ["-p", "mock", "chat", "anything"])
    assert result.exit_code == 0
    assert "failed" in result.output


def test_the_unsafe_flag_warns_loudly(
    tmp_path: Path,
    scripted,  # type: ignore[no-untyped-def]
) -> None:
    scripted(
        MockProvider.call("write_file", {"path": "outputs/r.md", "content": "x"}),
        MockProvider.text("Done."),
    )
    result = runner.invoke(app, ["-p", "mock", "chat", "write a file", "--unsafe-no-approvals"])
    assert "WARNING" in result.output
    assert "unsafe" in result.output.lower()
    # It really does skip the prompt: the write lands with no input supplied.
    assert (tmp_path / "workspace" / "outputs" / "r.md").exists()


# -- the interactive loop ---------------------------------------------------
def test_interactive_session_accepts_a_goal_then_quits(
    tmp_path: Path,
    scripted,  # type: ignore[no-untyped-def]
) -> None:
    scripted(MockProvider.text("Answered."))
    result = runner.invoke(app, ["-p", "mock", "chat"], input="do a thing\n/quit\n")
    assert result.exit_code == 0
    assert "Answered." in result.output


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("/help", "/status"),
        ("/status", "approval mode"),
        ("/tools", "read_file"),
        ("/model", "mock"),
        ("/memory", "no durable facts"),
        ("/tasks", "no tasks"),
        ("/clear", "conversation cleared"),
    ],
)
def test_each_slash_command(
    scripted,
    command: str,
    expected: str,  # type: ignore[no-untyped-def]
) -> None:
    scripted()
    result = runner.invoke(app, ["-p", "mock", "chat"], input=f"{command}\n/quit\n")
    assert result.exit_code == 0
    assert expected in result.output


def test_an_unknown_slash_command_is_reported(scripted) -> None:  # type: ignore[no-untyped-def]
    scripted()
    result = runner.invoke(app, ["-p", "mock", "chat"], input="/nonsense\n/quit\n")
    assert "unknown command" in result.output


def test_blank_input_is_ignored(scripted) -> None:  # type: ignore[no-untyped-def]
    scripted(MockProvider.text("Answered."))
    result = runner.invoke(app, ["-p", "mock", "chat"], input="\n\n/quit\n")
    assert result.exit_code == 0


def test_end_of_input_exits_cleanly(scripted) -> None:  # type: ignore[no-untyped-def]
    """Ctrl-D must leave the session without a traceback."""
    scripted()
    result = runner.invoke(app, ["-p", "mock", "chat"], input="")
    assert result.exit_code == 0
    assert "bye" in result.output


def test_resuming_an_unknown_session_is_refused(scripted) -> None:  # type: ignore[no-untyped-def]
    scripted()
    result = runner.invoke(app, ["-p", "mock", "chat", "hi", "--session", "sess_nope"])
    assert result.exit_code != 0
    assert "no session named" in result.output
