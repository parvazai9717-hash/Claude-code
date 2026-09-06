"""The inspection and deletion commands, with real data present.

The existing CLI tests cover these commands' refusal paths — unknown task,
unknown fact, unknown session. These cover the other half: that each one does its
job when the data actually exists, and that the destructive ones confirm first.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent.cli import app
from agent.providers.mock import MockProvider

runner = CliRunner()


# `~` is redirected into a temporary directory by the shared `isolated_home`
# fixture in conftest.py, which also clears LOCAL_AGENT_* and resets CLI state.
pytestmark = pytest.mark.usefixtures("isolated_home")


def _run_a_task(scripted, *responses: object) -> str:  # type: ignore[no-untyped-def]
    """Execute a real run and return its task id."""
    scripted(*responses)
    result = runner.invoke(app, ["-p", "mock", "chat", "do some work"])
    assert result.exit_code == 0
    listed = runner.invoke(app, ["-p", "mock", "task", "list"])
    for line in listed.output.splitlines():
        if "task_" in line:
            return line.split("task_")[1].split()[0].strip().rstrip("│").strip()
    raise AssertionError("no task was recorded")


# -- task events / recover --------------------------------------------------
def test_task_events_shows_the_recorded_lifecycle(scripted) -> None:  # type: ignore[no-untyped-def]
    task_id = _run_a_task(
        scripted,
        MockProvider.call("get_current_time", {}),
        MockProvider.text("It is now."),
    )
    result = runner.invoke(app, ["-p", "mock", "task", "events", f"task_{task_id}"])
    assert result.exit_code == 0
    assert "task_started" in result.output
    assert "tool_result" in result.output
    assert "task_finished" in result.output


def test_task_recover_reports_nothing_when_all_is_well(scripted) -> None:  # type: ignore[no-untyped-def]
    _run_a_task(scripted, MockProvider.text("done"))
    result = runner.invoke(app, ["-p", "mock", "task", "recover"])
    assert result.exit_code == 0
    assert "nothing to recover" in result.output


def test_task_recover_rescues_a_task_left_running(
    scripted,
    tmp_path: Path,  # type: ignore[no-untyped-def]
) -> None:
    """A task stranded by a dead process must become resumable, not stay 'running'."""
    from agent.memory.database import Database
    from agent.memory.tasks import TaskStore
    from agent.task_state import TaskState, TaskStatus

    # Materialise the database the CLI will use, then strand a task in it.
    runner.invoke(app, ["-p", "mock", "config", "show"])
    database = Database(tmp_path / "home" / ".local-agent" / "agent.sqlite3")
    stranded = TaskState(goal="interrupted work")
    stranded.transition_to(TaskStatus.RUNNING)
    TaskStore(database).save(stranded)
    database.close()

    result = runner.invoke(app, ["-p", "mock", "task", "recover"])
    assert result.exit_code == 0
    assert stranded.id in result.output

    status = runner.invoke(app, ["-p", "mock", "task", "status", stranded.id])
    assert "paused" in status.output


# -- memory -----------------------------------------------------------------
def _store_a_fact(scripted) -> None:  # type: ignore[no-untyped-def]
    scripted(
        MockProvider.call("remember_fact", {"fact": "the user prefers metric units"}),
        MockProvider.text("Saved."),
    )
    assert runner.invoke(app, ["-p", "mock", "chat", "remember that"], input="y\n").exit_code == 0


def test_memory_list_shows_an_approved_fact(scripted) -> None:  # type: ignore[no-untyped-def]
    _store_a_fact(scripted)
    result = runner.invoke(app, ["-p", "mock", "memory", "list"])
    assert result.exit_code == 0
    assert "metric units" in result.output


def test_memory_remove_confirms_before_deleting(scripted) -> None:  # type: ignore[no-untyped-def]
    _store_a_fact(scripted)
    declined = runner.invoke(app, ["-p", "mock", "memory", "remove", "1"], input="n\n")
    assert "cancelled" in declined.output
    assert "metric units" in runner.invoke(app, ["-p", "mock", "memory", "list"]).output

    accepted = runner.invoke(app, ["-p", "mock", "memory", "remove", "1"], input="y\n")
    assert "deleted fact 1" in accepted.output
    assert "no durable facts" in runner.invoke(app, ["-p", "mock", "memory", "list"]).output


# -- sessions ---------------------------------------------------------------
def _session_id(output: str) -> str:
    for line in output.splitlines():
        if "sess_" in line:
            return "sess_" + line.split("sess_")[1].split()[0].strip().rstrip("│").strip()
    raise AssertionError("no session was recorded")


def test_sessions_show_prints_the_conversation(scripted) -> None:  # type: ignore[no-untyped-def]
    scripted(
        MockProvider.call("get_current_time", {}),
        MockProvider.text("The time is recorded above."),
    )
    runner.invoke(app, ["-p", "mock", "chat", "what time is it"])
    session = _session_id(runner.invoke(app, ["-p", "mock", "sessions", "list"]).output)

    result = runner.invoke(app, ["-p", "mock", "sessions", "show", session])
    assert result.exit_code == 0
    assert "user" in result.output
    assert "get_current_time" in result.output  # the tool call is shown
    assert "The time is recorded above." in result.output


def test_sessions_delete_confirms_before_deleting(scripted) -> None:  # type: ignore[no-untyped-def]
    scripted(MockProvider.text("Hello."))
    runner.invoke(app, ["-p", "mock", "chat", "say hello"])
    session = _session_id(runner.invoke(app, ["-p", "mock", "sessions", "list"]).output)

    declined = runner.invoke(app, ["-p", "mock", "sessions", "delete", session], input="n\n")
    assert "cancelled" in declined.output
    assert session in runner.invoke(app, ["-p", "mock", "sessions", "list"]).output

    accepted = runner.invoke(app, ["-p", "mock", "sessions", "delete", session], input="y\n")
    assert f"deleted session {session}" in accepted.output
    assert "no sessions" in runner.invoke(app, ["-p", "mock", "sessions", "list"]).output


# -- skills -----------------------------------------------------------------
def test_skills_show_prints_metadata_and_workflow(tmp_path: Path) -> None:
    skill = tmp_path / "skills" / "reporter"
    skill.mkdir(parents=True)
    skill.write_text if False else (skill / "SKILL.md").write_text(
        "---\n"
        "name: reporter\n"
        "description: Write a report from workspace files.\n"
        "required_tools:\n  - read_file\n"
        "limitations:\n  - read-only\n"
        "---\n\n"
        "# Reporter\n\nStep one: list the files.\n"
    )
    result = runner.invoke(app, ["-p", "mock", "skills", "show", "reporter"])
    assert result.exit_code == 0
    assert "Write a report" in result.output
    assert "needs tools: read_file" in result.output
    assert "limitations: read-only" in result.output
    assert "Step one: list the files." in result.output
