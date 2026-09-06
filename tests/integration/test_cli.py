"""CLI behaviour, exercised through Typer's test runner.

Every invocation is pointed at a temporary HOME, workspace and database, so the
suite never reads or writes the real user environment.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent.cli import app

runner = CliRunner()


# `~` is redirected into a temporary directory by the shared `isolated_home`
# fixture in conftest.py, which also clears LOCAL_AGENT_* and resets CLI state.
pytestmark = pytest.mark.usefixtures("isolated_home")


def _invoke(*args: str):  # type: ignore[no-untyped-def]
    return runner.invoke(app, list(args))


def test_help_lists_the_commands() -> None:
    result = _invoke("--help")
    assert result.exit_code == 0
    for command in ("chat", "doctor", "task", "memory", "sessions", "config", "clear-data"):
        assert command in result.output


def test_version() -> None:
    result = _invoke("--version")
    assert result.exit_code == 0 and "local-agent" in result.output


def test_doctor_runs_offline_with_the_mock_provider() -> None:
    result = _invoke("-p", "mock", "doctor")
    assert result.exit_code == 0
    assert "workspace" in result.output
    assert "database" in result.output
    assert "not enabled in this release" in result.output


def test_doctor_reports_a_missing_gemini_key() -> None:
    result = _invoke("-p", "gemini", "doctor")
    assert result.exit_code == 0
    assert "GEMINI_API_KEY is not set" in result.output


def test_config_show_never_prints_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "sk-should-never-appear-1234")
    result = _invoke("-p", "gemini", "config", "show")
    assert result.exit_code == 0
    assert "sk-should-never-appear-1234" not in result.output
    assert "GEMINI_API_KEY" in result.output


def test_config_show_json() -> None:
    result = _invoke("-p", "mock", "config", "show", "--json")
    assert result.exit_code == 0 and '"provider"' in result.output


def test_invalid_provider_is_rejected() -> None:
    result = _invoke("-p", "not-a-provider", "config", "show")
    assert result.exit_code != 0


def test_invalid_approval_mode_is_rejected() -> None:
    result = _invoke("-a", "whatever", "doctor")
    assert result.exit_code != 0


def test_task_create_list_status_and_cancel() -> None:
    created = _invoke("-p", "mock", "task", "create", "a test goal")
    assert created.exit_code == 0
    task_id = created.output.split("created task ")[1].split()[0].strip()

    listed = _invoke("-p", "mock", "task", "list")
    assert task_id[:12] in listed.output

    status = _invoke("-p", "mock", "task", "status", task_id)
    assert status.exit_code == 0 and "a test goal" in status.output

    cancelled = _invoke("-p", "mock", "task", "cancel", task_id)
    assert cancelled.exit_code == 0

    again = _invoke("-p", "mock", "task", "run", task_id)
    assert again.exit_code != 0
    assert "already cancelled" in again.output


def test_task_commands_reject_an_unknown_id() -> None:
    for command in ("status", "pause", "resume", "cancel", "events", "run"):
        result = _invoke("-p", "mock", "task", command, "task_doesnotexist")
        assert result.exit_code != 0
        assert "no task matching" in result.output


def test_task_pause_marks_the_flag() -> None:
    created = _invoke("-p", "mock", "task", "create", "pausable")
    task_id = created.output.split("created task ")[1].split()[0].strip()
    paused = _invoke("-p", "mock", "task", "pause", task_id)
    assert paused.exit_code == 0 and "pause requested" in paused.output


def test_resume_refuses_a_task_that_is_not_paused() -> None:
    created = _invoke("-p", "mock", "task", "create", "not paused")
    task_id = created.output.split("created task ")[1].split()[0].strip()
    result = _invoke("-p", "mock", "task", "resume", task_id)
    assert result.exit_code != 0 and "not paused" in result.output


def test_memory_list_is_empty_initially() -> None:
    result = _invoke("-p", "mock", "memory", "list")
    assert result.exit_code == 0 and "no durable facts" in result.output


def test_memory_remove_rejects_an_unknown_id() -> None:
    result = _invoke("-p", "mock", "memory", "remove", "999")
    assert result.exit_code != 0 and "no fact with id" in result.output


def test_sessions_list_is_empty_initially() -> None:
    result = _invoke("-p", "mock", "sessions", "list")
    assert result.exit_code == 0 and "no sessions" in result.output


def test_sessions_show_rejects_an_unknown_id() -> None:
    result = _invoke("-p", "mock", "sessions", "show", "sess_nope")
    assert result.exit_code != 0 and "no session named" in result.output


def test_skills_list_reads_the_configured_directory(tmp_path: Path) -> None:
    skills = tmp_path / "skills" / "demo"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text(
        "---\nname: demo\ndescription: A demo skill.\nrequired_tools:\n  - read_file\n---\nbody\n"
    )
    result = _invoke("-p", "mock", "skills", "list")
    assert result.exit_code == 0 and "demo" in result.output


def test_skills_list_reports_a_broken_skill(tmp_path: Path) -> None:
    broken = tmp_path / "skills" / "broken"
    broken.mkdir(parents=True)
    (broken / "SKILL.md").write_text("no front matter")
    result = _invoke("-p", "mock", "skills", "list")
    assert "invalid" in result.output


def test_skills_show_rejects_an_unknown_skill() -> None:
    result = _invoke("-p", "mock", "skills", "show", "nope")
    assert result.exit_code != 0


def test_clear_data_requires_confirmation() -> None:
    _invoke("-p", "mock", "task", "create", "something")
    declined = runner.invoke(app, ["-p", "mock", "clear-data"], input="n\n")
    assert "cancelled" in declined.output
    assert "task" in _invoke("-p", "mock", "task", "list").output.lower()

    accepted = _invoke("-p", "mock", "clear-data", "--yes")
    assert accepted.exit_code == 0 and "deleted" in accepted.output
    assert "no tasks" in _invoke("-p", "mock", "task", "list").output


def test_clear_data_on_an_empty_database() -> None:
    result = _invoke("-p", "mock", "clear-data", "--yes")
    assert result.exit_code == 0 and "nothing to delete" in result.output


def test_a_config_file_is_picked_up(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("provider: mock\nmax_steps: 9\n")
    result = _invoke("config", "show")
    assert result.exit_code == 0
    assert "mock" in result.output


def test_environment_overrides_the_config_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "config.yaml").write_text("provider: ollama\n")
    monkeypatch.setenv("LOCAL_AGENT_PROVIDER", "mock")
    result = _invoke("config", "show")
    assert "mock" in result.output


def test_the_workspace_is_created_on_first_use(tmp_path: Path) -> None:
    _invoke("-p", "mock", "config", "show")
    workspace = tmp_path / "workspace"
    assert workspace.is_dir()
    for name in ("files", "projects", "downloads", "outputs", "temp", "state"):
        assert (workspace / name).is_dir()


def test_clear_data_keeps_connectors_but_says_so() -> None:
    """A command claiming to clear everything must not silently leave data behind."""
    _invoke("-p", "mock", "connectors", "add", "leftover", "--command", "npx")
    _invoke("-p", "mock", "task", "create", "something")

    result = _invoke("-p", "mock", "clear-data", "--yes")
    assert result.exit_code == 0
    assert "kept" in result.output
    assert "leftover" in result.output
    assert "--connectors" in result.output, "the way to remove them must be shown"

    # The connector really is still there.
    assert "leftover" in _invoke("-p", "mock", "connectors", "list").output


def test_clear_data_removes_connectors_when_asked() -> None:
    _invoke("-p", "mock", "connectors", "add", "leftover", "--command", "npx")
    _invoke("-p", "mock", "task", "create", "something")

    result = _invoke("-p", "mock", "clear-data", "--yes", "--connectors")
    assert result.exit_code == 0
    assert "connector definition" in result.output
    assert "no connectors configured" in _invoke("-p", "mock", "connectors", "list").output


def test_clear_data_on_an_empty_database_still_reports_kept_connectors() -> None:
    _invoke("-p", "mock", "connectors", "add", "leftover", "--command", "npx")
    result = _invoke("-p", "mock", "clear-data", "--yes")
    assert "nothing to delete" in result.output
    assert "kept" in result.output
