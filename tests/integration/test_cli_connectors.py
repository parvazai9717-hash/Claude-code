"""The connector CLI commands."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    for key in list(os.environ):
        if key.startswith("LOCAL_AGENT_"):
            monkeypatch.delenv(key, raising=False)
    yield home
    from agent import cli

    cli._state.clear()


def _invoke(*args: str):  # type: ignore[no-untyped-def]
    return runner.invoke(app, ["-p", "mock", *args])


def test_listing_is_empty_initially() -> None:
    result = _invoke("connectors", "list")
    assert result.exit_code == 0
    assert "no connectors configured" in result.output


def test_adding_a_stdio_connector() -> None:
    result = _invoke(
        "connectors",
        "add",
        "github",
        "--command",
        "npx",
        "--arg",
        "-y",
        "--arg",
        "server-github",
        "--env",
        "GITHUB_TOKEN",
    )
    assert result.exit_code == 0
    assert "added connector github" in result.output
    assert "GITHUB_TOKEN" in result.output
    listed = _invoke("connectors", "list")
    assert "github" in listed.output
    assert "mcp_stdio" in listed.output


def test_a_new_connector_is_disabled(tmp_path: Path) -> None:
    _invoke("connectors", "add", "demo", "--command", "npx")
    result = _invoke("connectors", "list", "--json")
    assert '"enabled": false' in result.output.lower()


def test_adding_an_http_connector() -> None:
    result = _invoke(
        "connectors",
        "add",
        "remote",
        "--url",
        "https://example.com/mcp",
        "--header-env",
        "Authorization=MY_TOKEN",
    )
    assert result.exit_code == 0
    assert "MY_TOKEN" in result.output


def test_exactly_one_transport_is_required() -> None:
    neither = _invoke("connectors", "add", "demo")
    assert neither.exit_code != 0 and "exactly one" in neither.output
    both = _invoke("connectors", "add", "demo", "--command", "npx", "--url", "https://x.com")
    assert both.exit_code != 0


def test_a_malformed_header_env_is_rejected() -> None:
    result = _invoke(
        "connectors", "add", "demo", "--url", "https://x.com", "--header-env", "nonsense"
    )
    assert result.exit_code != 0
    assert "HEADER=ENV_VAR" in result.output


def test_an_invalid_name_is_rejected() -> None:
    result = _invoke("connectors", "add", "Bad Name", "--command", "npx")
    assert result.exit_code != 0
    assert "invalid connector definition" in result.output


def test_a_path_qualified_command_is_rejected() -> None:
    result = _invoke("connectors", "add", "demo", "--command", "/usr/bin/thing")
    assert result.exit_code != 0
    assert "bare program name" in result.output


def test_a_duplicate_needs_replace() -> None:
    _invoke("connectors", "add", "demo", "--command", "npx")
    duplicate = _invoke("connectors", "add", "demo", "--command", "npx")
    assert duplicate.exit_code != 0 and "already exists" in duplicate.output
    replaced = _invoke("connectors", "add", "demo", "--command", "uvx", "--replace")
    assert replaced.exit_code == 0


def test_enable_and_disable() -> None:
    _invoke("connectors", "add", "demo", "--command", "npx")
    enabled = _invoke("connectors", "enable", "demo")
    assert enabled.exit_code == 0 and "require approval" in enabled.output
    assert '"enabled": true' in _invoke("connectors", "list", "--json").output.lower()
    disabled = _invoke("connectors", "disable", "demo")
    assert disabled.exit_code == 0
    assert '"enabled": false' in _invoke("connectors", "list", "--json").output.lower()


def test_enable_rejects_an_unknown_connector() -> None:
    result = _invoke("connectors", "enable", "nope")
    assert result.exit_code != 0 and "no connector named" in result.output


def test_remove_confirms_first() -> None:
    _invoke("connectors", "add", "demo", "--command", "npx")
    declined = runner.invoke(app, ["-p", "mock", "connectors", "remove", "demo"], input="n\n")
    assert "cancelled" in declined.output
    assert "demo" in _invoke("connectors", "list").output

    accepted = _invoke("connectors", "remove", "demo", "--yes")
    assert accepted.exit_code == 0
    assert "no connectors configured" in _invoke("connectors", "list").output


def test_remove_rejects_an_unknown_connector() -> None:
    result = _invoke("connectors", "remove", "nope", "--yes")
    assert result.exit_code != 0 and "no connector named" in result.output


def test_test_reports_a_disabled_connector() -> None:
    _invoke("connectors", "add", "demo", "--command", "npx")
    result = _invoke("connectors", "test")
    assert result.exit_code == 0
    assert "disabled" in result.output


def test_doctor_reports_connectors_and_media() -> None:
    _invoke("connectors", "add", "demo", "--command", "npx", "--enable")
    result = _invoke("doctor")
    assert result.exit_code == 0
    assert "connectors" in result.output
    assert "media input" in result.output
    assert "video disabled" in result.output


def test_the_read_only_flag_is_recorded() -> None:
    _invoke(
        "connectors",
        "add",
        "demo",
        "--command",
        "npx",
        "--read-only",
        "search",
        "--allow",
        "search",
    )
    output = _invoke("connectors", "list", "--json").output
    assert '"read_only_tools"' in output and '"search"' in output
