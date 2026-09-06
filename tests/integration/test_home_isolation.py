"""The test suite must never write to the real home directory.

This exists because it once did. `os.path.expanduser` — which backs both
`Path.home()` and `Path.expanduser()` — reads a *different environment variable
per platform*: `HOME` on POSIX, but `USERPROFILE` (falling back to `HOMEDRIVE` +
`HOMEPATH`) on Windows, where `HOME` is ignored outright.

The CLI fixtures originally set only `HOME`. On Linux and macOS that isolated
everything and the suite looked clean. On Windows it isolated nothing: the CLI's
default `data_dir` of `~/.local-agent` resolved to the real user profile, and a
plain `pytest` run left its tasks, facts and connectors in it.

These tests assert the property directly, so the same mistake cannot pass again
on any platform.
"""

from __future__ import annotations

import ntpath
import os
import posixpath
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent.cli import app

runner = CliRunner()

pytestmark = pytest.mark.usefixtures("isolated_home")


def test_tilde_resolves_inside_the_temporary_home(tmp_path: Path) -> None:
    resolved = Path("~").expanduser().resolve()
    assert resolved == (tmp_path / "home").resolve()


def test_every_variable_expanduser_consults_is_redirected(tmp_path: Path) -> None:
    """Set them all, because which one is consulted depends on the platform."""
    home = (tmp_path / "home").resolve()
    for variable in ("HOME", "USERPROFILE"):
        assert Path(os.environ[variable]).resolve() == home, f"{variable} not redirected"
    # HOMEDRIVE + HOMEPATH is the Windows fallback when USERPROFILE is absent.
    assert "HOMEPATH" in os.environ


def test_posix_and_windows_expansion_both_land_in_the_temporary_home(
    tmp_path: Path,
) -> None:
    """Check the rule each platform actually applies, not just the current one.

    `posixpath` reads HOME; `ntpath` reads USERPROFILE. Exercising both here
    means a Linux CI run still catches a Windows-only isolation hole.
    """
    home = (tmp_path / "home").resolve()
    assert Path(posixpath.expanduser("~")).resolve() == home
    assert Path(ntpath.expanduser("~")).resolve() == home


def test_setting_only_home_would_not_isolate_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The precise mistake that was made, pinned so it cannot be made again."""
    elsewhere = tmp_path / "not-the-temporary-home"
    elsewhere.mkdir()
    monkeypatch.setenv("USERPROFILE", str(elsewhere))
    # POSIX expansion still follows HOME and looks fine...
    assert Path(posixpath.expanduser("~")).resolve() == (tmp_path / "home").resolve()
    # ...while Windows expansion has escaped. This asymmetry is the whole bug.
    assert Path(ntpath.expanduser("~")).resolve() == elsewhere.resolve()


def test_the_cli_writes_its_database_inside_the_temporary_home(tmp_path: Path) -> None:
    """The symptom a user actually saw: a real profile full of test data."""
    result = runner.invoke(app, ["-p", "mock", "task", "create", "a throwaway task"])
    assert result.exit_code == 0

    database = tmp_path / "home" / ".local-agent" / "agent.sqlite3"
    assert database.is_file(), "the CLI database did not land in the temporary home"

    # And nothing was created beside the temporary tree.
    assert not (Path.cwd().parent / ".local-agent").exists()


def test_the_cli_writes_its_connectors_inside_the_temporary_home(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["-p", "mock", "connectors", "add", "throwaway", "--command", "npx"]
    )
    assert result.exit_code == 0
    assert (tmp_path / "home" / ".local-agent" / "connectors.json").is_file()
