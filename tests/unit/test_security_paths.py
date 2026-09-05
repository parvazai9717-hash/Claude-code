"""Workspace containment: traversal, symlinks, credential files, size limits."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.errors import LimitExceededError, PathEscapeError, SecretProtectedError
from agent.security.limits import enforce_file_size, truncate_output
from agent.security.paths import PathPolicy, WorkspacePaths


@pytest.fixture
def policy(workspace: Path) -> PathPolicy:
    return PathPolicy.for_workspace(workspace, max_file_bytes=1000)


def test_relative_path_resolves_inside(policy: PathPolicy, workspace: Path) -> None:
    assert policy.resolve("files/a.txt") == (workspace / "files" / "a.txt").resolve()


@pytest.mark.parametrize(
    "candidate",
    [
        "../outside.txt",
        "files/../../outside.txt",
        "../../etc/passwd",
        "files/../../../root/.bashrc",
    ],
)
def test_traversal_is_rejected(policy: PathPolicy, candidate: str) -> None:
    with pytest.raises(PathEscapeError):
        policy.resolve(candidate)


def test_absolute_path_outside_is_rejected(policy: PathPolicy) -> None:
    with pytest.raises(PathEscapeError):
        policy.resolve("/etc/passwd")


def test_absolute_path_inside_is_allowed(policy: PathPolicy, workspace: Path) -> None:
    target = workspace / "files" / "ok.txt"
    assert policy.resolve(str(target)) == target.resolve()


def test_home_relative_path_is_rejected(policy: PathPolicy) -> None:
    with pytest.raises(PathEscapeError, match="home-relative"):
        policy.resolve("~/.ssh/id_rsa")


def test_empty_and_null_paths_are_rejected(policy: PathPolicy) -> None:
    with pytest.raises(PathEscapeError):
        policy.resolve("")
    with pytest.raises(PathEscapeError):
        policy.resolve("files/a\x00.txt")


def test_symlink_escape_is_rejected(policy: PathPolicy, workspace: Path, tmp_path: Path) -> None:
    secret = tmp_path / "outside_secret.txt"
    secret.write_text("secret")
    link = workspace / "files" / "escape"
    link.symlink_to(secret)
    with pytest.raises(PathEscapeError):
        policy.resolve("files/escape")


def test_symlink_inside_workspace_is_still_refused_by_default(
    policy: PathPolicy, workspace: Path
) -> None:
    """Even a link that stays inside is refused unless symlinks are enabled."""
    target = workspace / "files" / "real.txt"
    target.write_text("data")
    link = workspace / "files" / "link.txt"
    link.symlink_to(target)
    with pytest.raises(PathEscapeError, match="symbolic link"):
        policy.resolve("files/link.txt")


def test_symlink_through_a_parent_directory_is_rejected(
    policy: PathPolicy, workspace: Path, tmp_path: Path
) -> None:
    outside_dir = tmp_path / "outside_dir"
    outside_dir.mkdir()
    (outside_dir / "f.txt").write_text("x")
    (workspace / "files" / "linkdir").symlink_to(outside_dir)
    with pytest.raises(PathEscapeError):
        policy.resolve("files/linkdir/f.txt")


@pytest.mark.parametrize(
    "name",
    [".env", ".env.production", "id_rsa", "credentials.json", "key.pem", "server.key", ".netrc"],
)
def test_credential_files_are_protected(policy: PathPolicy, name: str) -> None:
    with pytest.raises(SecretProtectedError):
        policy.resolve(f"files/{name}")


@pytest.mark.parametrize("directory", [".ssh", ".aws", ".gnupg", ".docker"])
def test_credential_directories_are_protected(policy: PathPolicy, directory: str) -> None:
    with pytest.raises(SecretProtectedError):
        policy.resolve(f"{directory}/config")


def test_hidden_files_are_excluded_by_default(policy: PathPolicy) -> None:
    with pytest.raises(SecretProtectedError, match="hidden"):
        policy.resolve("files/.hidden")


def test_hidden_files_can_be_allowed_explicitly(workspace: Path) -> None:
    policy = PathPolicy.for_workspace(workspace, allow_hidden=True)
    assert policy.resolve("files/.hidden").name == ".hidden"
    # A credential file stays refused even when hidden files are permitted.
    with pytest.raises(SecretProtectedError):
        policy.resolve("files/.env")


def test_must_exist_raises_for_missing(policy: PathPolicy) -> None:
    with pytest.raises(FileNotFoundError):
        policy.resolve("files/missing.txt", must_exist=True)


def test_relative_display_never_leaks_the_absolute_prefix(
    policy: PathPolicy, workspace: Path
) -> None:
    assert policy.relative(workspace / "files" / "a.txt") == "files/a.txt"


def test_workspace_paths_layout(workspace: Path) -> None:
    layout = WorkspacePaths(root=workspace)
    for directory in (
        layout.files,
        layout.projects,
        layout.downloads,
        layout.outputs,
        layout.temp,
        layout.state,
    ):
        assert directory.is_dir()


def test_truncate_output_marks_truncation() -> None:
    text, truncated = truncate_output("a" * 100, 10)
    assert truncated is True
    assert "truncated" in text
    short, untouched = truncate_output("abc", 10)
    assert (short, untouched) == ("abc", False)


def test_enforce_file_size() -> None:
    enforce_file_size(10, 100, "f")
    with pytest.raises(LimitExceededError, match="above the"):
        enforce_file_size(200, 100, "f")
