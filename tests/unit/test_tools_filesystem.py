"""Filesystem and search tools, including their limits."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.errors import LimitExceededError, PathEscapeError, SecretProtectedError
from agent.tools.base import ToolContext
from agent.tools.filesystem import (
    WORKSPACE_SUBDIRS,
    ListFilesTool,
    ReadFileTool,
    WriteFileTool,
    initialize_workspace,
    looks_binary,
)
from agent.tools.search import SearchFilesTool
from agent.tools.time_tool import GetCurrentTimeTool


def test_workspace_initialisation_is_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    initialize_workspace(root)
    initialize_workspace(root)
    for name in WORKSPACE_SUBDIRS:
        assert (root / name).is_dir()


async def test_get_current_time(context: ToolContext) -> None:
    output = await GetCurrentTimeTool().run({}, context)
    assert "local" in output and "utc" in output and isinstance(output["unix"], int)


# -- list_files -------------------------------------------------------------
async def test_list_files_returns_relative_paths(context: ToolContext, workspace: Path) -> None:
    (workspace / "files" / "a.txt").write_text("a")
    (workspace / "files" / "sub").mkdir()
    (workspace / "files" / "sub" / "b.txt").write_text("b")
    output = await ListFilesTool().run({"path": "files", "depth": 2}, context)
    paths = {entry["path"] for entry in output["entries"]}
    assert "files/a.txt" in paths and "files/sub/b.txt" in paths
    assert all(not p.startswith("/") for p in paths)


async def test_list_files_respects_depth(context: ToolContext, workspace: Path) -> None:
    (workspace / "files" / "sub").mkdir()
    (workspace / "files" / "sub" / "deep.txt").write_text("x")
    output = await ListFilesTool().run({"path": "files", "depth": 1}, context)
    assert "files/sub/deep.txt" not in {e["path"] for e in output["entries"]}


async def test_list_files_excludes_hidden_and_secrets(
    context: ToolContext, workspace: Path
) -> None:
    (workspace / "files" / ".env").write_text("SECRET=1")
    (workspace / "files" / ".hidden").write_text("x")
    (workspace / "files" / "visible.txt").write_text("x")
    output = await ListFilesTool().run({"path": "files"}, context)
    paths = {e["path"] for e in output["entries"]}
    assert paths == {"files/visible.txt"}


async def test_list_files_hides_secrets_even_when_hidden_are_requested(
    context: ToolContext, workspace: Path
) -> None:
    (workspace / "files" / ".env").write_text("SECRET=1")
    (workspace / "files" / ".notes").write_text("x")
    output = await ListFilesTool().run({"path": "files", "include_hidden": True}, context)
    paths = {e["path"] for e in output["entries"]}
    assert "files/.notes" in paths
    assert "files/.env" not in paths


async def test_list_files_limit_marks_truncation(context: ToolContext, workspace: Path) -> None:
    for index in range(20):
        (workspace / "files" / f"f{index}.txt").write_text("x")
    output = await ListFilesTool().run({"path": "files", "limit": 5}, context)
    assert output["truncated"] is True and output["count"] == 5


async def test_list_files_rejects_escape(context: ToolContext) -> None:
    with pytest.raises(PathEscapeError):
        await ListFilesTool().run({"path": "../.."}, context)


async def test_list_files_skips_symlinks(context: ToolContext, workspace: Path) -> None:
    target = workspace / "files" / "real.txt"
    target.write_text("x")
    (workspace / "files" / "link.txt").symlink_to(target)
    output = await ListFilesTool().run({"path": "files"}, context)
    assert {e["path"] for e in output["entries"]} == {"files/real.txt"}


# -- read_file --------------------------------------------------------------
async def test_read_file(context: ToolContext, workspace: Path) -> None:
    (workspace / "files" / "a.txt").write_text("line1\nline2\nline3\n")
    output = await ReadFileTool().run({"path": "files/a.txt"}, context)
    assert output["content"] == "line1\nline2\nline3"
    assert output["total_lines"] == 3


async def test_read_file_line_window(context: ToolContext, workspace: Path) -> None:
    (workspace / "files" / "a.txt").write_text("\n".join(f"l{i}" for i in range(10)))
    output = await ReadFileTool().run(
        {"path": "files/a.txt", "start_line": 3, "max_lines": 2}, context
    )
    assert output["content"] == "l2\nl3"
    assert output["truncated"] is True


async def test_read_file_rejects_oversize(context: ToolContext, workspace: Path) -> None:
    context.paths = context.paths.__class__(root=context.paths.root, max_file_bytes=10)
    (workspace / "files" / "big.txt").write_text("x" * 100)
    with pytest.raises(LimitExceededError):
        await ReadFileTool().run({"path": "files/big.txt"}, context)


async def test_read_file_rejects_binary(context: ToolContext, workspace: Path) -> None:
    (workspace / "files" / "b.bin").write_bytes(b"\x00\x01\x02binary")
    with pytest.raises(Exception) as exc_info:
        await ReadFileTool().run({"path": "files/b.bin"}, context)
    assert "binary" in str(exc_info.value)


async def test_read_file_rejects_secrets(context: ToolContext, workspace: Path) -> None:
    (workspace / "files" / ".env").write_text("GEMINI_API_KEY=x")
    with pytest.raises(SecretProtectedError):
        await ReadFileTool().run({"path": "files/.env"}, context)


async def test_read_file_rejects_directory(context: ToolContext) -> None:
    with pytest.raises(Exception, match="directory"):
        await ReadFileTool().run({"path": "files"}, context)


async def test_read_file_missing(context: ToolContext) -> None:
    with pytest.raises(FileNotFoundError):
        await ReadFileTool().run({"path": "files/nope.txt"}, context)


def test_looks_binary(tmp_path: Path) -> None:
    text = tmp_path / "t.txt"
    text.write_text("plain")
    binary = tmp_path / "b.bin"
    binary.write_bytes(b"\x00\x01")
    assert looks_binary(text) is False
    assert looks_binary(binary) is True


# -- write_file -------------------------------------------------------------
async def test_write_file_creates(context: ToolContext, workspace: Path) -> None:
    output = await WriteFileTool().run({"path": "outputs/new.md", "content": "hello"}, context)
    assert output["created"] is True
    assert (workspace / "outputs" / "new.md").read_text() == "hello"


async def test_write_file_overwrite_keeps_a_backup(context: ToolContext, workspace: Path) -> None:
    target = workspace / "outputs" / "x.txt"
    target.write_text("old")
    output = await WriteFileTool().run({"path": "outputs/x.txt", "content": "new"}, context)
    assert target.read_text() == "new"
    assert output["backup"] is not None
    assert (workspace / output["backup"]).read_text() == "old"


async def test_write_file_append(context: ToolContext, workspace: Path) -> None:
    target = workspace / "outputs" / "x.txt"
    target.write_text("a")
    await WriteFileTool().run({"path": "outputs/x.txt", "content": "b", "mode": "append"}, context)
    assert target.read_text() == "ab"


async def test_write_file_create_mode_refuses_existing(
    context: ToolContext, workspace: Path
) -> None:
    (workspace / "outputs" / "x.txt").write_text("a")
    with pytest.raises(Exception, match="already exists"):
        await WriteFileTool().run(
            {"path": "outputs/x.txt", "content": "b", "mode": "create"}, context
        )


async def test_write_file_rejects_escape(context: ToolContext) -> None:
    with pytest.raises(PathEscapeError):
        await WriteFileTool().run({"path": "../evil.txt", "content": "x"}, context)


async def test_write_file_rejects_secret_locations(context: ToolContext) -> None:
    with pytest.raises(SecretProtectedError):
        await WriteFileTool().run({"path": "files/.env", "content": "KEY=1"}, context)


async def test_write_file_rejects_oversize(context: ToolContext) -> None:
    context.paths = context.paths.__class__(root=context.paths.root, max_file_bytes=10)
    with pytest.raises(Exception, match="limit"):
        await WriteFileTool().run({"path": "outputs/x.txt", "content": "y" * 100}, context)


def test_write_file_is_classified_as_needing_approval_and_verification() -> None:
    definition = WriteFileTool().definition()
    assert definition.requires_approval is True
    assert definition.requires_verification is True
    assert definition.read_only is False
    assert definition.reversible is False


# -- search_files -----------------------------------------------------------
async def test_search_files_finds_matches(context: ToolContext, workspace: Path) -> None:
    (workspace / "files" / "a.py").write_text("def hello():\n    return 1\n")
    (workspace / "files" / "b.py").write_text("x = 2\n")
    output = await SearchFilesTool().run({"query": "hello", "path": "files"}, context)
    assert output["match_count"] == 1
    assert output["matches"][0]["path"] == "files/a.py"
    assert output["matches"][0]["line"] == 1


async def test_search_files_regex_and_case(context: ToolContext, workspace: Path) -> None:
    (workspace / "files" / "a.txt").write_text("Alpha\nbeta\n")
    assert (await SearchFilesTool().run({"query": "alpha", "path": "files"}, context))[
        "match_count"
    ] == 1
    assert (
        await SearchFilesTool().run(
            {"query": "alpha", "path": "files", "case_sensitive": True}, context
        )
    )["match_count"] == 0
    assert (
        await SearchFilesTool().run({"query": r"^b\w+", "path": "files", "regex": True}, context)
    )["match_count"] == 1


async def test_search_files_invalid_regex(context: ToolContext) -> None:
    with pytest.raises(Exception, match="regular expression"):
        await SearchFilesTool().run({"query": "[unclosed", "regex": True}, context)


async def test_search_files_glob_filter(context: ToolContext, workspace: Path) -> None:
    (workspace / "files" / "a.py").write_text("needle")
    (workspace / "files" / "a.txt").write_text("needle")
    output = await SearchFilesTool().run({"query": "needle", "glob": "*.py"}, context)
    assert output["match_count"] == 1


async def test_search_files_skips_secrets_and_hidden(context: ToolContext, workspace: Path) -> None:
    (workspace / "files" / ".env").write_text("needle")
    (workspace / "files" / ".hidden").write_text("needle")
    output = await SearchFilesTool().run({"query": "needle", "path": "files"}, context)
    assert output["match_count"] == 0
    assert output["files_skipped"] >= 2


async def test_search_files_respects_max_results(context: ToolContext, workspace: Path) -> None:
    (workspace / "files" / "many.txt").write_text("\n".join(["needle"] * 50))
    output = await SearchFilesTool().run(
        {"query": "needle", "path": "files", "max_results": 5}, context
    )
    assert output["match_count"] == 5 and output["truncated"] is True


async def test_search_files_skips_binary(context: ToolContext, workspace: Path) -> None:
    (workspace / "files" / "b.bin").write_bytes(b"\x00needle")
    output = await SearchFilesTool().run({"query": "needle", "path": "files"}, context)
    assert output["match_count"] == 0
