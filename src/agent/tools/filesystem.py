"""Workspace filesystem tools: `list_files`, `read_file`, `write_file`.

Every path goes through :class:`agent.security.paths.PathPolicy` before any I/O,
so containment, symlink escapes and credential files are handled in one place
rather than re-checked (and eventually forgotten) in each tool.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from ..errors import ErrorCategory, InvalidArgumentsError, ToolError
from ..messages import RiskCategory, RiskLevel
from ..security.limits import enforce_file_size, truncate_output
from .base import Tool, ToolContext

#: The persistent workspace layout created at startup.
WORKSPACE_SUBDIRS = ("files", "projects", "downloads", "outputs", "temp", "state")

#: Bytes inspected when deciding whether a file is binary.
_BINARY_SNIFF_BYTES = 8192


def initialize_workspace(root: Path) -> Path:
    """Create the workspace and its standard subdirectories. Idempotent."""
    root = Path(root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    for name in WORKSPACE_SUBDIRS:
        (root / name).mkdir(parents=True, exist_ok=True)
    return root.resolve()


def looks_binary(path: Path) -> bool:
    """Heuristic binary check: a NUL byte in the first few KiB."""
    try:
        with path.open("rb") as handle:
            chunk = handle.read(_BINARY_SNIFF_BYTES)
    except OSError:
        return False
    return b"\x00" in chunk


class ListFilesTool(Tool):
    name = "list_files"
    description = (
        "List files and directories inside the workspace. Returns workspace-relative paths. "
        "Use this to inspect the workspace before reading or writing anything."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Workspace-relative directory to list. Defaults to the root.",
                "default": ".",
            },
            "depth": {
                "type": "integer",
                "description": "How many directory levels to descend (1 = this directory only).",
                "default": 1,
                "minimum": 1,
                "maximum": 10,
            },
            "include_hidden": {
                "type": "boolean",
                "description": "Include dotfiles. Protected credential paths stay excluded.",
                "default": False,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of entries to return.",
                "default": 200,
                "minimum": 1,
                "maximum": 5000,
            },
        },
        "additionalProperties": False,
    }
    risk = RiskLevel.READ_ONLY
    risk_category = RiskCategory.READ
    read_only = True

    async def run(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        policy = context.paths
        start = policy.resolve(arguments.get("path", "."), must_exist=True)
        if not start.is_dir():
            raise InvalidArgumentsError(f"{policy.relative(start)} is not a directory")

        depth = min(int(arguments.get("depth", 1)), context.config.limits.max_directory_depth)
        include_hidden = bool(arguments.get("include_hidden", False))
        limit = min(int(arguments.get("limit", 200)), context.config.limits.max_list_entries)

        entries: list[dict[str, Any]] = []
        truncated = False
        for current_root, dirnames, filenames in os.walk(start, followlinks=False):
            current = Path(current_root)
            level = len(current.relative_to(start).parts)
            # depth 1 means "this directory only", so stop descending once the
            # next level would exceed the requested depth.
            if level + 1 >= depth:
                dirnames[:] = []
            # Filter directories in place so os.walk never descends into them.
            dirnames[:] = sorted(
                d
                for d in dirnames
                if (include_hidden or not d.startswith("."))
                and not policy.is_protected(current / d)[0]
            )
            for name in sorted(filenames):
                if not include_hidden and name.startswith("."):
                    continue
                candidate = current / name
                if policy.is_protected(candidate)[0]:
                    continue
                if candidate.is_symlink():
                    # Never report a link target that might sit outside the workspace.
                    continue
                if len(entries) >= limit:
                    truncated = True
                    break
                try:
                    size = candidate.stat().st_size
                except OSError:
                    continue
                entries.append(
                    {"path": policy.relative(candidate), "type": "file", "size_bytes": size}
                )
            if truncated:
                break
            for name in dirnames:
                if len(entries) >= limit:
                    truncated = True
                    break
                entries.append({"path": policy.relative(current / name), "type": "directory"})
            if truncated:
                break

        return {
            "root": policy.relative(start) or ".",
            "count": len(entries),
            "truncated": truncated,
            "entries": entries,
        }


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "Read a UTF-8 text file from the workspace. Refuses binary files, files above the "
        "configured size limit, and credential files. Always read before you modify."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Workspace-relative file path."},
            "start_line": {
                "type": "integer",
                "description": "1-based first line to return.",
                "default": 1,
                "minimum": 1,
            },
            "max_lines": {
                "type": "integer",
                "description": "Maximum number of lines to return.",
                "default": 500,
                "minimum": 1,
                "maximum": 10000,
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    }
    risk = RiskLevel.READ_ONLY
    risk_category = RiskCategory.READ
    read_only = True

    async def run(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        policy = context.paths
        target = policy.resolve(arguments["path"], must_exist=True)
        if target.is_dir():
            raise InvalidArgumentsError(
                f"{policy.relative(target)} is a directory; use list_files instead"
            )

        size = target.stat().st_size
        enforce_file_size(size, policy.max_file_bytes, policy.relative(target))
        if looks_binary(target):
            raise ToolError(
                f"{policy.relative(target)} appears to be a binary file and cannot be read as text",
                category=ErrorCategory.INVALID_ARGUMENTS,
            )

        text = target.read_text(encoding="utf-8")
        lines = text.splitlines()
        start = int(arguments.get("start_line", 1))
        max_lines = int(arguments.get("max_lines", 500))
        selected = lines[start - 1 : start - 1 + max_lines]
        content, truncated = truncate_output("\n".join(selected), context.max_output_chars)

        return {
            "path": policy.relative(target),
            "size_bytes": size,
            "total_lines": len(lines),
            "start_line": start,
            "returned_lines": len(selected),
            "truncated": truncated or (start - 1 + len(selected)) < len(lines),
            "content": content,
        }


class WriteFileTool(Tool):
    name = "write_file"
    description = (
        "Create or modify a text file inside the workspace. Requires human approval. "
        "Writes atomically and reports what changed. Read the file first when overwriting."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Workspace-relative file path."},
            "content": {"type": "string", "description": "The text to write."},
            "mode": {
                "type": "string",
                "description": (
                    "'overwrite' replaces the file, 'append' adds to the end, "
                    "'create' fails if the file already exists."
                ),
                "enum": ["overwrite", "append", "create"],
                "default": "overwrite",
            },
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }
    risk = RiskLevel.MEDIUM
    risk_category = RiskCategory.WRITE
    read_only = False
    requires_approval = True
    requires_verification = True
    reversible = False

    async def run(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        policy = context.paths
        target = policy.resolve(arguments["path"])
        content: str = arguments["content"]
        mode: str = arguments.get("mode", "overwrite")

        if target.is_dir():
            raise InvalidArgumentsError(f"{policy.relative(target)} is a directory")
        if len(content.encode("utf-8")) > policy.max_file_bytes:
            raise ToolError(
                f"refusing to write {len(content)} characters, above the "
                f"{policy.max_file_bytes} byte limit",
                category=ErrorCategory.LIMIT_EXCEEDED,
            )

        existed = target.exists()
        if mode == "create" and existed:
            raise ToolError(
                f"{policy.relative(target)} already exists; use mode 'overwrite' to replace it",
                category=ErrorCategory.INVALID_ARGUMENTS,
            )

        previous_bytes = target.stat().st_size if existed else 0
        backup_path: str | None = None
        target.parent.mkdir(parents=True, exist_ok=True)

        if mode == "append":
            if existed:
                # Keep a backup so an unwanted append can be undone by hand.
                backup = target.with_suffix(target.suffix + ".bak")
                shutil.copy2(target, backup)
                backup_path = policy.relative(backup)
            with target.open("a", encoding="utf-8") as handle:
                handle.write(content)
            final_text = target.read_text(encoding="utf-8")
        else:
            if existed:
                backup = target.with_suffix(target.suffix + ".bak")
                shutil.copy2(target, backup)
                backup_path = policy.relative(backup)
            # Atomic replace: write to a temp file in the same directory, then rename.
            handle_fd, temp_name = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
            try:
                with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
                    handle.write(content)
                os.replace(temp_name, target)
            except BaseException:
                Path(temp_name).unlink(missing_ok=True)
                raise
            final_text = content

        return {
            "path": policy.relative(target),
            "mode": mode,
            "created": not existed,
            "bytes_before": previous_bytes,
            "bytes_after": target.stat().st_size,
            "lines_after": len(final_text.splitlines()),
            "backup": backup_path,
        }
