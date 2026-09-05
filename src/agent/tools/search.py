"""`search_files` — read-only text search inside the workspace."""

from __future__ import annotations

import re
from typing import Any

from ..errors import InvalidArgumentsError
from ..messages import RiskCategory, RiskLevel
from .base import Tool, ToolContext
from .filesystem import looks_binary

#: Files larger than this are skipped during a search rather than failing the call.
_SEARCH_FILE_LIMIT_FACTOR = 1


class SearchFilesTool(Tool):
    name = "search_files"
    description = (
        "Search for text inside workspace files and return matching lines with their "
        "file paths and line numbers. Use this to locate code or content before reading."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The text or regular expression to search for.",
                "minLength": 1,
                "maxLength": 500,
            },
            "path": {
                "type": "string",
                "description": "Workspace-relative directory to search. Defaults to the root.",
                "default": ".",
            },
            "glob": {
                "type": "string",
                "description": "Filename pattern to restrict the search, e.g. '*.py'.",
                "default": "*",
            },
            "regex": {
                "type": "boolean",
                "description": "Treat the query as a regular expression.",
                "default": False,
            },
            "case_sensitive": {"type": "boolean", "default": False},
            "max_results": {
                "type": "integer",
                "description": "Maximum number of matching lines to return.",
                "default": 50,
                "minimum": 1,
                "maximum": 1000,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    risk = RiskLevel.READ_ONLY
    risk_category = RiskCategory.READ
    read_only = True

    async def run(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        policy = context.paths
        root = policy.resolve(arguments.get("path", "."), must_exist=True)
        if not root.is_dir():
            raise InvalidArgumentsError(f"{policy.relative(root)} is not a directory")

        query: str = arguments["query"]
        use_regex = bool(arguments.get("regex", False))
        case_sensitive = bool(arguments.get("case_sensitive", False))
        max_results = min(
            int(arguments.get("max_results", 50)), context.config.limits.max_search_results
        )
        pattern_text = query if use_regex else re.escape(query)
        try:
            pattern = re.compile(pattern_text, 0 if case_sensitive else re.IGNORECASE)
        except re.error as exc:
            raise InvalidArgumentsError(f"invalid regular expression: {exc}") from exc

        glob: str = arguments.get("glob", "*") or "*"
        matches: list[dict[str, Any]] = []
        files_scanned = 0
        files_skipped = 0
        truncated = False

        for candidate in sorted(root.rglob(glob)):
            if len(matches) >= max_results:
                truncated = True
                break
            if not candidate.is_file() or candidate.is_symlink():
                continue
            if policy.is_protected(candidate)[0] or policy.is_hidden(candidate):
                files_skipped += 1
                continue
            try:
                size = candidate.stat().st_size
            except OSError:
                files_skipped += 1
                continue
            if size > policy.max_file_bytes * _SEARCH_FILE_LIMIT_FACTOR or looks_binary(candidate):
                files_skipped += 1
                continue
            try:
                text = candidate.read_text(encoding="utf-8", errors="strict")
            except (OSError, UnicodeDecodeError):
                files_skipped += 1
                continue
            files_scanned += 1
            relative = policy.relative(candidate)
            for number, line in enumerate(text.splitlines(), start=1):
                if len(matches) >= max_results:
                    truncated = True
                    break
                if pattern.search(line):
                    snippet = line.strip()
                    if len(snippet) > 300:
                        snippet = snippet[:300] + " …"
                    matches.append({"path": relative, "line": number, "text": snippet})

        return {
            "query": query,
            "root": policy.relative(root) or ".",
            "match_count": len(matches),
            "files_scanned": files_scanned,
            "files_skipped": files_skipped,
            "truncated": truncated,
            "matches": matches,
        }
