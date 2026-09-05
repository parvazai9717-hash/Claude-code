"""`verify_result` — read-only evidence gathering.

Verification is a real observation, not a claim. This tool re-inspects the world
after a consequential action and returns evidence, or says plainly that evidence
could not be obtained. The runtime marks an action verified only when this tool
(or an equivalent read-only observation) reports `verified: true`.
"""

from __future__ import annotations

import shutil
from typing import Any

from ..errors import ErrorCategory, InvalidArgumentsError, ToolError
from ..messages import RiskCategory, RiskLevel
from ..security.permissions import PermissionChecker
from .base import Tool, ToolContext
from .filesystem import looks_binary
from .shell import build_child_env


class VerifyResultTool(Tool):
    name = "verify_result"
    description = (
        "Check that a previous action actually took effect, using a read-only observation. "
        "Methods: 'file_exists' (a path is present), 'file_contains' (a file contains an "
        "expected string), 'file_absent' (a path is gone), 'command_succeeds' (an allowlisted "
        "read-only command exits zero). Never claim success without calling this first."
    )
    parameters = {
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "enum": ["file_exists", "file_contains", "file_absent", "command_succeeds"],
                "description": "How to gather evidence.",
            },
            "path": {
                "type": "string",
                "description": "Workspace-relative path, for the file_* methods.",
            },
            "expected": {
                "type": "string",
                "description": "Text that must appear in the file, for 'file_contains'.",
                "maxLength": 5000,
            },
            "command": {
                "type": "array",
                "description": "Allowlisted argv to run, for 'command_succeeds'.",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 64,
            },
            "claim": {
                "type": "string",
                "description": "The claim being checked, in one short sentence.",
                "maxLength": 500,
            },
        },
        "required": ["method"],
        "additionalProperties": False,
    }
    # Verification is read-only by construction, so it never needs approval.
    risk = RiskLevel.READ_ONLY
    risk_category = RiskCategory.READ
    read_only = True
    requires_approval = False

    async def run(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        method: str = arguments["method"]
        claim: str = arguments.get("claim", "")
        handler = {
            "file_exists": self._file_exists,
            "file_absent": self._file_absent,
            "file_contains": self._file_contains,
            "command_succeeds": self._command_succeeds,
        }[method]
        result = await handler(arguments, context)
        result["method"] = method
        result["claim"] = claim
        return result

    # -- methods ------------------------------------------------------------
    async def _file_exists(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        path = self._require(arguments, "path", "file_exists")
        try:
            target = context.paths.resolve(path)
        except FileNotFoundError:
            return {"verified": False, "evidence": f"{path} does not exist"}
        if not target.exists():
            return {"verified": False, "evidence": f"{path} does not exist"}
        stat = target.stat()
        kind = "directory" if target.is_dir() else "file"
        return {
            "verified": True,
            "evidence": f"{path} exists as a {kind} of {stat.st_size} bytes",
            "size_bytes": stat.st_size,
        }

    async def _file_absent(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        path = self._require(arguments, "path", "file_absent")
        target = context.paths.resolve(path)
        if target.exists():
            return {"verified": False, "evidence": f"{path} still exists"}
        return {"verified": True, "evidence": f"{path} is absent, as expected"}

    async def _file_contains(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> dict[str, Any]:
        path = self._require(arguments, "path", "file_contains")
        expected = arguments.get("expected")
        if not expected:
            raise InvalidArgumentsError("'file_contains' requires 'expected'")
        target = context.paths.resolve(path)
        if not target.exists():
            return {"verified": False, "evidence": f"{path} does not exist"}
        if target.is_dir():
            return {"verified": False, "evidence": f"{path} is a directory, not a file"}
        if target.stat().st_size > context.paths.max_file_bytes:
            return {
                "verified": False,
                "evidence": f"{path} is too large to verify by content",
                "unavailable_reason": "file exceeds the configured size limit",
            }
        if looks_binary(target):
            return {
                "verified": False,
                "evidence": f"{path} is binary and cannot be checked as text",
                "unavailable_reason": "binary file",
            }
        text = target.read_text(encoding="utf-8", errors="replace")
        found = expected in text
        if found:
            line_number = text[: text.index(expected)].count("\n") + 1
            return {
                "verified": True,
                "evidence": f"{path} contains the expected text at line {line_number}",
                "line": line_number,
            }
        return {
            "verified": False,
            "evidence": f"{path} exists but does not contain the expected text",
        }

    async def _command_succeeds(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> dict[str, Any]:
        import asyncio

        command = arguments.get("command")
        if not command:
            raise InvalidArgumentsError("'command_succeeds' requires 'command'")
        checker = PermissionChecker(context.config)
        decision = checker.check_shell_command(command)
        if not decision.allowed:
            return {
                "verified": False,
                "evidence": "",
                "unavailable_reason": f"cannot verify by command: {decision.reason}",
            }
        argv = checker.parse_command(command)
        assert isinstance(argv, list)
        env = build_child_env(str(context.paths.root))
        program = shutil.which(argv[0], path=env["PATH"])
        if program is None:
            return {
                "verified": False,
                "evidence": "",
                "unavailable_reason": f"'{argv[0]}' is not installed, so evidence is unavailable",
            }
        process = await asyncio.create_subprocess_exec(
            program,
            *argv[1:],
            cwd=str(context.paths.root),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=context.timeout
            )
        except TimeoutError:
            process.kill()
            await process.wait()
            return {
                "verified": False,
                "evidence": "",
                "unavailable_reason": "the verification command timed out",
            }
        exit_code = process.returncode or 0
        stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
        stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
        tail = (stdout or stderr)[-500:]
        return {
            "verified": exit_code == 0,
            "exit_code": exit_code,
            "evidence": f"`{' '.join(argv)}` exited {exit_code}" + (f": {tail}" if tail else ""),
        }

    @staticmethod
    def _require(arguments: dict[str, Any], key: str, method: str) -> str:
        value = arguments.get(key)
        if not value:
            raise ToolError(
                f"'{method}' requires '{key}'", category=ErrorCategory.INVALID_ARGUMENTS
            )
        return str(value)
