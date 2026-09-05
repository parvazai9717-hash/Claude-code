"""`run_shell` — controlled command execution.

Commands run as **argv without a shell**, from the workspace directory, in a
scrubbed environment, under a timeout, in their own process group so a timeout
kills the whole tree. The allowlist is checked twice: once by the permission
layer before approval, and once here immediately before spawning.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
from typing import Any

from ..errors import ErrorCategory, PermissionDeniedError, ToolError
from ..messages import RiskCategory, RiskLevel
from ..security.limits import truncate_output
from ..security.permissions import PermissionChecker
from .base import Tool, ToolContext

#: Environment variables passed through to the child. Everything else is dropped,
#: so a credential in the parent environment can never reach a subprocess.
_ENV_ALLOWLIST = ("PATH", "LANG", "LC_ALL", "TERM", "TZ")

#: Grace period between SIGTERM and SIGKILL when a command times out.
_KILL_GRACE_SECONDS = 2.0


def build_child_env(workspace: str) -> dict[str, str]:
    """A minimal, credential-free environment for a child process."""
    env = {name: os.environ[name] for name in _ENV_ALLOWLIST if name in os.environ}
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    env["HOME"] = workspace
    env["PWD"] = workspace
    # Signal to child tooling that this is a constrained, non-interactive context.
    env["LOCAL_AGENT_SANDBOX"] = "1"
    return env


class RunShellTool(Tool):
    name = "run_shell"
    description = (
        "Run a single allowlisted command inside the workspace and return its exit status, "
        "stdout and stderr. Requires human approval. Pass the command as a list of arguments "
        "(argv); shell metacharacters such as pipes and redirection are not supported."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "array",
                "description": "The command as an argv list, e.g. ['ls', '-la'].",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 64,
            },
            "cwd": {
                "type": "string",
                "description": "Workspace-relative working directory. Defaults to the root.",
                "default": ".",
            },
            "timeout_seconds": {
                "type": "number",
                "description": "Override the tool timeout, capped by configuration.",
                "minimum": 1,
                "maximum": 600,
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    }
    risk = RiskLevel.HIGH
    risk_category = RiskCategory.SHELL
    read_only = False
    requires_approval = True
    requires_verification = True
    reversible = False

    async def run(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        policy = context.paths
        checker = PermissionChecker(context.config)

        # Re-check the allowlist immediately before spawning. The permission layer
        # already did this, but a second check here means the tool is safe even if
        # it is ever invoked outside the registry.
        decision = checker.check_shell_command(arguments["command"])
        if not decision.allowed:
            raise PermissionDeniedError(decision.reason)
        argv = checker.parse_command(arguments["command"])
        assert isinstance(argv, list)  # guaranteed by the decision above

        program_path = shutil.which(argv[0], path=build_child_env(str(policy.root))["PATH"])
        if program_path is None:
            raise ToolError(
                f"'{argv[0]}' is allowlisted but not installed on this system",
                category=ErrorCategory.NOT_FOUND,
            )

        cwd = policy.resolve(arguments.get("cwd", "."), must_exist=True)
        if not cwd.is_dir():
            raise ToolError(
                f"{policy.relative(cwd)} is not a directory",
                category=ErrorCategory.INVALID_ARGUMENTS,
            )

        timeout = float(arguments.get("timeout_seconds") or context.timeout)
        timeout = min(timeout, context.config.limits.tool_timeout_seconds)

        process = await asyncio.create_subprocess_exec(
            program_path,
            *argv[1:],
            cwd=str(cwd),
            env=build_child_env(str(policy.root)),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
            # A new process group means a timeout can kill children too.
            start_new_session=True,
        )

        timed_out = False
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
        except TimeoutError:
            timed_out = True
            stdout_bytes, stderr_bytes = await self._terminate(process)

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        limit = context.max_output_chars // 2
        stdout, stdout_truncated = truncate_output(stdout, limit)
        stderr, stderr_truncated = truncate_output(stderr, limit)

        exit_code = process.returncode
        return {
            "command": argv,
            "cwd": policy.relative(cwd) or ".",
            "exit_code": exit_code,
            "success": exit_code == 0 and not timed_out,
            "timed_out": timed_out,
            "stdout": stdout,
            "stderr": stderr,
            "output_truncated": stdout_truncated or stderr_truncated,
        }

    @staticmethod
    async def _terminate(process: asyncio.subprocess.Process) -> tuple[bytes, bytes]:
        """Kill a timed-out process group and collect whatever output exists."""
        for send_signal in (signal.SIGTERM, signal.SIGKILL):
            if process.returncode is not None:
                break
            try:
                os.killpg(os.getpgid(process.pid), send_signal)
            except (ProcessLookupError, PermissionError):
                break
            try:
                await asyncio.wait_for(process.wait(), timeout=_KILL_GRACE_SECONDS)
            except TimeoutError:
                continue
        try:
            return await asyncio.wait_for(process.communicate(), timeout=_KILL_GRACE_SECONDS)
        except (TimeoutError, ValueError):
            return b"", b""
