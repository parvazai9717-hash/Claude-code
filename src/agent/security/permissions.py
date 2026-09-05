"""Permission decisions.

Permissions answer a question the *model cannot answer for itself*: may this tool
run at all, in this configuration? Approval (a human question) comes afterwards,
and only if permission was granted.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass

from ..config import SHELL_FORBIDDEN, ApprovalMode, Config
from ..errors import ErrorCategory
from ..messages import RiskCategory, ToolDefinition

#: Risk categories that count as "risky" under `ApprovalMode.RISKY`.
RISKY_CATEGORIES: frozenset[RiskCategory] = frozenset(
    {
        RiskCategory.WRITE,
        RiskCategory.SHELL,
        RiskCategory.NETWORK,
        RiskCategory.DELETE,
        RiskCategory.INSTALL,
        RiskCategory.AUTH,
        RiskCategory.EXTERNAL,
        RiskCategory.MEMORY,
    }
)


@dataclass(frozen=True)
class PermissionDecision:
    """Whether a tool may proceed, and whether a human must be asked first."""

    allowed: bool
    requires_approval: bool = False
    reason: str = ""
    error_category: ErrorCategory | None = None

    @classmethod
    def allow(cls, *, requires_approval: bool = False, reason: str = "") -> PermissionDecision:
        return cls(allowed=True, requires_approval=requires_approval, reason=reason)

    @classmethod
    def deny(
        cls, reason: str, category: ErrorCategory = ErrorCategory.PERMISSION_DENIED
    ) -> PermissionDecision:
        return cls(allowed=False, reason=reason, error_category=category)


class PermissionChecker:
    """Applies configuration policy to a proposed tool call."""

    def __init__(self, config: Config) -> None:
        self.config = config

    # -- tool-level policy --------------------------------------------------
    def check_tool(
        self, definition: ToolDefinition, arguments: dict[str, object] | None = None
    ) -> PermissionDecision:
        """Decide whether `definition` may run with `arguments`."""
        mode = self.config.approval_mode

        if definition.risk_category is RiskCategory.SHELL and not self.config.shell_enabled:
            return PermissionDecision.deny("shell execution is disabled in this configuration")

        if definition.risk_category is RiskCategory.SHELL and arguments:
            shell_decision = self.check_shell_command(arguments.get("command"))
            if not shell_decision.allowed:
                return shell_decision

        if definition.read_only and not definition.requires_approval:
            return PermissionDecision.allow(reason="read-only tool")

        if mode is ApprovalMode.ALWAYS:
            return PermissionDecision.allow(requires_approval=True, reason="approval mode 'always'")

        if mode is ApprovalMode.RISKY:
            risky = (
                definition.risk_category in RISKY_CATEGORIES
                or definition.requires_approval
                or not definition.read_only
            )
            return PermissionDecision.allow(
                requires_approval=risky,
                reason="risky action" if risky else "not classified as risky",
            )

        # ApprovalMode.AUTOMATIC: only the explicit allowlist runs unattended.
        if definition.name in self.config.auto_approve_tools:
            return PermissionDecision.allow(
                reason=f"'{definition.name}' is on the auto-approve allowlist"
            )
        return PermissionDecision.deny(
            f"'{definition.name}' is not on the auto-approve allowlist and approval mode "
            "is 'automatic', so no human can be asked",
            category=ErrorCategory.PERMISSION_DENIED,
        )

    # -- shell policy -------------------------------------------------------
    def check_shell_command(self, command: object) -> PermissionDecision:
        """Validate a shell command against the deny-by-default allowlist.

        Accepts either an argv list or a single string (which is parsed with
        `shlex` and must not contain shell metacharacters).
        """
        argv = self.parse_command(command)
        if isinstance(argv, PermissionDecision):
            return argv
        program = argv[0]
        # Reject a path-qualified program: only bare, allowlisted names may run.
        if "/" in program or "\\" in program:
            return PermissionDecision.deny(
                f"'{program}' must be a bare command name, not a path",
                category=ErrorCategory.PERMISSION_DENIED,
            )
        if program in SHELL_FORBIDDEN:
            return PermissionDecision.deny(
                f"'{program}' is permanently forbidden and cannot be allowlisted"
            )
        if program not in self.config.shell_allowed_commands:
            return PermissionDecision.deny(
                f"'{program}' is not in the shell allowlist "
                f"({', '.join(sorted(self.config.shell_allowed_commands))})"
            )
        return PermissionDecision.allow(requires_approval=True, reason="allowlisted command")

    @staticmethod
    def parse_command(command: object) -> list[str] | PermissionDecision:
        """Normalize a command into argv, refusing shell metacharacters.

        Commands are executed as argv without a shell, so metacharacters cannot
        do anything — but their presence means the model expected shell semantics
        it will not get, and silently running a different command is worse than
        refusing.
        """
        if isinstance(command, list):
            if not command or not all(isinstance(part, str) for part in command):
                return PermissionDecision.deny(
                    "shell command must be a non-empty list of strings",
                    category=ErrorCategory.INVALID_ARGUMENTS,
                )
            argv = [str(part) for part in command]
        elif isinstance(command, str):
            text = command.strip()
            if not text:
                return PermissionDecision.deny(
                    "shell command must not be empty",
                    category=ErrorCategory.INVALID_ARGUMENTS,
                )
            forbidden_chars = set("|&;<>$`\n\\") & set(text)
            if forbidden_chars:
                return PermissionDecision.deny(
                    "shell metacharacters are not supported; pass an argv list instead "
                    f"(found: {''.join(sorted(forbidden_chars))})",
                    category=ErrorCategory.PERMISSION_DENIED,
                )
            try:
                argv = shlex.split(text)
            except ValueError as exc:
                return PermissionDecision.deny(
                    f"could not parse the command: {exc}",
                    category=ErrorCategory.INVALID_ARGUMENTS,
                )
            if not argv:
                return PermissionDecision.deny(
                    "shell command must not be empty",
                    category=ErrorCategory.INVALID_ARGUMENTS,
                )
        else:
            return PermissionDecision.deny(
                "shell command must be a string or a list of strings",
                category=ErrorCategory.INVALID_ARGUMENTS,
            )
        return argv
