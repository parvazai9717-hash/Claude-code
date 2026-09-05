"""Tool interface and execution context.

A tool is a small, auditable unit of capability. It declares a schema, a risk
classification and whether it needs approval; it receives an already-validated
argument dictionary and a :class:`ToolContext` carrying the security policy it
must use. A tool never reads configuration or the environment directly.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..config import Config
from ..errors import AgentError, ErrorCategory
from ..messages import RiskCategory, RiskLevel, ToolDefinition, ToolResult
from ..security.paths import PathPolicy
from ..security.redaction import Redactor


@dataclass
class ToolContext:
    """Everything a tool is allowed to reach.

    Passing this explicitly — rather than letting tools import globals — is what
    makes the workspace boundary testable: a test simply hands over a policy
    rooted at a temporary directory.
    """

    config: Config
    paths: PathPolicy
    redactor: Redactor
    workspace: Any = None
    #: Set for tools that need to record something against the current task.
    task_id: str | None = None
    step: int = 0
    #: Optional storage handle, injected for memory tools.
    store: Any = None

    @property
    def max_output_chars(self) -> int:
        return self.config.limits.max_tool_output_chars

    @property
    def timeout(self) -> float:
        return self.config.limits.tool_timeout_seconds


class Tool(ABC):
    """Base class for every tool."""

    #: Stable, model-visible name. Must be unique in a registry.
    name: str = ""
    description: str = ""
    #: JSON Schema for the arguments. `additionalProperties` defaults to false.
    parameters: dict[str, Any] = {}
    risk: RiskLevel = RiskLevel.READ_ONLY
    risk_category: RiskCategory = RiskCategory.READ
    read_only: bool = True
    requires_approval: bool = False
    requires_verification: bool = False
    reversible: bool = True

    def definition(self) -> ToolDefinition:
        """The registry-facing description of this tool."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            risk=self.risk,
            risk_category=self.risk_category,
            read_only=self.read_only,
            requires_approval=self.requires_approval,
            requires_verification=self.requires_verification,
            reversible=self.reversible,
        )

    @abstractmethod
    async def run(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        """Execute the tool.

        Args:
            arguments: Already validated against `parameters`.
            context: Security policy and configuration for this call.

        Returns:
            A JSON-serialisable result payload. It is redacted and truncated by
            the registry before it reaches the model.

        Raises:
            AgentError: For any structured, categorised failure.
        """

    # -- helpers ------------------------------------------------------------
    def ok(self, call_id: str, output: dict[str, Any], *, duration_ms: int = 0) -> ToolResult:
        return ToolResult(
            call_id=call_id, tool_name=self.name, ok=True, output=output, duration_ms=duration_ms
        )

    def fail(
        self,
        call_id: str,
        message: str,
        category: ErrorCategory = ErrorCategory.TOOL_FAILED,
        *,
        duration_ms: int = 0,
    ) -> ToolResult:
        return ToolResult(
            call_id=call_id,
            tool_name=self.name,
            ok=False,
            error_category=category,
            error_message=message,
            duration_ms=duration_ms,
        )


class Timer:
    """Small context manager for millisecond durations."""

    def __init__(self) -> None:
        self.start = 0.0
        self.elapsed_ms = 0

    def __enter__(self) -> Timer:
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed_ms = int((time.perf_counter() - self.start) * 1000)


def error_from_exception(exc: Exception) -> tuple[ErrorCategory, str]:
    """Map an arbitrary exception onto a category and a safe message."""
    if isinstance(exc, AgentError):
        return exc.category, exc.message
    if isinstance(exc, FileNotFoundError):
        return ErrorCategory.NOT_FOUND, str(exc)
    if isinstance(exc, PermissionError):
        return ErrorCategory.PERMISSION_DENIED, "the operating system refused access"
    if isinstance(exc, IsADirectoryError):
        return ErrorCategory.INVALID_ARGUMENTS, "that path is a directory, not a file"
    if isinstance(exc, NotADirectoryError):
        return ErrorCategory.INVALID_ARGUMENTS, "that path is a file, not a directory"
    if isinstance(exc, UnicodeDecodeError):
        return ErrorCategory.INVALID_ARGUMENTS, "the file is not valid UTF-8 text"
    if isinstance(exc, TimeoutError):
        return ErrorCategory.TOOL_TIMEOUT, "the operation timed out"
    if isinstance(exc, OSError):
        return ErrorCategory.TOOL_FAILED, f"operating system error: {exc.strerror or 'unknown'}"
    return ErrorCategory.TOOL_FAILED, f"{type(exc).__name__}: {exc}"
