"""Provider-neutral data model.

Everything that crosses the runtime/provider boundary is expressed with the types
in this module. Providers translate these into their own wire format and back;
the runtime never sees a Gemini or Ollama shape.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .errors import ErrorCategory

Role = Literal["system", "user", "assistant", "tool"]


def _now() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    """Generate a short, provider-neutral identifier."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class RiskLevel(StrEnum):
    """How much damage a tool can do if it is called wrongly."""

    READ_ONLY = "read_only"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RiskCategory(StrEnum):
    """What kind of side effect a tool has. Drives approval policy."""

    READ = "read"
    WRITE = "write"
    SHELL = "shell"
    NETWORK = "network"
    DELETE = "delete"
    INSTALL = "install"
    AUTH = "auth"
    EXTERNAL = "external"
    MEMORY = "memory"


class ToolCall(BaseModel):
    """A model's *request* to run a tool. It is a proposal, never an authorisation."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: new_id("call"))
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    # Safe, non-secret provider metadata retained only when debugging is enabled.
    raw_metadata: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """The runtime's *answer* to a tool call, after execution and redaction."""

    call_id: str
    tool_name: str
    ok: bool
    output: dict[str, Any] = Field(default_factory=dict)
    error_category: ErrorCategory | None = None
    error_message: str | None = None
    duration_ms: int = 0
    timed_out: bool = False
    truncated: bool = False
    verified: bool | None = None
    verification_evidence: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_model_text(self) -> str:
        """Render a compact, already-redacted summary for the model's context."""
        import json

        payload: dict[str, Any] = {"ok": self.ok}
        if self.ok:
            payload["output"] = self.output
        else:
            payload["error"] = {
                "category": self.error_category.value if self.error_category else "unknown",
                "message": self.error_message or "",
            }
        if self.truncated:
            payload["truncated"] = True
        if self.timed_out:
            payload["timed_out"] = True
        if self.verified is not None:
            payload["verified"] = self.verified
            if self.verification_evidence:
                payload["evidence"] = self.verification_evidence
        return json.dumps(payload, ensure_ascii=False, default=str)


class Message(BaseModel):
    """One turn of conversation in normalized form."""

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    # Set on `tool` messages: which call this message answers.
    tool_call_id: str | None = None
    name: str | None = None
    created_at: datetime = Field(default_factory=_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str = "", tool_calls: list[ToolCall] | None = None) -> Message:
        return cls(role="assistant", content=content, tool_calls=tool_calls or [])

    @classmethod
    def from_tool_result(cls, result: ToolResult) -> Message:
        return cls(
            role="tool",
            content=result.to_model_text(),
            tool_call_id=result.call_id,
            name=result.tool_name,
        )


class ToolDefinition(BaseModel):
    """The registry's description of a tool, in a provider-neutral schema."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    risk: RiskLevel = RiskLevel.READ_ONLY
    risk_category: RiskCategory = RiskCategory.READ
    read_only: bool = True
    requires_approval: bool = False
    #: Whether a successful call must be followed by verification evidence.
    requires_verification: bool = False
    #: Whether the action can be undone (informs the approval prompt).
    reversible: bool = True

    def json_schema(self) -> dict[str, Any]:
        """The argument schema, guaranteed to be a well-formed JSON-Schema object."""
        schema = dict(self.parameters) if self.parameters else {}
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})
        # Unknown arguments are rejected by default.
        schema.setdefault("additionalProperties", False)
        return schema


class Usage(BaseModel):
    """Token accounting, when the provider reports it."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class FinishReason(StrEnum):
    STOP = "stop"
    TOOL_CALLS = "tool_calls"
    LENGTH = "length"
    SAFETY = "safety"
    ERROR = "error"
    UNKNOWN = "unknown"


class ProviderCapabilities(BaseModel):
    """What a provider/model can actually do. Explicit, never assumed."""

    native_tool_calls: bool = False
    streaming: bool = False
    system_instruction: bool = True
    max_context_tokens: int | None = None


class ModelResponse(BaseModel):
    """A normalized model reply."""

    text: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: FinishReason = FinishReason.UNKNOWN
    provider: str = ""
    model: str = ""
    usage: Usage | None = None
    capabilities: ProviderCapabilities | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class ProviderStatus(BaseModel):
    """Result of a provider health check. Never contains a credential."""

    provider: str
    ok: bool
    model: str | None = None
    detail: str = ""
    error_category: ErrorCategory | None = None
    capabilities: ProviderCapabilities | None = None
    available_models: list[str] = Field(default_factory=list)
