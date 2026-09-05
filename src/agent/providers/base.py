"""The model-provider boundary.

The runtime speaks only :class:`~agent.messages.Message`,
:class:`~agent.messages.ToolDefinition` and :class:`~agent.messages.ModelResponse`.
Each provider translates those into its own wire format and back, and maps its
own failures onto the shared :class:`~agent.errors.ErrorCategory` taxonomy.

Rules every provider must satisfy:

1. No provider-specific type ever escapes the adapter.
2. Timeouts and cancellation are supported.
3. Errors distinguish auth, unavailable, timeout, rate limit, invalid request,
   malformed response and unknown.
4. Credentials come only from the environment, and are never logged, echoed or
   placed in model-visible context.
5. Capabilities — above all, native tool calling — are stated explicitly.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

from ..errors import ProviderMalformedResponseError
from ..messages import (
    Message,
    ModelResponse,
    ProviderCapabilities,
    ProviderStatus,
    ToolCall,
    ToolDefinition,
)


@runtime_checkable
class ModelProvider(Protocol):
    """The interface the runtime depends on."""

    name: str
    model: str

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        *,
        temperature: float = 0.2,
        max_output_tokens: int | None = None,
    ) -> ModelResponse:
        """Produce one model reply from normalized inputs."""
        ...  # pragma: no cover - protocol definition

    async def health_check(self) -> ProviderStatus:
        """Report reachability and configuration without performing real work."""
        ...  # pragma: no cover - protocol definition

    def capabilities(self) -> ProviderCapabilities:
        """State what this provider and model can actually do."""
        ...  # pragma: no cover - protocol definition

    async def aclose(self) -> None:
        """Release network resources."""
        ...  # pragma: no cover - protocol definition


class BaseProvider(ABC):
    """Shared helpers for concrete providers."""

    name: str = "base"

    def __init__(self, model: str, *, timeout: float = 120.0) -> None:
        self.model = model
        self.timeout = timeout

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        *,
        temperature: float = 0.2,
        max_output_tokens: int | None = None,
    ) -> ModelResponse: ...

    @abstractmethod
    async def health_check(self) -> ProviderStatus: ...

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(native_tool_calls=False)

    async def aclose(self) -> None:
        """Default: nothing to release."""
        return None

    # -- shared conversions -------------------------------------------------
    @staticmethod
    def split_system(messages: list[Message]) -> tuple[str, list[Message]]:
        """Separate system instructions from the conversation body.

        Most APIs carry the system prompt out of band; those that do not can
        re-prepend the returned text themselves.
        """
        system_parts = [m.content for m in messages if m.role == "system" and m.content]
        body = [m for m in messages if m.role != "system"]
        return "\n\n".join(system_parts), body

    @staticmethod
    def parse_arguments(raw: Any, tool_name: str) -> dict[str, Any]:
        """Coerce provider-supplied tool arguments into a dictionary.

        Models sometimes emit a JSON *string* where an object is expected; that is
        recoverable. Anything else is a malformed response, and saying so plainly
        is better than guessing what the model meant.
        """
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                return {}
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ProviderMalformedResponseError(
                    f"tool call '{tool_name}' had arguments that are not valid JSON",
                    details={"tool": tool_name},
                ) from exc
            if not isinstance(parsed, dict):
                raise ProviderMalformedResponseError(
                    f"tool call '{tool_name}' arguments must be a JSON object"
                )
            return parsed
        raise ProviderMalformedResponseError(
            f"tool call '{tool_name}' had arguments of unsupported type {type(raw).__name__}"
        )

    @staticmethod
    def make_tool_call(name: Any, arguments: Any, call_id: str | None = None) -> ToolCall:
        """Build a normalized tool call, rejecting a missing or non-string name."""
        if not isinstance(name, str) or not name.strip():
            raise ProviderMalformedResponseError("the model returned a tool call with no name")
        parsed = BaseProvider.parse_arguments(arguments, name)
        if call_id:
            return ToolCall(id=str(call_id), name=name, arguments=parsed)
        return ToolCall(name=name, arguments=parsed)
