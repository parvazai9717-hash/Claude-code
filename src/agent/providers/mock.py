"""A deterministic provider for tests and offline smoke runs.

`MockProvider` replays a script of :class:`~agent.messages.ModelResponse` objects,
so a full agent loop — tool calls, verification, replanning after a failure — can
be exercised with no network, no API key and no local model server.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ..errors import ProviderUnavailableError
from ..messages import (
    FinishReason,
    Message,
    ModelResponse,
    ProviderCapabilities,
    ProviderStatus,
    ToolCall,
    ToolDefinition,
    Usage,
)
from .base import BaseProvider

#: A script entry is either a ready response or a callable that builds one from
#: the conversation so far, which is how conditional test scenarios are written.
ScriptEntry = ModelResponse | Callable[[list[Message], list[ToolDefinition]], ModelResponse]


class MockProvider(BaseProvider):
    """Replays scripted responses in order."""

    name = "mock"

    def __init__(
        self,
        script: Sequence[ScriptEntry] | None = None,
        *,
        model: str = "mock-model",
        supports_tools: bool = True,
        supports_vision: bool = True,
        supports_audio: bool = True,
        supports_video: bool = False,
        healthy: bool = True,
        repeat_last: bool = False,
    ) -> None:
        super().__init__(model=model, timeout=5.0)
        self.script: list[ScriptEntry] = list(script or [])
        self.supports_tools = supports_tools
        self.supports_vision = supports_vision
        self.supports_audio = supports_audio
        self.supports_video = supports_video
        self.healthy = healthy
        #: Attachments the runtime actually sent, for assertions.
        self.received_attachments: list[Any] = []
        #: When the script runs out, either repeat the last entry or stop cleanly.
        self.repeat_last = repeat_last
        #: Every call the runtime made, for assertions.
        self.calls: list[dict[str, Any]] = []
        self._index = 0

    # -- scripting helpers --------------------------------------------------
    @staticmethod
    def text(content: str) -> ModelResponse:
        """A final textual answer with no tool calls."""
        return ModelResponse(
            text=content,
            finish_reason=FinishReason.STOP,
            provider="mock",
            model="mock-model",
            usage=Usage(input_tokens=0, output_tokens=len(content.split())),
        )

    @staticmethod
    def call(
        name: str, arguments: dict[str, Any] | None = None, *, text: str = ""
    ) -> ModelResponse:
        """A response that requests one tool call."""
        return ModelResponse(
            text=text or None,
            tool_calls=[ToolCall(name=name, arguments=arguments or {})],
            finish_reason=FinishReason.TOOL_CALLS,
            provider="mock",
            model="mock-model",
        )

    @staticmethod
    def calls_many(requests: Sequence[tuple[str, dict[str, Any]]]) -> ModelResponse:
        """A response requesting several tool calls in one turn."""
        return ModelResponse(
            tool_calls=[ToolCall(name=name, arguments=args) for name, args in requests],
            finish_reason=FinishReason.TOOL_CALLS,
            provider="mock",
            model="mock-model",
        )

    def enqueue(self, *entries: ScriptEntry) -> MockProvider:
        self.script.extend(entries)
        return self

    @property
    def remaining(self) -> int:
        return max(0, len(self.script) - self._index)

    # -- provider interface -------------------------------------------------
    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        *,
        temperature: float = 0.2,
        max_output_tokens: int | None = None,
    ) -> ModelResponse:
        for message in messages:
            self.received_attachments.extend(message.attachments)
        self.calls.append(
            {
                "messages": list(messages),
                "tools": [t.name for t in tools],
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            }
        )
        if not self.healthy:
            raise ProviderUnavailableError("the mock provider is configured as unhealthy")

        if self._index >= len(self.script):
            if self.repeat_last and self.script:
                entry = self.script[-1]
            else:
                # An exhausted script means the loop ran longer than the test
                # expected. Ending cleanly makes that visible without hanging.
                return ModelResponse(
                    text="(mock script exhausted)",
                    finish_reason=FinishReason.STOP,
                    provider=self.name,
                    model=self.model,
                )
        else:
            entry = self.script[self._index]
            self._index += 1

        response = entry(messages, tools) if callable(entry) else entry
        if response.tool_calls and not self.supports_tools:
            raise ProviderUnavailableError("this mock provider is configured without tool support")
        return response.model_copy(update={"provider": self.name, "model": self.model})

    async def health_check(self) -> ProviderStatus:
        return ProviderStatus(
            provider=self.name,
            ok=self.healthy,
            model=self.model,
            detail="mock provider (offline, deterministic)"
            if self.healthy
            else "mock provider configured as unhealthy",
            capabilities=self.capabilities(),
            available_models=[self.model],
        )

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            native_tool_calls=self.supports_tools,
            streaming=False,
            system_instruction=True,
            max_context_tokens=32_000,
            vision=self.supports_vision,
            audio=self.supports_audio,
            video=self.supports_video,
        )
