"""Ollama adapter.

Speaks the Ollama-compatible HTTP API (`/api/chat`, `/api/tags`, `/api/show`)
over `httpx`. Local models vary in capability, so this adapter checks whether the
selected model advertises tool support and raises a clear capability error rather
than fabricating a tool call from free text.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from ..errors import (
    ProviderError,
    ProviderInvalidRequestError,
    ProviderMalformedResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    ProviderUnsupportedError,
)
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

#: Ollama reports capabilities in `/api/show`; this is the tool-calling marker.
_TOOLS_CAPABILITY = "tools"


class OllamaProvider(BaseProvider):
    """Talks to a local Ollama-compatible server."""

    name = "ollama"

    def __init__(
        self,
        model: str = "llama3.1",
        *,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(model=model, timeout=timeout)
        self.base_url = base_url.rstrip("/")
        self._client = client
        self._owns_client = client is None
        #: Populated by `health_check`; None means "not yet determined".
        self._supports_tools: bool | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
        return self._client

    # -- conversion ---------------------------------------------------------
    def _to_payload_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert normalized messages into Ollama chat messages."""
        payload: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "tool":
                payload.append(
                    {
                        "role": "tool",
                        "content": message.content,
                        # Ollama accepts a name hint for matching the call.
                        "tool_name": message.name or "tool",
                    }
                )
                continue
            entry: dict[str, Any] = {"role": message.role, "content": message.content}
            if message.tool_calls:
                entry["tool_calls"] = [
                    {"function": {"name": call.name, "arguments": call.arguments}}
                    for call in message.tool_calls
                ]
            payload.append(entry)
        return payload

    @staticmethod
    def _to_payload_tools(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.json_schema(),
                },
            }
            for tool in tools
        ]

    def _from_payload(self, payload: dict[str, Any]) -> ModelResponse:
        message = payload.get("message")
        if not isinstance(message, dict):
            raise ProviderMalformedResponseError(
                "the Ollama response did not contain a message object"
            )
        text = message.get("content") or None
        tool_calls: list[ToolCall] = []
        for raw in message.get("tool_calls") or []:
            if not isinstance(raw, dict):
                raise ProviderMalformedResponseError(
                    "a tool call in the response was not an object"
                )
            function = raw.get("function") or {}
            tool_calls.append(
                self.make_tool_call(function.get("name"), function.get("arguments"), raw.get("id"))
            )

        raw_reason = str(payload.get("done_reason") or "").lower()
        if tool_calls:
            finish = FinishReason.TOOL_CALLS
        elif raw_reason == "length":
            finish = FinishReason.LENGTH
        elif raw_reason == "stop" or payload.get("done"):
            finish = FinishReason.STOP
        else:
            finish = FinishReason.UNKNOWN

        usage = None
        if "prompt_eval_count" in payload or "eval_count" in payload:
            prompt_tokens = payload.get("prompt_eval_count")
            output_tokens = payload.get("eval_count")
            usage = Usage(
                input_tokens=prompt_tokens,
                output_tokens=output_tokens,
                total_tokens=(prompt_tokens or 0) + (output_tokens or 0) or None,
            )

        return ModelResponse(
            text=text,
            tool_calls=tool_calls,
            finish_reason=finish,
            provider=self.name,
            model=str(payload.get("model") or self.model),
            usage=usage,
            capabilities=self.capabilities(),
        )

    # -- provider interface -------------------------------------------------
    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        *,
        temperature: float = 0.2,
        max_output_tokens: int | None = None,
    ) -> ModelResponse:
        if tools and self._supports_tools is False:
            raise ProviderUnsupportedError(
                f"the local model {self.model!r} does not support tool calling. "
                "Choose a tool-capable model (for example llama3.1, qwen2.5 or mistral-nemo), "
                "or run without tools.",
                details={"model": self.model},
            )

        options: dict[str, Any] = {"temperature": temperature}
        if max_output_tokens is not None:
            options["num_predict"] = max_output_tokens

        body: dict[str, Any] = {
            "model": self.model,
            "messages": self._to_payload_messages(messages),
            "stream": False,
            "options": options,
        }
        if tools:
            body["tools"] = self._to_payload_tools(tools)

        client = self._get_client()
        try:
            response = await asyncio.wait_for(
                client.post("/api/chat", json=body), timeout=self.timeout
            )
        except TimeoutError as exc:
            raise ProviderTimeoutError(f"Ollama did not respond within {self.timeout:g}s") from exc
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(f"the Ollama request timed out: {exc}") from exc
        except httpx.ConnectError as exc:
            raise ProviderUnavailableError(
                f"cannot reach the Ollama server at {self.base_url}. "
                "Start it with `ollama serve` or correct `ollama_base_url`."
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(f"the Ollama request failed: {exc}") from exc

        if response.status_code >= 400:
            raise self._map_status(response)

        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderMalformedResponseError("the Ollama response was not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ProviderMalformedResponseError("the Ollama response was not a JSON object")
        return self._from_payload(payload)

    def _map_status(self, response: httpx.Response) -> ProviderError:
        text = response.text[:300]
        lowered = text.lower()
        if response.status_code == 404:
            return ProviderInvalidRequestError(
                f"the model {self.model!r} is not installed on the Ollama server. "
                f"Install it with: ollama pull {self.model}"
            )
        tool_rejection = "does not support tools" in lowered or (
            "tool" in lowered and response.status_code == 400
        )
        if tool_rejection:
            self._supports_tools = False
            return ProviderUnsupportedError(
                f"the local model {self.model!r} does not support tool calling: {text}"
            )
        if response.status_code == 400:
            return ProviderInvalidRequestError(f"Ollama rejected the request: {text}")
        if response.status_code in (502, 503, 504):
            return ProviderUnavailableError(
                f"the Ollama server is unavailable (HTTP {response.status_code})", retryable=True
            )
        return ProviderError(f"Ollama returned HTTP {response.status_code}: {text}")

    async def health_check(self) -> ProviderStatus:
        client = self._get_client()
        try:
            response = await asyncio.wait_for(
                client.get("/api/tags"), timeout=min(self.timeout, 15.0)
            )
        except (TimeoutError, httpx.TimeoutException):
            return ProviderStatus(
                provider=self.name,
                ok=False,
                model=self.model,
                detail=f"the Ollama server at {self.base_url} did not respond in time",
                error_category=ProviderTimeoutError("").category,
            )
        except httpx.HTTPError as exc:
            return ProviderStatus(
                provider=self.name,
                ok=False,
                model=self.model,
                detail=(
                    f"cannot reach the Ollama server at {self.base_url}: {exc}. "
                    "Start it with `ollama serve`."
                ),
                error_category=ProviderUnavailableError("").category,
            )

        if response.status_code >= 400:
            return ProviderStatus(
                provider=self.name,
                ok=False,
                model=self.model,
                detail=f"the Ollama server returned HTTP {response.status_code}",
                error_category=ProviderUnavailableError("").category,
            )

        try:
            models = [str(m.get("name", "")) for m in response.json().get("models", [])]
        except (ValueError, AttributeError):
            models = []
        installed = any(
            name == self.model or name.split(":")[0] == self.model.split(":")[0] for name in models
        )
        supports_tools = await self._probe_tool_support() if installed else None
        self._supports_tools = supports_tools

        if not installed:
            detail = (
                f"the server is reachable but {self.model!r} is not installed. "
                f"Install it with: ollama pull {self.model}"
            )
        elif supports_tools is False:
            detail = (
                f"{self.model} is installed but does not advertise tool support; "
                "the agent needs a tool-capable model"
            )
        else:
            detail = f"reachable; {self.model} is installed and supports tool calling"

        return ProviderStatus(
            provider=self.name,
            ok=installed and supports_tools is not False,
            model=self.model,
            detail=detail,
            capabilities=self.capabilities(),
            available_models=sorted(models)[:50],
        )

    async def _probe_tool_support(self) -> bool | None:
        """Ask `/api/show` whether the model advertises tool calling.

        Returns None when the server does not report capabilities, in which case
        the adapter stays optimistic and lets a real request decide.
        """
        try:
            response = await asyncio.wait_for(
                self._get_client().post("/api/show", json={"model": self.model}),
                timeout=min(self.timeout, 15.0),
            )
            if response.status_code >= 400:
                return None
            payload = response.json()
        except (TimeoutError, httpx.HTTPError, ValueError):
            return None
        capabilities = payload.get("capabilities")
        if isinstance(capabilities, list):
            return _TOOLS_CAPABILITY in capabilities
        template = str(payload.get("template", ""))
        if template:
            return ".Tools" in template or "tools" in template.lower()
        return None

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            native_tool_calls=self._supports_tools is not False,
            streaming=True,
            system_instruction=True,
            max_context_tokens=None,
        )

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
        self._client = None
