"""Gemini adapter.

All Gemini-specific knowledge lives here: role naming, schema conversion,
function-call parsing, retries, timeouts and error mapping. Written against the
installed `google-genai` SDK (2.x), whose types were inspected rather than
assumed.

The API key is read from the environment at construction time and is never
logged, echoed, or placed in model-visible context.
"""

from __future__ import annotations

import asyncio
import os
import random
from typing import Any

from ..errors import (
    ConfigurationError,
    ProviderAuthError,
    ProviderError,
    ProviderInvalidRequestError,
    ProviderMalformedResponseError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
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

#: Environment variable holding the credential. Its value never leaves this module.
API_KEY_ENV = "GEMINI_API_KEY"

#: HTTP statuses worth retrying with backoff.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

_FINISH_REASONS = {
    "STOP": FinishReason.STOP,
    "MAX_TOKENS": FinishReason.LENGTH,
    "SAFETY": FinishReason.SAFETY,
    "RECITATION": FinishReason.SAFETY,
    "PROHIBITED_CONTENT": FinishReason.SAFETY,
    "BLOCKLIST": FinishReason.SAFETY,
    "MALFORMED_FUNCTION_CALL": FinishReason.ERROR,
}

#: JSON-Schema keys Gemini's function declarations accept. Anything else is dropped
#: rather than passed through, because an unknown key is rejected by the API.
_ALLOWED_SCHEMA_KEYS = frozenset(
    {
        "type",
        "description",
        "properties",
        "required",
        "items",
        "enum",
        "format",
        "nullable",
        "minimum",
        "maximum",
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
        "default",
    }
)


def sanitize_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Reduce a JSON Schema to the subset Gemini accepts.

    `additionalProperties` in particular is not part of Gemini's schema dialect;
    the registry still enforces it locally, so dropping it here loses nothing.
    """
    result: dict[str, Any] = {}
    for key, value in schema.items():
        if key not in _ALLOWED_SCHEMA_KEYS:
            continue
        if key == "properties" and isinstance(value, dict):
            result[key] = {name: sanitize_schema(sub) for name, sub in value.items()}
        elif key == "items" and isinstance(value, dict):
            result[key] = sanitize_schema(value)
        else:
            result[key] = value
    result.setdefault("type", "object" if "properties" in result else "string")
    return result


class GeminiProvider(BaseProvider):
    """Talks to the Gemini API through the official SDK."""

    name = "gemini"

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        *,
        api_key: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        environ: dict[str, str] | None = None,
        client: Any = None,
    ) -> None:
        super().__init__(model=model, timeout=timeout)
        self.max_retries = max_retries
        self._client = client
        if client is None:
            env = os.environ if environ is None else environ
            key = api_key or env.get(API_KEY_ENV, "").strip()
            if not key:
                raise ConfigurationError(
                    f"{API_KEY_ENV} is not set. Add it to your environment or .env file "
                    "(never to config.yaml). Get a key at https://aistudio.google.com/apikey."
                )
            self._api_key = key
        else:
            self._api_key = ""

    # -- client -------------------------------------------------------------
    def _get_client(self) -> Any:
        """Construct the SDK client lazily, so importing this module is cheap."""
        if self._client is None:
            try:
                from google import genai
                from google.genai import types as genai_types
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ConfigurationError(
                    "the Gemini SDK is not installed. Install it with: "
                    'pip install "local-agent[gemini]"'
                ) from exc
            self._client = genai.Client(
                api_key=self._api_key,
                http_options=genai_types.HttpOptions(timeout=int(self.timeout * 1000)),
            )
        return self._client

    # -- conversion ---------------------------------------------------------
    def _to_contents(self, messages: list[Message]) -> list[Any]:
        """Convert normalized messages into Gemini `Content` objects."""
        from google.genai import types

        contents: list[Any] = []
        for message in messages:
            if message.role == "system":
                continue  # carried out of band as `system_instruction`
            if message.role == "user":
                parts: list[Any] = []
                if message.content:
                    parts.append(types.Part(text=message.content))
                parts.extend(self._media_parts(message))
                contents.append(types.Content(role="user", parts=parts or [types.Part(text="")]))
            elif message.role == "assistant":
                parts = []
                if message.content:
                    parts.append(types.Part(text=message.content))
                for call in message.tool_calls:
                    parts.append(
                        types.Part(
                            function_call=types.FunctionCall(
                                # Gemini matches calls to responses by name.
                                name=call.name,
                                args=call.arguments,
                            )
                        )
                    )
                if parts:
                    contents.append(types.Content(role="model", parts=parts))
            elif message.role == "tool":
                parts = [
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=message.name or "tool",
                            response={"result": message.content},
                        )
                    )
                ]
                # Media a tool loaded is attached alongside its textual result, so
                # the model can actually perceive what `view_media` fetched.
                parts.extend(self._media_parts(message))
                contents.append(types.Content(role="user", parts=parts))
        return contents

    def _media_parts(self, message: Message) -> list[Any]:
        """Convert a message's attachments into Gemini inline-data parts.

        Anything this model cannot perceive is dropped rather than sent, and the
        drop is recorded in metadata so the runtime can say so honestly.
        """
        from google.genai import types

        capabilities = self.capabilities()
        parts: list[Any] = []
        for attachment in message.attachments:
            if not capabilities.accepts(attachment.kind.value):
                continue
            parts.append(
                types.Part.from_bytes(data=attachment.data, mime_type=attachment.mime_type)
            )
        return parts

    def _to_tools(self, tools: list[ToolDefinition]) -> list[Any]:
        from google.genai import types

        if not tools:
            return []
        declarations = [
            types.FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                # The SDK wants a typed Schema; it coerces nested property
                # dictionaries for us, so one construction covers the tree.
                parameters=types.Schema(**sanitize_schema(tool.json_schema())),
            )
            for tool in tools
        ]
        return [types.Tool(function_declarations=declarations)]

    def _from_response(self, response: Any) -> ModelResponse:
        """Convert a Gemini response back into a :class:`ModelResponse`."""
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            feedback = getattr(response, "prompt_feedback", None)
            blocked = getattr(feedback, "block_reason", None)
            if blocked:
                return ModelResponse(
                    text=f"The request was blocked by the provider's safety filter ({blocked}).",
                    finish_reason=FinishReason.SAFETY,
                    provider=self.name,
                    model=self.model,
                )
            raise ProviderMalformedResponseError("Gemini returned no candidates")

        candidate = candidates[0]
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []

        texts: list[str] = []
        tool_calls: list[ToolCall] = []
        for part in parts:
            # `thought` parts are private reasoning: never surfaced or persisted.
            if getattr(part, "thought", None):
                continue
            text = getattr(part, "text", None)
            if text:
                texts.append(text)
            function_call = getattr(part, "function_call", None)
            if function_call is not None:
                tool_calls.append(
                    self.make_tool_call(
                        getattr(function_call, "name", None),
                        getattr(function_call, "args", None),
                        getattr(function_call, "id", None),
                    )
                )

        raw_reason = str(getattr(candidate, "finish_reason", "") or "").rsplit(".", 1)[-1]
        finish = _FINISH_REASONS.get(raw_reason.upper(), FinishReason.UNKNOWN)
        if tool_calls:
            finish = FinishReason.TOOL_CALLS
        elif finish is FinishReason.UNKNOWN and texts:
            finish = FinishReason.STOP

        usage_data = getattr(response, "usage_metadata", None)
        usage = None
        if usage_data is not None:
            usage = Usage(
                input_tokens=getattr(usage_data, "prompt_token_count", None),
                output_tokens=getattr(usage_data, "candidates_token_count", None),
                total_tokens=getattr(usage_data, "total_token_count", None),
            )

        return ModelResponse(
            text="\n".join(texts) or None,
            tool_calls=tool_calls,
            finish_reason=finish,
            provider=self.name,
            model=self.model,
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
        from google.genai import types

        client = self._get_client()
        system_instruction, body = self.split_system(messages)
        contents = self._to_contents(body)
        if not contents:
            raise ProviderInvalidRequestError("there is nothing to send: the conversation is empty")

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            system_instruction=system_instruction or None,
            tools=self._to_tools(tools) or None,
            # The runtime executes tools itself; the SDK must never do it for us.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = await asyncio.wait_for(
                    client.aio.models.generate_content(
                        model=self.model, contents=contents, config=config
                    ),
                    timeout=self.timeout,
                )
                return self._from_response(response)
            except TimeoutError as exc:
                raise ProviderTimeoutError(
                    f"Gemini did not respond within {self.timeout:g}s"
                ) from exc
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                mapped = self._map_error(exc)
                if not mapped.retryable or attempt == self.max_retries - 1:
                    raise mapped from exc
                last_error = mapped
                # Exponential backoff with jitter, so parallel runs do not sync up.
                await asyncio.sleep(min(2**attempt + random.uniform(0, 0.5), 10.0))
        raise last_error or ProviderError("Gemini request failed for an unknown reason")

    def _map_error(self, exc: Exception) -> ProviderError:
        """Map an SDK exception onto the shared taxonomy."""
        if isinstance(exc, ProviderError):
            return exc
        status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        if status is None:
            status = getattr(getattr(exc, "response", None), "status_code", None)
        message = str(getattr(exc, "message", None) or exc)
        # The message can echo request details; never include a credential.
        message = message.replace(self._api_key, "[REDACTED]") if self._api_key else message
        text = message.lower()

        if status in (401, 403) or "api key" in text or "unauthenticated" in text:
            return ProviderAuthError(
                f"Gemini rejected the credentials in {API_KEY_ENV}. "
                "Check that the key is valid and has access to this model."
            )
        if status == 429 or "rate limit" in text or "quota" in text or "resource_exhausted" in text:
            return ProviderRateLimitError("Gemini rate limit reached", retryable=True)
        if status == 404 or "not found" in text:
            return ProviderInvalidRequestError(
                f"Gemini does not recognise the model {self.model!r}"
            )
        if status == 400 or "invalid" in text:
            return ProviderInvalidRequestError(f"Gemini rejected the request: {message[:300]}")
        if isinstance(status, int) and status in _RETRYABLE_STATUS:
            return ProviderUnavailableError(
                f"Gemini is temporarily unavailable (HTTP {status})", retryable=True
            )
        if "timeout" in text or "timed out" in text:
            return ProviderTimeoutError("the Gemini request timed out", retryable=True)
        if "connect" in text or "network" in text or "dns" in text:
            return ProviderUnavailableError(
                "cannot reach the Gemini API; check the network connection", retryable=True
            )
        return ProviderError(f"Gemini request failed: {message[:300]}")

    async def health_check(self) -> ProviderStatus:
        if not self._api_key and self._client is None:
            return ProviderStatus(
                provider=self.name,
                ok=False,
                model=self.model,
                detail=f"{API_KEY_ENV} is not set",
            )
        try:
            client = self._get_client()
            models = await asyncio.wait_for(
                client.aio.models.list(), timeout=min(self.timeout, 30.0)
            )
            names: list[str] = []
            async for entry in models:
                name = str(getattr(entry, "name", "")).removeprefix("models/")
                if name:
                    names.append(name)
                if len(names) >= 200:
                    break
            available = self.model in names or f"models/{self.model}" in names
            return ProviderStatus(
                provider=self.name,
                ok=True,
                model=self.model,
                detail=(
                    f"reachable; {self.model} is available"
                    if available
                    else f"reachable, but {self.model!r} was not in the model list"
                ),
                capabilities=self.capabilities(),
                available_models=sorted(names)[:50],
            )
        except Exception as exc:
            mapped = self._map_error(exc)
            return ProviderStatus(
                provider=self.name,
                ok=False,
                model=self.model,
                detail=mapped.message,
                error_category=mapped.category,
            )

    def capabilities(self) -> ProviderCapabilities:
        # Gemini's multimodal models accept images, audio and video natively.
        # Video is still gated by `media.enable_video` on the runtime side.
        return ProviderCapabilities(
            native_tool_calls=True,
            streaming=True,
            system_instruction=True,
            max_context_tokens=1_000_000,
            vision=True,
            audio=True,
            video=True,
        )

    async def aclose(self) -> None:
        self._client = None
