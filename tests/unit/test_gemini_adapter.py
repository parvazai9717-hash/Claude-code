"""Gemini adapter: schema conversion, response parsing, error mapping.

These tests use a fake client rather than the network, so they run with no API
key and no connectivity. A live call against the real API is a manual test.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent.errors import (
    ConfigurationError,
    ProviderAuthError,
    ProviderInvalidRequestError,
    ProviderMalformedResponseError,
    ProviderRateLimitError,
    ProviderUnavailableError,
)
from agent.messages import FinishReason, Message, ToolDefinition

pytest.importorskip("google.genai", reason="the Gemini SDK is an optional extra")

from agent.providers.gemini import GeminiProvider, sanitize_schema

TOOLS = [
    ToolDefinition(
        name="read_file",
        description="read a file",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "path"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    )
]


class _FakePart:
    def __init__(self, text: str | None = None, function_call: Any = None, thought: Any = None):
        self.text = text
        self.function_call = function_call
        self.thought = thought


class _FakeFunctionCall:
    def __init__(self, name: str, args: Any, id: str | None = None):
        self.name = name
        self.args = args
        self.id = id


class _FakeCandidate:
    def __init__(self, parts: list[Any], finish_reason: str = "STOP"):
        self.content = type("C", (), {"parts": parts})()
        self.finish_reason = finish_reason


class _FakeResponse:
    def __init__(self, candidates: list[Any], usage: Any = None, prompt_feedback: Any = None):
        self.candidates = candidates
        self.usage_metadata = usage
        self.prompt_feedback = prompt_feedback


class _FakeModels:
    def __init__(self, response: Any = None, error: Exception | None = None):
        self._response = response
        self._error = error
        self.last_kwargs: dict[str, Any] = {}

    async def generate_content(self, **kwargs: Any) -> Any:
        self.last_kwargs = kwargs
        if self._error is not None:
            raise self._error
        return self._response


class _FakeClient:
    def __init__(self, response: Any = None, error: Exception | None = None):
        self.aio = type("Aio", (), {"models": _FakeModels(response, error)})()


def _provider(response: Any = None, error: Exception | None = None) -> GeminiProvider:
    return GeminiProvider(model="gemini-2.5-flash", client=_FakeClient(response, error))


# -- construction -----------------------------------------------------------
def test_missing_api_key_is_actionable() -> None:
    with pytest.raises(ConfigurationError, match="GEMINI_API_KEY"):
        GeminiProvider(environ={})


def test_api_key_is_read_from_the_environment() -> None:
    provider = GeminiProvider(environ={"GEMINI_API_KEY": "sk-test-key-value-123456"})
    assert provider.model == "gemini-2.5-flash"
    # The key is held privately and never exposed through a public attribute.
    assert "sk-test-key-value-123456" not in repr(provider.capabilities())


# -- schema conversion ------------------------------------------------------
def test_sanitize_schema_drops_unsupported_keys() -> None:
    cleaned = sanitize_schema(TOOLS[0].json_schema())
    assert "additionalProperties" not in cleaned
    assert cleaned["properties"]["path"]["type"] == "string"
    assert cleaned["required"] == ["path"]


def test_sanitize_schema_recurses_into_arrays() -> None:
    cleaned = sanitize_schema(
        {
            "type": "object",
            "properties": {
                "xs": {"type": "array", "items": {"type": "string", "additionalProperties": False}}
            },
        }
    )
    assert "additionalProperties" not in cleaned["properties"]["xs"]["items"]


# -- response parsing -------------------------------------------------------
async def test_text_response() -> None:
    provider = _provider(_FakeResponse([_FakeCandidate([_FakePart(text="hello")])]))
    response = await provider.generate([Message.user("hi")], [])
    assert response.text == "hello"
    assert response.finish_reason is FinishReason.STOP
    assert response.provider == "gemini"


async def test_function_call_response() -> None:
    call = _FakeFunctionCall("read_file", {"path": "a.txt"})
    provider = _provider(
        _FakeResponse([_FakeCandidate([_FakePart(function_call=call)], finish_reason="STOP")])
    )
    response = await provider.generate([Message.user("hi")], TOOLS)
    assert response.tool_calls[0].name == "read_file"
    assert response.tool_calls[0].arguments == {"path": "a.txt"}
    assert response.finish_reason is FinishReason.TOOL_CALLS


async def test_thought_parts_are_never_surfaced() -> None:
    """Private reasoning must not reach the runtime, the logs or storage."""
    parts = [_FakePart(text="private reasoning", thought=True), _FakePart(text="public answer")]
    provider = _provider(_FakeResponse([_FakeCandidate(parts)]))
    response = await provider.generate([Message.user("hi")], [])
    assert response.text == "public answer"
    assert "private reasoning" not in (response.text or "")


async def test_usage_is_normalized() -> None:
    usage = type(
        "U", (), {"prompt_token_count": 10, "candidates_token_count": 5, "total_token_count": 15}
    )()
    provider = _provider(_FakeResponse([_FakeCandidate([_FakePart(text="x")])], usage=usage))
    response = await provider.generate([Message.user("hi")], [])
    assert response.usage is not None and response.usage.total_tokens == 15


async def test_max_tokens_maps_to_length() -> None:
    provider = _provider(
        _FakeResponse([_FakeCandidate([_FakePart(text="x")], finish_reason="MAX_TOKENS")])
    )
    response = await provider.generate([Message.user("hi")], [])
    assert response.finish_reason is FinishReason.LENGTH


async def test_no_candidates_is_malformed() -> None:
    provider = _provider(_FakeResponse([]))
    with pytest.raises(ProviderMalformedResponseError):
        await provider.generate([Message.user("hi")], [])


async def test_safety_block_is_reported_not_crashed() -> None:
    feedback = type("F", (), {"block_reason": "SAFETY"})()
    provider = _provider(_FakeResponse([], prompt_feedback=feedback))
    response = await provider.generate([Message.user("hi")], [])
    assert response.finish_reason is FinishReason.SAFETY


async def test_empty_conversation_is_rejected() -> None:
    provider = _provider(_FakeResponse([_FakeCandidate([_FakePart(text="x")])]))
    with pytest.raises(ProviderInvalidRequestError, match="empty"):
        await provider.generate([Message.system("only a system message")], [])


async def test_automatic_function_calling_is_disabled() -> None:
    """The runtime must execute tools, so the SDK must never do it itself."""
    provider = _provider(_FakeResponse([_FakeCandidate([_FakePart(text="x")])]))
    await provider.generate([Message.user("hi")], TOOLS)
    config = provider._client.aio.models.last_kwargs["config"]  # type: ignore[union-attr]
    assert config.automatic_function_calling.disable is True


async def test_system_message_is_sent_out_of_band() -> None:
    provider = _provider(_FakeResponse([_FakeCandidate([_FakePart(text="x")])]))
    await provider.generate([Message.system("be careful"), Message.user("hi")], [])
    kwargs = provider._client.aio.models.last_kwargs  # type: ignore[union-attr]
    assert kwargs["config"].system_instruction == "be careful"
    assert all(getattr(c, "role", "") != "system" for c in kwargs["contents"])


# -- error mapping ----------------------------------------------------------
class _ApiError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@pytest.mark.parametrize(
    ("code", "message", "expected"),
    [
        (401, "unauthenticated", ProviderAuthError),
        (403, "permission denied for api key", ProviderAuthError),
        (404, "model not found", ProviderInvalidRequestError),
        (400, "invalid argument", ProviderInvalidRequestError),
    ],
)
def test_error_mapping(code: int, message: str, expected: type[Exception]) -> None:
    mapped = _provider()._map_error(_ApiError(code, message))
    assert isinstance(mapped, expected)


def test_rate_limit_is_retryable() -> None:
    mapped = _provider()._map_error(_ApiError(429, "resource exhausted"))
    assert isinstance(mapped, ProviderRateLimitError) and mapped.retryable


def test_server_error_is_retryable() -> None:
    mapped = _provider()._map_error(_ApiError(503, "service unavailable"))
    assert isinstance(mapped, ProviderUnavailableError) and mapped.retryable


def test_error_messages_never_echo_the_key() -> None:
    provider = GeminiProvider(environ={"GEMINI_API_KEY": "sk-secret-key-abcdef123456"})
    mapped = provider._map_error(
        _ApiError(400, "request with key sk-secret-key-abcdef123456 failed")
    )
    assert "sk-secret-key-abcdef123456" not in mapped.message


async def test_non_retryable_error_is_raised_immediately() -> None:
    provider = _provider(error=_ApiError(401, "unauthenticated"))
    with pytest.raises(ProviderAuthError):
        await provider.generate([Message.user("hi")], [])


async def test_health_check_reports_failure_without_raising() -> None:
    provider = _provider(error=_ApiError(401, "unauthenticated"))

    async def failing_list() -> Any:
        raise _ApiError(401, "unauthenticated")

    provider._client.aio.models.list = failing_list  # type: ignore[union-attr]
    status = await provider.health_check()
    assert status.ok is False and status.error_category is not None


def test_capabilities_declare_native_tool_calls() -> None:
    assert _provider().capabilities().native_tool_calls is True
