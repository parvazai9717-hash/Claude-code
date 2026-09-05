"""Provider abstraction, the mock provider, and malformed-response handling."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from agent.config import Config
from agent.errors import (
    ConfigurationError,
    ErrorCategory,
    ProviderInvalidRequestError,
    ProviderMalformedResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    ProviderUnsupportedError,
)
from agent.messages import FinishReason, Message, ToolDefinition
from agent.providers.base import BaseProvider
from agent.providers.factory import create_provider
from agent.providers.mock import MockProvider
from agent.providers.ollama import OllamaProvider

TOOLS = [ToolDefinition(name="read_file", description="read", parameters={"type": "object"})]


# -- base helpers -----------------------------------------------------------
def test_split_system_separates_instructions() -> None:
    system, body = BaseProvider.split_system(
        [Message.system("a"), Message.user("b"), Message.system("c")]
    )
    assert system == "a\n\nc"
    assert [m.role for m in body] == ["user"]


def test_parse_arguments_accepts_dict_and_json_string() -> None:
    assert BaseProvider.parse_arguments({"a": 1}, "t") == {"a": 1}
    assert BaseProvider.parse_arguments('{"a": 1}', "t") == {"a": 1}
    assert BaseProvider.parse_arguments(None, "t") == {}
    assert BaseProvider.parse_arguments("", "t") == {}


def test_parse_arguments_rejects_malformed() -> None:
    with pytest.raises(ProviderMalformedResponseError):
        BaseProvider.parse_arguments("{not json", "t")
    with pytest.raises(ProviderMalformedResponseError):
        BaseProvider.parse_arguments("[1,2]", "t")
    with pytest.raises(ProviderMalformedResponseError):
        BaseProvider.parse_arguments(42, "t")


def test_tool_call_without_a_name_is_malformed() -> None:
    with pytest.raises(ProviderMalformedResponseError, match="no name"):
        BaseProvider.make_tool_call(None, {})
    with pytest.raises(ProviderMalformedResponseError):
        BaseProvider.make_tool_call("", {})


# -- mock -------------------------------------------------------------------
async def test_mock_replays_the_script() -> None:
    provider = MockProvider(
        [MockProvider.call("read_file", {"path": "a"}), MockProvider.text("done")]
    )
    first = await provider.generate([Message.user("go")], TOOLS)
    assert first.has_tool_calls and first.tool_calls[0].name == "read_file"
    second = await provider.generate([Message.user("go")], TOOLS)
    assert second.text == "done" and second.finish_reason is FinishReason.STOP


async def test_mock_records_what_the_runtime_sent() -> None:
    provider = MockProvider([MockProvider.text("ok")])
    await provider.generate([Message.user("hi")], TOOLS, temperature=0.7, max_output_tokens=99)
    assert provider.calls[0]["tools"] == ["read_file"]
    assert provider.calls[0]["temperature"] == 0.7
    assert provider.calls[0]["max_output_tokens"] == 99


async def test_mock_exhausted_script_ends_cleanly() -> None:
    provider = MockProvider([MockProvider.text("a")])
    await provider.generate([], TOOLS)
    second = await provider.generate([], TOOLS)
    assert second.finish_reason is FinishReason.STOP


async def test_mock_callable_entries_see_the_conversation() -> None:
    def respond(messages: list[Message], tools: list[ToolDefinition]):
        return MockProvider.text(f"saw {len(messages)} messages")

    provider = MockProvider([respond])
    response = await provider.generate([Message.user("a"), Message.user("b")], TOOLS)
    assert response.text == "saw 2 messages"


async def test_mock_unhealthy_raises() -> None:
    with pytest.raises(ProviderUnavailableError):
        await MockProvider(healthy=False).generate([], TOOLS)


async def test_mock_health_check() -> None:
    status = await MockProvider().health_check()
    assert status.ok and status.capabilities is not None
    assert status.capabilities.native_tool_calls is True


# -- factory ----------------------------------------------------------------
def test_factory_builds_mock(config: Config) -> None:
    assert create_provider(config).name == "mock"


def test_factory_builds_ollama(config: Config) -> None:
    config.provider = "ollama"
    config.ollama_model = "llama3.1"
    provider = create_provider(config)
    assert provider.name == "ollama" and provider.model == "llama3.1"


def test_factory_rejects_unknown_provider(config: Config) -> None:
    config.provider = "nope"  # type: ignore[assignment]
    with pytest.raises(ConfigurationError, match="unknown provider"):
        create_provider(config)


def test_factory_reports_a_missing_gemini_key(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    config.provider = "gemini"
    with pytest.raises(ConfigurationError, match="GEMINI_API_KEY"):
        create_provider(config)


# -- ollama -----------------------------------------------------------------
def _ollama(handler: Any, **kwargs: Any) -> OllamaProvider:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url="http://test")
    return OllamaProvider(model="llama3.1", client=client, **kwargs)


async def test_ollama_parses_a_text_reply() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "llama3.1",
                "message": {"role": "assistant", "content": "hello"},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 10,
                "eval_count": 4,
            },
        )

    response = await _ollama(handler).generate([Message.user("hi")], [])
    assert response.text == "hello"
    assert response.finish_reason is FinishReason.STOP
    assert response.usage is not None and response.usage.total_tokens == 14


async def test_ollama_parses_tool_calls() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "read_file", "arguments": {"path": "a.txt"}}}
                    ],
                },
                "done": True,
            },
        )

    response = await _ollama(handler).generate([Message.user("hi")], TOOLS)
    assert response.tool_calls[0].name == "read_file"
    assert response.tool_calls[0].arguments == {"path": "a.txt"}
    assert response.finish_reason is FinishReason.TOOL_CALLS


async def test_ollama_sends_normalized_tools() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "x"}})

    await _ollama(handler).generate([Message.user("hi")], TOOLS)
    assert captured["tools"][0]["function"]["name"] == "read_file"
    assert captured["stream"] is False


async def test_ollama_missing_model_is_actionable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="model not found")

    with pytest.raises(ProviderInvalidRequestError, match="ollama pull"):
        await _ollama(handler).generate([Message.user("hi")], [])


async def test_ollama_malformed_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"nonsense": True})

    with pytest.raises(ProviderMalformedResponseError):
        await _ollama(handler).generate([Message.user("hi")], [])


async def test_ollama_non_json_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>not json</html>")

    with pytest.raises(ProviderMalformedResponseError):
        await _ollama(handler).generate([Message.user("hi")], [])


async def test_ollama_connection_error_is_actionable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(ProviderUnavailableError, match="ollama serve"):
        await _ollama(handler).generate([Message.user("hi")], [])


async def test_ollama_timeout_maps_to_timeout_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    with pytest.raises(ProviderTimeoutError):
        await _ollama(handler).generate([Message.user("hi")], [])


async def test_ollama_refuses_tools_on_an_incapable_model() -> None:
    """A model without tool support must error, never fake a call."""
    provider = _ollama(lambda r: httpx.Response(200, json={}))
    provider._supports_tools = False
    with pytest.raises(ProviderUnsupportedError, match="does not support tool calling"):
        await provider.generate([Message.user("hi")], TOOLS)


async def test_ollama_health_check_reports_missing_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "other-model"}]})
        return httpx.Response(404)

    status = await _ollama(handler).health_check()
    assert status.ok is False and "ollama pull" in status.detail


async def test_ollama_health_check_detects_tool_support() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "llama3.1:latest"}]})
        return httpx.Response(200, json={"capabilities": ["completion", "tools"]})

    status = await _ollama(handler).health_check()
    assert status.ok is True and "supports tool calling" in status.detail


async def test_ollama_health_check_flags_a_model_without_tools() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "llama3.1"}]})
        return httpx.Response(200, json={"capabilities": ["completion"]})

    status = await _ollama(handler).health_check()
    assert status.ok is False and "does not advertise tool support" in status.detail


async def test_ollama_health_check_unreachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    status = await _ollama(handler).health_check()
    assert status.ok is False
    assert status.error_category is ErrorCategory.PROVIDER_UNAVAILABLE
