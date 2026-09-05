"""Normalized data model and the event bus."""

from __future__ import annotations

import json
from pathlib import Path

from agent.errors import ErrorCategory
from agent.events import (
    Event,
    EventBus,
    EventType,
    JsonlEventWriter,
    summarize_events,
)
from agent.messages import (
    FinishReason,
    Message,
    ModelResponse,
    RiskCategory,
    RiskLevel,
    ToolCall,
    ToolDefinition,
    ToolResult,
    new_id,
)
from agent.security.redaction import Redactor


# -- messages ---------------------------------------------------------------
def test_message_constructors() -> None:
    assert Message.system("s").role == "system"
    assert Message.user("u").role == "user"
    assistant = Message.assistant("a", [ToolCall(name="t")])
    assert assistant.role == "assistant" and assistant.tool_calls[0].name == "t"


def test_tool_result_renders_compact_json_for_the_model() -> None:
    result = ToolResult(call_id="c", tool_name="read_file", ok=True, output={"path": "a"})
    payload = json.loads(result.to_model_text())
    assert payload == {"ok": True, "output": {"path": "a"}}


def test_failed_tool_result_carries_a_category() -> None:
    result = ToolResult(
        call_id="c",
        tool_name="t",
        ok=False,
        error_category=ErrorCategory.PERMISSION_DENIED,
        error_message="nope",
    )
    payload = json.loads(result.to_model_text())
    assert payload["error"]["category"] == "permission_denied"
    assert payload["error"]["message"] == "nope"


def test_tool_result_flags_are_surfaced() -> None:
    result = ToolResult(
        call_id="c",
        tool_name="t",
        ok=True,
        truncated=True,
        timed_out=True,
        verified=False,
        verification_evidence="nothing found",
    )
    payload = json.loads(result.to_model_text())
    assert payload["truncated"] and payload["timed_out"]
    assert payload["verified"] is False and payload["evidence"] == "nothing found"


def test_message_from_tool_result_links_the_call() -> None:
    result = ToolResult(call_id="call_1", tool_name="read_file", ok=True)
    message = Message.from_tool_result(result)
    assert message.role == "tool"
    assert message.tool_call_id == "call_1"
    assert message.name == "read_file"


def test_tool_definition_schema_defaults_are_strict() -> None:
    schema = ToolDefinition(name="t", description="d").json_schema()
    assert schema["additionalProperties"] is False
    assert schema["type"] == "object"


def test_tool_definition_is_immutable() -> None:
    definition = ToolDefinition(name="t", description="d")
    try:
        definition.name = "other"  # type: ignore[misc]
    except Exception as exc:
        assert "frozen" in str(exc).lower() or "immutable" in str(exc).lower()
    else:  # pragma: no cover - would indicate a model config regression
        raise AssertionError("ToolDefinition must be frozen")


def test_risk_classifications_exist() -> None:
    assert RiskLevel.READ_ONLY.value == "read_only"
    assert RiskCategory.SHELL.value == "shell"


def test_model_response_has_tool_calls() -> None:
    assert ModelResponse().has_tool_calls is False
    assert ModelResponse(tool_calls=[ToolCall(name="t")]).has_tool_calls is True


def test_ids_are_prefixed_and_unique() -> None:
    first, second = new_id("task"), new_id("task")
    assert first.startswith("task_") and first != second


def test_finish_reasons_cover_the_expected_cases() -> None:
    assert {r.value for r in FinishReason} >= {"stop", "tool_calls", "length", "safety", "error"}


# -- events -----------------------------------------------------------------
def test_bus_dispatches_to_subscribers() -> None:
    bus = EventBus()
    seen: list[Event] = []
    bus.subscribe(seen.append)
    bus.emit_event(EventType.TASK_STARTED, task_id="t1", message="go")
    assert len(seen) == 1 and seen[0].task_id == "t1"


def test_unsubscribe_stops_delivery() -> None:
    bus = EventBus()
    seen: list[Event] = []
    unsubscribe = bus.subscribe(seen.append)
    bus.emit_event(EventType.MESSAGE)
    unsubscribe()
    bus.emit_event(EventType.MESSAGE)
    assert len(seen) == 1


def test_a_failing_subscriber_cannot_break_a_run() -> None:
    bus = EventBus()
    seen: list[Event] = []

    def broken(event: Event) -> None:
        raise RuntimeError("subscriber exploded")

    bus.subscribe(broken)
    bus.subscribe(seen.append)
    bus.emit_event(EventType.MESSAGE, message="still delivered")
    assert len(seen) == 1


def test_events_are_redacted_centrally() -> None:
    redactor = Redactor(environ={"MY_API_KEY": "sk-event-secret-1234567"})
    bus = EventBus(redactor=redactor.redact)
    seen: list[Event] = []
    bus.subscribe(seen.append)
    bus.emit_event(
        EventType.TOOL_RESULT,
        message="key is sk-event-secret-1234567",
        data={"api_key": "sk-event-secret-1234567"},
    )
    assert "sk-event-secret-1234567" not in seen[0].message
    assert "sk-event-secret-1234567" not in str(seen[0].data)


def test_history_is_bounded_and_readable() -> None:
    bus = EventBus()
    for index in range(5):
        bus.emit_event(EventType.MESSAGE, message=str(index))
    assert len(bus.history) == 5


def test_jsonl_writer_appends_one_object_per_line(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "events.jsonl"
    writer = JsonlEventWriter(path)
    writer(Event(type=EventType.TASK_STARTED, message="a"))
    writer(Event(type=EventType.TASK_FINISHED, message="b"))
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["type"] == "task_started"


def test_summarize_events_counts_by_type() -> None:
    events = [
        Event(type=EventType.TOOL_RESULT),
        Event(type=EventType.TOOL_RESULT),
        Event(type=EventType.REPLAN),
    ]
    assert summarize_events(events) == {"tool_result": 2, "replan": 1}
