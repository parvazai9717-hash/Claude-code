"""Registry behaviour: validation, permissions, approvals, limits, redaction."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agent.config import ApprovalMode, Config
from agent.errors import AgentError, ErrorCategory, InvalidArgumentsError
from agent.events import EventBus, EventType
from agent.messages import RiskCategory, RiskLevel, ToolCall
from agent.security.approvals import ApprovalDecision, AutoDenyApprover, PolicyApprover
from agent.security.permissions import PermissionChecker
from agent.tools.base import Tool, ToolContext
from agent.tools.registry import ToolRegistry, validate_arguments


class EchoTool(Tool):
    name = "echo_tool"
    description = "Echo a value back."
    parameters = {
        "type": "object",
        "properties": {
            "value": {"type": "string"},
            "count": {"type": "integer", "minimum": 1, "maximum": 5, "default": 1},
        },
        "required": ["value"],
        "additionalProperties": False,
    }

    async def run(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        return {"echoed": arguments["value"] * int(arguments.get("count", 1))}


class BoomTool(Tool):
    name = "boom_tool"
    description = "Always raises."

    async def run(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        raise RuntimeError("internal failure detail")


class SlowTool(Tool):
    name = "slow_tool"
    description = "Sleeps forever."

    async def run(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        await asyncio.sleep(30)
        return {}


class WriteyTool(Tool):
    name = "writey_tool"
    description = "A side-effecting tool."
    read_only = False
    requires_approval = True
    risk = RiskLevel.MEDIUM
    risk_category = RiskCategory.WRITE

    async def run(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        return {"did": "something"}


class SecretTool(Tool):
    name = "secret_tool"
    description = "Returns something secret-shaped."

    async def run(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        return {
            "api_key": "sk-test-secret-value-1234",
            "note": "token is sk-test-secret-value-1234",
        }


def _registry(config: Config, approver: Any = None) -> ToolRegistry:
    return ToolRegistry(
        permissions=PermissionChecker(config),
        approver=approver or PolicyApprover(default=ApprovalDecision.APPROVE_ONCE),
        events=EventBus(),
    )


# -- schema validation ------------------------------------------------------
def test_validate_fills_defaults_and_accepts_valid() -> None:
    schema = EchoTool().definition().json_schema()
    assert validate_arguments(schema, {"value": "x"}) == {"value": "x", "count": 1}


def test_unknown_arguments_are_rejected() -> None:
    schema = EchoTool().definition().json_schema()
    with pytest.raises(InvalidArgumentsError, match="unknown argument"):
        validate_arguments(schema, {"value": "x", "surprise": 1})


def test_missing_required_argument_is_rejected() -> None:
    schema = EchoTool().definition().json_schema()
    with pytest.raises(InvalidArgumentsError, match="missing required"):
        validate_arguments(schema, {})


def test_wrong_type_is_rejected() -> None:
    schema = EchoTool().definition().json_schema()
    with pytest.raises(InvalidArgumentsError, match="must be of type"):
        validate_arguments(schema, {"value": 42})


def test_boolean_does_not_satisfy_integer() -> None:
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}}
    with pytest.raises(InvalidArgumentsError, match="got boolean"):
        validate_arguments(schema, {"n": True})


def test_bounds_and_enums_are_enforced() -> None:
    schema = {
        "type": "object",
        "properties": {
            "n": {"type": "integer", "minimum": 1, "maximum": 3},
            "mode": {"type": "string", "enum": ["a", "b"]},
            "s": {"type": "string", "maxLength": 2},
        },
    }
    with pytest.raises(InvalidArgumentsError, match=">= 1"):
        validate_arguments(schema, {"n": 0})
    with pytest.raises(InvalidArgumentsError, match="<= 3"):
        validate_arguments(schema, {"n": 9})
    with pytest.raises(InvalidArgumentsError, match="one of"):
        validate_arguments(schema, {"mode": "c"})
    with pytest.raises(InvalidArgumentsError, match="at most 2 characters"):
        validate_arguments(schema, {"s": "abc"})


def test_array_items_are_validated() -> None:
    schema = {
        "type": "object",
        "properties": {"xs": {"type": "array", "items": {"type": "string"}, "maxItems": 2}},
    }
    assert validate_arguments(schema, {"xs": ["a"]}) == {"xs": ["a"]}
    with pytest.raises(InvalidArgumentsError):
        validate_arguments(schema, {"xs": [1]})
    with pytest.raises(InvalidArgumentsError, match="at most 2 items"):
        validate_arguments(schema, {"xs": ["a", "b", "c"]})


def test_non_object_arguments_are_rejected() -> None:
    with pytest.raises(InvalidArgumentsError):
        validate_arguments({"type": "object"}, ["not", "a", "dict"])  # type: ignore[arg-type]


# -- registration -----------------------------------------------------------
def test_duplicate_registration_is_rejected(config: Config) -> None:
    registry = _registry(config)
    registry.register(EchoTool())
    with pytest.raises(AgentError, match="already registered"):
        registry.register(EchoTool())


def test_invalid_tool_name_is_rejected(config: Config) -> None:
    class Bad(EchoTool):
        name = "Bad Name!"

    with pytest.raises(AgentError, match="invalid tool name"):
        _registry(config).register(Bad())


def test_definitions_are_sorted_and_normalized(config: Config) -> None:
    registry = _registry(config)
    registry.register_all([WriteyTool(), EchoTool()])
    names = [d.name for d in registry.definitions()]
    assert names == ["echo_tool", "writey_tool"]
    assert registry.definitions()[0].json_schema()["additionalProperties"] is False


# -- execution --------------------------------------------------------------
async def test_successful_execution(config: Config, context: ToolContext) -> None:
    registry = _registry(config)
    registry.register(EchoTool())
    result = await registry.execute(
        ToolCall(name="echo_tool", arguments={"value": "ab", "count": 2}), context
    )
    assert result.ok and result.output == {"echoed": "abab"}


async def test_unknown_tool_returns_structured_failure(
    config: Config, context: ToolContext
) -> None:
    result = await _registry(config).execute(ToolCall(name="nope", arguments={}), context)
    assert result.ok is False
    assert result.error_category is ErrorCategory.UNKNOWN_TOOL


async def test_invalid_arguments_return_structured_failure(
    config: Config, context: ToolContext
) -> None:
    registry = _registry(config)
    registry.register(EchoTool())
    result = await registry.execute(ToolCall(name="echo_tool", arguments={}), context)
    assert result.ok is False
    assert result.error_category is ErrorCategory.INVALID_ARGUMENTS


async def test_tool_exception_becomes_a_failure_not_a_raise(
    config: Config, context: ToolContext
) -> None:
    registry = _registry(config)
    registry.register(BoomTool())
    result = await registry.execute(ToolCall(name="boom_tool", arguments={}), context)
    assert result.ok is False
    assert result.error_category is ErrorCategory.TOOL_FAILED


async def test_timeout_is_reported(config: Config, context: ToolContext) -> None:
    config.limits.tool_timeout_seconds = 0.05
    registry = _registry(config)
    registry.register(SlowTool())
    result = await registry.execute(ToolCall(name="slow_tool", arguments={}), context)
    assert result.ok is False and result.timed_out
    assert result.error_category is ErrorCategory.TOOL_TIMEOUT


async def test_approval_denial_is_reported(config: Config, context: ToolContext) -> None:
    registry = _registry(config, approver=AutoDenyApprover())
    registry.register(WriteyTool())
    result = await registry.execute(ToolCall(name="writey_tool", arguments={}), context)
    assert result.ok is False
    assert result.error_category is ErrorCategory.APPROVAL_DENIED


async def test_cancel_at_the_prompt_stops_the_run(config: Config, context: ToolContext) -> None:
    approver = PolicyApprover(default=ApprovalDecision.CANCEL)
    registry = _registry(config, approver=approver)
    registry.register(WriteyTool())
    result = await registry.execute(ToolCall(name="writey_tool", arguments={}), context)
    assert result.error_category is ErrorCategory.CANCELLED


async def test_approve_for_run_is_not_asked_twice(config: Config, context: ToolContext) -> None:
    approver = PolicyApprover(default=ApprovalDecision.APPROVE_FOR_RUN)
    registry = _registry(config, approver=approver)
    registry.register(WriteyTool())
    for _ in range(3):
        assert (await registry.execute(ToolCall(name="writey_tool"), context)).ok
    assert len(approver.seen) == 1


async def test_read_only_tools_are_never_sent_to_the_approver(
    config: Config, context: ToolContext
) -> None:
    approver = PolicyApprover(default=ApprovalDecision.DENY)
    registry = _registry(config, approver=approver)
    registry.register(EchoTool())
    assert (
        await registry.execute(ToolCall(name="echo_tool", arguments={"value": "x"}), context)
    ).ok
    assert approver.seen == []


async def test_automatic_mode_denies_without_asking(config: Config, context: ToolContext) -> None:
    config.approval_mode = ApprovalMode.AUTOMATIC
    approver = PolicyApprover(default=ApprovalDecision.APPROVE_ONCE)
    registry = _registry(config, approver=approver)
    registry.register(WriteyTool())
    result = await registry.execute(ToolCall(name="writey_tool"), context)
    assert result.ok is False
    assert result.error_category is ErrorCategory.PERMISSION_DENIED
    assert approver.seen == [], "the approver must not be consulted in automatic mode"


async def test_output_is_redacted(config: Config, context: ToolContext) -> None:
    registry = _registry(config)
    registry.register(SecretTool())
    result = await registry.execute(ToolCall(name="secret_tool"), context)
    assert "sk-test-secret-value-1234" not in str(result.output)


async def test_long_output_is_truncated(config: Config, context: ToolContext) -> None:
    config.limits.max_tool_output_chars = 50

    class BigTool(Tool):
        name = "big_tool"
        description = "Returns a lot."

        async def run(self, arguments: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
            return {"text": "x" * 500}

    registry = _registry(config)
    registry.register(BigTool())
    result = await registry.execute(ToolCall(name="big_tool"), context)
    assert result.truncated and "truncated" in result.output["text"]


async def test_events_are_emitted_for_a_call(config: Config, context: ToolContext) -> None:
    bus = EventBus()
    seen: list[EventType] = []
    bus.subscribe(lambda e: seen.append(e.type))
    registry = ToolRegistry(
        permissions=PermissionChecker(config),
        approver=PolicyApprover(default=ApprovalDecision.APPROVE_ONCE),
        events=bus,
    )
    registry.register(WriteyTool())
    await registry.execute(ToolCall(name="writey_tool"), context)
    assert EventType.TOOL_REQUESTED in seen
    assert EventType.APPROVAL_REQUESTED in seen
    assert EventType.TOOL_RESULT in seen


def test_the_model_cannot_register_tools(config: Config) -> None:
    """Registration is a Python-only operation; no tool exposes it."""
    registry = _registry(config)
    registry.register_all([EchoTool(), WriteyTool()])
    for definition in registry.definitions():
        schema = str(definition.json_schema()) + definition.description
        assert "register" not in schema.lower()
