"""Tool registry: the single gate every tool call passes through.

The registry owns argument validation, permission checks, approval requests,
timeouts, redaction, truncation and structured error reporting. A tool that is
not registered here cannot run, and the model cannot register one: registration
happens only from Python, at startup.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterable, Sequence
from typing import Any

from ..errors import (
    AgentError,
    ErrorCategory,
    InvalidArgumentsError,
    UnknownToolError,
)
from ..events import EventBus, EventType
from ..messages import ToolCall, ToolDefinition, ToolResult
from ..security.approvals import (
    ApprovalDecision,
    ApprovalRequest,
    Approver,
    AutoDenyApprover,
    describe_request,
)
from ..security.limits import truncate_output
from ..security.permissions import PermissionChecker
from .base import Timer, Tool, ToolContext, error_from_exception

#: Tool names must look like identifiers so no provider mangles them.
NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


# --------------------------------------------------------------------------
# Minimal JSON-Schema validation
# --------------------------------------------------------------------------
_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
    "null": (type(None),),
}


def validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    """Validate and coerce `arguments` against a JSON-Schema object.

    Supports the subset the tools actually use: `type`, `properties`, `required`,
    `additionalProperties`, `enum`, `default`, `items`, numeric bounds and string
    lengths. Unknown properties are rejected unless the schema opts in.

    Returns:
        A new dictionary containing only declared properties, with defaults filled.

    Raises:
        InvalidArgumentsError: With a message naming the offending field.
    """
    if not isinstance(arguments, dict):
        raise InvalidArgumentsError("tool arguments must be a JSON object")

    properties: dict[str, Any] = schema.get("properties", {}) or {}
    required: list[str] = list(schema.get("required", []) or [])
    allow_extra = bool(schema.get("additionalProperties", False))

    unknown = sorted(set(arguments) - set(properties))
    if unknown and not allow_extra:
        raise InvalidArgumentsError(
            f"unknown argument(s): {', '.join(unknown)}; "
            f"allowed: {', '.join(sorted(properties)) or '(none)'}"
        )

    missing = [key for key in required if key not in arguments or arguments[key] is None]
    if missing:
        raise InvalidArgumentsError(f"missing required argument(s): {', '.join(sorted(missing))}")

    validated: dict[str, Any] = {}
    for key, spec in properties.items():
        if key not in arguments or arguments[key] is None:
            if "default" in spec:
                validated[key] = spec["default"]
            continue
        validated[key] = _validate_value(key, arguments[key], spec)

    if allow_extra:
        for key in unknown:
            validated[key] = arguments[key]
    return validated


def _validate_value(field: str, value: Any, spec: dict[str, Any]) -> Any:
    declared = spec.get("type")
    types = declared if isinstance(declared, list) else [declared] if declared else []
    if types:
        accepted: tuple[type, ...] = tuple(t for name in types for t in _TYPE_MAP.get(name, ()))
        # `bool` is a subclass of `int`; do not let it satisfy integer/number.
        if accepted and isinstance(value, bool) and "boolean" not in types:
            raise InvalidArgumentsError(
                f"argument '{field}' must be of type {'/'.join(types)}, got boolean"
            )
        if accepted and not isinstance(value, accepted):
            raise InvalidArgumentsError(
                f"argument '{field}' must be of type {'/'.join(types)}, got {type(value).__name__}"
            )

    if "enum" in spec and value not in spec["enum"]:
        raise InvalidArgumentsError(
            f"argument '{field}' must be one of {spec['enum']}, got {value!r}"
        )

    if isinstance(value, str):
        if "minLength" in spec and len(value) < spec["minLength"]:
            raise InvalidArgumentsError(
                f"argument '{field}' must be at least {spec['minLength']} characters"
            )
        if "maxLength" in spec and len(value) > spec["maxLength"]:
            raise InvalidArgumentsError(
                f"argument '{field}' must be at most {spec['maxLength']} characters"
            )

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in spec and value < spec["minimum"]:
            raise InvalidArgumentsError(f"argument '{field}' must be >= {spec['minimum']}")
        if "maximum" in spec and value > spec["maximum"]:
            raise InvalidArgumentsError(f"argument '{field}' must be <= {spec['maximum']}")

    if isinstance(value, list):
        if "maxItems" in spec and len(value) > spec["maxItems"]:
            raise InvalidArgumentsError(
                f"argument '{field}' must have at most {spec['maxItems']} items"
            )
        if "minItems" in spec and len(value) < spec["minItems"]:
            raise InvalidArgumentsError(
                f"argument '{field}' must have at least {spec['minItems']} items"
            )
        item_spec = spec.get("items")
        if isinstance(item_spec, dict):
            return [_validate_value(f"{field}[{i}]", v, item_spec) for i, v in enumerate(value)]

    return value


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
class ToolRegistry:
    """Holds tools and executes calls under policy."""

    def __init__(
        self,
        *,
        permissions: PermissionChecker,
        approver: Approver | None = None,
        events: EventBus | None = None,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        self.permissions = permissions
        self.approver: Approver = approver or AutoDenyApprover()
        self.events = events or EventBus()
        #: Tools the human approved for the whole run.
        self._run_approved: set[str] = set()

    # -- registration -------------------------------------------------------
    def register(self, tool: Tool) -> None:
        """Add a tool. Raises on a duplicate or malformed name."""
        name = tool.name
        if not NAME_PATTERN.match(name or ""):
            raise AgentError(
                f"invalid tool name {name!r}: must match {NAME_PATTERN.pattern}",
                category=ErrorCategory.CONFIGURATION,
            )
        if name in self._tools:
            raise AgentError(
                f"a tool named {name!r} is already registered",
                category=ErrorCategory.CONFIGURATION,
            )
        self._tools[name] = tool

    def register_all(self, tools: Iterable[Tool]) -> None:
        for tool in tools:
            self.register(tool)

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    # -- lookup -------------------------------------------------------------
    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise UnknownToolError(
                f"no tool named {name!r}; available: {', '.join(sorted(self._tools))}"
            ) from None

    def names(self) -> list[str]:
        return sorted(self._tools)

    def definitions(self, only: Sequence[str] | None = None) -> list[ToolDefinition]:
        """Normalized definitions, optionally restricted to a subset."""
        selected = (
            self._tools if only is None else {n: self._tools[n] for n in only if n in self._tools}
        )
        return [tool.definition() for _, tool in sorted(selected.items())]

    # -- execution ----------------------------------------------------------
    async def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        """Validate, authorise and run one tool call.

        This method never raises for an expected failure: every problem comes back
        as a structured, unsuccessful :class:`ToolResult`, because the model must
        be able to see and react to it.
        """
        self.events.emit_event(
            EventType.TOOL_REQUESTED,
            task_id=context.task_id,
            step=context.step,
            message=f"tool requested: {call.name}",
            data={"tool": call.name, "arguments": call.arguments},
        )

        # 1. The tool must exist.
        try:
            tool = self.get(call.name)
        except UnknownToolError as exc:
            return self._failure(call, exc.category, exc.message)

        definition = tool.definition()

        # 2. Arguments must validate against the declared schema.
        try:
            arguments = validate_arguments(definition.json_schema(), call.arguments)
        except InvalidArgumentsError as exc:
            return self._failure(call, exc.category, exc.message)

        # 3. Policy must permit the call.
        decision = self.permissions.check_tool(definition, arguments)
        if not decision.allowed:
            self.events.emit_event(
                EventType.APPROVAL_RESULT,
                task_id=context.task_id,
                step=context.step,
                message=f"permission denied: {decision.reason}",
                data={"tool": call.name, "allowed": False},
            )
            return self._failure(
                call,
                decision.error_category or ErrorCategory.PERMISSION_DENIED,
                decision.reason,
            )

        # 4. A human must approve, unless already approved for this run.
        approved = True
        if decision.requires_approval and call.name not in self._run_approved:
            approved, cancelled, note = self._request_approval(call, definition, arguments, context)
            if cancelled:
                return self._failure(
                    call, ErrorCategory.CANCELLED, "the run was cancelled at the approval prompt"
                )
            if not approved:
                return self._failure(
                    call,
                    ErrorCategory.APPROVAL_DENIED,
                    f"a human denied this action{f': {note}' if note else ''}",
                )

        # 5. Execute under a timeout.
        return await self._run_tool(tool, call, arguments, context)

    def _request_approval(
        self,
        call: ToolCall,
        definition: ToolDefinition,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> tuple[bool, bool, str]:
        """Ask the approver. Returns `(approved, cancelled, note)`."""
        summary, target = describe_request(definition, arguments)
        request = ApprovalRequest(
            tool=definition,
            arguments=arguments,
            task_id=context.task_id,
            step=context.step,
            summary=summary,
            target=target,
        )
        self.events.emit_event(
            EventType.APPROVAL_REQUESTED,
            task_id=context.task_id,
            step=context.step,
            message=f"approval requested for {call.name}",
            data={"tool": call.name, "summary": summary, "target": target},
        )
        response = self.approver.request(request)
        if response.decision is ApprovalDecision.APPROVE_FOR_RUN:
            self._run_approved.add(call.name)
        self.events.emit_event(
            EventType.APPROVAL_RESULT,
            task_id=context.task_id,
            step=context.step,
            message=f"approval {response.decision.value} for {call.name}",
            data={"tool": call.name, "decision": response.decision.value},
        )
        return (
            response.approved,
            response.decision is ApprovalDecision.CANCEL,
            response.note,
        )

    async def _run_tool(
        self, tool: Tool, call: ToolCall, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        self.events.emit_event(
            EventType.TOOL_STARTED,
            task_id=context.task_id,
            step=context.step,
            message=f"running {call.name}",
            data={"tool": call.name},
        )
        timer = Timer()
        try:
            with timer:
                output = await asyncio.wait_for(
                    tool.run(arguments, context), timeout=context.timeout
                )
        except TimeoutError:
            result = ToolResult(
                call_id=call.id,
                tool_name=call.name,
                ok=False,
                error_category=ErrorCategory.TOOL_TIMEOUT,
                error_message=f"{call.name} exceeded the {context.timeout:g}s tool timeout",
                duration_ms=timer.elapsed_ms,
                timed_out=True,
            )
            self._emit_result(result, context)
            return result
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            category, message = error_from_exception(exc)
            result = ToolResult(
                call_id=call.id,
                tool_name=call.name,
                ok=False,
                error_category=category,
                error_message=str(context.redactor.redact_text(message)),
                duration_ms=timer.elapsed_ms,
            )
            self._emit_result(result, context)
            return result

        safe_output, truncated = self._sanitize(output, context)
        result = ToolResult(
            call_id=call.id,
            tool_name=call.name,
            ok=True,
            output=safe_output,
            duration_ms=timer.elapsed_ms,
            truncated=truncated,
        )
        self._emit_result(result, context)
        return result

    def _sanitize(
        self, output: dict[str, Any], context: ToolContext
    ) -> tuple[dict[str, Any], bool]:
        """Redact secrets, then clamp long string fields."""
        redacted = context.redactor.redact(output)
        if not isinstance(redacted, dict):
            return {"value": redacted}, False
        truncated_any = False
        clamped: dict[str, Any] = {}
        for key, value in redacted.items():
            if isinstance(value, str):
                text, was_truncated = truncate_output(value, context.max_output_chars)
                truncated_any = truncated_any or was_truncated
                clamped[key] = text
            else:
                clamped[key] = value
        return clamped, truncated_any

    def _emit_result(self, result: ToolResult, context: ToolContext) -> None:
        self.events.emit_event(
            EventType.TOOL_RESULT,
            task_id=context.task_id,
            step=context.step,
            message=(
                f"{result.tool_name} {'succeeded' if result.ok else 'failed'} "
                f"in {result.duration_ms}ms"
            ),
            data={
                "tool": result.tool_name,
                "ok": result.ok,
                "duration_ms": result.duration_ms,
                "error_category": result.error_category.value if result.error_category else None,
                "truncated": result.truncated,
            },
        )

    def _failure(self, call: ToolCall, category: ErrorCategory, message: str) -> ToolResult:
        result = ToolResult(
            call_id=call.id,
            tool_name=call.name,
            ok=False,
            error_category=category,
            error_message=message,
        )
        self.events.emit_event(
            EventType.TOOL_RESULT,
            message=f"{call.name} rejected: {category.value}",
            data={"tool": call.name, "ok": False, "error_category": category.value},
        )
        return result
