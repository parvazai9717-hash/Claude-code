"""Runtime limit tracking.

Every bound in :class:`agent.config.Limits` is enforced here rather than being
scattered through the runtime, so a new limit only has to be checked in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Limits
from ..errors import ErrorCategory, LimitExceededError


@dataclass
class LimitCheck:
    """The outcome of a single limit test."""

    ok: bool
    limit_name: str = ""
    limit: int | float = 0
    used: int | float = 0
    message: str = ""


@dataclass
class LimitTracker:
    """Counts consumption against the configured limits for one run."""

    limits: Limits
    steps_used: int = 0
    total_tool_calls: int = 0
    calls_this_step: int = 0
    _step_calls: dict[int, int] = field(default_factory=dict)

    # -- steps --------------------------------------------------------------
    def start_step(self, step: int) -> LimitCheck:
        """Register the start of a step, refusing to exceed `max_steps`."""
        if self.steps_used >= self.limits.max_steps:
            return LimitCheck(
                ok=False,
                limit_name="max_steps",
                limit=self.limits.max_steps,
                used=self.steps_used,
                message=f"reached the maximum of {self.limits.max_steps} agent steps",
            )
        self.steps_used += 1
        self.calls_this_step = 0
        self._step_calls[step] = 0
        return LimitCheck(ok=True)

    # -- tool calls ---------------------------------------------------------
    def check_tool_call(self) -> LimitCheck:
        """Test — without consuming — whether another tool call is permitted."""
        if self.total_tool_calls >= self.limits.max_total_tool_calls:
            return LimitCheck(
                ok=False,
                limit_name="max_total_tool_calls",
                limit=self.limits.max_total_tool_calls,
                used=self.total_tool_calls,
                message=(
                    f"reached the maximum of {self.limits.max_total_tool_calls} total tool calls"
                ),
            )
        if self.calls_this_step >= self.limits.max_tool_calls_per_step:
            return LimitCheck(
                ok=False,
                limit_name="max_tool_calls_per_step",
                limit=self.limits.max_tool_calls_per_step,
                used=self.calls_this_step,
                message=(
                    f"reached the maximum of {self.limits.max_tool_calls_per_step} "
                    "tool calls in a single step"
                ),
            )
        return LimitCheck(ok=True)

    def consume_tool_call(self) -> None:
        """Record one executed tool call."""
        self.total_tool_calls += 1
        self.calls_this_step += 1

    def check_retry(self, attempts: int) -> LimitCheck:
        """Whether an action that has already failed `attempts` times may retry."""
        if attempts > self.limits.max_retries_per_action:
            return LimitCheck(
                ok=False,
                limit_name="max_retries_per_action",
                limit=self.limits.max_retries_per_action,
                used=attempts,
                message=(
                    f"this action already failed {attempts} times "
                    f"(limit {self.limits.max_retries_per_action})"
                ),
            )
        return LimitCheck(ok=True)

    def check_replan(self, replans: int) -> LimitCheck:
        if replans >= self.limits.max_replans:
            return LimitCheck(
                ok=False,
                limit_name="max_replans",
                limit=self.limits.max_replans,
                used=replans,
                message=f"reached the maximum of {self.limits.max_replans} replans",
            )
        return LimitCheck(ok=True)

    def remaining_steps(self) -> int:
        return max(0, self.limits.max_steps - self.steps_used)


def truncate_output(text: str, max_chars: int) -> tuple[str, bool]:
    """Clamp `text` to `max_chars`, reporting whether truncation happened.

    The marker keeps the model honest about the fact that it is seeing a slice.
    """
    if len(text) <= max_chars:
        return text, False
    kept = text[:max_chars]
    omitted = len(text) - max_chars
    return f"{kept}\n... [truncated: {omitted} more characters]", True


def enforce_file_size(size: int, max_bytes: int, path_display: str) -> None:
    """Raise when a file exceeds the configured read limit."""
    if size > max_bytes:
        raise LimitExceededError(
            f"{path_display} is {size} bytes, above the {max_bytes} byte limit",
            category=ErrorCategory.LIMIT_EXCEEDED,
            details={"size": size, "limit": max_bytes},
        )
