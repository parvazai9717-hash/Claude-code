"""`get_current_time` — a read-only clock.

The agent has no reliable sense of time from the model alone, so this tool exists
to make "what time is it" a fact rather than a guess.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ..messages import RiskCategory, RiskLevel
from .base import Tool, ToolContext


class GetCurrentTimeTool(Tool):
    name = "get_current_time"
    description = (
        "Return the current local and UTC time. Use this instead of guessing the date or time."
    )
    parameters = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    risk = RiskLevel.READ_ONLY
    risk_category = RiskCategory.READ
    read_only = True
    requires_approval = False

    async def run(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        now_local = datetime.now().astimezone()
        now_utc = datetime.now(UTC)
        return {
            "local": now_local.isoformat(timespec="seconds"),
            "utc": now_utc.isoformat(timespec="seconds"),
            "timezone": str(now_local.tzinfo),
            "unix": int(now_utc.timestamp()),
        }
