"""`remember_fact` — propose a durable fact, subject to human approval.

The tool writes to SQLite only after the registry's approval step has already
succeeded, and it always stores the row with `approved = 1` for that reason: an
unapproved proposal never reaches this code at all.
"""

from __future__ import annotations

from typing import Any

from ..errors import ErrorCategory, ToolError
from ..messages import RiskCategory, RiskLevel
from .base import Tool, ToolContext


class RememberFactTool(Tool):
    name = "remember_fact"
    description = (
        "Propose a durable fact or user preference to store in long-term memory. "
        "Requires human approval. Use it only for stable, useful facts the user would want "
        "remembered across sessions — never for secrets, credentials or transient details."
    )
    parameters = {
        "type": "object",
        "properties": {
            "fact": {
                "type": "string",
                "description": "The fact to remember, in one clear sentence.",
                "minLength": 3,
                "maxLength": 500,
            },
            "category": {
                "type": "string",
                "description": "A short grouping label, e.g. 'preference' or 'project'.",
                "default": "general",
                "maxLength": 50,
            },
        },
        "required": ["fact"],
        "additionalProperties": False,
    }
    risk = RiskLevel.LOW
    risk_category = RiskCategory.MEMORY
    read_only = False
    requires_approval = True
    reversible = True

    async def run(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        store = context.store
        if store is None:
            raise ToolError(
                "durable memory is not available in this run",
                category=ErrorCategory.CAPABILITY_UNAVAILABLE,
            )
        fact: str = arguments["fact"].strip()
        category: str = arguments.get("category", "general")

        # Redaction runs on the way in as well as the way out: a fact that looks
        # like a credential must never be written to disk in the first place.
        safe_fact = context.redactor.redact_text(fact)
        if safe_fact != fact:
            raise ToolError(
                "that fact appears to contain a credential and will not be stored",
                category=ErrorCategory.SECRET_PROTECTED,
            )

        # Reaching this point means the registry's approval step already passed.
        fact_id = store.add(safe_fact, category=category, source="agent", approved=True)
        return {
            "id": fact_id,
            "fact": safe_fact,
            "category": category,
            "stored": True,
            "total_facts": store.count(approved_only=True),
        }
