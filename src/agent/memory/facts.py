"""Durable memory: facts a human explicitly approved.

Nothing reaches this table without approval. The model can *propose* a fact via
the `remember_fact` tool; the runtime asks a human; only an approved proposal is
stored, and only approved rows are ever loaded back into a prompt.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ..errors import LimitExceededError
from .database import Database


class FactStore:
    """Reads and writes approved durable facts."""

    def __init__(self, database: Database, max_facts: int = 500) -> None:
        self.db = database
        self.max_facts = max_facts

    def add(
        self, fact: str, *, category: str = "general", source: str = "", approved: bool = False
    ) -> int:
        """Store a fact. `approved` must be true for it to be recalled later."""
        text = fact.strip()
        if not text:
            raise ValueError("a fact cannot be empty")
        if self.count() >= self.max_facts:
            raise LimitExceededError(
                f"durable memory already holds {self.max_facts} facts; "
                "remove some with `local-agent memory remove`"
            )
        cursor = self.db.execute(
            "INSERT OR REPLACE INTO facts (fact, category, source, approved, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (text, category, source, int(approved), datetime.now(UTC).isoformat()),
        )
        return int(cursor.lastrowid or 0)

    def list_facts(self, *, approved_only: bool = True, limit: int = 200) -> list[dict[str, Any]]:
        sql = "SELECT * FROM facts"
        if approved_only:
            sql += " WHERE approved = 1"
        sql += " ORDER BY id DESC LIMIT ?"
        return [dict(row) for row in self.db.query(sql, (limit,))]

    def get(self, fact_id: int) -> dict[str, Any] | None:
        row = self.db.query_one("SELECT * FROM facts WHERE id = ?", (fact_id,))
        return dict(row) if row else None

    def remove(self, fact_id: int) -> bool:
        cursor = self.db.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
        return cursor.rowcount > 0

    def clear(self) -> int:
        cursor = self.db.execute("DELETE FROM facts")
        return cursor.rowcount

    def count(self, *, approved_only: bool = False) -> int:
        sql = "SELECT COUNT(*) AS n FROM facts"
        if approved_only:
            sql += " WHERE approved = 1"
        row = self.db.query_one(sql)
        return int(row["n"]) if row else 0

    def prompt_block(self, limit: int = 25) -> str:
        """Render approved facts for inclusion in the system prompt."""
        facts = self.list_facts(approved_only=True, limit=limit)
        if not facts:
            return ""
        lines = [f"- {row['fact']}" for row in facts]
        return "Facts the user previously approved for long-term memory:\n" + "\n".join(lines)
