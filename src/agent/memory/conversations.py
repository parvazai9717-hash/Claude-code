"""Conversation history: persisted sessions and their messages.

The five memory layers are kept deliberately separate:

1. **Working memory** — the in-memory `list[Message]` the runtime holds.
2. **Conversation history** — this module: durable sessions and messages.
3. **Durable memory** — :mod:`agent.memory.facts`: human-approved facts only.
4. **Task state** — :mod:`agent.memory.tasks`.
5. **Summaries** — :mod:`agent.memory.summaries`, for long conversations.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from ..messages import Message, ToolCall, new_id
from .database import Database


def _now() -> str:
    return datetime.now(UTC).isoformat()


class ConversationStore:
    """Reads and writes sessions and messages."""

    def __init__(self, database: Database) -> None:
        self.db = database

    # -- sessions -----------------------------------------------------------
    def create_session(
        self, *, session_id: str | None = None, title: str = "", provider: str = "", model: str = ""
    ) -> str:
        identifier = session_id or new_id("sess")
        timestamp = _now()
        self.db.execute(
            "INSERT OR IGNORE INTO sessions (id, title, provider, model, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (identifier, title, provider, model, timestamp, timestamp),
        )
        return identifier

    def touch_session(self, session_id: str) -> None:
        self.db.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (_now(), session_id))

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT s.*, (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id)"
            " AS message_count FROM sessions s ORDER BY s.updated_at DESC LIMIT ?",
            (limit,),
        )
        return [dict(row) for row in rows]

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        row = self.db.query_one("SELECT * FROM sessions WHERE id = ?", (session_id,))
        return dict(row) if row else None

    def find_session(self, prefix: str) -> dict[str, Any] | None:
        """Resolve a session by full id or unambiguous prefix.

        A listing can truncate an id to fit the terminal, so a user copying one
        off their own screen may hand back only its first characters. Matching a
        unique prefix — and refusing an ambiguous one — is what makes that work.
        """
        exact = self.get_session(prefix)
        if exact is not None:
            return exact
        cleaned = prefix.rstrip(".\u2026 ")
        rows = self.db.query("SELECT id FROM sessions WHERE id LIKE ? LIMIT 2", (f"{cleaned}%",))
        if len(rows) == 1:
            return self.get_session(rows[0]["id"])
        return None

    def delete_session(self, session_id: str) -> int:
        """Delete a session and everything attached to it. Returns rows removed."""
        with self.db.transaction() as connection:
            cursor = connection.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
            )
            message_count = int(cursor.fetchone()[0])
            connection.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            connection.execute("DELETE FROM summaries WHERE session_id = ?", (session_id,))
            connection.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        return message_count

    # -- messages -----------------------------------------------------------
    def add_message(self, session_id: str, message: Message, task_id: str | None = None) -> None:
        """Append one message. The caller is responsible for redaction."""
        self.create_session(session_id=session_id)
        self.db.execute(
            "INSERT INTO messages"
            " (session_id, task_id, role, content, tool_calls, tool_call_id, name, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                task_id,
                message.role,
                message.content,
                json.dumps([call.model_dump(mode="json") for call in message.tool_calls]),
                message.tool_call_id,
                message.name,
                message.created_at.isoformat(),
            ),
        )
        self.touch_session(session_id)

    def add_messages(
        self, session_id: str, messages: list[Message], task_id: str | None = None
    ) -> None:
        for message in messages:
            self.add_message(session_id, message, task_id)

    def get_messages(self, session_id: str, limit: int | None = None) -> list[Message]:
        sql = "SELECT * FROM messages WHERE session_id = ? ORDER BY id"
        parameters: tuple[Any, ...] = (session_id,)
        if limit is not None:
            sql += " DESC LIMIT ?"
            parameters = (session_id, limit)
        rows = self.db.query(sql, parameters)
        if limit is not None:
            rows = list(reversed(rows))
        return [self._to_message(row) for row in rows]

    def count_messages(self, session_id: str) -> int:
        row = self.db.query_one(
            "SELECT COUNT(*) AS n FROM messages WHERE session_id = ?", (session_id,)
        )
        return int(row["n"]) if row else 0

    @staticmethod
    def _to_message(row: Any) -> Message:
        raw_calls = json.loads(row["tool_calls"] or "[]")
        return Message(
            role=row["role"],
            content=row["content"] or "",
            tool_calls=[ToolCall(**call) for call in raw_calls],
            tool_call_id=row["tool_call_id"],
            name=row["name"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )
