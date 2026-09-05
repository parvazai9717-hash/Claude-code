"""Conversation summarisation and context management.

When a conversation grows past the configured message limit, older turns are
compacted into a summary so the run can continue without silently dropping
context. Summarisation is extractive and deterministic — it never asks the model
to invent a summary, so it cannot introduce a claim that was never made.

A retrieval interface is defined here so semantic search can be added later
without changing any caller.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol

from ..messages import Message
from .database import Database

#: Characters kept from each summarised message.
_EXCERPT_CHARS = 160


class Retriever(Protocol):
    """Future extension point for semantic retrieval.

    The first release ships only :class:`RecentMessageRetriever`. A vector-backed
    implementation can be dropped in without touching the runtime.
    """

    def retrieve(
        self, query: str, *, session_id: str, limit: int
    ) -> list[str]: ...  # pragma: no cover - protocol definition


class SummaryStore:
    """Stores and retrieves conversation summaries."""

    def __init__(self, database: Database) -> None:
        self.db = database

    def add(self, session_id: str, summary: str, message_count: int) -> int:
        cursor = self.db.execute(
            "INSERT INTO summaries (session_id, summary, message_count, created_at)"
            " VALUES (?, ?, ?, ?)",
            (session_id, summary, message_count, datetime.now(UTC).isoformat()),
        )
        return int(cursor.lastrowid or 0)

    def latest(self, session_id: str) -> dict[str, Any] | None:
        row = self.db.query_one(
            "SELECT * FROM summaries WHERE session_id = ? ORDER BY id DESC LIMIT 1", (session_id,)
        )
        return dict(row) if row else None

    def list_summaries(self, session_id: str, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM summaries WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        )
        return [dict(row) for row in rows]


def summarize_messages(messages: list[Message]) -> str:
    """Build an extractive summary of a run of messages.

    Only what was actually said or done is recorded: user goals, assistant
    conclusions, and the tools that ran with their outcomes.
    """
    if not messages:
        return ""
    goals: list[str] = []
    conclusions: list[str] = []
    tool_names: dict[str, int] = {}
    for message in messages:
        if message.role == "user" and message.content.strip():
            goals.append(message.content.strip()[:_EXCERPT_CHARS])
        elif message.role == "assistant":
            for call in message.tool_calls:
                tool_names[call.name] = tool_names.get(call.name, 0) + 1
            if message.content.strip():
                conclusions.append(message.content.strip()[:_EXCERPT_CHARS])
    parts: list[str] = [f"Summary of {len(messages)} earlier messages."]
    if goals:
        parts.append("User asked about: " + "; ".join(goals[-5:]))
    if tool_names:
        rendered = ", ".join(f"{name} x{count}" for name, count in sorted(tool_names.items()))
        parts.append(f"Tools used: {rendered}")
    if conclusions:
        parts.append("Assistant concluded: " + "; ".join(conclusions[-3:]))
    return "\n".join(parts)


def compact_conversation(
    messages: list[Message], max_messages: int
) -> tuple[list[Message], str | None]:
    """Trim a conversation to `max_messages`, folding the rest into a summary.

    The system message is always kept first, and the trim point never splits an
    assistant tool-call message from the `tool` messages that answer it — a
    dangling tool result is rejected by several providers.

    Returns:
        `(messages, summary_text)`; `summary_text` is None when nothing was cut.
    """
    if len(messages) <= max_messages:
        return messages, None

    system = [m for m in messages if m.role == "system"]
    body = [m for m in messages if m.role != "system"]
    keep = max(1, max_messages - len(system) - 1)  # -1 leaves room for the summary
    cut_index = len(body) - keep

    # Move the cut forward until it does not start on an orphaned tool result.
    while cut_index < len(body) and body[cut_index].role == "tool":
        cut_index += 1

    older, recent = body[:cut_index], body[cut_index:]
    if not older:
        return messages, None

    summary_text = summarize_messages(older)
    summary_message = Message(
        role="system",
        content=f"[Earlier conversation, compacted]\n{summary_text}",
        metadata={"compacted": True, "replaced_messages": len(older)},
    )
    return [*system, summary_message, *recent], summary_text


class RecentMessageRetriever:
    """The default retriever: the most recent messages, newest last.

    Deliberately simple. It exists so the runtime already depends on the
    :class:`Retriever` interface rather than on a concrete implementation.
    """

    def __init__(self, database: Database) -> None:
        from .conversations import ConversationStore

        self.conversations = ConversationStore(database)

    def retrieve(self, query: str, *, session_id: str, limit: int = 10) -> list[str]:
        messages = self.conversations.get_messages(session_id, limit=limit)
        return [m.content for m in messages if m.content.strip()]
