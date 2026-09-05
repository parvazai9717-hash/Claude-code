"""Lifecycle events and structured JSONL logging.

Events are the only progress channel the CLI, the log file and the SQLite event
table share. Every event payload passes through :mod:`agent.security.redaction`
before it is written or displayed, so an event can never leak a credential.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class EventType(StrEnum):
    """Every observable moment in a run."""

    TASK_CREATED = "task_created"
    TASK_STARTED = "task_started"
    TASK_STATUS_CHANGED = "task_status_changed"
    TASK_FINISHED = "task_finished"

    PHASE_ENTERED = "phase_entered"
    STEP_STARTED = "step_started"
    STEP_FINISHED = "step_finished"

    PLAN_UPDATED = "plan_updated"
    OBSERVATION = "observation"

    PROVIDER_REQUEST = "provider_request"
    PROVIDER_RESPONSE = "provider_response"
    PROVIDER_ERROR = "provider_error"

    TOOL_REQUESTED = "tool_requested"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_RESULT = "approval_result"
    TOOL_STARTED = "tool_started"
    TOOL_RESULT = "tool_result"

    VERIFICATION_STARTED = "verification_started"
    VERIFICATION_RESULT = "verification_result"

    REPLAN = "replan"
    LIMIT_REACHED = "limit_reached"
    ERROR = "error"
    MESSAGE = "message"
    FINAL_ANSWER = "final_answer"


class Event(BaseModel):
    """One structured, already-safe record."""

    type: EventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    task_id: str | None = None
    session_id: str | None = None
    step: int | None = None
    phase: str | None = None
    provider: str | None = None
    model: str | None = None
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)

    def to_json(self) -> str:
        payload = self.model_dump(mode="json", exclude_none=True)
        return json.dumps(payload, ensure_ascii=False, default=str)


EventHandler = Callable[[Event], None]


class EventBus:
    """Fan-out for lifecycle events.

    The bus applies redaction once, centrally, so every subscriber — terminal,
    JSONL log, SQLite — receives the same already-safe payload. A failing
    subscriber is isolated: it can never abort a run.
    """

    def __init__(self, redactor: Callable[[Any], Any] | None = None) -> None:
        self._handlers: list[EventHandler] = []
        self._redactor = redactor
        self._lock = threading.Lock()
        self._history: list[Event] = []
        self._keep_history = 500

    def subscribe(self, handler: EventHandler) -> Callable[[], None]:
        """Register a handler; returns a callable that unsubscribes it."""
        with self._lock:
            self._handlers.append(handler)

        def _unsubscribe() -> None:
            with self._lock:
                if handler in self._handlers:
                    self._handlers.remove(handler)

        return _unsubscribe

    def emit(self, event: Event) -> Event:
        """Redact, record and dispatch an event. Returns the redacted event."""
        if self._redactor is not None:
            safe_data = self._redactor(event.data)
            safe_message = self._redactor(event.message)
            event = event.model_copy(
                update={
                    "data": safe_data if isinstance(safe_data, dict) else {},
                    "message": safe_message if isinstance(safe_message, str) else "",
                }
            )
        with self._lock:
            self._history.append(event)
            if len(self._history) > self._keep_history:
                del self._history[: -self._keep_history]
            handlers = list(self._handlers)
        for handler in handlers:
            try:
                handler(event)
            except Exception:
                logging.getLogger("agent.events").debug("event handler failed", exc_info=True)
        return event

    def emit_event(self, type: EventType, **kwargs: Any) -> Event:
        """Convenience constructor + emit."""
        return self.emit(Event(type=type, **kwargs))

    @property
    def history(self) -> list[Event]:
        with self._lock:
            return list(self._history)


class JsonlEventWriter:
    """Append events to a JSONL file, one compact object per line."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def __call__(self, event: Event) -> None:
        line = event.to_json()
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def configure_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure the standard library logger used for diagnostics.

    Lifecycle information belongs on the :class:`EventBus`; this logger carries
    diagnostics only. Neither channel ever receives an unredacted secret.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
        handlers=handlers,
        force=True,
    )


def summarize_events(events: Iterable[Event]) -> dict[str, int]:
    """Count events by type — used by `task events` and by tests."""
    counts: dict[str, int] = {}
    for event in events:
        counts[event.type.value] = counts.get(event.type.value, 0) + 1
    return counts
