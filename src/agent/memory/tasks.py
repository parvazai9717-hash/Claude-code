"""Task persistence and recovery.

A task's full :class:`~agent.task_state.TaskState` is stored as JSON alongside
indexed columns, so a task survives a process restart and `task resume` can
revalidate it before continuing.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from ..errors import ErrorCategory, StorageError
from ..events import Event, EventType
from ..messages import ToolResult
from ..task_state import ActionRecord, TaskState, TaskStatus, VerificationRecord
from .database import Database


def _now() -> str:
    return datetime.now(UTC).isoformat()


class TaskStore:
    """Persists task state, tool calls, verifications and events."""

    def __init__(self, database: Database) -> None:
        self.db = database

    # -- task state ---------------------------------------------------------
    def save(self, state: TaskState) -> None:
        """Insert or update a task. The full state is stored as JSON."""
        payload = state.model_dump_json()
        self.db.execute(
            "INSERT INTO tasks"
            " (id, session_id, goal, status, provider, model, workspace, state_json,"
            "  created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(id) DO UPDATE SET"
            "  status = excluded.status, provider = excluded.provider, model = excluded.model,"
            "  workspace = excluded.workspace, state_json = excluded.state_json,"
            "  updated_at = excluded.updated_at",
            (
                state.id,
                state.session_id,
                state.goal,
                state.status.value,
                state.provider,
                state.model,
                state.workspace,
                payload,
                state.created_at.isoformat(),
                _now(),
            ),
        )

    def load(self, task_id: str) -> TaskState | None:
        row = self.db.query_one("SELECT state_json FROM tasks WHERE id = ?", (task_id,))
        if row is None:
            return None
        try:
            return TaskState.model_validate_json(row["state_json"])
        except Exception as exc:
            raise StorageError(f"task {task_id} could not be loaded: {exc}") from exc

    def find(self, prefix: str) -> TaskState | None:
        """Load a task by full id or unambiguous prefix.

        A listing can truncate an id to fit the terminal, so the trailing ellipsis
        a user may copy along with it is stripped before matching.
        """
        exact = self.load(prefix)
        if exact is not None:
            return exact
        cleaned = prefix.rstrip(".\u2026 ")
        rows = self.db.query("SELECT id FROM tasks WHERE id LIKE ? LIMIT 2", (f"{cleaned}%",))
        if len(rows) == 1:
            return self.load(rows[0]["id"])
        return None

    def list_tasks(
        self, *, status: TaskStatus | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        sql = "SELECT id, goal, status, provider, model, created_at, updated_at FROM tasks"
        parameters: tuple[Any, ...] = ()
        if status is not None:
            sql += " WHERE status = ?"
            parameters = (status.value,)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        parameters += (limit,)
        return [dict(row) for row in self.db.query(sql, parameters)]

    def delete(self, task_id: str) -> bool:
        with self.db.transaction() as connection:
            connection.execute("DELETE FROM events WHERE task_id = ?", (task_id,))
            connection.execute("DELETE FROM tool_calls WHERE task_id = ?", (task_id,))
            connection.execute("DELETE FROM verifications WHERE task_id = ?", (task_id,))
            cursor = connection.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            return cursor.rowcount > 0

    def request_pause(self, task_id: str) -> bool:
        """Set the pause flag on a stored task; the runtime honours it at a safe point."""
        return self._set_flag(task_id, "pause_requested", True)

    def request_cancel(self, task_id: str, reason: str = "") -> bool:
        state = self.load(task_id)
        if state is None:
            return False
        state.cancel_requested = True
        state.cancel_reason = reason
        self.save(state)
        return True

    def _set_flag(self, task_id: str, field: str, value: bool) -> bool:
        state = self.load(task_id)
        if state is None:
            return False
        setattr(state, field, value)
        self.save(state)
        return True

    def recover_interrupted(self) -> list[str]:
        """Mark tasks left `running` by a crashed process as paused.

        A process that dies mid-run leaves a task in `running` with nothing
        executing it. Recovery moves those to `paused` so `task resume` can pick
        them up rather than the CLI reporting a task that is not actually alive.
        """
        rows = self.db.query(
            "SELECT id FROM tasks WHERE status IN (?, ?)",
            (TaskStatus.RUNNING.value, TaskStatus.WAITING_FOR_APPROVAL.value),
        )
        recovered: list[str] = []
        for row in rows:
            state = self.load(row["id"])
            if state is None:
                continue
            state.transition_to(TaskStatus.PAUSED)
            state.record_failure(
                ErrorCategory.UNKNOWN,
                "the previous process exited while this task was running",
            )
            self.save(state)
            recovered.append(state.id)
        return recovered

    # -- tool calls and verifications --------------------------------------
    def record_tool_result(
        self,
        result: ToolResult,
        *,
        task_id: str | None,
        session_id: str | None,
        arguments: dict[str, Any],
        approved: bool | None = None,
    ) -> None:
        self.db.execute(
            "INSERT INTO tool_calls"
            " (task_id, session_id, call_id, tool_name, arguments, ok, output,"
            "  error_category, error_message, duration_ms, approved, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                task_id,
                session_id,
                result.call_id,
                result.tool_name,
                json.dumps(arguments, default=str),
                int(result.ok),
                json.dumps(result.output, default=str),
                result.error_category.value if result.error_category else None,
                result.error_message,
                result.duration_ms,
                None if approved is None else int(approved),
                _now(),
            ),
        )

    def record_verification(self, record: VerificationRecord, task_id: str | None) -> None:
        self.db.execute(
            "INSERT INTO verifications"
            " (task_id, action_id, method, verified, evidence, detail, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                task_id,
                record.action_id,
                record.method,
                int(record.verified),
                record.evidence,
                record.detail,
                record.created_at.isoformat(),
            ),
        )

    def tool_calls_for(self, task_id: str, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM tool_calls WHERE task_id = ? ORDER BY id LIMIT ?", (task_id, limit)
        )
        return [dict(row) for row in rows]

    # -- events -------------------------------------------------------------
    def record_event(self, event: Event) -> None:
        self.db.execute(
            "INSERT INTO events (task_id, session_id, type, step, phase, message, data, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event.task_id,
                event.session_id,
                event.type.value,
                event.step,
                event.phase,
                event.message,
                json.dumps(event.data, default=str),
                event.timestamp.isoformat(),
            ),
        )

    def events_for(self, task_id: str, limit: int = 200) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM events WHERE task_id = ? ORDER BY id LIMIT ?", (task_id, limit)
        )
        results: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            try:
                item["data"] = json.loads(item.get("data") or "{}")
            except json.JSONDecodeError:
                item["data"] = {}
            results.append(item)
        return results

    def event_writer(self) -> Any:
        """An :class:`~agent.events.EventBus` subscriber that persists events."""

        def _write(event: Event) -> None:
            # Only lifecycle-significant events are persisted; per-token noise is not.
            if event.type in {EventType.PROVIDER_REQUEST}:
                return
            self.record_event(event)

        return _write

    # -- action bookkeeping -------------------------------------------------
    @staticmethod
    def action_from_result(
        result: ToolResult, step: int, arguments: dict[str, Any], approved: bool | None
    ) -> ActionRecord:
        return ActionRecord(
            step=step,
            call_id=result.call_id,
            tool_name=result.tool_name,
            arguments=arguments,
            ok=result.ok,
            approved=approved,
            error_category=result.error_category,
            summary=(result.error_message or "")[:300] if not result.ok else "succeeded",
            duration_ms=result.duration_ms,
        )
