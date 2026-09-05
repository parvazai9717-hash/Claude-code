"""SQLite connection management and schema.

The schema is created idempotently and versioned with SQLite's own
``user_version`` pragma, so upgrading is a matter of appending a migration rather
than shipping a migration framework.

Everything written here has already passed through redaction: the database is
treated as an output channel, not as a private store.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from ..errors import StorageError

SCHEMA_VERSION = 1

#: Applied in order; index i upgrades the database from version i to i+1.
MIGRATIONS: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS sessions (
        id           TEXT PRIMARY KEY,
        title        TEXT NOT NULL DEFAULT '',
        provider     TEXT NOT NULL DEFAULT '',
        model        TEXT NOT NULL DEFAULT '',
        created_at   TEXT NOT NULL,
        updated_at   TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS messages (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id   TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        task_id      TEXT,
        role         TEXT NOT NULL,
        content      TEXT NOT NULL DEFAULT '',
        tool_calls   TEXT NOT NULL DEFAULT '[]',
        tool_call_id TEXT,
        name         TEXT,
        created_at   TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);

    CREATE TABLE IF NOT EXISTS tasks (
        id           TEXT PRIMARY KEY,
        session_id   TEXT NOT NULL,
        goal         TEXT NOT NULL,
        status       TEXT NOT NULL,
        provider     TEXT NOT NULL DEFAULT '',
        model        TEXT NOT NULL DEFAULT '',
        workspace    TEXT NOT NULL DEFAULT '',
        state_json   TEXT NOT NULL,
        created_at   TEXT NOT NULL,
        updated_at   TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status, updated_at);

    CREATE TABLE IF NOT EXISTS tool_calls (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id        TEXT,
        session_id     TEXT,
        call_id        TEXT NOT NULL,
        tool_name      TEXT NOT NULL,
        arguments      TEXT NOT NULL DEFAULT '{}',
        ok             INTEGER NOT NULL DEFAULT 0,
        output         TEXT NOT NULL DEFAULT '{}',
        error_category TEXT,
        error_message  TEXT,
        duration_ms    INTEGER NOT NULL DEFAULT 0,
        approved       INTEGER,
        created_at     TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_tool_calls_task ON tool_calls(task_id, id);

    CREATE TABLE IF NOT EXISTS verifications (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id     TEXT,
        action_id   TEXT,
        method      TEXT NOT NULL,
        verified    INTEGER NOT NULL,
        evidence    TEXT NOT NULL DEFAULT '',
        detail      TEXT NOT NULL DEFAULT '',
        created_at  TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS facts (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        fact        TEXT NOT NULL,
        category    TEXT NOT NULL DEFAULT 'general',
        source      TEXT NOT NULL DEFAULT '',
        approved    INTEGER NOT NULL DEFAULT 0,
        created_at  TEXT NOT NULL,
        UNIQUE(fact)
    );

    CREATE TABLE IF NOT EXISTS events (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id     TEXT,
        session_id  TEXT,
        type        TEXT NOT NULL,
        step        INTEGER,
        phase       TEXT,
        message     TEXT NOT NULL DEFAULT '',
        data        TEXT NOT NULL DEFAULT '{}',
        created_at  TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_events_task ON events(task_id, id);

    CREATE TABLE IF NOT EXISTS summaries (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id   TEXT NOT NULL,
        summary      TEXT NOT NULL,
        message_count INTEGER NOT NULL DEFAULT 0,
        created_at   TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_summaries_session ON summaries(session_id, id);
    """,
]


class Database:
    """A thin, thread-safe wrapper around a SQLite connection.

    The agent is single-process and mostly single-threaded, but Typer commands and
    the asyncio runtime can touch the database from different threads, so access
    is serialised with a lock and `check_same_thread` is disabled deliberately.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).expanduser()
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        try:
            self._connection = sqlite3.connect(
                str(self.path), check_same_thread=False, isolation_level=None
            )
        except sqlite3.Error as exc:
            raise StorageError(f"cannot open the database at {self.path}: {exc}") from exc
        self._connection.row_factory = sqlite3.Row
        self._configure()
        self.migrate()

    def _configure(self) -> None:
        with self._lock:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA synchronous = NORMAL")

    # -- schema -------------------------------------------------------------
    @property
    def version(self) -> int:
        with self._lock:
            row = self._connection.execute("PRAGMA user_version").fetchone()
        return int(row[0]) if row else 0

    def migrate(self) -> None:
        """Apply any outstanding migrations. Safe to call repeatedly."""
        with self._lock:
            current = self.version
            for index in range(current, len(MIGRATIONS)):
                try:
                    self._connection.executescript(MIGRATIONS[index])
                    self._connection.execute(f"PRAGMA user_version = {index + 1}")
                except sqlite3.Error as exc:
                    raise StorageError(f"migration {index + 1} failed: {exc}") from exc

    # -- access -------------------------------------------------------------
    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run statements in a single transaction, rolling back on error."""
        with self._lock:
            try:
                self._connection.execute("BEGIN")
                yield self._connection
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
            else:
                self._connection.execute("COMMIT")

    def execute(self, sql: str, parameters: tuple | dict = ()) -> sqlite3.Cursor:
        with self._lock:
            try:
                return self._connection.execute(sql, parameters)
            except sqlite3.Error as exc:
                raise StorageError(f"database error: {exc}") from exc

    def query(self, sql: str, parameters: tuple | dict = ()) -> list[sqlite3.Row]:
        return list(self.execute(sql, parameters).fetchall())

    def query_one(self, sql: str, parameters: tuple | dict = ()) -> sqlite3.Row | None:
        rows = self.execute(sql, parameters).fetchmany(1)
        return rows[0] if rows else None

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    # -- maintenance --------------------------------------------------------
    def clear_all(self) -> dict[str, int]:
        """Delete every row from every table. Returns per-table deletion counts."""
        tables = [
            "events",
            "verifications",
            "tool_calls",
            "summaries",
            "messages",
            "tasks",
            "facts",
            "sessions",
        ]
        counts: dict[str, int] = {}
        with self.transaction() as connection:
            for table in tables:
                count = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                connection.execute(f"DELETE FROM {table}")
                counts[table] = int(count)
        self.execute("VACUUM")
        return counts

    def stats(self) -> dict[str, int]:
        """Row counts per table, for `doctor` and `clear-data` confirmation."""
        tables = [
            "sessions",
            "messages",
            "tasks",
            "tool_calls",
            "verifications",
            "facts",
            "events",
            "summaries",
        ]
        return {
            table: int(self.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in tables
        }

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
