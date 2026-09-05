"""SQLite persistence: schema, conversations, facts, tasks, summaries."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.errors import LimitExceededError, StorageError
from agent.memory.conversations import ConversationStore
from agent.memory.database import SCHEMA_VERSION, Database
from agent.memory.facts import FactStore
from agent.memory.summaries import SummaryStore, compact_conversation, summarize_messages
from agent.memory.tasks import TaskStore
from agent.messages import Message, ToolCall, ToolResult
from agent.task_state import TaskState, TaskStatus


# -- database ---------------------------------------------------------------
def test_schema_is_created_at_the_current_version(database: Database) -> None:
    assert database.version == SCHEMA_VERSION


def test_migration_is_idempotent(database: Database) -> None:
    database.migrate()
    database.migrate()
    assert database.version == SCHEMA_VERSION


def test_reopening_preserves_data(tmp_path: Path) -> None:
    path = tmp_path / "d" / "a.sqlite3"
    first = Database(path)
    FactStore(first).add("remembered", approved=True)
    first.close()
    second = Database(path)
    assert len(FactStore(second).list_facts()) == 1
    second.close()


def test_stats_and_clear_all(database: Database) -> None:
    ConversationStore(database).add_message(
        ConversationStore(database).create_session(session_id="s1"), Message.user("hi")
    )
    FactStore(database).add("f", approved=True)
    assert database.stats()["messages"] == 1
    deleted = database.clear_all()
    assert deleted["messages"] == 1
    assert sum(database.stats().values()) == 0


def test_transaction_rolls_back(database: Database) -> None:
    store = FactStore(database)
    store.add("kept", approved=True)
    with pytest.raises(RuntimeError), database.transaction() as connection:
        connection.execute(
            "INSERT INTO facts (fact, category, source, approved, created_at)"
            " VALUES ('rolled back','g','',1,'now')"
        )
        raise RuntimeError("boom")
    assert [f["fact"] for f in store.list_facts()] == ["kept"]


# -- conversations ----------------------------------------------------------
def test_messages_round_trip_including_tool_calls(database: Database) -> None:
    store = ConversationStore(database)
    session = store.create_session(title="t", provider="mock", model="m")
    store.add_message(session, Message.user("hello"))
    store.add_message(
        session, Message.assistant("thinking", [ToolCall(name="read_file", arguments={"p": "a"})])
    )
    store.add_message(
        session, Message.from_tool_result(ToolResult(call_id="c", tool_name="read_file", ok=True))
    )
    messages = store.get_messages(session)
    assert [m.role for m in messages] == ["user", "assistant", "tool"]
    assert messages[1].tool_calls[0].name == "read_file"
    assert messages[1].tool_calls[0].arguments == {"p": "a"}


def test_session_listing_counts_messages(database: Database) -> None:
    store = ConversationStore(database)
    session = store.create_session(title="t")
    store.add_message(session, Message.user("a"))
    store.add_message(session, Message.user("b"))
    assert store.list_sessions()[0]["message_count"] == 2


def test_get_messages_limit_returns_the_most_recent(database: Database) -> None:
    store = ConversationStore(database)
    session = store.create_session()
    for index in range(5):
        store.add_message(session, Message.user(f"m{index}"))
    recent = store.get_messages(session, limit=2)
    assert [m.content for m in recent] == ["m3", "m4"]


def test_deleting_a_session_removes_its_messages(database: Database) -> None:
    store = ConversationStore(database)
    session = store.create_session()
    store.add_message(session, Message.user("a"))
    assert store.delete_session(session) == 1
    assert store.get_session(session) is None
    assert store.get_messages(session) == []


# -- facts ------------------------------------------------------------------
def test_only_approved_facts_are_recalled(database: Database) -> None:
    store = FactStore(database)
    store.add("approved fact", approved=True)
    store.add("unapproved proposal", approved=False)
    assert [f["fact"] for f in store.list_facts(approved_only=True)] == ["approved fact"]
    assert len(store.list_facts(approved_only=False)) == 2


def test_prompt_block_only_contains_approved_facts(database: Database) -> None:
    store = FactStore(database)
    store.add("user prefers metric units", approved=True)
    store.add("never approved", approved=False)
    block = store.prompt_block()
    assert "metric units" in block and "never approved" not in block


def test_empty_prompt_block(database: Database) -> None:
    assert FactStore(database).prompt_block() == ""


def test_fact_limit_is_enforced(database: Database) -> None:
    store = FactStore(database, max_facts=2)
    store.add("a", approved=True)
    store.add("b", approved=True)
    with pytest.raises(LimitExceededError):
        store.add("c", approved=True)


def test_facts_are_deduplicated(database: Database) -> None:
    store = FactStore(database)
    store.add("same", approved=True)
    store.add("same", approved=True)
    assert store.count() == 1


def test_removing_a_fact(database: Database) -> None:
    store = FactStore(database)
    fact_id = store.add("x", approved=True)
    assert store.remove(fact_id) is True
    assert store.remove(fact_id) is False


def test_empty_fact_is_rejected(database: Database) -> None:
    with pytest.raises(ValueError):
        FactStore(database).add("   ")


# -- tasks ------------------------------------------------------------------
def test_task_round_trips(database: Database) -> None:
    store = TaskStore(database)
    task = TaskState(goal="do a thing", provider="mock", model="m")
    task.set_plan(["step one"])
    task.transition_to(TaskStatus.RUNNING)
    store.save(task)
    loaded = store.load(task.id)
    assert loaded is not None
    assert loaded.goal == "do a thing"
    assert loaded.status is TaskStatus.RUNNING
    assert loaded.plan.steps[0].description == "step one"


def test_saving_twice_updates_rather_than_duplicates(database: Database) -> None:
    store = TaskStore(database)
    task = TaskState(goal="g")
    store.save(task)
    task.transition_to(TaskStatus.RUNNING)
    store.save(task)
    assert len(store.list_tasks()) == 1
    loaded = store.load(task.id)
    assert loaded is not None and loaded.status is TaskStatus.RUNNING


def test_find_by_prefix(database: Database) -> None:
    store = TaskStore(database)
    task = TaskState(goal="g")
    store.save(task)
    assert store.find(task.id[:10]) is not None
    assert store.find("nomatch") is None


def test_listing_filters_by_status(database: Database) -> None:
    store = TaskStore(database)
    running = TaskState(goal="a")
    running.transition_to(TaskStatus.RUNNING)
    store.save(running)
    store.save(TaskState(goal="b"))
    assert len(store.list_tasks(status=TaskStatus.RUNNING)) == 1


def test_pause_and_cancel_flags_persist(database: Database) -> None:
    store = TaskStore(database)
    task = TaskState(goal="g")
    store.save(task)
    store.request_pause(task.id)
    assert store.load(task.id).pause_requested is True  # type: ignore[union-attr]
    store.request_cancel(task.id, "user asked")
    reloaded = store.load(task.id)
    assert reloaded is not None
    assert reloaded.cancel_requested is True and reloaded.cancel_reason == "user asked"


def test_recover_interrupted_moves_running_to_paused(database: Database) -> None:
    """A process that dies mid-run must not leave a task looking alive."""
    store = TaskStore(database)
    task = TaskState(goal="g")
    task.transition_to(TaskStatus.RUNNING)
    store.save(task)
    recovered = store.recover_interrupted()
    assert recovered == [task.id]
    reloaded = store.load(task.id)
    assert reloaded is not None and reloaded.status is TaskStatus.PAUSED
    assert reloaded.failures[-1].message.startswith("the previous process exited")


def test_corrupt_task_row_is_reported(database: Database) -> None:
    database.execute(
        "INSERT INTO tasks (id, session_id, goal, status, provider, model, workspace,"
        " state_json, created_at, updated_at)"
        " VALUES ('bad','s','g','created','','','','{not json','now','now')"
    )
    with pytest.raises(StorageError):
        TaskStore(database).load("bad")


def test_tool_results_and_verifications_are_recorded(database: Database) -> None:
    store = TaskStore(database)
    task = TaskState(goal="g")
    store.save(task)
    store.record_tool_result(
        ToolResult(call_id="c1", tool_name="write_file", ok=True, output={"path": "a"}),
        task_id=task.id,
        session_id=task.session_id,
        arguments={"path": "a"},
        approved=True,
    )
    rows = store.tool_calls_for(task.id)
    assert rows[0]["tool_name"] == "write_file" and rows[0]["approved"] == 1


def test_deleting_a_task_removes_its_records(database: Database) -> None:
    store = TaskStore(database)
    task = TaskState(goal="g")
    store.save(task)
    store.record_tool_result(
        ToolResult(call_id="c", tool_name="t", ok=True),
        task_id=task.id,
        session_id=task.session_id,
        arguments={},
    )
    assert store.delete(task.id) is True
    assert store.tool_calls_for(task.id) == []


# -- summaries --------------------------------------------------------------
def test_summarize_is_extractive() -> None:
    messages = [
        Message.user("please summarise the files"),
        Message.assistant("", [ToolCall(name="read_file", arguments={})]),
        Message.assistant("the directory holds three notes"),
    ]
    summary = summarize_messages(messages)
    assert "please summarise the files" in summary
    assert "read_file" in summary
    assert "three notes" in summary


def test_compaction_keeps_the_system_message_and_recent_turns() -> None:
    messages = [Message.system("sys")] + [Message.user(f"m{i}") for i in range(20)]
    compacted, summary = compact_conversation(messages, max_messages=8)
    assert summary is not None
    assert compacted[0].role == "system"
    assert compacted[1].metadata.get("compacted") is True
    assert len(compacted) <= 8
    assert compacted[-1].content == "m19"


def test_compaction_is_a_no_op_below_the_limit() -> None:
    messages = [Message.user("a"), Message.user("b")]
    compacted, summary = compact_conversation(messages, max_messages=10)
    assert summary is None and compacted == messages


def test_compaction_never_orphans_a_tool_result() -> None:
    """A `tool` message without its assistant call is rejected by some providers."""
    messages = [Message.system("s")]
    for index in range(6):
        messages.append(Message.assistant("", [ToolCall(id=f"c{index}", name="read_file")]))
        messages.append(
            Message.from_tool_result(
                ToolResult(call_id=f"c{index}", tool_name="read_file", ok=True)
            )
        )
    compacted, _ = compact_conversation(messages, max_messages=6)
    body = [m for m in compacted if m.role != "system"]
    assert body[0].role != "tool"


def test_summary_store(database: Database) -> None:
    store = SummaryStore(database)
    store.add("s1", "first summary", 10)
    store.add("s1", "second summary", 20)
    latest = store.latest("s1")
    assert latest is not None and latest["summary"] == "second summary"
    assert len(store.list_summaries("s1")) == 2
