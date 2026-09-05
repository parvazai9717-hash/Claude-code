"""Persistence, memory approval and event recording through the real runtime."""

from __future__ import annotations

from pathlib import Path

from agent.config import Config
from agent.errors import ErrorCategory
from agent.memory.conversations import ConversationStore
from agent.memory.database import Database
from agent.memory.facts import FactStore
from agent.memory.tasks import TaskStore
from agent.providers.mock import MockProvider
from agent.runtime import build_runner
from agent.security.approvals import PolicyApprover
from agent.task_state import TaskState


async def test_a_run_is_fully_recorded(
    config: Config, workspace: Path, tmp_path: Path, approve_all: PolicyApprover
) -> None:
    database = Database(tmp_path / "run.sqlite3")
    (workspace / "files" / "a.txt").write_text("data")
    runner = build_runner(
        config,
        provider=MockProvider(
            [
                MockProvider.call("read_file", {"path": "files/a.txt"}),
                MockProvider.call("write_file", {"path": "outputs/b.md", "content": "data"}),
                MockProvider.call(
                    "verify_result", {"method": "file_exists", "path": "outputs/b.md"}
                ),
                MockProvider.text("Done."),
            ]
        ),
        approver=approve_all,
        database=database,
    )
    task = TaskState(goal="copy a.txt into a report")
    result = await runner.run(task)

    store = TaskStore(database)
    reloaded = store.load(task.id)
    assert reloaded is not None
    assert reloaded.status is result.task.status
    assert len(reloaded.actions) == 3

    calls = store.tool_calls_for(task.id)
    assert [c["tool_name"] for c in calls] == ["read_file", "write_file", "verify_result"]

    events = store.events_for(task.id)
    types = {e["type"] for e in events}
    assert {"task_started", "tool_result", "task_finished"} <= types
    database.close()


async def test_a_fact_is_stored_only_after_approval(
    config: Config, tmp_path: Path, approve_all: PolicyApprover, deny_all: PolicyApprover
) -> None:
    database = Database(tmp_path / "facts.sqlite3")
    facts = FactStore(database)

    # Denied: nothing is written.
    denied_runner = build_runner(
        config,
        provider=MockProvider(
            [
                MockProvider.call("remember_fact", {"fact": "the user prefers metric units"}),
                MockProvider.text("Not saved."),
            ]
        ),
        approver=deny_all,
        database=database,
    )
    denied = await denied_runner.run(TaskState(goal="remember something"))
    assert denied.task.actions[0].error_category is ErrorCategory.APPROVAL_DENIED
    assert facts.count() == 0

    # Approved: the fact is stored and recalled.
    approved_runner = build_runner(
        config,
        provider=MockProvider(
            [
                MockProvider.call("remember_fact", {"fact": "the user prefers metric units"}),
                MockProvider.text("Saved."),
            ]
        ),
        approver=approve_all,
        database=database,
    )
    await approved_runner.run(TaskState(goal="remember something"))
    stored = facts.list_facts(approved_only=True)
    assert len(stored) == 1 and stored[0]["fact"] == "the user prefers metric units"
    assert "metric units" in approved_runner.build_system_message().content
    database.close()


async def test_a_credential_shaped_fact_is_refused_even_when_approved(
    config: Config, tmp_path: Path, approve_all: PolicyApprover
) -> None:
    database = Database(tmp_path / "facts2.sqlite3")
    runner = build_runner(
        config,
        provider=MockProvider(
            [
                MockProvider.call(
                    "remember_fact",
                    {"fact": "the api key is AIzaSyA1B2C3D4E5F6G7H8I9J0K1L2M3N4O5P6"},
                ),
                MockProvider.text("That could not be stored."),
            ]
        ),
        approver=approve_all,
        database=database,
    )
    result = await runner.run(TaskState(goal="remember the key"))
    assert result.task.actions[0].error_category is ErrorCategory.SECRET_PROTECTED
    assert FactStore(database).count() == 0
    database.close()


async def test_conversation_history_survives_and_reloads(
    config: Config, tmp_path: Path, approve_all: PolicyApprover
) -> None:
    database = Database(tmp_path / "conv.sqlite3")
    conversations = ConversationStore(database)
    runner = build_runner(
        config,
        provider=MockProvider([MockProvider.text("First answer.")]),
        approver=approve_all,
        database=database,
    )
    task = TaskState(goal="say something")
    result = await runner.run(task)
    conversations.add_messages(task.session_id, result.messages, task_id=task.id)

    restored = ConversationStore(Database(tmp_path / "conv.sqlite3")).get_messages(task.session_id)
    assert any(m.role == "assistant" and "First answer." in m.content for m in restored)
    database.close()


async def test_events_are_written_to_the_jsonl_log(
    config: Config, tmp_path: Path, approve_all: PolicyApprover
) -> None:
    from agent.events import EventBus, JsonlEventWriter
    from agent.security.redaction import Redactor

    log_path = tmp_path / "events.jsonl"
    redactor = Redactor(environ={"SOME_API_KEY": "sk-log-secret-abcdef12345"})
    bus = EventBus(redactor=redactor.redact)
    bus.subscribe(JsonlEventWriter(log_path))
    runner = build_runner(
        config,
        provider=MockProvider(
            [MockProvider.call("get_current_time", {}), MockProvider.text("done")]
        ),
        approver=approve_all,
        events=bus,
    )
    await runner.run(TaskState(goal="what time is it"))

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) > 3
    assert all(line.startswith("{") for line in lines)
    assert "sk-log-secret-abcdef12345" not in log_path.read_text()


async def test_conversation_compaction_keeps_the_run_going(
    config: Config, tmp_path: Path, approve_all: PolicyApprover
) -> None:
    """A long run must compact rather than overflow."""
    config.limits.max_conversation_messages = 8
    config.limits.max_steps = 12
    database = Database(tmp_path / "compact.sqlite3")
    script = [MockProvider.call("get_current_time", {}) for _ in range(10)]
    script.append(MockProvider.text("finished"))
    runner = build_runner(
        config, provider=MockProvider(script), approver=approve_all, database=database
    )
    result = await runner.run(TaskState(goal="check the time repeatedly"))
    assert len(runner.messages) <= config.limits.max_conversation_messages + 2
    assert any(m.metadata.get("compacted") for m in runner.messages)
    assert result.task.total_tool_calls >= 8
    database.close()
