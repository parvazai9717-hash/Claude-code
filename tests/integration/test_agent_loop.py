"""End-to-end runtime scenarios, all offline.

Each test drives the real runtime with the real tool registry, the real security
layer and a real SQLite database — only the model is scripted. Nothing here
touches the network, an API key, an Ollama server, a browser or the real home
directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.config import Config
from agent.errors import ErrorCategory
from agent.memory.database import Database
from agent.memory.tasks import TaskStore
from agent.providers.mock import MockProvider
from agent.runtime import AgentRunner, build_runner
from agent.security.approvals import PolicyApprover
from agent.task_state import TaskState, TaskStatus


def _runner(
    config: Config,
    script: list,
    approver: PolicyApprover,
    database: Database | None = None,
) -> AgentRunner:
    return build_runner(
        config,
        provider=MockProvider(script),
        approver=approver,
        database=database,
    )


# -- 1. conversation with no tool calls -------------------------------------
async def test_plain_conversation(config: Config, approve_all: PolicyApprover) -> None:
    runner = _runner(config, [MockProvider.text("Hello. I did not need any tools.")], approve_all)
    result = await runner.run(TaskState(goal="say hello"))
    assert result.outcome == "completed"
    assert "Hello" in result.text
    assert result.task.total_tool_calls == 0
    assert result.task.status is TaskStatus.COMPLETED


# -- 2. read a file, then answer with verification --------------------------
async def test_read_then_answer(
    config: Config, workspace: Path, approve_all: PolicyApprover
) -> None:
    (workspace / "files" / "notes.txt").write_text("the answer is 42\n")
    runner = _runner(
        config,
        [
            MockProvider.call("read_file", {"path": "files/notes.txt"}),
            MockProvider.text("The file says the answer is 42."),
        ],
        approve_all,
    )
    result = await runner.run(TaskState(goal="what does notes.txt say?"))
    assert result.outcome == "completed"
    assert result.task.actions[0].tool_name == "read_file"
    assert result.task.actions[0].ok is True
    # A read is recorded as an observation, not as an unverified claim.
    assert result.task.observations[0].source == "read_file"
    assert result.unverified == []


# -- 3. write requiring approval --------------------------------------------
async def test_approved_write_is_verified(
    config: Config, workspace: Path, approve_all: PolicyApprover
) -> None:
    runner = _runner(
        config,
        [
            MockProvider.call("write_file", {"path": "outputs/r.md", "content": "hello there"}),
            MockProvider.call(
                "verify_result",
                {"method": "file_contains", "path": "outputs/r.md", "expected": "hello there"},
            ),
            MockProvider.text("Wrote outputs/r.md and verified its contents."),
        ],
        approve_all,
    )
    result = await runner.run(TaskState(goal="write a report"))
    assert result.outcome == "completed"
    assert (workspace / "outputs" / "r.md").read_text() == "hello there"
    assert len(approve_all.seen) == 1, "the write must be approved exactly once"
    assert approve_all.seen[0].tool.name == "write_file"
    assert result.task.verifications[-1].verified is True
    assert result.unverified == []


# -- 4. approval denial ------------------------------------------------------
async def test_denied_write_is_reported_honestly(
    config: Config, workspace: Path, deny_all: PolicyApprover
) -> None:
    runner = _runner(
        config,
        [
            MockProvider.call("write_file", {"path": "outputs/r.md", "content": "x"}),
            MockProvider.text("The write was denied, so nothing was created."),
        ],
        deny_all,
    )
    result = await runner.run(TaskState(goal="write a report"))
    assert not (workspace / "outputs" / "r.md").exists()
    denied = [a for a in result.task.actions if a.error_category is ErrorCategory.APPROVAL_DENIED]
    assert len(denied) == 1
    assert result.outcome == "partial"
    assert "Denied by the human" in result.summary()


# -- 5. tool failure and replan ---------------------------------------------
async def test_failure_triggers_a_replan(
    config: Config, workspace: Path, approve_all: PolicyApprover
) -> None:
    (workspace / "files" / "real.txt").write_text("found it")
    runner = _runner(
        config,
        [
            MockProvider.call("read_file", {"path": "files/missing.txt"}),
            MockProvider.call("read_file", {"path": "files/real.txt"}),
            MockProvider.text("The first path was wrong; the second worked."),
        ],
        approve_all,
    )
    result = await runner.run(TaskState(goal="read the notes"))
    assert result.task.failures[0].category is ErrorCategory.NOT_FOUND
    assert result.task.replan_count >= 1
    assert result.task.actions[-1].ok is True
    assert result.outcome == "partial", "a run with a recorded failure is not a clean success"


# -- 6. verification failure and bounded retry ------------------------------
async def test_failed_verification_is_not_reported_as_success(
    config: Config, approve_all: PolicyApprover
) -> None:
    runner = _runner(
        config,
        [
            MockProvider.call("write_file", {"path": "outputs/a.md", "content": "actual text"}),
            MockProvider.call(
                "verify_result",
                {"method": "file_contains", "path": "outputs/a.md", "expected": "not present"},
            ),
            MockProvider.text("The verification failed; the file does not contain that text."),
        ],
        approve_all,
    )
    result = await runner.run(TaskState(goal="write and check"))
    assert result.task.verifications[-1].verified is False
    assert any(f.category is ErrorCategory.VERIFICATION_FAILED for f in result.task.failures)
    assert result.outcome != "completed"


async def test_repeated_identical_failures_are_bounded(
    config: Config, approve_all: PolicyApprover
) -> None:
    """The same failing call must not be retried forever."""
    config.limits.max_retries_per_action = 1
    config.limits.max_steps = 10
    script = [MockProvider.call("read_file", {"path": "files/missing.txt"})] * 8
    script.append(MockProvider.text("giving up"))
    runner = _runner(config, script, approve_all)
    result = await runner.run(TaskState(goal="keep failing"))
    limited = [a for a in result.task.actions if a.error_category is ErrorCategory.LIMIT_EXCEEDED]
    assert limited, "the retry limit must eventually stop the identical call"
    assert result.task.status.value in {"completed", "failed"}


# -- 7. pause and resume across a restart -----------------------------------
async def test_pause_stops_before_the_next_action(
    config: Config, workspace: Path, tmp_path: Path, approve_all: PolicyApprover
) -> None:
    database = Database(tmp_path / "d.sqlite3")
    runner = _runner(
        config,
        [
            MockProvider.call("write_file", {"path": "outputs/a.md", "content": "first"}),
            MockProvider.call("write_file", {"path": "outputs/b.md", "content": "second"}),
        ],
        approve_all,
        database,
    )
    task = TaskState(goal="write two files")

    # Pause as soon as the first write lands, before the second one runs.
    def pause_after_first_write(event) -> None:  # type: ignore[no-untyped-def]
        if event.type.value == "tool_result" and event.data.get("tool") == "write_file":
            runner.request_pause()

    runner.events.subscribe(pause_after_first_write)
    result = await runner.run(task)

    assert result.outcome == "paused"
    assert result.task.status is TaskStatus.PAUSED
    assert (workspace / "outputs" / "a.md").exists()
    assert not (workspace / "outputs" / "b.md").exists(), "pause must precede the next action"
    database.close()


async def test_resume_after_a_simulated_restart(
    config: Config, workspace: Path, tmp_path: Path, approve_all: PolicyApprover
) -> None:
    path = tmp_path / "restart.sqlite3"

    # --- process 1: start, then pause -----------------------------------
    first_db = Database(path)
    first = _runner(
        config,
        [MockProvider.call("write_file", {"path": "outputs/a.md", "content": "first"})],
        approve_all,
        first_db,
    )
    task = TaskState(goal="two-part job")
    first.events.subscribe(
        lambda e: (
            first.request_pause()
            if e.type.value == "tool_result" and e.data.get("tool") == "write_file"
            else None
        )
    )
    await first.run(task)
    task_id = task.id
    first_db.close()

    # --- process 2: reload from disk and continue ------------------------
    second_db = Database(path)
    store = TaskStore(second_db)
    reloaded = store.load(task_id)
    assert reloaded is not None
    assert reloaded.status is TaskStatus.PAUSED
    assert reloaded.actions, "the completed action survived the restart"

    second = _runner(
        config,
        [
            MockProvider.call("write_file", {"path": "outputs/b.md", "content": "second"}),
            MockProvider.text("Finished the second half after resuming."),
        ],
        approve_all,
        second_db,
    )
    result = await second.run(reloaded)
    assert result.outcome in {"completed", "unverified", "partial"}
    assert (workspace / "outputs" / "b.md").exists()
    assert result.task.id == task_id
    second_db.close()


# -- 8. cancellation ---------------------------------------------------------
async def test_cancellation_stops_future_actions(
    config: Config, workspace: Path, approve_all: PolicyApprover
) -> None:
    runner = _runner(
        config,
        [
            MockProvider.call("write_file", {"path": "outputs/a.md", "content": "first"}),
            MockProvider.call("write_file", {"path": "outputs/b.md", "content": "second"}),
        ],
        approve_all,
    )
    runner.events.subscribe(
        lambda e: (
            runner.request_cancel("test cancellation")
            if e.type.value == "tool_result" and e.data.get("tool") == "write_file"
            else None
        )
    )
    result = await runner.run(TaskState(goal="write two files"))
    assert result.outcome == "cancelled"
    assert result.task.status is TaskStatus.CANCELLED
    assert not (workspace / "outputs" / "b.md").exists()


async def test_cancellation_requested_before_the_run_starts(
    config: Config, approve_all: PolicyApprover
) -> None:
    runner = _runner(
        config, [MockProvider.call("write_file", {"path": "a", "content": "x"})], approve_all
    )
    task = TaskState(goal="g")
    task.cancel_requested = True
    result = await runner.run(task)
    assert result.outcome == "cancelled"
    assert result.task.total_tool_calls == 0


# -- 9. provider switching ---------------------------------------------------
@pytest.mark.parametrize("provider_name", ["mock", "ollama", "gemini"])
async def test_the_runtime_is_provider_agnostic(
    config: Config, workspace: Path, approve_all: PolicyApprover, provider_name: str
) -> None:
    """The same scenario must behave identically whatever the provider is called.

    The scripted responses stand in for whatever adapter produced them: the
    runtime never inspects provider identity.
    """
    (workspace / "files" / "n.txt").write_text("content here")
    config.provider = provider_name  # type: ignore[assignment]
    provider = MockProvider(
        [
            MockProvider.call("read_file", {"path": "files/n.txt"}),
            MockProvider.text("Read it."),
        ],
        model=f"{provider_name}-model",
    )
    provider.name = provider_name  # type: ignore[assignment]
    runner = build_runner(config, provider=provider, approver=approve_all)
    result = await runner.run(TaskState(goal="read the file"))
    assert result.outcome == "completed"
    assert result.task.provider == provider_name
    assert result.task.actions[0].tool_name == "read_file"


# -- 10. limits --------------------------------------------------------------
async def test_max_steps_stops_the_run(config: Config, approve_all: PolicyApprover) -> None:
    config.limits.max_steps = 3
    runner = _runner(
        config,
        [MockProvider.call("get_current_time", {})] * 10,
        approve_all,
    )
    result = await runner.run(TaskState(goal="loop forever"))
    assert result.outcome == "partial"
    assert "maximum of 3 agent steps" in result.text
    assert result.task.current_step == 3


async def test_max_total_tool_calls_stops_the_run(
    config: Config, approve_all: PolicyApprover
) -> None:
    config.limits.max_total_tool_calls = 2
    config.limits.max_steps = 20
    runner = _runner(config, [MockProvider.call("get_current_time", {})] * 10, approve_all)
    result = await runner.run(TaskState(goal="loop"))
    assert result.outcome == "partial"
    assert "total tool calls" in result.text


async def test_provider_failure_ends_the_run_cleanly(
    config: Config, approve_all: PolicyApprover
) -> None:
    runner = build_runner(config, provider=MockProvider(healthy=False), approver=approve_all)
    result = await runner.run(TaskState(goal="anything"))
    assert result.outcome == "failed"
    assert result.task.status is TaskStatus.FAILED
    assert result.task.failures[0].category is ErrorCategory.PROVIDER_UNAVAILABLE


# -- 11. multimodal ----------------------------------------------------------
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
WAV_BYTES = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 64


async def test_the_model_receives_the_image_it_asked_to_see(
    config: Config, workspace: Path, approve_all: PolicyApprover
) -> None:
    (workspace / "files" / "chart.png").write_bytes(PNG_BYTES)
    provider = MockProvider(
        [
            MockProvider.call("view_media", {"path": "files/chart.png"}),
            MockProvider.text("The chart shows an upward trend."),
        ]
    )
    runner = build_runner(config, provider=provider, approver=approve_all)
    result = await runner.run(TaskState(goal="what does the chart show?"))
    assert result.outcome == "completed"
    assert [a.kind.value for a in provider.received_attachments] == ["image"]
    assert provider.received_attachments[0].data == PNG_BYTES


async def test_audio_reaches_a_model_that_accepts_it(
    config: Config, workspace: Path, approve_all: PolicyApprover
) -> None:
    (workspace / "files" / "clip.wav").write_bytes(WAV_BYTES)
    provider = MockProvider(
        [
            MockProvider.call("view_media", {"path": "files/clip.wav"}),
            MockProvider.text("The clip is silent."),
        ]
    )
    runner = build_runner(config, provider=provider, approver=approve_all)
    await runner.run(TaskState(goal="what is in the clip?"))
    assert [a.kind.value for a in provider.received_attachments] == ["audio"]


async def test_a_text_only_model_is_never_offered_view_media(
    config: Config, approve_all: PolicyApprover
) -> None:
    """A tool whose results the model cannot perceive is worse than no tool."""
    provider = MockProvider([], supports_vision=False, supports_audio=False)
    runner = build_runner(config, provider=provider, approver=approve_all)
    assert "view_media" not in runner.tools.names()
    assert "view_media" not in runner.build_system_message().content


async def test_video_is_refused_while_disabled(
    config: Config, workspace: Path, approve_all: PolicyApprover
) -> None:
    (workspace / "files" / "clip.mp4").write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 64)
    provider = MockProvider(
        [
            MockProvider.call("view_media", {"path": "files/clip.mp4"}),
            MockProvider.text("I could not watch the video."),
        ],
        supports_video=True,
    )
    runner = build_runner(config, provider=provider, approver=approve_all)
    result = await runner.run(TaskState(goal="watch the clip"))
    assert result.task.actions[0].ok is False
    assert result.task.actions[0].error_category is ErrorCategory.CAPABILITY_UNAVAILABLE
    assert provider.received_attachments == []


async def test_media_cannot_be_loaded_from_outside_the_workspace(
    config: Config, tmp_path: Path, approve_all: PolicyApprover
) -> None:
    secret = tmp_path / "private.png"
    secret.write_bytes(PNG_BYTES)
    provider = MockProvider(
        [
            MockProvider.call("view_media", {"path": "../private.png"}),
            MockProvider.text("I could not read outside the workspace."),
        ]
    )
    runner = build_runner(config, provider=provider, approver=approve_all)
    result = await runner.run(TaskState(goal="look at the private image"))
    assert result.task.actions[0].error_category is ErrorCategory.PATH_ESCAPE
    assert provider.received_attachments == []
