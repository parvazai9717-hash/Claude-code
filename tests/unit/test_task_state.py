"""Task status transitions and record keeping."""

from __future__ import annotations

import pytest

from agent.errors import ErrorCategory
from agent.task_state import (
    ActionRecord,
    InvalidTransitionError,
    Phase,
    TaskState,
    TaskStatus,
    VerificationRecord,
)


def test_new_task_starts_created() -> None:
    task = TaskState(goal="g")
    assert task.status is TaskStatus.CREATED
    assert task.phase is Phase.PLAN
    assert not task.is_terminal


def test_valid_transition_sequence() -> None:
    task = TaskState(goal="g")
    task.transition_to(TaskStatus.RUNNING)
    assert task.started_at is not None
    task.transition_to(TaskStatus.WAITING_FOR_APPROVAL)
    task.transition_to(TaskStatus.RUNNING)
    task.transition_to(TaskStatus.PAUSED)
    task.transition_to(TaskStatus.RUNNING)
    task.transition_to(TaskStatus.COMPLETED)
    assert task.is_terminal
    assert task.finished_at is not None


@pytest.mark.parametrize(
    ("start", "target"),
    [
        (TaskStatus.CREATED, TaskStatus.PAUSED),
        (TaskStatus.CREATED, TaskStatus.COMPLETED),
        (TaskStatus.COMPLETED, TaskStatus.RUNNING),
        (TaskStatus.CANCELLED, TaskStatus.RUNNING),
        (TaskStatus.FAILED, TaskStatus.RUNNING),
        (TaskStatus.PAUSED, TaskStatus.COMPLETED),
    ],
)
def test_invalid_transitions_are_rejected(start: TaskStatus, target: TaskStatus) -> None:
    task = TaskState(goal="g")
    task.status = start
    with pytest.raises(InvalidTransitionError):
        task.transition_to(target)


def test_transition_to_same_status_is_a_no_op() -> None:
    task = TaskState(goal="g")
    task.transition_to(TaskStatus.CREATED)
    assert task.status is TaskStatus.CREATED


def test_terminal_statuses_are_final() -> None:
    for status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
        task = TaskState(goal="g")
        task.status = status
        assert task.is_terminal
        assert not task.can_transition_to(TaskStatus.RUNNING)


def test_plan_summary_marks_completion() -> None:
    task = TaskState(goal="g")
    task.set_plan(["first", "second"])
    task.plan.steps[0].done = True
    assert "[x] first" in task.plan.summary()
    assert "[ ] second" in task.plan.summary()
    assert len(task.plan.open_steps) == 1
    assert task.plan.revision == 1


def test_records_are_accumulated() -> None:
    task = TaskState(goal="g")
    task.current_step = 2
    task.record_observation("read_file", "read a.txt")
    task.record_action(ActionRecord(step=2, call_id="c1", tool_name="write_file", ok=True))
    task.record_failure(ErrorCategory.TOOL_FAILED, "boom", tool_name="run_shell")
    assert len(task.observations) == len(task.actions) == len(task.failures) == 1
    assert task.total_tool_calls == 1


def test_retry_counting_is_per_signature() -> None:
    task = TaskState(goal="g")
    assert task.note_retry("write:x") == 1
    assert task.note_retry("write:x") == 2
    assert task.retries_for("write:x") == 2
    assert task.retries_for("other") == 0


def test_unverified_actions_excludes_verified_ones() -> None:
    task = TaskState(goal="g")
    action = task.record_action(ActionRecord(step=1, call_id="c", tool_name="write_file", ok=True))
    assert task.unverified_actions() == [action]
    task.record_verification(
        VerificationRecord(step=1, action_id=action.id, method="file_exists", verified=True)
    )
    assert task.unverified_actions() == []


def test_failed_verification_does_not_mark_an_action_verified() -> None:
    task = TaskState(goal="g")
    action = task.record_action(ActionRecord(step=1, call_id="c", tool_name="write_file", ok=True))
    task.record_verification(
        VerificationRecord(step=1, action_id=action.id, method="file_exists", verified=False)
    )
    assert task.unverified_actions() == [action]


def test_context_summary_contains_no_hidden_reasoning() -> None:
    task = TaskState(goal="g")
    task.set_plan(["do a thing"])
    task.record_observation("list_files", "3 entries")
    summary = task.context_summary()
    assert "do a thing" in summary and "3 entries" in summary


def test_round_trips_through_json() -> None:
    task = TaskState(goal="g")
    task.transition_to(TaskStatus.RUNNING)
    task.set_plan(["a"])
    task.record_action(ActionRecord(step=1, call_id="c", tool_name="read_file", ok=True))
    restored = TaskState.model_validate_json(task.model_dump_json())
    assert restored.id == task.id
    assert restored.status is TaskStatus.RUNNING
    assert restored.plan.steps[0].description == "a"
    assert restored.actions[0].tool_name == "read_file"
