#!/usr/bin/env python3
"""A complete agent loop, offline.

Runs the real runtime, the real tool registry and the real security layer against
a scripted :class:`~agent.providers.mock.MockProvider`, in a throwaway workspace.
No network, no API key, no Ollama server, no browser.

    python scripts/demo_offline.py

The scenario deliberately includes a *failure*: the agent first reads a path that
does not exist, replans, then reads the right one, writes a report, and verifies
the write before claiming success.

Because a failure is on record, the run finishes as ``partial`` rather than
``completed`` — even though every remaining action succeeded and the write is
verified. That distinction is the point of the exercise.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

# Allow running straight from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent.config import ApprovalMode, Config
from agent.events import Event, EventType
from agent.providers.mock import MockProvider
from agent.runtime import build_runner
from agent.security.approvals import ApprovalDecision, PolicyApprover
from agent.task_state import TaskState

GOAL = "Read files/notes.txt and write a summary to outputs/report.md"


def show(event: Event) -> None:
    """Print the lifecycle events a user would see in the terminal."""
    interesting = {
        EventType.TOOL_REQUESTED,
        EventType.APPROVAL_REQUESTED,
        EventType.TOOL_RESULT,
        EventType.VERIFICATION_RESULT,
        EventType.REPLAN,
    }
    if event.type in interesting:
        print(f"  [{event.type.value}] {event.message}")


async def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="local-agent-demo-"))
    config = Config(
        provider="mock",
        workspace=root / "workspace",
        data_dir=root / "data",
        skills_dir=Path(__file__).resolve().parent.parent / "skills",
        approval_mode=ApprovalMode.RISKY,
    )
    config.ensure_directories()
    (config.workspace / "files" / "notes.txt").write_text(
        "The project uses Python 3.11, SQLite for storage, and no vector database.\n"
    )

    provider = MockProvider(
        [
            # 1. Look around.
            MockProvider.call("list_files", {"path": "files"}),
            # 2. Get the path wrong — the runtime must record the failure and replan.
            MockProvider.call("read_file", {"path": "files/nope.txt"}),
            # 3. Read the right file.
            MockProvider.call("read_file", {"path": "files/notes.txt"}),
            # 4. Write the report (requires approval).
            MockProvider.call(
                "write_file",
                {
                    "path": "outputs/report.md",
                    "content": "# Summary\n\nPython 3.11, SQLite storage, no vector database.\n",
                },
            ),
            # 5. Verify the write before claiming anything.
            MockProvider.call(
                "verify_result",
                {
                    "method": "file_contains",
                    "path": "outputs/report.md",
                    "expected": "SQLite storage",
                    "claim": "the report was written",
                },
            ),
            MockProvider.text(
                "I read files/notes.txt and wrote outputs/report.md, "
                "verified to contain the expected summary."
            ),
        ]
    )

    # A scripted approver stands in for the human at the terminal.
    approver = PolicyApprover(default=ApprovalDecision.APPROVE_ONCE)
    runner = build_runner(
        config, provider=provider, approver=approver, database=config.database_path
    )
    runner.events.subscribe(show)

    print(f"workspace: {config.workspace}")
    print(f"goal:      {GOAL}\n")
    result = await runner.run(TaskState(goal=GOAL))

    print(f"\noutcome: {result.outcome}")
    print(result.summary())
    print(f"\napprovals requested: {[r.tool.name for r in approver.seen]}")
    print(f"report on disk:      {(config.workspace / 'outputs' / 'report.md').exists()}")
    print(
        f"steps: {result.task.current_step}  tool calls: {result.task.total_tool_calls}  "
        f"replans: {result.task.replan_count}"
    )

    # The expected outcome is `partial`, not `completed`: the work finished and the
    # write is verified, but a failure is on record, and the runtime reports that
    # rather than rounding a run with a failure up to a clean success.
    ok = (
        result.outcome == "partial"
        and (config.workspace / "outputs" / "report.md").exists()
        and any(v.verified for v in result.task.verifications)
        and result.unverified == []
        and result.task.replan_count >= 1
    )
    print("\nDEMO PASSED" if ok else "\nDEMO FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
