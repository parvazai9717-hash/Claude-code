"""Shared fixtures.

Every fixture is rooted at a `tmp_path`. Nothing in the suite reads or writes the
real home directory, the real workspace, or the real environment.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from agent.config import ApprovalMode, Config
from agent.events import EventBus
from agent.memory.database import Database
from agent.security.approvals import ApprovalDecision, PolicyApprover
from agent.security.paths import PathPolicy
from agent.security.permissions import PermissionChecker
from agent.security.redaction import Redactor
from agent.tools.base import ToolContext
from agent.tools.filesystem import initialize_workspace
from agent.tools.registry import ToolRegistry


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """An initialised workspace inside the test's temporary directory."""
    return initialize_workspace(tmp_path / "workspace")


@pytest.fixture
def config(tmp_path: Path, workspace: Path) -> Config:
    """A configuration pointed entirely at temporary directories."""
    return Config(
        provider="mock",
        workspace=workspace,
        data_dir=tmp_path / "data",
        skills_dir=tmp_path / "skills",
        approval_mode=ApprovalMode.RISKY,
    )


@pytest.fixture
def redactor() -> Redactor:
    """A redactor built from a fixed environment, not the real one."""
    return Redactor(environ={"TEST_API_KEY": "sk-test-secret-value-1234"})


@pytest.fixture
def paths(config: Config) -> PathPolicy:
    return PathPolicy.for_workspace(config.workspace, max_file_bytes=config.limits.max_file_bytes)


@pytest.fixture
def context(config: Config, paths: PathPolicy, redactor: Redactor) -> ToolContext:
    return ToolContext(config=config, paths=paths, redactor=redactor, task_id="task_test", step=1)


@pytest.fixture
def database(tmp_path: Path) -> Iterator[Database]:
    db = Database(tmp_path / "data" / "test.sqlite3")
    yield db
    db.close()


@pytest.fixture
def approve_all() -> PolicyApprover:
    return PolicyApprover(default=ApprovalDecision.APPROVE_ONCE)


@pytest.fixture
def deny_all() -> PolicyApprover:
    return PolicyApprover(default=ApprovalDecision.DENY)


@pytest.fixture
def registry(config: Config, approve_all: PolicyApprover) -> ToolRegistry:
    """A registry with the full default tool set and an approve-everything policy."""
    from agent.tools import build_default_tools

    reg = ToolRegistry(
        permissions=PermissionChecker(config), approver=approve_all, events=EventBus()
    )
    reg.register_all(build_default_tools(include_memory=False))
    return reg
