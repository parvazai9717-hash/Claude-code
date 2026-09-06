"""The management API — the surface a UI drives."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.config import Config
from agent.errors import ConfigurationError
from agent.manage import AgentManager
from agent.memory.database import Database


@pytest.fixture
def manager(config: Config, database: Database) -> AgentManager:
    return AgentManager(config=config, database=database)


# -- connectors -------------------------------------------------------------
def test_connectors_start_empty(manager: AgentManager) -> None:
    assert manager.list_connectors() == []


def test_adding_a_connector(manager: AgentManager) -> None:
    entry = manager.add_connector(
        name="github", kind="mcp_stdio", command="npx", args=["-y", "server"], env=["GH_TOKEN"]
    )
    assert entry["name"] == "github"
    assert entry["enabled"] is False, "a new connector must not be live immediately"
    listed = manager.list_connectors()
    assert listed[0]["tool_prefix"] == "mcp__github__"


def test_a_connector_never_exposes_a_credential(
    manager: AgentManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GH_TOKEN", "sk-should-not-appear-anywhere")
    manager.add_connector(name="github", kind="mcp_stdio", command="npx", env=["GH_TOKEN"])
    rendered = str(manager.list_connectors())
    assert "sk-should-not-appear-anywhere" not in rendered
    assert "GH_TOKEN" in rendered


def test_missing_credentials_are_reported(
    manager: AgentManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NOT_SET_ANYWHERE", raising=False)
    manager.add_connector(name="demo", kind="mcp_stdio", command="npx", env=["NOT_SET_ANYWHERE"])
    assert manager.list_connectors()[0]["missing_credentials"] == ["NOT_SET_ANYWHERE"]


def test_an_invalid_connector_is_rejected_with_a_reason(manager: AgentManager) -> None:
    with pytest.raises(ConfigurationError, match="invalid connector definition"):
        manager.add_connector(name="Bad Name", kind="mcp_stdio", command="npx")
    with pytest.raises(ConfigurationError, match="bare program name"):
        manager.add_connector(name="demo", kind="mcp_stdio", command="/usr/bin/thing")


def test_enable_disable_and_remove(manager: AgentManager) -> None:
    manager.add_connector(name="demo", kind="mcp_stdio", command="npx")
    assert manager.set_connector_enabled("demo", True)["enabled"] is True  # type: ignore[index]
    assert manager.set_connector_enabled("demo", False)["enabled"] is False  # type: ignore[index]
    assert manager.set_connector_enabled("missing", True) is None
    assert manager.remove_connector("demo") is True
    assert manager.remove_connector("demo") is False


def test_connectors_persist_across_managers(config: Config, database: Database) -> None:
    AgentManager(config=config, database=database).add_connector(
        name="demo", kind="mcp_http", url="https://example.com/mcp"
    )
    reloaded = AgentManager(config=config, database=database).list_connectors()
    assert [c["name"] for c in reloaded] == ["demo"]


async def test_testing_a_connector_reports_disabled(manager: AgentManager) -> None:
    manager.add_connector(name="demo", kind="mcp_stdio", command="npx")
    statuses = await manager.test_connectors()
    assert statuses[0]["enabled"] is False
    assert "disabled" in statuses[0]["detail"]


# -- tools, providers, media ------------------------------------------------
def test_listing_tools_includes_the_risk_classification(manager: AgentManager) -> None:
    tools = {t["name"]: t for t in manager.list_tools()}
    assert tools["read_file"]["read_only"] is True
    assert tools["write_file"]["requires_approval"] is True
    assert tools["run_shell"]["risk"] == "high"
    assert tools["write_file"]["reversible"] is False
    assert all(t["source"] == "built-in" for t in tools.values())


def test_shell_can_be_removed_from_the_tool_list(manager: AgentManager) -> None:
    manager.config.shell_enabled = False
    assert "run_shell" not in {t["name"] for t in manager.list_tools()}


def test_listing_providers_reports_capabilities(manager: AgentManager) -> None:
    providers = {p["name"]: p for p in manager.list_providers()}
    assert providers["mock"]["active"] is True
    assert providers["ollama"]["credential_env_var"] is None
    assert providers["gemini"]["credential_env_var"] == "GEMINI_API_KEY"


def test_media_support_reflects_the_active_provider(manager: AgentManager) -> None:
    support = manager.media_support()
    assert support["accepted"]["image"] is True
    assert support["accepted"]["video"] is False, "video is off unless explicitly enabled"
    assert "image/png" in support["supported_types"]["image"]


def test_media_support_honours_the_video_switch(manager: AgentManager) -> None:
    manager.config.media.enable_video = True
    assert manager.media_support()["video_enabled"] is True


# -- skills, history, stats -------------------------------------------------
def test_listing_skills(manager: AgentManager, tmp_path: Path) -> None:
    skill = tmp_path / "skills" / "demo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: demo\ndescription: A demo.\nrequired_tools:\n  - read_file\n---\nbody\n"
    )
    manager.config.skills_dir = tmp_path / "skills"
    entries = {s["name"]: s for s in manager.list_skills()}
    assert entries["demo"]["available"] is True


def test_listing_a_broken_skill_explains_why(manager: AgentManager, tmp_path: Path) -> None:
    broken = tmp_path / "skills" / "broken"
    broken.mkdir(parents=True)
    (broken / "SKILL.md").write_text("no front matter")
    manager.config.skills_dir = tmp_path / "skills"
    entry = next(s for s in manager.list_skills() if s["name"] == "broken")
    assert entry["available"] is False and entry["reason"]


def test_stats_summarises_the_configuration(manager: AgentManager) -> None:
    manager.add_connector(name="demo", kind="mcp_stdio", command="npx", enabled=True)
    stats = manager.stats()
    assert stats["provider"] == "mock"
    assert stats["connectors"] == {"configured": 1, "enabled": 1}
    assert "tasks" in stats["database"]


def test_safe_config_never_contains_a_key(
    manager: AgentManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "sk-never-render-this-value")
    assert "sk-never-render-this-value" not in str(manager.safe_config())
