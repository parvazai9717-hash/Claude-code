"""Skill metadata, discovery, validation and permission matching."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.config import ApprovalMode, Config
from agent.errors import SkillError
from agent.skills.loader import discover_skills
from agent.skills.manifest import parse_skill_markdown
from agent.skills.registry import SkillRegistry

VALID = """\
---
name: reporter
description: Write a report from workspace files.
activation:
  - the user asks for a report
required_tools:
  - list_files
  - read_file
allowed_paths:
  - files
requires_approval: false
limitations:
  - read-only
---

# Reporter

1. List the files.
2. Read them.
"""

ALL_TOOLS = ["list_files", "read_file", "write_file", "run_shell", "verify_result"]


def _write_skill(root: Path, name: str, text: str) -> Path:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "SKILL.md").write_text(text)
    return directory


# -- parsing ----------------------------------------------------------------
def test_valid_manifest_parses() -> None:
    manifest = parse_skill_markdown(VALID)
    assert manifest.name == "reporter"
    assert manifest.required_tools == ["list_files", "read_file"]
    assert manifest.limitations == ["read-only"]
    assert "List the files" in manifest.body


def test_missing_front_matter_is_rejected() -> None:
    with pytest.raises(SkillError, match="front-matter"):
        parse_skill_markdown("# Just a heading\n")


def test_malformed_yaml_is_rejected() -> None:
    with pytest.raises(SkillError, match="not valid YAML"):
        parse_skill_markdown("---\nname: [unclosed\n---\nbody\n")


def test_missing_required_fields_are_rejected() -> None:
    with pytest.raises(SkillError, match="description"):
        parse_skill_markdown("---\nname: noderscription\n---\nbody\n")


def test_invalid_name_is_rejected() -> None:
    with pytest.raises(SkillError, match="invalid skill metadata"):
        parse_skill_markdown("---\nname: Bad Name\ndescription: d\n---\n")


def test_a_scalar_is_accepted_where_a_list_is_expected() -> None:
    manifest = parse_skill_markdown(
        "---\nname: scalar\ndescription: d\nrequired_tools: read_file\n---\n"
    )
    assert manifest.required_tools == ["read_file"]


def test_non_mapping_front_matter_is_rejected() -> None:
    with pytest.raises(SkillError, match="mapping"):
        parse_skill_markdown("---\n- a\n- b\n---\nbody\n")


# -- discovery --------------------------------------------------------------
def test_discovery_finds_valid_skills(tmp_path: Path) -> None:
    _write_skill(tmp_path, "reporter", VALID)
    manifests, problems = discover_skills(tmp_path)
    assert [m.name for m in manifests] == ["reporter"]
    assert problems == []


def test_discovery_reports_problems_without_aborting(tmp_path: Path) -> None:
    _write_skill(tmp_path, "reporter", VALID)
    _write_skill(tmp_path, "broken", "no front matter here")
    (tmp_path / "empty").mkdir()
    manifests, problems = discover_skills(tmp_path)
    assert [m.name for m in manifests] == ["reporter"]
    assert {name for name, _ in problems} == {"broken", "empty"}


def test_name_must_match_the_directory(tmp_path: Path) -> None:
    _write_skill(tmp_path, "other-name", VALID)
    manifests, problems = discover_skills(tmp_path)
    assert manifests == []
    assert "does not match the directory name" in problems[0][1]


def test_missing_directory_is_not_an_error(tmp_path: Path) -> None:
    manifests, problems = discover_skills(tmp_path / "nope")
    assert manifests == [] and problems == []


def test_oversized_skill_file_is_skipped(tmp_path: Path) -> None:
    _write_skill(tmp_path, "big", "---\nname: big\ndescription: d\n---\n" + "x" * 300_000)
    _, problems = discover_skills(tmp_path)
    assert "too large" in problems[0][1]


# -- permission matching ----------------------------------------------------
def test_available_when_all_tools_are_registered(tmp_path: Path, config: Config) -> None:
    _write_skill(tmp_path, "reporter", VALID)
    registry = SkillRegistry.from_directory(tmp_path)
    check = registry.check("reporter", available_tools=ALL_TOOLS, config=config)
    assert check.available is True


def test_a_skill_cannot_grant_itself_a_missing_tool(tmp_path: Path, config: Config) -> None:
    _write_skill(
        tmp_path,
        "needy",
        "---\nname: needy\ndescription: d\nrequired_tools:\n  - browser_open\n---\n",
    )
    registry = SkillRegistry.from_directory(tmp_path)
    check = registry.check("needy", available_tools=ALL_TOOLS, config=config)
    assert check.available is False
    assert check.missing_tools == ("browser_open",)


def test_automatic_mode_blocks_approval_gated_skills(tmp_path: Path, config: Config) -> None:
    _write_skill(
        tmp_path,
        "writer",
        "---\nname: writer\ndescription: d\nrequired_tools:\n  - write_file\n"
        "requires_approval: true\n---\n",
    )
    config.approval_mode = ApprovalMode.AUTOMATIC
    config.auto_approve_tools = []
    registry = SkillRegistry.from_directory(tmp_path)
    check = registry.check("writer", available_tools=ALL_TOOLS, config=config)
    assert check.available is False
    assert "automatic" in check.reason


def test_automatic_mode_allows_an_allowlisted_skill(tmp_path: Path, config: Config) -> None:
    _write_skill(
        tmp_path,
        "writer",
        "---\nname: writer\ndescription: d\nrequired_tools:\n  - write_file\n"
        "requires_approval: true\n---\n",
    )
    config.approval_mode = ApprovalMode.AUTOMATIC
    config.auto_approve_tools = ["write_file"]
    registry = SkillRegistry.from_directory(tmp_path)
    assert registry.check("writer", available_tools=ALL_TOOLS, config=config).available


def test_unknown_skill_is_an_error(tmp_path: Path, config: Config) -> None:
    registry = SkillRegistry.from_directory(tmp_path)
    with pytest.raises(SkillError, match="no skill named"):
        registry.check("nope", available_tools=ALL_TOOLS, config=config)


def test_prompt_block_lists_only_usable_skills(tmp_path: Path, config: Config) -> None:
    _write_skill(tmp_path, "reporter", VALID)
    _write_skill(
        tmp_path, "needy", "---\nname: needy\ndescription: d\nrequired_tools:\n  - nope\n---\n"
    )
    registry = SkillRegistry.from_directory(tmp_path)
    block = registry.prompt_block(available_tools=ALL_TOOLS, config=config)
    assert "reporter" in block and "needy" not in block
    assert "grants no extra permissions" in block


def test_the_shipped_example_skill_is_valid_and_read_only() -> None:
    """The example skill must parse and must not ask for write access."""
    registry = SkillRegistry.from_directory(Path("skills"))
    assert registry.problems == []
    example = registry.get("example")
    assert set(example.required_tools) <= {"list_files", "read_file", "search_files"}
