"""The system prompt reflects live configuration, never a hard-coded list."""

from __future__ import annotations

from agent.config import ApprovalMode, Config
from agent.messages import RiskCategory, RiskLevel, ToolDefinition
from agent.prompts import (
    build_planning_hint,
    build_replan_hint,
    build_system_prompt,
    build_verification_hint,
)

TOOLS = [
    ToolDefinition(name="read_file", description="Read a file.", read_only=True),
    ToolDefinition(
        name="write_file",
        description="Write a file.",
        read_only=False,
        requires_approval=True,
        risk=RiskLevel.MEDIUM,
        risk_category=RiskCategory.WRITE,
    ),
]


def test_prompt_lists_the_real_tools(config: Config) -> None:
    prompt = build_system_prompt(config, TOOLS)
    assert "read_file" in prompt and "write_file" in prompt
    assert "Require human approval: write_file" in prompt
    assert "There are no other tools" in prompt


def test_prompt_states_the_active_approval_mode(config: Config) -> None:
    for mode in ApprovalMode:
        config.approval_mode = mode
        assert mode.value in build_system_prompt(config, TOOLS)


def test_prompt_states_the_shell_allowlist(config: Config) -> None:
    config.shell_allowed_commands = ["ls", "cat"]
    prompt = build_system_prompt(config, TOOLS)
    assert "cat, ls" in prompt
    assert "argv list" in prompt


def test_prompt_says_when_shell_is_disabled(config: Config) -> None:
    config.shell_enabled = False
    assert "Shell execution is disabled entirely" in build_system_prompt(config, TOOLS)


def test_prompt_includes_the_workspace(config: Config) -> None:
    assert str(config.workspace.expanduser()) in build_system_prompt(config, TOOLS)


def test_prompt_carries_the_required_honesty_rules(config: Config) -> None:
    prompt = build_system_prompt(config, TOOLS)
    for phrase in (
        "verification evidence",
        "Inspect before you modify",
        "reversible",
        "Report uncertainty and failure honestly",
        "Never ask for, reveal, guess or infer secrets",
    ):
        assert phrase in prompt


def test_facts_and_skills_blocks_are_appended(config: Config) -> None:
    prompt = build_system_prompt(
        config, TOOLS, facts_block="FACTS HERE", skills_block="SKILLS HERE"
    )
    assert "FACTS HERE" in prompt and "SKILLS HERE" in prompt


def test_empty_blocks_are_omitted(config: Config) -> None:
    prompt = build_system_prompt(config, TOOLS, facts_block="", skills_block="")
    assert "\n\n\n" not in prompt


def test_hints_are_concrete() -> None:
    assert "my goal" in build_planning_hint("my goal")
    replan = build_replan_hint("read_file failed", 2)
    assert "2 attempt(s) left" in replan and "Do not repeat the identical call" in replan
    assert "verify_result" in build_verification_hint("write_file on step 1")
