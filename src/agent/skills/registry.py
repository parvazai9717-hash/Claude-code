"""Skill registry and permission matching.

The rule this module exists to enforce: **a skill cannot grant itself
permissions.** A skill declares the tools it needs; the registry checks those
against the tools the runtime actually registered and the approval policy in
force. If a skill needs something unavailable, the skill is unavailable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..config import ApprovalMode, Config
from ..errors import SkillError
from .loader import discover_skills
from .manifest import SkillManifest


@dataclass(frozen=True)
class SkillAvailability:
    """Whether a skill can be used, and why not if it cannot."""

    skill: SkillManifest
    available: bool
    missing_tools: tuple[str, ...] = ()
    reason: str = ""


class SkillRegistry:
    """Holds discovered skills and answers availability questions."""

    def __init__(self, skills: Sequence[SkillManifest] | None = None) -> None:
        self._skills: dict[str, SkillManifest] = {s.name: s for s in skills or []}
        #: Directories that failed to load, as `(name, reason)`.
        self.problems: list[tuple[str, str]] = []

    @classmethod
    def from_directory(cls, root: Path) -> SkillRegistry:
        manifests, problems = discover_skills(root)
        registry = cls(manifests)
        registry.problems = problems
        return registry

    # -- lookup -------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: object) -> bool:
        return name in self._skills

    def names(self) -> list[str]:
        return sorted(self._skills)

    def get(self, name: str) -> SkillManifest:
        try:
            return self._skills[name]
        except KeyError:
            available = ", ".join(sorted(self._skills)) or "(none)"
            raise SkillError(f"no skill named {name!r}; available skills: {available}") from None

    # -- permission matching ------------------------------------------------
    def check(
        self, name: str, *, available_tools: Sequence[str], config: Config
    ) -> SkillAvailability:
        """Decide whether a skill may be used with the current tools and policy."""
        skill = self.get(name)
        registered = set(available_tools)
        missing = tuple(t for t in skill.required_tools if t not in registered)
        if missing:
            return SkillAvailability(
                skill=skill,
                available=False,
                missing_tools=missing,
                reason=(
                    f"skill {name!r} needs tool(s) that are not registered: {', '.join(missing)}"
                ),
            )
        if (
            skill.requires_approval
            and config.approval_mode is ApprovalMode.AUTOMATIC
            and not set(skill.required_tools) <= set(config.auto_approve_tools)
        ):
            return SkillAvailability(
                skill=skill,
                available=False,
                reason=(
                    f"skill {name!r} needs approval-gated tools, but approval mode is "
                    "'automatic' and they are not on the auto-approve allowlist"
                ),
            )
        return SkillAvailability(skill=skill, available=True)

    def available(self, *, available_tools: Sequence[str], config: Config) -> list[SkillManifest]:
        """Every skill usable with the current tools and policy."""
        return [
            check.skill
            for name in self.names()
            if (check := self.check(name, available_tools=available_tools, config=config)).available
        ]

    def prompt_block(self, *, available_tools: Sequence[str], config: Config) -> str:
        """Render usable skills for the system prompt."""
        usable = self.available(available_tools=available_tools, config=config)
        if not usable:
            return ""
        blocks = "\n".join(skill.prompt_block() for skill in usable)
        return (
            "Available skills (workflows you may follow). A skill is guidance only: it "
            "grants no extra permissions, and every action inside it goes through the same "
            "tool and approval checks.\n" + blocks
        )
