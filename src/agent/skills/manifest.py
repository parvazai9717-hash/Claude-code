"""Skill manifests.

A skill is a *recommendation*, not a permission. A `SKILL.md` file declares what
a workflow needs; the registry then checks those needs against what is globally
permitted. A skill that asks for a tool the runtime does not have simply becomes
unavailable — it never causes the tool to be granted.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

from ..errors import SkillError

#: `SKILL.md` starts with a YAML front-matter block delimited by `---`.
FRONT_MATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)

NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{1,63}$")


class SkillManifest(BaseModel):
    """The validated metadata of one skill."""

    name: str
    description: str
    #: Free-text conditions describing when the skill applies.
    activation: list[str] = Field(default_factory=list)
    #: Tool names the workflow needs. Must all be globally permitted.
    required_tools: list[str] = Field(default_factory=list)
    #: Workspace-relative paths the skill expects to work within.
    allowed_paths: list[str] = Field(default_factory=list)
    #: Domains the skill would need, if network access is ever enabled.
    allowed_domains: list[str] = Field(default_factory=list)
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    #: Whether following this workflow involves actions needing approval.
    requires_approval: bool = True
    #: Stated limitations. Rendered to the model so it does not over-trust the skill.
    limitations: list[str] = Field(default_factory=list)

    #: Filled in by the loader.
    path: Path | None = None
    body: str = ""

    @field_validator("name")
    @classmethod
    def _check_name(cls, value: str) -> str:
        if not NAME_PATTERN.match(value):
            raise ValueError(f"skill name must match {NAME_PATTERN.pattern}")
        return value

    @field_validator(
        "activation",
        "required_tools",
        "allowed_paths",
        "allowed_domains",
        "inputs",
        "outputs",
        "limitations",
        mode="before",
    )
    @classmethod
    def _listify(cls, value: Any) -> Any:
        """Accept a single string where a list is expected — a common authoring slip."""
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value

    def prompt_block(self) -> str:
        """A compact description for the model's context."""
        lines = [f"- {self.name}: {self.description}"]
        if self.activation:
            lines.append(f"  use when: {'; '.join(self.activation)}")
        if self.required_tools:
            lines.append(f"  needs tools: {', '.join(self.required_tools)}")
        if self.limitations:
            lines.append(f"  limitations: {'; '.join(self.limitations)}")
        return "\n".join(lines)


def parse_skill_markdown(text: str, *, path: Path | None = None) -> SkillManifest:
    """Parse a `SKILL.md` file into a validated manifest.

    Raises:
        SkillError: When the front matter is missing, unparseable or invalid.
    """
    match = FRONT_MATTER.match(text.lstrip("﻿"))
    if match is None:
        raise SkillError(
            "SKILL.md must begin with a YAML front-matter block delimited by '---' lines",
            details={"path": str(path) if path else ""},
        )
    raw_front, body = match.group(1), match.group(2)
    try:
        data = yaml.safe_load(raw_front)
    except yaml.YAMLError as exc:
        raise SkillError(f"the skill front matter is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise SkillError("the skill front matter must be a YAML mapping")

    try:
        manifest = SkillManifest(**data)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise SkillError(f"invalid skill metadata: {details}") from exc

    manifest.path = path
    manifest.body = body.strip()
    return manifest
