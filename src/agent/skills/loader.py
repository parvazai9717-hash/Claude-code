"""Skill discovery from the filesystem."""

from __future__ import annotations

from pathlib import Path

from ..errors import SkillError
from .manifest import SkillManifest, parse_skill_markdown

SKILL_FILENAME = "SKILL.md"

#: A skill directory is not allowed to be arbitrarily deep or large.
MAX_SKILL_FILE_BYTES = 200_000


def discover_skills(root: Path) -> tuple[list[SkillManifest], list[tuple[str, str]]]:
    """Find every skill under `root`.

    A malformed skill never aborts discovery: it is reported so the CLI can show
    it, while the valid skills stay usable.

    Returns:
        `(manifests, problems)` where each problem is `(directory_name, reason)`.
    """
    root = Path(root).expanduser()
    manifests: list[SkillManifest] = []
    problems: list[tuple[str, str]] = []
    if not root.is_dir():
        return manifests, problems

    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        skill_file = entry / SKILL_FILENAME
        if not skill_file.is_file():
            problems.append((entry.name, f"no {SKILL_FILENAME} in this directory"))
            continue
        try:
            if skill_file.stat().st_size > MAX_SKILL_FILE_BYTES:
                problems.append((entry.name, f"{SKILL_FILENAME} is too large to load"))
                continue
            manifest = parse_skill_markdown(skill_file.read_text(encoding="utf-8"), path=skill_file)
        except SkillError as exc:
            problems.append((entry.name, exc.message))
            continue
        except (OSError, UnicodeDecodeError) as exc:
            problems.append((entry.name, f"could not read {SKILL_FILENAME}: {exc}"))
            continue
        if manifest.name != entry.name:
            problems.append(
                (entry.name, f"declared name {manifest.name!r} does not match the directory name")
            )
            continue
        manifests.append(manifest)
    return manifests, problems
