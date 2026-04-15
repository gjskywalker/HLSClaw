from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class SkillProperties:
    name: str
    description: str


def _read_skill_md(skill_dir: Path) -> str:
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        raise FileNotFoundError(f"Missing SKILL.md in {skill_dir}")
    return skill_md.read_text(encoding="utf-8")


def _parse_front_matter(text: str) -> dict[str, str]:
    if not text.startswith("---"):
        return {}

    lines = text.splitlines()
    data: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def read_properties(skill_dir: str | Path) -> SkillProperties:
    path = Path(skill_dir)
    text = _read_skill_md(path)
    front_matter = _parse_front_matter(text)
    return SkillProperties(
        name=front_matter.get("name", path.name),
        description=front_matter.get("description", ""),
    )


def validate(skill_dir: str | Path) -> list[str]:
    path = Path(skill_dir)
    problems: list[str] = []
    if not path.is_dir():
        return [f"Skill path is not a directory: {path}"]

    try:
        props = read_properties(path)
    except FileNotFoundError as exc:
        return [str(exc)]
    except UnicodeDecodeError as exc:
        return [f"Could not decode SKILL.md: {exc}"]

    if not props.name:
        problems.append("Skill front matter must include a non-empty name")
    if not props.description:
        problems.append("Skill front matter should include a description")
    return problems


def to_prompt(skill_dirs: Iterable[str | Path]) -> str:
    sections: list[str] = []
    for skill_dir in skill_dirs:
        path = Path(skill_dir)
        props = read_properties(path)
        text = _read_skill_md(path).strip()
        sections.append(
            f"<skill name=\"{props.name}\" description=\"{props.description}\">\n"
            f"{text}\n"
            "</skill>"
        )
    return "\n\n".join(sections)
