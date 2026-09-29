"""Blocker registries for HiL-style tasks (Section 4.5).

A HiL task has an instruction that reads as complete but withholds a few details the tests
depend on. The withheld details live in a blocker registry, which the ``ask_human`` tool uses to
answer questions. The registry is a JSON file, ``ask-human-data/blocker_registry.json``, in the
same shape HiL-Bench ships::

    {"blockers": [{"id": "snake_case_id",
                   "type": "missing parameters",
                   "description": "what is underdetermined (never the answer)",
                   "resolution": "the withheld fact, what a human would reply",
                   "example_questions": ["three or more phrasings", "...", "..."]}]}

``validate_task`` catches structural problems only. Whether a withheld fact is really
unrecoverable from the environment is a judgement; the authoritative test is differential (the
no-tool variant must fail and the full-information variant must pass), which costs solver
rollouts.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REGISTRY_RELPATH = Path("ask-human-data") / "blocker_registry.json"

# HiL-Bench's blocker taxonomy (Table 6 of the paper uses five of these).
BLOCKER_TYPES = (
    "question",
    "business info",
    "schema",
    "missing parameters",
    "ambiguous requirements",
    "contradictory requirements",
)
MIN_BLOCKERS, MAX_BLOCKERS = 3, 5
MIN_EXAMPLE_QUESTIONS = 3
REQUIRED_TASK_ITEMS = ("instruction.md", "task.toml", "tests", "solution", "environment")


@dataclass
class Blocker:
    id: str
    type: str
    description: str
    resolution: str
    example_questions: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Blocker:
        return cls(
            id=str(data.get("id") or ""),
            type=str(data.get("type") or ""),
            description=str(data.get("description") or ""),
            resolution=str(data.get("resolution") or ""),
            example_questions=[str(q) for q in data.get("example_questions") or []],
        )


def registry_path(task_dir: Path) -> Path:
    return Path(task_dir) / REGISTRY_RELPATH


def load_registry(path: Path) -> list[Blocker]:
    data = json.loads(Path(path).read_text())
    blockers = data.get("blockers") if isinstance(data, dict) else None
    if not isinstance(blockers, list):
        raise ValueError(f"{path}: registry has no 'blockers' list")  # noqa: TRY004 (bad data file)
    return [Blocker.from_dict(b) for b in blockers if isinstance(b, dict)]


def write_registry(path: Path, blockers: list[Blocker]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"blockers": [asdict(b) for b in blockers]}, indent=2) + "\n")


def validate_blockers(blockers: list[Blocker]) -> list[str]:
    """Structural problems with a registry. An empty list is not a quality judgement."""
    errors: list[str] = []
    if not MIN_BLOCKERS <= len(blockers) <= MAX_BLOCKERS:
        errors.append(f"{len(blockers)} blockers (want {MIN_BLOCKERS}-{MAX_BLOCKERS})")
    seen: set[str] = set()
    for i, b in enumerate(blockers):
        for name in ("id", "type", "description", "resolution"):
            if not getattr(b, name).strip():
                errors.append(f"blocker[{i}] missing {name}")
        if b.id in seen:
            errors.append(f"duplicate blocker id {b.id!r}")
        seen.add(b.id)
        if len(b.example_questions) < MIN_EXAMPLE_QUESTIONS:
            errors.append(f"blocker[{i}] has fewer than {MIN_EXAMPLE_QUESTIONS} example_questions")
        # A description that restates its resolution hands the agent the answer while telling
        # it the answer is missing.
        d, r = b.description.lower(), b.resolution.lower()
        if len(r) > 25 and r[:25] in d:
            errors.append(f"blocker[{i}] description leaks its resolution")
    return errors


def blocker_warnings(instruction: str, blockers: list[Blocker]) -> list[str]:
    """Things a reviewer should look at first. Not grounds for rejection on their own."""
    warnings = [f"blocker {b.id!r} has type {b.type!r}, outside the HiL-Bench taxonomy"
                for b in blockers if b.type and b.type not in BLOCKER_TYPES]
    for leak in leak_check(instruction, blockers):
        values = ", ".join(repr(h["value"]) for h in leak["leaked"])
        warnings.append(f"blocker {leak['id']!r}: {values} already appear in the instruction "
                        "and the resolution adds no new value")
    return warnings


def validate_task(task_dir: Path) -> list[str]:
    """Structural problems with an authored HiL task directory and its registry."""
    task_dir = Path(task_dir)
    errors = [f"missing {item}" for item in REQUIRED_TASK_ITEMS if not (task_dir / item).exists()]
    path = registry_path(task_dir)
    if not path.is_file():
        return errors + [f"missing {REGISTRY_RELPATH}"]
    try:
        blockers = load_registry(path)
    except (ValueError, json.JSONDecodeError) as exc:
        return errors + [f"registry is not usable: {exc}"]
    return errors + validate_blockers(blockers)


def leak_check(instruction: str, blockers: list[Blocker]) -> list[dict[str, Any]]:
    """Flag blockers whose answer may still be visible in the underspecified instruction.

    A question has to share vocabulary with the instruction to be askable, so shared words mean
    nothing. What matters is the distinguishing value: numbers of two or more digits, quoted or
    backticked literals, and snake_case or kebab-case identifiers in the resolution. A blocker is
    flagged only if its resolution introduces no such value that the instruction lacks, since a
    resolution routinely restates known values to disambiguate a new one.

    This is a cheap pre-filter tuned to avoid false positives; its silence is not evidence that
    a blocker is sound.
    """
    flags = []
    for b in blockers:
        res = b.resolution.strip()
        if not res:
            continue
        candidates: set[str] = set(re.findall(r"\b\d{2,}\b", res))
        for pattern in (r"`([^`]{4,60})`", r"'([^']{4,60})'", r'"([^"]{4,60})"'):
            candidates |= set(re.findall(pattern, res))
        candidates |= set(re.findall(r"\b[A-Za-z][A-Za-z0-9]*(?:[_-][A-Za-z0-9]+)+\b", res))
        if not candidates:
            continue
        hits, novel = [], []
        for token in sorted(candidates):
            i = instruction.find(token)
            if i < 0:
                novel.append(token)
                continue
            context = instruction[max(0, i - 60):i + len(token) + 60].replace("\n", " ")
            hits.append({"value": token, "context": context})
        if hits and not novel:
            flags.append({"id": b.id, "leaked": hits[:4],
                          "note": "resolution introduces no value absent from the instruction"})
    return flags
