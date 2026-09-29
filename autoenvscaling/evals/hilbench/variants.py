"""Derive the three HiL-Bench variants of a task from one authored task and its registry.

HiL-Bench ships every task in three instruction settings over one shared environment::

    <task>/full_info/    the complete instruction
    <task>/ask_human/    the underspecified instruction; the ask_human tool is available
    <task>/baseline/     the same underspecified instruction; no tool (the control)
    <task>/shared/ask-human-data/blocker_registry.json

On the real benchmark, ``ask_human/instruction.md`` and ``baseline/instruction.md`` are
byte-identical (tool availability is the only difference), and ``full_info/instruction.md`` is
the underspecified instruction followed by a ``## BLOCKER DETAILS`` section that lists each
blocker's description and resolution. So the proposer authors one underspecified task plus the
registry, and the other two variants are string operations with no model call. This is how the
paper's 500 generated training tasks were turned into HiL-Bench's format.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from autoenvscaling.evals.hilbench.registry import (
    REGISTRY_RELPATH,
    Blocker,
    blocker_warnings,
    load_registry,
    registry_path,
    validate_task,
)

VARIANTS = ("full_info", "ask_human", "baseline")
COPIED_ITEMS = ("environment", "solution", "tests", "task.toml")


def blocker_details(blockers: list[Blocker]) -> str:
    """The appendix that turns the underspecified instruction into the full_info one.

    Byte-compatible with HiL-Bench: the underspecified instruction already ends in a newline, so
    one more opens the blank line before the header; the header is followed directly by the
    first ``###``; blockers are separated by a blank line.
    """
    body = "\n".join(f"### {b.description.strip()}\n{b.resolution.strip()}\n" for b in blockers)
    return "\n## BLOCKER DETAILS\n" + body


def full_info_instruction(underspecified: str, blockers: list[Blocker]) -> str:
    return underspecified + blocker_details(blockers)


def derive_variants(task_dir: Path, out_dir: Path | None = None) -> dict[str, Any]:
    """Write full_info/, ask_human/, baseline/ and shared/ for one authored task.

    ``out_dir`` defaults to ``<task_dir>/variants``. Returns a summary with structural errors,
    reviewer warnings, and whether the variants were written. Variants are not written when the
    registry or the instruction is unusable.
    """
    task_dir = Path(task_dir)
    out_dir = Path(out_dir) if out_dir is not None else task_dir / "variants"
    errors = validate_task(task_dir)
    summary: dict[str, Any] = {"task": task_dir.name, "errors": errors, "warnings": [],
                               "derived": False}
    fatal = [e for e in errors if e.startswith(("missing ask-human-data", "registry",
                                                 "missing instruction.md"))]
    if fatal:
        return summary

    blockers = load_registry(registry_path(task_dir))
    underspecified = (task_dir / "instruction.md").read_text()
    summary["n_blockers"] = len(blockers)
    summary["types"] = [b.type for b in blockers]
    summary["warnings"] = blocker_warnings(underspecified, blockers)

    for variant in VARIANTS:
        target = out_dir / variant
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        for item in COPIED_ITEMS:
            source = task_dir / item
            if source.is_dir():
                shutil.copytree(source, target / item)
            elif source.is_file():
                shutil.copy2(source, target / item)
        text = (full_info_instruction(underspecified, blockers) if variant == "full_info"
                else underspecified)
        (target / "instruction.md").write_text(text)

    shared = out_dir / "shared" / REGISTRY_RELPATH.parent
    shared.mkdir(parents=True, exist_ok=True)
    shutil.copy2(registry_path(task_dir), shared / REGISTRY_RELPATH.name)
    summary["derived"] = True
    summary["variants_dir"] = str(out_dir)
    return summary
