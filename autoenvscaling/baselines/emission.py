"""How a tool-free model returns a task: every file in a ``<file path="...">`` block.

The Prompt and Prompt + Env Feedback baselines cannot write files themselves, so the model emits
each file of the Harbor task as::

    <file path="environment/Dockerfile">
    FROM ubuntu:24.04
    ...
    </file>

``parse_files`` extracts the blocks and ``materialize`` writes them under a task directory. Paths
must be relative and stay inside the task directory. A later block for the same path replaces an
earlier one, which lets a repair round emit only the files it changes. The original parser for
these baselines (``flat_parse.py``) is not in the source archive; this format is a reconstruction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

FORMAT_INSTRUCTIONS = """Return the task as files. Write each file in full inside its own block:

<file path="relative/path/in/the/task">
file contents
</file>

Paths are relative to the task directory (for example `task.toml`, `environment/Dockerfile`,
`tests/test_outputs.py`). Do not wrap the blocks in Markdown code fences. Text outside the blocks is
ignored."""

_BLOCK = re.compile(r'<file\s+path\s*=\s*"([^"]+)"\s*>\n?(.*?)</file>', re.DOTALL)
# Scripts that must be executable once written.
_EXECUTABLE_SUFFIXES = (".sh",)


@dataclass
class ParseResult:
    files: dict[str, str] = field(default_factory=dict)
    rejected: list[str] = field(default_factory=list)  # unsafe or invalid paths

    @property
    def ok(self) -> bool:
        return bool(self.files)


def parse_files(reply: str) -> ParseResult:
    result = ParseResult()
    for raw_path, content in _BLOCK.findall(reply or ""):
        path = _clean(raw_path)
        if path is None:
            result.rejected.append(raw_path)
            continue
        result.files[path] = content
    return result


def _clean(raw: str) -> str | None:
    text = raw.strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    if not text or text.startswith("/") or ":" in text.split("/")[0]:
        return None
    parts = PurePosixPath(text).parts
    if any(p in ("..", "") for p in parts) or parts[0] == ".git":
        return None
    return PurePosixPath(*parts).as_posix()


def materialize(reply: str, task_dir: Path) -> ParseResult:
    """Write every file block in ``reply`` under ``task_dir`` (existing files are overwritten)."""
    result = parse_files(reply)
    task_dir = Path(task_dir)
    root = task_dir.resolve()
    for rel, content in result.files.items():
        target = task_dir / rel
        if not target.resolve().is_relative_to(root):
            result.rejected.append(rel)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        if target.suffix in _EXECUTABLE_SUFFIXES:
            target.chmod(0o755)
    for rel in result.rejected:
        result.files.pop(rel, None)
    return result
