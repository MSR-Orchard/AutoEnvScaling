"""A generated environment is a Harbor task directory (https://www.harborframework.com/docs/tasks).

    <task>/
      task.toml             metadata and resource limits
      instruction.md        what the solver sees
      environment/          build context; environment/Dockerfile builds the container
      solution/solve.sh     the proposer's reference solution (hidden from the solver)
      tests/test.sh         the verifier (hidden from the solver); writes the reward

The solver sees only the instruction and works inside the environment's container.
"""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path

REQUIRED_FILES = ("task.toml", "instruction.md", "environment/Dockerfile", "solution/solve.sh",
                  "tests/test.sh")


@dataclass(frozen=True)
class HarborTask:
    path: Path

    @property
    def task_id(self) -> str:
        return self.path.name

    @property
    def instruction(self) -> str:
        return (self.path / "instruction.md").read_text()

    @property
    def config(self) -> dict:
        with open(self.path / "task.toml", "rb") as fh:
            return tomllib.load(fh)

    def missing_files(self) -> list[str]:
        return [rel for rel in REQUIRED_FILES if not (self.path / rel).is_file()]

    def is_complete(self) -> bool:
        return not self.missing_files()

    def digest(self) -> str:
        """Content hash over every file, so a revised task gets a new identity."""
        h = hashlib.sha256()
        for file in sorted(p for p in self.path.rglob("*") if p.is_file()):
            h.update(str(file.relative_to(self.path)).encode())
            h.update(file.read_bytes())
        return h.hexdigest()[:16]
