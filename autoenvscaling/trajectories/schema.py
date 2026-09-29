"""Trajectory records for both roles.

Both roles act in a terminal the same way: given a prompt and a container, the policy issues a
command or tool call and reads its output, producing tau = (x, a_1, o_1, ..., a_T, o_T). The roles
differ only in the prompt, the container, and how the trajectory is scored.

A `ProposerTrajectory` is written for every proposer episode. It is the unit of proposer
training: RL groups for self-play (all trajectories for one assignment, scored with r_P) and
the fine-tuning set for the Cold-Start checkpoint (high-quality admitted episodes).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Turn:
    """One message. role is "system", "user", "assistant", or "tool"."""
    role: str
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None


@dataclass
class Assignment:
    """What the proposer is asked to build from (Section 3.1, assignments from solver rollouts)."""
    assignment_id: str
    parent_task_id: str
    solver_pass_rate: float
    common_failure: str = ""
    direction: str = "auto"          # "simpler" when the solver keeps failing, "harder" when it succeeds
    domain: str = "terminal"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProposerTrajectory:
    trajectory_id: str
    assignment: Assignment
    model: str
    harness: str                      # "claude-code", "codex", or "pi"
    turns: list[Turn] = field(default_factory=list)
    task_dir: str | None = None       # the Harbor task the episode ended with, if any
    task_digest: str | None = None
    validation: dict[str, Any] | None = None     # ValidationReport.to_dict()
    calibration: dict[str, Any] | None = None    # CalibrationReport.to_dict()
    reward: float | None = None                  # r_P
    revision: int = 0                            # 0 for the first submission, up to max_revisions
    wall_time_s: float | None = None
    usage: dict[str, Any] = field(default_factory=dict)  # tokens / cost reported by the harness

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProposerTrajectory":
        data = dict(data)
        data["assignment"] = Assignment(**data["assignment"])
        data["turns"] = [Turn(**t) for t in data.get("turns", [])]
        return cls(**data)


@dataclass
class SolverTrajectory:
    trajectory_id: str
    task_id: str
    model: str
    turns: list[Turn] = field(default_factory=list)
    tests_passed: int = 0
    tests_total: int = 0
    reward: float | None = None      # r_S
    verifier_output: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def write_jsonl(path: Path, records: list[ProposerTrajectory | SolverTrajectory]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for rec in records:
            fh.write(rec.to_json() + "\n")


def read_proposer_jsonl(path: Path) -> list[ProposerTrajectory]:
    with open(path) as fh:
        return [ProposerTrajectory.from_dict(json.loads(line)) for line in fh if line.strip()]
