"""Outcomes of the host's admission checks (Section 3.2 of the paper).

Validation: the environment builds, the reference solution receives full reward, an empty
solution receives less than half, no instruction shares a 13-gram with a held-out task, and
the task passes Harbor's rubric review (`harbor check`).

Calibration: the current solver attempts the environment several times; it is kept if the
mean reward is in [0.25, 0.75] and the rewards vary (std >= 0.1).
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field

# Check names, in the order the host runs them.
BUILD = "build"
REFERENCE = "reference_solution"
EMPTY = "empty_solution"
CONTAMINATION = "contamination"
RUBRIC = "rubric_review"


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    data: dict = field(default_factory=dict)


@dataclass
class ValidationReport:
    task_id: str
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)

    @property
    def infra_error(self) -> bool:
        """True if a check could not run (sandbox or reviewer failure), as opposed to failing."""
        return any(c.data.get("infra_error") for c in self.checks)

    @property
    def failed_checks(self) -> list[str]:
        return [c.name for c in self.checks if not c.passed]

    def to_dict(self) -> dict:
        return {"task_id": self.task_id, "passed": self.passed, "checks": [asdict(c) for c in self.checks]}


@dataclass
class CalibrationReport:
    task_id: str
    rewards: list[float] = field(default_factory=list)
    band_low: float = 0.25
    band_high: float = 0.75
    min_std: float = 0.1
    hint_probe_reward: float | None = None  # diagnostic rollout with reference hints, below band only

    @property
    def mean(self) -> float:
        return statistics.fmean(self.rewards) if self.rewards else 0.0

    @property
    def std(self) -> float:
        return statistics.pstdev(self.rewards) if len(self.rewards) > 1 else 0.0

    @property
    def in_band(self) -> bool:
        return self.band_low <= self.mean <= self.band_high

    @property
    def passed(self) -> bool:
        return bool(self.rewards) and self.in_band and self.std >= self.min_std

    @property
    def verdict(self) -> str:
        if not self.rewards:
            return "not_run"
        if self.mean < self.band_low:
            return "too_hard"
        if self.mean > self.band_high:
            return "too_easy"
        if self.std < self.min_std:
            return "no_variance"
        return "admitted"

    def to_dict(self) -> dict:
        return {**asdict(self), "mean": self.mean, "std": self.std, "passed": self.passed,
                "verdict": self.verdict}
