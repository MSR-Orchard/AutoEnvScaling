"""The solver's training pool (Section 3.3).

Admitted environments form the pool. At each refresh the host reruns the updated solver on the
pool, removes environments that fall outside the calibration band, and asks the proposer for
replacements based on the latest rollouts.

On disk a pool is one directory::

    <root>/pool.jsonl             one PoolEntry per line, rewritten atomically on every save
    <root>/tasks/<task_id>/       a copy of each admitted Harbor task
    <root>/rollouts/<task_id>.jsonl   solver rollouts recorded on the environment, newest last
    <root>/events.jsonl           append-only log of admissions, removals and promotions

An entry's `task_id` is the content digest of its task directory (`HarborTask.digest`), so the
same environment admitted twice is stored once and a revised environment gets a new identity.

Entries have one of three states. `active` entries are what the solver trains on, at most
`capacity` of them. Admissions that arrive while the active pool is full wait in `reserve` and
are promoted, after a fresh check against the current solver, when slots open. `retired`
entries left the pool at a review and are kept for the record.
"""

from __future__ import annotations

import asyncio
import json
import random
import shutil
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from autoenvscaling.config import CalibrationSettings
from autoenvscaling.proposer.assignment import build_assignment, most_common_failure
from autoenvscaling.results import CalibrationReport
from autoenvscaling.tasks import HarborTask
from autoenvscaling.trajectories.export import read_jsonl, solver_prompt_row, write_jsonl_atomic
from autoenvscaling.trajectories.schema import (
    Assignment,
    ProposerTrajectory,
    SolverTrajectory,
    Turn,
)

ACTIVE = "active"
RESERVE = "reserve"
RETIRED = "retired"

POOL_FILE = "pool.jsonl"
EVENTS_FILE = "events.jsonl"

TOO_EASY = "too_easy"
TOO_HARD = "too_hard"


@dataclass
class PoolEntry:
    task_id: str                          # content digest of the task directory
    task_dir: str                         # relative to the pool root, or absolute
    status: str = ACTIVE
    source: str = ""                      # e.g. "proposer:<model>", "tmax", "seed"
    added_step: int = 0
    assignment_id: str | None = None
    parent_task_id: str | None = None
    trajectory_id: str | None = None
    calibration: dict[str, Any] | None = None   # CalibrationReport.to_dict() at admission
    reviews: list[dict[str, Any]] = field(default_factory=list)
    exposures: int = 0                    # solver batches this environment appeared in
    retired_step: int | None = None
    retired_reason: str | None = None

    @property
    def latest_mean(self) -> float | None:
        if self.reviews:
            return self.reviews[-1]["mean"]
        return None if self.calibration is None else self.calibration.get("mean")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PoolEntry:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class ReplacementRequest:
    """What the host asks the proposer for after removing an environment."""
    parent_task_id: str
    parent_task_dir: str
    solver_pass_rate: float
    verdict: str                  # the calibration verdict that removed the parent
    common_failure: str = ""

    @property
    def direction(self) -> str:
        """A harder variant when the solver succeeds, a simpler one when it keeps failing."""
        if self.verdict == TOO_EASY:
            return "harder"
        if self.verdict == TOO_HARD:
            return "simpler"
        return "auto"

    def to_assignment(self, assignment_id: str, rollouts: Sequence[SolverTrajectory] = (),
                      domain: str = "terminal") -> Assignment:
        """An assignment for the proposer, built from the solver's latest rollouts on the parent.

        Pass rate, common failure and reward statistics come from
        `proposer.assignment.build_assignment`. When the parent was removed for being too easy
        or too hard, that verdict sets the direction (harder or simpler); otherwise the
        builder's pass-rate rule does.
        """
        rollouts = [t for t in rollouts if t.task_id == self.parent_task_id]
        if rollouts:
            assignment = build_assignment(self.parent_task_id, rollouts,
                                          assignment_id=assignment_id, domain=domain)
        else:
            assignment = Assignment(assignment_id=assignment_id, parent_task_id=self.parent_task_id,
                                    solver_pass_rate=self.solver_pass_rate,
                                    common_failure=self.common_failure, domain=domain)
        if self.direction != "auto":
            assignment.direction = self.direction
        assignment.extra = {**assignment.extra, "parent_task_dir": self.parent_task_dir,
                            "removal_verdict": self.verdict}
        return assignment


@dataclass
class ReviewOutcome:
    step: int
    kept: list[str] = field(default_factory=list)
    removed: dict[str, str] = field(default_factory=dict)     # task_id -> verdict
    promoted: list[str] = field(default_factory=list)
    requests: list[ReplacementRequest] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"step": self.step, "kept": len(self.kept), "removed": self.removed,
                "promoted": self.promoted, "requests": [asdict(r) for r in self.requests]}


# A rerun takes one entry and returns fresh solver attempts on it.
Rerun = Callable[["PoolEntry"], Awaitable[list[SolverTrajectory]]]


class Pool:
    """A persisted pool of Harbor tasks. Call `save()` after mutating it."""

    def __init__(self, root: Path, capacity: int = 256):
        self.root = Path(root)
        self.capacity = capacity
        self.entries: dict[str, PoolEntry] = {}

    # ---- persistence ---------------------------------------------------------------------
    @classmethod
    def load(cls, root: Path, capacity: int = 256) -> Pool:
        pool = cls(root, capacity)
        path = pool.root / POOL_FILE
        if path.exists():
            for row in read_jsonl(path):
                entry = PoolEntry.from_dict(row)
                pool.entries[entry.task_id] = entry
        return pool

    def save(self) -> None:
        write_jsonl_atomic(self.root / POOL_FILE, (asdict(e) for e in self.entries.values()))

    def _log(self, event: str, **data: Any) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.root / EVENTS_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"time": time.time(), "event": event, **data}) + "\n")

    # ---- views ---------------------------------------------------------------------------
    def with_status(self, status: str) -> list[PoolEntry]:
        return [e for e in self.entries.values() if e.status == status]

    def active(self) -> list[PoolEntry]:
        return self.with_status(ACTIVE)

    def reserve(self) -> list[PoolEntry]:
        return self.with_status(RESERVE)

    @property
    def open_slots(self) -> int:
        return max(0, self.capacity - len(self.active()))

    def task_path(self, entry: PoolEntry) -> Path:
        path = Path(entry.task_dir)
        return path if path.is_absolute() else self.root / path

    def task(self, entry: PoolEntry) -> HarborTask:
        return HarborTask(self.task_path(entry))

    def summary(self) -> dict[str, Any]:
        counts = Counter(e.status for e in self.entries.values())
        means = [m for e in self.active() if (m := e.latest_mean) is not None]
        return {"active": counts[ACTIVE], "reserve": counts[RESERVE], "retired": counts[RETIRED],
                "capacity": self.capacity,
                "active_mean_reward": sum(means) / len(means) if means else None}

    # ---- admission -----------------------------------------------------------------------
    def admit(self, task: HarborTask, *, step: int = 0, source: str = "",
              trajectory: ProposerTrajectory | None = None,
              calibration: CalibrationReport | dict[str, Any] | None = None,
              copy: bool = True) -> PoolEntry:
        """Add an admitted environment. It is active if there is room, otherwise in reserve.

        With `copy=True` the task directory is copied under the pool root, so the pool does not
        depend on the episode's output directory. Admitting the same content twice returns the
        existing entry.
        """
        task_id = task.digest()
        if task_id in self.entries:
            return self.entries[task_id]
        if copy:
            dest = self.root / "tasks" / task_id
            if not dest.exists():
                shutil.copytree(task.path, dest)
            task_dir = str(dest.relative_to(self.root))
        else:
            task_dir = str(Path(task.path).resolve())
        if isinstance(calibration, CalibrationReport):
            calibration = calibration.to_dict()
        if calibration is None and trajectory is not None:
            calibration = trajectory.calibration
        entry = PoolEntry(
            task_id=task_id, task_dir=task_dir, status=ACTIVE if self.open_slots else RESERVE,
            source=source, added_step=step, calibration=calibration,
            assignment_id=trajectory.assignment.assignment_id if trajectory else None,
            parent_task_id=trajectory.assignment.parent_task_id if trajectory else None,
            trajectory_id=trajectory.trajectory_id if trajectory else None)
        self.entries[task_id] = entry
        self._log("admit", task_id=task_id, status=entry.status, step=step, source=source)
        return entry

    def seed(self, task_dirs: Iterable[Path], *, source: str = "seed", copy: bool = True) -> list[PoolEntry]:
        """Admit existing Harbor tasks, e.g. a frontier-proposer pool or the Tmax baseline pool."""
        entries = []
        for path in task_dirs:
            task = HarborTask(Path(path))
            missing = task.missing_files()
            if missing:
                raise ValueError(f"{path} is not a complete Harbor task (missing {missing})")
            entries.append(self.admit(task, source=source, copy=copy))
        return entries

    # ---- sampling and rollouts ------------------------------------------------------------
    def sample(self, n: int, rng: random.Random, exclude: Iterable[str] = ()) -> list[PoolEntry]:
        """Up to `n` distinct active entries, uniformly at random, skipping `exclude`."""
        skip = set(exclude)
        candidates = sorted((e for e in self.active() if e.task_id not in skip), key=lambda e: e.task_id)
        return rng.sample(candidates, min(n, len(candidates)))

    def record_rollouts(self, task_id: str, trajectories: Sequence[SolverTrajectory],
                        step: int, *, count_exposure: bool = True) -> None:
        """Append solver attempts on an environment. They feed the proposer's next assignments."""
        path = self.root / "rollouts" / f"{task_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            for traj in trajectories:
                row = json.loads(traj.to_json())
                row["step"] = step
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        if count_exposure and task_id in self.entries:
            self.entries[task_id].exposures += 1

    def latest_rollouts(self, task_id: str, limit: int = 16) -> list[dict[str, Any]]:
        path = self.root / "rollouts" / f"{task_id}.jsonl"
        return read_jsonl(path)[-limit:] if path.exists() else []

    def latest_rollout_trajectories(self, task_id: str, limit: int = 16) -> list[SolverTrajectory]:
        known = SolverTrajectory.__dataclass_fields__
        out = []
        for row in self.latest_rollouts(task_id, limit):
            row = {k: v for k, v in row.items() if k in known}
            row["turns"] = [Turn(**turn) for turn in row.get("turns", [])]
            out.append(SolverTrajectory(**row))
        return out

    # ---- review ----------------------------------------------------------------------------
    def apply_review(self, reports: dict[str, CalibrationReport], step: int, *,
                     require_variance: bool = True) -> ReviewOutcome:
        """Bookkeeping for one review, given fresh calibration reports keyed by task id.

        Active entries outside the band (or without reward variance, if `require_variance`) are
        retired and each yields a replacement request. Reserve entries that are now in band are
        promoted into the freed slots, oldest admission first; reserve entries that are out of
        band are retired without a replacement request. Entries with no report are kept as is.
        """
        outcome = ReviewOutcome(step=step)

        def verdict_ok(report: CalibrationReport) -> bool:
            return report.passed if require_variance else (bool(report.rewards) and report.in_band)

        for task_id, report in reports.items():
            entry = self.entries.get(task_id)
            if entry is None or entry.status == RETIRED or not report.rewards:
                continue
            entry.reviews.append({"step": step, "mean": report.mean, "std": report.std,
                                  "verdict": report.verdict, "rewards": list(report.rewards)})
            if verdict_ok(report):
                if entry.status == ACTIVE:
                    outcome.kept.append(task_id)
                continue
            was_active = entry.status == ACTIVE
            self._retire(entry, step, report.verdict)
            if was_active:
                outcome.removed[task_id] = report.verdict
                outcome.requests.append(ReplacementRequest(
                    parent_task_id=task_id, parent_task_dir=str(self.task_path(entry)),
                    solver_pass_rate=report.mean, verdict=report.verdict,
                    common_failure=most_common_failure(self.latest_rollout_trajectories(task_id))))
        outcome.promoted = self.promote_reserve(step, reviewed_only=set(reports))
        self._log("review", **outcome.to_dict())
        return outcome

    def _retire(self, entry: PoolEntry, step: int, reason: str) -> None:
        entry.status = RETIRED
        entry.retired_step = step
        entry.retired_reason = reason

    def promote_reserve(self, step: int, reviewed_only: set[str] | None = None) -> list[str]:
        """Move reserve entries into open active slots, oldest admission first.

        With `reviewed_only`, only reserve entries in that set (those that just passed a review)
        are eligible.
        """
        promoted = []
        waiting = sorted(self.reserve(), key=lambda e: (e.added_step, e.task_id))
        for entry in waiting:
            if not self.open_slots:
                break
            if reviewed_only is not None and entry.task_id not in reviewed_only:
                continue
            entry.status = ACTIVE
            promoted.append(entry.task_id)
            self._log("promote", task_id=entry.task_id, step=step)
        return promoted

    async def review(self, rerun: Rerun, step: int, *, calibration: CalibrationSettings,
                     concurrency: int = 8, include_reserve: bool = True,
                     require_variance: bool = True) -> ReviewOutcome:
        """Rerun the current solver on every active (and reserve) entry and apply the verdicts.

        The fresh attempts are recorded as rollouts, so the replacement requests and the next
        proposer workspace see the latest solver behaviour.
        """
        entries = self.active() + (self.reserve() if include_reserve else [])
        gate = asyncio.Semaphore(concurrency)

        async def one(entry: PoolEntry) -> tuple[str, list[SolverTrajectory]]:
            async with gate:
                return entry.task_id, await rerun(entry)

        results = await asyncio.gather(*(one(e) for e in entries))
        reports: dict[str, CalibrationReport] = {}
        for task_id, trajs in results:
            self.record_rollouts(task_id, trajs, step, count_exposure=False)
            rewards = [t.reward for t in trajs if t.reward is not None]
            reports[task_id] = CalibrationReport(task_id, rewards, band_low=calibration.band_low,
                                                 band_high=calibration.band_high,
                                                 min_std=calibration.min_reward_std)
        return self.apply_review(reports, step, require_variance=require_variance)

    # ---- export --------------------------------------------------------------------------
    def export_prompt_data(self, path: Path, images: dict[str, str] | None = None) -> int:
        """Write the active pool as Miles prompt rows (atomically). Returns the row count.

        ``images`` maps task ids to built environment images; carry mode starts each episode's
        sandbox from ``metadata["image"]`` and refuses rows without one.
        """
        rows = []
        for entry in sorted(self.active(), key=lambda e: e.task_id):
            task = self.task(entry)
            metadata = {"source": entry.source}
            if images and entry.task_id in images:
                metadata["image"] = images[entry.task_id]
            rows.append(solver_prompt_row(task, task_id=entry.task_id, metadata=metadata))
        return write_jsonl_atomic(path, rows)
