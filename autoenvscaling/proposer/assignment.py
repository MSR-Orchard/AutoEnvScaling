"""Generation assignments built from solver rollouts (Section 3.1 of the paper).

Each proposer episode starts from an assignment that names a parent environment from the
training pool and gives the solver's pass rate and most common failure on it. The proposer then
builds a simpler variant when the solver keeps failing, or a harder one (a longer horizon or an
additional skill) when the solver succeeds.

Definitions used here:

* An attempt's reward is its ``SolverTrajectory.reward`` (the fraction of tests passed), or
  ``tests_passed / tests_total`` when the reward was not filled in. Attempts with neither are
  infrastructure errors and are ignored.
* The pass rate is the fraction of attempts that pass every test (reward 1.0).
* The most common failure is read from the failing attempts' verifier output: the most frequent
  ``FAILED <test>`` lines from pytest, then the most frequent ``N failed, M passed`` summary,
  falling back to the typical number of tests passed.
* Direction: "harder" when the pass rate is at least ``harder_at`` (default 0.5), else "simpler".

Ported from the research code's assignment builder; its archive-specific linking (generation and
admission archives, content-hash joins, absolute paths), its Terminal-Bench string scan (the
host's contamination check covers this), and its hashed "difficulty transformation" vocabulary
(the paper uses only the simpler/harder direction) are not carried over.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

from autoenvscaling.trajectories.schema import Assignment, SolverTrajectory

SIMPLER = "simpler"
HARDER = "harder"
DEFAULT_HARDER_AT = 0.5

_FAILED_LINE = re.compile(r"^FAILED\s+(\S+)")
_SUMMARY_LINE = re.compile(r"^=*\s*(\d+ failed(?:, \d+ passed)?[^=]*?)\s*=*$")


def attempt_reward(trajectory: SolverTrajectory) -> float | None:
    """The attempt's reward in [0, 1], or None when the attempt produced no verdict."""
    if trajectory.reward is not None:
        return float(trajectory.reward)
    if trajectory.tests_total:
        return trajectory.tests_passed / trajectory.tests_total
    return None


def completed(rollouts: Iterable[SolverTrajectory]) -> list[SolverTrajectory]:
    return [t for t in rollouts if attempt_reward(t) is not None]


def is_success(trajectory: SolverTrajectory) -> bool:
    reward = attempt_reward(trajectory)
    return reward is not None and reward >= 1.0


def pass_rate(rollouts: Iterable[SolverTrajectory]) -> float | None:
    done = completed(rollouts)
    if not done:
        return None
    return sum(1 for t in done if is_success(t)) / len(done)


def reward_summary(rollouts: Iterable[SolverTrajectory]) -> dict:
    """Per-environment reward statistics, as written to ``past_rollouts/index.jsonl``."""
    rollouts = list(rollouts)
    rewards = [attempt_reward(t) for t in completed(rollouts)]
    n = len(rewards)
    return {
        "n": n,
        "n_errored": len(rollouts) - n,
        "rewards": [round(r, 4) for r in rewards],
        "mean_reward": round(statistics.fmean(rewards), 4) if n else None,
        "reward_std": round(statistics.pstdev(rewards), 4) if n > 1 else (0.0 if n else None),
        "pass_rate": round(sum(1 for r in rewards if r >= 1.0) / n, 4) if n else None,
    }


def most_common_failure(rollouts: Iterable[SolverTrajectory]) -> str:
    """A short description of how the failing attempts failed ("" if none failed)."""
    failing = [t for t in completed(rollouts) if not is_success(t)]
    if not failing:
        return ""
    failed_tests: Counter[str] = Counter()
    summaries: Counter[str] = Counter()
    for t in failing:
        seen: set[str] = set()
        for line in (t.verifier_output or "").splitlines():
            line = line.strip()
            match = _FAILED_LINE.match(line)
            if match:
                name = match.group(1)
                if name not in seen:  # count each test once per attempt
                    failed_tests[name] += 1
                    seen.add(name)
                continue
            match = _SUMMARY_LINE.match(line)
            if match:
                summaries[match.group(1).strip()] += 1
    parts = []
    if failed_tests:
        top = sorted(failed_tests.items(), key=lambda kv: (-kv[1], kv[0]))[:2]
        parts.append("failing tests: " + ", ".join(f"{name} ({count}/{len(failing)} attempts)"
                                                   for name, count in top))
    if summaries:
        parts.append(min(summaries.items(), key=lambda kv: (-kv[1], kv[0]))[0])
    if not parts:
        with_totals = [t for t in failing if t.tests_total]
        if with_totals:
            passed = statistics.median(t.tests_passed for t in with_totals)
            total = statistics.median(t.tests_total for t in with_totals)
            parts.append(f"typically {passed:g} of {total:g} tests pass")
        else:
            parts.append("verifier failed (no test summary found)")
    return "; ".join(parts)


def choose_direction(solver_pass_rate: float | None, harder_at: float = DEFAULT_HARDER_AT) -> str:
    if solver_pass_rate is None:
        return SIMPLER
    return HARDER if solver_pass_rate >= harder_at else SIMPLER


def assignment_id_for(parent_task_id: str, rollouts: Sequence[SolverTrajectory]) -> str:
    """Deterministic id from the parent and the rollouts that informed the assignment."""
    h = hashlib.sha256(parent_task_id.encode())
    for trajectory_id in sorted(t.trajectory_id for t in rollouts):
        h.update(b"\0" + trajectory_id.encode())
    return f"asg-{h.hexdigest()[:12]}"


def build_assignment(parent_task_id: str, rollouts: Sequence[SolverTrajectory], *,
                     assignment_id: str | None = None, domain: str = "terminal",
                     category: str | None = None,
                     harder_at: float = DEFAULT_HARDER_AT) -> Assignment:
    """One assignment for a parent environment from the solver's rollouts on it."""
    rollouts = [t for t in rollouts if t.task_id == parent_task_id]
    rate = pass_rate(rollouts)
    summary = reward_summary(rollouts)
    extra: dict = {"mean_reward": summary["mean_reward"], "reward_std": summary["reward_std"],
                   "n_attempts": summary["n"], "harder_at": harder_at}
    if category:
        extra["category"] = category
    return Assignment(
        assignment_id=assignment_id or assignment_id_for(parent_task_id, rollouts),
        parent_task_id=parent_task_id,
        solver_pass_rate=round(rate, 4) if rate is not None else 0.0,
        common_failure=most_common_failure(rollouts),
        direction=choose_direction(rate, harder_at),
        domain=domain,
        extra=extra,
    )


def round_robin(task_ids: Iterable[str], categories: Mapping[str, str] | None = None) -> list[str]:
    """Interleave task ids across categories so no category dominates (deterministic)."""
    groups: dict[str, list[str]] = {}
    for task_id in task_ids:
        groups.setdefault((categories or {}).get(task_id, ""), []).append(task_id)
    for ids in groups.values():
        ids.sort()
    ordered: list[str] = []
    for i in range(max((len(v) for v in groups.values()), default=0)):
        for cat in sorted(groups):
            if i < len(groups[cat]):
                ordered.append(groups[cat][i])
    return ordered


def build_assignments(rollouts_by_task: Mapping[str, Sequence[SolverTrajectory]], *,
                      categories: Mapping[str, str] | None = None, n: int | None = None,
                      domain: str = "terminal",
                      harder_at: float = DEFAULT_HARDER_AT) -> list[Assignment]:
    """Assignments for up to n parents, taken round-robin across categories."""
    parents = [t for t in round_robin(rollouts_by_task, categories)
               if completed(rollouts_by_task[t])]
    if n is not None:
        parents = parents[:n]
    return [build_assignment(t, rollouts_by_task[t], domain=domain,
                             category=(categories or {}).get(t), harder_at=harder_at)
            for t in parents]


def render_assignment(assignment: Assignment) -> str:
    """The assignment as the proposer reads it (``assignment/ASSIGNMENT.md``)."""
    rate = f"{assignment.solver_pass_rate:.2f}"
    extra = assignment.extra or {}
    mean = extra.get("mean_reward")
    lines = [
        f"# Assignment {assignment.assignment_id}",
        "",
        (f"- Parent environment: `{assignment.parent_task_id}` "
         f"(`past_environments/{assignment.parent_task_id}/`)"),
        f"- Solver pass rate on the parent: {rate}"
        + (f" (mean reward {mean:.2f} over {extra.get('n_attempts')} attempts)"
           if isinstance(mean, (int, float)) else ""),
        f"- Most common failure: {assignment.common_failure or 'none; the solver passed every attempt'}",
        f"- Domain: {assignment.domain}",
    ]
    if extra.get("category"):
        lines.append(f"- Category: {extra['category']}")
    if assignment.direction == SIMPLER:
        goal = ("The solver keeps failing this environment. Build a simpler variant that "
                "targets the same weakness: fewer compounding steps, a clearer starting state, "
                "or more scaffolding, so that the solver sometimes succeeds.")
    elif assignment.direction == HARDER:
        goal = ("The solver succeeds on this environment. Build a harder variant: a longer "
                "horizon or an additional skill, so that the solver sometimes fails.")
    else:
        goal = ("Decide from the rollouts whether a simpler or a harder variant is needed so "
                "that the solver sometimes succeeds and sometimes fails.")
    lines += ["", f"Direction: **{assignment.direction}**. {goal}", ""]
    return "\n".join(lines)


def save_assignment(assignment: Assignment, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(assignment), indent=2, sort_keys=True) + "\n")


def load_assignment(path: Path) -> Assignment:
    return Assignment(**json.loads(Path(path).read_text()))
