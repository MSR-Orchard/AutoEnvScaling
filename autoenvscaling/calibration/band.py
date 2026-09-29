"""Solver calibration: keep an environment the current solver sometimes solves.

The rule (Appendix, "Solver-Guided Calibration"): run ``initial_rollouts`` (4) solver attempts;
if their mean reward is within ``boundary_margin`` (0.15) of either band edge, add attempts up to
``max_rollouts`` (8). The environment is kept if the mean reward is in [``band_low``,
``band_high``] = [0.25, 0.75] and the reward standard deviation is at least ``min_reward_std``
(0.1). Additional attempts extend the sample; completed attempts are never redrawn.

An environment below the band gets one diagnostic attempt with hints from the reference solution
(``hint_probe_below_band``). Its reward is not part of the calibration sample; it tells the
proposer whether the task is fair but hard (the hints help) or underdetermined (they do not).

Attempts that never validly ran (endpoint or sandbox failures) carry no reward and are left out of
the sample. Each stage tops up such attempts once; if no attempt produced a reward at all,
``SolverUnavailableError`` is raised instead of reporting the task as too hard.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from autoenvscaling.calibration.solver import SolverEndpoint, reference_hints, run_solver
from autoenvscaling.config import CalibrationSettings, Settings
from autoenvscaling.results import CalibrationReport
from autoenvscaling.tasks import HarborTask
from autoenvscaling.trajectories.schema import SolverTrajectory, write_jsonl
from autoenvscaling.validation.harbor_runner import HarborRunner, TrialRunner

logger = logging.getLogger(__name__)
_EPS = 1e-9


class SolverUnavailableError(RuntimeError):
    """No solver attempt produced a reward, so the task's difficulty is unknown."""


def needs_more_rollouts(mean: float, cal: CalibrationSettings) -> bool:
    """True when the mean is within ``boundary_margin`` of a band edge (either side of it)."""
    distance = min(abs(mean - cal.band_low), abs(mean - cal.band_high))
    return distance <= cal.boundary_margin + _EPS


def calibration_dir(task: HarborTask, settings: Settings) -> Path:
    """Where the Harbor jobs, trajectories and report for one version of a task are kept."""
    return Path(settings.work_dir) / "calibration" / f"{task.task_id}-{task.digest()}"


async def calibrate(task: HarborTask, solver: SolverEndpoint, settings: Settings, *,
                    runner: TrialRunner | None = None) -> CalibrationReport:
    """4 rollouts, expanded to 8 when the mean is within 0.15 of a band edge; hint probe below band.

    Writes ``calibration.json`` and ``trajectories.jsonl`` (every attempt, including the hint
    probe) to ``calibration_dir(task, settings)``. ``runner`` optionally replaces the Harbor runner.
    """
    cal = settings.calibration
    out_dir = calibration_dir(task, settings)
    if runner is None:
        runner = HarborRunner(out_dir / "jobs", env_type=settings.sandbox.harbor_env)

    async def draw(n: int, hints: str | None = None) -> list[SolverTrajectory]:
        return await run_solver(task, solver, n=n, harbor_env=settings.sandbox.harbor_env,
                                hints=hints, runner=runner)

    attempts = await _draw_scored(draw, cal.initial_rollouts)
    rewards = _rewards(attempts)
    if not rewards:
        _write(out_dir, None, attempts, [], solver)
        raise SolverUnavailableError(
            f"none of {len(attempts)} solver attempts on {task.task_id} produced a reward; "
            f"first error: {_first_error(attempts)}")
    expanded = False
    mean = sum(rewards) / len(rewards)
    if needs_more_rollouts(mean, cal) and len(rewards) < cal.max_rollouts:
        expanded = True
        attempts += await _draw_scored(draw, cal.max_rollouts - len(rewards))
        rewards = _rewards(attempts)[:cal.max_rollouts]

    report = CalibrationReport(task_id=task.task_id, rewards=rewards, band_low=cal.band_low,
                               band_high=cal.band_high, min_std=cal.min_reward_std)
    probe: list[SolverTrajectory] = []
    if cal.hint_probe_below_band and report.verdict == "too_hard":
        hints = reference_hints(task)
        if hints is None:
            logger.info("no reference-solution steps to hint for %s", task.task_id)
        else:
            try:
                probe = await draw(1, hints=hints)
            except Exception:
                logger.exception("hint probe failed to run for %s", task.task_id)
            scored = _rewards(probe)
            report.hint_probe_reward = scored[0] if scored else None
    logger.info("calibrated %s: mean=%.3f std=%.3f n=%d expanded=%s verdict=%s", task.task_id,
                report.mean, report.std, len(report.rewards), expanded, report.verdict)
    _write(out_dir, report, attempts, probe, solver, expanded=expanded)
    return report


async def _draw_scored(draw, n: int) -> list[SolverTrajectory]:
    """n attempts; attempts without a reward are topped up once."""
    attempts = await draw(n)
    missing = n - len(_rewards(attempts))
    if missing > 0:
        logger.warning("%d of %d solver attempts produced no reward; retrying them once",
                       missing, n)
        attempts += await draw(missing)
    return attempts


def _rewards(attempts: list[SolverTrajectory]) -> list[float]:
    return [a.reward for a in attempts if a.reward is not None]


def _first_error(attempts: list[SolverTrajectory]) -> str:
    for a in attempts:
        if a.verifier_output:
            return a.verifier_output[-500:]
    return "no attempts ran"


def _write(out_dir: Path, report: CalibrationReport | None, attempts: list[SolverTrajectory],
           probe: list[SolverTrajectory], solver: SolverEndpoint, *,
           expanded: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "report": report.to_dict() if report else None,
        "expanded": expanded,
        "solver_model": solver.model,
        "attempts": [{"trajectory_id": a.trajectory_id, "reward": a.reward,
                      "tests_passed": a.tests_passed, "tests_total": a.tests_total}
                     for a in attempts],
        "hint_probe": [{"trajectory_id": a.trajectory_id, "reward": a.reward} for a in probe],
    }
    (out_dir / "calibration.json").write_text(json.dumps(record, indent=2))
    write_jsonl(out_dir / "trajectories.jsonl", [*attempts, *probe])
