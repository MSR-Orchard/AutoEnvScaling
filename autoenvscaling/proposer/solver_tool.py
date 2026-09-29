"""``solver_model``: run the current solver on a candidate environment from inside the sandbox.

Section 3.2 of the paper: the proposer can run the calibration check itself before submitting.
This tool calls the same solver runner the host's calibration uses
(``autoenvscaling.calibration.solver.run_solver``) and reports each attempt's reward, verifier
output and trajectory, plus the mean, spread and band verdict the host would compute.

Usage (inside the proposer sandbox)::

    solver_model output/tasks/my-task            # settings.calibration.initial_rollouts attempts
    solver_model output/tasks/my-task -n 8 --out logs/solver_model/run2.json

Configuration comes from the environment the host forwards: ``AES_SOLVER_BASE_URL``,
``AES_SOLVER_MODEL`` and ``AES_SOLVER_API_KEY`` for the solver endpoint, and the usual ``AES_*``
settings (``AES_SANDBOX_HARBOR_ENV``, ``AES_CALIBRATION_BAND_LOW``, ...).

Exit codes: 0 the attempts ran (whatever the rewards), 2 usage or configuration error, 3 the
solver run failed.

Differences from the research code's in-sandbox calibrator: the solver runner, its agent and its
network staging are owned by ``autoenvscaling.calibration.solver`` rather than reimplemented
here, and the band verdict comes from ``CalibrationReport``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from autoenvscaling.calibration.solver import SolverEndpoint, run_solver
from autoenvscaling.config import Settings
from autoenvscaling.proposer.assignment import attempt_reward
from autoenvscaling.results import CalibrationReport
from autoenvscaling.tasks import HarborTask

VERIFIER_TAIL_CHARS = 2000


def endpoint_from_env(env: dict[str, str] | None = None) -> SolverEndpoint:
    env = dict(os.environ if env is None else env)
    base_url, model = env.get("AES_SOLVER_BASE_URL"), env.get("AES_SOLVER_MODEL")
    if not base_url or not model:
        raise ValueError("solver_model needs AES_SOLVER_BASE_URL and AES_SOLVER_MODEL")
    return SolverEndpoint(base_url=base_url, model=model)


def build_report(task: HarborTask, trajectories: list, settings: Settings,
                 wall_time_s: float) -> dict:
    cal = settings.calibration
    rewards = [r for r in (attempt_reward(t) for t in trajectories) if r is not None]
    report = CalibrationReport(task.task_id, rewards, band_low=cal.band_low,
                               band_high=cal.band_high, min_std=cal.min_reward_std)
    attempts = []
    for t in trajectories:
        attempts.append({
            "trajectory_id": t.trajectory_id,
            "reward": attempt_reward(t),
            "tests_passed": t.tests_passed,
            "tests_total": t.tests_total,
            "n_turns": len(t.turns),
            "verifier_tail": (t.verifier_output or "")[-VERIFIER_TAIL_CHARS:],
            "turns": [asdict(turn) for turn in t.turns],
        })
    return {
        "task_dir": str(task.path),
        "attempts_requested": len(trajectories),
        "attempts_with_reward": len(rewards),
        "rewards": rewards,
        "mean_reward": report.mean if rewards else None,
        "reward_std": report.std if rewards else None,
        "band": [cal.band_low, cal.band_high],
        "min_reward_std": cal.min_reward_std,
        "verdict": report.verdict,
        "wall_time_s": round(wall_time_s, 1),
        "attempts": attempts,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="solver_model", description=__doc__.split("\n\n")[0])
    ap.add_argument("task_dir", help="the Harbor task to try")
    ap.add_argument("-n", type=int, default=None, help="number of solver attempts")
    ap.add_argument("--hints", type=Path, default=None,
                    help="file with hints to give the solver (diagnostic only)")
    ap.add_argument("--out", type=Path, default=None, help="where to write the full JSON report")
    args = ap.parse_args(argv)

    task = HarborTask(Path(args.task_dir).resolve())
    if not task.path.is_dir():
        print(f"solver_model: no such task directory: {task.path}", file=sys.stderr)
        return 2
    missing = task.missing_files()
    if missing:
        print(f"solver_model: task is missing {missing}", file=sys.stderr)
        return 2
    settings = Settings.load()
    try:
        endpoint = endpoint_from_env()
    except ValueError as exc:
        print(f"solver_model: {exc}", file=sys.stderr)
        return 2
    n = args.n or settings.calibration.initial_rollouts
    hints = args.hints.read_text() if args.hints else None

    started = time.monotonic()
    try:
        trajectories = asyncio.run(run_solver(task, endpoint, n=n,
                                              harbor_env=settings.sandbox.harbor_env, hints=hints))
    except Exception as exc:  # noqa: BLE001  report it and let the proposer decide
        print(json.dumps({"error": f"{type(exc).__name__}: {str(exc)[:500]}"}))
        return 3
    report = build_report(task, list(trajectories), settings, time.monotonic() - started)

    out = args.out or Path("logs") / "solver_model" / f"{task.task_id}-{int(time.time())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    compact = {k: report[k] for k in ("attempts_with_reward", "rewards", "mean_reward",
                                      "reward_std", "band", "verdict", "wall_time_s")}
    compact["report"] = str(out)
    print(json.dumps(compact))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
