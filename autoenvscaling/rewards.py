"""Rewards and advantages for the two roles (Section 3.3, Equations 1-3).

Solver:   r_S = fraction of the environment's tests that pass (partial credit).
Proposer: r_P = -1    if validation fails,
                -0.25 if validation passes but calibration fails,
                 1    if both pass (the environment enters the training pool).

Advantages are computed separately per role. Proposer rewards are centred within a group of
trajectories that share one generation assignment; solver rewards are centred across attempts
on the same environment. Rewards are mean-centred without dividing by the group standard
deviation, and groups whose rewards do not vary are dropped (following DAPO).
"""

from __future__ import annotations

from collections.abc import Sequence

from autoenvscaling.results import CalibrationReport, ValidationReport

PROPOSER_INVALID = -1.0
PROPOSER_UNCALIBRATED = -0.25
PROPOSER_ADMITTED = 1.0


def solver_reward(tests_passed: int, tests_total: int) -> float:
    if tests_total <= 0:
        raise ValueError("an environment must have at least one test")
    if not 0 <= tests_passed <= tests_total:
        raise ValueError(f"tests_passed={tests_passed} outside [0, {tests_total}]")
    return tests_passed / tests_total


def proposer_reward(validation: ValidationReport | bool,
                    calibration: CalibrationReport | bool | None) -> float:
    """r_P for one proposer trajectory. Calibration only runs after validation passes."""
    valid = validation if isinstance(validation, bool) else validation.passed
    if not valid:
        return PROPOSER_INVALID
    calibrated = (calibration if isinstance(calibration, bool)
                  else calibration is not None and calibration.passed)
    return PROPOSER_ADMITTED if calibrated else PROPOSER_UNCALIBRATED


def group_advantages(rewards: Sequence[float], eps: float = 1e-8) -> list[float] | None:
    """Mean-centred advantages for one group, or None if the rewards do not vary."""
    if not rewards:
        return None
    mean = sum(rewards) / len(rewards)
    if max(rewards) - min(rewards) <= eps:
        return None
    return [r - mean for r in rewards]
