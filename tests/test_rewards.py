import pytest

from autoenvscaling.results import CalibrationReport, CheckResult, ValidationReport
from autoenvscaling.rewards import (PROPOSER_ADMITTED, PROPOSER_INVALID, PROPOSER_UNCALIBRATED,
                                    group_advantages, proposer_reward, solver_reward)


def test_solver_reward_is_fraction_of_tests():
    assert solver_reward(3, 4) == 0.75
    assert solver_reward(0, 5) == 0.0
    with pytest.raises(ValueError):
        solver_reward(1, 0)
    with pytest.raises(ValueError):
        solver_reward(5, 4)


def _validation(passed: bool) -> ValidationReport:
    return ValidationReport("t", [CheckResult("build", True), CheckResult("reference_solution", passed)])


def test_proposer_reward_cases():
    in_band = CalibrationReport("t", [0.0, 1.0, 0.5, 0.5])
    too_easy = CalibrationReport("t", [1.0, 1.0, 1.0, 1.0])
    assert proposer_reward(_validation(False), None) == PROPOSER_INVALID == -1.0
    assert proposer_reward(_validation(True), too_easy) == PROPOSER_UNCALIBRATED == -0.25
    assert proposer_reward(_validation(True), in_band) == PROPOSER_ADMITTED == 1.0
    assert proposer_reward(True, None) == PROPOSER_UNCALIBRATED
    # A failed validation stays -1 even if a calibration report is present.
    assert proposer_reward(_validation(False), in_band) == PROPOSER_INVALID


def test_calibration_band_and_variance():
    assert CalibrationReport("t", [0.25, 0.25, 0.25, 0.25]).verdict == "no_variance"
    assert CalibrationReport("t", [0.0, 0.0, 0.0, 0.1]).verdict == "too_hard"
    assert CalibrationReport("t", [1.0, 1.0, 0.9, 1.0]).verdict == "too_easy"
    assert CalibrationReport("t", [0.0, 0.5, 1.0, 0.5]).passed


def test_group_advantages_mean_centred_and_drop_constant_groups():
    assert group_advantages([1.0, -1.0, -0.25, 1.0]) == pytest.approx([0.8125, -1.1875, -0.4375, 0.8125])
    assert group_advantages([1.0, 1.0, 1.0]) is None
    assert group_advantages([]) is None
