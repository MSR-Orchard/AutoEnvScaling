import json

import pytest

from autoenvscaling.calibration import band as B
from autoenvscaling.calibration.solver import SolverEndpoint
from autoenvscaling.config import CalibrationSettings, Settings
from autoenvscaling.tasks import HarborTask
from autoenvscaling.validation.harbor_runner import RunResult, TrialOutcome


@pytest.mark.parametrize("mean, expand", [
    (0.05, False), (0.10, True), (0.25, True), (0.40, True), (0.41, False), (0.5, False),
    (0.59, False), (0.60, True), (0.75, True), (0.90, True), (0.91, False), (1.0, False)])
def test_expansion_rule_is_within_margin_of_either_edge(mean, expand):
    assert B.needs_more_rollouts(mean, CalibrationSettings()) is expand


class ScriptedRunner:
    """Returns the next scripted batch of rewards per call; None means an infrastructure failure."""

    def __init__(self, *batches):
        self.batches = list(batches)
        self.calls = []

    async def run(self, task_dir, agent, *, n_attempts=1, label="trial", instruction=None,
                  public_network=False):
        rewards = self.batches.pop(0)
        assert len(rewards) == n_attempts
        self.calls.append((label, n_attempts, instruction))
        outcomes = [TrialOutcome(trial_name=f"{label}{len(self.calls)}_{i}", reward=r,
                                 infra_error=r is None,
                                 exception_type=None if r is not None else "ApiError")
                    for i, r in enumerate(rewards)]
        return RunResult(outcomes=outcomes)


@pytest.fixture
def settings(tmp_path):
    s = Settings()
    s.work_dir = tmp_path / "work"
    return s


@pytest.fixture
def task(tmp_path):
    root = tmp_path / "task"
    (root / "solution").mkdir(parents=True)
    (root / "instruction.md").write_text("Repair the index.\n")
    (root / "solution" / "solve.sh").write_text("#!/bin/bash\npython3 /solution/fix.py\n")
    return HarborTask(root)


SOLVER = SolverEndpoint(base_url="http://solver.test/v1", model="solver-9b")


async def test_in_band_with_four_rollouts(task, settings):
    runner = ScriptedRunner([0.25, 0.75, 0.5, 0.5])
    report = await B.calibrate(task, SOLVER, settings, runner=runner)
    assert report.rewards == [0.25, 0.75, 0.5, 0.5] and report.passed
    assert runner.calls == [("solver", 4, None)]
    record = json.loads((B.calibration_dir(task, settings) / "calibration.json").read_text())
    assert record["report"]["verdict"] == "admitted" and not record["expanded"]
    lines = (B.calibration_dir(task, settings) / "trajectories.jsonl").read_text().splitlines()
    assert len(lines) == 4


async def test_near_an_edge_expands_to_eight(task, settings):
    runner = ScriptedRunner([0.0, 0.5, 0.5, 0.2], [0.5, 0.0, 0.25, 0.25])
    report = await B.calibrate(task, SOLVER, settings, runner=runner)
    assert [c[1] for c in runner.calls] == [4, 4] and len(report.rewards) == 8
    assert report.rewards[:4] == [0.0, 0.5, 0.5, 0.2]


async def test_below_band_runs_one_hint_probe(task, settings):
    runner = ScriptedRunner([0.0, 0.0, 0.0, 0.0], [0.5])
    report = await B.calibrate(task, SOLVER, settings, runner=runner)
    assert report.verdict == "too_hard" and report.hint_probe_reward == 0.5
    label, n, instruction = runner.calls[-1]
    assert (label, n) == ("hint_probe", 1)
    assert "## Hints" in instruction and "1. python3 /solution/fix.py" in instruction
    assert report.rewards == [0.0] * 4  # the probe is diagnostic, not part of the sample


async def test_no_probe_when_disabled_or_not_below_band(task, settings):
    easy = await B.calibrate(task, SOLVER, settings, runner=ScriptedRunner([1.0] * 4))
    assert easy.verdict == "too_easy" and easy.hint_probe_reward is None
    settings.calibration.hint_probe_below_band = False
    runner = ScriptedRunner([0.0] * 4)
    hard = await B.calibrate(task, SOLVER, settings, runner=runner)
    assert hard.verdict == "too_hard" and len(runner.calls) == 1


async def test_infra_failures_are_topped_up_once_and_excluded(task, settings):
    runner = ScriptedRunner([0.5, None, 0.5, None], [1.0, 0.0])
    report = await B.calibrate(task, SOLVER, settings, runner=runner)
    assert [c[1] for c in runner.calls] == [4, 2]
    assert sorted(report.rewards) == [0.0, 0.5, 0.5, 1.0]


async def test_no_reward_at_all_raises(task, settings):
    runner = ScriptedRunner([None] * 4, [None] * 4)
    with pytest.raises(B.SolverUnavailableError, match="none of 8"):
        await B.calibrate(task, SOLVER, settings, runner=runner)
