from pathlib import Path

import pytest

from autoenvscaling.config import Settings
from autoenvscaling.results import BUILD, CONTAMINATION, EMPTY, REFERENCE, RUBRIC
from autoenvscaling.tasks import HarborTask
from autoenvscaling.validation import admission as A
from autoenvscaling.validation.contamination import NgramIndex
from autoenvscaling.validation.harbor_runner import RunResult, TrialOutcome
from autoenvscaling.validation.rubric import CriterionVerdict, ReviewerSpec, RubricResult

HELDOUT = ("The nightly backup job in /etc/cron.d/backup silently stopped writing archives to "
           "/srv/backups after the last system upgrade.")


def make_task(root: Path,
              instruction: str = "Summarise the web server logs in /app/logs.\n") -> HarborTask:
    for rel in ("environment", "solution", "tests"):
        (root / rel).mkdir(parents=True, exist_ok=True)
    (root / "task.toml").write_text('schema_version = "1.4"\n\n[verifier]\ntimeout_sec = 120.0\n')
    (root / "instruction.md").write_text(instruction)
    (root / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (root / "solution" / "solve.sh").write_text("#!/bin/bash\npython3 /app/summarise.py\n")
    (root / "tests" / "test.sh").write_text("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")
    return HarborTask(root)


class FakeRunner:
    def __init__(self, **by_label):
        self.by_label = by_label
        self.calls = []

    async def run(self, task_dir, agent, *, n_attempts=1, label="trial", instruction=None,
                  public_network=False):
        self.calls.append((label, agent.name))
        outcome = self.by_label[label]
        if isinstance(outcome, Exception):
            raise outcome
        return RunResult(outcomes=[outcome], job_dir=Path("/jobs") / label)


def scored(reward):
    return TrialOutcome(trial_name="t", reward=reward)


@pytest.fixture
def settings(tmp_path):
    s = Settings()
    s.work_dir = tmp_path / "work"
    return s


@pytest.fixture
def heldout():
    return NgramIndex.build({"tb:cron": HELDOUT}, n=13, sources=["fixture"])


@pytest.fixture
def review(monkeypatch):
    state = {"result": RubricResult(verdicts=[CriterionVerdict("typos", "pass")]), "calls": 0}

    async def fake_review(task_dir, reviewer, **kwargs):
        state["calls"] += 1
        state["kwargs"] = kwargs
        return state["result"]

    monkeypatch.setattr(A, "review_task", fake_review)
    return state


REVIEWER = ReviewerSpec(agent="claude-code", model="claude-opus-5")


async def test_all_checks_pass_in_order(tmp_path, settings, heldout, review):
    runner = FakeRunner(reference=scored(1.0), empty=scored(0.2))
    report = await A.validate(make_task(tmp_path / "t"), settings, runner=runner,
                              reviewer=REVIEWER, heldout=heldout)
    assert [c.name for c in report.checks] == [BUILD, REFERENCE, EMPTY, CONTAMINATION, RUBRIC]
    assert report.passed and runner.calls == [("reference", "oracle"), ("empty", "nop")]
    assert review["kwargs"]["skip"] == ("binary_reward",)
    assert review["kwargs"]["rubric"] == "terminal-bench"


async def test_incomplete_task_fails_build_and_skips_trials(tmp_path, settings, heldout, review):
    task = make_task(tmp_path / "t")
    (task.path / "tests" / "test.sh").unlink()
    runner = FakeRunner()
    report = await A.validate(task, settings, runner=runner, reviewer=REVIEWER, heldout=heldout)
    by = {c.name: c for c in report.checks}
    assert not by[BUILD].passed and "missing tests/test.sh" in by[BUILD].detail
    assert by[REFERENCE].data["skipped"] and by[EMPTY].data["skipped"]
    assert by[CONTAMINATION].passed  # cheap, so it still runs
    assert by[RUBRIC].data["skipped"] and review["calls"] == 0
    assert runner.calls == []


async def test_bare_task_name_is_accepted(tmp_path, settings):
    task = make_task(tmp_path / "t")
    (task.path / "task.toml").write_text('[task]\nname = "log-summary"\n')
    assert A.structure_errors(task) == []


async def test_invalid_task_toml_is_a_structure_error(tmp_path, settings, heldout):
    task = make_task(tmp_path / "t")
    (task.path / "task.toml").write_text("[environment]\ncpus = 'many'\n")
    report = await A.validate(task, settings, runner=FakeRunner(), heldout=heldout,
                              checks=A.EXECUTION_CHECKS)
    assert "task.toml is invalid" in report.checks[0].detail


async def test_build_failure_skips_reference_and_empty(tmp_path, settings, heldout, review):
    fail = TrialOutcome(trial_name="t", exception_type="ImageBuildError",
                        exception_message="Image build for im-1 failed.",
                        failed_phase="environment")
    runner = FakeRunner(reference=fail)
    report = await A.validate(make_task(tmp_path / "t"), settings, runner=runner,
                              reviewer=REVIEWER, heldout=heldout)
    assert report.failed_checks == [BUILD, REFERENCE, EMPTY, RUBRIC]
    assert "ImageBuildError" in report.checks[0].data["build_log"]
    assert not A.has_infra_error(report)


async def test_reference_below_full_reward_fails(tmp_path, settings, heldout, review):
    runner = FakeRunner(reference=TrialOutcome(trial_name="t", reward=0.75,
                                               verifier_output="3 passed, 1 failed"))
    report = await A.validate(make_task(tmp_path / "t"), settings, runner=runner,
                              reviewer=REVIEWER, heldout=heldout)
    by = {c.name: c for c in report.checks}
    assert by[BUILD].passed and not by[REFERENCE].passed
    assert "0.75" in by[REFERENCE].detail
    assert "1 failed" in by[REFERENCE].data["verifier_output"]
    assert by[EMPTY].data["skipped"] and by[RUBRIC].data["skipped"]
    assert runner.calls == [("reference", "oracle")]


async def test_gameable_tests_fail_the_empty_solution(tmp_path, settings, heldout, review):
    runner = FakeRunner(reference=scored(1.0), empty=scored(0.5))
    report = await A.validate(make_task(tmp_path / "t"), settings, runner=runner,
                              reviewer=REVIEWER, heldout=heldout)
    assert report.failed_checks == [EMPTY, RUBRIC]
    assert "less than 0.5" in report.checks[2].detail


async def test_verifier_crash_on_empty_solution_is_a_task_failure(tmp_path, settings, heldout):
    crash = TrialOutcome(trial_name="t", exception_type="RewardFileNotFoundError",
                         failed_phase="verifier")
    runner = FakeRunner(reference=scored(1.0), empty=crash)
    report = await A.validate(make_task(tmp_path / "t"), settings, runner=runner,
                              heldout=heldout, checks=A.EXECUTION_CHECKS)
    assert not report.checks[2].passed and not A.has_infra_error(report)


async def test_infrastructure_failures_are_marked(tmp_path, settings, heldout, review):
    infra = TrialOutcome(trial_name="t", exception_type="AuthError", failed_phase="environment",
                         infra_error=True)
    report = await A.validate(make_task(tmp_path / "t"), settings,
                              runner=FakeRunner(reference=infra), reviewer=REVIEWER,
                              heldout=heldout)
    assert A.has_infra_error(report) and report.checks[0].data["infra_error"]
    crashed = await A.validate(make_task(tmp_path / "t2"), settings,
                               runner=FakeRunner(reference=RuntimeError("no modal token")),
                               reviewer=REVIEWER, heldout=heldout)
    assert A.has_infra_error(crashed)


async def test_contamination_blocks_the_rubric_review(tmp_path, settings, heldout, review):
    task = make_task(tmp_path / "t", "Hi. " + HELDOUT + " Please fix it.\n")
    report = await A.validate(task, settings, runner=FakeRunner(reference=scored(1.0),
                                                                empty=scored(0.0)),
                              reviewer=REVIEWER, heldout=heldout)
    by = {c.name: c for c in report.checks}
    assert not by[CONTAMINATION].passed and "tb:cron" in by[CONTAMINATION].detail
    assert by[RUBRIC].data["skipped"] and review["calls"] == 0


async def test_rubric_failure_lists_criteria(tmp_path, settings, heldout, review):
    review["result"] = RubricResult(verdicts=[CriterionVerdict("typos", "fail", "misspelt path"),
                                              CriterionVerdict("verifiable", "pass")])
    report = await A.validate(make_task(tmp_path / "t"), settings,
                              runner=FakeRunner(reference=scored(1.0), empty=scored(0.0)),
                              reviewer=REVIEWER, heldout=heldout)
    rub = report.checks[-1]
    assert not rub.passed and rub.data["failed_criteria"] == ["typos"]
    review["result"] = RubricResult(error="Check agent did not produce a valid result")
    report = await A.validate(make_task(tmp_path / "t2"), settings,
                              runner=FakeRunner(reference=scored(1.0), empty=scored(0.0)),
                              reviewer=REVIEWER, heldout=heldout)
    assert A.has_infra_error(report)


async def test_missing_reviewer_fails_before_running_anything(tmp_path, settings, monkeypatch):
    monkeypatch.delenv("AES_REVIEWER_MODEL", raising=False)
    runner = FakeRunner()
    with pytest.raises(ValueError, match="AES_REVIEWER_MODEL"):
        await A.validate(make_task(tmp_path / "t"), settings, runner=runner)
    assert runner.calls == []


async def test_check_subsets(tmp_path, settings, heldout):
    report = await A.validate(make_task(tmp_path / "t"), settings,
                              runner=FakeRunner(reference=scored(1.0), empty=scored(0.0)),
                              checks=A.EXECUTION_CHECKS)
    assert [c.name for c in report.checks] == [BUILD, REFERENCE, EMPTY] and report.passed
    with pytest.raises(ValueError):
        await A.validate(make_task(tmp_path / "t2"), settings, checks=(BUILD, REFERENCE))
    only = await A.validate(make_task(tmp_path / "t3"), settings, checks=(CONTAMINATION,),
                            heldout=heldout)
    assert [c.name for c in only.checks] == [CONTAMINATION]


async def test_admit_calibrates_only_valid_tasks(tmp_path, settings, heldout, monkeypatch):
    import autoenvscaling.calibration.band as band
    calls = []

    async def fake_calibrate(task, solver, settings):
        calls.append(task.task_id)
        return "calibration"

    monkeypatch.setattr(band, "calibrate", fake_calibrate)
    ok = await A.admit(make_task(tmp_path / "ok"), None, settings, checks=A.EXECUTION_CHECKS,
                       runner=FakeRunner(reference=scored(1.0), empty=scored(0.0)))
    bad = await A.admit(make_task(tmp_path / "bad"), None, settings, checks=A.EXECUTION_CHECKS,
                        runner=FakeRunner(reference=scored(0.0)))
    assert ok[1] == "calibration" and bad[1] is None and calls == ["ok"]
