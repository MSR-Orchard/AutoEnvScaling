import json
from pathlib import Path

import pytest

from autoenvscaling.evals.harbor_results import attempt_from_result, load_attempts
from autoenvscaling.evals.scoring import (
    INFRA_ERROR,
    SCORED,
    VERIFIER_BROKEN,
    Attempt,
    classify,
    summarize,
)


def test_classify_separates_model_outcomes_from_infrastructure():
    assert classify(1.0, None) == SCORED
    assert classify(0.0, None) == SCORED
    assert classify(None, None) == VERIFIER_BROKEN
    # A timeout is the model's outcome: the verifier scored what was left behind.
    assert classify(0.0, "AgentTimeoutError") == SCORED
    assert classify(None, "AgentTimeoutError") == INFRA_ERROR
    # Provider and sandbox failures are never scored, even if a reward exists.
    assert classify(0.0, "ApiRateLimitError") == INFRA_ERROR
    assert classify(1.0, "RuntimeError") == INFRA_ERROR


def _a(task, reward, status=SCORED, t="0"):
    return Attempt(task=task, status=status, reward=reward, started_at=t)


def test_avg_at_k_is_mean_over_task_attempts():
    attempts = [_a("a", 1.0, t="1"), _a("a", 0.0, t="2"), _a("b", 0.0, t="1"),
                _a("b", 0.0, t="2")]
    rep = summarize(attempts, k=2)
    assert rep.avg_at_k == pytest.approx(0.25)
    assert rep.pass_at_k == pytest.approx(0.5)
    assert rep.per_attempt_means == [pytest.approx(0.5), pytest.approx(0.0)]
    assert rep.n_tasks_incomplete == 0


def test_infra_retries_never_increase_k():
    attempts = [_a("a", None, INFRA_ERROR, "0"), _a("a", None, INFRA_ERROR, "1"),
                _a("a", 1.0, t="2"), _a("a", 0.0, t="3"), _a("a", 1.0, t="4")]
    rep = summarize(attempts, k=2)
    # Only the first two scored attempts count; the infra errors are counted, not scored.
    assert rep.tasks[0]["rewards"] == [1.0, 0.0]
    assert rep.avg_at_k == pytest.approx(0.5)
    assert rep.n_infra_errors == 2


def test_missing_attempts_excluded_or_zero():
    attempts = [_a("a", 1.0), _a("b", None, VERIFIER_BROKEN)]
    excluded = summarize(attempts, k=2)
    assert excluded.avg_at_k == pytest.approx(1.0)
    assert excluded.n_tasks_scored == 1 and excluded.n_tasks_incomplete == 2
    zero = summarize(attempts, k=2, missing="zero")
    assert zero.avg_at_k == pytest.approx(0.25)  # (1+0)/2 for a, 0 for b
    assert zero.n_verifier_broken == 1


def test_expected_tasks_without_trials_are_incomplete():
    rep = summarize([_a("a", 1.0)], k=1, expected_tasks=["a", "b"])
    assert rep.n_tasks == 2 and rep.n_tasks_incomplete == 1 and rep.avg_at_k == 1.0


def _trial(job: Path, name: str, task: str, reward, exception=None, started="2026-01-01T00:00:00",
           metadata=None):
    d = job / name
    d.mkdir(parents=True)
    result = {
        "task_name": task, "trial_name": name, "started_at": started,
        "verifier_result": None if reward is None else {"rewards": {"reward": reward}},
        "exception_info": None if exception is None else {"exception_type": exception},
        "agent_result": {"metadata": metadata or {}},
    }
    (d / "result.json").write_text(json.dumps(result))


def test_load_attempts_from_harbor_job_dirs(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    (job / "result.json").write_text(json.dumps({"id": "job-level result, not a trial"}))
    _trial(job, "t1", "hello", 1.0)
    _trial(job, "t2", "hello", 0.0, metadata={"pi_exit_code": 124})
    _trial(job, "t3", "hello", None, exception="ApiRateLimitError")
    _trial(job, "t4", "world", 0.0, metadata={"context_limited": True})
    attempts = load_attempts([job])
    assert len(attempts) == 4
    by_name = {a.trial_name: a for a in attempts}
    assert by_name["t2"].timed_out and by_name["t2"].is_scored
    assert by_name["t3"].status == INFRA_ERROR
    assert by_name["t4"].context_limited
    rep = summarize(attempts, k=2)
    assert rep.n_timed_out == 1 and rep.n_context_limited == 1


def test_attempt_ignores_non_numeric_rewards():
    a = attempt_from_result({"task_name": "x", "trial_name": "y",
                             "verifier_result": {"rewards": {"reward": True}}})
    assert a.status == VERIFIER_BROKEN
