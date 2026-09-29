"""Assignments from solver rollouts: pass rate, most common failure, direction."""

from autoenvscaling.proposer.assignment import (
    HARDER,
    SIMPLER,
    build_assignment,
    build_assignments,
    load_assignment,
    most_common_failure,
    pass_rate,
    render_assignment,
    reward_summary,
    round_robin,
    save_assignment,
)
from autoenvscaling.trajectories.schema import SolverTrajectory


def traj(i, task="parent", reward=None, passed=0, total=4, verifier=""):
    return SolverTrajectory(trajectory_id=f"{task}-{i}", task_id=task, model="solver",
                            tests_passed=passed, tests_total=total, reward=reward,
                            verifier_output=verifier)


PYTEST_FAIL = """\
FAILED test_outputs.py::test_checkpoint_loaded - AssertionError: buffers loaded as weights
FAILED test_outputs.py::test_metrics - assert 0.1 > 0.9
========================= 2 failed, 2 passed in 1.23s ==========================
"""


def test_pass_rate_ignores_attempts_without_a_verdict():
    rollouts = [traj(0, reward=1.0), traj(1, reward=0.5), traj(2, total=0)]
    assert pass_rate(rollouts) == 0.5
    assert pass_rate([traj(0, total=0)]) is None
    summary = reward_summary(rollouts)
    assert summary["n"] == 2 and summary["n_errored"] == 1
    assert summary["mean_reward"] == 0.75 and summary["reward_std"] == 0.25


def test_most_common_failure_reads_pytest_output():
    rollouts = [traj(0, reward=0.5, verifier=PYTEST_FAIL),
                traj(1, reward=0.75, verifier=PYTEST_FAIL.splitlines()[0]),
                traj(2, reward=1.0)]
    failure = most_common_failure(rollouts)
    assert failure.startswith("failing tests: test_outputs.py::test_checkpoint_loaded (2/2")
    assert "2 failed, 2 passed" in failure
    assert most_common_failure([traj(0, reward=1.0)]) == ""
    assert most_common_failure([traj(0, passed=1, total=4)]) == "typically 1 of 4 tests pass"


def test_direction_is_simpler_when_failing_and_harder_when_succeeding():
    failing = build_assignment("parent", [traj(i, reward=0.25) for i in range(4)])
    assert failing.direction == SIMPLER and failing.solver_pass_rate == 0.0
    succeeding = build_assignment("parent", [traj(i, reward=1.0) for i in range(3)]
                                  + [traj(9, reward=0.0)])
    assert succeeding.direction == HARDER and succeeding.solver_pass_rate == 0.75
    assert succeeding.extra["n_attempts"] == 4


def test_assignment_ids_are_deterministic_and_rollouts_filtered_by_parent():
    rollouts = [traj(0, reward=1.0), traj(1, task="other", reward=0.0)]
    a = build_assignment("parent", rollouts)
    b = build_assignment("parent", list(reversed(rollouts)))
    assert a.assignment_id == b.assignment_id and a.assignment_id.startswith("asg-")
    assert a.extra["n_attempts"] == 1


def test_round_robin_across_categories():
    cats = {"a1": "a", "a2": "a", "a3": "a", "b1": "b"}
    assert round_robin(["a3", "a1", "b1", "a2"], cats) == ["a1", "b1", "a2", "a3"]
    by_task = {t: [traj(0, task=t, reward=0.5)] for t in cats}
    by_task["empty"] = [traj(0, task="empty", total=0)]
    assignments = build_assignments(by_task, categories=cats, n=3)
    assert [x.parent_task_id for x in assignments] == ["a1", "b1", "a2"]
    assert assignments[1].extra["category"] == "b"


def test_render_and_save(tmp_path):
    a = build_assignment("parent", [traj(0, reward=0.0, verifier=PYTEST_FAIL)], category="ml")
    text = render_assignment(a)
    assert "`parent`" in text and "Direction: **simpler**" in text and "Category: ml" in text
    save_assignment(a, tmp_path / "a.json")
    assert load_assignment(tmp_path / "a.json") == a
