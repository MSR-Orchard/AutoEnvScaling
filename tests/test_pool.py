import random
from pathlib import Path

import pytest

from autoenvscaling.config import CalibrationSettings
from autoenvscaling.pool import ACTIVE, RESERVE, RETIRED, Pool, ReplacementRequest
from autoenvscaling.results import CalibrationReport
from autoenvscaling.tasks import REQUIRED_FILES, HarborTask
from autoenvscaling.trajectories.export import read_jsonl
from autoenvscaling.trajectories.schema import Assignment, ProposerTrajectory, SolverTrajectory


def make_task(root: Path, name: str, body: str | None = None) -> HarborTask:
    path = root / name
    for rel in REQUIRED_FILES:
        f = path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(f"{rel} for {body or name}\n")
    return HarborTask(path)


def _report(task_id: str, rewards: list[float]) -> CalibrationReport:
    return CalibrationReport(task_id, rewards)


@pytest.fixture
def pool(tmp_path):
    return Pool(tmp_path / "pool", capacity=3)


def test_admit_copies_dedupes_and_overflows_to_reserve(tmp_path, pool):
    tasks = [make_task(tmp_path / "src", f"t{i}") for i in range(4)]
    traj = ProposerTrajectory("tr0", Assignment("a0", "parent", 0.9), "m", "pi")
    entries = [pool.admit(t, step=i, source="proposer:m", trajectory=traj if i == 0 else None)
               for i, t in enumerate(tasks)]
    assert [e.status for e in entries] == [ACTIVE, ACTIVE, ACTIVE, RESERVE]
    assert entries[0].assignment_id == "a0" and entries[0].parent_task_id == "parent"
    assert pool.task(entries[0]).instruction == "instruction.md for t0\n"
    assert pool.task_path(entries[0]).is_relative_to(pool.root)
    # Same content again: no new entry.
    dup = make_task(tmp_path / "elsewhere", "t0")
    assert pool.admit(dup) is entries[0]
    assert len(pool.entries) == 4 and pool.open_slots == 0

    pool.save()
    reloaded = Pool.load(pool.root, capacity=3)
    assert {e.task_id: e.status for e in reloaded.entries.values()} == \
        {e.task_id: e.status for e in entries}


def test_seed_rejects_incomplete_tasks(tmp_path, pool):
    good = make_task(tmp_path, "good")
    bad = tmp_path / "bad"
    bad.mkdir()
    with pytest.raises(ValueError, match="missing"):
        pool.seed([good.path, bad])


def test_sample_is_uniform_without_replacement_and_respects_exclude(tmp_path):
    pool = Pool(tmp_path / "p", capacity=10)
    ids = [pool.admit(make_task(tmp_path, f"t{i}")).task_id for i in range(5)]
    rng = random.Random(0)
    picked = pool.sample(3, rng)
    assert len({e.task_id for e in picked}) == 3
    rest = pool.sample(10, rng, exclude=[e.task_id for e in picked])
    assert {e.task_id for e in rest} == set(ids) - {e.task_id for e in picked}


def test_apply_review_retires_out_of_band_promotes_reserve_and_requests_replacements(tmp_path, pool):
    t = [pool.admit(make_task(tmp_path, f"t{i}"), step=i) for i in range(5)]
    assert [e.status for e in t] == [ACTIVE, ACTIVE, ACTIVE, RESERVE, RESERVE]
    pool.record_rollouts(t[1].task_id, [
        SolverTrajectory("x", t[1].task_id, "s", reward=0.0, verifier_output="ok\nFAILED test_io"),
        SolverTrajectory("y", t[1].task_id, "s", reward=0.25, verifier_output="FAILED test_io"),
        SolverTrajectory("z", t[1].task_id, "s", reward=1.0, verifier_output="all passed"),
    ], step=3)
    reports = {
        t[0].task_id: _report(t[0].task_id, [0.0, 0.5, 1.0, 0.5]),   # in band: kept
        t[1].task_id: _report(t[1].task_id, [0.0, 0.0, 0.0, 0.25]),  # too hard
        t[2].task_id: _report(t[2].task_id, [1.0, 1.0, 1.0, 0.75]),  # too easy
        t[3].task_id: _report(t[3].task_id, [0.25, 0.75, 0.5, 0.5]), # reserve, in band: promoted
        t[4].task_id: _report(t[4].task_id, [1.0, 1.0, 1.0, 1.0]),   # reserve, too easy: retired
    }
    outcome = pool.apply_review(reports, step=16)
    assert outcome.kept == [t[0].task_id]
    assert outcome.removed == {t[1].task_id: "too_hard", t[2].task_id: "too_easy"}
    assert outcome.promoted == [t[3].task_id]
    assert t[4].status == RETIRED and t[4].retired_reason == "too_easy"
    requests = {r.parent_task_id: r for r in outcome.requests}
    assert set(requests) == {t[1].task_id, t[2].task_id}
    assert requests[t[1].task_id].direction == "simpler"
    assert requests[t[1].task_id].common_failure == "failing tests: test_io (2/2 attempts)"
    assert requests[t[2].task_id].direction == "harder"
    assert requests[t[2].task_id].solver_pass_rate == pytest.approx(0.9375)
    assert pool.open_slots == 1
    assert t[0].reviews[-1]["verdict"] == "admitted"

    a = requests[t[2].task_id].to_assignment("r1-a0")
    assert a.parent_task_id == t[2].task_id and a.direction == "harder"
    assert a.extra["parent_task_dir"].endswith(t[2].task_id)


def test_apply_review_without_variance_requirement(tmp_path, pool):
    e = pool.admit(make_task(tmp_path, "t"))
    out = pool.apply_review({e.task_id: _report(e.task_id, [0.5, 0.5, 0.5, 0.5])}, 1,
                            require_variance=False)
    assert out.kept == [e.task_id]
    out = pool.apply_review({e.task_id: _report(e.task_id, [0.5, 0.5, 0.5, 0.5])}, 2)
    assert out.removed == {e.task_id: "no_variance"}
    assert out.requests[0].direction == "auto"


async def test_review_reruns_solver_and_records_rollouts(tmp_path, pool):
    easy = pool.admit(make_task(tmp_path, "easy"))
    mid = pool.admit(make_task(tmp_path, "mid"))
    plan = {easy.task_id: [1.0, 1.0, 1.0, 1.0], mid.task_id: [0.0, 1.0, 0.5, 0.5]}

    async def rerun(entry):
        return [SolverTrajectory(f"{entry.task_id}-{i}", entry.task_id, "s", reward=r)
                for i, r in enumerate(plan[entry.task_id])]

    outcome = await pool.review(rerun, step=16, calibration=CalibrationSettings(), concurrency=2)
    assert outcome.removed == {easy.task_id: "too_easy"}
    assert len(pool.latest_rollouts(mid.task_id)) == 4
    assert mid.exposures == 0            # review attempts are not training exposures


def test_export_prompt_data_lists_active_entries(tmp_path, pool):
    entries = [pool.admit(make_task(tmp_path, f"t{i}"), source="tmax") for i in range(4)]
    n = pool.export_prompt_data(tmp_path / "prompt.jsonl")
    rows = read_jsonl(tmp_path / "prompt.jsonl")
    assert n == 3 == len(rows)
    assert {r["label"] for r in rows} == {e.task_id for e in entries if e.status == ACTIVE}
    assert rows[0]["metadata"]["source"] == "tmax"
    assert Path(rows[0]["metadata"]["task_dir"]).is_absolute()


def test_replacement_request_direction():
    assert ReplacementRequest("p", "d", 0.1, "too_hard").direction == "simpler"
    assert ReplacementRequest("p", "d", 0.9, "too_easy").direction == "harder"
    assert ReplacementRequest("p", "d", 0.5, "no_variance").direction == "auto"


def test_latest_rollout_trajectories_round_trip(tmp_path, pool):
    e = pool.admit(make_task(tmp_path, "t"))
    traj = SolverTrajectory("x", e.task_id, "s", reward=0.5, tests_passed=1, tests_total=2)
    pool.record_rollouts(e.task_id, [traj], step=4)
    (back,) = pool.latest_rollout_trajectories(e.task_id)
    assert back == traj
