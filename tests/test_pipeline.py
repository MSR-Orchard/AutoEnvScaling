import json
import os
import random
from pathlib import Path

import pytest

from autoenvscaling.calibration.solver import SolverEndpoint
from autoenvscaling.config import Settings
from autoenvscaling.pipeline import (
    PROPOSER_BASE_URL_ENV,
    TASK_FILES,
    GenerationPipeline,
    ProposerSpec,
    build_workspace,
    plan_assignments,
)
from autoenvscaling.pool import ACTIVE, Pool, ReplacementRequest
from autoenvscaling.results import CalibrationReport, CheckResult, ValidationReport
from autoenvscaling.tasks import REQUIRED_FILES, HarborTask
from autoenvscaling.trajectories.export import proposer_rl_groups
from autoenvscaling.trajectories.schema import (
    Assignment,
    ProposerTrajectory,
    SolverTrajectory,
    Turn,
    read_proposer_jsonl,
)

SOLVER = SolverEndpoint(base_url="http://solver.invalid/v1", model="solver")


def write_task(path: Path, outcome: str) -> None:
    """A complete Harbor task whose `outcome` file tells the fake checks what to report."""
    for rel in REQUIRED_FILES:
        f = path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(f"{rel} {path}\n")
    (path / "outcome").write_text(outcome)


class FakeProposer:
    """Episodes follow a script per (assignment, sample): one outcome per revision.

    Outcomes: "invalid", "easy" (valid, too easy), "ok" (admitted), "infra" (validation hits an
    infrastructure error), "none" (no task), "crash" (the episode raises).
    """

    def __init__(self, script):
        self.script = script
        self.calls = []

    async def __call__(self, assignment, *, harness, model, settings, workspace_dir, out_dir,
                       feedback=None, revision=0, previous_task_dir=None):
        sample = int(Path(out_dir).parent.name[1:])
        ws = Path(workspace_dir)
        previous = (Path(previous_task_dir) / "outcome").read_text() if previous_task_dir else None
        self.calls.append({"assignment": assignment.assignment_id, "sample": sample,
                           "revision": revision, "feedback": feedback,
                           "base_url": os.environ.get(PROPOSER_BASE_URL_ENV), "workspace": ws,
                           "previous": previous,
                           "memory": sorted(p.name for p in (ws / "memory").glob("*"))})
        outcome = self.script[(assignment.assignment_id, sample)][revision]
        if outcome == "crash":
            raise RuntimeError("sandbox died")
        out_dir = Path(out_dir)
        (out_dir / "memory").mkdir(parents=True, exist_ok=True)
        (out_dir / "memory" / f"{assignment.assignment_id}-s{sample}-r{revision}.md").write_text("note")
        task_dir = None
        if outcome != "none":
            task_dir = out_dir / "output" / "tasks" / "task"
            write_task(task_dir, outcome)
        return ProposerTrajectory(
            trajectory_id="", assignment=assignment, model=model, harness=harness,
            turns=[Turn("user", "go"), Turn("assistant", "done")],
            task_dir=str(task_dir) if task_dir else None)


async def fake_validate(task: HarborTask, settings) -> ValidationReport:
    outcome = (task.path / "outcome").read_text()
    if outcome == "infra":
        return ValidationReport(task.task_id, [CheckResult("build", False, "docker daemon down",
                                                           {"infra_error": True})])
    ok = outcome != "invalid"
    return ValidationReport(task.task_id, [CheckResult("build", True),
                                           CheckResult("reference_solution", ok, "" if ok else "reward 0.4")])


async def fake_calibrate(task: HarborTask, solver, settings) -> CalibrationReport:
    outcome = (task.path / "outcome").read_text()
    rewards = [1.0, 1.0, 1.0, 1.0] if outcome == "easy" else [0.0, 1.0, 0.5, 0.5]
    return CalibrationReport(task.task_id, rewards)


def make_pipeline(tmp_path, script, *, pool=None, proposer=None, **kwargs):
    fake = FakeProposer(script)
    kwargs = {"validate_fn": fake_validate, "calibrate_fn": fake_calibrate, **kwargs}
    pipe = GenerationPipeline(Settings(), proposer=proposer or ProposerSpec("claude-code", "opus"),
                              solver=SOLVER, out_dir=tmp_path / "rounds", pool=pool,
                              episode_fn=fake, **kwargs)
    return pipe, fake


def _assignment(aid: str) -> Assignment:
    return Assignment(aid, "parent", 0.9, direction="harder")


async def test_revision_loop_stops_at_admission_and_caps_revisions(tmp_path):
    script = {
        ("a", 0): ["ok"],                          # admitted first try
        ("b", 0): ["invalid", "easy", "ok"],       # admitted on the second revision
        ("c", 0): ["invalid", "invalid", "easy"],  # never admitted: 1 + max_revisions episodes
    }
    pipe, fake = make_pipeline(tmp_path, script)
    result = await pipe.run_round([_assignment(a) for a in "abc"], round_id="r1")

    rewards = {t.trajectory_id: t.reward for t in result.trajectories}
    assert rewards == {"a-s000-r0": 1.0,
                       "b-s000-r0": -1.0, "b-s000-r1": -0.25, "b-s000-r2": 1.0,
                       "c-s000-r0": -1.0, "c-s000-r1": -1.0, "c-s000-r2": -0.25}
    assert len(result.admitted_tasks) == 2 and result.admitted == []

    b_calls = [c for c in fake.calls if c["assignment"] == "b"]
    assert b_calls[0]["feedback"] is None and b_calls[0]["previous"] is None
    assert "VALIDATION FAILED" in b_calls[1]["feedback"] and "reward 0.4" in b_calls[1]["feedback"]
    assert "CALIBRATION FAILED" in b_calls[2]["feedback"] and "[0.25, 0.75]" in b_calls[2]["feedback"]
    # Each revision is handed the task it is revising.
    assert b_calls[1]["previous"] == "invalid" and b_calls[2]["previous"] == "easy"
    # Calibration runs only after validation passes.
    by_id = {t.trajectory_id: t for t in result.trajectories}
    assert by_id["c-s000-r0"].calibration is None
    assert by_id["c-s000-r2"].calibration["verdict"] == "too_easy"

    on_disk = read_proposer_jsonl(result.trajectories_path)
    assert sorted(t.trajectory_id for t in on_disk) == sorted(rewards)
    summary = json.loads((tmp_path / "rounds" / "r1" / "round.json").read_text())
    assert summary["episodes"] == 7 and summary["admitted_tasks"] == 2


async def test_host_failures_are_unscored_and_end_the_chain(tmp_path):
    script = {("a", 0): ["none", "crash"], ("b", 0): ["infra", "ok"]}
    pipe, _ = make_pipeline(tmp_path, script)
    result = await pipe.run_round([_assignment("a"), _assignment("b")], round_id="r")
    by_id = {t.trajectory_id: t for t in result.trajectories}
    assert by_id["a-s000-r0"].reward == -1.0          # no task is the proposer's failure
    assert by_id["a-s000-r0"].validation["checks"][0]["name"] == TASK_FILES
    assert by_id["a-s000-r1"].reward is None and "sandbox died" in by_id["a-s000-r1"].usage["error"]
    assert by_id["b-s000-r0"].reward is None           # infrastructure error: no revision
    assert "b-s000-r1" not in by_id
    assert result.summary()["infra_errors"] == 2


async def test_admission_exception_is_unscored(tmp_path):
    async def broken_validate(task, settings):
        raise ConnectionError("harbor unreachable")

    pipe, _ = make_pipeline(tmp_path, {("a", 0): ["ok"]}, validate_fn=broken_validate)
    result = await pipe.run_round([_assignment("a")])
    assert result.trajectories[0].reward is None
    assert "harbor unreachable" in result.trajectories[0].usage["error"]


async def test_shared_policy_round_groups_and_pool_admission(tmp_path):
    pool = Pool(tmp_path / "pool", capacity=2)
    endpoint = SolverEndpoint(base_url="http://policy.invalid/v1", model="policy")
    script = {("g", s): o for s, o in enumerate([["ok"], ["invalid"] * 3, ["easy", "ok"], ["ok"]])}
    pipe, fake = make_pipeline(tmp_path, script, pool=pool, proposer=ProposerSpec.shared(endpoint))
    result = await pipe.run_round([_assignment("g")], episodes_per_assignment=4, round_id="r", step=16)

    assert {c["base_url"] for c in fake.calls} == {"http://policy.invalid/v1"}
    assert os.environ.get(PROPOSER_BASE_URL_ENV) is None
    assert [e.status for e in result.admitted] == [ACTIVE, ACTIVE, "reserve"]
    assert all(e.source == "proposer:policy" and e.added_step == 16 for e in result.admitted)
    assert Pool.load(pool.root, capacity=2).entries.keys() == pool.entries.keys()

    groups, _ = proposer_rl_groups(result.trajectories)
    assert len(groups) == 1 and len(groups[0].rewards) == 7
    assert sum(s.advantage for s in groups[0].samples) == pytest.approx(0.0)


async def test_memory_and_tools_carry_over_to_the_next_round(tmp_path):
    pipe, fake = make_pipeline(tmp_path, {("a", 0): ["invalid", "ok"], ("b", 0): ["ok"],
                                          ("c", 0): ["ok"]}, max_concurrent_chains=1)
    await pipe.run_round([_assignment("a"), _assignment("b")], round_id="r1")
    # Saved after every episode: the state holds the most recently finished episode's copy.
    assert sorted(p.name for p in (pipe.state_dir / "memory").iterdir()) == ["b-s0-r0.md"]
    assert fake.calls[0]["memory"] == []
    await pipe.run_round([_assignment("c")], round_id="r2")
    assert fake.calls[-1]["memory"] == ["b-s0-r0.md"]


def test_reviewer_is_the_proposers_model_unless_pinned():
    from autoenvscaling.pipeline import reviewer_spec

    frontier = reviewer_spec(Settings(), ProposerSpec("claude-code", "claude-opus-5"))
    assert (frontier.agent, frontier.model) == ("claude-code", "claude-opus-5")
    shared = reviewer_spec(Settings(), ProposerSpec("pi", "policy", True, "http://policy.invalid/v1"))
    assert shared.agent == "pi" and shared.model == "openai/policy"
    assert shared.agent_env["OPENAI_BASE_URL"] == "http://policy.invalid/v1"
    pinned = Settings()
    pinned.validation.reviewer_model = "gpt-5.6-sol"
    spec = reviewer_spec(pinned, ProposerSpec("pi", "policy", True, "http://policy.invalid/v1"))
    assert (spec.agent, spec.model) == ("pi", "gpt-5.6-sol")


def _pool_with_parent(tmp_path) -> tuple[Pool, str]:
    pool = Pool(tmp_path / "pool", capacity=4)
    parent = tmp_path / "src" / "parent"
    write_task(parent, "ok")
    entry = pool.admit(HarborTask(parent))
    pool.record_rollouts(entry.task_id, [
        SolverTrajectory("x", entry.task_id, "s", reward=0.0, verifier_output="FAILED test_a"),
        SolverTrajectory("y", entry.task_id, "s", reward=1.0, verifier_output="ok")], step=1)
    return pool, entry.task_id


def test_workspace_refresh_and_solver_feedback_ablation(tmp_path):
    pool, task_id = _pool_with_parent(tmp_path)
    assignment = Assignment("a", task_id, 0.5)
    ws = build_workspace(tmp_path / "ws1", pool, [assignment], True)
    assert (ws / "past_environments" / task_id / "instruction.md").exists()
    assert (ws / "past_rollouts" / "failed" / task_id / "x.json").exists()
    assert json.loads((ws / "past_rollouts" / "index.jsonl").read_text())["task_id"] == task_id

    ws = build_workspace(tmp_path / "ws2", pool, [assignment], False)
    assert (ws / "past_environments" / task_id).exists()
    assert not any((ws / "past_rollouts" / "failed").iterdir())


async def test_round_without_solver_feedback_hides_rollouts(tmp_path):
    pool, task_id = _pool_with_parent(tmp_path)
    pipe, fake = make_pipeline(tmp_path, {("a", 0): ["ok"]}, pool=pool, solver_feedback=False)
    await pipe.run_round([Assignment("a", task_id, 0.5)], round_id="r")
    assert not any((fake.calls[0]["workspace"] / "past_rollouts" / "failed").iterdir())


def test_plan_assignments_uses_rollouts_and_removal_verdicts(tmp_path):
    pool, task_id = _pool_with_parent(tmp_path)
    for i in range(2):
        p = tmp_path / "src" / f"t{i}"
        write_task(p, f"ok{i}")
        pool.admit(HarborTask(p))
    removed = ReplacementRequest(task_id, "dir", 0.95, "too_easy")
    out = plan_assignments(pool, [removed], 3, round_id="r", rng=random.Random(0))
    assert len({a.assignment_id for a in out}) == 3
    first = out[0]
    assert first.parent_task_id == task_id and first.direction == "harder"
    assert first.solver_pass_rate == 0.5 and "test_a" in first.common_failure
    assert first.extra["removal_verdict"] == "too_easy"
    assert {a.parent_task_id for a in out[1:]} <= set(pool.entries) - {task_id}

    too_hard = ReplacementRequest(task_id, "dir", 0.1, "too_hard")
    blind = plan_assignments(pool, [too_hard], 1, round_id="r", rng=random.Random(0),
                             solver_feedback=False)
    assert blind[0].direction == "simpler" and blind[0].common_failure == ""
