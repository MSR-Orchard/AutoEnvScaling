import json
from pathlib import Path

import pytest

from autoenvscaling.calibration.solver import SolverEndpoint
from autoenvscaling.config import Settings
from autoenvscaling.pipeline import RoundResult
from autoenvscaling.pool import Pool
from autoenvscaling.selfplay import (
    FRONTIER,
    SHARED,
    STATIC,
    FixedEndpointTrainer,
    LoopConfig,
    Schedule,
    SelfPlayLoop,
    StepHandle,
)
from autoenvscaling.selfplay.schedule import GENERATE, PROPOSER_UPDATE, REVIEW
from autoenvscaling.tasks import REQUIRED_FILES, HarborTask
from autoenvscaling.trajectories.export import read_jsonl
from autoenvscaling.trajectories.schema import ProposerTrajectory, SolverTrajectory, Turn


# ---- schedule ------------------------------------------------------------------------------
def test_paper_schedule():
    s = Schedule(total_solver_steps=288)
    assert s.events_after(15) == ()
    assert s.events_after(16) == (REVIEW, GENERATE, PROPOSER_UPDATE)
    assert s.events_after(288) == ()                 # nothing after the final step
    assert s.proposer_updates == 17 and s.planned_updates == 305


def test_schedule_variants():
    assert Schedule(48, refresh_pool=False).events_after(16) == ()
    assert Schedule(48, train_proposer=False).events_after(32) == (REVIEW, GENERATE)
    s = Schedule(8, proposer_interval=4, pool_review_interval=2)
    assert [e for _, e in s.timeline() if e != "solver_update"] == \
        [REVIEW, REVIEW, GENERATE, PROPOSER_UPDATE, REVIEW]
    with pytest.raises(ValueError):
        Schedule(8, proposer_interval=0)


# ---- fakes ---------------------------------------------------------------------------------
def make_task(root: Path, name: str) -> HarborTask:
    for rel in REQUIRED_FILES:
        f = root / name / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(f"{rel} {name}\n")
    return HarborTask(root / name)


class FakeTrainer:
    """Each update index serves its own endpoint, so tests can check which weights were used."""

    def __init__(self):
        self.index = 0
        self.outstanding = None
        self.submitted = []
        self.closed = False

    async def acquire(self):
        if self.outstanding is None:
            self.outstanding = self.index
        return StepHandle(self.outstanding, SolverEndpoint(f"http://policy/{self.outstanding}", "policy"))

    async def submit(self, handle, groups, *, role):
        assert handle.update_index == self.outstanding
        self.submitted.append((handle.update_index, role, [g.group_id for g in groups]))
        self.outstanding = None
        self.index += 1

    async def close(self):
        self.closed = True


def small_settings() -> Settings:
    s = Settings()
    s.selfplay.solver_envs_per_batch = 2
    s.selfplay.solver_rollouts_per_env = 2
    s.selfplay.proposer_interval = 2
    s.selfplay.pool_review_interval = 2
    s.selfplay.proposer_group_size = 3
    return s


def seeded_pool(tmp_path, kinds: dict[str, str], capacity: int = 8) -> Pool:
    pool = Pool(tmp_path / "pool", capacity=capacity)
    for name, kind in kinds.items():
        pool.admit(make_task(tmp_path / "src", name), source=kind)
    pool.save()
    return pool


REWARDS = {"mixed": [0.0, 1.0], "solved": [1.0, 1.0], "failed": [0.0, 0.0]}


class Harness:
    """Fake solver, reviewer and proposer pipeline that record what they were asked to do."""

    def __init__(self, tmp_path, pool, review_verdicts=None, proposer_rewards=(1.0, -1.0, -0.25)):
        self.tmp_path = tmp_path
        self.pool = pool
        self.rollouts = []
        self.reviews = []
        self.rounds = []
        self.review_verdicts = review_verdicts or {}
        self.proposer_rewards = proposer_rewards

    async def rollout(self, entry, task, n, endpoint):
        self.rollouts.append((entry.task_id, n, endpoint.base_url))
        kind = entry.source if entry.source in REWARDS else "mixed"
        return [SolverTrajectory(f"{entry.task_id}-{i}", entry.task_id, "policy",
                                 turns=[Turn("user", "do"), Turn("assistant", "done")],
                                 reward=REWARDS[kind][i % 2], verifier_output="FAILED x")
                for i in range(n)]

    async def review(self, entry, task, endpoint):
        self.reviews.append((entry.task_id, endpoint.base_url))
        rewards = self.review_verdicts.get(entry.source, [0.0, 1.0, 0.5, 0.5])
        return [SolverTrajectory(f"rev-{entry.task_id}-{i}", entry.task_id, "policy", reward=r)
                for i, r in enumerate(rewards)]

    def pipeline_factory(self, proposer, endpoint):
        harness = self

        class FakePipeline:
            async def run_round(self, assignments, *, episodes_per_assignment, round_id, step):
                harness.rounds.append({"proposer": proposer, "endpoint": endpoint.base_url,
                                       "assignments": list(assignments),
                                       "episodes": episodes_per_assignment, "step": step})
                result = RoundResult(round_id, step)
                for a in assignments:
                    for s in range(episodes_per_assignment):
                        reward = harness.proposer_rewards[s % len(harness.proposer_rewards)]
                        result.trajectories.append(ProposerTrajectory(
                            f"{a.assignment_id}-{s}", a, proposer.model, proposer.harness,
                            turns=[Turn("user", "build"), Turn("assistant", "ok")], reward=reward))
                    task = make_task(harness.tmp_path / round_id, a.assignment_id)
                    result.admitted.append(harness.pool.admit(task, step=step, source="mixed"))
                return result

        return FakePipeline()

    def loop(self, settings, config, trainer, run_dir):
        return SelfPlayLoop(settings, config, pool=self.pool, trainer=trainer, run_dir=run_dir,
                            rollout_fn=self.rollout, review_fn=self.review,
                            pipeline_factory=self.pipeline_factory)


def metric_events(run_dir: Path) -> list[str]:
    with open(run_dir / "metrics.jsonl") as fh:
        return [json.loads(line)["event"] for line in fh]


# ---- loop ----------------------------------------------------------------------------------
async def test_shared_policy_schedule_and_endpoints(tmp_path):
    pool = seeded_pool(tmp_path, {"m1": "mixed", "m2": "mixed", "m3": "mixed", "s1": "solved",
                                  "easy": "easy"})
    h = Harness(tmp_path, pool, review_verdicts={"easy": [1.0, 1.0, 1.0, 1.0]})
    trainer = FakeTrainer()
    state = await h.loop(small_settings(), LoopConfig(total_solver_steps=5, mode=SHARED), trainer,
                         tmp_path / "run").run()

    roles = [role for _, role, _ in trainer.submitted]
    assert roles == ["solver", "solver", "proposer", "solver", "solver", "proposer", "solver"]
    assert state.solver_steps == 5 and state.proposer_updates == 2 and trainer.closed
    # Every solver batch is full, and contains only groups with reward variance.
    for _, role, groups in trainer.submitted:
        if role == "solver":
            assert len(groups) == 2
            assert all(pool.entries[g].source != "solved" for g in groups)
    # Rollouts for update i used the endpoint for update i.
    solver_indices = [i for i, role, _ in trainer.submitted if role == "solver"]
    assert {int(u.rsplit("/", 1)[1]) for _, _, u in h.rollouts} <= set(solver_indices)
    # The refresh after step 2 ran on the weights of update 2 (the first proposer update).
    first_round = h.rounds[0]
    assert first_round["endpoint"] == "http://policy/2" and first_round["step"] == 2
    assert first_round["proposer"].trainable and first_round["proposer"].base_url == "http://policy/2"
    assert first_round["episodes"] == 3 and len(first_round["assignments"]) == 1
    # The too-easy environment was removed and became the first assignment's parent.
    easy_id = next(e.task_id for e in pool.entries.values() if e.source == "easy")
    assert pool.entries[easy_id].status == "retired"
    assert first_round["assignments"][0].parent_task_id == easy_id
    assert first_round["assignments"][0].direction == "harder"
    assert {u for _, u in h.reviews if _ == easy_id} == {"http://policy/2"}

    events = metric_events(tmp_path / "run")
    assert events[0] == "start" and events[-1] == "done"
    assert events.count("proposer_update") == 2


async def test_frontier_mode_refills_open_slots_without_proposer_updates(tmp_path):
    pool = seeded_pool(tmp_path, {"m1": "mixed", "m2": "mixed", "e1": "easy", "e2": "easy"},
                       capacity=4)
    h = Harness(tmp_path, pool, review_verdicts={"easy": [1.0, 1.0, 1.0, 1.0]})
    trainer = FakeTrainer()
    config = LoopConfig(total_solver_steps=3, mode=FRONTIER, proposer_harness="claude-code",
                        proposer_model="claude-opus-5", solver_feedback=False)
    await h.loop(small_settings(), config, trainer, tmp_path / "run").run()

    assert [role for _, role, _ in trainer.submitted] == ["solver"] * 3
    (round_,) = h.rounds
    assert round_["proposer"].model == "claude-opus-5" and not round_["proposer"].trainable
    assert len(round_["assignments"]) == 2 and round_["episodes"] == 1
    assert all(a.common_failure == "" for a in round_["assignments"])
    # The refresh used the handle of update 2, which then served solver step 3.
    assert round_["endpoint"] == "http://policy/2"
    assert trainer.submitted[2][0] == 2


async def test_static_mode_never_refreshes(tmp_path):
    pool = seeded_pool(tmp_path, {"m1": "mixed", "m2": "mixed"})
    h = Harness(tmp_path, pool)
    trainer = FixedEndpointTrainer(SolverEndpoint("http://fixed/v1", "policy"), tmp_path / "batches")
    await h.loop(small_settings(), LoopConfig(total_solver_steps=4, mode=STATIC), trainer,
                 tmp_path / "run").run()
    assert [r for _, r, _ in trainer.submitted] == ["solver"] * 4
    assert h.reviews == [] and h.rounds == []
    rows = read_jsonl(tmp_path / "batches" / "batch_000000.jsonl")
    assert len(rows) == 4 and {r["role"] for r in rows} == {"solver"}


async def test_constant_proposer_groups_skip_the_update_and_reuse_the_handle(tmp_path):
    pool = seeded_pool(tmp_path, {"m1": "mixed", "m2": "mixed"})
    h = Harness(tmp_path, pool, proposer_rewards=(-1.0,))
    trainer = FakeTrainer()
    state = await h.loop(small_settings(), LoopConfig(total_solver_steps=3), trainer,
                         tmp_path / "run").run()
    assert [r for _, r, _ in trainer.submitted] == ["solver"] * 3
    assert state.proposer_updates == 0
    assert h.rounds[0]["endpoint"] == "http://policy/2"
    events = metric_events(tmp_path / "run")
    assert "proposer_update_skipped" in events


async def test_degenerate_pool_raises(tmp_path):
    pool = seeded_pool(tmp_path, {"s1": "solved", "f1": "failed"})
    h = Harness(tmp_path, pool)
    with pytest.raises(RuntimeError, match="degenerate"):
        await h.loop(small_settings(), LoopConfig(total_solver_steps=2), FakeTrainer(),
                     tmp_path / "run").run()


async def test_empty_pool_raises(tmp_path):
    pool = Pool(tmp_path / "pool")
    with pytest.raises(RuntimeError, match="seed it first"):
        await Harness(tmp_path, pool).loop(small_settings(), LoopConfig(total_solver_steps=1),
                                           FakeTrainer(), tmp_path / "run").run()


async def test_resume_continues_from_saved_state(tmp_path):
    pool = seeded_pool(tmp_path, {"m1": "mixed", "m2": "mixed"})
    h = Harness(tmp_path, pool)
    run_dir = tmp_path / "run"
    await h.loop(small_settings(), LoopConfig(total_solver_steps=1, mode=STATIC), FakeTrainer(),
                 run_dir).run()
    trainer = FakeTrainer()
    state = await h.loop(small_settings(), LoopConfig(total_solver_steps=3, mode=STATIC), trainer,
                         run_dir).run()
    assert state.solver_steps == 3 and len(trainer.submitted) == 2


def test_loop_config_validation():
    with pytest.raises(ValueError):
        LoopConfig(mode="bogus")
    with pytest.raises(ValueError):
        LoopConfig(mode=FRONTIER)
    assert not LoopConfig(mode=SHARED, train_proposer=False).trains_proposer


def test_cli_seed_prompts_schedule(tmp_path, capsys):
    from autoenvscaling.selfplay.__main__ import main

    tasks = [make_task(tmp_path / "src", f"t{i}").path for i in range(3)]
    assert main(["seed", "--pool-dir", str(tmp_path / "pool"), "--source", "tmax",
                 *map(str, tasks)]) == 0
    assert json.loads(capsys.readouterr().out)["active"] == 3
    assert main(["prompts", "--pool-dir", str(tmp_path / "pool"),
                 "--out", str(tmp_path / "prompts.jsonl")]) == 0
    assert len(read_jsonl(tmp_path / "prompts.jsonl")) == 3
    capsys.readouterr()
    assert main(["schedule", "--steps", "32"]) == 0
    assert "planned trainer updates (Miles --num-rollout): 33" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["run", "--pool-dir", str(tmp_path / "pool"), "--run-dir", str(tmp_path / "run")])
    with pytest.raises(SystemExit):
        main(["run", "--mode", "frontier", "--pool-dir", "p", "--run-dir", "r", "--bridge-dir", "b"])
