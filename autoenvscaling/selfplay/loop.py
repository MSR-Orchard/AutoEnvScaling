"""The training loop: solver RL on the pool, pool refreshes, and proposer rounds.

One loop covers the three setups in the paper:

* `mode="shared"`: one policy plays both roles (Sections 3.3 and 4.3). The solver trains every
  step; after every k solver steps the pool is reviewed with the updated solver, the policy (as
  proposer, in the Pi harness) generates replacements, and it trains on those proposer
  trajectories. `train_proposer=False` keeps the shared weights but removes the proposer
  update, which is the "without proposer training" run in Section 4.3.
* `mode="frontier"`: a proposer with fixed weights (Claude Opus 5, GPT-5.6-sol) refills the
  pool at each refresh, one episode per open slot; only the solver trains (Section 4.2).
  `solver_feedback=False` removes solver trajectories from the proposer's workspace.
* `mode="static"`: no refresh at all. Training on a fixed Tmax pool, or on a pool a frontier
  proposer generated ahead of time, is this mode.

Solver batches follow the appendix: 16 environments x 16 attempts, oversampling environments
and discarding groups whose rewards do not vary until the batch is filled.

Rollouts, reviews and proposer episodes are injectable (`rollout_fn`, `review_fn`,
`pipeline_factory`) so the schedule can be tested without sandboxes. The defaults run the
solver through `calibration.solver.run_solver` and the proposer through
`pipeline.GenerationPipeline`.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from autoenvscaling.calibration.solver import SolverEndpoint
from autoenvscaling.config import Settings
from autoenvscaling.pipeline import (
    AssignmentBuilder,
    GenerationPipeline,
    ProposerSpec,
    plan_assignments,
)
from autoenvscaling.pool import Pool, PoolEntry, ReplacementRequest
from autoenvscaling.selfplay.schedule import GENERATE, PROPOSER_UPDATE, REVIEW, Schedule
from autoenvscaling.selfplay.trainer import StepHandle, Trainer
from autoenvscaling.tasks import HarborTask
from autoenvscaling.trajectories.export import (
    PROPOSER,
    SOLVER,
    ChatFormat,
    GroupingStats,
    RLGroup,
    proposer_rl_groups,
    solver_rl_groups,
)
from autoenvscaling.trajectories.schema import SolverTrajectory

SHARED = "shared"
FRONTIER = "frontier"
STATIC = "static"
MODES = (SHARED, FRONTIER, STATIC)

RolloutFn = Callable[[PoolEntry, HarborTask, int, SolverEndpoint], Awaitable[list[SolverTrajectory]]]
ReviewFn = Callable[[PoolEntry, HarborTask, SolverEndpoint], Awaitable[list[SolverTrajectory]]]
PipelineFactory = Callable[[ProposerSpec, SolverEndpoint], GenerationPipeline]


@dataclass
class LoopConfig:
    total_solver_steps: int = 288
    mode: str = SHARED
    train_proposer: bool = True            # shared mode only
    proposer_harness: str = "pi"           # the shared policy always runs in Pi
    proposer_model: str = ""               # frontier mode: the fixed-weight proposer
    solver_feedback: bool = True
    oversample_envs: int = 8               # extra environments drawn per sampling wave
    max_sampling_waves: int = 3
    max_empty_solver_steps: int = 3        # consecutive batches with no usable group before giving up
    review_rollouts: int | None = None     # default: settings.calibration.initial_rollouts
    require_variance: bool = True          # review keeps environments with std >= min_reward_std only
    proposer_groups_per_round: int = 1     # shared mode: assignments per round, each a group
    frontier_episodes_per_assignment: int = 1
    max_concurrent_solver_envs: int = 32
    max_concurrent_chains: int = 16
    seed: int = 0
    # The harnesses' system prompts and tool schemas, used when re-tokenizing trajectories.
    solver_chat: ChatFormat | None = None
    proposer_chat: ChatFormat | None = None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.mode == FRONTIER and not self.proposer_model:
            raise ValueError("frontier mode needs proposer_model")

    @property
    def trains_proposer(self) -> bool:
        return self.mode == SHARED and self.train_proposer


@dataclass
class LoopState:
    solver_steps: int = 0
    proposer_updates: int = 0
    rounds: int = 0

    @classmethod
    def load(cls, path: Path) -> LoopState:
        return cls(**json.loads(path.read_text())) if path.exists() else cls()

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self)))
        tmp.replace(path)


@dataclass
class SolverBatch:
    groups: list[RLGroup] = field(default_factory=list)
    stats: GroupingStats = field(default_factory=GroupingStats)
    envs_tried: int = 0
    waves: int = 0
    infra_errors: int = 0
    extra_groups_discarded: int = 0


class SelfPlayLoop:
    def __init__(self, settings: Settings, config: LoopConfig, *, pool: Pool, trainer: Trainer,
                 run_dir: Path, rollout_fn: RolloutFn | None = None,
                 review_fn: ReviewFn | None = None,
                 pipeline_factory: PipelineFactory | None = None,
                 assignment_builder: AssignmentBuilder | None = None):
        self.settings = settings
        self.config = config
        self.pool = pool
        self.trainer = trainer
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.rollout_fn = rollout_fn or self._run_solver
        self.review_fn = review_fn or self._review_solver
        self.pipeline_factory = pipeline_factory or self._make_pipeline
        self.assignment_builder = assignment_builder
        sp = settings.selfplay
        self.schedule = Schedule(total_solver_steps=config.total_solver_steps,
                                 proposer_interval=sp.proposer_interval,
                                 pool_review_interval=sp.pool_review_interval,
                                 refresh_pool=config.mode != STATIC,
                                 train_proposer=config.trains_proposer)
        self.state_path = self.run_dir / "state.json"
        self.state = LoopState.load(self.state_path)
        self.rng = random.Random(config.seed + self.state.solver_steps)
        self.pending_requests: list[ReplacementRequest] = []

    # ---- defaults backed by the real solver and proposer ------------------------------------
    async def _run_solver(self, entry: PoolEntry, task: HarborTask, n: int,
                          endpoint: SolverEndpoint) -> list[SolverTrajectory]:
        from autoenvscaling.calibration.solver import run_solver
        return await run_solver(task, endpoint, n=n, harbor_env=self.settings.sandbox.harbor_env)

    async def _review_solver(self, entry: PoolEntry, task: HarborTask,
                             endpoint: SolverEndpoint) -> list[SolverTrajectory]:
        n = self.config.review_rollouts or self.settings.calibration.initial_rollouts
        return await self.rollout_fn(entry, task, n, endpoint)

    def _make_pipeline(self, proposer: ProposerSpec, solver: SolverEndpoint) -> GenerationPipeline:
        return GenerationPipeline(self.settings, proposer=proposer, solver=solver,
                                  out_dir=self.run_dir / "rounds", pool=self.pool,
                                  solver_feedback=self.config.solver_feedback,
                                  max_concurrent_chains=self.config.max_concurrent_chains)

    def proposer_spec(self, endpoint: SolverEndpoint) -> ProposerSpec:
        if self.config.mode == SHARED:
            return ProposerSpec.shared(endpoint, harness=self.config.proposer_harness)
        return ProposerSpec(self.config.proposer_harness, self.config.proposer_model)

    # ---- bookkeeping ------------------------------------------------------------------------
    def log(self, event: str, **data: Any) -> None:
        row = {"time": time.time(), "event": event, **data}
        with open(self.run_dir / "metrics.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")

    # ---- solver step ------------------------------------------------------------------------
    async def collect_solver_batch(self, endpoint: SolverEndpoint, step: int) -> SolverBatch:
        """Draw environments in waves until `solver_envs_per_batch` groups have reward variance."""
        sp = self.settings.selfplay
        batch = SolverBatch()
        tried: set[str] = set()
        gate = asyncio.Semaphore(self.config.max_concurrent_solver_envs)

        async def attempt(entry: PoolEntry) -> list[SolverTrajectory] | None:
            async with gate:
                try:
                    return await self.rollout_fn(entry, self.pool.task(entry),
                                                 sp.solver_rollouts_per_env, endpoint)
                except Exception:  # noqa: BLE001 - an infrastructure failure costs one environment
                    return None

        while len(batch.groups) < sp.solver_envs_per_batch and batch.waves < self.config.max_sampling_waves:
            need = sp.solver_envs_per_batch - len(batch.groups)
            entries = self.pool.sample(need + self.config.oversample_envs, self.rng, exclude=tried)
            if not entries:
                break
            batch.waves += 1
            batch.envs_tried += len(entries)
            tried.update(e.task_id for e in entries)
            results = await asyncio.gather(*(attempt(e) for e in entries))
            trajectories: list[SolverTrajectory] = []
            for entry, trajs in zip(entries, results):
                if trajs is None:
                    batch.infra_errors += 1
                    continue
                self.pool.record_rollouts(entry.task_id, trajs, step)
                trajectories.extend(trajs)
            groups, stats = solver_rl_groups(trajectories, chat=self.config.solver_chat)
            batch.stats = batch.stats.merge(stats)
            order = {e.task_id: i for i, e in enumerate(entries)}
            groups.sort(key=lambda g: order[g.group_id])      # keep the random draw order
            batch.groups.extend(groups[:need])
            batch.extra_groups_discarded += max(0, len(groups) - need)
        self.pool.save()
        return batch

    # ---- refresh ----------------------------------------------------------------------------
    async def refresh(self, endpoint: SolverEndpoint, step: int, events: Sequence[str]) -> list[RLGroup]:
        """Review and/or generate. Returns proposer groups to train on (possibly empty)."""
        if REVIEW in events:
            outcome = await self.pool.review(
                lambda e: self.review_fn(e, self.pool.task(e), endpoint), step,
                calibration=self.settings.calibration,
                concurrency=self.config.max_concurrent_solver_envs,
                require_variance=self.config.require_variance)
            self.pending_requests.extend(outcome.requests)
            self.log("review", step=step, kept=len(outcome.kept), removed=outcome.removed,
                     promoted=outcome.promoted, pool=self.pool.summary())
        self.pool.save()
        if GENERATE not in events:
            return []
        return await self.generate(endpoint, step, train=PROPOSER_UPDATE in events)

    async def generate(self, endpoint: SolverEndpoint, step: int, *, train: bool) -> list[RLGroup]:
        proposer = self.proposer_spec(endpoint)
        if self.config.mode == SHARED:
            n_assignments = self.config.proposer_groups_per_round
            episodes = self.settings.selfplay.proposer_group_size
        else:
            n_assignments = self.pool.open_slots
            episodes = self.config.frontier_episodes_per_assignment
        requests, self.pending_requests = self.pending_requests, []
        if n_assignments <= 0:
            self.log("generate", step=step, skipped="no open slots")
            return []
        round_id = f"round-{step:06d}"
        assignments = plan_assignments(self.pool, requests, n_assignments, round_id=round_id,
                                       rng=self.rng, solver_feedback=self.config.solver_feedback,
                                       builder=self.assignment_builder)
        if not assignments:
            self.log("generate", step=step, skipped="no parents")
            return []
        pipeline = self.pipeline_factory(proposer, endpoint)
        result = await pipeline.run_round(assignments, episodes_per_assignment=episodes,
                                          round_id=round_id, step=step)
        self.state.rounds += 1
        groups: list[RLGroup] = []
        stats = GroupingStats()
        if train:
            groups, stats = proposer_rl_groups(result.trajectories, chat=self.config.proposer_chat)
        self.log("generate", step=step, proposer=asdict(proposer), round=result.summary(),
                 proposer_groups=stats.to_dict(), pool=self.pool.summary())
        return groups

    # ---- main loop --------------------------------------------------------------------------
    async def run(self) -> LoopState:
        if not self.pool.active():
            raise RuntimeError(f"pool at {self.pool.root} has no active environments; seed it first")
        self.log("start", config=asdict(self.config), schedule=asdict(self.schedule),
                 planned_updates=self.schedule.planned_updates, state=asdict(self.state))
        handle: StepHandle | None = None
        empty = 0
        while self.state.solver_steps < self.config.total_solver_steps:
            handle = handle or await self.trainer.acquire()
            step = self.state.solver_steps + 1
            batch = await self.collect_solver_batch(handle.endpoint, step)
            if not batch.groups:
                empty += 1
                self.log("solver_step_empty", step=step, envs_tried=batch.envs_tried,
                         infra_errors=batch.infra_errors, groups=batch.stats.to_dict())
                if empty >= self.config.max_empty_solver_steps:
                    raise RuntimeError(f"no solver group with reward variance in {empty} tries at "
                                       f"step {step}; the pool is degenerate for this solver")
                continue
            empty = 0
            await self.trainer.submit(handle, batch.groups, role=SOLVER)
            update_index, handle = handle.update_index, None
            self.state.solver_steps = step
            self.state.save(self.state_path)
            rewards = [r for g in batch.groups for r in g.rewards]
            self.log("solver_update", step=step, update_index=update_index,
                     groups=len(batch.groups), mean_reward=sum(rewards) / len(rewards),
                     envs_tried=batch.envs_tried, waves=batch.waves,
                     infra_errors=batch.infra_errors,
                     extra_groups_discarded=batch.extra_groups_discarded,
                     grouping=batch.stats.to_dict())

            events = self.schedule.events_after(step)
            if not events:
                continue
            handle = await self.trainer.acquire()
            proposer_groups = await self.refresh(handle.endpoint, step, events)
            if proposer_groups:
                await self.trainer.submit(handle, proposer_groups, role=PROPOSER)
                update_index, handle = handle.update_index, None
                self.state.proposer_updates += 1
                self.state.save(self.state_path)
                self.log("proposer_update", step=step, update_index=update_index,
                         groups=len(proposer_groups),
                         samples=sum(len(g.samples) for g in proposer_groups))
            elif PROPOSER_UPDATE in events:
                self.log("proposer_update_skipped", step=step,
                         reason="no proposer group with reward variance")
        await self.trainer.close()
        self.log("done", state=asdict(self.state))
        return self.state
