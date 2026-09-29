"""One generation round: assignments in, scored proposer trajectories and admitted environments out.

For every assignment the round runs one or more independent proposer episodes. Each episode
ends with a Harbor task, which the host validates and, only if validation passes, calibrates
against the current solver. A task that fails either check goes back to the proposer with the
host's feedback (`calibration.feedback.revision_feedback`), at most
`settings.calibration.max_revisions` times. Every episode, first attempt or revision, becomes
one `ProposerTrajectory` with its reward r_P (`rewards.proposer_reward`). Tasks that pass both
checks are admitted to the training pool.

The same round serves both kinds of proposer:

* a frontier proposer with fixed weights (`ProposerSpec(harness, model)`), which improves only
  through its harness, typically one episode per assignment;
* the shared policy (`ProposerSpec.shared(endpoint)`), which runs in the Pi harness against
  the endpoint serving the current weights, with `proposer_group_size` episodes per assignment
  so the trajectories can form an RL group.

Workspace. Each round builds one workspace (`proposer.workspace.ProposerWorkspace`) from the
pool: the latest solver rollouts (left out when solver feedback is ablated), past environments,
and the proposer's memory and tools saved in `<out_dir>/proposer_state`. Every episode of the
round starts from it; a revision also gets the rejected task (`run_episode(previous_task_dir=)`)
so the proposer revises it in place. After each episode its memory and tools are saved to the
state directory (`proposer.workspace.save_persistent`), so the next round starts from the most
recently finished episode's copy.

Rubric review. `validate` grades the rubric with the proposer's own model
(`ReviewerSpec.for_proposer`), unless `settings.validation.reviewer_agent` or `reviewer_model`
pins a reviewer.

Failures of the host rather than the proposer are not scored: an exception from the episode,
validation or calibration, or a validation report flagged as an infrastructure error
(`ValidationReport.infra_error`), leaves the trajectory with `reward=None`, so it is never
trained on, and ends the chain without a revision.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import shutil
import time
from collections.abc import Awaitable, Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autoenvscaling.calibration.feedback import revision_feedback, should_revise
from autoenvscaling.calibration.solver import SolverEndpoint
from autoenvscaling.config import Settings
from autoenvscaling.pool import Pool, PoolEntry, ReplacementRequest
from autoenvscaling.proposer.workspace import ProposerWorkspace, save_persistent
from autoenvscaling.results import CalibrationReport, CheckResult, ValidationReport
from autoenvscaling.rewards import PROPOSER_ADMITTED, proposer_reward
from autoenvscaling.tasks import HarborTask
from autoenvscaling.trajectories.schema import Assignment, ProposerTrajectory
from autoenvscaling.validation.rubric import ReviewerSpec

TASK_FILES = "task_files"          # check name used when an episode ends without a complete task
PROPOSER_BASE_URL_ENV = "AES_PROPOSER_BASE_URL"
STATE_DIR = "proposer_state"       # memory/ and tools/ carried from round to round

EpisodeFn = Callable[..., Awaitable[ProposerTrajectory]]
ValidateFn = Callable[[HarborTask, Settings], Awaitable[ValidationReport]]
CalibrateFn = Callable[[HarborTask, SolverEndpoint, Settings], Awaitable[CalibrationReport]]
# (workspace_dir, pool, assignments, include_solver_rollouts, state_dir) -> workspace_dir
WorkspaceFn = Callable[[Path, "Pool | None", Sequence[Assignment], bool, "Path | None"], Path]
AssignmentBuilder = Callable[[Sequence[ReplacementRequest]], list[Assignment]]


@dataclass
class ProposerSpec:
    harness: str                     # "claude-code", "codex", "pi", or "auto"
    model: str
    trainable: bool = False          # True for the shared policy with open weights
    base_url: str | None = None      # OpenAI-compatible endpoint for open-weight models

    @classmethod
    def shared(cls, endpoint: SolverEndpoint, harness: str = "pi") -> ProposerSpec:
        """The policy that also plays the solver, served at `endpoint`."""
        return cls(harness=harness, model=endpoint.model, trainable=True, base_url=endpoint.base_url)


@dataclass
class RoundResult:
    round_id: str
    step: int
    trajectories: list[ProposerTrajectory] = field(default_factory=list)
    admitted: list[PoolEntry] = field(default_factory=list)
    admitted_tasks: list[HarborTask] = field(default_factory=list)
    trajectories_path: Path | None = None

    def summary(self) -> dict[str, Any]:
        scored = [t for t in self.trajectories if t.reward is not None]
        valid = [t for t in scored if t.reward > -1]
        return {"round_id": self.round_id, "step": self.step,
                "episodes": len(self.trajectories), "scored": len(scored),
                "infra_errors": len(self.trajectories) - len(scored),
                "valid_rate": len(valid) / len(scored) if scored else None,
                "admit_rate": (sum(t.reward == PROPOSER_ADMITTED for t in scored) / len(scored)
                               if scored else None),
                "admitted_tasks": len(self.admitted_tasks),
                "pool_admissions": len(self.admitted)}


# ---------------------------------------------------------------------------------------------
# Assignments and workspace
# ---------------------------------------------------------------------------------------------
def plan_assignments(pool: Pool, requests: Sequence[ReplacementRequest], n: int, *, round_id: str,
                     rng: random.Random, solver_feedback: bool = True,
                     builder: AssignmentBuilder | None = None) -> list[Assignment]:
    """Assignments for a round: replacement requests first, topped up from the active pool.

    Each assignment is built from the solver's latest rollouts on its parent
    (`ReplacementRequest.to_assignment`). Top-up parents are drawn uniformly from the active
    pool. Without solver feedback the common failure, which is read from solver rollouts, is
    cleared, matching the ablation that removes solver trajectories from the workspace.
    """
    chosen = list(requests[:n])
    taken = {r.parent_task_id for r in chosen}
    for entry in pool.sample(n - len(chosen), rng, exclude=taken) if n > len(chosen) else []:
        mean = entry.latest_mean if entry.latest_mean is not None else 0.5
        chosen.append(ReplacementRequest(parent_task_id=entry.task_id,
                                         parent_task_dir=str(pool.task_path(entry)),
                                         solver_pass_rate=mean, verdict="in_pool"))
    if builder is not None:
        assignments = builder(chosen)
    else:
        assignments = [r.to_assignment(f"{round_id}-a{i:03d}",
                                       rollouts=pool.latest_rollout_trajectories(r.parent_task_id))
                       for i, r in enumerate(chosen)]
    if not solver_feedback:
        for a in assignments:
            a.common_failure = ""
    return assignments


def build_workspace(workspace_dir: Path, pool: Pool | None, assignments: Sequence[Assignment],
                    include_solver_rollouts: bool, state_dir: Path | None = None, *,
                    max_environments: int = 64, rollouts_per_env: int = 16) -> Path:
    """Build the round's workspace with `ProposerWorkspace.build`.

    Past environments are the assignment parents plus the most recently admitted active
    environments, up to `max_environments`; past rollouts are the latest `rollouts_per_env`
    solver attempts on each of them, unless solver feedback is ablated.
    """
    environments: list[HarborTask] = []
    rollouts = []
    if pool is not None:
        parents = [a.parent_task_id for a in assignments]
        recent = [e.task_id for e in sorted(pool.active(), key=lambda e: (-e.added_step, e.task_id))]
        selected: list[str] = []
        for task_id in parents + recent:
            if task_id in pool.entries and task_id not in selected:
                selected.append(task_id)
        selected = selected[:max(max_environments, len(set(parents)))]
        environments = [pool.task(pool.entries[t]) for t in selected]
        if include_solver_rollouts:
            rollouts = [r for t in selected for r in pool.latest_rollout_trajectories(t, rollouts_per_env)]
    if Path(workspace_dir).exists():
        shutil.rmtree(workspace_dir)
    state = state_dir if state_dir is not None and Path(state_dir).is_dir() else None
    ProposerWorkspace.build(Path(workspace_dir), rollouts=rollouts, environments=environments,
                            state_dir=state)
    return Path(workspace_dir)


def reviewer_spec(settings: Settings, proposer: ProposerSpec) -> ReviewerSpec:
    """Who grades the rubric review: a pinned reviewer from settings, else the proposer's model."""
    v = settings.validation
    if v.reviewer_agent or v.reviewer_model:
        return ReviewerSpec(agent=v.reviewer_agent or proposer.harness,
                            model=v.reviewer_model or proposer.model)
    env = dict(os.environ)
    if proposer.base_url:
        env[PROPOSER_BASE_URL_ENV] = proposer.base_url
    return ReviewerSpec.for_proposer(proposer.harness, proposer.model, env=env)


@contextmanager
def _env_var(name: str, value: str | None) -> Iterator[None]:
    if value is None:
        yield
        return
    old = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


# ---------------------------------------------------------------------------------------------
# The round
# ---------------------------------------------------------------------------------------------
class GenerationPipeline:
    """Runs generation rounds.

    `pool` receives the admitted environments. `workspace_pool` supplies the past environments
    and solver rollouts for the proposer's workspace; it defaults to `pool` and differs when a
    new pool is built from the parents in a seed pool.
    """

    def __init__(self, settings: Settings, *, proposer: ProposerSpec, solver: SolverEndpoint,
                 out_dir: Path, pool: Pool | None = None, workspace_pool: Pool | None = None,
                 solver_feedback: bool = True, max_concurrent_chains: int = 16,
                 episode_fn: EpisodeFn | None = None, validate_fn: ValidateFn | None = None,
                 calibrate_fn: CalibrateFn | None = None, workspace_fn: WorkspaceFn | None = None):
        if episode_fn is None:
            from autoenvscaling.proposer.episode import run_episode as episode_fn
        if validate_fn is None:
            from autoenvscaling.validation.admission import validate

            reviewer = reviewer_spec(settings, proposer)

            async def validate_fn(task: HarborTask, settings: Settings) -> ValidationReport:
                return await validate(task, settings, reviewer=reviewer)
        if calibrate_fn is None:
            from autoenvscaling.calibration.band import calibrate as calibrate_fn
        self.settings = settings
        self.proposer = proposer
        self.solver = solver
        self.out_dir = Path(out_dir)
        self.pool = pool
        self.workspace_pool = workspace_pool if workspace_pool is not None else pool
        self.solver_feedback = solver_feedback
        self.max_concurrent_chains = max_concurrent_chains
        self.episode_fn = episode_fn
        self.validate_fn = validate_fn
        self.calibrate_fn = calibrate_fn
        self.workspace_fn = workspace_fn or build_workspace

    @property
    def max_revisions(self) -> int:
        return self.settings.calibration.max_revisions

    @property
    def state_dir(self) -> Path:
        return self.out_dir / STATE_DIR

    async def score(self, traj: ProposerTrajectory) -> tuple[ValidationReport, CalibrationReport | None]:
        """Validate the episode's task, and calibrate it only if validation passed."""
        task = HarborTask(Path(traj.task_dir)) if traj.task_dir else None
        if task is None or not task.path.is_dir() or not task.is_complete():
            missing = "no task was submitted" if task is None or not task.path.is_dir() else \
                f"missing files: {', '.join(task.missing_files())}"
            return ValidationReport(traj.trajectory_id, [CheckResult(TASK_FILES, False, missing)]), None
        traj.task_digest = task.digest()
        validation = await self.validate_fn(task, self.settings)
        if not validation.passed:
            return validation, None
        return validation, await self.calibrate_fn(task, self.solver, self.settings)

    async def run_chain(self, assignment: Assignment, *, sample: int, round_dir: Path,
                        workspace_dir: Path, sink: Callable[[ProposerTrajectory], None]
                        ) -> list[ProposerTrajectory]:
        """One episode plus up to `max_revisions` revisions, stopping at the first admission."""
        chain: list[ProposerTrajectory] = []
        chain_dir = round_dir / "episodes" / assignment.assignment_id / f"s{sample:03d}"
        feedback: str | None = None
        previous_task_dir: Path | None = None
        for revision in range(self.max_revisions + 1):
            trajectory_id = f"{assignment.assignment_id}-s{sample:03d}-r{revision}"
            episode_dir = chain_dir / f"r{revision}"
            started = time.monotonic()
            try:
                traj = await self.episode_fn(
                    assignment, harness=self.proposer.harness, model=self.proposer.model,
                    settings=self.settings, workspace_dir=workspace_dir, out_dir=episode_dir,
                    feedback=feedback, revision=revision, previous_task_dir=previous_task_dir)
            except Exception as exc:  # noqa: BLE001 - one broken episode must not stop the round
                traj = ProposerTrajectory(trajectory_id, assignment, self.proposer.model,
                                          self.proposer.harness, revision=revision,
                                          usage={"error": f"episode: {type(exc).__name__}: {exc}"})
                traj.wall_time_s = time.monotonic() - started
                chain.append(traj)
                sink(traj)
                break
            traj.trajectory_id = traj.trajectory_id or trajectory_id
            traj.revision = revision
            save_persistent(episode_dir, self.state_dir)
            try:
                validation, calibration = await self.score(traj)
            except Exception as exc:  # noqa: BLE001 - host checks hit infrastructure errors too
                traj.reward = None
                traj.usage = {**traj.usage, "error": f"admission: {type(exc).__name__}: {exc}"}
                chain.append(traj)
                sink(traj)
                break
            traj.validation = validation.to_dict()
            traj.calibration = calibration.to_dict() if calibration is not None else None
            traj.reward = None if validation.infra_error else proposer_reward(validation, calibration)
            chain.append(traj)
            sink(traj)
            if not should_revise(validation, calibration, revision, self.settings):
                break
            feedback = revision_feedback(validation, calibration)
            if traj.task_dir and Path(traj.task_dir).is_dir():
                previous_task_dir = Path(traj.task_dir)
        return chain

    async def run_round(self, assignments: Sequence[Assignment], *, episodes_per_assignment: int = 1,
                        round_id: str | None = None, step: int = 0) -> RoundResult:
        """Run every (assignment, sample) chain concurrently and admit what passes."""
        round_id = round_id or f"round-{step:06d}"
        round_dir = self.out_dir / round_id
        round_dir.mkdir(parents=True, exist_ok=True)
        workspace_dir = self.workspace_fn(round_dir / "workspace", self.workspace_pool, assignments,
                                          self.solver_feedback, self.state_dir)
        traj_path = round_dir / "proposer_trajectories.jsonl"
        traj_path.unlink(missing_ok=True)

        def sink(traj: ProposerTrajectory) -> None:
            with open(traj_path, "a", encoding="utf-8") as fh:
                fh.write(traj.to_json() + "\n")

        gate = asyncio.Semaphore(self.max_concurrent_chains)

        async def one(assignment: Assignment, sample: int) -> list[ProposerTrajectory]:
            async with gate:
                return await self.run_chain(assignment, sample=sample, round_dir=round_dir,
                                            workspace_dir=workspace_dir, sink=sink)

        jobs = [(a, s) for a in assignments for s in range(episodes_per_assignment)]
        with _env_var(PROPOSER_BASE_URL_ENV, self.proposer.base_url):
            chains = await asyncio.gather(*(one(a, s) for a, s in jobs))

        result = RoundResult(round_id=round_id, step=step, trajectories_path=traj_path)
        for chain in chains:
            result.trajectories.extend(chain)
            final = chain[-1] if chain else None
            if final is None or final.reward != PROPOSER_ADMITTED:
                continue
            task = HarborTask(Path(final.task_dir))
            result.admitted_tasks.append(task)
            if self.pool is not None:
                result.admitted.append(self.pool.admit(
                    task, step=step, source=f"proposer:{self.proposer.model}",
                    trajectory=final))
        if self.pool is not None:
            self.pool.save()
        (round_dir / "round.json").write_text(json.dumps(result.summary(), indent=2))
        return result
