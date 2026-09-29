"""Turn per-attempt results into benchmark scores.

The paper reports avg@k on Terminal-Bench (k = 5): each task is attempted k times, each attempt
is scored by the task's verifier, and the score is the mean reward over all task-attempts. This
module computes that number together with pass@k and the counts a reader needs to trust it.

Three rules decide what counts as an attempt:

* An attempt is **scored** when the verifier produced a numeric reward. An agent timeout is a
  real model outcome: the verifier still scores the state the agent left behind.
* An attempt is an **infrastructure error** when the trial failed for a reason outside the
  model (sandbox, network, provider API). It is never scored as reward 0, and retrying it does
  not increase k: a task that needed three sandboxes to produce one clean attempt has still had
  one attempt.
* An attempt is **verifier broken** when the trial finished without an exception but the
  verifier reported no reward. That is a harness failure, not a model score of 0.

Per task, the first k scored attempts (by start time) are used. Tasks with fewer than k scored
attempts are reported as incomplete so they can be rerun.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

SCORED = "scored"
INFRA_ERROR = "infra_error"
VERIFIER_BROKEN = "verifier_broken"

# Exceptions after which the verifier's reward is still the model's outcome. Harbor records the
# exception and then runs the verifier anyway.
MODEL_OUTCOME_EXCEPTIONS = frozenset({
    "AgentTimeoutError",
    "ContextWindowExceededError",
    "OutputTokenExceededError",
    "AgentSafetyRefusalError",
})
TIMEOUT_EXCEPTIONS = frozenset({"AgentTimeoutError"})


def classify(reward: float | None, exception_type: str | None) -> str:
    """Status of one trial from its verifier reward and the exception Harbor recorded."""
    if exception_type is None:
        return SCORED if reward is not None else VERIFIER_BROKEN
    if exception_type in MODEL_OUTCOME_EXCEPTIONS and reward is not None:
        return SCORED
    return INFRA_ERROR


@dataclass
class Attempt:
    """One trial of one task."""

    task: str
    status: str
    reward: float | None = None
    started_at: str | None = None
    exception_type: str | None = None
    timed_out: bool = False
    context_limited: bool = False
    trial_name: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_scored(self) -> bool:
        return self.status == SCORED and self.reward is not None


@dataclass
class TaskScore:
    task: str
    rewards: list[float]
    n_infra_errors: int = 0
    n_verifier_broken: int = 0

    @property
    def mean(self) -> float | None:
        return statistics.fmean(self.rewards) if self.rewards else None

    def passed(self, threshold: float = 1.0) -> bool:
        return any(r >= threshold for r in self.rewards)


@dataclass
class ScoreReport:
    k: int
    avg_at_k: float | None
    pass_at_k: float | None
    n_tasks: int
    n_tasks_scored: int
    n_tasks_incomplete: int
    n_attempts_scored: int
    n_infra_errors: int
    n_verifier_broken: int
    n_timed_out: int
    n_context_limited: int
    # Mean reward of the i-th scored attempt across tasks, for i in 0..k-1. The spread of these
    # k numbers is the run-to-run variation of a pass@1 score.
    per_attempt_means: list[float] = field(default_factory=list)
    run_std: float | None = None
    task_stderr: float | None = None
    missing: str = "exclude"
    tasks: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_attempts(attempts: Iterable[Attempt], k: int) -> dict[str, TaskScore]:
    """Group attempts by task and keep the first k scored attempts of each."""
    if k < 1:
        raise ValueError("k must be at least 1")
    by_task: dict[str, list[Attempt]] = defaultdict(list)
    for attempt in attempts:
        by_task[attempt.task].append(attempt)
    scores: dict[str, TaskScore] = {}
    for task, group in sorted(by_task.items()):
        scored = sorted((a for a in group if a.is_scored), key=lambda a: a.started_at or "")
        scores[task] = TaskScore(
            task=task,
            rewards=[float(a.reward) for a in scored[:k]],  # type: ignore[arg-type]
            n_infra_errors=sum(a.status == INFRA_ERROR for a in group),
            n_verifier_broken=sum(a.status == VERIFIER_BROKEN for a in group),
        )
    return scores


def summarize(attempts: Iterable[Attempt], k: int, *,
              missing: Literal["exclude", "zero"] = "exclude",
              pass_threshold: float = 1.0,
              expected_tasks: Iterable[str] | None = None) -> ScoreReport:
    """avg@k, pass@k and harness-health counts over a set of attempts.

    ``missing`` decides what happens to a task with fewer than k scored attempts: "exclude"
    averages over the attempts it has (a task with none drops out of the score), "zero" fills
    the missing attempts with reward 0. Either way the task is counted in
    ``n_tasks_incomplete``. ``expected_tasks`` adds tasks that produced no trial at all.
    """
    if missing not in ("exclude", "zero"):
        raise ValueError(f"missing must be 'exclude' or 'zero', got {missing!r}")
    attempts = list(attempts)
    scores = select_attempts(attempts, k)
    for task in expected_tasks or ():
        scores.setdefault(task, TaskScore(task=task, rewards=[]))

    task_means: list[float] = []
    passes: list[int] = []
    columns: list[list[float]] = [[] for _ in range(k)]
    incomplete = 0
    for score in scores.values():
        rewards = list(score.rewards)
        if len(rewards) < k:
            incomplete += 1
            if missing == "zero":
                rewards += [0.0] * (k - len(rewards))
        if not rewards:
            continue
        task_means.append(statistics.fmean(rewards))
        passes.append(int(any(r >= pass_threshold for r in rewards)))
        for i, r in enumerate(rewards[:k]):
            columns[i].append(r)

    per_attempt = [statistics.fmean(c) for c in columns if c]
    return ScoreReport(
        k=k,
        avg_at_k=statistics.fmean(task_means) if task_means else None,
        pass_at_k=statistics.fmean(passes) if passes else None,
        n_tasks=len(scores),
        n_tasks_scored=len(task_means),
        n_tasks_incomplete=incomplete,
        n_attempts_scored=sum(len(s.rewards) for s in scores.values()),
        n_infra_errors=sum(a.status == INFRA_ERROR for a in attempts),
        n_verifier_broken=sum(a.status == VERIFIER_BROKEN for a in attempts),
        n_timed_out=sum(a.timed_out for a in attempts if a.is_scored),
        n_context_limited=sum(a.context_limited for a in attempts if a.is_scored),
        per_attempt_means=per_attempt,
        run_std=statistics.stdev(per_attempt) if len(per_attempt) > 1 else None,
        task_stderr=(statistics.stdev(task_means) / math.sqrt(len(task_means))
                     if len(task_means) > 1 else None),
        missing=missing,
        tasks=[{"task": s.task, "rewards": s.rewards, "mean": s.mean,
                "passed": s.passed(pass_threshold), "n_infra_errors": s.n_infra_errors,
                "n_verifier_broken": s.n_verifier_broken} for s in scores.values()],
    )
