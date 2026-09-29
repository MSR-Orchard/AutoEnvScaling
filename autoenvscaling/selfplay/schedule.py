"""When things happen in shared-policy training (Section 3.3, Appendix "Training Settings").

The solver trains at every step. Every `proposer_interval` (k = 16) solver steps the proposer
generates new environments and, when it shares weights with the solver, trains on the resulting
proposer trajectories. The pool is reviewed every `pool_review_interval` (16) solver steps. With
the paper's settings both intervals coincide, so a refresh after solver step 16 is: review the
pool with the updated solver, generate replacements, then one proposer update.

Nothing is scheduled after the final solver step, because environments generated then would
never be trained on.
"""

from __future__ import annotations

from dataclasses import dataclass

REVIEW = "review"
GENERATE = "generate"
PROPOSER_UPDATE = "proposer_update"


@dataclass(frozen=True)
class Schedule:
    total_solver_steps: int
    proposer_interval: int = 16
    pool_review_interval: int = 16
    refresh_pool: bool = True        # False for a static pool (the Tmax baseline)
    train_proposer: bool = True      # False for fixed-weight proposers and the ablation

    def __post_init__(self) -> None:
        if self.total_solver_steps < 0 or self.proposer_interval < 1 or self.pool_review_interval < 1:
            raise ValueError("steps must be >= 0 and intervals >= 1")

    def events_after(self, step: int) -> tuple[str, ...]:
        """Events that follow solver step `step` (1-indexed), in the order they run."""
        if not self.refresh_pool or step >= self.total_solver_steps:
            return ()
        events = []
        if step % self.pool_review_interval == 0:
            events.append(REVIEW)
        if step % self.proposer_interval == 0:
            events.append(GENERATE)
            if self.train_proposer:
                events.append(PROPOSER_UPDATE)
        return tuple(events)

    @property
    def proposer_updates(self) -> int:
        return sum(PROPOSER_UPDATE in self.events_after(s) for s in range(1, self.total_solver_steps + 1))

    @property
    def planned_updates(self) -> int:
        """Solver plus proposer updates: the trainer's step budget (e.g. Miles --num-rollout).

        This is an upper bound. A generation round whose proposer groups all have constant
        rewards produces no proposer update.
        """
        return self.total_solver_steps + self.proposer_updates

    def timeline(self) -> list[tuple[int, str]]:
        """(solver step, event) pairs, with "solver_update" for the step itself."""
        out: list[tuple[int, str]] = []
        for step in range(1, self.total_solver_steps + 1):
            out.append((step, "solver_update"))
            out.extend((step, e) for e in self.events_after(step))
        return out
