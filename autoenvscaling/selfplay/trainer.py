"""The boundary between the self-play loop and whatever updates the policy weights.

The loop owns the schedule, the pool and every rollout. A `Trainer` only does two things:

* `acquire()` waits until the trainer is ready for its next update and returns a handle whose
  `endpoint` serves the current weights. Everything the loop runs until it calls `submit`
  (solver attempts, pool reviews, proposer episodes and their calibration) uses that endpoint,
  so rollouts always come from the weights they are trained against.
* `submit(handle, groups, role)` hands over one batch of RL groups (see
  `trajectories.export.RLGroup`) and triggers exactly one policy update.

`autoenvscaling.training.miles.bridge.MilesBridgeTrainer` implements this for Miles.
`FixedEndpointTrainer` below serves a fixed endpoint and only records batches, for dry runs
and for testing the orchestration against a static inference server.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from autoenvscaling.calibration.solver import SolverEndpoint
from autoenvscaling.trajectories.export import RLGroup, write_rl_samples


@dataclass(frozen=True)
class StepHandle:
    update_index: int              # the trainer's own counter (Miles: the rollout id)
    endpoint: SolverEndpoint       # serves the weights the next update starts from


@runtime_checkable
class Trainer(Protocol):
    async def acquire(self) -> StepHandle: ...

    async def submit(self, handle: StepHandle, groups: Sequence[RLGroup], *, role: str) -> None: ...

    async def close(self) -> None: ...


class FixedEndpointTrainer:
    """Never changes the weights; writes each submitted batch to `out_dir/batch_<n>.jsonl`."""

    def __init__(self, endpoint: SolverEndpoint, out_dir: Path | None = None):
        self.endpoint = endpoint
        self.out_dir = Path(out_dir) if out_dir else None
        self.next_index = 0
        self.submitted: list[tuple[int, str, int]] = []   # (update index, role, groups)
        self._outstanding: int | None = None

    async def acquire(self) -> StepHandle:
        if self._outstanding is None:
            self._outstanding = self.next_index
        return StepHandle(self._outstanding, self.endpoint)

    async def submit(self, handle: StepHandle, groups: Sequence[RLGroup], *, role: str) -> None:
        if handle.update_index != self._outstanding:
            raise RuntimeError(f"stale handle {handle.update_index}, expected {self._outstanding}")
        if self.out_dir is not None:
            write_rl_samples(self.out_dir / f"batch_{handle.update_index:06d}.jsonl", groups)
        self.submitted.append((handle.update_index, role, len(groups)))
        self._outstanding = None
        self.next_index += 1

    async def close(self) -> None:
        return None
