"""Partial rollout and over-sampling for terminal episodes (Miles `carry` mode).

The paper's training runs did not wait for every episode of a batch: an unfinished episode was
*parked* at the weight update (sandbox and conversation kept alive) and *resumed* under the new
weights, keeping the token ids and log-probabilities already sampled; batches were filled by
over-sampling environments and discarding groups whose rewards did not vary. This package is
that implementation, taken from the research code and put on the release's interfaces:

    registry.py    parked episodes (sandbox + turn state), evicted exactly once
    episode.py     run or resume one episode; the sample repairs the trainer needs
    lifecycle.py   weight-version clock, group validity, the carry buffer, the generation gate
    rollout.py     the Miles rollout function: over-sampling, filtering, the cut, the buffer
    proxy.py       the recording proxy between Pi and the router (token ids and log-probs)
    driver.py      Pi in a sandbox of the task image, parked by holding its request

`docs/partial-rollout.md` separates what was verified from the final run, what is new in this
port, and what could not be verified.
"""

from autoenvscaling.training.carry.episode import (
    CarryConfig,
    EpisodeDriver,
    TurnOutcome,
    TurnStatus,
    run_episode_with_carry,
)
from autoenvscaling.training.carry.registry import EpisodeRegistry, EvictReason, ResumeBoundary
from autoenvscaling.training.carry.rollout import CarrySettings, collect_batch, generate_rollout

__all__ = ["CarryConfig", "CarrySettings", "EpisodeDriver", "EpisodeRegistry", "EvictReason",
           "ResumeBoundary", "TurnOutcome", "TurnStatus", "collect_batch", "generate_rollout",
           "run_episode_with_carry"]
