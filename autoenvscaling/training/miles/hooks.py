"""Miles reward post-processing: the paper's advantage, A = r - mean(group), without std division.

Wired in with ``--custom-reward-post-process-path
autoenvscaling.training.miles.hooks.center_rewards_by_group``. Miles calls it as
``fn(args, samples) -> (raw_rewards, rewards)`` on the flat list of samples of one update, and
broadcasts `rewards` over each sample's response tokens.

This hook is required with DPPO: Miles centres rewards itself only for grpo, gspo and
reinforce_plus_plus_baseline, so under ``--advantage-estimator dppo`` raw rewards would be used
as advantages.

When the self-play loop produced the batch (`bridge` mode), each sample carries the advantage
`trajectories.export` computed with `rewards.group_advantages` over its whole group, including
group members that could not be tokenized; that value is used as is. Otherwise (Miles' own
rollouts in `generate_fn` mode) rewards are centred over samples sharing a `group_index`.

The module imports nothing from Miles, so it is testable on its own.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any


def _reward(sample: Any, args: Any) -> float:
    reward = sample.reward
    key = getattr(args, "reward_key", None)
    if key and isinstance(reward, dict):
        reward = reward[key]
    return float(reward)


def center_rewards_by_group(args: Any, samples: Sequence[Any]) -> tuple[list[float], list[float]]:
    raw = [_reward(s, args) for s in samples]
    given = [(s.metadata or {}).get("advantage") for s in samples]
    if samples and all(a is not None for a in given):
        return raw, [float(a) for a in given]

    members: dict[Any, list[int]] = defaultdict(list)
    for i, sample in enumerate(samples):
        members[sample.group_index].append(i)
    centred = list(raw)
    for indices in members.values():
        mean = sum(raw[i] for i in indices) / len(indices)
        for i in indices:
            centred[i] = raw[i] - mean
    return raw, centred
