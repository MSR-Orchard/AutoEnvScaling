"""Read the trials of a Harbor job from disk.

A Harbor job directory holds one subdirectory per trial, each with a ``result.json`` (Harbor's
``TrialResult``). Reading results from disk rather than from the in-memory job object means a
report can be rebuilt at any time, and several jobs (for example a rerun of failed tasks) can be
merged into one score.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from autoenvscaling.evals.scoring import (
    TIMEOUT_EXCEPTIONS,
    Attempt,
    classify,
)

# Exit codes the Pi agent reports when GNU timeout ended the run.
_TIMEOUT_EXIT_CODES = {124, 137, 143}


def trial_results(job_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Every trial ``result.json`` directly under ``job_dir``, with its trial directory."""
    found = []
    for path in sorted(Path(job_dir).glob("*/result.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and "trial_name" in data and "task_name" in data:
            found.append((path.parent, data))
    return found


def _reward(result: dict[str, Any], reward_key: str) -> float | None:
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    value = rewards.get(reward_key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def attempt_from_result(result: dict[str, Any], *, reward_key: str = "reward") -> Attempt:
    """Classify one Harbor ``TrialResult`` (as a dict) into an ``Attempt``."""
    exception = result.get("exception_info") or {}
    exception_type = exception.get("exception_type")
    reward = _reward(result, reward_key)
    metadata = (result.get("agent_result") or {}).get("metadata") or {}
    exit_code = metadata.get("pi_exit_code")
    status = classify(reward, exception_type)
    return Attempt(
        task=result["task_name"],
        status=status,
        reward=reward if status == "scored" else None,
        started_at=result.get("started_at"),
        exception_type=exception_type,
        timed_out=exception_type in TIMEOUT_EXCEPTIONS or exit_code in _TIMEOUT_EXIT_CODES,
        context_limited=bool(metadata.get("context_limited"))
        or exception_type == "ContextWindowExceededError",
        trial_name=result.get("trial_name"),
        extra={"model_id": metadata.get("model_id"), "pi_turns": metadata.get("pi_turns")},
    )


def load_attempts(job_dirs: Iterable[Path], *, reward_key: str = "reward") -> list[Attempt]:
    """All attempts from one or more Harbor job directories."""
    attempts: list[Attempt] = []
    for job_dir in job_dirs:
        for _, result in trial_results(Path(job_dir)):
            attempts.append(attempt_from_result(result, reward_key=reward_key))
    return attempts
