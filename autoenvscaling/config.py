"""Run settings. Every default below is the value used in the paper.

Settings come from, in increasing priority: the defaults here, a TOML file passed with
``--config``, and ``AES_*`` environment variables (e.g. ``AES_SANDBOX_BACKEND=docker``).
Nothing in the package hard-codes a machine-specific path; paths are relative to
``work_dir`` unless given absolutely.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class SandboxSettings:
    backend: str = "modal"          # "modal" or "docker"
    harbor_env: str = "modal"       # Harbor environment type for validation/calibration runs
    proposer_image: str = "ghcr.io/msr-orchard/autoenvscaling-proposer:latest"
    proposer_cpu: int = 16          # Appendix, sandbox resources
    proposer_memory_gib: int = 32
    proposer_timeout_s: int = 7200  # 2-hour episode budget, no turn limit
    solver_cpu: int = 4
    solver_memory_gib: int = 8


@dataclass
class ValidationSettings:
    reference_min_reward: float = 1.0   # the reference solution must receive full reward
    empty_max_reward: float = 0.5       # an empty solution must receive less than this
    ngram_n: int = 13                   # contamination check, following Tmax
    # Held-out sets for the contamination check (Harbor registry names, comma-separated):
    # Terminal-Bench 2.1 and 4.0, as in the paper's decontamination appendix.
    heldout_dataset: str = "terminal-bench/terminal-bench-2-1,terminal-bench/terminal-bench@4.0.0"
    rubric: str = "terminal-bench"      # rubric used by `harbor check`
    # Criteria skipped in the rubric review, as in the paper's runs: binary_reward (our tests
    # give partial credit) and the contributor-metadata criteria, which generated tasks do not
    # carry (a README, time estimates, tags, explanations, CTRF reporting, a separate verifier).
    rubric_skip: tuple[str, ...] = (
        "binary_reward",
        "task_readme", "expert_time_estimate", "category_and_tags", "task_name",
        "difficulty_explanation_quality", "solution_explanation_quality",
        "verification_explanation_quality", "ctrf_reporting", "separate_verifier_configured",
    )
    # The rubric review is graded by the proposer's own model. Leave these empty to use the
    # proposer's harness and model; set them to pin a reviewer (e.g. for baselines).
    reviewer_agent: str = ""
    reviewer_model: str = ""


@dataclass
class CalibrationSettings:
    band_low: float = 0.25
    band_high: float = 0.75
    min_reward_std: float = 0.1
    initial_rollouts: int = 4
    max_rollouts: int = 8
    boundary_margin: float = 0.15       # expand to max_rollouts when the mean is this close to a band edge
    hint_probe_below_band: bool = True  # one diagnostic rollout with reference-solution hints
    max_revisions: int = 2


@dataclass
class SelfPlaySettings:
    learning_rate: float = 1e-6
    solver_envs_per_batch: int = 16
    solver_rollouts_per_env: int = 16
    proposer_group_size: int = 128      # proposer trajectories per assignment
    proposer_interval: int = 16         # k: proposer generates and trains every k solver steps
    pool_size: int = 256
    pool_review_interval: int = 16
    dppo_tv_threshold: float = 0.1


@dataclass
class Settings:
    work_dir: Path = Path("runs")
    sandbox: SandboxSettings = field(default_factory=SandboxSettings)
    validation: ValidationSettings = field(default_factory=ValidationSettings)
    calibration: CalibrationSettings = field(default_factory=CalibrationSettings)
    selfplay: SelfPlaySettings = field(default_factory=SelfPlaySettings)

    @classmethod
    def load(cls, path: str | Path | None = None, env: dict[str, str] | None = None) -> "Settings":
        settings = cls()
        if path is not None:
            with open(path, "rb") as fh:
                _apply(settings, tomllib.load(fh))
        _apply_env(settings, os.environ if env is None else env)
        settings.work_dir = Path(settings.work_dir)
        return settings


def _apply(obj: Any, data: dict[str, Any]) -> None:
    names = {f.name: f for f in fields(obj)}
    for key, value in data.items():
        if key not in names:
            raise KeyError(f"unknown setting {type(obj).__name__}.{key}")
        current = getattr(obj, key)
        if isinstance(value, dict):
            _apply(current, value)
        else:
            setattr(obj, key, _coerce(current, value))


def _apply_env(settings: Settings, env: dict[str, str]) -> None:
    """AES_<SECTION>_<NAME>=value, e.g. AES_CALIBRATION_BAND_LOW=0.3; AES_WORK_DIR for top level."""
    for key, raw in env.items():
        if not key.startswith("AES_"):
            continue
        parts = key[4:].lower()
        if parts == "work_dir":
            settings.work_dir = Path(raw)
            continue
        section, _, name = parts.partition("_")
        target = getattr(settings, section, None)
        if target is None or not hasattr(target, name):
            continue
        setattr(target, name, _coerce(getattr(target, name), raw))


def _coerce(current: Any, value: Any) -> Any:
    if isinstance(value, str):
        if isinstance(current, bool):
            return value.lower() in {"1", "true", "yes"}
        if isinstance(current, int):
            return int(value)
        if isinstance(current, float):
            return float(value)
        if isinstance(current, tuple):
            return tuple(v.strip() for v in value.split(",") if v.strip())
        if isinstance(current, Path):
            return Path(value)
    if isinstance(current, tuple) and isinstance(value, list):
        return tuple(value)
    return value
