"""Turn a `MilesConfig` into a command, run it, and read back what it produced.

The launcher is a subprocess boundary: nothing here imports Miles. `plan()` returns the exact
command and environment, so a run that costs GPU-days can be checked before it starts.

Machine-specific locations come from the caller or the environment:

    MILES_DIR      Miles checkout with third_party/miles applied   (required to run)
    MEGATRON_DIR   Megatron-LM checkout inside the Miles image      (required to run)
    MILES_IMAGE    Apptainer image; when set, the job runs through launch/apptainer.sh
    AES_BINDS      bind mounts for Apptainer (see launch/apptainer.sh)
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autoenvscaling.training.miles.bridge import STOP_FILE
from autoenvscaling.training.miles.config import BRIDGE, CARRY, MilesConfig

# Variables the carry's Ray workers read (training.carry.rollout.CarrySettings and the proxy key).
CARRY_ENV_PREFIXES = ("AES_CARRY_", "AES_PROXY_API_KEY")

LAUNCH_DIR = Path(__file__).resolve().parent / "launch"
REPO_ROOT = Path(__file__).resolve().parents[3]
LATEST_FILE = "latest_checkpointed_iteration.txt"


@dataclass
class TrainingResult:
    ok: bool
    returncode: int
    checkpoint: str | None
    iteration: int | None
    detail: dict[str, Any] = field(default_factory=dict)


class ConfigError(ValueError):
    """The configuration would fail or train something other than what it declares."""


def latest_checkpoint(save_dir: Path) -> tuple[int, Path] | None:
    """The newest checkpoint Miles recorded under `save_dir`, as (iteration, directory)."""
    save_dir = Path(save_dir)
    marker = save_dir / LATEST_FILE
    if marker.exists():
        text = marker.read_text().strip()
        if text.isdigit():
            iteration = int(text)
            return iteration, save_dir / f"iter_{iteration:07d}"
    found = [(int(m.group(1)), d) for d in save_dir.glob("iter_*")
             if d.is_dir() and (m := re.fullmatch(r"iter_(\d+)", d.name))]
    return max(found) if found else None


class MilesLauncher:
    def __init__(self, *, miles_dir: str | None = None, megatron_dir: str | None = None,
                 image: str | None = None, binds: str | None = None,
                 bridge_dir: str | Path | None = None, forward_env: tuple[str, ...] = (),
                 env: dict[str, str] | None = None):
        base = dict(os.environ if env is None else env)
        self.base_env = base
        self.miles_dir = miles_dir or base.get("MILES_DIR", "")
        self.megatron_dir = megatron_dir or base.get("MEGATRON_DIR", "")
        self.image = image or base.get("MILES_IMAGE") or None
        self.binds = binds or base.get("AES_BINDS", "")
        self.bridge_dir = str(bridge_dir) if bridge_dir else None
        self.forward_env = forward_env

    def command(self, config: MilesConfig) -> list[str]:
        script = LAUNCH_DIR / ("apptainer.sh" if self.image else "train.sh")
        return ["bash", str(script), *config.train_args()]

    def environment(self, config: MilesConfig) -> dict[str, str]:
        """Only the variables the launch scripts read; the caller's environment is inherited."""
        env = {"MILES_DIR": self.miles_dir, "MEGATRON_DIR": self.megatron_dir,
               "MODEL_ARGS_SCRIPT": config.model_args_script, "AES_ROOT": str(REPO_ROOT),
               "NUM_GPUS": str(config.gpus_per_node)}
        if self.image:
            env["MILES_IMAGE"] = self.image
            env["AES_BINDS"] = self.binds
        if config.mode == BRIDGE:
            if not self.bridge_dir:
                raise ConfigError("bridge mode needs bridge_dir (the control directory the "
                                  "self-play loop also uses)")
            env["AES_BRIDGE_DIR"] = self.bridge_dir
        forward = list(self.forward_env)
        if config.mode == CARRY:
            # The carry settings travel as environment variables to the rollout workers.
            forward += [name for name in sorted(self.base_env)
                        if name.startswith(CARRY_ENV_PREFIXES) and name not in forward]
        if forward:
            env["AES_FORWARD_ENV"] = " ".join(forward)
        return env

    def problems(self, config: MilesConfig) -> list[str]:
        out = list(config.problems())
        if not self.miles_dir:
            out.append("MILES_DIR is not set")
        if not self.megatron_dir:
            out.append("MEGATRON_DIR is not set")
        if config.mode == BRIDGE and not self.bridge_dir:
            out.append("bridge mode needs bridge_dir")
        if config.mode == CARRY:
            from autoenvscaling.training.carry.rollout import CarrySettings
            out += [f"carry: {p}" for p in
                    CarrySettings.from_env(self.base_env).problems(config.n_samples_per_prompt)]
        return out

    def plan(self, config: MilesConfig) -> str:
        """The command as one shell line, with the variables it needs."""
        env = self.environment(config) if config.mode != BRIDGE or self.bridge_dir else {}
        assignments = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
        return f"{assignments} {shlex.join(self.command(config))}".strip()

    def run(self, config: MilesConfig, *, log_path: Path | None = None) -> TrainingResult:
        problems = self.problems(config)
        if problems:
            raise ConfigError("; ".join(problems))
        Path(config.save).mkdir(parents=True, exist_ok=True)
        env = {**self.base_env, **self.environment(config)}
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as log:
                proc = subprocess.run(self.command(config), env=env, stdout=log,
                                      stderr=subprocess.STDOUT, check=False)
        else:
            proc = subprocess.run(self.command(config), env=env, check=False)
        found = latest_checkpoint(Path(config.save))
        # Success needs a checkpoint on disk and a zero exit. In bridge mode the job also ends
        # (with an OrchestratorFinished error) when the self-play loop stops before Miles has
        # used all of num_rollout, which happens when a proposer update was skipped.
        stopped = (config.mode == BRIDGE and self.bridge_dir is not None
                   and (Path(self.bridge_dir) / STOP_FILE).exists())
        ok = found is not None and (proc.returncode == 0 or stopped)
        return TrainingResult(ok=ok, returncode=proc.returncode,
                              checkpoint=str(found[1]) if found else None,
                              iteration=found[0] if found else None,
                              detail={"num_rollout": config.num_rollout,
                                      "stopped_by_loop": stopped,
                                      "command": shlex.join(self.command(config))})
