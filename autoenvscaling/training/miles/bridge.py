"""A file handshake that lets the self-play loop drive a running Miles job.

Miles owns its training loop: for each rollout id it calls the configured rollout function,
trains one step on what that function returns, and pushes the new weights to its SGLang
engines. In `bridge` mode the rollout function is `training.miles.rollout.generate_rollout`,
which does not generate anything itself. It announces that the engines are awake with the
latest weights and waits for the self-play loop to deliver a finished batch:

    <control_dir>/ready_<id>.json    written by Miles: {"rollout_id", "router_url", "time"}
    <control_dir>/batch_<id>.jsonl   written by the loop (atomically): RL sample rows
    <control_dir>/stop.json          written by the loop when it has nothing more to train

While Miles waits, the loop runs solver attempts, pool reviews and proposer episodes against
the router, so every rollout comes from exactly the weights it is trained against. The loop
must not send requests after it writes a batch: Miles then moves the engines' memory to the
trainer, and a request that reaches an engine in that state can crash the job (the patch in
third_party/miles detaches engines from the router during that window).

`control_dir` must be on storage both processes can see.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from autoenvscaling.calibration.solver import SolverEndpoint
from autoenvscaling.selfplay.trainer import StepHandle
from autoenvscaling.trajectories.export import (
    RLGroup,
    read_jsonl,
    rl_sample_rows,
    write_jsonl_atomic,
)

STOP_FILE = "stop.json"
_READY_RE = re.compile(r"^ready_(\d+)\.json$")


def ready_path(control_dir: Path, rollout_id: int) -> Path:
    return Path(control_dir) / f"ready_{rollout_id:06d}.json"


def batch_path(control_dir: Path, rollout_id: int) -> Path:
    return Path(control_dir) / f"batch_{rollout_id:06d}.jsonl"


# ---- the Miles side (synchronous; runs inside Miles' rollout manager) --------------------------
def write_ready(control_dir: Path, rollout_id: int, router_url: str) -> None:
    path = ready_path(control_dir, rollout_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"rollout_id": rollout_id, "router_url": router_url, "time": time.time()}))
    tmp.replace(path)


def wait_for_batch(control_dir: Path, rollout_id: int, *, poll_s: float = 5.0,
                   timeout_s: float | None = None) -> list[dict[str, Any]] | None:
    """Block until the batch for `rollout_id` exists. Returns None if the loop stopped instead."""
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    batch = batch_path(control_dir, rollout_id)
    while True:
        if batch.exists():
            return read_jsonl(batch)
        if (Path(control_dir) / STOP_FILE).exists():
            return None
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError(f"no batch for rollout {rollout_id} in {control_dir} after {timeout_s}s")
        time.sleep(poll_s)


# ---- the loop side -------------------------------------------------------------------------------
def pending_ready(control_dir: Path) -> list[int]:
    """Rollout ids Miles is waiting on: announced, with no batch yet."""
    out = []
    for path in Path(control_dir).glob("ready_*.json"):
        m = _READY_RE.match(path.name)
        if m and not batch_path(control_dir, int(m.group(1))).exists():
            out.append(int(m.group(1)))
    return sorted(out)


class MilesBridgeTrainer:
    """`selfplay.trainer.Trainer` over the file handshake.

    `public_base_url` is the address under which sandboxes reach the Miles router (for example
    a tunnel); without it the router's own address is used, which works only when the sandboxes
    share its network.
    """

    def __init__(self, control_dir: Path, *, model: str = "policy", public_base_url: str | None = None,
                 api_key_env: str = "AES_SOLVER_API_KEY", poll_s: float = 5.0,
                 timeout_s: float | None = None):
        self.control_dir = Path(control_dir)
        self.control_dir.mkdir(parents=True, exist_ok=True)
        (self.control_dir / STOP_FILE).unlink(missing_ok=True)
        self.model = model
        self.public_base_url = public_base_url
        self.api_key_env = api_key_env
        self.poll_s = poll_s
        self.timeout_s = timeout_s
        self._outstanding: StepHandle | None = None

    def _endpoint(self, router_url: str) -> SolverEndpoint:
        base = self.public_base_url or f"{router_url.rstrip('/')}/v1"
        return SolverEndpoint(base_url=base, model=self.model, api_key_env=self.api_key_env)

    async def acquire(self) -> StepHandle:
        if self._outstanding is not None:
            return self._outstanding
        deadline = None if self.timeout_s is None else time.monotonic() + self.timeout_s
        while True:
            pending = pending_ready(self.control_dir)
            if pending:
                rollout_id = pending[0]
                info = json.loads(ready_path(self.control_dir, rollout_id).read_text())
                self._outstanding = StepHandle(rollout_id, self._endpoint(info["router_url"]))
                return self._outstanding
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError(f"Miles did not ask for a batch within {self.timeout_s}s")
            await asyncio.sleep(self.poll_s)

    async def submit(self, handle: StepHandle, groups: Sequence[RLGroup], *, role: str) -> None:
        if self._outstanding is None or handle.update_index != self._outstanding.update_index:
            raise RuntimeError(f"batch for rollout {handle.update_index} was not requested")
        rows = rl_sample_rows(groups)
        if not rows:
            raise ValueError("refusing to send an empty batch; Miles cannot train on it")
        if any(row["role"] != role for row in rows):
            raise ValueError(f"batch mixes roles; expected only {role!r}")
        write_jsonl_atomic(batch_path(self.control_dir, handle.update_index), rows)
        self._outstanding = None

    async def close(self) -> None:
        (self.control_dir / STOP_FILE).write_text(json.dumps({"time": time.time()}))
