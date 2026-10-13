"""Lifecycle hooks at Miles' rollout and buffer boundaries: the weight-version clock, group
validity, the carry buffer and the generation gate.

A sample group is atomic for the group-relative objective: its token buffer and the live
sandbox registry must describe the same episodes, and a failed member invalidates the whole
group. Everything here is decided per group.

Weight versions
---------------
Miles enforces ``--max-weight-staleness`` only in its fully asynchronous data buffer; under the
colocated, synchronous rollout the flag is inert, so the carry owns the bound. The clock is
anchored to the first rollout id this process sees (Miles continues rollout numbering across a
checkpoint resume, and only version *differences* matter), and "current version" advances by
one per rollout because colocation runs one optimizer step per rollout.

This is the research code's ``spare/miles/pi_rollout.py`` with the proxy control behind a small
protocol so it can be faked in tests.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import threading
import urllib.error
import urllib.request
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from autoenvscaling.training.carry.episode import episode_id, get_registry
from autoenvscaling.training.carry.registry import EpisodeRegistry, EvictReason

logger = logging.getLogger(__name__)

QUIESCE_ATTEMPTS = 3
QUIESCE_ABORT_INTERVAL_S = 1.0
_STILL_ACTIVE = re.compile(r"\d+ engine calls still active after abort")


# ---- weight-version clock ------------------------------------------------------------------------
@dataclass
class WeightClock:
    base_rollout_id: int | None = None
    active_version: int | None = None

    def begin(self, rollout_id: int) -> int:
        if self.base_rollout_id is None:
            self.base_rollout_id = int(rollout_id)
        self.active_version = int(rollout_id) - self.base_rollout_id
        return self.active_version

    def current_version(self) -> int:
        return 0 if self.active_version is None else self.active_version


CLOCK = WeightClock()


# ---- the generation gate (the proxy in front of the router) -------------------------------------
class GateControl(Protocol):
    """Park/unpark/quiesce the generation gate that holds agents' in-flight requests.

    ``park()`` closes the global gate before the engines are aborted and put to sleep;
    ``quiesce()`` returns once every admitted engine call has settled (it raises with a
    "N engine calls still active after abort" message while some are); ``unpark()`` reopens the
    global gate once the engines serve the new weights; ``check()`` verifies the gate speaks the
    carry lifecycle protocol.
    """

    def park(self) -> None: ...

    def unpark(self) -> None: ...

    def quiesce(self) -> dict: ...

    def check(self) -> None: ...


class StillActive(RuntimeError):
    """The gate reports engine calls still in flight after an abort sweep."""


class HttpGate:
    """The proxy's control endpoints over HTTP (``training.carry.proxy``)."""

    def __init__(self, base_url: str, timeout_s: float = 30.0, api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.api_key = api_key

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post(self, action: str) -> dict:
        req = urllib.request.Request(f"{self.base_url}/{action}", data=b"{}", headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 503:
                try:
                    detail = json.loads(exc.read()).get("detail")
                except (ValueError, AttributeError):
                    detail = None
                if isinstance(detail, str) and _STILL_ACTIVE.fullmatch(detail):
                    raise StillActive(detail) from exc
            raise

    def park(self) -> None:
        self._post("park")

    def unpark(self) -> None:
        self._post("unpark")

    def quiesce(self) -> dict:
        return self._post("quiesce")

    def check(self) -> None:
        req = urllib.request.Request(f"{self.base_url}/config", headers=self._headers())
        with urllib.request.urlopen(req, timeout=self.timeout_s) as response:
            config = json.loads(response.read())
        if config.get("carry_lifecycle_version", 0) < 1:
            raise RuntimeError("the proxy does not implement the carry lifecycle (version >= 1)")


async def after_engine_abort(gate: GateControl, *,
                             abort_engines: Callable[[], Awaitable[None]] | None = None) -> None:
    """Confirm quiescence before any weight update, keeping abort sweeps going meanwhile.

    A proxy-admitted request can still be queued in the router when the first abort sweep
    reaches an engine, so sweeps repeat while the gate is closed until the proxy acknowledges
    that every engine call has settled. A timeout never authorises training: the cut fails
    closed.
    """
    stop = asyncio.Event()

    async def repeat():
        while True:
            try:
                await asyncio.wait_for(stop.wait(), timeout=QUIESCE_ABORT_INTERVAL_S)
                return
            except TimeoutError:
                try:
                    await asyncio.wait_for(abort_engines(), timeout=5.0)
                except Exception as exc:  # noqa: BLE001 - keep sweeping; quiescence decides
                    logger.warning("[carry] abort sweep failed: %s", exc)

    repeater = asyncio.create_task(repeat()) if abort_engines is not None else None
    try:
        for attempt in range(QUIESCE_ATTEMPTS):
            try:
                result = await asyncio.to_thread(gate.quiesce)
            except StillActive as exc:
                if repeater is None or attempt + 1 == QUIESCE_ATTEMPTS:
                    raise
                logger.warning("[carry] %s; continuing abort sweeps (%d/%d)", exc, attempt + 1,
                               QUIESCE_ATTEMPTS)
                continue
            if not result.get("quiescent"):
                raise RuntimeError("the proxy has not acknowledged engine quiescence")
            break
    finally:
        stop.set()
        if repeater is not None:
            await repeater


# ---- group validity --------------------------------------------------------------------------------
@dataclass
class ValidityRules:
    n_samples_per_prompt: int
    max_weight_staleness: int | None = 4
    episode_timeout_s: float | None = None
    max_context_len: int = 0


def _status(sample: Any) -> str:
    status = getattr(sample, "status", "pending")
    return getattr(status, "value", status)


def _stale(sample: Any, rules: ValidityRules, version: int) -> bool:
    md = getattr(sample, "metadata", None) or {}
    admitted = md.get("admission_version")
    limit = rules.max_weight_staleness
    return admitted is not None and limit is not None and version - int(admitted) > limit


def _completed_error(sample: Any, rules: ValidityRules) -> str | None:
    md = getattr(sample, "metadata", None) or {}
    if md.get("admission_version") is None:
        return "missing_admission_version"
    tokens = getattr(sample, "tokens", None) or []
    length = int(getattr(sample, "response_length", 0) or 0)
    lps = getattr(sample, "rollout_log_probs", None) or []
    mask = getattr(sample, "loss_mask", None) or []
    if not (0 < length < len(tokens)) or (rules.max_context_len and len(tokens) > rules.max_context_len):
        return "invalid_token_lengths"
    if len(lps) != length or len(mask) != length or not any(mask):
        return "invalid_logprobs_or_mask"
    if any(m not in (0, 1) for m in mask) or any(not math.isfinite(lp) for lp in lps):
        return "invalid_logprobs_or_mask"
    reward = getattr(sample, "reward", None)
    if not isinstance(reward, (int, float)) or not math.isfinite(reward):
        return "invalid_reward"
    return None


def completed_sample_error(sample: Any, rules: ValidityRules, version: int) -> str | None:
    """Why a sample that came back from the buffer already completed cannot be trained on."""
    md = getattr(sample, "metadata", None) or {}
    if md.get("carry_error") or getattr(sample, "remove_sample", False):
        return "episode_failed"
    if _stale(sample, rules, version):
        return "stale"
    return _completed_error(sample, rules)


def group_error(group: Sequence[Any], rules: ValidityRules, *, version: int,
                for_training: bool = False,
                registry: EpisodeRegistry | None = None) -> str | None:
    """None if the group is whole and usable; otherwise the first reason it is not.

    Before training every member must be completed with consistent tokens, log-probabilities
    and mask; before a resume every unfinished member must still own a parked sandbox.
    """
    registry = registry or get_registry()
    if len(group) != rules.n_samples_per_prompt:
        return "incomplete_group"
    for sample in group:
        md = getattr(sample, "metadata", None) or {}
        if md.get("carry_error") or getattr(sample, "remove_sample", False):
            return "episode_failed"
        if _stale(sample, rules, version):
            return "stale"
        status = _status(sample)
        if status in ("completed", "truncated"):
            error = _completed_error(sample, rules)
            if error:
                return error
        elif for_training:
            return "unfinished_episode"
        else:
            entry = registry.get(episode_id(sample))
            if entry is None:
                if status != "pending" or md.get("admission_version") is not None:
                    return "missing_sandbox"
            else:
                if not entry.is_parked:
                    return "episode_already_active"
                limit = rules.max_weight_staleness
                if limit is not None and entry.staleness(version) > limit:
                    return "stale"
                if rules.episode_timeout_s is not None and entry.expired(rules.episode_timeout_s):
                    return "timeout"
    return None


def discard_group(group: Sequence[Any], reason: str = "group_voided", *,
                  registry: EpisodeRegistry | None = None) -> None:
    """Drop a group whole: evict its sandboxes and mark every member removed."""
    registry = registry or get_registry()
    why = {"stale": EvictReason.STALE, "timeout": EvictReason.TIMEOUT}.get(reason, EvictReason.GROUP_VOIDED)
    if group:
        logger.info("[carry] discard_group first_episode=%s reason=%s samples=%d",
                    episode_id(group[0]), reason, len(group))
    for sample in group:
        registry.evict(episode_id(sample), why)
        sample.remove_sample = True
        if getattr(sample, "metadata", None) is None:
            sample.metadata = {}
        sample.metadata["carry_error"] = reason


def preflight_group(group: Sequence[Any], rules: ValidityRules, *, version: int,
                    registry: EpisodeRegistry | None = None) -> bool:
    reason = group_error(group, rules, version=version, registry=registry)
    if reason:
        discard_group(group, reason, registry=registry)
        return False
    return True


def validate_training_batch(groups: Sequence[Sequence[Any]], rules: ValidityRules, *,
                            version: int, registry: EpisodeRegistry | None = None) -> dict[str, float]:
    """Recheck the final trainer input at the last boundary and report what it contains."""
    samples = []
    for group in groups:
        reason = group_error(group, rules, version=version, for_training=True, registry=registry)
        if reason:
            discard_group(group, reason, registry=registry)
            raise RuntimeError(f"training batch invalid at the final boundary: {reason}")
        samples.extend(group)
    ages = [version - int(s.metadata["admission_version"]) for s in samples]
    resumed = sum(int((s.metadata or {}).get("resumes", 0)) > 0 for s in samples)
    metrics = {"carry/weight_version": float(version),
               "carry/trainable_groups": float(len(groups)),
               "carry/trainable_samples": float(len(samples)),
               "carry/max_staleness": float(max(ages, default=0)),
               "carry/mean_staleness": (sum(ages) / len(ages)) if ages else 0.0,
               "carry/resumed_trainable_samples": float(resumed)}
    logger.info("[carry] trainable batch version=%d groups=%d samples=%d max_staleness=%d "
                "resumed_samples=%d", version, len(groups), len(samples), max(ages, default=0), resumed)
    return metrics


# ---- the carry buffer -------------------------------------------------------------------------------
def reconcile_buffer(groups: Sequence[Sequence[Any]], rules: ValidityRules, *, next_version: int,
                     carry_allowance: int, registry: EpisodeRegistry | None = None) -> list:
    """Keep whole resumable groups, then release every unreachable sandbox.

    Called after all generation tasks settled at the cut and before the trainer starts. Groups
    are judged at the version the *next* rollout will run under. Completed members cost no
    sandbox; unfinished members each hold one, and their total must fit the carry allowance.
    """
    registry = registry or get_registry()
    kept, reachable = [], set()
    for group in groups:
        reason = group_error(group, rules, version=next_version, registry=registry)
        if reason:
            discard_group(group, reason, registry=registry)
            continue
        kept.append(group)
        reachable.update(episode_id(s) for s in group if _status(s) not in ("completed", "truncated"))
    if len(reachable) > carry_allowance:
        raise RuntimeError(f"retained work ({len(reachable)} parked episodes) exceeds the carry "
                           f"allowance ({carry_allowance})")
    released = registry.retain(reachable)
    logger.info("[carry] buffer reconciled: groups=%d parked=%d unreachable_released=%d",
                len(kept), registry.parked_count(), released)
    return kept


def take_buffered_groups(data_source: Any, rules: ValidityRules, *, version: int,
                         registry: EpisodeRegistry | None = None) -> list:
    """Claim the whole buffer before any fresh group is requested.

    Every unfinished carried sample then belongs to the in-flight count, including the short
    interval before its coroutine resumes the registry; completed members return immediately
    and consume no new sandbox.
    """
    owner = getattr(data_source, "__self__", None)
    buffer = getattr(owner, "buffer", None)
    if not isinstance(buffer, list):
        raise TypeError("the carry needs Miles' buffered data source "
                           "(miles.rollout.data_source.RolloutDataSourceWithBuffer)")
    count = len(buffer)
    groups = data_source(count) if count else []
    if owner.buffer or len(groups) != count:
        raise RuntimeError("the carry must claim its entire buffer before fresh admission")
    return [g for g in groups if preflight_group(g, rules, version=version, registry=registry)]


# ---- capacity ----------------------------------------------------------------------------------------
@dataclass
class Capacity:
    """Three quantities the research launcher sized from first principles.

    ``max_live_sandboxes`` is the backend ceiling; ``max_concurrent_episodes`` the generation
    budget (episodes admitted at once); ``carry_allowance`` the sandboxes the carry may hold,
    which must cover every active episode because a resume reuses its slot. The invariant
    ``n_samples <= concurrent <= allowance <= ceiling`` is what keeps admission from wedging.
    """

    max_live_sandboxes: int
    max_concurrent_episodes: int
    carry_allowance: int

    def problems(self, n_samples_per_prompt: int) -> list[str]:
        out = []
        if self.max_concurrent_episodes < n_samples_per_prompt:
            out.append("max_concurrent_episodes must hold at least one group")
        if not self.max_concurrent_episodes <= self.carry_allowance <= self.max_live_sandboxes:
            out.append("need max_concurrent_episodes <= carry_allowance <= max_live_sandboxes")
        return out


# ---- heartbeat -----------------------------------------------------------------------------------------
@dataclass
class Heartbeat:
    registry: EpisodeRegistry
    interval_s: float
    episode_timeout_s: float
    on_sweep: Callable[[], None] | None = None     # e.g. refresh the process's liveness lease
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def start(self) -> None:
        if self.interval_s <= 0 or (self._thread is not None and self._thread.is_alive()):
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="carry-heartbeat", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                if self.on_sweep is not None:
                    self.on_sweep()
                self.registry.evict_timed_out(self.episode_timeout_s)
                self.registry.heartbeat_all()
            except Exception as exc:  # noqa: BLE001 - upkeep must never kill the thread
                logger.warning("[carry] heartbeat sweep failed: %s", exc)

    def stop(self) -> None:
        self._stop.set()
