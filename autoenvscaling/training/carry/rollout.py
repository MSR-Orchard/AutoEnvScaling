"""The Miles rollout function for `carry` mode: over-sampled groups, the constant-reward filter,
and partial rollout of terminal episodes across weight updates.

Wired in with ``--rollout-function-path autoenvscaling.training.carry.rollout.generate_rollout``
(see ``config.MilesConfig``, mode ``carry``). Miles calls it once per rollout with the engines
serving the current weights and a buffered data source; it returns exactly
``rollout_batch_size`` groups whose rewards vary, and hands every unfinished or surplus group
back to Miles' buffer for the next rollout.

Batch-completion semantics (the final run's, see docs/partial-rollout.md)
-----------------------------------------------------------------------
* Carried groups are claimed from the buffer first; their completed members are reused as they
  are, their unfinished members resume in their parked sandboxes under the new weights.
* Fresh groups are drawn ``over_sampling_batch_size`` at a time and admitted against an
  episode budget: ``deficit = max_concurrent_episodes - in_flight - held``. The budget bounds
  resources; over-sampling only sizes each refill. A group whose rewards do not vary is dropped
  and the pool refills; the loop continues until ``rollout_batch_size`` groups passed the filter,
  with no limit on draws (there are no "waves"), so a batch is never under-filled.
* Then the cut: the proxy's gate closes, the engines abort their requests, the proxy confirms
  quiescence, in-flight episodes park with their sandbox and turn state, and every partial or
  surplus-completed group goes to the buffer. Groups that would be stale at the next version,
  or whose sandbox is gone, are discarded whole.

``collect_batch`` is the loop without Miles (fakes for the data source, the group runner, the
filter and the engines), so it is tested offline; ``generate_rollout`` wires Miles around it.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from autoenvscaling.training.carry import episode as ep
from autoenvscaling.training.carry.lifecycle import (
    CLOCK,
    Capacity,
    GateControl,
    Heartbeat,
    HttpGate,
    ValidityRules,
    after_engine_abort,
    completed_sample_error,
    discard_group,
    group_error,
    reconcile_buffer,
    take_buffered_groups,
    validate_training_batch,
)
from autoenvscaling.training.carry.registry import EpisodeRegistry

logger = logging.getLogger(__name__)

ENV_PREFIX = "AES_CARRY_"


# ---- settings ---------------------------------------------------------------------------------------
@dataclass
class CarrySettings:
    """Runtime settings of carry mode, passed to the Ray workers as ``AES_CARRY_*`` variables.

    The defaults are the final run's values (its launcher sized them against a 1024-sandbox
    ceiling: 768 episodes generating, carry allowed to hold every active episode).
    """

    max_live_sandboxes: int = 1024          # backend ceiling
    max_concurrent_episodes: int = 768      # generation budget (episodes admitted at once)
    carry_allowance: int = 768              # sandboxes the carry may hold across the cut
    episode_timeout_s: float = 2700.0       # active (unpaused) time budget of one episode
    max_paused_s: float = 7200.0            # wall backstop for an episode parked and never resumed
    setup_timeout_s: float = 900.0
    verify_timeout_s: float = 900.0
    max_turns: int = 128
    abort_drain_timeout_s: float = 300.0    # bound on waiting for in-flight tasks at the cut
    abort_request_timeout_s: float = 60.0   # bound on the engines acknowledging abort_request
    heartbeat_s: float = 300.0              # parked-sandbox keepalive sweep
    proxy_url: str = "http://127.0.0.1:30000"        # the recording proxy, from the trainer
    proxy_public_url: str = ""                       # the same proxy, from inside sandboxes
    proxy_api_key_env: str = "AES_PROXY_API_KEY"
    served_model: str = "policy"
    # Sandboxes follow the repository's own solver settings unless overridden here:
    # `Settings.sandbox.backend` (Modal by default), `solver_cpu`, `solver_memory_gib`, and the
    # solver network mode (`AES_SOLVER_NETWORK`, allowlist by default).
    sandbox_backend: str = ""                        # "" -> Settings.sandbox.backend
    sandbox_cpu: int = 0                             # 0 -> Settings.sandbox.solver_cpu
    sandbox_memory_gib: int = 0                      # 0 -> Settings.sandbox.solver_memory_gib
    sandbox_workdir: str = "/app"                    # the Terminal-Bench task convention
    network: str = ""                                # "" -> AES_SOLVER_NETWORK ("allowlist" | "public")
    run_id: str = "aes-carry"                        # names sandboxes so orphans can be reaped
    pi_version: str = ""                             # default: calibration.solver.PI_VERSION
    pi_package: str = ""                             # default: the scope Harbor uses for that version
    state_dir: str = ""                              # where per-rollout carry manifests go

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> CarrySettings:
        env = os.environ if env is None else env
        values: dict[str, Any] = {}
        for f in fields(cls):
            raw = env.get(ENV_PREFIX + f.name.upper())
            if raw is None:
                continue
            values[f.name] = _coerce(f.default, raw)
        settings = cls(**values)
        from autoenvscaling.config import Settings
        repo = Settings.load(env=env).sandbox
        settings.sandbox_backend = settings.sandbox_backend or repo.backend
        settings.sandbox_cpu = settings.sandbox_cpu or repo.solver_cpu
        settings.sandbox_memory_gib = settings.sandbox_memory_gib or repo.solver_memory_gib
        settings.network = settings.network or env.get("AES_SOLVER_NETWORK", "allowlist").strip().lower()
        return settings

    def to_env(self) -> dict[str, str]:
        return {ENV_PREFIX + f.name.upper(): str(getattr(self, f.name)) for f in fields(self)}

    @property
    def capacity(self) -> Capacity:
        return Capacity(self.max_live_sandboxes, self.max_concurrent_episodes, self.carry_allowance)

    def problems(self, n_samples_per_prompt: int) -> list[str]:
        out = self.capacity.problems(n_samples_per_prompt)
        if self.episode_timeout_s <= 0:
            out.append("episode_timeout_s must be positive")
        if not self.proxy_public_url:
            out.append("proxy_public_url is required (sandboxes cannot reach the trainer's loopback)")
        if self.network not in ("", "allowlist", "public"):
            out.append("network must be 'allowlist' or 'public'")
        if self.sandbox_backend not in ("", "modal", "docker"):
            out.append("sandbox_backend must be 'modal' or 'docker'")
        return out


def _coerce(default: Any, raw: str) -> Any:
    if isinstance(default, bool):
        return raw.lower() in {"1", "true", "yes"}
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


# ---- the loop -----------------------------------------------------------------------------------------
@dataclass
class LoopConfig:
    rollout_batch_size: int
    n_samples_per_prompt: int
    over_sampling_batch_size: int
    max_concurrent_episodes: int
    partial_rollout: bool = True
    abort_drain_timeout_s: float = 300.0
    abort_request_timeout_s: float = 60.0
    wait_timeout_s: float = 1.0          # how often admission is re-evaluated while waiting
    admit_log_interval_s: float = 60.0


class RolloutState:
    """What one rollout's loop shares with its episodes: the abort flag and the counters."""

    def __init__(self) -> None:
        self.aborted = False
        self.pendings: set[asyncio.Task] = set()
        self.remaining_batch_size = 0
        self.inflight_samples = 0
        self._last_admit_log = 0.0

    def submit(self, groups: Sequence[Sequence[Any]], run_group) -> None:
        # Count episodes at submit time: the admission loop is synchronous and would otherwise
        # keep submitting before any task had a chance to run.
        self.inflight_samples += sum(len(g) for g in groups)
        for group in groups:
            self.pendings.add(asyncio.create_task(run_group(group)))
        self.remaining_batch_size += len(groups)


@dataclass
class BatchResult:
    data: list                     # exactly rollout_batch_size groups, sorted by first index
    aborted: list                  # partial and surplus groups for the buffer
    metrics: dict[str, float] = field(default_factory=dict)
    groups_drawn: int = 0
    groups_dropped: dict[str, int] = field(default_factory=dict)


class CarryHooks:
    """Lifecycle callbacks ``collect_batch`` needs; ``MilesHooks`` is the real one."""

    def __init__(self, rules: ValidityRules, *, registry: EpisodeRegistry | None = None,
                 gate: GateControl | None = None,
                 abort_engines: Callable[[], Awaitable[None]] | None = None,
                 held: Callable[[], int] | None = None) -> None:
        self.rules = rules
        self.registry = registry or ep.get_registry()
        self.gate = gate
        self.abort_engines = abort_engines
        self.held = held or (lambda: 0)

    def version(self) -> int:
        return CLOCK.current_version()

    def take_buffered_groups(self, data_source) -> list:
        return take_buffered_groups(data_source, self.rules, version=self.version(),
                                    registry=self.registry)

    def group_error(self, group, *, for_training: bool) -> str | None:
        return group_error(group, self.rules, version=self.version(), for_training=for_training,
                           registry=self.registry)

    def discard_group(self, group, reason: str) -> None:
        discard_group(group, reason, registry=self.registry)

    def validate_training_batch(self, groups) -> dict[str, float]:
        return validate_training_batch(groups, self.rules, version=self.version(),
                                       registry=self.registry)

    async def before_cut(self) -> None:
        if self.gate is not None:
            await asyncio.to_thread(self.gate.park)

    async def engines_abort(self) -> None:
        if self.abort_engines is not None:
            await self.abort_engines()

    async def after_abort(self) -> None:
        if self.gate is not None:
            await after_engine_abort(self.gate, abort_engines=self.abort_engines)


async def collect_batch(cfg: LoopConfig, *, data_source: Callable[[int], list],
                        run_group: Callable[[Sequence[Any]], Awaitable[Sequence[Any]]],
                        dynamic_filter: Callable[[Sequence[Any]], tuple[bool, str | None]],
                        hooks: CarryHooks, state: RolloutState, rollout_id: int) -> BatchResult:
    """Fill one batch: ``rollout_batch_size`` groups that pass the filter, then cut.

    ``data_source(n)`` returns up to n groups (buffer first, then fresh prompts);
    ``run_group(group)`` runs or resumes every member and returns the group; ``dynamic_filter``
    is Miles' ``check_reward_nonzero_std`` or a stand-in.
    """
    target = cfg.rollout_batch_size
    data: list = []
    extra_completed: list = []
    drawn = 0
    dropped: dict[str, int] = {}

    carried = hooks.take_buffered_groups(data_source)
    if carried:
        state.submit(carried, run_group)
        logger.info("[carry] carried groups=%d samples=%d", len(carried),
                    sum(len(g) for g in carried))

    while len(data) < target:
        # Keep the generation pool full, measured in episodes, within the resource budget.
        while True:
            need_batch = state.remaining_batch_size < target
            held = hooks.held()
            deficit = cfg.max_concurrent_episodes - state.inflight_samples - held
            now = time.monotonic()
            if now - state._last_admit_log >= cfg.admit_log_interval_s:
                state._last_admit_log = now
                logger.info("[admit] target=%d inflight=%d held=%d deficit=%d pendings=%d "
                            "remaining_batch=%d have=%d/%d", cfg.max_concurrent_episodes,
                            state.inflight_samples, held, deficit, len(state.pendings),
                            state.remaining_batch_size, len(data), target)
            need_pool = cfg.partial_rollout and deficit >= cfg.n_samples_per_prompt
            if not (need_batch or need_pool):
                break
            budget_groups = max(0, deficit // cfg.n_samples_per_prompt)
            want = cfg.over_sampling_batch_size if need_batch else max(1, budget_groups)
            n = min(want, budget_groups)
            if n <= 0:
                if not state.pendings:
                    # Nothing running and no budget: only cleanup can free a slot. Wait rather
                    # than admit work the backend cannot hold.
                    await asyncio.sleep(0.25)
                    continue
                break
            groups = data_source(n)
            if not groups:
                break
            drawn += len(groups)
            state.submit(groups, run_group)

        if not state.pendings:
            raise RuntimeError("the rollout data source is exhausted before a trainable batch "
                               "was collected")
        done, state.pendings = await asyncio.wait(state.pendings, timeout=cfg.wait_timeout_s,
                                                  return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            group = list(task.result())
            assert len(group) == cfg.n_samples_per_prompt
            reason = hooks.group_error(group, for_training=True)
            if reason is None:
                keep, why = dynamic_filter(group)
                reason = None if keep else (why or "filtered")
                if reason is not None:
                    # A filtered group is finished: its sandboxes are released, its samples
                    # are not trained on. (Miles' own loop never puts them back either.)
                    hooks.discard_group(group, reason)
            else:
                hooks.discard_group(group, reason)
            if reason is not None:
                dropped[reason] = dropped.get(reason, 0) + 1
                state.remaining_batch_size -= 1
                continue
            if len(data) < target:
                data.append(group)
            else:
                extra_completed.append(group)

    aborted = await cut(cfg, state=state, hooks=hooks, rollout_id=rollout_id)
    aborted.extend(extra_completed)
    assert len(data) == target, f"got {len(data)} groups, expected {target}"
    data.sort(key=lambda g: g[0].index)
    metrics = {"carry/groups_drawn": float(drawn + len(carried)),
               "carry/groups_dropped": float(sum(dropped.values())),
               "carry/groups_carried_in": float(len(carried)),
               "carry/groups_carried_out": float(len(aborted)),
               "carry/surplus_completed_groups": float(len(extra_completed))}
    for reason, n in dropped.items():
        metrics[f"carry/dropped_{reason}"] = float(n)
    metrics.update(hooks.validate_training_batch(data))
    return BatchResult(data=data, aborted=aborted, metrics=metrics, groups_drawn=drawn,
                       groups_dropped=dropped)


async def cut(cfg: LoopConfig, *, state: RolloutState, hooks: CarryHooks,
              rollout_id: int) -> list:
    """Park everything in flight: close the gate, abort the engines, drain the tasks."""
    assert not state.aborted
    state.aborted = True
    await hooks.before_cut()
    try:
        # The first abort_request must be bounded too: a router that never answers would
        # otherwise hold the cut forever. Quiescence is still confirmed (or refused) below.
        await asyncio.wait_for(hooks.engines_abort(), timeout=cfg.abort_request_timeout_s)
    except TimeoutError:
        logger.warning("the engines did not acknowledge abort_request within %.0fs; continuing",
                       cfg.abort_request_timeout_s)
    await hooks.after_abort()

    aborted: list = []
    loop = asyncio.get_running_loop()
    deadline = loop.time() + cfg.abort_drain_timeout_s
    while state.pendings:
        remaining = deadline - loop.time()
        if remaining <= 0:
            # A wedged task must bound GPU idle at minutes, not forever. Cancelling a straggler
            # is safe: a resumed episode that finds no registry entry restarts from turn 0.
            logger.warning("abort drain timed out after %.0fs with %d task(s) pending; "
                           "cancelling them", cfg.abort_drain_timeout_s, len(state.pendings))
            for task in state.pendings:
                task.cancel()
            await asyncio.gather(*state.pendings, return_exceptions=True)
            state.pendings = set()
            break
        done, state.pendings = await asyncio.wait(state.pendings, timeout=remaining,
                                                  return_when=asyncio.FIRST_COMPLETED)
        if not cfg.partial_rollout:
            continue
        for task in done:
            try:
                group = list(task.result())
            except asyncio.CancelledError:
                continue
            except Exception:
                logger.exception("failed group discarded during the abort drain")
                continue
            for sample in group:
                if getattr(sample, "response", None) and "start_rollout_id" not in sample.metadata:
                    sample.metadata["start_rollout_id"] = rollout_id
            aborted.append(group)
    if cfg.partial_rollout:
        logger.info("collected %d partial groups for the buffer", len(aborted))
    return aborted


# ---- restart manifest --------------------------------------------------------------------------------
def write_manifest(state_dir: Path, rollout_id: int, *, buffered_groups: int,
                   parked_episodes: Sequence[str]) -> Path:
    """What the carry held after this rollout. Read back after a restart to say what was lost."""
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / f"carry_manifest_{rollout_id:06d}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"rollout_id": rollout_id, "buffered_groups": buffered_groups,
                               "parked_episodes": list(parked_episodes), "time": time.time()}))
    tmp.replace(path)
    return path


def read_previous_manifest(state_dir: Path, rollout_id: int) -> dict[str, Any] | None:
    path = state_dir / f"carry_manifest_{rollout_id - 1:06d}.json"
    return json.loads(path.read_text()) if path.exists() else None


# ---- Miles wiring -------------------------------------------------------------------------------------
_RUNTIME: dict[str, Any] = {}


def _runtime(args: Any) -> dict[str, Any]:
    """Process-wide objects: settings, driver, gate, heartbeat. Built on the first rollout."""
    if _RUNTIME:
        return _RUNTIME
    from autoenvscaling.training.carry.driver import SandboxEpisodeDriver

    settings = CarrySettings.from_env()
    problems = settings.problems(args.n_samples_per_prompt)
    problems += config_problems(args)
    if problems:
        raise RuntimeError("carry mode misconfigured: " + "; ".join(problems))
    rules = ValidityRules(n_samples_per_prompt=args.n_samples_per_prompt,
                          max_weight_staleness=args.max_weight_staleness,
                          episode_timeout_s=settings.episode_timeout_s,
                          max_context_len=int(getattr(args, "rollout_max_context_len", 0) or 0))
    driver = SandboxEpisodeDriver(settings, args)
    gate = HttpGate(settings.proxy_url, api_key=os.environ.get(settings.proxy_api_key_env, ""))
    _RUNTIME.update(settings=settings, rules=rules, driver=driver, gate=gate,
                    registry=ep.get_registry(),
                    heartbeat=Heartbeat(ep.get_registry(), settings.heartbeat_s,
                                        settings.episode_timeout_s),
                    first_rollout=True)
    atexit.register(_shutdown_sync)
    return _RUNTIME


def config_problems(args: Any) -> list[str]:
    """Miles arguments the carry cannot run under (the final run's invariants, made explicit)."""
    out = []
    if not getattr(args, "partial_rollout", False):
        out.append("--partial-rollout is required")
    if not getattr(args, "use_rollout_logprobs", False):
        out.append("--use-rollout-logprobs is required: carried tokens train on the engine's "
                   "recorded log-probabilities")
    if getattr(args, "mask_offpolicy_in_partial_rollout", False):
        out.append("--mask-offpolicy-in-partial-rollout must be off: carried tokens are trained "
                   "on, not masked")
    if getattr(args, "recompute_logprobs_via_prefill", False):
        out.append("--recompute-logprobs-via-prefill must be off: behaviour log-probabilities "
                   "are never recomputed")
    stale = getattr(args, "max_weight_staleness", None)
    if stale is not None and stale < 0:
        out.append("--max-weight-staleness must be non-negative")
    if getattr(args, "over_sampling_batch_size", 0) < getattr(args, "rollout_batch_size", 0):
        out.append("--over-sampling-batch-size must be >= --rollout-batch-size")
    return out


def _shutdown_sync() -> None:
    rt = _RUNTIME
    if not rt:
        return
    try:
        rt["heartbeat"].stop()
        try:
            rt["gate"].park()
        except Exception as exc:  # noqa: BLE001
            logger.warning("[carry] could not park the gate at shutdown: %s", exc)
        rt["registry"].drain()
    except Exception as exc:  # noqa: BLE001
        logger.warning("[carry] shutdown incomplete: %s", exc)


async def _abort_engines(args: Any) -> None:
    """Abort every request on every engine behind the router (Miles' own abort, lifted)."""
    from miles.utils.http_utils import get, post

    router = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    try:
        import sglang_router
        from packaging.version import parse
        legacy = parse(sglang_router.__version__) <= parse("0.2.1")
    except Exception:  # noqa: BLE001
        legacy = False
    if legacy or getattr(args, "use_miles_router", False):
        urls = (await get(f"{router}/list_workers"))["urls"]
    else:
        urls = [w["url"] for w in (await get(f"{router}/workers"))["workers"]]
    results = await asyncio.gather(*(post(f"{u}/abort_request", {"abort_all": True}) for u in urls),
                                   return_exceptions=True)
    for url, result in zip(urls, results):
        if isinstance(result, Exception):
            logger.warning("failed to abort worker at %s: %s", url, result)


def _status_enum():
    from miles.utils.types import Sample
    return Sample.Status


async def _run_sample(args: Any, sample: Any, sampling_params: dict, *, rt: dict[str, Any],
                      state: RolloutState, rollout_id: int) -> Any:
    Status = _status_enum()
    try:
        if sample.status in (Status.COMPLETED, Status.TRUNCATED):
            reason = completed_sample_error(sample, rt["rules"], CLOCK.current_version())
            if reason:
                discard_group([sample], reason, registry=rt["registry"])
                sample.status = Status.ABORTED
            return sample
        sample = await ep.run_episode(
            sample=sample, sampling_params=sampling_params, driver=rt["driver"],
            registry=rt["registry"], should_abort=lambda: state.aborted,
            weight_version=CLOCK.current_version(), rollout_id=rollout_id,
            config=ep.CarryConfig(max_weight_staleness=args.max_weight_staleness,
                                  episode_timeout_s=rt["settings"].episode_timeout_s),
            tokenizer=getattr(rt["driver"], "tokenizer", None))
        sample.status = {"completed": Status.COMPLETED}.get(ep.get_status(sample), Status.ABORTED)
        return sample
    finally:
        state.inflight_samples -= 1


def _make_run_group(args: Any, sampling_params: dict, *, rt: dict[str, Any], state: RolloutState,
                    rollout_id: int):
    async def run_group(group):
        if state.aborted:
            state.inflight_samples -= len(group)
            return list(group)
        tasks = [asyncio.create_task(_run_sample(args, s, dict(sampling_params), rt=rt, state=state,
                                                 rollout_id=rollout_id)) for s in group]
        try:
            return list(await asyncio.gather(*tasks))
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            discard_group(list(group), "group_cancelled", registry=rt["registry"])
            raise
    return run_group


def generate_rollout(args: Any, rollout_id: int, data_source: Any, evaluation: bool = False):
    """Miles rollout-function entry point (legacy signature)."""
    if evaluation:
        raise NotImplementedError("carry mode does not evaluate inside Miles; evaluate "
                                  "checkpoints separately")
    from miles.rollout.base_types import RolloutFnTrainOutput
    from miles.rollout.filter_hub.base_types import call_dynamic_filter
    from miles.utils.async_utils import run
    from miles.utils.misc import load_function

    rt = _runtime(args)
    settings: CarrySettings = rt["settings"]
    filter_fn = load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path else None

    def dynamic_filter(group):
        if filter_fn is None:
            return True, None
        out = call_dynamic_filter(filter_fn, args, group)
        return bool(out.keep), out.reason

    sampling_params = {"temperature": args.rollout_temperature, "top_p": args.rollout_top_p,
                       "top_k": args.rollout_top_k, "max_new_tokens": args.rollout_max_response_len}
    cfg = LoopConfig(rollout_batch_size=args.rollout_batch_size,
                     n_samples_per_prompt=args.n_samples_per_prompt,
                     over_sampling_batch_size=args.over_sampling_batch_size,
                     max_concurrent_episodes=settings.max_concurrent_episodes,
                     partial_rollout=bool(args.partial_rollout),
                     abort_drain_timeout_s=settings.abort_drain_timeout_s,
                     abort_request_timeout_s=settings.abort_request_timeout_s)
    output, aborted = run(_generate(args, rollout_id, data_source, cfg, rt, dynamic_filter,
                                    sampling_params))
    data_source.add_samples(aborted)
    data_source.buffer = reconcile_buffer(
        data_source.buffer, rt["rules"], next_version=CLOCK.current_version() + 1,
        carry_allowance=settings.carry_allowance, registry=rt["registry"])
    if settings.state_dir:
        write_manifest(Path(settings.state_dir), rollout_id,
                       buffered_groups=len(data_source.buffer),
                       parked_episodes=rt["registry"].episode_ids())
    if rollout_id == args.num_rollout - 1:
        run(_shutdown(rt))
    return RolloutFnTrainOutput(samples=output.data, metrics=output.metrics)


async def _generate(args, rollout_id, data_source, cfg, rt, dynamic_filter, sampling_params):
    state = RolloutState()
    hooks = CarryHooks(rt["rules"], registry=rt["registry"], gate=rt["gate"],
                       abort_engines=lambda: _abort_engines(args),
                       held=rt["driver"].pending_cleanup_count)
    await _begin_rollout(args, rollout_id, rt)
    try:
        run_group = _make_run_group(args, sampling_params, rt=rt, state=state, rollout_id=rollout_id)
        result = await collect_batch(cfg, data_source=data_source.get_samples, run_group=run_group,
                                     dynamic_filter=dynamic_filter, hooks=hooks, state=state,
                                     rollout_id=rollout_id)
    except BaseException:
        # Fail closed: park the gate, cancel what is running, release every sandbox.
        state.aborted = True
        try:
            await hooks.before_cut()
        finally:
            for task in state.pendings:
                task.cancel()
            await asyncio.gather(*state.pendings, return_exceptions=True)
            await _shutdown(rt)
        raise
    result.metrics.update(rt["registry"].metrics())
    result.metrics.update(ep.COUNTERS.drain())
    result.metrics["carry/pending_cleanup"] = float(rt["driver"].pending_cleanup_count())
    return result, result.aborted


async def _begin_rollout(args: Any, rollout_id: int, rt: dict[str, Any]) -> None:
    settings: CarrySettings = rt["settings"]
    await asyncio.to_thread(rt["gate"].check)
    version = CLOCK.begin(rollout_id)
    if rt.pop("first_rollout", False):
        # Restart handling, explicit: the carry buffer and registry live in this process, so
        # after a restart nothing is carried. Say what the previous process held, and reap the
        # sandboxes it left behind (they are not resumable without their registry entries).
        if settings.state_dir:
            previous = read_previous_manifest(Path(settings.state_dir), rollout_id)
            if previous:
                logger.warning("[carry] restart at rollout %d: the previous process held %d "
                               "buffered groups and %d parked episodes; they restart from "
                               "turn 0 (no duplicates: the data source resumes at its "
                               "checkpointed offset)", rollout_id, previous["buffered_groups"],
                               len(previous["parked_episodes"]))
        reaped = await rt["driver"].reap_orphans()
        if reaped:
            logger.warning("[carry] reaped %d orphaned sandboxes from a previous process", reaped)
        rt["heartbeat"].start()
    # Miles calls the rollout only once the weights and KV memory are ready; per-episode gates
    # stay closed until those episodes are actually admitted.
    await asyncio.to_thread(rt["gate"].unpark)
    logger.info("[carry] weights ready, version=%d (rollout %d)", version, rollout_id)


async def _shutdown(rt: dict[str, Any]) -> None:
    rt["heartbeat"].stop()
    try:
        await asyncio.to_thread(rt["gate"].park)
    finally:
        rt["registry"].drain()
    await rt["driver"].wait_cleanup(30.0)


# ---- the log hook ----------------------------------------------------------------------------------
def log_rollout_data(rollout_id: int, args: Any, samples: Any, rollout_extra_metrics: Any,
                     rollout_time: float) -> bool:
    """``--custom-rollout-log-function-path`` hook: add the carry's view of the rollout.

    Returns False so Miles still performs its own logging.
    """
    try:
        metrics = dict(ep.get_registry().metrics())
        metrics["carry/rollout_seconds"] = float(rollout_time)
        if isinstance(rollout_extra_metrics, dict):
            rollout_extra_metrics.update(metrics)
        else:
            logger.info("[carry] metrics %s", metrics)
    except Exception as exc:  # noqa: BLE001 - metrics must never break a rollout
        logger.warning("[carry] metric collection failed: %s", exc)
    return False
