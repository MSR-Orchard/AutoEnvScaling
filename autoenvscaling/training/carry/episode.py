"""Carry core for terminal episodes: run or resume one episode under the current weights.

Framework-agnostic on purpose: it imports nothing from Miles, so the invariants are testable
without a cluster. The Miles-facing rollout function is ``training.carry.rollout``.

The step cycle it participates in
---------------------------------
1. rollout phase: episodes run; the engine's own token ids and log-probabilities accumulate on
   the sample through the recording proxy (``training.carry.proxy``);
2. cut: the rollout function flips its abort flag; the in-flight generation is held at the
   proxy and the episode is *parked* (sandbox kept alive, turn state stored in the registry);
3. train phase: engines sleep, the optimizer steps, the weight version bumps;
4. wake: the parked sample comes back from Miles' buffer with its tokens; the episode resumes
   under the new policy and its later tokens are sampled by the new weights.

Correctness notes
-----------------
* **Behaviour log-probabilities are never recomputed.** They are the engine's per-token values,
  accumulated across park/resume into ``sample.rollout_log_probs``. That is what makes a
  mixed-policy trajectory sound under ``--use-rollout-logprobs``; recomputing them would corrupt
  the importance ratio. ``--mask-offpolicy-in-partial-rollout`` must therefore stay off: carried
  tokens are *trained on* with their stored log-probabilities rather than masked away.
* **KV state is not assumed to survive the flip.** Under colocation the engines free their memory
  for training, so a resumed turn re-prefills its context under the new weights. Resume depends
  only on ``sample.tokens`` and the live sandbox, never on engine-side state.
* **Staleness is stamped at admission** and carried on the registry entry, so queue wait does not
  age a group.

This is the research code's ``spare/miles/terminal_generate.py`` plus the sample repairs of its
``miles_adapter.py``, with the names made generic.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from autoenvscaling.training.carry.registry import EpisodeRegistry, EvictReason

logger = logging.getLogger(__name__)

_REGISTRY: EpisodeRegistry | None = None


def get_registry() -> EpisodeRegistry:
    """One registry per rollout-worker process; it outlives a training step by design."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = EpisodeRegistry()
    return _REGISTRY


def reset_registry(registry: EpisodeRegistry | None = None) -> EpisodeRegistry:
    """Test hook / explicit re-init. Drains any survivors first so a reset never strands sandboxes."""
    global _REGISTRY
    if _REGISTRY is not None:
        _REGISTRY.drain()
    _REGISTRY = registry if registry is not None else EpisodeRegistry()
    return _REGISTRY


class TurnStatus(str, Enum):
    DONE = "done"          # episode reached a terminal state; verify it
    ABORTED = "aborted"    # the cut landed mid-episode; park it
    FAILED = "failed"      # unrecoverable; drop the episode


class EpisodeActiveBudgetExpired(TimeoutError):
    """An episode reached its configured active execution limit (an expected outcome)."""


@dataclass
class TurnOutcome:
    status: TurnStatus
    turn_state: Any
    turn_index: int
    error: str | None = None


class EpisodeDriver(Protocol):
    """The terminal-harness side, injected so this module stays testable.

    The driver owns the agent loop and the engine calls; this module owns only the carry.
    ``run_turns`` must append the engine's own per-token log-probabilities to
    ``sample.rollout_log_probs`` and the generated ids to ``sample.tokens`` (with ``loss_mask``
    and ``response_length`` consistent), and must return promptly once ``should_abort()`` is true.
    """

    async def create_sandbox(self, sample: Any) -> Any: ...

    async def run_turns(self, *, sample: Any, sandbox: Any, turn_state: Any,
                        sampling_params: dict, should_abort: Callable[[], bool],
                        start_turn_index: int) -> TurnOutcome: ...

    async def verify(self, *, sample: Any, sandbox: Any, turn_state: Any) -> float: ...


@dataclass
class CarryConfig:
    max_weight_staleness: int | None = 4
    episode_timeout_s: float | None = None


@dataclass
class CarryCounters:
    """Per-rollout counters, reset each time ``log_rollout_data`` reports them."""

    parked: int = 0
    resumed: int = 0
    misses: int = 0
    neg_logprob_sum: float = 0.0
    neg_logprob_n: int = 0

    def note_entropy_sample(self, log_probs, loss_mask) -> None:
        """Monte-Carlo entropy estimate from one sample's log-probabilities.

        The tokens were sampled from the policy at temperature 1 and the log-probabilities are
        the engine's, so the mean negative log-probability of the sampled tokens is an unbiased
        entropy estimate. Only masked-in tokens count: injected turns (tool results) carry a
        placeholder 0.0 and would fake a collapse.
        """
        if not log_probs or not loss_mask or len(loss_mask) != len(log_probs):
            return
        pairs = [lp for lp, m in zip(log_probs, loss_mask) if m]
        if pairs:
            self.neg_logprob_sum += -float(sum(pairs))
            self.neg_logprob_n += len(pairs)

    def drain(self) -> dict[str, float]:
        out = {"carry/parked_this_rollout": float(self.parked),
               "carry/resumed_this_rollout": float(self.resumed),
               "carry/resume_misses_this_rollout": float(self.misses)}
        if self.neg_logprob_n:
            out["entropy/actor_mean_neg_logprob"] = self.neg_logprob_sum / self.neg_logprob_n
            out["entropy/actor_tokens_measured"] = float(self.neg_logprob_n)
        self.parked = self.resumed = self.misses = 0
        self.neg_logprob_sum, self.neg_logprob_n = 0.0, 0
        return out


COUNTERS = CarryCounters()


async def run_episode_with_carry(*, sample: Any, sampling_params: dict, driver: EpisodeDriver,
                                 registry: EpisodeRegistry, should_abort: Callable[[], bool],
                                 weight_version: int, rollout_id: int, config: CarryConfig) -> Any:
    """Run or resume one terminal episode. Returns the mutated sample.

    Sets ``sample._carry_status`` to "completed" or "aborted"; the rollout function maps that
    onto Miles' status enum.
    """
    eid = episode_id(sample)
    gid = group_id(sample)

    resumed = False
    entry = None
    if registry.get(eid) is not None or has_partial_response(sample):
        entry = registry.resume(eid, resumed_at_version=weight_version,
                                token_index=len(getattr(sample, "tokens", []) or []),
                                max_staleness=config.max_weight_staleness,
                                timeout_s=config.episode_timeout_s)
        resumed = entry is not None
        if resumed:
            COUNTERS.resumed += 1
        else:
            # A miss is not an error: after a worker restart every resume misses. Restarting
            # the episode costs work, never correctness. The partial tokens on the sample no
            # longer correspond to any live sandbox, so they are discarded, not continued.
            COUNTERS.misses += 1
            logger.warning("[carry] no registry entry for %s despite partial response; "
                           "restarting episode", eid)
            reset_partial_response(sample)

    if resumed:
        sandbox = entry.sandbox
        turn_state = entry.turn_state
        start_turn_index = entry.turn_index
        admission_version = entry.admission_version
    else:
        sandbox = await driver.create_sandbox(sample)
        turn_state = None
        start_turn_index = 0
        admission_version = weight_version   # stamped when generation actually starts

    logger.info("[carry] episode %s start resumed=%s turn=%d tokens=%d", eid, resumed,
                start_turn_index, len(getattr(sample, "tokens", []) or []))
    try:
        outcome = await driver.run_turns(sample=sample, sandbox=sandbox, turn_state=turn_state,
                                         sampling_params=sampling_params, should_abort=should_abort,
                                         start_turn_index=start_turn_index)
    except asyncio.CancelledError:
        # A forced abort-drain cancellation has no resumable outcome to park. Release
        # ownership and preserve the cancellation for gather().
        logger.info("[carry] episode %s cancelled; releasing sandbox", eid)
        registry.evict(eid, EvictReason.DRAIN) or _teardown_unregistered(registry, sandbox)
        raise
    except Exception as exc:
        if isinstance(exc, EpisodeActiveBudgetExpired):
            set_metadata(sample, "carry_failure_kind", "active_budget")
            logger.warning("[carry] episode %s budget exhausted", eid)
        else:
            logger.exception("[carry] episode %s failed", eid)
        registry.evict(eid, EvictReason.FAILED) or _teardown_unregistered(registry, sandbox)
        set_status(sample, "aborted")
        set_metadata(sample, "carry_error", f"{type(exc).__name__}: {exc}")
        return sample

    if outcome.status is TurnStatus.ABORTED:
        logger.info("[carry] episode %s parked at turn %d", eid, outcome.turn_index)
        COUNTERS.parked += 1
        registry.park(episode_id=eid, group_id=gid, sandbox=sandbox, turn_state=outcome.turn_state,
                      admission_version=admission_version,
                      start_rollout_id=start_rollout_id(sample, rollout_id),
                      parked_at_version=weight_version, turn_index=outcome.turn_index)
        set_metadata(sample, "episode_id", eid)
        set_metadata(sample, "start_rollout_id", start_rollout_id(sample, rollout_id))
        set_metadata(sample, "admission_version", admission_version)
        set_status(sample, "aborted")
        return sample

    if outcome.status is TurnStatus.FAILED:
        logger.warning("[carry] episode %s FAILED at turn %d: %s", eid, outcome.turn_index,
                       outcome.error or "episode failed")
        registry.evict(eid, EvictReason.FAILED) or _teardown_unregistered(registry, sandbox)
        set_status(sample, "aborted")
        set_metadata(sample, "carry_error", outcome.error or "episode failed")
        return sample

    # DONE: verify, then release the sandbox. The registry entry may not exist (an episode
    # that never parked), so tear down directly in that case.
    try:
        reward = await driver.verify(sample=sample, sandbox=sandbox, turn_state=outcome.turn_state)
        sample.reward = reward
    finally:
        if not registry.evict(eid, EvictReason.COMPLETED):
            _teardown_unregistered(registry, sandbox)
    logger.info("[carry] episode %s done at turn %d reward=%s response_len=%s", eid,
                outcome.turn_index, getattr(sample, "reward", None),
                getattr(sample, "response_length", None))
    set_status(sample, "completed")
    set_metadata(sample, "admission_version", admission_version)
    set_metadata(sample, "resumes", len(entry.resume_boundaries) if entry else 0)
    if entry:
        set_metadata(sample, "resume_boundaries",
                     [[b.token_index, b.turn_index, b.parked_at_version, b.resumed_at_version]
                      for b in entry.resume_boundaries])
    return sample


async def run_episode(*, sample: Any, sampling_params: dict, driver: EpisodeDriver,
                      registry: EpisodeRegistry, should_abort: Callable[[], bool],
                      weight_version: int, rollout_id: int, config: CarryConfig,
                      tokenizer: Any = None) -> Any:
    """``run_episode_with_carry`` with the containment and sample repairs the trainer needs.

    Any exception in any one episode is contained: the sample comes back aborted with the error
    in ``metadata["carry_error"]`` and the rollout continues (the research runs died whole from
    single-episode errors before this wrapper existed). Afterwards the per-token fields are made
    consistent, because a sample with mismatched ``rollout_log_probs``/``response_length`` or
    with no prompt token kills the whole optimizer step inside Megatron rather than one episode.
    """
    try:
        sample = await run_episode_with_carry(
            sample=sample, sampling_params=sampling_params, driver=driver, registry=registry,
            should_abort=should_abort, weight_version=weight_version, rollout_id=rollout_id,
            config=config)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - one episode must never kill the rollout
        logger.warning("episode failed (contained): %s: %s", type(exc).__name__, str(exc)[:300])
        set_status(sample, "aborted")
        set_metadata(sample, "carry_error", f"{type(exc).__name__}: {exc}")
        sample.remove_sample = True
    normalize_sample(sample, tokenizer=tokenizer)
    return sample


def normalize_sample(sample: Any, *, tokenizer: Any = None) -> None:
    """Make a sample safe for Miles' batch assembly without changing what it means.

    * a missing reward becomes 0.0 (the zero-std filter needs a number; an all-failed group has
      zero variance and is discarded, which is right);
    * a log-probability vector that does not match ``response_length`` voids the response (the
      sample stays in its group, inert, so a mixed group is not poisoned);
    * a sample must keep at least one prompt token, otherwise Miles' loss-mask padding fails.
    """
    if getattr(sample, "reward", None) is None:
        sample.reward = 0.0
    if getattr(sample, "rollout_log_probs", None) is None:
        sample.rollout_log_probs = []
    resp_len = int(getattr(sample, "response_length", 0) or 0)
    if len(sample.rollout_log_probs) != resp_len:
        logger.warning("episode %s: logprob/response_length mismatch (%d vs %s); voiding response",
                       getattr(sample, "index", "?"), len(sample.rollout_log_probs), resp_len)
        prompt_len = len(sample.tokens) - resp_len
        sample.tokens = sample.tokens[:prompt_len] if prompt_len > 0 else sample.tokens
        _void_response(sample)
        resp_len = 0
    if len(sample.tokens) - resp_len < 1:
        logger.warning("episode %s: no prompt tokens left (tokens=%d response_length=%d); "
                       "voiding response", getattr(sample, "index", "?"), len(sample.tokens), resp_len)
        _void_response(sample)
        if not sample.tokens:
            if tokenizer is not None and getattr(sample, "prompt", None):
                try:
                    sample.tokens = list(tokenizer.encode(str(sample.prompt), add_special_tokens=False))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("could not reseed prompt tokens: %s", exc)
            if not sample.tokens:
                # One in-vocabulary token keeps prompt_length == 1; the sample is aborted with an
                # empty loss mask and reward 0, so the token contributes nothing to the loss.
                sample.tokens = [0]
        return
    COUNTERS.note_entropy_sample(sample.rollout_log_probs, getattr(sample, "loss_mask", None))


def _void_response(sample: Any) -> None:
    sample.response = ""
    sample.response_length = 0
    sample.rollout_log_probs = []
    if getattr(sample, "loss_mask", None) is not None:
        sample.loss_mask = []
    set_status(sample, "aborted")
    sample.reward = 0.0


# ---- sample accessors (tolerant of plain objects so tests need no Miles types) -----------------
def episode_id(sample: Any) -> str:
    md = getattr(sample, "metadata", None) or {}
    if md.get("episode_id"):
        return str(md["episode_id"])
    # Stable across park/resume: Miles' sample index identifies the rollout within its group.
    return f"{group_id(sample)}-r{getattr(sample, 'index', 0)}"


def group_id(sample: Any) -> str:
    md = getattr(sample, "metadata", None) or {}
    if getattr(sample, "group_index", None) is not None:
        return f"group-{sample.group_index}"
    for key in ("group_id", "task_id", "prompt_id"):
        if md.get(key):
            return str(md[key])
    return hashlib.md5(str(getattr(sample, "prompt", "g0")).encode()).hexdigest()[:12]


def start_rollout_id(sample: Any, rollout_id: int) -> int:
    md = getattr(sample, "metadata", None) or {}
    return int(md.get("start_rollout_id", rollout_id))


def has_partial_response(sample: Any) -> bool:
    return int(getattr(sample, "response_length", 0) or 0) > 0 or bool(
        getattr(sample, "rollout_log_probs", None))


def reset_partial_response(sample: Any) -> None:
    """Drop orphaned partial tokens: they describe a sandbox that no longer exists.

    Miles' ``Sample`` has no separate prompt-token field, so the prompt is recovered as the
    tokens minus the generated tail.
    """
    prompt = list(getattr(sample, "prompt_tokens", []) or [])
    if not prompt:
        toks = list(getattr(sample, "tokens", []) or [])
        rl = int(getattr(sample, "response_length", 0) or 0)
        prompt = toks[: len(toks) - rl] if 0 < rl < len(toks) else toks
    sample.tokens = prompt
    sample.response_length = 0
    sample.response = ""
    sample.reward = None
    sample.remove_sample = False
    md = getattr(sample, "metadata", None) or {}
    for key in ("admission_version", "carry_error", "carry_failure_kind", "start_rollout_id",
                "resumes", "resume_boundaries"):
        md.pop(key, None)
    sample.rollout_log_probs = []
    if getattr(sample, "loss_mask", None) is not None:
        sample.loss_mask = []
    if getattr(sample, "weight_versions", None):
        sample.weight_versions = []


def set_status(sample: Any, status: str) -> None:
    sample._carry_status = status


def get_status(sample: Any) -> str:
    return getattr(sample, "_carry_status", "aborted")


def set_metadata(sample: Any, key: str, value: Any) -> None:
    md = getattr(sample, "metadata", None)
    if md is None:
        md = {}
        sample.metadata = md
    md[key] = value


def _teardown_unregistered(registry: EpisodeRegistry, sandbox: Any) -> None:
    """A sandbox that never made it into the registry still must not leak."""
    if sandbox is None:
        return
    try:
        registry._teardown(sandbox)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[carry] teardown of unregistered sandbox failed: %s", exc)
