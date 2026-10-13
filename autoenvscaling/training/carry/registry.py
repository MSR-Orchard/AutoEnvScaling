"""Registry of terminal episodes that survive across Miles rollout iterations.

Why this exists
---------------
Miles' ``--partial-rollout`` recycles an unfinished *text* sample: the prompt plus the tokens
generated so far go back into the data buffer, and the next rollout iteration continues
generating. That works because a text continuation's entire state *is* the token sequence.

An agentic terminal episode's state is not. It includes a live sandbox, a running container with
a filesystem the agent has been mutating for possibly dozens of turns. Miles' buffer cannot hold
that, which is why its agentic generate functions assert partial rollout off
(``generate_hub/multi_turn.py``). This registry is the missing half: the buffer keeps the tokens,
the registry keeps everything else, and the two are rejoined by ``episode_id`` when the sample
comes back around.

Lifetime
--------
Miles' ``RolloutManager`` is a Ray actor constructed once and driven with ``generate(rollout_id)``
once per training step; the rollout function runs in a thread of that same process. Colocation
offloads GPU memory between phases but never disposes the actor, so module-level state here
outlives a training step, which is precisely what the carry needs. If that ever stops being true,
the carry breaks loudly rather than silently: ``resume()`` simply misses and the episode restarts.

Leak discipline
---------------
A parked episode holds a sandbox. Every entry therefore has exactly one eviction path
(completion, staleness drop, group void, timeout, or drain) and ``evict()`` tears the sandbox down
exactly once, recording why. ``assert_no_leaks()`` is the test hook.

This module is the research code's ``spare/miles/episode_registry.py`` with the sandbox-service
specifics removed; the behaviour is the one the paper's final run trained with.
"""

from __future__ import annotations

import concurrent.futures as cf
import logging
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class EvictReason(str, Enum):
    """Why an episode left the registry. Every entry ends on exactly one."""

    COMPLETED = "completed"          # episode finished; reward collected
    STALE = "stale"                  # aged past --max-weight-staleness
    GROUP_VOIDED = "group_voided"    # the group it belongs to was discarded
    TIMEOUT = "timeout"              # episode exceeded its wall-clock budget
    DRAIN = "drain"                  # run shutting down
    FAILED = "failed"                # episode errored unrecoverably


@dataclass(frozen=True)
class ResumeBoundary:
    """Where a parked episode was cut and which policy resumed it.

    Every boundary is stamped so a carried trajectory can be attributed per token: the tokens
    before ``token_index`` were sampled under ``parked_at_version`` (or earlier boundaries), the
    tokens after it under ``resumed_at_version``.
    """

    token_index: int      # len(tokens) at the moment of the cut
    turn_index: int       # which turn was interrupted
    parked_at_version: int
    resumed_at_version: int


@dataclass
class EpisodeEntry:
    episode_id: str
    group_id: str
    sandbox: Any                      # the live sandbox; None once torn down
    turn_state: Any                   # harness-owned conversation state
    admission_version: int            # policy version when generation was ADMITTED
    created_at: float
    start_rollout_id: int
    resume_boundaries: list[ResumeBoundary] = field(default_factory=list)
    parked_at_version: int | None = None
    turn_index: int = 0
    is_parked: bool = True

    def staleness(self, current_version: int) -> int:
        """Optimizer updates elapsed since this episode was admitted.

        Stamped at *generation admission*, not at enqueue, so queue wait does not age a group.
        """
        return current_version - self.admission_version

    def expired(self, timeout_s: float) -> bool:
        check = getattr(self.sandbox, "episode_expired", None)
        if check is not None:
            return check(timeout_s)
        return time.monotonic() - self.created_at >= timeout_s


class EpisodeRegistry:
    """Process-level store of parked episodes, keyed by episode id.

    Thread-safe: the rollout function runs under ``asyncio.to_thread`` and eviction can be driven
    from a heartbeat thread concurrently with generation.
    """

    def __init__(self, teardown: Callable[[Any], None] | None = None) -> None:
        self._entries: dict[str, EpisodeEntry] = {}
        self._lock = threading.RLock()
        self._teardown = teardown if teardown is not None else _default_teardown
        self.evictions: dict[EvictReason, int] = {r: 0 for r in EvictReason}
        self._torn_down: set[str] = set()
        self._double_teardowns: list[str] = []

    # ---- park ------------------------------------------------------------------------------
    def park(self, *, episode_id: str, group_id: str, sandbox: Any, turn_state: Any,
             admission_version: int, start_rollout_id: int, parked_at_version: int,
             turn_index: int = 0) -> EpisodeEntry:
        """Store an unfinished episode so a later iteration can resume it."""
        with self._lock:
            existing = self._entries.get(episode_id)
            if existing is not None and existing.sandbox is not sandbox:
                # Reusing an id must not discard ownership of the old instance.
                self.evict(episode_id, EvictReason.FAILED)
                existing = None
            if existing is not None:
                # Re-parking after a resume: keep the original admission version (staleness is
                # measured from admission, not from the most recent park).
                existing.sandbox = sandbox
                existing.turn_state = turn_state
                existing.parked_at_version = parked_at_version
                existing.turn_index = turn_index
                existing.is_parked = True
                if pause := getattr(sandbox, "pause_execution", None):
                    pause()
                return existing
            entry = EpisodeEntry(
                episode_id=episode_id, group_id=group_id, sandbox=sandbox, turn_state=turn_state,
                admission_version=admission_version,
                created_at=getattr(sandbox, "created_at", time.monotonic()),
                start_rollout_id=start_rollout_id, parked_at_version=parked_at_version,
                turn_index=turn_index)
            # A fresh park with a new sandbox: any teardown recorded against this id belongs to
            # a previous episode that merely shared it (Miles reuses sample indices after a
            # checkpoint resume). Scope the double-free guard to one episode instance.
            self._torn_down.discard(episode_id)
            if pause := getattr(sandbox, "pause_execution", None):
                pause()
            self._entries[episode_id] = entry
            return entry

    # ---- resume ----------------------------------------------------------------------------
    def resume(self, episode_id: str, *, resumed_at_version: int, token_index: int,
               max_staleness: int | None = None,
               timeout_s: float | None = None) -> EpisodeEntry | None:
        """Hand back a parked episode and record the resume boundary.

        Returns None on a miss, which the caller must treat as "start a fresh episode" rather
        than an error: a miss is what happens after a process restart, and restarting the
        episode is always safe.
        """
        with self._lock:
            entry = self._entries.get(episode_id)
            if entry is None:
                return None
            if max_staleness is not None and entry.staleness(resumed_at_version) > max_staleness:
                self.evict(episode_id, EvictReason.STALE)
                return None
            if timeout_s is not None and entry.expired(timeout_s):
                self.evict(episode_id, EvictReason.TIMEOUT)
                return None
            if not entry.is_parked:
                raise RuntimeError(f"episode {episode_id} is already active")
            if resume := getattr(entry.sandbox, "resume_execution", None):
                resume()
            entry.is_parked = False
            entry.resume_boundaries.append(ResumeBoundary(
                token_index=token_index, turn_index=entry.turn_index,
                parked_at_version=(entry.parked_at_version if entry.parked_at_version is not None
                                   else entry.admission_version),
                resumed_at_version=resumed_at_version))
            return entry

    def get(self, episode_id: str) -> EpisodeEntry | None:
        with self._lock:
            return self._entries.get(episode_id)

    # ---- evict -----------------------------------------------------------------------------
    def evict(self, episode_id: str, reason: EvictReason, *, expected_sandbox: Any = None) -> bool:
        """Remove one episode and tear its sandbox down exactly once."""
        with self._lock:
            entry = self._entries.get(episode_id)
            if entry is None:
                return False
            if expected_sandbox is not None and entry.sandbox is not expected_sandbox:
                return False
            self._entries.pop(episode_id)
            self.evictions[reason] += 1
            self._teardown_once(entry, reason)
            return True

    def heartbeat_all(self, max_workers: int = 32) -> tuple[int, int]:
        """Extend owned sandboxes; only a confirmed absence authorises eviction.

        ``sandbox.heartbeat()`` returns True (alive), False (confirmed gone) or None (unknown).
        Unknown transport failures retain ownership, and an old heartbeat result cannot evict a
        replacement sandbox that reused the episode id.
        """
        with self._lock:
            items = [(eid, e.sandbox) for eid, e in self._entries.items() if e.sandbox is not None]
        if not items:
            return (0, 0)

        def beat(item):
            episode_id, sandbox = item
            try:
                return episode_id, sandbox, sandbox.heartbeat()
            except Exception as exc:  # noqa: BLE001 - unknown, keep ownership
                logger.warning("[carry] heartbeat unknown for %s: %s", episode_id, exc)
                return episode_id, sandbox, None

        alive = dead = 0
        with cf.ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as pool:
            for episode_id, sandbox, ok in pool.map(beat, items):
                if ok is True:
                    alive += 1
                elif ok is False:
                    dead += self.evict(episode_id, EvictReason.TIMEOUT, expected_sandbox=sandbox)
        return alive, dead

    def retain(self, episode_ids: Iterable[str]) -> int:
        """Release every entry no longer reachable from the partial buffer.

        Call only after all generation tasks have settled at a rollout cut.
        """
        keep = set(episode_ids)
        with self._lock:
            discard = set(self._entries) - keep
            for episode_id in discard:
                self.evict(episode_id, EvictReason.GROUP_VOIDED)
            return len(discard)

    def parked_count(self) -> int:
        with self._lock:
            return sum(e.is_parked for e in self._entries.values())

    def evict_group(self, group_id: str, reason: EvictReason) -> int:
        """Evict every episode of a group: a voided group must not leave stragglers."""
        with self._lock:
            ids = [e.episode_id for e in self._entries.values() if e.group_id == group_id]
            for episode_id in ids:
                self.evict(episode_id, reason)
            return len(ids)

    def evict_stale(self, current_version: int, max_staleness: int) -> list[str]:
        """Drop episodes aged past ``max_staleness`` optimizer updates.

        Per episode, not per group: ``group_id`` identifies the task, which every episode of that
        task shares across rollouts and admission versions. Evicting per group would take fresh
        parks down with genuinely old ones (measured on the research runs before this change).
        """
        with self._lock:
            stale = [e.episode_id for e in self._entries.values()
                     if e.staleness(current_version) > max_staleness]
            for episode_id in stale:
                self.evict(episode_id, EvictReason.STALE)
            return sorted(stale)

    def evict_timed_out(self, budget_s: float) -> list[str]:
        with self._lock:
            ids = [e.episode_id for e in self._entries.values() if e.expired(budget_s)]
            for episode_id in ids:
                self.evict(episode_id, EvictReason.TIMEOUT)
            return ids

    def drain(self) -> int:
        """Evict everything. Call before the run exits so no sandbox outlives its process."""
        with self._lock:
            ids = list(self._entries)
            for episode_id in ids:
                self.evict(episode_id, EvictReason.DRAIN)
            return len(ids)

    # ---- inspection ------------------------------------------------------------------------
    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def reap_expired(self, ttl_seconds: float) -> int:
        """Evict entries whose sandbox has certainly outlived the backend's lifetime limit.

        Entries leave this registry on completion, staleness, voiding, timeout or drain, all of
        which require the episode to be *resumed* first. A sandbox backend that deletes
        containers by itself after a lifetime limit gives no callback, so a parked episode that
        is never resumed would keep its entry and its admission permit forever. Evicting on age
        returns the permit; a resumed episode that finds no entry restarts from turn 0.
        """
        if ttl_seconds <= 0:
            return 0
        now = time.monotonic()   # created_at is monotonic too
        with self._lock:
            expired = [eid for eid, e in self._entries.items() if now - e.created_at >= ttl_seconds]
        reaped = sum(self.evict(episode_id, EvictReason.TIMEOUT) for episode_id in expired)
        if reaped:
            logger.info("[carry] reaped %d registry entries past the %.0fs sandbox lifetime",
                        reaped, ttl_seconds)
        return reaped

    def trim_to(self, max_entries: int) -> int:
        """Evict the LEAST-PROGRESSED parked episodes until at most ``max_entries`` remain.

        The carry pool needs a hard bound: parked episodes hold sandbox permits, and if carry
        grows into the whole ceiling every new episode blocks on admission and the rollout
        wedges. Least-progressed first because turn count is the value of a carried episode (a
        re-admitted episode restarts from turn 0); ``created_at`` breaks ties.
        """
        if max_entries < 0:
            return 0
        with self._lock:
            if len(self._entries) <= max_entries:
                return 0
            ordered = sorted(self._entries.items(),
                             key=lambda kv: (kv[1].turn_index, kv[1].created_at))
            surplus = [eid for eid, _ in ordered[: len(self._entries) - max_entries]]
        trimmed = sum(self.evict(episode_id, EvictReason.TIMEOUT) for episode_id in surplus)
        if trimmed:
            logger.info("[carry] trimmed %d parked episode(s) to hold the carry pool at %d",
                        trimmed, max_entries)
        return trimmed

    def episode_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._entries)

    def group_ids(self) -> set[str]:
        with self._lock:
            return {e.group_id for e in self._entries.values()}

    def metrics(self) -> dict[str, float]:
        with self._lock:
            m: dict[str, float] = {
                "registry/parked_episodes": float(sum(e.is_parked for e in self._entries.values())),
                "registry/active_episodes": float(sum(not e.is_parked for e in self._entries.values())),
                "registry/parked_groups": float(len({e.group_id for e in self._entries.values()})),
                "registry/double_teardowns": float(len(self._double_teardowns)),
            }
            for reason, n in self.evictions.items():
                m[f"registry/evicted_{reason.value}"] = float(n)
            return m

    def assert_no_leaks(self, expected_torn_down: Iterable[str] | None = None) -> None:
        """Test hook: registry empty, nothing torn down twice, everything expected torn down."""
        with self._lock:
            assert not self._entries, f"registry not empty: {sorted(self._entries)}"
            assert not self._double_teardowns, (
                f"sandbox torn down more than once: {self._double_teardowns}")
            if expected_torn_down is not None:
                missing = set(expected_torn_down) - self._torn_down
                assert not missing, f"sandboxes never torn down: {sorted(missing)}"

    # ---- internal --------------------------------------------------------------------------
    def _teardown_once(self, entry: EpisodeEntry, reason: EvictReason) -> None:
        if entry.episode_id in self._torn_down:
            self._double_teardowns.append(entry.episode_id)
            logger.error("[registry] double teardown of %s (%s)", entry.episode_id, reason.value)
            return
        self._torn_down.add(entry.episode_id)
        if entry.sandbox is None:
            return
        try:
            self._teardown(entry.sandbox)
        except Exception as exc:  # noqa: BLE001 - teardown must never break the training loop
            logger.warning("[registry] sandbox teardown failed for %s (%s): %s",
                           entry.episode_id, reason.value, exc)
        finally:
            entry.sandbox = None


def _default_teardown(sandbox: Any) -> None:
    close = getattr(sandbox, "close", None)
    if close is not None:
        close()
