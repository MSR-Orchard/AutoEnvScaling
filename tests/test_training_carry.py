"""Partial rollout and over-sampling (training.carry): carry across weight updates, batch
filling, filtering, token and log-probability preservation, and restart behaviour. Everything
runs offline with a fake episode driver and a fake Miles data source."""

from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace

import pytest

from autoenvscaling.training.carry import episode as ep
from autoenvscaling.training.carry import lifecycle as lc
from autoenvscaling.training.carry import rollout as ro
from autoenvscaling.training.carry.registry import EpisodeRegistry, EvictReason
from autoenvscaling.training.miles.config import CARRY, MilesConfig, carry_solver
from autoenvscaling.training.miles.launcher import MilesLauncher

N = 2   # samples per prompt in these tests


# ---- fakes -----------------------------------------------------------------------------------------
class FakeSandbox:
    def __init__(self, name, on_close=None):
        self.name = name
        self.closed = 0
        self.paused = 0
        self.resumed = 0
        self.alive = True
        self.on_close = on_close

    def close(self):
        self.closed += 1
        if self.on_close is not None:
            self.on_close(self)

    def pause_execution(self):
        self.paused += 1

    def resume_execution(self):
        self.resumed += 1

    def heartbeat(self):
        return self.alive


def sample(index, group_index=0, prompt_tokens=(1, 2, 3), **extra):
    s = SimpleNamespace(index=index, group_index=group_index, prompt="task", tokens=list(prompt_tokens),
                        response="", response_length=0, rollout_log_probs=None, loss_mask=None,
                        reward=None, status="pending", metadata={}, remove_sample=False,
                        weight_versions=[])
    for k, v in extra.items():
        setattr(s, k, v)
    return s


class FakeDriver:
    """Each run_turns call generates ``tokens_per_turn`` tokens; episodes listed in ``slow``
    keep generating until the cut and park; the rest finish after ``turns`` turns."""

    def __init__(self, *, rewards=None, slow=(), turns=2, tokens_per_turn=3):
        self.rewards = rewards or {}
        self.slow = set(slow)
        self.turns = turns
        self.tokens_per_turn = tokens_per_turn
        self.sandboxes: dict[str, FakeSandbox] = {}
        self.created = 0
        self.fail = set()
        self.closing: set = set()          # sandboxes whose deletion is not yet confirmed
        self.close_delay_s = 0.0

    def _closing_hook(self, sb):
        if self.close_delay_s > 0:
            self.closing.add(sb)
            asyncio.get_running_loop().call_later(self.close_delay_s, self.closing.discard, sb)

    async def create_sandbox(self, s):
        self.created += 1
        sb = FakeSandbox(f"sb-{ep.episode_id(s)}-{self.created}", on_close=self._closing_hook)
        self.sandboxes[ep.episode_id(s)] = sb
        return sb

    def _generate(self, s, version):
        ids = [100 * (version + 1) + i for i in range(self.tokens_per_turn)]
        s.tokens.extend(ids)
        s.rollout_log_probs = (s.rollout_log_probs or []) + [-0.5 - version] * len(ids)
        s.loss_mask = (s.loss_mask or []) + [1] * len(ids)
        s.response_length += len(ids)
        s.response += "x" * len(ids)

    async def run_turns(self, *, sample, sandbox, turn_state, sampling_params, should_abort,
                        start_turn_index):
        eid = ep.episode_id(sample)
        if eid in self.fail:
            raise RuntimeError("boom")
        turn = start_turn_index
        version = sampling_params.get("version", 0)
        while True:
            if should_abort():
                return ep.TurnOutcome(ep.TurnStatus.ABORTED, {"turn": turn}, turn)
            self._generate(sample, version)
            turn += 1
            if eid not in self.slow and turn >= self.turns:
                return ep.TurnOutcome(ep.TurnStatus.DONE, {"turn": turn}, turn)
            await asyncio.sleep(0.005)

    async def verify(self, *, sample, sandbox, turn_state):
        return self.rewards.get(ep.episode_id(sample), 1.0)


@pytest.fixture
def registry():
    reg = ep.reset_registry()
    lc.CLOCK.base_rollout_id = None
    lc.CLOCK.active_version = None
    ep.COUNTERS.drain()
    yield reg
    ep.reset_registry()


# ---- registry ----------------------------------------------------------------------------------------
def test_registry_park_resume_and_eviction_paths():
    reg = EpisodeRegistry()
    sb = FakeSandbox("a")
    reg.park(episode_id="e1", group_id="g", sandbox=sb, turn_state={"turn": 3}, admission_version=0,
             start_rollout_id=0, parked_at_version=0, turn_index=3)
    assert sb.paused == 1 and reg.parked_count() == 1
    entry = reg.resume("e1", resumed_at_version=2, token_index=40)
    assert entry is not None and entry.sandbox is sb and sb.resumed == 1 and not entry.is_parked
    assert entry.resume_boundaries[0].token_index == 40 and entry.resume_boundaries[0].parked_at_version == 0
    assert entry.resume_boundaries[0].resumed_at_version == 2 and entry.staleness(2) == 2
    with pytest.raises(RuntimeError):
        reg.resume("e1", resumed_at_version=2, token_index=40)   # already active
    # re-park keeps the admission version and accumulates boundaries
    reg.park(episode_id="e1", group_id="g", sandbox=sb, turn_state={"turn": 5}, admission_version=99,
             start_rollout_id=0, parked_at_version=2, turn_index=5)
    assert reg.get("e1").admission_version == 0 and reg.get("e1").turn_index == 5
    # staleness: parked at 0, resumed at 6 with a bound of 4 -> evicted, miss
    assert reg.resume("e1", resumed_at_version=6, token_index=50, max_staleness=4) is None
    assert sb.closed == 1 and reg.evictions[EvictReason.STALE] == 1 and len(reg) == 0
    assert reg.evict("e1", EvictReason.COMPLETED) is False
    reg.assert_no_leaks(expected_torn_down=["e1"])


def test_registry_group_eviction_retain_trim_and_heartbeat():
    reg = EpisodeRegistry()
    boxes = {}
    for i, turn in enumerate([5, 1, 3]):
        boxes[i] = FakeSandbox(str(i))
        reg.park(episode_id=f"e{i}", group_id="g0" if i < 2 else "g1", sandbox=boxes[i], turn_state=None,
                 admission_version=0, start_rollout_id=0, parked_at_version=0, turn_index=turn)
    assert reg.group_ids() == {"g0", "g1"}
    assert reg.trim_to(2) == 1 and "e1" not in reg.episode_ids()        # least progressed first
    assert boxes[1].closed == 1
    boxes[2].alive = False
    alive, dead = reg.heartbeat_all()
    assert (alive, dead) == (1, 1) and boxes[2].closed == 1               # confirmed absence only
    boxes[0].alive = None
    assert reg.heartbeat_all() == (0, 0) and boxes[0].closed == 0          # unknown keeps ownership
    assert reg.retain([]) == 1 and len(reg) == 0
    assert reg.metrics()["registry/evicted_group_voided"] == 1.0
    reg.assert_no_leaks()
    # a fresh park reusing an id must not inherit the old instance's teardown record
    reg.park(episode_id="e0", group_id="g0", sandbox=FakeSandbox("new"), turn_state=None,
             admission_version=0, start_rollout_id=0, parked_at_version=0)
    assert reg.evict("e0", EvictReason.DRAIN) and not reg._double_teardowns


# ---- carry across weight updates ------------------------------------------------------------------------
async def _run(s, driver, registry, *, version, abort, rollout_id=None):
    return await ep.run_episode(sample=s, sampling_params={"version": version}, driver=driver,
                                registry=registry, should_abort=abort, weight_version=version,
                                rollout_id=version if rollout_id is None else rollout_id,
                                config=ep.CarryConfig(max_weight_staleness=4, episode_timeout_s=None))


async def test_episode_parks_and_resumes_under_new_weights_keeping_tokens(registry):
    driver = FakeDriver(slow=["group-0-r0"], rewards={"group-0-r0": 0.5})
    s = sample(0)
    flag = {"abort": False}

    async def cut_soon():
        await asyncio.sleep(0.03)
        flag["abort"] = True

    asyncio.create_task(cut_soon())
    await _run(s, driver, registry, version=0, abort=lambda: flag["abort"])
    assert ep.get_status(s) == "aborted" and s.metadata["admission_version"] == 0
    assert s.metadata["start_rollout_id"] == 0 and len(registry) == 1
    parked_tokens, parked_lps = list(s.tokens), list(s.rollout_log_probs)
    assert len(parked_lps) == s.response_length > 0 and all(lp == -0.5 for lp in parked_lps)
    sb = driver.sandboxes["group-0-r0"]
    assert registry.get("group-0-r0").sandbox is sb and sb.paused == 1 and sb.closed == 0

    # The optimizer stepped: resume under version 1 in the SAME sandbox; nothing is recomputed.
    driver.slow.clear()
    await _run(s, driver, registry, version=1, abort=lambda: False)
    assert ep.get_status(s) == "completed" and s.reward == 0.5
    assert s.tokens[:len(parked_tokens)] == parked_tokens
    assert s.rollout_log_probs[:len(parked_lps)] == parked_lps
    new = s.rollout_log_probs[len(parked_lps):]
    assert new and all(lp == -1.5 for lp in new)               # sampled by the new policy
    assert len(s.rollout_log_probs) == s.response_length == len(s.loss_mask)
    assert s.metadata["admission_version"] == 0 and s.metadata["resumes"] == 1
    boundary = s.metadata["resume_boundaries"][0]
    assert boundary[0] == len(parked_tokens) and boundary[2:] == [0, 1]
    assert driver.created == 1 and sb.resumed == 1 and sb.closed == 1
    registry.assert_no_leaks(expected_torn_down=["group-0-r0"])
    counters = ep.COUNTERS.drain()
    assert counters["carry/parked_this_rollout"] == 1 and counters["carry/resumed_this_rollout"] == 1


async def test_restart_miss_discards_partial_tokens_and_restarts(registry):
    driver = FakeDriver(rewards={"group-0-r0": 1.0})
    s = sample(0, prompt_tokens=(7, 8))
    s.tokens += [500, 501]
    s.response_length = 2
    s.rollout_log_probs = [-1.0, -1.0]
    s.loss_mask = [1, 1]
    s.metadata = {"admission_version": 0, "resumes": 1}
    await _run(s, driver, registry, version=3, abort=lambda: False)   # empty registry: a miss
    assert ep.get_status(s) == "completed" and s.tokens[:2] == [7, 8] and 500 not in s.tokens
    assert s.metadata["admission_version"] == 3 and s.metadata["resumes"] == 0
    assert ep.COUNTERS.drain()["carry/resume_misses_this_rollout"] == 1
    registry.assert_no_leaks()


async def test_stale_parked_episode_is_evicted_and_restarted(registry):
    driver = FakeDriver(slow=["group-0-r0"])
    s = sample(0)
    flag = {"abort": False}
    asyncio.get_running_loop().call_later(0.02, lambda: flag.__setitem__("abort", True))
    await _run(s, driver, registry, version=0, abort=lambda: flag["abort"])
    old_sb = driver.sandboxes["group-0-r0"]
    driver.slow.clear()
    await _run(s, driver, registry, version=5, abort=lambda: False)    # 5 - 0 > 4
    assert old_sb.closed == 1 and driver.created == 2 and s.metadata["admission_version"] == 5
    assert registry.evictions[EvictReason.STALE] == 1
    registry.assert_no_leaks()


async def test_failed_episode_is_contained_and_its_sandbox_released(registry):
    driver = FakeDriver()
    driver.fail.add("group-0-r0")
    s = sample(0)
    await _run(s, driver, registry, version=0, abort=lambda: False)
    assert ep.get_status(s) == "aborted" and "boom" in s.metadata["carry_error"]
    assert s.reward == 0.0 and driver.sandboxes["group-0-r0"].closed == 1
    registry.assert_no_leaks()


def test_normalize_sample_repairs():
    s = sample(0)
    ep.normalize_sample(s)
    assert s.reward == 0.0 and s.rollout_log_probs == []
    s = sample(1, tokens=[1, 2, 3, 4], response_length=2, rollout_log_probs=[-1.0], loss_mask=[1, 1], reward=1.0)
    ep.normalize_sample(s)
    assert s.response_length == 0 and s.tokens == [1, 2] and s.reward == 0.0 and ep.get_status(s) == "aborted"
    s = sample(2, tokens=[], response_length=0, rollout_log_probs=[], reward=1.0)
    ep.normalize_sample(s)
    assert s.tokens == [0] and s.response_length == 0


# ---- lifecycle: group validity, buffer, capacity ---------------------------------------------------------
def completed(index, group_index=0, reward=1.0, admission=0):
    s = sample(index, group_index, tokens=[1, 2, 3, 4, 5], response_length=2, rollout_log_probs=[-1.0, -2.0],
               loss_mask=[1, 1], reward=reward, status="completed")
    s.metadata = {"admission_version": admission}
    return s


def test_group_error_cases(registry):
    rules = lc.ValidityRules(n_samples_per_prompt=2, max_weight_staleness=4)
    good = [completed(0), completed(1, reward=0.0)]
    assert lc.group_error(good, rules, version=0, for_training=True) is None
    assert lc.group_error(good[:1], rules, version=0) == "incomplete_group"
    assert lc.group_error(good, rules, version=5, for_training=True) == "stale"
    bad = [completed(0), completed(1)]
    bad[1].rollout_log_probs = [-1.0]
    assert lc.group_error(bad, rules, version=0) == "invalid_logprobs_or_mask"
    bad[1].rollout_log_probs = [-1.0, float("nan")]
    assert lc.group_error(bad, rules, version=0) == "invalid_logprobs_or_mask"
    pending = [completed(0), sample(1)]
    assert lc.group_error(pending, rules, version=0, for_training=True) == "unfinished_episode"
    assert lc.group_error(pending, rules, version=0) is None          # pending, never admitted
    pending[1].metadata["admission_version"] = 0
    assert lc.group_error(pending, rules, version=0) == "missing_sandbox"
    registry.park(episode_id="group-0-r1", group_id="group-0", sandbox=FakeSandbox("x"), turn_state=None,
                  admission_version=0, start_rollout_id=0, parked_at_version=0)
    assert lc.group_error(pending, rules, version=0) is None
    assert lc.group_error(pending, rules, version=5) == "stale"
    failed = [completed(0), completed(1)]
    failed[0].metadata["carry_error"] = "x"
    assert lc.group_error(failed, rules, version=0) == "episode_failed"
    lc.discard_group(pending, "stale")
    assert registry.evictions[EvictReason.STALE] == 1 and all(s.remove_sample for s in pending)


def test_reconcile_buffer_keeps_resumable_groups_and_releases_the_rest(registry):
    rules = lc.ValidityRules(n_samples_per_prompt=2, max_weight_staleness=4)
    live = [completed(0, 0, admission=3), sample(1, 0)]
    live[1].metadata["admission_version"] = 3
    registry.park(episode_id="group-0-r1", group_id="group-0", sandbox=FakeSandbox("live"), turn_state=None,
                  admission_version=3, start_rollout_id=3, parked_at_version=3)
    old = [sample(2, 1), sample(3, 1)]
    for s in old:
        s.metadata["admission_version"] = 0
        registry.park(episode_id=ep.episode_id(s), group_id="group-1", sandbox=FakeSandbox("old"),
                      turn_state=None, admission_version=0, start_rollout_id=0, parked_at_version=0)
    orphan = FakeSandbox("orphan")
    registry.park(episode_id="group-9-r9", group_id="group-9", sandbox=orphan, turn_state=None,
                  admission_version=0, start_rollout_id=0, parked_at_version=0)
    kept = lc.reconcile_buffer([live, old], rules, next_version=5, carry_allowance=10)
    assert kept == [live] and orphan.closed == 1 and registry.episode_ids() == ["group-0-r1"]
    assert registry.evictions[EvictReason.STALE] == 2 and registry.evictions[EvictReason.GROUP_VOIDED] == 1
    with pytest.raises(RuntimeError):
        lc.reconcile_buffer([live], rules, next_version=1, carry_allowance=0)


def test_take_buffered_groups_claims_everything(registry):
    rules = lc.ValidityRules(n_samples_per_prompt=2)

    class Source:
        def __init__(self):
            self.buffer = [[completed(0, 0), completed(1, 0, 0.0)], [completed(2, 1), completed(3, 1)]]

        def get_samples(self, n):
            out, self.buffer = self.buffer[:n], self.buffer[n:]
            return out

    src = Source()
    src.buffer[1][0].metadata["carry_error"] = "x"
    groups = lc.take_buffered_groups(src.get_samples, rules, version=0)
    assert len(groups) == 1 and src.buffer == [] and groups[0][0].index == 0
    with pytest.raises(TypeError):
        lc.take_buffered_groups(lambda n: [], rules, version=0)


def test_validate_training_batch_and_capacity(registry):
    rules = lc.ValidityRules(n_samples_per_prompt=2)
    g = [completed(0, admission=1), completed(1, reward=0.0, admission=3)]
    g[0].metadata["resumes"] = 2
    m = lc.validate_training_batch([g], rules, version=3)
    assert m["carry/max_staleness"] == 2 and m["carry/resumed_trainable_samples"] == 1
    assert m["carry/trainable_samples"] == 2
    with pytest.raises(RuntimeError):
        lc.validate_training_batch([[completed(0), sample(1)]], rules, version=0)
    assert lc.Capacity(1024, 768, 768).problems(16) == []
    assert lc.Capacity(1024, 768, 256).problems(16)          # the launcher's old 256 fails
    assert lc.Capacity(1024, 8, 1024).problems(16)


def test_weight_clock_is_anchored_to_the_first_rollout():
    clock = lc.WeightClock()
    assert clock.current_version() == 0
    assert clock.begin(77) == 0 and clock.begin(78) == 1 and clock.current_version() == 1


# ---- batch filling: over-sampling, filtering, the cut and the carry ------------------------------------
class FakeSource:
    """Miles' buffered data source: the buffer first, then fresh prompts in order."""

    def __init__(self, rewards):
        self.buffer = []
        self.rewards = rewards       # group_index -> list of rewards per member
        self.next_group = 0
        self.next_index = 0
        self.drawn = 0

    def get_samples(self, n):
        out, self.buffer = self.buffer[:n], self.buffer[n:]
        while len(out) < n and self.next_group < len(self.rewards):
            g = [sample(self.next_index + i, self.next_group) for i in range(N)]
            self.next_index += N
            self.next_group += 1
            self.drawn += 1
            out.append(g)
        return out


class FakeGate:
    def __init__(self):
        self.calls = []

    def park(self):
        self.calls.append("park")

    def unpark(self):
        self.calls.append("unpark")

    def quiesce(self):
        self.calls.append("quiesce")
        return {"quiescent": True}

    def check(self):
        pass


def make_run_group(driver, registry, state, *, version, rollout_id):
    """The per-group runner `generate_rollout` builds, over the fake driver."""

    async def run_sample(s):
        try:
            if s.status in ("completed", "truncated"):
                reason = lc.completed_sample_error(s, RULES, version)
                if reason:
                    lc.discard_group([s], reason, registry=registry)
                    s.status = "aborted"
                return s
            await ep.run_episode(sample=s, sampling_params={"version": version}, driver=driver,
                                 registry=registry, should_abort=lambda: state.aborted,
                                 weight_version=version, rollout_id=rollout_id,
                                 config=ep.CarryConfig(max_weight_staleness=4))
            s.status = "completed" if ep.get_status(s) == "completed" else "aborted"
            return s
        finally:
            state.inflight_samples -= 1

    async def run_group(group):
        if state.aborted:
            state.inflight_samples -= len(group)
            return list(group)
        return list(await asyncio.gather(*(run_sample(s) for s in group)))

    return run_group


RULES = lc.ValidityRules(n_samples_per_prompt=N, max_weight_staleness=4)


def zero_std(group):
    rewards = [s.reward for s in group]
    keep = max(rewards) - min(rewards) > 1e-8
    return keep, None if keep else f"zero_std_{rewards[0]}"


async def rollout(source, driver, registry, *, rollout_id, rewards, cfg, gate, peak):
    version = lc.CLOCK.begin(rollout_id)
    state = ro.RolloutState()
    hooks = ro.CarryHooks(RULES, registry=registry, gate=gate, abort_engines=None, held=lambda: 0)
    run_group = make_run_group(driver, registry, state, version=version, rollout_id=rollout_id)
    driver.rewards = rewards

    async def watch():
        while not state.aborted:
            peak[0] = max(peak[0], state.inflight_samples)
            await asyncio.sleep(0.001)

    watcher = asyncio.create_task(watch())
    result = await ro.collect_batch(cfg, data_source=source.get_samples, run_group=run_group,
                                    dynamic_filter=zero_std, hooks=hooks, state=state, rollout_id=rollout_id)
    watcher.cancel()
    source.buffer = lc.reconcile_buffer(result.aborted, RULES, next_version=version + 1, carry_allowance=100,
                                        registry=registry)
    return result


async def test_batch_fills_by_oversampling_filters_constant_groups_and_carries_the_rest(registry):
    # groups 0..9: 0,1 constant (dropped); 2,3,4 mixed; 5 slow (parks); 6,7 mixed; 8,9 mixed
    rewards = {}
    for g in range(10):
        rewards[g] = [1.0, 1.0] if g in (0, 1) else [1.0, 0.0]
    per_episode = {f"group-{g}-r{N * g + i}": rewards[g][i] for g in rewards for i in range(N)}
    source = FakeSource(rewards)
    driver = FakeDriver(slow=["group-5-r10", "group-5-r11"])
    cfg = ro.LoopConfig(rollout_batch_size=3, n_samples_per_prompt=N, over_sampling_batch_size=4,
                        max_concurrent_episodes=6, abort_drain_timeout_s=2, wait_timeout_s=0.01)
    gate = FakeGate()
    peak = [0]
    result = await rollout(source, driver, registry, rollout_id=10, rewards=per_episode, cfg=cfg, gate=gate,
                           peak=peak)
    assert len(result.data) == 3 and all(zero_std(g)[0] for g in result.data)
    assert [g[0].group_index for g in result.data] == sorted(g[0].group_index for g in result.data)
    assert result.groups_dropped.get("zero_std_1.0") == 2
    assert peak[0] <= cfg.max_concurrent_episodes                     # the episode budget bounds admission
    assert gate.calls == ["park", "quiesce"]                           # the cut: gate closed, quiescence confirmed
    assert result.metrics["carry/trainable_groups"] == 3 and result.metrics["carry/weight_version"] == 0
    # The slow group parked with its sandboxes. Any other group still running at the cut is a
    # valid surplus that parks too, and completed surplus groups are carried as they are.
    def parked_members(groups):
        return sorted(ep.episode_id(s) for g in groups for s in g if s.status == "aborted")

    slow = next(g for g in result.aborted if g[0].group_index == 5)
    assert all(s.status == "aborted" for s in slow)
    assert all(s.metadata["start_rollout_id"] == 10 for s in slow)
    parked_ids = parked_members(result.aborted)
    assert {"group-5-r10", "group-5-r11"} <= set(parked_ids)
    assert registry.episode_ids() == parked_ids                 # exactly the parked episodes hold a sandbox
    assert all(s.status in ("completed", "aborted") for g in result.aborted for s in g)
    assert source.buffer == result.aborted
    # every dropped, trained or completed episode released its sandbox; only parked ones remain
    assert sum(sb.closed for sb in driver.sandboxes.values()) == driver.created - len(parked_ids)

    # Next rollout: the buffer is claimed first, parked episodes resume under version 1 and keep
    # the tokens they sampled under version 0; carried completed groups are reused as they are.
    driver.slow.clear()
    before = {ep.episode_id(s): (list(s.tokens), list(s.rollout_log_probs)) for s in slow}
    cfg2 = ro.LoopConfig(rollout_batch_size=len(source.buffer) + 1, n_samples_per_prompt=N,
                         over_sampling_batch_size=2, max_concurrent_episodes=6, abort_drain_timeout_s=2,
                         wait_timeout_s=0.01)
    result2 = await rollout(source, driver, registry, rollout_id=11, rewards=per_episode, cfg=cfg2, gate=gate,
                            peak=peak)
    assert result2.metrics["carry/groups_carried_in"] == len(result.aborted)
    resumed = next(g for g in result2.data if g[0].group_index == 5)
    for s in resumed:
        toks, lps = before[ep.episode_id(s)]
        assert s.tokens[:len(toks)] == toks and s.rollout_log_probs[:len(lps)] == lps
        assert len(s.rollout_log_probs) > len(lps) and s.metadata["resumes"] == 1
        assert s.metadata["admission_version"] == 0
    assert result2.metrics["carry/resumed_trainable_samples"] >= 2
    assert result2.metrics["carry/max_staleness"] == 1
    # whatever was still running at this cut parked again; nothing else holds a sandbox
    assert registry.episode_ids() == parked_members(result2.aborted)
    registry.drain()
    registry.assert_no_leaks()


async def test_abort_drain_cancels_stragglers_and_their_sandboxes_are_released(registry):
    class StuckDriver(FakeDriver):
        async def run_turns(self, **kw):
            await asyncio.sleep(3600)   # ignores should_abort

    source = FakeSource({0: [1.0, 0.0], 1: [1.0, 0.0]})
    driver = StuckDriver()
    done = FakeDriver(rewards={"group-0-r0": 1.0, "group-0-r1": 0.0})

    async def run_group_factory(state):
        pass

    state = ro.RolloutState()
    hooks = ro.CarryHooks(RULES, registry=registry, gate=FakeGate(), held=lambda: 0)
    version = lc.CLOCK.begin(0)

    async def run_group(group):
        # group 0 completes through the normal driver, group 1 wedges
        drv = done if group[0].group_index == 0 else driver
        try:
            await asyncio.gather(*(ep.run_episode(sample=s, sampling_params={}, driver=drv, registry=registry,
                                                  should_abort=lambda: state.aborted, weight_version=version,
                                                  rollout_id=0, config=ep.CarryConfig()) for s in group))
            for s in group:
                s.status = "completed" if ep.get_status(s) == "completed" else "aborted"
            return group
        finally:
            state.inflight_samples -= len(group)

    cfg = ro.LoopConfig(rollout_batch_size=1, n_samples_per_prompt=N, over_sampling_batch_size=2,
                        max_concurrent_episodes=4, abort_drain_timeout_s=0.05, wait_timeout_s=0.01)
    result = await ro.collect_batch(cfg, data_source=source.get_samples, run_group=run_group, dynamic_filter=zero_std,
                                    hooks=hooks, state=state, rollout_id=0)
    assert len(result.data) == 1 and result.aborted == [] and not state.pendings
    assert all(sb.closed == 1 for sb in driver.sandboxes.values())    # cancelled episodes release
    registry.assert_no_leaks()


async def test_admission_refills_free_slots_counting_closing_sandboxes_once(registry):
    """Capacity 6 with four active (slow) episodes: exactly the two free slots are refilled, a
    finished group's sandboxes hold capacity only while their deletion is unconfirmed, and the
    total of in-flight episodes plus closing sandboxes never exceeds the capacity."""
    rewards = {g: [1.0, 0.0] for g in range(6)}
    per_episode = {f"group-{g}-r{N * g + i}": rewards[g][i] for g in rewards for i in range(N)}
    source = FakeSource(rewards)
    driver = FakeDriver(slow=["group-0-r0", "group-0-r1", "group-1-r2", "group-1-r3"], rewards=per_episode)
    driver.close_delay_s = 0.03
    state = ro.RolloutState()
    version = lc.CLOCK.begin(0)
    hooks = ro.CarryHooks(RULES, registry=registry, gate=FakeGate(), held=lambda: len(driver.closing))
    cfg = ro.LoopConfig(rollout_batch_size=2, n_samples_per_prompt=N, over_sampling_batch_size=2,
                        max_concurrent_episodes=6, abort_drain_timeout_s=2, wait_timeout_s=0.005)
    run_group = make_run_group(driver, registry, state, version=version, rollout_id=0)
    peak = [0]
    refilled = []

    async def watch():
        while not state.aborted:
            total = state.inflight_samples + len(driver.closing)
            peak[0] = max(peak[0], total)
            await asyncio.sleep(0.001)

    watcher = asyncio.create_task(watch())
    result = await ro.collect_batch(cfg, data_source=source.get_samples, run_group=run_group,
                                    dynamic_filter=zero_std, hooks=hooks, state=state, rollout_id=0)
    watcher.cancel()
    refilled = [g[0].group_index for g in result.data]
    assert peak[0] <= 6, f"capacity exceeded: {peak[0]}"
    assert peak[0] == 6                                                # the two free slots were used
    assert refilled == [2, 3] and source.drawn == 4                    # one fresh group at a time
    assert len(registry) == 4                                          # the slow groups parked
    registry.drain()


def test_run_identity_defaults_to_the_save_directory(tmp_path):
    a = ro.resolve_identity(ro.CarrySettings(), str(tmp_path / "ckpt-a"))
    again = ro.resolve_identity(ro.CarrySettings(), str(tmp_path / "ckpt-a"))
    b = ro.resolve_identity(ro.CarrySettings(), str(tmp_path / "ckpt-b"))
    assert a.run_id == again.run_id and a.run_id.startswith("aes-ckpt-a-")      # stable across restarts
    assert a.run_id != b.run_id                                                 # distinct jobs
    assert a.state_dir == str((tmp_path / "ckpt-a").resolve() / "carry")
    explicit = ro.resolve_identity(ro.CarrySettings(run_id="mine", state_dir="/s"), str(tmp_path / "x"))
    assert (explicit.run_id, explicit.state_dir) == ("mine", "/s")
    with pytest.raises(RuntimeError, match="AES_CARRY_RUN_ID"):
        ro.resolve_identity(ro.CarrySettings(), "")


async def test_cut_bounds_the_initial_abort_request(registry):
    source = FakeSource({0: [1.0, 0.0]})
    driver = FakeDriver(rewards={"group-0-r0": 1.0, "group-0-r1": 0.0})
    state = ro.RolloutState()
    version = lc.CLOCK.begin(0)

    async def never_returns():
        await asyncio.sleep(3600)

    gate = FakeGate()
    hooks = ro.CarryHooks(RULES, registry=registry, gate=gate, abort_engines=never_returns, held=lambda: 0)
    cfg = ro.LoopConfig(rollout_batch_size=1, n_samples_per_prompt=N, over_sampling_batch_size=1,
                        max_concurrent_episodes=4, abort_request_timeout_s=0.05, abort_drain_timeout_s=1,
                        wait_timeout_s=0.01)
    run_group = make_run_group(driver, registry, state, version=version, rollout_id=0)
    result = await asyncio.wait_for(
        ro.collect_batch(cfg, data_source=source.get_samples, run_group=run_group, dynamic_filter=zero_std,
                         hooks=hooks, state=state, rollout_id=0), 5)
    assert len(result.data) == 1 and gate.calls == ["park", "quiesce"]


async def test_exhausted_source_raises_instead_of_training_short(registry):
    source = FakeSource({0: [1.0, 1.0]})
    driver = FakeDriver(rewards={"group-0-r0": 1.0, "group-0-r1": 1.0})
    state = ro.RolloutState()
    version = lc.CLOCK.begin(0)
    cfg = ro.LoopConfig(rollout_batch_size=1, n_samples_per_prompt=N, over_sampling_batch_size=1,
                        max_concurrent_episodes=4, wait_timeout_s=0.01)
    hooks = ro.CarryHooks(RULES, registry=registry, gate=FakeGate(), held=lambda: 0)
    run_group = make_run_group(driver, registry, state, version=version, rollout_id=0)
    with pytest.raises(RuntimeError, match="exhausted"):
        await ro.collect_batch(cfg, data_source=source.get_samples, run_group=run_group, dynamic_filter=zero_std,
                               hooks=hooks, state=state, rollout_id=0)


# ---- restart ---------------------------------------------------------------------------------------------
def test_restart_manifest_reports_what_the_previous_process_held(tmp_path):
    ro.write_manifest(tmp_path, 41, buffered_groups=3, parked_episodes=["a", "b"])
    previous = ro.read_previous_manifest(tmp_path, 42)
    assert previous["buffered_groups"] == 3 and previous["parked_episodes"] == ["a", "b"]
    assert ro.read_previous_manifest(tmp_path, 41) is None
    assert json.loads((tmp_path / "carry_manifest_000041.json").read_text())["rollout_id"] == 41


def test_config_problems_name_the_final_runs_invariants():
    args = SimpleNamespace(partial_rollout=True, use_rollout_logprobs=True, mask_offpolicy_in_partial_rollout=False,
                           recompute_logprobs_via_prefill=False, max_weight_staleness=4,
                           over_sampling_batch_size=40, rollout_batch_size=16)
    assert ro.config_problems(args) == []
    args.use_rollout_logprobs = False
    args.mask_offpolicy_in_partial_rollout = True
    args.over_sampling_batch_size = 8
    problems = " ".join(ro.config_problems(args))
    assert "--use-rollout-logprobs" in problems and "mask-offpolicy" in problems and "over-sampling" in problems


# ---- configuration -----------------------------------------------------------------------------------------
def _value(args, flag):
    return args[args.index(flag) + 1]


def test_carry_preset_renders_the_final_runs_arguments():
    cfg = carry_solver(hf_checkpoint="h", ref_load="r", save="s", prompt_data="pool.jsonl")
    assert cfg.mode == CARRY and cfg.problems() == []
    args = cfg.train_args()
    assert _value(args, "--rollout-batch-size") == "16" and _value(args, "--n-samples-per-prompt") == "16"
    assert _value(args, "--global-batch-size") == "256" and _value(args, "--over-sampling-batch-size") == "40"
    assert "--partial-rollout" in args and "--use-rollout-logprobs" in args and "--rollout-shuffle" in args
    assert _value(args, "--max-weight-staleness") == "4" and _value(args, "--num-rollout") == "200"
    assert _value(args, "--rollout-function-path") == "autoenvscaling.training.carry.rollout.generate_rollout"
    assert _value(args, "--dynamic-sampling-filter-path").endswith("check_reward_nonzero_std")
    assert _value(args, "--custom-rollout-log-function-path").endswith("carry.rollout.log_rollout_data")
    assert _value(args, "--advantage-estimator") == "dppo" and _value(args, "--rollout-max-context-len") == "98304"
    assert "--mask-offpolicy-in-partial-rollout" not in args
    assert "needs prompt_data" in " ".join(MilesConfig(mode=CARRY, hf_checkpoint="h", ref_load="r", save="s").problems())
    assert cfg.with_(over_sampling_batch_size=8).problems()


def test_settings_from_env_and_launcher_forwarding():
    env = {"AES_CARRY_MAX_CONCURRENT_EPISODES": "32", "AES_CARRY_CARRY_ALLOWANCE": "32",
           "AES_CARRY_MAX_LIVE_SANDBOXES": "64", "AES_CARRY_PROXY_PUBLIC_URL": "https://proxy.example",
           "AES_CARRY_EPISODE_TIMEOUT_S": "600", "AES_PROXY_API_KEY": "k", "MILES_DIR": "/m", "MEGATRON_DIR": "/g"}
    settings = ro.CarrySettings.from_env(env)
    assert settings.max_concurrent_episodes == 32 and settings.episode_timeout_s == 600.0
    # sandboxes follow the repository's solver settings unless overridden
    assert (settings.sandbox_backend, settings.sandbox_cpu, settings.sandbox_memory_gib, settings.network) == \
        ("modal", 4, 8, "allowlist")
    repo = ro.CarrySettings.from_env({"AES_SANDBOX_BACKEND": "docker", "AES_SANDBOX_SOLVER_CPU": "2",
                                      "AES_SOLVER_NETWORK": "public", "AES_CARRY_SANDBOX_MEMORY_GIB": "16"})
    assert (repo.sandbox_backend, repo.sandbox_cpu, repo.sandbox_memory_gib, repo.network) == ("docker", 2, 16, "public")
    assert ro.CarrySettings.from_env({"AES_CARRY_NETWORK": "open"}).problems(16)
    gate = lc.HttpGate("http://proxy:30000/", api_key="k")
    assert gate._headers()["Authorization"] == "Bearer k" and gate.base_url == "http://proxy:30000"
    assert settings.problems(16) == [] and settings.problems(64)
    assert settings.to_env()["AES_CARRY_CARRY_ALLOWANCE"] == "32"
    assert ro.CarrySettings.from_env({}).problems(16) == [("proxy_public_url is required (sandboxes cannot reach "
                                                           "the trainer's loopback)")]
    launcher = MilesLauncher(env=env)
    cfg = carry_solver(hf_checkpoint="h", ref_load="r", save="s", prompt_data="pool.jsonl")
    forwarded = launcher.environment(cfg)["AES_FORWARD_ENV"].split()
    assert "AES_PROXY_API_KEY" in forwarded and "AES_CARRY_PROXY_PUBLIC_URL" in forwarded
    assert launcher.problems(cfg) == []
    assert any("carry:" in p for p in MilesLauncher(env={"MILES_DIR": "/m", "MEGATRON_DIR": "/g"}).problems(cfg))


def test_cli_carry_flag(capsys):
    from autoenvscaling.training.miles.__main__ import main
    os.environ.pop("AES_CARRY_PROXY_PUBLIC_URL", None)
    rc = main(["plan", "--preset", "carry-solver", "--hf-checkpoint", "h", "--ref-load", "r", "--save", "s",
               "--prompt-data", "p.jsonl"])
    out = capsys.readouterr()
    assert "--partial-rollout" in out.out and rc == 1 and "proxy_public_url" in out.err
    rc = main(["plan", "--preset", "shared-policy", "--carry", "--hf-checkpoint", "h", "--ref-load", "r", "--save", "s",
               "--prompt-data", "p.jsonl", "--set", "over_sampling_batch_size=24"])
    assert "--over-sampling-batch-size 24" in capsys.readouterr().out
