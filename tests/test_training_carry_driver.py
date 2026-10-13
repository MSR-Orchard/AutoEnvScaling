"""The sandbox episode driver with a fake sandbox backend and a fake proxy: Pi is installed and
launched once, a park holds it alive, a resume rejoins it, the recording lands on the sample
unchanged, verification reads the Harbor reward file, and the sandbox is released exactly once."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from autoenvscaling.sandbox.base import ExecResult
from autoenvscaling.training.carry import driver as dr
from autoenvscaling.training.carry.episode import TurnStatus
from autoenvscaling.training.carry.rollout import CarrySettings


class FakeSandbox:
    def __init__(self, spec):
        self.spec = spec
        self.calls = []
        self.files = {}
        self.uploaded = []
        self.terminated = 0
        self.pi_done = asyncio.Event()
        self.pi_rc = 0
        self.reward_text = "0.75"          # what this run's test.sh writes; None writes nothing
        self.verifier_files = {}           # /logs/verifier as the container sees it

    async def exec(self, *cmd, timeout_s=None, workdir=None, env=None):
        script = cmd[-1]
        self.calls.append((cmd, timeout_s, env, workdir))
        if "command -v curl" in script:
            return ExecResult(0, "/usr/bin/curl")
        if "npm install" in script:
            return ExecResult(0, "0.80.6")
        if "models.json" in script:
            return ExecResult(0, "PLANT_OK")
        if "pi --print" in script:
            await self.pi_done.wait()
            return ExecResult(self.pi_rc, "EXIT=0")
        if cmd[0] == "python3":
            return ExecResult(0, '{"agent_ended": true, "stop_reason": "stop", "assistant_error": false, '
                                 '"content_blocks": 1, "http_status": null}')
        if script.startswith("rm -rf /tests /logs/verifier"):
            self.verifier_files.clear()
            return ExecResult(0)
        if "test.sh" in script:
            if self.reward_text is not None:
                self.verifier_files["reward.txt"] = self.reward_text
            if "reward.txt" in self.verifier_files:
                return ExecResult(0, f"RC=0\nREWARD_TXT\n{self.verifier_files['reward.txt']}\n")
            return ExecResult(0, "RC=1\nNO_REWARD\nno tests ran\n")
        if script == "true":
            return ExecResult(0)
        return ExecResult(1, "", f"unexpected: {script}")

    async def write_file(self, path, data):
        self.files[path] = data

    async def read_file(self, path):
        return self.files[path]

    async def upload_dir(self, local_dir, remote_dir):
        self.uploaded.append((Path(local_dir), remote_dir))

    async def download_dir(self, remote_dir, local_dir):
        pass

    async def terminate(self):
        self.terminated += 1


class FakeBackend:
    name = "fake"

    def __init__(self):
        self.created = []
        self.removed = []

    async def create(self, spec):
        sb = FakeSandbox(spec)
        self.created.append(sb)
        return sb

    async def remove_by_prefix(self, prefix):
        self.removed.append(prefix)
        return 2


class FakeProxy:
    def __init__(self, recording):
        self.recording_data = recording
        self.gates = []
        self.configured = []
        self.deleted = []

    def configure(self, sglang_url, served_id, max_turns, max_context):
        self.configured.append((sglang_url, served_id, max_turns, max_context))
        return {"sglang_url": sglang_url}

    def gate(self, action, episode_id):
        self.gates.append((action, episode_id))

    def recording(self, episode_id):
        return dict(self.recording_data)

    def delete(self, episode_id):
        self.deleted.append(episode_id)


def recording(n_prompt=4, n_gen=6, turns=2):
    tokens = list(range(1, n_prompt + n_gen + 1))
    return {"tokens": tokens, "logprobs": [0.0] * n_prompt + [-0.3] * n_gen,
            "loss_mask": [0] * n_prompt + [1] * n_gen, "prompt_len": n_prompt, "n_tokens": len(tokens),
            "turns": turns, "tool_results_seen": 1, "stop_reason": None, "error": None,
            "weight_versions": [3, 4]}


def make_driver(tmp_path, rec=None, **overrides):
    values = {"max_live_sandboxes": 4, "max_concurrent_episodes": 2, "carry_allowance": 2,
              "proxy_public_url": "https://proxy.example", "episode_timeout_s": 60, "run_id": "run1",
              "pi_version": "0.80.6", "sandbox_backend": "fake", "sandbox_cpu": 2, "sandbox_memory_gib": 8,
              "network": "allowlist"}
    values.update(overrides)
    settings = CarrySettings(**values)
    args = SimpleNamespace(sglang_router_ip="10.0.0.1", sglang_router_port=30001, rollout_max_context_len=1000,
                           rollout_max_response_len=100, hf_checkpoint=None)
    backend = FakeBackend()
    proxy = FakeProxy(rec or recording())
    driver = dr.SandboxEpisodeDriver(settings, args, backend=backend, proxy=proxy)
    return driver, backend, proxy


def sample(tmp_path, with_tests=True):
    task_dir = tmp_path / "task"
    if with_tests:
        (task_dir / "tests").mkdir(parents=True)
        (task_dir / "tests" / "test.sh").write_text("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")
    return SimpleNamespace(index=0, group_index=0, prompt=[{"role": "user", "content": "fix the build"}],
                           tokens=[], response_length=0, rollout_log_probs=None, loss_mask=None, reward=None,
                           metadata={"image": "ghcr.io/example/task:1", "task_dir": str(task_dir),
                                     "instruction": "fix the build"}, weight_versions=[])


async def test_full_episode_installs_pi_runs_once_verifies_and_releases(tmp_path):
    driver, backend, proxy = make_driver(tmp_path)
    s = sample(tmp_path)
    sb = await driver.create_sandbox(s)
    fake = backend.created[0]
    assert fake.spec.image == "ghcr.io/example/task:1" and fake.spec.name == f"run1-{sb.episode_id}"
    assert proxy.configured == [("http://10.0.0.1:30001", "policy", 128, 1000)]
    install = next(c[0][-1] for c in fake.calls if "npm install" in c[0][-1])
    assert "npm install -g --ignore-scripts @earendil-works/pi-coding-agent@0.80.6" in install   # Harbor's package
    assert f"/v1/ep/{sb.episode_id}" in fake.files[f"{dr.PI_CONFIG_DIR}/models.json"].decode()   # per-episode route
    assert fake.spec.cpu == 2 and fake.spec.memory_gib == 8 and fake.spec.workdir == "/app"
    assert "proxy.example" in fake.spec.domain_allowlist and "registry.npmjs.org" in fake.spec.domain_allowlist
    assert driver.pending_cleanup_count() == 1

    run = asyncio.create_task(driver.run_turns(sample=s, sandbox=sb, turn_state=None, sampling_params={},
                                               should_abort=lambda: False, start_turn_index=0))
    await asyncio.sleep(0.02)
    assert not run.done() and fake.files[f"/tmp/aes_pi_prompt_{sb.episode_id}.txt"] == b"fix the build"
    launch = next(c for c in fake.calls if "pi --print" in c[0][-1])
    assert launch[2] == {"OPENAI_API_KEY": "proxy", "PI_CODING_AGENT_DIR": dr.PI_CONFIG_DIR}
    assert "fix the build" not in launch[0][-1]
    fake.pi_done.set()
    outcome = await asyncio.wait_for(run, 2)
    assert outcome.status is TurnStatus.DONE and outcome.turn_index == 2
    assert s.tokens == list(range(1, 11)) and s.response_length == 6
    assert s.rollout_log_probs == [-0.3] * 6 and s.loss_mask == [1] * 6 and s.weight_versions == ["3", "4"]
    assert s.metadata["pi_terminal_status"]["agent_ended"] is True

    reward = await driver.verify(sample=s, sandbox=sb, turn_state=None)
    assert reward == 0.75 and fake.uploaded == [(tmp_path / "task" / "tests", "/tests")]
    reset = next(i for i, c in enumerate(fake.calls) if c[0][-1].startswith("rm -rf /tests /logs/verifier"))
    run = next(i for i, c in enumerate(fake.calls) if "test.sh" in c[0][-1])
    assert reset < run and fake.calls[run][3] == "/app"                  # fresh dirs, task workdir
    await driver.wait_cleanup(2)
    assert fake.terminated == 1 and proxy.deleted == [sb.episode_id] and driver.pending_cleanup_count() == 0
    assert sum(1 for c in fake.calls if "pi --print" in c[0][-1]) == 1


async def test_park_keeps_pi_alive_and_resume_rejoins_it(tmp_path, monkeypatch):
    monkeypatch.setattr(dr, "ABORT_POLL_S", 0.01)
    driver, backend, proxy = make_driver(tmp_path, rec=recording(n_gen=3, turns=1))
    s = sample(tmp_path)
    sb = await driver.create_sandbox(s)
    fake = backend.created[0]
    flag = {"abort": False}
    run = asyncio.create_task(driver.run_turns(sample=s, sandbox=sb, turn_state=None, sampling_params={},
                                               should_abort=lambda: flag["abort"], start_turn_index=0))
    await asyncio.sleep(0.02)
    flag["abort"] = True
    outcome = await asyncio.wait_for(run, 2)
    assert outcome.status is TurnStatus.ABORTED and outcome.turn_index == 1
    assert proxy.gates == [("park", sb.episode_id)] and fake.terminated == 0
    assert sb.waiter is not None and not sb.waiter.done()               # Pi is still running
    assert s.tokens == list(range(1, 8)) and s.response_length == 3     # the partial recording

    # the registry would pause the clock at park and resume it later
    sb.pause_execution()
    sb.resume_execution()
    proxy.recording_data = recording(n_gen=6, turns=2)
    flag["abort"] = False
    fake.pi_done.set()
    outcome = await asyncio.wait_for(driver.run_turns(sample=s, sandbox=sb, turn_state=None, sampling_params={},
                                                      should_abort=lambda: False, start_turn_index=1), 2)
    assert outcome.status is TurnStatus.DONE and proxy.gates[-1] == ("unpark", sb.episode_id)
    assert s.tokens == list(range(1, 11)) and s.rollout_log_probs == [-0.3] * 6
    assert sum(1 for c in fake.calls if "pi --print" in c[0][-1]) == 1  # never launched twice
    assert await asyncio.to_thread(sb.heartbeat) is True
    assert sb.heartbeat() is None                                       # never blocks the loop thread
    sb.close()
    await driver.wait_cleanup(2)
    assert fake.terminated == 1 and await asyncio.to_thread(sb.heartbeat) is None   # cleanup owns it


async def test_failed_pi_and_missing_recording_are_failures_not_rewards(tmp_path):
    driver, backend, proxy = make_driver(tmp_path, rec={"n_tokens": 0, "turns": 0})
    s = sample(tmp_path)
    sb = await driver.create_sandbox(s)
    fake = backend.created[0]
    fake.pi_done.set()
    outcome = await driver.run_turns(sample=s, sandbox=sb, turn_state=None, sampling_params={},
                                     should_abort=lambda: False, start_turn_index=0)
    assert outcome.status is TurnStatus.FAILED and "no tokens" in outcome.error
    fake2 = FakeSandbox(fake.spec)
    fake2.pi_rc = 137
    fake2.pi_done.set()
    sb2 = dr.CarrySandbox(fake2, "ep-x", loop=asyncio.get_running_loop(), release=lambda: None,
                          proxy=proxy, max_paused_s=10)
    with pytest.raises(RuntimeError, match="Pi process failed"):
        await driver.run_turns(sample=s, sandbox=sb2, turn_state=None, sampling_params={},
                               should_abort=lambda: False, start_turn_index=0)
    with pytest.raises(ValueError, match="no image"):
        await driver.create_sandbox(SimpleNamespace(metadata={}, prompt=""))


async def test_reap_orphans_and_setup_failure_release_the_slot(tmp_path):
    driver, backend, _proxy = make_driver(tmp_path)
    assert await driver.reap_orphans() == 2 and backend.removed == ["run1-"]

    class BrokenInstall(FakeSandbox):
        async def exec(self, *cmd, timeout_s=None, workdir=None, env=None):
            if "npm install" in cmd[-1]:
                return ExecResult(1, "", "npm ERR! network")
            return await super().exec(*cmd, timeout_s=timeout_s, workdir=workdir, env=env)

    async def create(spec):
        sb = BrokenInstall(spec)
        backend.created.append(sb)
        return sb

    backend.create = create
    with pytest.raises(RuntimeError, match="pi install failed"):
        await driver.create_sandbox(sample(tmp_path))
    await driver.wait_cleanup(2)
    assert backend.created[-1].terminated == 1 and driver.pending_cleanup_count() == 0
    assert driver._live._value == 4                                     # the admission slot came back


async def test_stale_or_missing_reward_file_never_scores(tmp_path):
    driver, backend, _proxy = make_driver(tmp_path)
    s = sample(tmp_path)
    sb = await driver.create_sandbox(s)
    fake = backend.created[0]
    fake.verifier_files["reward.txt"] = "1"        # planted by the agent or left by an earlier run
    fake.reward_text = None                         # this run's tests write nothing
    with pytest.raises(RuntimeError, match="no reward"):
        await driver.verify(sample=s, sandbox=sb, turn_state=None)
    await driver.wait_cleanup(2)
    assert fake.terminated == 1                     # the sandbox is still released


async def test_network_and_pi_package_settings(tmp_path):
    (tmp_path / "task").mkdir()
    (tmp_path / "task" / "task.toml").write_text('[environment]\nnetwork_mode = "allowlist"\n'
                                                 'allowed_hosts = ["pypi.org"]\n[agent]\nuser = "solver"\n')
    driver, backend, _proxy = make_driver(tmp_path)
    sb = await driver.create_sandbox(sample(tmp_path))
    assert "pypi.org" in backend.created[-1].spec.domain_allowlist
    sb.close()
    driver_public, backend_public, _ = make_driver(tmp_path, network="public")
    await driver_public.create_sandbox(sample(tmp_path, with_tests=False))
    assert backend_public.created[-1].spec.domain_allowlist is None
    assert dr.pi_package_spec("0.80.6") == "@earendil-works/pi-coding-agent@0.80.6"
    assert dr.pi_package_spec("0.73.1") == "@mariozechner/pi-coding-agent@0.73.1"
    assert dr.pi_package_spec("0.80.6", "my/pi") == "my/pi@0.80.6"
    assert dr.task_allowed_hosts(str(tmp_path / "missing")) == []


def test_apply_recording_and_reward_parsing():
    s = SimpleNamespace(tokens=[], rollout_log_probs=None, loss_mask=None, response_length=0, weight_versions=[])
    dr.apply_recording(s, recording(), max_context=100)
    assert s.response_length == 6 and len(s.rollout_log_probs) == 6 and s.loss_mask == [1] * 6
    with pytest.raises(RuntimeError, match="exceeds"):
        dr.apply_recording(s, recording(), max_context=5)
    bad = recording()
    bad["logprobs"] = bad["logprobs"][:-1]
    with pytest.raises(RuntimeError, match="inconsistent"):
        dr.apply_recording(s, bad)
    with pytest.raises(RuntimeError, match="prompt_len"):
        dr.apply_recording(s, {**recording(), "prompt_len": 0})
    assert dr.parse_reward_output("RC=0\nREWARD_TXT\n0.5\n") == 0.5
    assert dr.parse_reward_output("RC=1\nREWARD_JSON\n{\"reward\": 0.25}\n") == 0.25
    assert dr.parse_reward_output("RC=1\nREWARD_JSON\n{\"t1\": 1, \"t2\": 0}\n") == 0.5
    with pytest.raises(RuntimeError, match="no reward"):
        dr.parse_reward_output("RC=1\nNO_REWARD\nImportError: pytest\n")
    with pytest.raises(RuntimeError, match="no reward marker"):
        dr.parse_reward_output("RC=0\n1\n")                    # a bare number is not this run's file
    with pytest.raises(RuntimeError):
        dr.parse_reward_output("RC=0\nREWARD_TXT\nnan\n")
