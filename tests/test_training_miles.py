import asyncio
import subprocess
import threading
from types import SimpleNamespace

import pytest

from autoenvscaling.selfplay.schedule import Schedule
from autoenvscaling.training.miles import bridge
from autoenvscaling.training.miles import launcher as launcher_mod
from autoenvscaling.training.miles.__main__ import main as miles_main
from autoenvscaling.training.miles.config import (
    BRIDGE_ROLLOUT_FN,
    GENERATE_FN,
    REWARD_HOOK,
    SFT,
    MilesConfig,
    cold_start_sft,
    exp2_solver,
    shared_policy,
)
from autoenvscaling.training.miles.hooks import center_rewards_by_group
from autoenvscaling.training.miles.launcher import ConfigError, MilesLauncher, latest_checkpoint
from autoenvscaling.training.miles.rollout import batch_metrics, encode_rows, group_encoded
from autoenvscaling.trajectories.export import RLGroup, RLSample, rl_sample_rows

PATHS = {"hf_checkpoint": "ckpt/hf", "ref_load": "ckpt/torch_dist", "save": "out/save"}


def _value(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


# ---- configuration -------------------------------------------------------------------------
def test_defaults_are_the_paper_settings():
    c = shared_policy(**PATHS)
    args = c.train_args()
    assert _value(args, "--lr") == "1e-06" and _value(args, "--lr-decay-style") == "constant"
    assert _value(args, "--advantage-estimator") == "dppo"
    assert _value(args, "--dppo-divergence-type") == "tv"
    assert _value(args, "--dppo-divergence-threshold") == "0.1"
    assert _value(args, "--kl-loss-coef") == "0.0" and _value(args, "--entropy-coef") == "0.0"
    assert _value(args, "--rollout-batch-size") == "16" and _value(args, "--n-samples-per-prompt") == "16"
    assert _value(args, "--global-batch-size") == "256"
    assert int(_value(args, "--num-rollout")) == Schedule(288).planned_updates
    assert _value(args, "--custom-reward-post-process-path") == REWARD_HOOK
    assert c.problems() == []


def test_bridge_mode_arguments():
    args = shared_policy(**PATHS).train_args()
    assert _value(args, "--rollout-function-path") == BRIDGE_ROLLOUT_FN
    for flag in ("--disable-rollout-global-dataset", "--use-dynamic-global-batch-size", "--colocate"):
        assert flag in args
    assert "--prompt-data" not in args and "--use-rollout-logprobs" not in args
    assert _value(args, "--sglang-served-model-name") == "policy"


def test_generate_fn_mode_and_problems():
    c = MilesConfig(mode=GENERATE_FN, **PATHS)
    assert "prompt_data" in " ".join(c.problems())
    c = c.with_(prompt_data="pool.jsonl", generate_function_path="my.module.generate")
    args = c.train_args()
    assert _value(args, "--prompt-data") == "pool.jsonl"
    assert _value(args, "--custom-generate-function-path") == "my.module.generate"
    assert _value(args, "--input-key") == "prompt" and "--apply-chat-template" in args
    assert "check_reward_nonzero_std" in _value(args, "--dynamic-sampling-filter-path")
    assert c.problems() == []
    assert "does not divide" in " ".join(c.with_(tensor_parallel=3).problems())
    assert MilesConfig().problems()   # paths are never defaulted


def test_exp2_and_cold_start_presets():
    small = exp2_solver("qwen3.5-9b", **PATHS)
    args = small.train_args()
    assert small.model_args_script.endswith("qwen3.5-9B.sh")
    assert "--expert-model-parallel-size" not in args
    assert _value(args, "--seq-length") == "131072"
    big = exp2_solver("qwen3.5-35b-a3b", **PATHS)
    assert "--expert-model-parallel-size" in big.train_args()
    with pytest.raises(ValueError):
        exp2_solver("gpt")

    sft = cold_start_sft(hf_checkpoint="hf", save="out", prompt_data="sft.jsonl")
    args = sft.train_args()
    assert sft.mode == SFT and sft.problems() == []
    assert _value(args, "--input-key") == "messages" and _value(args, "--tool-key") == "tools"
    assert "--loss-type" in args and "--rollout-num-gpus-per-engine" not in args


# ---- launcher ------------------------------------------------------------------------------
def test_plan_and_environment():
    launcher = MilesLauncher(miles_dir="miles", megatron_dir="megatron", bridge_dir="ctl", env={})
    config = shared_policy(**PATHS)
    env = launcher.environment(config)
    assert env["AES_BRIDGE_DIR"] == "ctl" and env["MODEL_ARGS_SCRIPT"].endswith("qwen3.6-35B-A3B.sh")
    plan = launcher.plan(config)
    assert plan.startswith("MILES_DIR=miles MEGATRON_DIR=megatron") and "train.sh" in plan
    assert "apptainer.sh" in MilesLauncher(image="img.sif", bridge_dir="c", env={}).plan(config)
    assert MilesLauncher(env={}).problems(config) == [
        "MILES_DIR is not set", "MEGATRON_DIR is not set", "bridge mode needs bridge_dir"]
    with pytest.raises(ConfigError):
        MilesLauncher(env={}).environment(config)


def test_latest_checkpoint(tmp_path):
    assert latest_checkpoint(tmp_path) is None
    (tmp_path / "iter_0000007").mkdir()
    (tmp_path / "iter_0000012").mkdir()
    assert latest_checkpoint(tmp_path) == (12, tmp_path / "iter_0000012")
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("7\n")
    assert latest_checkpoint(tmp_path) == (7, tmp_path / "iter_0000007")


FAKE_TRAIN = """#!/usr/bin/env bash
save=""
while [ $# -gt 0 ]; do
  if [ "$1" = "--save" ]; then save="$2"; fi
  shift
done
if [ "${FAKE_MODE}" != "no_ckpt" ]; then
  mkdir -p "$save/iter_0000003"; echo 3 > "$save/latest_checkpointed_iteration.txt"
fi
[ "${FAKE_MODE}" = "fail" ] && exit 1
exit 0
"""


@pytest.mark.parametrize("mode,ok", [("ok", True), ("no_ckpt", False), ("fail", False)])
def test_run_requires_zero_exit_and_a_checkpoint(tmp_path, monkeypatch, mode, ok):
    launch = tmp_path / "launch"
    launch.mkdir()
    (launch / "train.sh").write_text(FAKE_TRAIN)
    monkeypatch.setattr(launcher_mod, "LAUNCH_DIR", launch)
    config = shared_policy(hf_checkpoint="hf", ref_load="ref", save=str(tmp_path / "save"))
    ctl = tmp_path / "ctl"
    result = MilesLauncher(miles_dir="m", megatron_dir="g", bridge_dir=ctl,
                           env={"FAKE_MODE": mode, "PATH": "/usr/bin:/bin"}).run(config)
    assert result.ok is ok
    if mode == "fail":
        # The job also ends with an error when the self-play loop stops early; that is success.
        ctl.mkdir()
        (ctl / bridge.STOP_FILE).write_text("{}")
        again = MilesLauncher(miles_dir="m", megatron_dir="g", bridge_dir=ctl,
                              env={"FAKE_MODE": mode, "PATH": "/usr/bin:/bin"}).run(config)
        assert again.ok and again.detail["stopped_by_loop"]


def test_launch_scripts_parse():
    for script in ("train.sh", "apptainer.sh"):
        subprocess.run(["bash", "-n", str(launcher_mod.LAUNCH_DIR / script)], check=True)


def test_cli_plan(capsys):
    rc = miles_main(["plan", "--preset", "exp2-qwen3.5-9b", "--hf-checkpoint", "hf", "--ref-load", "r",
                     "--save", "s", "--bridge-dir", "ctl", "--set", "rollout_batch_size=8",
                     "--num-rollout", "25"])
    out = capsys.readouterr()
    assert "--rollout-batch-size 8" in out.out and "--num-rollout 25" in out.out
    assert rc in (0, 1)       # 1 when MILES_DIR/MEGATRON_DIR are unset in this environment
    assert miles_main(["presets"]) == 0


# ---- bridge --------------------------------------------------------------------------------
def _groups(role="solver") -> list[RLGroup]:
    msgs = [{"role": "user", "content": "do"}, {"role": "assistant", "content": "done"}]
    return [RLGroup("env1", role, [1.0, 0.0], [RLSample("t1", 1.0, 0.5, msgs), RLSample("t2", 0.0, -0.5, msgs)])]


async def test_bridge_handshake(tmp_path):
    trainer = bridge.MilesBridgeTrainer(tmp_path, model="policy", poll_s=0.01, timeout_s=5)
    received = {}

    def miles_side():
        bridge.write_ready(tmp_path, 7, "http://router.invalid:3000")
        received[7] = bridge.wait_for_batch(tmp_path, 7, poll_s=0.01, timeout_s=5)
        bridge.write_ready(tmp_path, 8, "http://router.invalid:3000")
        received[8] = bridge.wait_for_batch(tmp_path, 8, poll_s=0.01, timeout_s=5)

    thread = threading.Thread(target=miles_side)
    thread.start()
    handle = await trainer.acquire()
    assert handle.update_index == 7
    assert handle.endpoint.base_url == "http://router.invalid:3000/v1" and handle.endpoint.model == "policy"
    assert await trainer.acquire() == handle         # idempotent until submitted
    with pytest.raises(ValueError):
        await trainer.submit(handle, _groups(), role="proposer")
    await trainer.submit(handle, _groups(), role="solver")
    with pytest.raises(RuntimeError):
        await trainer.submit(handle, _groups(), role="solver")
    handle = await trainer.acquire()
    assert handle.update_index == 8
    await trainer.close()
    await asyncio.to_thread(thread.join, 5)
    assert [r["trajectory_id"] for r in received[7]] == ["t1", "t2"]
    assert received[8] is None                         # stopped


async def test_bridge_public_url_and_empty_batch(tmp_path):
    trainer = bridge.MilesBridgeTrainer(tmp_path, public_base_url="https://tunnel.example/v1",
                                        poll_s=0.01, timeout_s=5)
    bridge.write_ready(tmp_path, 0, "http://router.invalid:3000")
    handle = await trainer.acquire()
    assert handle.endpoint.base_url == "https://tunnel.example/v1"
    with pytest.raises(ValueError):
        await trainer.submit(handle, [], role="solver")
    assert bridge.pending_ready(tmp_path) == [0]


# ---- Miles-side conversion and reward hook --------------------------------------------------
def fake_encode(messages, tools):
    tokens, mask = [], []
    for m in messages:
        n = len(m["content"].split()) + 1
        tokens += [len(tokens) + i for i in range(n)]
        mask += [1 if m["role"] == "assistant" else 0] * n
    return tokens, mask


def test_encode_rows_masks_prompt_and_truncates():
    groups = _groups()
    groups[0].samples[1].messages = [{"role": "user", "content": "a b c d e f"},
                                     {"role": "assistant", "content": "x y z"}]
    rows = rl_sample_rows(groups)
    encoded = encode_rows(rows, fake_encode, max_tokens=8)
    first, second = encoded
    assert first["tokens"] == [0, 1, 2, 3] and first["response_length"] == 2
    assert first["loss_mask"] == [1, 1] and not first["remove_sample"]
    assert first["prompt"] == [{"role": "user", "content": "do"}]
    assert first["metadata"]["advantage"] == 0.5 and first["metadata"]["role"] == "solver"
    assert len(second["tokens"]) == 8 and second["remove_sample"] and second["metadata"]["truncated"]
    assert second["response_length"] == 1 and len(second["loss_mask"]) == 1
    assert [len(g) for g in group_encoded(encoded)] == [2]
    assert batch_metrics(rows, encoded)["aes/removed_samples"] == 1.0


def test_center_rewards_by_group():
    s = lambda g, r, adv=None: SimpleNamespace(group_index=g, reward=r,
                                               metadata={} if adv is None else {"advantage": adv})
    args = SimpleNamespace(reward_key=None)
    raw, centred = center_rewards_by_group(args, [s(0, 1.0), s(0, 0.0), s(1, 0.5), s(1, 0.25)])
    assert raw == [1.0, 0.0, 0.5, 0.25] and centred == [0.5, -0.5, 0.125, -0.125]
    raw, adv = center_rewards_by_group(args, [s(0, 1.0, 0.9), s(0, -1.0, -1.1)])
    assert adv == [0.9, -1.1]
    keyed = SimpleNamespace(group_index=0, reward={"score": 2.0}, metadata={})
    assert center_rewards_by_group(SimpleNamespace(reward_key="score"), [keyed]) == ([2.0], [0.0])
