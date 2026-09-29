import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

from autoenvscaling import cli
from autoenvscaling.config import Settings
from autoenvscaling.evals.hilbench.__main__ import main as hil_main

REPO = Path(__file__).resolve().parents[1]
EXPERIMENTS = ["evals", "exp1_authoring", "exp2_solver_rl", "exp3_self_play", "cold_start",
               "cross_domain", "hilbench"]
LAUNCH_ARGS = {"evals": ["some/model", "both"], "exp2_solver_rl": ["qwen3.5-35b-a3b"]}
WITH_SUBCOMMANDS = {"selfplay", "train", "export", "baselines", "eval", "hilbench"}


def test_every_experiment_has_readme_and_launcher():
    for name in EXPERIMENTS:
        assert (REPO / "cmd" / name / "README.md").is_file(), name
        assert (REPO / "cmd" / name / "run.sh").is_file(), name


@pytest.mark.parametrize("name", [n for n in EXPERIMENTS if (REPO / "cmd" / n / "config.toml").exists()])
def test_configs_load_as_package_settings(name):
    settings = Settings.load(REPO / "cmd" / name / "config.toml", env={})
    assert str(settings.work_dir).startswith("runs/")
    # The paper's resources are never overridden by an experiment config.
    assert (settings.sandbox.proposer_cpu, settings.sandbox.proposer_memory_gib) == (16, 32)
    assert (settings.sandbox.solver_cpu, settings.sandbox.solver_memory_gib) == (4, 8)


def _dry_run(name, tmp_path):
    trace = tmp_path / "trace.txt"
    env = {**os.environ, "DRY_RUN": "1", "AES_RUN_DIR": str(tmp_path / "run"),
           "AES_TRACE": str(trace)}
    result = subprocess.run(["bash", str(REPO / "cmd" / name / "run.sh"), *LAUNCH_ARGS.get(name, [])],
                            env=env, capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "run" / "launcher.log").is_file()
    return result.stdout, [shlex.split(line) for line in trace.read_text().splitlines()]


class _Parsed(Exception):
    pass


def _command_path(argv):
    """The command and, for commands with subcommands, the subcommand (skipping --config X)."""
    path, rest = [argv[0]], argv[1:]
    if argv[0] in WITH_SUBCOMMANDS:
        i = 0
        while rest[i].startswith("-"):
            i += 2 if rest[i] == "--config" else 1
        path.append(rest[i])
    return path


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_launchers_call_only_existing_commands(name, tmp_path, monkeypatch, capsys):
    _, calls = _dry_run(name, tmp_path)
    assert calls
    original = argparse.ArgumentParser.parse_args

    def parse_only(self, args=None, namespace=None):
        raise _Parsed(original(self, args, namespace))

    for argv in calls:
        with pytest.raises(SystemExit) as help_exit:     # the command exists and has --help
            cli.main([*_command_path(argv), "--help"])
        assert help_exit.value.code == 0, argv
        monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse_only)
        with pytest.raises(_Parsed):                     # and accepts these exact arguments
            cli.main(argv)
        monkeypatch.setattr(argparse.ArgumentParser, "parse_args", original)
    capsys.readouterr()


def test_launchers_mark_missing_steps(tmp_path):
    expected = {"exp1_authoring": 2, "exp3_self_play": 1, "cross_domain": 7, "hilbench": 3}
    for name, count in expected.items():
        stdout, _ = _dry_run(name, tmp_path / name)
        assert stdout.count("not in this release yet") == count, name


def test_hilbench_cli(tmp_path, capsys):
    assert hil_main(["prompt", "--index", "1"]) == 0
    assert "Author this task in the following domain" in capsys.readouterr().out
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text(json.dumps({"task": "swe_1", "task_type": "swe", "mode": "ask_human",
                                    "sample": 0, "reward": 1.0, "stop_reason": "completed",
                                    "verify": {"n_tests_reported": 3}}) + "\n")
    assert hil_main(["report", str(episodes), "--model", "m"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["m"]["all.ask_human"]["success_rate"] == 1.0
    assert report["m"]["all.ask_human"]["ask_rate"] == 0.0
    assert hil_main(["derive", str(tmp_path / "missing")]) == 1
