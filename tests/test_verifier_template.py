"""Run the real scoring template with real pytest, confined to temporary task directories."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

TEMPLATE = (Path(__file__).resolve().parents[1] / "skills" / "auto_env_scaling"
            / "verifier_template.sh")


def run_template(tmp_path, source, *, extra_env=None, stale=False):
    tests = tmp_path / "tests"
    logs = tmp_path / "logs"
    tests.mkdir()
    logs.mkdir()
    (tests / "test_outputs.py").write_text(source)
    if stale:
        (logs / "reward.txt").write_text("1\n")
        (logs / "ctrf.json").write_text(json.dumps(
            {"results": {"summary": {"tests": 100, "passed": 100}}}))
        (logs / "junit.xml").write_text('<testsuites><testsuite tests="100"/></testsuites>')
    env = {**os.environ, "PATH": f"{Path(sys.executable).parent}:{os.environ['PATH']}",
           "AES_VERIFIER_TESTS_DIR": str(tests), "AES_VERIFIER_LOG_DIR": str(logs),
           **(extra_env or {})}
    result = subprocess.run(["bash", str(TEMPLATE)], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=30, check=False)
    return result, logs


def test_partial_credit_comes_from_fresh_report_not_stale_success(tmp_path):
    result, logs = run_template(tmp_path, "def test_yes(): assert True\n"
                               "def test_no(): assert False\n", stale=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (logs / "reward.txt").read_text() == "0.5000\n"
    assert (logs.stat().st_mode & 0o777) == 0o700
    assert (logs / "reward.txt").stat().st_mode & 0o777 == 0o600
    if (logs / "ctrf.json").exists():
        summary = json.loads((logs / "ctrf.json").read_text())["results"]["summary"]
        assert summary["tests"] == 2 and summary["passed"] == 1


@pytest.mark.parametrize("source", ["def broken(\n", "raise RuntimeError('collection failure')\n",
                                    "# no tests\n"])
def test_collection_errors_and_empty_suites_never_reuse_or_write_reward(tmp_path, source):
    result, logs = run_template(tmp_path, source, stale=True)
    assert result.returncode != 0
    assert not (logs / "reward.txt").exists()


def test_ignores_inherited_pytest_options_plugins_and_python_path(tmp_path):
    poison = tmp_path / "poison"
    poison.mkdir()
    marker = tmp_path / "poison-was-loaded"
    (poison / "pytest.py").write_text(f"open({str(marker)!r},'w').write('bad')\n")
    (tmp_path / "pytest.ini").write_text("[pytest]\naddopts = --unknown-poison-option\n")
    result, logs = run_template(tmp_path, "def test_yes(): assert True\n", extra_env={
        "PYTEST_ADDOPTS": "--unknown-poison-option", "PYTEST_PLUGINS": "missing_poison_plugin",
        "PYTHONPATH": str(poison)})
    assert result.returncode == 0, result.stdout + result.stderr
    assert (logs / "reward.txt").read_text() == "1.0000\n"
    assert not marker.exists()


def test_failed_checks_score_zero(tmp_path):
    result, logs = run_template(tmp_path, "def test_no(): assert False\n")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (logs / "reward.txt").read_text() == "0.0000\n"


@pytest.mark.parametrize("cases", [[True, False, True], [True, True, False]])
def test_parameterized_cases_count_individually_for_reward_and_reports(tmp_path, cases):
    source = ("import pytest\n"
              f"@pytest.mark.parametrize('passes', {cases!r})\n"
              "def test_parameterized(passes): assert passes\n"
              "def test_other(): assert True\n")
    result, logs = run_template(tmp_path, source)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (logs / "reward.txt").read_text() == "0.7500\n"
    report = json.loads((logs / "ctrf.json").read_text())["results"]
    assert report["summary"]["tests"] == 4
    assert report["summary"]["passed"] == 3
    assert report["summary"]["failed"] == 1
    assert len({test["name"] for test in report["tests"]}) == 4


@pytest.mark.skipif(sys.platform != "linux" or os.geteuid() != 0 or not shutil.which("setpriv"),
                    reason="requires real Linux privilege separation")
def test_agent_cannot_forge_a_pass_in_the_full_grading_pipeline(tmp_path, tmp_path_factory):
    tmp_path_factory.getbasetemp().chmod(0o711)
    tmp_path.chmod(0o711)
    if any(not (parent.stat().st_mode & 0o001) for parent in tmp_path.parents):
        pytest.skip("use a traversable --basetemp for unprivileged execution tests")
    runner = TEMPLATE.with_name("run_agent.py")
    child_python = shutil.which("python3", path="/usr/bin:/bin")
    logs = tmp_path / "logs"
    attack = (f"from pathlib import Path; p=Path({str(logs / 'ctrf.json')!r}); "
              "p.write_text('{\"results\":{\"summary\":{\"tests\":2,\"passed\":2}}}')")
    source = f"""
import json,subprocess,sys

def agent(code):
    process = subprocess.run([
        sys.executable, '-I', {str(runner)!r}, '--runtime-dir', {str(tmp_path / 'runtime')!r},
        '--protect', {str(tmp_path / 'tests')!r}, '--protect', {str(logs)!r},
        '--', {child_python!r}, '-c', code], capture_output=True, text=True, check=True, timeout=15)
    return json.loads(process.stdout)

def test_good_program():
    result = agent("print(42)")
    assert result['returncode'] == 0 and result['stdout'].strip() == '42'

def test_attempted_report_forgery():
    result = agent({attack!r})
    assert result['returncode'] == 0  # PermissionError must fail this check, not forge a pass.
"""
    result, logs = run_template(tmp_path, source)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (logs / "reward.txt").read_text() == "0.5000\n"


def test_missing_test_directory_is_a_verifier_error(tmp_path):
    env = {**os.environ, "AES_VERIFIER_TESTS_DIR": str(tmp_path / "absent"),
           "AES_VERIFIER_LOG_DIR": str(tmp_path / "logs")}
    result = subprocess.run(["bash", str(TEMPLATE)], env=env, capture_output=True, timeout=10,
                            check=False)
    assert result.returncode != 0
    assert not (tmp_path / "logs/reward.txt").exists()
