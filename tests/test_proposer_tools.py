"""In-sandbox tools: aes-validate, web_search, web_fetch and solver_model (all without network)."""

import importlib.machinery
import importlib.util
import json
import socket
import sys
from pathlib import Path

import pytest

from autoenvscaling.proposer import solver_tool
from autoenvscaling.trajectories.schema import SolverTrajectory, Turn

TOOLS = Path(__file__).resolve().parents[1] / "docker" / "proposer" / "tools"


def load_tool(name: str):
    """Import an extensionless tool script without leaving bytecode next to it."""
    loader = importlib.machinery.SourceFileLoader(name.replace("-", "_"), str(TOOLS / name))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    previous, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def make_task(root: Path) -> Path:
    for rel in ("task.toml", "instruction.md", "environment/Dockerfile", "solution/solve.sh",
                "tests/test.sh"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("x")
    return root


# ---- aes-validate -------------------------------------------------------------------------

def write_job(job_dir: Path, reward: float, verifier: str = "1 passed"):
    trial = job_dir / "trial-0"
    (trial / "verifier").mkdir(parents=True)
    (trial / "verifier" / "test-stdout.txt").write_text(verifier)
    (job_dir / "result.json").write_text(json.dumps({"stats": {"evals": {"oracle__adhoc": {
        "reward_stats": {"reward": {str(reward): ["trial-0"]}}, "exception_stats": {}}}}}))


def test_aes_validate_reads_rewards_and_judges(tmp_path):
    tool = load_tool("aes-validate")
    write_job(tmp_path / "job", 1.0)
    assert tool.read_rewards(tmp_path / "job") == ([1.0], [])
    # Fallback to a trial's reward.txt when there is no job result.
    trial = tmp_path / "job2" / "t" / "verifier"
    trial.mkdir(parents=True)
    (trial / "reward.txt").write_text("0.25\n")
    assert tool.read_rewards(tmp_path / "job2")[0] == [0.25]
    assert tool.judge("reference", [1.0], 1.0, 0.5) == "pass"
    assert tool.judge("reference", [0.9], 1.0, 0.5) == "fail"
    assert tool.judge("empty", [0.5], 1.0, 0.5) == "fail"
    assert tool.judge("empty", [], 1.0, 0.5) == "no_reward"
    cmd = tool.harbor_command(Path("/t"), "nop", "modal", Path("/j"), "empty-1")
    assert cmd[:8] == ["harbor", "run", "-p", "/t", "-a", "nop", "-e", "modal"]


def test_aes_validate_end_to_end_with_a_fake_harbor(tmp_path, monkeypatch, capsys):
    tool = load_tool("aes-validate")
    task = make_task(tmp_path / "task")
    rewards = {"oracle": 1.0, "nop": 0.75}  # the empty solution scores too high

    def fake_run(cmd, capture_output, text, timeout, check):
        jobs, name = Path(cmd[cmd.index("-o") + 1]), cmd[cmd.index("--job-name") + 1]
        write_job(jobs / name, rewards[cmd[cmd.index("-a") + 1]], "FAILED test_x")

        class P:
            returncode, stdout, stderr = 0, "", ""
        return P()

    monkeypatch.setattr(tool.subprocess, "run", fake_run)
    out = tmp_path / "report.json"
    code = tool.main([str(task), "--jobs-dir", str(tmp_path / "jobs"), "--out", str(out)])
    printed = capsys.readouterr().out
    assert code == 1
    assert "reference: reward=1 (want >= 1) -> pass" in printed
    assert "empty: reward=0.75 (want < 0.5) -> fail" in printed
    report = json.loads(out.read_text())
    assert report["checks"][1]["trials"][0]["verifier_tail"] == "FAILED test_x"

    assert tool.main([str(tmp_path / "nope")]) == 2


def test_aes_validate_report_stays_out_of_the_task(tmp_path, monkeypatch):
    tool = load_tool("aes-validate")
    task = make_task(tmp_path / "task")

    def fake_run(cmd, capture_output, text, timeout, check):
        jobs, name = Path(cmd[cmd.index("-o") + 1]), cmd[cmd.index("--job-name") + 1]
        write_job(jobs / name, 1.0 if cmd[cmd.index("-a") + 1] == "oracle" else 0.0, "")

        class P:
            returncode, stdout, stderr = 0, "", ""
        return P()

    monkeypatch.setattr(tool.subprocess, "run", fake_run)
    monkeypatch.setenv("AES_LOGS_DIR", str(tmp_path / "workspace-logs"))
    monkeypatch.chdir(task)  # the proposer often runs it from inside the task
    assert tool.main([".", "--jobs-dir", str(tmp_path / "jobs")]) == 0
    assert not (task / "logs").exists()
    assert len(list((tmp_path / "workspace-logs" / "aes-validate").glob("task-*.json"))) == 1


# ---- web tools ----------------------------------------------------------------------------

def test_web_search_filters_blocked_results(monkeypatch):
    tool = load_tool("web_search")
    monkeypatch.setenv("AES_BLOCKED_HOSTS", "tbench.ai")
    monkeypatch.delenv("AES_BLOCKED_URL_PATTERNS", raising=False)
    data = {"organic": [
        {"title": "ok", "link": "https://docs.python.org/3/", "snippet": "s"},
        {"title": "blocked host", "link": "https://www.tbench.ai/tasks", "snippet": "s"},
        {"title": "blocked repo", "link": "https://github.com/org/terminal-bench/x", "snippet": "s"},
    ]}
    filtered = tool.filter_results(data)
    assert [h["title"] for h in filtered["organic"]] == ["ok"] and filtered["_dropped"] == 2
    assert "removed" in tool.format_results(filtered)
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    assert tool.main(["query"]) == 2


def test_web_fetch_refuses_blocked_and_private_targets(monkeypatch):
    tool = load_tool("web_fetch")
    monkeypatch.setenv("AES_BLOCKED_HOSTS", "tbench.ai")
    addresses = {"docs.example": "93.184.216.34", "intranet.example": "10.0.0.5"}
    monkeypatch.setattr(tool.socket, "getaddrinfo",
                        lambda host, port: [(socket.AF_INET, None, None, "",
                                             (addresses[host], 0))])
    assert tool.refusal("https://docs.example/page") is None
    assert "blocked" in tool.refusal("https://sub.tbench.ai/x")
    assert "benchmark" in tool.refusal("https://docs.example/terminal-bench/task")
    assert "non-public" in tool.refusal("http://intranet.example/")
    assert "http" in tool.refusal("file:///etc/passwd")
    text = tool.html_to_text("<html><script>x()</script><p>Hello &amp; bye</p><br>Next</html>")
    assert [line.strip() for line in text.splitlines() if line.strip()] == ["Hello & bye", "Next"]


# ---- solver_model -------------------------------------------------------------------------

def test_solver_model_reports_the_calibration_verdict(tmp_path, monkeypatch, capsys):
    task = make_task(tmp_path / "my-task")
    seen = {}

    async def fake_run_solver(task_arg, endpoint, *, n, harbor_env, hints=None):
        seen.update(n=n, harbor_env=harbor_env, model=endpoint.model, task=task_arg.task_id)
        return [SolverTrajectory(trajectory_id=f"a{i}", task_id="my-task", model="solver",
                                 turns=[Turn("assistant", "try")], tests_passed=p, tests_total=4,
                                 reward=p / 4, verifier_output="FAILED t" * (p < 4))
                for i, p in enumerate([0, 2, 4, 2])]

    monkeypatch.setattr(solver_tool, "run_solver", fake_run_solver)
    monkeypatch.setenv("AES_SOLVER_BASE_URL", "http://solver.example/v1")
    monkeypatch.setenv("AES_SOLVER_MODEL", "solver-9b")
    monkeypatch.setenv("AES_SANDBOX_HARBOR_ENV", "docker")
    out = tmp_path / "report.json"
    assert solver_tool.main([str(task), "--out", str(out)]) == 0
    compact = json.loads(capsys.readouterr().out)
    assert seen == {"n": 4, "harbor_env": "docker", "model": "solver-9b", "task": "my-task"}
    assert compact["rewards"] == [0.0, 0.5, 1.0, 0.5] and compact["verdict"] == "admitted"
    report = json.loads(out.read_text())
    assert report["attempts"][0]["turns"][0]["content"] == "try"


def test_solver_model_configuration_errors(tmp_path, monkeypatch):
    monkeypatch.delenv("AES_SOLVER_BASE_URL", raising=False)
    assert solver_tool.main([str(make_task(tmp_path / "t"))]) == 2
    assert solver_tool.main([str(tmp_path / "missing")]) == 2
    with pytest.raises(ValueError):
        solver_tool.endpoint_from_env({})
