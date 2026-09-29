import json
import subprocess
import sys
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

import pytest
from harbor.models.agent.context import AgentContext

from autoenvscaling.evals.hilbench import ask_human, generation, metrics
from autoenvscaling.evals.hilbench import ask_human_client as client
from autoenvscaling.evals.hilbench.pi_agent import HilPiAgent
from autoenvscaling.evals.hilbench.registry import (
    Blocker,
    blocker_warnings,
    leak_check,
    validate_blockers,
    validate_task,
    write_registry,
)
from autoenvscaling.evals.hilbench.variants import blocker_details, derive_variants


def _blocker(i, **overrides):
    data = {"id": f"b{i}", "type": "missing parameters",
            "description": f"Which threshold {i} applies.",
            "resolution": f"Use a threshold of {10 + i} units.",
            "example_questions": ["What threshold?", "Which limit?", "How strict?"]}
    data.update(overrides)
    return Blocker(**data)


def _task(tmp_path, blockers, instruction="Reconcile the two exports and write reports.\n"):
    task = tmp_path / "env_001"
    for sub in ("environment", "solution", "tests"):
        (task / sub).mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    (task / "solution" / "solve.sh").write_text("#!/bin/sh\n")
    (task / "tests" / "test.sh").write_text("#!/bin/sh\n")
    (task / "task.toml").write_text('version = "1.0"\n')
    (task / "instruction.md").write_text(instruction)
    write_registry(task / "ask-human-data" / "blocker_registry.json", blockers)
    return task


def test_registry_structure_checks():
    good = [_blocker(i) for i in range(3)]
    assert validate_blockers(good) == []
    errors = validate_blockers(good[:2] + [_blocker(0, example_questions=["one"])])
    assert any("2 blockers" not in e for e in errors)
    assert any("duplicate blocker id" in e for e in errors)
    assert any("fewer than 3 example_questions" in e for e in errors)
    leaky = _blocker(9, description="Amounts agree when the difference is at most 0.02 either way",
                     resolution="Amounts agree when the difference is at most 0.02 either way.")
    assert any("leaks its resolution" in e for e in validate_blockers(good + [leaky]))


def test_leak_check_only_flags_resolutions_without_new_values():
    instruction = ("Sort rows by total_billable_volume. Keep 401 for a missing token and 400 "
                   "for bad input.")
    restating = Blocker("codes", "question", "which code for duplicates",
                        "Return 409 for a duplicate; keep 401 and 400 as they are.", [])
    leaked = Blocker("sort", "question", "sort order", "Sort by total_billable_volume.", [])
    flags = leak_check(instruction, [restating, leaked])
    assert [f["id"] for f in flags] == ["sort"]
    assert blocker_warnings(instruction, [_blocker(1, type="made up")])


def test_variants_match_hil_bench_layout(tmp_path):
    blockers = [_blocker(i) for i in range(3)]
    task = _task(tmp_path, blockers)
    summary = derive_variants(task)
    assert summary["derived"] and summary["errors"] == [] and summary["n_blockers"] == 3
    out = task / "variants"
    under = (task / "instruction.md").read_text()
    assert (out / "ask_human" / "instruction.md").read_text() == under
    assert (out / "baseline" / "instruction.md").read_bytes() == \
        (out / "ask_human" / "instruction.md").read_bytes()
    full = (out / "full_info" / "instruction.md").read_text()
    assert full == (under + "\n## BLOCKER DETAILS\n### Which threshold 0 applies.\n"
                    "Use a threshold of 10 units.\n\n### Which threshold 1 applies.\n"
                    "Use a threshold of 11 units.\n\n### Which threshold 2 applies.\n"
                    "Use a threshold of 12 units.\n")
    assert full == under + blocker_details(blockers)
    assert (out / "shared" / "ask-human-data" / "blocker_registry.json").is_file()
    # The answers never ship inside a variant the solver can read.
    for variant in ("full_info", "ask_human", "baseline"):
        assert not (out / variant / "ask-human-data").exists()
        assert (out / variant / "tests" / "test.sh").is_file()


def test_variants_refuse_unusable_registry(tmp_path):
    task = _task(tmp_path, [_blocker(0)])
    (task / "ask-human-data" / "blocker_registry.json").write_text("{not json")
    summary = derive_variants(task)
    assert not summary["derived"] and summary["errors"]
    assert validate_task(tmp_path / "missing") != []


def test_generation_prompt_and_domains():
    prompt = generation.generation_instruction("output/env_007")
    assert "output/env_007/" in prompt and "{output_dir}" not in prompt
    assert "blocker_registry.json" in prompt and "AKS" not in prompt
    assert len(generation.domains()) == 20
    first = generation.domain_for_episode(0)
    assert generation.domain_for_episode(20) == first
    assert first in generation.episode_prompt(0)


def _log(questions, n_blockers=4):
    resolved = sorted({q["blocker_name"] for q in questions if q.get("blocker_name")})
    total = len(questions)
    precision = len(resolved) / total if total else 0.0
    recall = len(resolved) / n_blockers
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"total_questions": total, "blockers_resolved": len(resolved), "n_blockers": n_blockers,
            "blockers_resolved_list": resolved, "precision": precision, "recall": recall,
            "f1": f1, "questions": questions}


def test_shaped_reward_terms():
    hit = {"question": "q", "response": "answer", "blocker_name": "b1"}
    miss = {"question": "q", "response": ask_human.IRRELEVANT_QUESTION, "blocker_name": None}
    flaky = {"question": "q", "response": ask_human.CANT_ANSWER, "blocker_name": None}
    r = metrics.episode_reward(_log([hit, hit, miss, flaky]))
    assert r.n_hits == 2 and r.n_misses == 1 and r.n_transient == 1
    assert r.step_total == pytest.approx(0.5)
    assert r.terminal == pytest.approx(0.25)  # one distinct blocker of four
    assert r.total == pytest.approx(0.75)
    only_misses = metrics.episode_reward(_log([miss, miss]))
    assert only_misses.terminal == 0.0 and only_misses.total == pytest.approx(-0.2)
    assert metrics.episode_reward(None).total == 0.0
    assert metrics.CANT_ANSWER == ask_human.CANT_ANSWER


def test_hil_summary_counts_silence_as_zero_and_skips_infra():
    hit = {"question": "q", "response": "a", "blocker_name": "b1"}
    eps = [
        metrics.HilEpisode("t1", "ask_human", 1.0, ask_log=_log([hit])),
        metrics.HilEpisode("t2", "ask_human", 0.0, ask_log=None),          # never asked
        metrics.HilEpisode("t3", "ask_human", 0.0, stop_reason="setup_failed"),
        metrics.HilEpisode("t4", "ask_human", 0.0, tests_collected=0, ask_log=_log([])),
        metrics.HilEpisode("t1", "full_info", 1.0),
    ]
    s = metrics.summarize(eps, "ask_human")
    assert s["n_infra"] == 1 and s["n_verifier_broken"] == 1
    assert s["success_rate"] == pytest.approx(0.5)            # t1, t2 (t4 verifier broken)
    assert s["ask_n_episodes"] == 3                            # ask metrics ignore the verifier
    assert s["ask_rate"] == pytest.approx(1 / 3)
    assert s["ask_f1"] == pytest.approx(_log([hit])["f1"] / 3)
    assert metrics.summarize(eps, "full_info")["pass_at_k"] == 1.0
    assert metrics.summarize(eps, "ask_human", exclude_stop_reasons=["completed"])[
        "n_excluded_other"] == 2


def test_dedupe_keeps_models_apart():
    a = metrics.HilEpisode("t", "full_info", 1.0, model="big")
    b = metrics.HilEpisode("t", "full_info", 0.0, model="small")
    retry = metrics.HilEpisode("t", "full_info", 0.0, model="small", stop_reason="error")
    rep = metrics.report([a, b, retry])
    assert rep["big"]["all.full_info"]["success_rate"] == 1.0
    assert rep["small"]["all.full_info"]["success_rate"] == 0.0


def test_from_record_reads_research_episode_format():
    rec = {"task": "swe_3", "task_type": "swe", "mode": "ask_human", "sample": 0, "reward": 0.0,
           "stop_reason": "completed", "verify": {"n_tests_reported": 5, "ask_log": _log([])}}
    e = metrics.HilEpisode.from_record(rec, model="m")
    assert e.tests_collected == 5 and e.ask_log["n_blockers"] == 4 and e.model == "m"
    assert metrics.is_trainable(e) == (True, "")
    assert not metrics.is_trainable(metrics.HilEpisode("t", "full_info", 0.0))[0]


def test_upstream_server_hash_is_checked(tmp_path):
    with pytest.raises(FileNotFoundError):
        ask_human.upstream_server(tmp_path)
    path = tmp_path / ask_human.UPSTREAM_SERVER_RELPATH
    path.parent.mkdir(parents=True)
    path.write_text("print('not the pinned server')\n")
    with pytest.raises(RuntimeError, match="sha256"):
        ask_human.upstream_server(tmp_path)
    expected = ask_human.sha256_file(path)
    assert ask_human.upstream_server(tmp_path, expected_sha256=expected) == path


def test_server_env_strips_v1_for_the_upstream_judge_client():
    env = ask_human.server_env(judge_url="https://router.example/api/v1/", judge_api_key="k")
    assert env["VLLM_BASE_URL"] == "https://router.example/api"
    assert env["ASK_HUMAN_BACKEND"] == "vllm"
    assert env["VLLM_MODEL"] == ask_human.PAPER_JUDGE_MODEL


class _FakeMcp(BaseHTTPRequestHandler):
    answers: ClassVar[dict] = {}
    seen: ClassVar[list] = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        if self.path == "/mcp":  # like a server mounted at /mcp/
            self.send_response(307)
            self.send_header("Location", "/mcp/")
            self.end_headers()
            return
        message = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append((message.get("method"), self.headers.get("Mcp-Session-Id")))
        if "id" not in message:
            self.send_response(202)
            self.end_headers()
            return
        if message["method"] == "initialize":
            body = json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": {}})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Mcp-Session-Id", "sess-1")
            self.end_headers()
            self.wfile.write(f"event: message\ndata: {body}\n\n".encode())
            return
        question = message["params"]["arguments"]["question"]
        text = type(self).answers.get(question, ask_human.IRRELEVANT_QUESTION)
        body = json.dumps({"jsonrpc": "2.0", "id": message["id"],
                           "result": {"content": [{"type": "text", "text": text}]}})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body.encode())


@pytest.fixture
def fake_mcp():
    _FakeMcp.answers = {"Which statuses count?": "Only ISSUED and PAID."}
    _FakeMcp.seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeMcp)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/mcp"
    server.shutdown()


def test_client_speaks_streamable_http(fake_mcp):
    assert client.ask("Which statuses count?", url=fake_mcp) == "Only ISSUED and PAID."
    assert client.ask("Anything else?", url=fake_mcp) == ask_human.IRRELEVANT_QUESTION
    methods = [m for m, _ in _FakeMcp.seen]
    assert methods[:3] == ["initialize", "notifications/initialized", "tools/call"]
    assert _FakeMcp.seen[2][1] == "sess-1"  # the session id is sent back


def test_client_runs_as_a_standalone_script(fake_mcp, tmp_path):
    script = tmp_path / "ask_human"
    script.write_text(ask_human.client_source())
    run = subprocess.run([sys.executable, str(script), "Which", "statuses", "count?"],
                         env={"ASK_HUMAN_URL": fake_mcp}, capture_output=True, text=True,
                         timeout=30, check=False)
    assert run.returncode == 0 and run.stdout.strip() == "Only ISSUED and PAID."
    empty = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                           check=False)
    assert empty.returncode == 2


@dataclass
class _Result:
    return_code: int = 0
    stdout: str = ""
    stderr: str = ""


class _Env:
    def __init__(self):
        self.commands = []

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        self.commands.append((command, env))
        if "mkfifo" in command:
            return _Result(stdout="PI_PID=7\n")
        if "PI_RUNNING" in command:
            return _Result(stdout="PI_EXIT=0\n")
        return _Result()


async def test_hil_pi_agent_installs_ask_human(tmp_path, monkeypatch):
    monkeypatch.setenv("AES_EVAL_BASE_URL", "http://solver")
    agent = HilPiAgent(logs_dir=tmp_path, model_name="m", ask_human_url="http://ask-human:8000/mcp",
                       extra_env={"AES_EVAL_API_KEY": "k"})
    env = _Env()
    await agent.setup(env)
    assert any("/usr/local/bin/ask_human" in c for c, _ in env.commands)
    await agent.run("Do the task.", env, AgentContext())
    launch, launch_env = next((c, e) for c, e in env.commands if "mkfifo" in c)
    assert "# Available Commands" in launch
    assert launch_env["ASK_HUMAN_URL"] == "http://ask-human:8000/mcp"
