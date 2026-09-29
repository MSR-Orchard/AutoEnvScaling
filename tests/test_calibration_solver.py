import json
from pathlib import Path

import pytest

from autoenvscaling.calibration import solver as S
from autoenvscaling.tasks import HarborTask
from autoenvscaling.validation.harbor_runner import RunResult, TrialOutcome

PI_EVENTS = [
    {"type": "session", "id": "s1"},
    {"type": "message_end",
     "message": {"role": "user", "content": [{"type": "text", "text": "the instruction"}]}},
    {"type": "message_update", "delta": "ignored"},
    {"type": "message_end", "message": {"role": "assistant", "content": [
        {"type": "thinking", "thinking": "look at the logs first"},
        {"type": "text", "text": "Listing files."},
        {"type": "toolCall", "id": "c1", "name": "bash", "arguments": {"command": "ls /app"}}]}},
    {"type": "tool_execution_end", "toolCallId": "c1", "toolName": "bash",
     "result": {"content": [{"type": "text", "text": "logs\n"}]}, "isError": False},
    {"type": "message_end",
     "message": {"role": "toolResult", "toolCallId": "c1", "toolName": "bash",
                 "content": [{"type": "text", "text": "logs\n"}]}},
    {"type": "message_end",
     "message": {"role": "assistant", "content": [{"type": "text", "text": "Done."}]}},
]


def _ndjson(events):
    return "\n".join(json.dumps(e) for e in events) + "\nnot json\n"


def test_pi_turns_parses_messages_and_tool_calls():
    turns = S.pi_turns(_ndjson(PI_EVENTS))
    assert [t.role for t in turns] == ["assistant", "tool", "assistant"]
    first = turns[0]
    assert first.content.startswith("<think>\nlook at the logs first\n</think>")
    assert "Listing files." in first.content
    call = first.tool_calls[0]
    assert call["id"] == "c1" and call["function"]["name"] == "bash"
    assert json.loads(call["function"]["arguments"]) == {"command": "ls /app"}
    assert (turns[1].tool_call_id, turns[1].name, turns[1].content) == ("c1", "bash", "logs\n")


def test_pi_turns_falls_back_to_tool_execution_events():
    events = [e for e in PI_EVENTS if (e.get("message") or {}).get("role") != "toolResult"]
    turns = S.pi_turns(_ndjson(events))
    assert [t.role for t in turns] == ["assistant", "tool", "assistant"]
    assert turns[1].tool_call_id == "c1"


def test_endpoint_agent_spec(monkeypatch):
    monkeypatch.setenv("AES_SOLVER_API_KEY", "secret")
    ep = S.SolverEndpoint(base_url="http://solver.internal.test:30000/v1", model="solver-9b")
    spec = ep.agent_spec()
    assert spec.name == "pi" and spec.model_name == "openai/solver-9b"
    assert spec.kwargs == {"model_api": "openai-completions", "version": S.PI_VERSION}
    assert spec.env == {"OPENAI_BASE_URL": "http://solver.internal.test:30000/v1",
                        "OPENAI_API_KEY": "secret"}
    assert spec.extra_allowed_hosts == () and spec.timeout_sec == 1200.0
    closed = S.SolverEndpoint(base_url="https://solver.example.org/v1", model="org/m",
                              public_network=False)
    assert closed.agent_spec().extra_allowed_hosts == ("solver.example.org",)
    assert closed.harbor_model_name == "openai/org/m"


def test_endpoint_from_env():
    ep = S.SolverEndpoint.from_env({"AES_SOLVER_BASE_URL": "http://x.test/v1",
                                    "AES_SOLVER_MODEL": "m"})
    assert (ep.base_url, ep.model) == ("http://x.test/v1", "m")
    with pytest.raises(ValueError):
        S.SolverEndpoint.from_env({})


def test_summarize_solution_and_hints(tmp_path):
    solve = """#!/bin/bash
# reference
set -euo pipefail
export PATH=/opt/bin:$PATH
DB=/app/db.sqlite python3 /solution/repair.py --in /app/data
if [ -f /app/out ]; then
  sqlite3 /app/db.sqlite 'VACUUM'
fi
python3 /solution/repair.py --again
"""
    assert S.summarize_solution(solve) == ["set -euo", "export PATH=/opt/bin:$PATH",
                                           "python3 /solution/repair.py", "if [",
                                           "sqlite3 /app/db.sqlite"]
    (tmp_path / "solution").mkdir()
    (tmp_path / "solution" / "solve.sh").write_text(solve)
    hints = S.reference_hints(HarborTask(tmp_path))
    assert hints.splitlines()[0] == "1. set -euo"
    assert S.reference_hints(HarborTask(tmp_path / "missing")) is None


class RecordingRunner:
    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.kwargs = None

    async def run(self, task_dir, agent, **kwargs):
        self.kwargs = kwargs
        self.agent = agent
        return RunResult(outcomes=self.outcomes)


def _task(root: Path) -> HarborTask:
    root.mkdir(parents=True)
    (root / "instruction.md").write_text("Fix the service.\n")
    return HarborTask(root)


async def test_run_solver_builds_trajectories(tmp_path):
    trial_dir = tmp_path / "trial"
    (trial_dir / "agent").mkdir(parents=True)
    (trial_dir / "agent" / "pi.txt").write_text(_ndjson(PI_EVENTS))
    outcomes = [
        TrialOutcome(trial_name="a", trial_dir=trial_dir, reward=0.5, tests_passed=2,
                     tests_total=4, verifier_output="2 passed, 2 failed"),
        TrialOutcome(trial_name="b", exception_type="AgentAuthenticationError",
                     exception_message="401", infra_error=True),
    ]
    runner = RecordingRunner(outcomes)
    ep = S.SolverEndpoint(base_url="http://x.test/v1", model="solver-9b")
    task = _task(tmp_path / "task")
    trajs = await S.run_solver(task, ep, n=2, harbor_env="modal", runner=runner)
    assert runner.kwargs["n_attempts"] == 2 and runner.kwargs["public_network"] is True
    assert runner.kwargs["instruction"] is None and runner.kwargs["label"] == "solver"
    a, b = trajs
    assert (a.reward, a.tests_passed, a.tests_total, a.model) == (0.5, 2, 4, "solver-9b")
    assert a.turns[0].role == "user" and a.turns[0].content == "Fix the service.\n"
    assert len(a.turns) == 4
    assert b.reward is None and "AgentAuthenticationError: 401" in b.verifier_output


async def test_run_solver_appends_hints(tmp_path):
    runner = RecordingRunner([TrialOutcome(trial_name="a", reward=0.1)])
    ep = S.SolverEndpoint(base_url="http://x.test/v1", model="solver-9b")
    trajs = await S.run_solver(_task(tmp_path / "task"), ep, n=1, harbor_env="modal",
                               hints="1. ls /app", runner=runner)
    shown = runner.kwargs["instruction"]
    assert shown.startswith("Fix the service.") and "## Hints" in shown and "1. ls /app" in shown
    assert runner.kwargs["label"] == "hint_probe" and trajs[0].turns[0].content == shown
    with pytest.raises(ValueError):
        await S.run_solver(_task(tmp_path / "t2"), ep, n=0, harbor_env="modal", runner=runner)
