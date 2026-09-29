import json
from pathlib import Path

import httpx
import pytest

from autoenvscaling.baselines import arms as A
from autoenvscaling.baselines import llm as L
from autoenvscaling.baselines.emission import materialize, parse_files
from autoenvscaling.baselines.prompts import build_prompt, env_feedback, repair_prompt
from autoenvscaling.baselines.run import describe, load_assignments, summarize, wilson
from autoenvscaling.results import (BUILD, EMPTY, REFERENCE, CheckResult, ValidationReport)

GOOD = """Here is the task.
<file path="task.toml">
schema_version = "1.4"
</file>
<file path="instruction.md">
Count the lines in /app/data.txt and write the number to /app/count.txt.
</file>
<file path="environment/Dockerfile">
FROM python:3.12-slim
RUN pip install pytest pytest-json-ctrf
</file>
<file path="solution/solve.sh">
#!/bin/bash
wc -l < /app/data.txt > /app/count.txt
</file>
<file path="tests/test_outputs.py">
def test_count():
    assert open('/app/count.txt').read().strip() == '3'
</file>
"""


def test_parse_files_rejects_unsafe_paths_and_keeps_the_last_block():
    reply = ('<file path="../etc/passwd">x</file><file path="/abs">x</file>'
             '<file path="./a/b.txt">one</file>\n<file path="a/b.txt">\ntwo\n</file>')
    parsed = parse_files(reply)
    assert parsed.files == {"a/b.txt": "two\n"}
    assert parsed.rejected == ["../etc/passwd", "/abs"]


def test_materialize_writes_files_and_marks_scripts_executable(tmp_path):
    parsed = materialize(GOOD, tmp_path / "task")
    assert parsed.ok and len(parsed.files) == 5
    solve = tmp_path / "task" / "solution" / "solve.sh"
    assert solve.read_text().startswith("#!/bin/bash") and solve.stat().st_mode & 0o111


def test_prompts_carry_description_contract_and_feedback():
    prompt = build_prompt("Build a log parsing task.", extra_guidance="Use harbor task init.")
    assert "no tools" in prompt and "Build a log parsing task." in prompt
    assert "tests/test_outputs.py" in prompt and "Use harbor task init." in prompt
    own = build_prompt("x", managed_verifier=False)
    assert "/logs/verifier/reward.txt" in own
    repair = repair_prompt(prompt, "old answer", "VALIDATION FAILED")
    assert repair.startswith(prompt) and "old answer" in repair and "VALIDATION FAILED" in repair


def test_env_feedback_returns_build_and_verifier_logs():
    build = ValidationReport("t", [CheckResult(BUILD, False, "did not build",
                                               {"build_log": "E: nosuchpkg"})])
    assert "Build log tail" in env_feedback(build) and "nosuchpkg" in env_feedback(build)
    ref = ValidationReport("t", [CheckResult(BUILD, True),
                                 CheckResult(REFERENCE, False, "reward 0",
                                             {"verifier_output": "AssertionError: 4 != 3"})])
    assert "4 != 3" in env_feedback(ref) and "reference solution" in env_feedback(ref)
    emp = ValidationReport("t", [CheckResult(BUILD, True), CheckResult(REFERENCE, True),
                                 CheckResult(EMPTY, False, "reward 0.6")])
    assert "do-nothing" in env_feedback(emp)


class FakeModel:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    async def complete(self, prompt):
        self.prompts.append(prompt)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        text, finish = reply if isinstance(reply, tuple) else (reply, "stop")
        return L.Completion(text=text, finish_reason=finish,
                            usage={"prompt_tokens": 100, "completion_tokens": 10})


class FakeScorer:
    def __init__(self, *reports):
        self.reports = list(reports)
        self.seen = []

    async def __call__(self, task_dir):
        self.seen.append(Path(task_dir))
        return self.reports.pop(0)


def ok():
    return ValidationReport("t", [CheckResult(n, True) for n in (BUILD, REFERENCE, EMPTY)])


def failed_reference():
    return ValidationReport("t", [CheckResult(BUILD, True),
                                  CheckResult(REFERENCE, False, "reward 0",
                                              {"verifier_output": "assert failed"}),
                                  CheckResult(EMPTY, False, "skipped", {"skipped": True})])


async def test_prompt_arm_scores_the_managed_task(tmp_path):
    scorer = FakeScorer(ok())
    result = await A.run_prompt_arm("N001", "count lines", tmp_path / "N001",
                                    complete=FakeModel(GOOD).complete, score=scorer)
    assert result.admitted and result.admitted_round == 1
    task = scorer.seen[0]
    assert (task / "tests" / "test.sh").read_text() == A.MANAGED_VERIFIER.read_text()
    assert (tmp_path / "N001" / "v1" / "reply.txt").read_text() == GOOD


async def test_prompt_arm_asks_once_for_missing_files(tmp_path):
    model = FakeModel("Plan: I will write a task.", GOOD)
    result = await A.run_prompt_arm("N002", "d", tmp_path / "N002", complete=model.complete,
                                    score=FakeScorer(ok()))
    assert result.rounds[0].continuation_used and len(model.prompts) == 2
    assert "You stopped before emitting the task files" in model.prompts[1]
    assert len(result.rounds[0].usage) == 2


async def test_prompt_arm_without_files_is_not_scored(tmp_path):
    scorer = FakeScorer()
    result = await A.run_prompt_arm("N003", "d", tmp_path / "N003",
                                    complete=FakeModel(("no files", "length")).complete,
                                    score=scorer)
    assert not result.admitted and result.rounds[0].rejection == "no_env_generated"
    assert scorer.seen == []


async def test_refusals_and_endpoint_failures_are_excluded(tmp_path):
    refused = await A.run_prompt_arm("N004", "d", tmp_path / "a",
                                     complete=FakeModel(L.PolicyRefusal("filter")).complete,
                                     score=FakeScorer())
    assert refused.excluded == A.EXCLUDED_REFUSAL
    down = await A.run_prompt_arm("N005", "d", tmp_path / "b",
                                  complete=FakeModel(L.EndpointError("HTTP 401")).complete,
                                  score=FakeScorer())
    assert down.excluded == A.EXCLUDED_INFRA


async def test_feedback_arm_repairs_on_top_of_the_previous_candidate(tmp_path):
    seed = await A.run_prompt_arm("N006", "d", tmp_path / "prompt" / "N006",
                                  complete=FakeModel(GOOD).complete,
                                  score=FakeScorer(failed_reference()))
    fix = '<file path="solution/solve.sh">\n#!/bin/bash\necho 3 > /app/count.txt\n</file>\n'
    model = FakeModel(fix)
    scorer = FakeScorer(ok())
    result = await A.run_feedback_arm("N006", "d", tmp_path / "feedback" / "N006", seed=seed,
                                      seed_dir=tmp_path / "prompt" / "N006",
                                      complete=model.complete, score=scorer)
    assert result.admitted and result.admitted_round == 2 and len(result.rounds) == 2
    assert "assert failed" in model.prompts[0] and GOOD.strip() in model.prompts[0]
    task = scorer.seen[0]
    assert (task / "instruction.md").is_file()  # carried over from round 1
    assert "echo 3" in (task / "solution" / "solve.sh").read_text()


async def test_feedback_arm_stops_after_two_repairs(tmp_path):
    seed = await A.run_prompt_arm("N007", "d", tmp_path / "prompt" / "N007",
                                  complete=FakeModel(GOOD).complete,
                                  score=FakeScorer(failed_reference()))
    model = FakeModel(GOOD, GOOD, GOOD)
    result = await A.run_feedback_arm("N007", "d", tmp_path / "feedback" / "N007", seed=seed,
                                      seed_dir=tmp_path / "prompt" / "N007",
                                      complete=model.complete,
                                      score=FakeScorer(failed_reference(), failed_reference()))
    assert not result.admitted and [r.version for r in result.rounds] == [1, 2, 3]
    assert len(model.prompts) == 2


async def test_feedback_arm_inherits_an_admitted_or_excluded_seed(tmp_path):
    seed = await A.run_prompt_arm("N008", "d", tmp_path / "p", complete=FakeModel(GOOD).complete,
                                  score=FakeScorer(ok()))
    model = FakeModel()
    result = await A.run_feedback_arm("N008", "d", tmp_path / "f", seed=seed,
                                      seed_dir=tmp_path / "p", complete=model.complete,
                                      score=FakeScorer())
    assert result.admitted_round == 1 and model.prompts == []
    excluded = A.ArmResult("N009", "prompt", excluded=A.EXCLUDED_REFUSAL)
    result = await A.run_feedback_arm("N009", "d", tmp_path / "g", seed=excluded,
                                      seed_dir=tmp_path / "p", complete=model.complete,
                                      score=FakeScorer())
    assert result.excluded == A.EXCLUDED_REFUSAL


def test_result_round_trip_and_summary(tmp_path):
    admitted = A.ArmResult("N1", "prompt", rounds=[A.Round(1, admitted=True,
                                                           usage=[{"prompt_tokens": 1000,
                                                                   "completion_tokens": 100}])])
    rejected = A.ArmResult("N2", "prompt", rounds=[A.Round(1, rejection=REFERENCE,
                                                           usage=[{"prompt_tokens": 1000,
                                                                   "completion_tokens": 100}])])
    refused = A.ArmResult("N3", "prompt", excluded=A.EXCLUDED_REFUSAL)
    for r in (admitted, rejected, refused):
        A.write_result(tmp_path / "prompt" / r.assignment_id / "result.json", r)
    again = A.ArmResult.from_dict(json.loads(
        (tmp_path / "prompt" / "N1" / "result.json").read_text()))
    assert again.admitted and again.rounds[0].usage[0]["prompt_tokens"] == 1000
    cell = summarize(tmp_path, usd_in=1.0, usd_out=10.0)["prompt"]
    assert (cell["assignments"], cell["scored"], cell["admitted"]) == (3, 2, 1)
    assert cell["valid_rate"] == 0.5 and cell["excluded"] == {"policy_refusal": 1}
    assert cell["cost_usd"] == pytest.approx(2000 / 1e6 + 200 / 1e5)
    assert cell["cost_per_valid_env_usd"] == pytest.approx(cell["cost_usd"])


def test_wilson_interval_matches_known_value():
    p, lo, hi = wilson(89, 100)
    assert p == 0.89
    assert lo == pytest.approx(0.8139, abs=1e-3) and hi == pytest.approx(0.9372, abs=1e-3)


def test_load_and_describe_assignments(tmp_path):
    path = tmp_path / "a.jsonl"
    path.write_text(json.dumps({"assignment_id": "N1", "description": "Free text."}) + "\n" +
                    json.dumps({"assignment_id": "N2", "parent_task_id": "cron-fix",
                                "solver_pass_rate": 0.0, "common_failure": "wrong path",
                                "direction": "simpler"}) + "\n")
    recs = load_assignments(path)
    assert describe(recs[0]) == "Free text."
    text = describe(recs[1])
    assert "cron-fix" in text and "0.00" in text and "simpler variant" in text
    listed = tmp_path / "b.json"
    listed.write_text(json.dumps({"assignments": [{"assignment_id": "N3", "description": "d"}]}))
    assert load_assignments(listed)[0]["assignment_id"] == "N3"


async def test_chat_client_retries_and_classifies(monkeypatch):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json={"model": "m", "usage": {"prompt_tokens": 5},
                                         "choices": [{"finish_reason": "stop",
                                                      "message": {"content": "hi"}}]})

    real = httpx.AsyncClient
    monkeypatch.setattr(L.httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(L.asyncio, "sleep", _no_sleep)
    client = L.ChatClient(base_url="http://model.test/v1", model="m", reasoning_effort="high")
    out = await client.complete("prompt")
    assert out.text == "hi" and len(calls) == 2 and calls[0]["reasoning_effort"] == "high"

    def refuse(request):
        return httpx.Response(200, json={"choices": [{"finish_reason": "content_filter",
                                                      "message": {"content": None}}]})
    monkeypatch.setattr(L.httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(refuse), **kw))
    with pytest.raises(L.PolicyRefusal):
        await client.complete("prompt")

    monkeypatch.setattr(L.httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(
                            lambda r: httpx.Response(401, text="bad key")), **kw))
    with pytest.raises(L.EndpointError, match="401"):
        await client.complete("prompt")


async def _no_sleep(_):
    return None
