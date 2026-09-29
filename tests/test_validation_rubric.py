from pathlib import Path

import pytest
from harbor.cli.quality_checker.models import (CheckReport, QualityCheckModel, QualityCheckResult,
                                               load_rubric)

from autoenvscaling.validation import rubric as R

SHIPPED = Path(__file__).resolve().parents[1] / "rubrics" / "terminal-bench.toml"


def test_shipped_rubric_is_the_terminal_bench_rubric_minus_binary_reward():
    text = SHIPPED.read_text()
    assert "harbor-framework/terminal-bench" in text and "Apache License 2.0" in text
    assert "ad8b8447d82d23456a8b99aa0b3c65e7abf0646c" in text
    names = [c.name for c in load_rubric(SHIPPED).criteria]
    assert len(names) == 34 and len(set(names)) == 34
    assert "binary_reward" not in names
    for expected in ("verifiable", "anti_cheat_robustness", "test_instruction_alignment",
                     "do_not_modify_enforced"):
        assert expected in names


def test_resolve_rubric():
    assert R.resolve_rubric("terminal-bench") == R.REPO_RUBRICS / "terminal-bench.toml"
    assert R.resolve_rubric(str(SHIPPED)) == SHIPPED
    assert R.resolve_rubric(R.HARBOR_DEFAULT) is None
    with pytest.raises(FileNotFoundError):
        R.resolve_rubric("no-such-rubric")


def test_filtered_rubric_drops_skipped_criteria(tmp_path):
    path, applied = R.write_filtered_rubric(SHIPPED, ["typos", "binary_reward"],
                                            tmp_path / "r.toml")
    assert "typos" not in applied and len(applied) == 33
    assert [c.name for c in load_rubric(path).criteria] == applied
    with pytest.raises(ValueError):
        R.write_filtered_rubric(SHIPPED, R.rubric_file_criteria(str(SHIPPED)), tmp_path / "x.toml")


def _report(checks=None, error=None):
    return CheckReport(results=[QualityCheckResult(task_name="t", checks=checks or {},
                                                   error=error, cost_usd=0.12)])


async def test_review_task_calls_harbor_check_and_parses_verdicts(tmp_path):
    seen = {}

    async def fake_run_checks(**kwargs):
        seen.update(kwargs)
        names = [c.name for c in load_rubric(kwargs["rubric_path"]).criteria]
        checks = {n: QualityCheckModel(outcome="pass", explanation="ok") for n in names}
        checks["typos"] = QualityCheckModel(outcome="fail", explanation="'recieve' in solve.sh")
        checks["artifact_efficiency"] = QualityCheckModel(outcome="not_applicable", explanation="")
        return _report(checks), tmp_path / "job"

    reviewer = R.ReviewerSpec(agent="codex", model="gpt-5.6-sol")
    result = await R.review_task(tmp_path / "task", reviewer, rubric="terminal-bench",
                                 skip=("binary_reward",), jobs_dir=tmp_path / "jobs",
                                 env_type="modal", run_checks=fake_run_checks)
    assert seen["agent"] == "codex" and seen["model"] == "gpt-5.6-sol"
    assert seen["environment"].value == "modal" and seen["jobs_dir"] == tmp_path / "jobs"
    assert not result.passed and [v.name for v in result.failed] == ["typos"]
    assert result.cost_usd == 0.12 and len(result.verdicts) == 34


def test_not_applicable_counts_as_pass_and_errors_do_not():
    applied = ["a", "b"]
    ok = R.parse_report(_report({"a": QualityCheckModel(outcome="pass", explanation=""),
                                 "b": QualityCheckModel(outcome="not_applicable",
                                                        explanation="")}), applied)
    assert ok.passed
    missing = R.parse_report(_report({"a": QualityCheckModel(outcome="pass", explanation="")}),
                             applied)
    assert missing.error and not missing.passed
    errored = R.parse_report(_report(error="Check agent did not produce a valid result"), applied)
    assert errored.error and not errored.passed


def test_reviewer_for_proposer():
    assert R.ReviewerSpec.for_proposer("codex", "gpt-5.6-sol") == R.ReviewerSpec("codex",
                                                                                 "gpt-5.6-sol")
    pi = R.ReviewerSpec.for_proposer("pi", "deepseek-v4-flash",
                                     {"AES_PROPOSER_BASE_URL": "http://p.test/v1",
                                      "AES_PROPOSER_API_KEY": "k"})
    assert pi.model == "openai/deepseek-v4-flash"
    assert pi.agent_kwargs == {"model_api": "openai-completions"}
    assert pi.agent_env == {"OPENAI_BASE_URL": "http://p.test/v1", "OPENAI_API_KEY": "k"}
    with pytest.raises(ValueError):
        R.ReviewerSpec.for_proposer("pi", "m", {})


def test_reviewer_from_env():
    assert R.ReviewerSpec.from_env({}) is None
    spec = R.ReviewerSpec.from_env({"AES_REVIEWER_MODEL": "claude-opus-5"})
    assert spec == R.ReviewerSpec(agent="claude-code", model="claude-opus-5")
    spec = R.ReviewerSpec.from_env({"AES_REVIEWER_MODEL": "openai/m", "AES_REVIEWER_AGENT": "pi"})
    assert spec.agent == "pi"
