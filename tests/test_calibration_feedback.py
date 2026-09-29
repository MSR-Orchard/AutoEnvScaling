from autoenvscaling.calibration.feedback import revision_feedback, should_revise
from autoenvscaling.config import Settings
from autoenvscaling.results import (BUILD, CONTAMINATION, EMPTY, REFERENCE, RUBRIC,
                                    CalibrationReport, CheckResult, ValidationReport)


def _skipped(name):
    return CheckResult(name, False, "not run", {"skipped": True})


def _valid():
    return ValidationReport("t", [CheckResult(n, True) for n in
                                  (BUILD, REFERENCE, EMPTY, CONTAMINATION, RUBRIC)])


def test_build_failure_feedback_includes_the_log_and_skipped_checks():
    report = ValidationReport("t", [
        CheckResult(BUILD, False, "the environment image did not build",
                    {"build_log": "E: Unable to locate package nosuchpkg"}),
        _skipped(REFERENCE), _skipped(EMPTY), CheckResult(CONTAMINATION, True), _skipped(RUBRIC)])
    text = revision_feedback(report, None)
    assert "VALIDATION FAILED" in text and "nosuchpkg" in text
    assert ("Not run because of the failure above: reference_solution, empty_solution, "
            "rubric_review") in text
    assert "repeats every check" in text


def test_reference_empty_contamination_and_rubric_feedback():
    report = ValidationReport("t", [
        CheckResult(BUILD, True),
        CheckResult(REFERENCE, False, "the reference solution got reward 0.5",
                    {"verifier_output": "FAILED test_outputs.py::test_index"}),
        _skipped(EMPTY),
        CheckResult(CONTAMINATION, False, "shares a 13-gram with tb:cron"),
        CheckResult(RUBRIC, False, "failed rubric criteria: typos",
                    {"verdicts": [{"name": "typos", "outcome": "fail", "explanation": "bad path"},
                                  {"name": "verifiable", "outcome": "pass", "explanation": ""}]}),
    ])
    text = revision_feedback(report)
    assert "test_index" in text and "reference solution" in text
    assert "held-out benchmark task" in text
    assert "- typos: bad path" in text and "verifiable" not in text
    empty = ValidationReport("t", [CheckResult(BUILD, True), CheckResult(REFERENCE, True),
                                   CheckResult(EMPTY, False, "do-nothing got reward 0.6")])
    assert "untouched environment" in revision_feedback(empty)


def test_infrastructure_failures_are_not_blamed_on_the_proposer():
    report = ValidationReport("t", [CheckResult(BUILD, False, "sandbox quota",
                                                {"infra_error": True})])
    assert "not because of your environment" in revision_feedback(report)
    assert not should_revise(report, None, 0, Settings())


def test_calibration_feedback_by_verdict():
    easy = CalibrationReport("t", [1.0, 1.0, 0.9, 1.0])
    assert "too easy" in revision_feedback(_valid(), easy)
    flat = CalibrationReport("t", [0.5, 0.5, 0.5, 0.5])
    assert "discriminate" in revision_feedback(_valid(), flat)
    hard = CalibrationReport("t", [0.0, 0.0, 0.1, 0.0], hint_probe_reward=0.6)
    assert "fair but hard" in revision_feedback(_valid(), hard)
    hard.hint_probe_reward = 0.0
    assert "underdetermined" in revision_feedback(_valid(), hard)
    hard.hint_probe_reward = None
    assert "too hard" in revision_feedback(_valid(), hard)
    ok = CalibrationReport("t", [0.0, 1.0, 0.5, 0.5])
    assert "No revision is needed" in revision_feedback(_valid(), ok)


def test_should_revise_allows_two_revisions():
    s = Settings()
    bad = ValidationReport("t", [CheckResult(BUILD, False, "no")])
    assert should_revise(bad, None, 0, s) and should_revise(bad, None, 1, s)
    assert not should_revise(bad, None, 2, s)
    admitted = CalibrationReport("t", [0.0, 1.0, 0.5, 0.5])
    assert not should_revise(_valid(), admitted, 0, s)
    assert should_revise(_valid(), CalibrationReport("t", [1.0] * 4), 0, s)
