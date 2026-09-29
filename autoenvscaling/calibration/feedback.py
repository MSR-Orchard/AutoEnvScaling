"""Revision feedback: tell the proposer which admission check failed and why.

An environment that fails validation or calibration goes back to the proposer, which may revise it
up to ``CalibrationSettings.max_revisions`` (2) times per assignment; every revision repeats all
checks. ``revision_feedback`` turns the two reports into the message for the revision episode, and
``should_revise`` decides whether another revision is allowed.

The wording follows the old host's revision prompt and feedback templates
(calibration_loop._PROMPT_REVISION and the Exp1 feedback lane).
"""

from __future__ import annotations

from autoenvscaling.config import Settings
from autoenvscaling.results import (BUILD, CONTAMINATION, EMPTY, REFERENCE, RUBRIC,
                                    CalibrationReport, CheckResult, ValidationReport)

LOG_CHARS = 3_000


def should_revise(validation: ValidationReport, calibration: CalibrationReport | None,
                  revision: int, settings: Settings) -> bool:
    """True if the environment was not admitted and the assignment has revisions left.

    ``revision`` is the revision number of the version just checked (0 for the first submission).
    A report whose outcome is unknown because of an infrastructure failure is not grounds for a
    revision: rerun the checks instead.
    """
    if any(c.data.get("infra_error") for c in validation.checks):
        return False
    admitted = validation.passed and calibration is not None and calibration.passed
    return not admitted and revision < settings.calibration.max_revisions


def revision_feedback(validation: ValidationReport,
                      calibration: CalibrationReport | None = None) -> str:
    """What failed and why, written for the proposer's revision episode."""
    if not validation.passed:
        body = _validation_feedback(validation)
    elif calibration is None:
        body = ("Your environment passed validation but has not been calibrated against the "
                "current solver yet.")
    elif calibration.passed:
        return ("Your environment passed validation and calibration "
                f"(mean reward {calibration.mean:.2f}, std {calibration.std:.2f}). "
                "No revision is needed.")
    else:
        body = _calibration_feedback(calibration)
    return (f"{body}\n\nRevise the environment to fix this and submit it again. The revised "
            "environment repeats every check: build, reference solution, empty solution, "
            "contamination, rubric review, and calibration against the current solver.")


def _validation_feedback(report: ValidationReport) -> str:
    failures = [c for c in report.checks if not c.passed and not c.data.get("skipped")]
    skipped = [c.name for c in report.checks if c.data.get("skipped")]
    parts = ["VALIDATION FAILED. The host checked your environment and it did not pass."]
    for check in failures:
        parts.append(_explain(check))
    if skipped:
        parts.append("Not run because of the failure above: " + ", ".join(skipped) + ".")
    return "\n\n".join(parts)


def _explain(check: CheckResult) -> str:
    data = check.data
    if data.get("infra_error"):
        return (f"[{check.name}] The host could not complete this check because of an "
                f"infrastructure problem, not because of your environment: {check.detail}")
    if check.name == BUILD:
        if data.get("stage") == "structure":
            return (f"[build] {check.detail}\nThe task must be a complete Harbor task: task.toml, "
                    "instruction.md, environment/Dockerfile, solution/solve.sh and tests/test.sh.")
        return (f"[build] {check.detail}\nBuild log (tail):\n{_tail(data.get('build_log'))}\n"
                "Fix environment/Dockerfile so the image builds.")
    if check.name == REFERENCE:
        return (f"[reference solution] {check.detail}\nYour own reference solution does not "
                "satisfy your own tests. Either solution/solve.sh does not accomplish the task, or "
                "the tests check something the instruction never asks for. Make them agree.\n"
                f"Verifier output (tail):\n{_tail(data.get('verifier_output'))}")
    if check.name == EMPTY:
        return (f"[empty solution] {check.detail}\nThe tests do not distinguish real work from an "
                "untouched environment. Make them check the task's actual effect on the "
                "environment, so that doing nothing earns less than half the reward.\n"
                f"Verifier output (tail):\n{_tail(data.get('verifier_output'))}")
    if check.name == CONTAMINATION:
        return (f"[contamination] {check.detail}\nThe instruction overlaps word for word with a "
                "held-out benchmark task. Write the task and its instruction yourself; do not copy "
                "benchmark tasks.")
    if check.name == RUBRIC:
        lines = [f"[rubric review] {check.detail}"]
        for verdict in data.get("verdicts", []):
            if verdict.get("outcome") == "fail":
                lines.append(f"- {verdict['name']}: {verdict.get('explanation', '').strip()}")
        return "\n".join(lines)
    return f"[{check.name}] {check.detail}"


def _calibration_feedback(cal: CalibrationReport) -> str:
    rewards = ", ".join(f"{r:.2f}" for r in cal.rewards)
    head = (f"CALIBRATION FAILED. The current solver attempted your environment {len(cal.rewards)} "
            f"times: rewards [{rewards}], mean {cal.mean:.2f}, std {cal.std:.2f}. An environment "
            f"is kept when the mean is in [{cal.band_low:.2f}, {cal.band_high:.2f}] and the std is "
            f"at least {cal.min_std:.2f}.")
    verdict = cal.verdict
    if verdict == "too_easy":
        advice = ("The environment is too easy for the current solver. Make it harder with a "
                  "longer horizon or an additional skill, so the solver sometimes fails.")
    elif verdict == "too_hard":
        if cal.hint_probe_reward is None:
            advice = ("The environment is too hard for the current solver. Check that the "
                      "instruction states everything the tests check, then make the task more "
                      "approachable: fewer compounding steps, a clearer starting state, or more "
                      "scaffolding.")
        elif cal.hint_probe_reward >= cal.band_low:
            advice = (f"The environment is too hard, but with high-level hints from your reference "
                      f"solution the solver reached reward {cal.hint_probe_reward:.2f}. The task "
                      "looks fair but hard: keep it and make the path more discoverable.")
        else:
            advice = (f"The environment is too hard, and even with high-level hints from your "
                      f"reference solution the solver reached only "
                      f"{cal.hint_probe_reward:.2f}. The task is probably underdetermined: "
                      "clarify the instruction so it states what the tests check, or add "
                      "scaffolding.")
    elif verdict == "no_variance":
        advice = ("Every attempt scored about the same, so the environment gives no learning "
                  "signal. Make the tests discriminate: independent tests for separate parts of "
                  "the task, so genuine solver decisions lead to different rewards. Do not add "
                  "randomness.")
    else:
        advice = "Calibration did not produce a usable result."
    return f"{head}\n{advice}"


def _tail(text: object) -> str:
    text = str(text or "").strip()
    return text[-LOG_CHARS:] if text else "(empty)"
