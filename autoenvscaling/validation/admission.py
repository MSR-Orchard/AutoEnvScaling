"""Host-side validation: build, reference solution, empty solution, contamination, rubric review.

``validate`` runs the checks of Section 3.2 in order and returns one ``CheckResult`` per check:

1. build: the task has every required file and a task.toml Harbor accepts, and its image builds.
   The build is exercised by the reference-solution trial (Harbor builds the image when it starts
   the environment), so a build failure is read from that trial rather than from a separate build.
2. reference_solution: the proposer's ``solution/solve.sh`` must receive full reward (1.0).
3. empty_solution: a do-nothing solution (Harbor's ``nop`` agent) must receive less than 0.5.
4. contamination: the instruction must not share a 13-gram with a held-out task.
5. rubric_review: ``harbor check`` with the Terminal-Bench rubric, graded by the proposer's model.

Checks that a failure makes moot are not run. They still appear in the report, with
``passed=False`` and ``data={"skipped": True, "reason": ...}``, so every report lists all five
checks. The contamination check is local and cheap, so it always runs. A check whose outcome is
unknown because the sandbox, the model endpoint or the network failed has
``data["infra_error"] = True``; callers should retry or drop such a task rather than score it
(see ``has_infra_error``).
"""

from __future__ import annotations

import logging
import tomllib
from collections.abc import Collection
from pathlib import Path

import tomli_w

from autoenvscaling.config import Settings
from autoenvscaling.results import (BUILD, CONTAMINATION, EMPTY, REFERENCE, RUBRIC, CheckResult,
                                    ValidationReport)
from autoenvscaling.tasks import HarborTask
from autoenvscaling.validation.contamination import NgramIndex, load_index
from autoenvscaling.validation.harbor_runner import (AgentSpec, HarborRunner, RunResult,
                                                     TrialOutcome, TrialRunner, build_log_tail,
                                                     normalize_task_name)
from autoenvscaling.validation.rubric import ReviewerSpec, review_task

logger = logging.getLogger(__name__)

ALL_CHECKS = (BUILD, REFERENCE, EMPTY, CONTAMINATION, RUBRIC)
EXECUTION_CHECKS = (BUILD, REFERENCE, EMPTY)
OUTPUT_TAIL_CHARS = 4_000


def has_infra_error(report: ValidationReport) -> bool:
    """True if any check's outcome is unknown because of an infrastructure failure."""
    return any(c.data.get("infra_error") for c in report.checks)


def validation_dir(task: HarborTask, settings: Settings) -> Path:
    """Where the Harbor jobs and review artifacts for one version of a task are kept."""
    return Path(settings.work_dir) / "validation" / f"{task.task_id}-{task.digest()}"


async def validate(task: HarborTask, settings: Settings, *, checks: Collection[str] = ALL_CHECKS,
                   runner: TrialRunner | None = None, reviewer: ReviewerSpec | None = None,
                   heldout: NgramIndex | None = None) -> ValidationReport:
    """Run every validation check in order and stop at the first failure that makes later ones moot.

    Keyword arguments (all optional): ``checks`` restricts the report to a subset of the checks
    (build, reference_solution and empty_solution share trials and always run together; the
    Section 4.1 baselines use only these three); ``runner`` replaces the Harbor trial runner;
    ``reviewer`` is the agent and model for the rubric review (default: ``AES_REVIEWER_AGENT`` and
    ``AES_REVIEWER_MODEL``); ``heldout`` is a prebuilt contamination index.
    """
    unknown = set(checks) - set(ALL_CHECKS)
    if unknown:
        raise ValueError(f"unknown checks: {sorted(unknown)}")
    selected = tuple(c for c in ALL_CHECKS if c in set(checks))
    run_exec = any(c in selected for c in EXECUTION_CHECKS)
    if run_exec and not all(c in selected for c in EXECUTION_CHECKS):
        raise ValueError("build, reference_solution and empty_solution must be run together")
    if RUBRIC in selected and reviewer is None:
        reviewer = ReviewerSpec.from_env()
        if reviewer is None:
            raise ValueError("the rubric review needs the proposer's model: pass reviewer= or set "
                             "AES_REVIEWER_MODEL (and AES_REVIEWER_AGENT)")

    work = validation_dir(task, settings)
    results: dict[str, CheckResult] = {}
    blocker: str | None = None  # the first failed check; later expensive checks are skipped

    if run_exec:
        if runner is None:
            runner = HarborRunner(work / "jobs", env_type=settings.sandbox.harbor_env)
        results.update(await _execution_checks(task, settings, runner))
        blocker = next((c for c in EXECUTION_CHECKS if not results[c].passed), None)

    if CONTAMINATION in selected:
        results[CONTAMINATION] = _contamination_check(task, settings, heldout)
        if blocker is None and not results[CONTAMINATION].passed:
            blocker = CONTAMINATION

    if RUBRIC in selected:
        if blocker is not None:
            results[RUBRIC] = _skipped(RUBRIC, blocker)
        else:
            results[RUBRIC] = await _rubric_check(task, settings, reviewer, work / "rubric")

    return ValidationReport(task_id=task.task_id, checks=[results[c] for c in selected])


async def _execution_checks(task: HarborTask, settings: Settings,
                            runner: TrialRunner) -> dict[str, CheckResult]:
    vset = settings.validation
    errors = structure_errors(task)
    if errors:
        return {BUILD: CheckResult(BUILD, False, "the task is not a valid Harbor task: "
                                   + "; ".join(errors), {"stage": "structure", "errors": errors}),
                REFERENCE: _skipped(REFERENCE, BUILD), EMPTY: _skipped(EMPTY, BUILD)}

    reference_run = await _run_one(runner, task, AgentSpec.reference(), "reference")
    ref = reference_run.outcomes[0] if reference_run.outcomes else None
    if ref is None or ref.infra_error:
        detail = ref.error_summary if ref else "Harbor returned no trial"
        return {BUILD: _infra(BUILD, "the reference trial could not run: " + detail, reference_run),
                REFERENCE: _skipped(REFERENCE, BUILD), EMPTY: _skipped(EMPTY, BUILD)}
    if ref.build_failed:
        data = {**_trial_data(ref, reference_run), "build_log": build_log_tail(ref)}
        return {BUILD: CheckResult(BUILD, False, "the environment image did not build: "
                                   + ref.error_summary[:500], data),
                REFERENCE: _skipped(REFERENCE, BUILD), EMPTY: _skipped(EMPTY, BUILD)}
    out = {BUILD: CheckResult(BUILD, True, "the image built and the environment started",
                              {"confirmed_by": "reference trial"})}

    if ref.reward is not None and ref.reward >= vset.reference_min_reward:
        out[REFERENCE] = CheckResult(REFERENCE, True, f"reference solution reward {ref.reward:g}",
                                     _trial_data(ref, reference_run))
    else:
        got = "no reward" if ref.reward is None else f"reward {ref.reward:g}"
        reason = f" ({ref.error_summary[:300]})" if ref.exception_type else ""
        out[REFERENCE] = CheckResult(
            REFERENCE, False,
            f"the reference solution got {got}{reason}; it must get {vset.reference_min_reward:g}",
            {**_trial_data(ref, reference_run),
             "verifier_output": ref.verifier_output[-OUTPUT_TAIL_CHARS:]})
        out[EMPTY] = _skipped(EMPTY, REFERENCE)
        return out

    empty_run = await _run_one(runner, task, AgentSpec.empty(), "empty")
    emp = empty_run.outcomes[0] if empty_run.outcomes else None
    if emp is None or emp.infra_error:
        detail = emp.error_summary if emp else "Harbor returned no trial"
        out[EMPTY] = _infra(EMPTY, "the empty-solution trial could not run: " + detail, empty_run)
    elif emp.reward is None:
        # The verifier crashed on an untouched container: the tests are broken, not the host.
        out[EMPTY] = CheckResult(
            EMPTY, False,
            f"the verifier produced no reward for the empty solution ({emp.error_summary[:300]})",
            {**_trial_data(emp, empty_run),
             "verifier_output": emp.verifier_output[-OUTPUT_TAIL_CHARS:]})
    elif emp.reward < vset.empty_max_reward:
        out[EMPTY] = CheckResult(EMPTY, True, f"empty solution reward {emp.reward:g}",
                                 _trial_data(emp, empty_run))
    else:
        out[EMPTY] = CheckResult(
            EMPTY, False,
            f"a do-nothing solution got reward {emp.reward:g}; it must get less than "
            f"{vset.empty_max_reward:g}",
            {**_trial_data(emp, empty_run),
             "verifier_output": emp.verifier_output[-OUTPUT_TAIL_CHARS:]})
    return out


async def _run_one(runner: TrialRunner, task: HarborTask, agent: AgentSpec,
                   label: str) -> RunResult:
    try:
        return await runner.run(task.path, agent, n_attempts=1, label=label)
    except Exception as exc:  # Harbor raises only when the job itself cannot start
        logger.exception("%s trial for %s failed to start", label, task.task_id)
        return RunResult(outcomes=[TrialOutcome(trial_name="", exception_type=type(exc).__name__,
                                                exception_message=str(exc), failed_phase="setup",
                                                infra_error=True)])


def structure_errors(task: HarborTask) -> list[str]:
    """Missing required files, an empty instruction, or a task.toml that Harbor rejects."""
    errors = [f"missing {rel}" for rel in task.missing_files()]
    if (task.path / "instruction.md").is_file() and not task.instruction.strip():
        errors.append("instruction.md is empty")
    config = task.path / "task.toml"
    if config.is_file():
        from harbor.models.task.config import TaskConfig
        try:
            # Validate what the trials will run: a bare task name gets an org prefix there.
            data = normalize_task_name(tomllib.loads(config.read_text()))
            TaskConfig.model_validate_toml(tomli_w.dumps(data))
        except Exception as exc:  # pydantic ValidationError or a TOML syntax error
            errors.append(f"task.toml is invalid: {str(exc)[:500]}")
    return errors


def _contamination_check(task: HarborTask, settings: Settings,
                         heldout: NgramIndex | None) -> CheckResult:
    vset = settings.validation
    if not (task.path / "instruction.md").is_file():
        return CheckResult(CONTAMINATION, False, "instruction.md is missing", {})
    try:
        index = heldout or load_index(vset.heldout_dataset,
                                      Path(settings.work_dir) / "contamination", n=vset.ngram_n)
    except Exception as exc:
        return CheckResult(CONTAMINATION, False,
                           f"could not load the held-out set {vset.heldout_dataset!r}: {exc}",
                           {"infra_error": True})
    matches = index.matches(task.instruction)
    data = {"n": index.n, "heldout": index.sources, "heldout_tasks": index.n_tasks,
            "matches": [{"heldout_task": m.heldout_task, "ngram": m.ngram} for m in matches]}
    if not matches:
        return CheckResult(CONTAMINATION, True,
                           f"no {index.n}-gram shared with {index.n_tasks} held-out tasks", data)
    tasks = sorted({m.heldout_task for m in matches})
    return CheckResult(CONTAMINATION, False,
                       f"the instruction shares a {index.n}-gram with held-out task(s) "
                       f"{', '.join(tasks[:5])}: \"{matches[0].ngram}\"", data)


async def _rubric_check(task: HarborTask, settings: Settings, reviewer: ReviewerSpec,
                        jobs_dir: Path) -> CheckResult:
    vset = settings.validation
    try:
        result = await review_task(task.path, reviewer, rubric=vset.rubric,
                                   skip=vset.rubric_skip, jobs_dir=jobs_dir,
                                   env_type=settings.sandbox.harbor_env)
    except Exception as exc:
        logger.exception("rubric review failed for %s", task.task_id)
        return CheckResult(RUBRIC, False, f"the rubric review could not run: {exc}",
                           {"infra_error": True, "reviewer_model": reviewer.model})
    data = {"rubric": vset.rubric, "skipped_criteria": list(vset.rubric_skip),
            "reviewer_agent": reviewer.agent, "reviewer_model": reviewer.model,
            "cost_usd": result.cost_usd,
            "job_dir": str(result.job_dir) if result.job_dir else None,
            "verdicts": [{"name": v.name, "outcome": v.outcome, "explanation": v.explanation}
                         for v in result.verdicts]}
    if result.error:
        # The reviewer did not produce a valid verdict file: that says nothing about the task.
        return CheckResult(RUBRIC, False,
                           f"the reviewer produced no valid verdicts: {result.error[:500]}",
                           {**data, "infra_error": True})
    if result.passed:
        return CheckResult(RUBRIC, True,
                           f"passed all {len(result.verdicts)} applied criteria", data)
    names = [v.name for v in result.failed]
    return CheckResult(RUBRIC, False, f"failed rubric criteria: {', '.join(names)}",
                       {**data, "failed_criteria": names})


def _skipped(name: str, blocker: str) -> CheckResult:
    return CheckResult(name, False, f"not run: the {blocker} check failed",
                       {"skipped": True, "reason": f"{blocker} failed"})


def _infra(name: str, detail: str, run: RunResult) -> CheckResult:
    return CheckResult(name, False, detail,
                       {"infra_error": True, "job_dir": str(run.job_dir) if run.job_dir else None,
                        "trials": [o.to_dict() for o in run.outcomes]})


def _trial_data(outcome: TrialOutcome, run: RunResult) -> dict:
    return {"reward": outcome.reward, "tests_passed": outcome.tests_passed,
            "tests_total": outcome.tests_total,
            "job_dir": str(run.job_dir) if run.job_dir else None, "trial": outcome.to_dict()}


async def admit(task: HarborTask, solver, settings: Settings, **validate_kwargs):
    """Validation, then calibration if validation passed: the full admission for one version.

    Returns ``(ValidationReport, CalibrationReport | None)``; feed both to
    ``autoenvscaling.rewards.proposer_reward`` and, for a revision, to
    ``autoenvscaling.calibration.feedback.revision_feedback``.
    """
    from autoenvscaling.calibration.band import calibrate

    report = await validate(task, settings, **validate_kwargs)
    if not report.passed:
        return report, None
    return report, await calibrate(task, solver, settings)
