"""The two Section 4.1 baselines, scored by the same host validation as AutoEnvScaling.

* Prompt: one tool-free model call turns a task description into the task's files. Nothing is
  executed during generation. If a finished reply contains no files, the model is asked once to
  emit them (the research runner's continuation step).
* Prompt + Env Feedback: starts from the Prompt arm's result for the same assignment (round 1 is
  shared, so the arms are paired). While the environment fails validation, the host sends the
  build or verifier logs back and the model revises, for at most two repair rounds (RST's limit,
  used in the research runs). Each repair is applied on top of the previous candidate: files the
  model does not re-emit keep their previous content.

An environment counts as valid when it builds, its reference solution receives full reward and an
empty solution receives less than 0.5 (the ``build``, ``reference_solution`` and
``empty_solution`` checks). Provider refusals and infrastructure failures are recorded as
``excluded`` and left out of the valid-environment rate, as in the paper's comparison.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from autoenvscaling.baselines.emission import materialize
from autoenvscaling.baselines.llm import Completion, EndpointError, PolicyRefusal
from autoenvscaling.baselines.prompts import (build_prompt, continuation_prompt, env_feedback,
                                              repair_prompt)
from autoenvscaling.results import ValidationReport

logger = logging.getLogger(__name__)

MANAGED_VERIFIER = Path(__file__).with_name("verifier_template.sh")
MAX_REPAIRS = 2
EXCLUDED_REFUSAL = "policy_refusal"
EXCLUDED_INFRA = "infra_error"

CompleteFn = Callable[[str], Awaitable[Completion]]
ScoreFn = Callable[[Path], Awaitable[ValidationReport]]


@dataclass
class Round:
    version: int
    task_dir: str | None = None
    admitted: bool = False
    infra_error: bool = False
    rejection: str | None = None  # first failed check, or "no_env_generated"
    report: dict[str, Any] | None = None
    usage: list[dict[str, Any]] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    continuation_used: bool = False
    feedback: str | None = None


@dataclass
class ArmResult:
    assignment_id: str
    arm: str
    rounds: list[Round] = field(default_factory=list)
    excluded: str | None = None
    error: str | None = None

    @property
    def admitted(self) -> bool:
        return any(r.admitted for r in self.rounds)

    @property
    def admitted_round(self) -> int | None:
        return next((r.version for r in self.rounds if r.admitted), None)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "admitted": self.admitted, "admitted_round": self.admitted_round}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArmResult":
        rounds = [Round(**r) for r in data.get("rounds", [])]
        return cls(assignment_id=data["assignment_id"], arm=data["arm"], rounds=rounds,
                   excluded=data.get("excluded"), error=data.get("error"))


@dataclass
class ArmConfig:
    managed_verifier: bool = True
    extra_guidance: str = ""
    max_repairs: int = MAX_REPAIRS


async def run_prompt_arm(assignment_id: str, description: str, out_dir: Path, *,
                         complete: CompleteFn, score: ScoreFn,
                         config: ArmConfig | None = None) -> ArmResult:
    """Generate once without tools, then score with host validation."""
    config = config or ArmConfig()
    result = ArmResult(assignment_id=assignment_id, arm="prompt")
    prompt = build_prompt(description, managed_verifier=config.managed_verifier,
                          extra_guidance=config.extra_guidance)
    version_dir = Path(out_dir) / "v1"
    version_dir.mkdir(parents=True, exist_ok=True)
    (version_dir / "prompt.txt").write_text(prompt)
    round_ = Round(version=1)
    result.rounds.append(round_)
    try:
        reply = await complete(prompt)
        round_.usage.append(reply.usage)
        text = reply.text
        task_dir = version_dir / "task"
        parsed = materialize(text, task_dir)
        if not parsed.ok and reply.finish_reason == "stop":
            extra = await complete(continuation_prompt(prompt, text))
            round_.usage.append(extra.usage)
            merged = text + "\n" + extra.text
            parsed = materialize(merged, task_dir)
            if parsed.ok:
                text, round_.continuation_used = merged, True
        (version_dir / "reply.txt").write_text(text)
    except PolicyRefusal as exc:
        result.excluded, result.error = EXCLUDED_REFUSAL, str(exc)
        return result
    except EndpointError as exc:
        result.excluded, result.error = EXCLUDED_INFRA, str(exc)
        return result
    await _score_round(round_, task_dir, parsed.files, score, config)
    if round_.infra_error:
        result.excluded = EXCLUDED_INFRA
    return result


async def run_feedback_arm(assignment_id: str, description: str, out_dir: Path, *,
                           seed: ArmResult, seed_dir: Path, complete: CompleteFn,
                           score: ScoreFn, config: ArmConfig | None = None) -> ArmResult:
    """Repair the Prompt arm's environment with environment feedback, up to ``max_repairs`` times.

    ``seed`` is the Prompt arm's result for the same assignment and ``seed_dir`` its directory.
    """
    config = config or ArmConfig()
    result = ArmResult(assignment_id=assignment_id, arm="feedback",
                       rounds=[Round(**asdict(r)) for r in seed.rounds[:1]],
                       excluded=seed.excluded, error=seed.error)
    if seed.excluded or not seed.rounds:
        return result
    base = build_prompt(description, managed_verifier=config.managed_verifier,
                        extra_guidance=config.extra_guidance)
    previous_dir = Path(seed_dir) / "v1"
    last = result.rounds[0]
    for version in range(2, 2 + config.max_repairs):
        if last.admitted or last.infra_error:
            break
        feedback = (env_feedback(ValidationReport(**_report_args(last.report))) if last.report
                    else "VALIDATION FAILED: your answer contained no task files. Emit every file "
                         "of the task in the required format.")
        prompt = repair_prompt(base, _read(previous_dir / "reply.txt"), feedback)
        version_dir = Path(out_dir) / f"v{version}"
        version_dir.mkdir(parents=True, exist_ok=True)
        (version_dir / "prompt.txt").write_text(prompt)
        round_ = Round(version=version, feedback=feedback)
        result.rounds.append(round_)
        try:
            reply = await complete(prompt)
        except PolicyRefusal as exc:
            result.excluded, result.error = EXCLUDED_REFUSAL, str(exc)
            return result
        except EndpointError as exc:
            result.excluded, result.error = EXCLUDED_INFRA, str(exc)
            return result
        round_.usage.append(reply.usage)
        (version_dir / "reply.txt").write_text(reply.text)
        task_dir = version_dir / "task"
        if (previous_dir / "task").is_dir():
            shutil.copytree(previous_dir / "task", task_dir, dirs_exist_ok=True)
        parsed = materialize(reply.text, task_dir)
        files = sorted(set(parsed.files) | set(_list_files(task_dir)))
        await _score_round(round_, task_dir, files, score, config)
        previous_dir, last = version_dir, round_
    if last.infra_error:
        result.excluded = EXCLUDED_INFRA
    return result


async def _score_round(round_: Round, task_dir: Path, files, score: ScoreFn,
                       config: ArmConfig) -> None:
    round_.files = sorted(files)
    if not round_.files:
        round_.rejection = "no_env_generated"
        return
    if config.managed_verifier:
        (task_dir / "tests").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(MANAGED_VERIFIER, task_dir / "tests" / "test.sh")
        (task_dir / "tests" / "test.sh").chmod(0o755)
    round_.task_dir = str(task_dir)
    report = await score(task_dir)
    round_.report = report.to_dict()
    round_.admitted = report.passed
    round_.infra_error = any(c.data.get("infra_error") for c in report.checks)
    if not report.passed:
        round_.rejection = report.failed_checks[0] if report.failed_checks else "unknown"


def _report_args(data: dict[str, Any]) -> dict[str, Any]:
    from autoenvscaling.results import CheckResult
    return {"task_id": data["task_id"], "checks": [CheckResult(**c) for c in data["checks"]]}


def _list_files(root: Path) -> list[str]:
    return [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()]


def _read(path: Path) -> str:
    try:
        return path.read_text()
    except OSError:
        return ""


def write_result(path: Path, result: ArmResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result.to_dict(), indent=2))
    tmp.replace(path)
