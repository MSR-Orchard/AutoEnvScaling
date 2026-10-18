"""Rubric review: Harbor's task review (``harbor check``) against the Terminal-Bench rubric.

``harbor check <task> -r <rubric> -a <agent> -m <model> -e <env>`` wraps the task in a review
task, runs the given agent on it in a sandbox, and has the agent write one verdict per rubric
criterion (``pass``, ``fail`` or ``not_applicable``, with an explanation). We call the same code
path through Harbor's Python API (``harbor.analyze.checker.run_checks``), which returns the parsed
report instead of printing it.

Following the paper, the reviewer is the proposer's own model. The shipped Terminal-Bench-derived
rubric omits ``binary_reward`` (our tests give partial credit), and its schema guidance follows
Harbor's current runtime/package fields and free-form metadata. Contributor criteria listed in
``ValidationSettings.rubric_skip`` are filtered out. A task passes only if no applied criterion
is ``fail``.

The rubric setting accepts a name or a path. ``terminal-bench`` resolves to
``rubrics/terminal-bench.toml`` in this repository; ``harbor-default`` uses the rubric that ships
with Harbor (``harbor/cli/quality_checker/default-rubric.toml``); anything else is read as a path.
Criteria named in ``ValidationSettings.rubric_skip`` are removed before the review runs.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomli_w

REPO_RUBRICS = Path(__file__).resolve().parents[2] / "rubrics"
HARBOR_DEFAULT = "harbor-default"


@dataclass(frozen=True)
class ReviewerSpec:
    """The agent and model that review the task (the proposer's own model, per the paper).

    ``agent`` is a Harbor agent name (``claude-code``, ``codex``, ``pi``, ...). For a model served
    behind an OpenAI-compatible endpoint, use ``agent="pi"``, ``model="openai/<name>"``,
    ``agent_kwargs={"model_api": "openai-completions"}`` and put ``OPENAI_API_KEY`` and
    ``OPENAI_BASE_URL`` in ``agent_env``.
    """

    agent: str
    model: str
    agent_kwargs: dict[str, Any] = field(default_factory=dict)
    agent_env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def for_proposer(cls, harness: str, model: str,
                     env: dict[str, str] | None = None) -> "ReviewerSpec":
        """Review with the proposer's own model, through the Harbor agent matching its harness.

        ``claude-code`` and ``codex`` map to Harbor's agents of the same name. ``pi`` reaches the
        model through the proposer's OpenAI-compatible endpoint (``AES_PROPOSER_BASE_URL`` and
        ``AES_PROPOSER_API_KEY``).
        """
        if harness != "pi":
            return cls(agent=harness, model=model)
        env = dict(os.environ if env is None else env)
        base_url = env.get("AES_PROPOSER_BASE_URL", "").strip()
        if not base_url:
            raise ValueError("set AES_PROPOSER_BASE_URL to review with a Pi proposer's model")
        return cls(agent="pi", model=model if model.startswith("openai/") else f"openai/{model}",
                   agent_kwargs={"model_api": "openai-completions"},
                   agent_env={"OPENAI_BASE_URL": base_url,
                              "OPENAI_API_KEY": env.get("AES_PROPOSER_API_KEY", "") or "EMPTY"})

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "ReviewerSpec | None":
        """Read ``AES_REVIEWER_AGENT`` (default ``claude-code``) and ``AES_REVIEWER_MODEL``.

        Returns None when no reviewer model is configured.
        """
        env = dict(os.environ if env is None else env)
        model = env.get("AES_REVIEWER_MODEL", "").strip()
        if not model:
            return None
        return cls(agent=env.get("AES_REVIEWER_AGENT", "claude-code").strip() or "claude-code",
                   model=model)


@dataclass
class CriterionVerdict:
    name: str
    outcome: str  # "pass", "fail" or "not_applicable"
    explanation: str = ""


@dataclass
class RubricResult:
    verdicts: list[CriterionVerdict] = field(default_factory=list)
    error: str | None = None
    cost_usd: float | None = None
    job_dir: Path | None = None

    @property
    def failed(self) -> list[CriterionVerdict]:
        return [v for v in self.verdicts if v.outcome == "fail"]

    @property
    def passed(self) -> bool:
        return self.error is None and bool(self.verdicts) and not self.failed


def resolve_rubric(rubric: str) -> Path | None:
    """Path of the rubric file for a setting value; None means Harbor's built-in default."""
    if rubric == HARBOR_DEFAULT:
        return None
    candidate = Path(rubric).expanduser()
    if candidate.is_file():
        return candidate
    for root in (Path.cwd() / "rubrics", REPO_RUBRICS):
        path = root / f"{rubric}.toml"
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"rubric {rubric!r} not found as a file or as rubrics/{rubric}.toml")


def load_criteria(path: Path | None) -> list[dict[str, Any]]:
    from harbor.cli.quality_checker.models import load_rubric
    return [c.model_dump() for c in load_rubric(path).criteria]


def write_filtered_rubric(path: Path | None, skip: Iterable[str],
                          dest: Path) -> tuple[Path, list[str]]:
    """Write the rubric without the skipped criteria; return its path and the applied names."""
    skip = set(skip)
    criteria = [c for c in load_criteria(path) if c["name"] not in skip]
    if not criteria:
        raise ValueError("every rubric criterion was skipped; nothing left to review")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(tomli_w.dumps({"criteria": criteria}))
    return dest, [c["name"] for c in criteria]


CheckFn = Callable[..., Awaitable[tuple[Any, Path]]]


async def review_task(task_dir: Path, reviewer: ReviewerSpec, *, rubric: str,
                      skip: Iterable[str], jobs_dir: Path, env_type: str = "modal",
                      run_checks: CheckFn | None = None) -> RubricResult:
    """Run ``harbor check`` on one task and return one verdict per applied criterion."""
    from harbor.models.environment_type import EnvironmentType

    if run_checks is None:
        from harbor.analyze.checker import run_checks
    jobs_dir = Path(jobs_dir)
    rubric_path, applied = write_filtered_rubric(resolve_rubric(rubric), skip,
                                                 jobs_dir / "rubric.toml")
    report, job_dir = await run_checks(
        path=Path(task_dir),
        agent=reviewer.agent,
        model=reviewer.model,
        rubric_path=rubric_path,
        environment=EnvironmentType(env_type),
        n_concurrent=1,
        n_attempts=1,
        jobs_dir=jobs_dir,
        agent_kwargs=dict(reviewer.agent_kwargs) or None,
        agent_env=dict(reviewer.agent_env) or None,
        quiet=True,
    )
    return parse_report(report, applied, job_dir=job_dir)


def parse_report(report: Any, applied: list[str], *, job_dir: Path | None = None) -> RubricResult:
    """Read a Harbor ``CheckReport`` for a single task."""
    results = list(getattr(report, "results", []) or [])
    if len(results) != 1:
        return RubricResult(error=f"expected one check result, got {len(results)}",
                            job_dir=job_dir)
    result = results[0]
    if result.error:
        return RubricResult(error=str(result.error), cost_usd=result.cost_usd, job_dir=job_dir)
    verdicts = []
    for name in applied:
        check = result.checks.get(name)
        if check is None:
            return RubricResult(error=f"reviewer returned no verdict for {name!r}",
                                cost_usd=result.cost_usd, job_dir=job_dir)
        outcome = getattr(check.outcome, "value", check.outcome)
        verdicts.append(CriterionVerdict(name=name, outcome=str(outcome),
                                         explanation=check.explanation))
    return RubricResult(verdicts=verdicts, cost_usd=result.cost_usd, job_dir=job_dir)


def rubric_file_criteria(rubric: str) -> list[str]:
    """Names of the criteria in a rubric setting, before skipping (for docs and tests)."""
    path = resolve_rubric(rubric)
    if path is None:
        return [c["name"] for c in load_criteria(None)]
    with open(path, "rb") as fh:
        return [c["name"] for c in tomllib.load(fh)["criteria"]]
