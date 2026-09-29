"""Terminal-Bench 2.1 and 4.0 through Harbor, with the Pi agent and avg@k.

The paper evaluates on held-out Terminal-Bench 2.1 and on Terminal-Bench 4.0, reporting avg@5:
every task is attempted five times and the score is the mean verifier reward over all
task-attempts (Appendix G.3). Solvers run in the Pi coding agent in sandboxes with 4 CPUs and
8 GiB of memory. Model cards report higher numbers for the same models with a different harness
and larger sandboxes; the comparisons in the paper hold the harness and resources fixed.

``build_job_config`` turns ``EvalSettings`` into a Harbor ``JobConfig`` (k attempts per task,
Harbor retries for infrastructure errors), ``run_eval`` runs it, and ``report`` reads the trial
results back from disk and scores them with ``autoenvscaling.evals.scoring``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from autoenvscaling.evals.endpoint import EndpointSpec
from autoenvscaling.evals.harbor_results import load_attempts
from autoenvscaling.evals.pi_agent import PI_AGENT_IMPORT, PI_VERSION
from autoenvscaling.evals.scoring import summarize


@dataclass(frozen=True)
class Benchmark:
    key: str
    title: str
    dataset: str          # Harbor dataset: "name@version" (registry) or "org/name@ref" (package)
    note: str = ""


BENCHMARKS: dict[str, Benchmark] = {
    "tb2.1": Benchmark(
        key="tb2.1", title="Terminal-Bench 2.1", dataset="terminal-bench/terminal-bench-2-1",
        note="Held out from training; the contamination check compares generated instructions "
             "against these tasks (13-gram overlap)."),
    "tb4": Benchmark(
        key="tb4", title="Terminal-Bench 4.0", dataset="terminal-bench/terminal-bench@4.0.0",
        note="Some tasks need Docker Compose or GPUs; the Harbor environment must support them "
             "or those tasks end as infrastructure errors."),
}

# Harbor exceptions that are the model's outcome, not the infrastructure's, so retrying them
# would bias the score toward lucky reruns.
NO_RETRY_EXCEPTIONS = {
    "AgentTimeoutError",
    "VerifierTimeoutError",
    "AgentSafetyRefusalError",
    "AgentAuthenticationError",
    "ModelNotFoundError",
    "ContextWindowExceededError",
    "OutputTokenExceededError",
    "ApiUsageLimitError",
}


@dataclass
class EvalSettings:
    benchmark: str = "tb2.1"
    endpoint: EndpointSpec = field(default_factory=lambda: EndpointSpec(model=""))
    k: int = 5                              # attempts per task; the paper reports avg@5
    harbor_env: str = "modal"               # Harbor environment type
    n_concurrent: int = 16
    cpus: int | None = 4                    # Appendix G.3: 4 CPUs and 8 GiB per solver sandbox
    memory_gib: int | None = 8
    agent_timeout_sec: float | None = None  # None keeps each task's declared timeout
    max_retries: int = 2                    # infrastructure retries; never increase k
    jobs_dir: Path = Path("runs/evals")
    job_name: str | None = None
    task_names: list[str] | None = None     # glob patterns
    exclude_task_names: list[str] | None = None
    n_tasks: int | None = None
    dataset_override: str | None = None     # a local task directory or another dataset spec
    tools: str | None = "bash"
    pi_version: str = PI_VERSION

    @property
    def spec(self) -> Benchmark:
        try:
            return BENCHMARKS[self.benchmark]
        except KeyError:
            raise ValueError(f"unknown benchmark {self.benchmark!r}; "
                             f"expected one of {sorted(BENCHMARKS)}") from None

    @property
    def dataset(self) -> str:
        return self.dataset_override or self.spec.dataset

    def default_job_name(self) -> str:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        model = self.endpoint.model.replace("/", "_") or "model"
        return f"{self.benchmark}-{model}-k{self.k}-{stamp}"


def dataset_config(spec: str, *, task_names: list[str] | None = None,
                   exclude_task_names: list[str] | None = None,
                   n_tasks: int | None = None) -> Any:
    """Harbor ``DatasetConfig`` for a dataset spec, parsed the way ``harbor run -d`` does.

    ``name@version`` is a registry dataset, ``org/name@ref`` a package dataset, and an
    existing directory a local dataset of task directories.
    """
    from harbor.models.job.config import DatasetConfig

    filters = {"task_names": task_names, "exclude_task_names": exclude_task_names,
               "n_tasks": n_tasks}
    path = Path(spec).expanduser()
    if path.is_dir():
        return DatasetConfig(path=path, **filters)
    name, _, version = spec.partition("@")
    if "/" in name:
        return DatasetConfig(name=name, ref=version or "latest", **filters)
    return DatasetConfig(name=name, version=version or None, **filters)


def agent_config(settings: EvalSettings, *, import_path: str = PI_AGENT_IMPORT,
                 extra_kwargs: dict[str, Any] | None = None) -> Any:
    """Harbor ``AgentConfig`` for the Pi agent against the settings' endpoint."""
    from harbor.models.trial.config import AgentConfig

    ep = settings.endpoint
    if not ep.model:
        raise ValueError("EvalSettings.endpoint.model must be the exact served model id")
    kwargs: dict[str, Any] = {
        "base_url_env": ep.base_url_env,
        "api_key_env": ep.api_key_env,
        "api": ep.api,
        "context_window": ep.context_window,
        "max_output_tokens": ep.max_output_tokens,
        "sampling": ep.sampling,
        "provider_routing": ep.provider_routing,
        "tools": settings.tools,
        "pi_version": settings.pi_version,
        "agent_timeout_sec": settings.agent_timeout_sec,
        **(extra_kwargs or {}),
    }
    # Harbor's own timeout is set a little above Pi's, so Pi's clean exit (and the verifier run
    # on the state it left) always wins the race.
    override = (settings.agent_timeout_sec + 120) if settings.agent_timeout_sec else None
    return AgentConfig(
        import_path=import_path,
        model_name=ep.model,
        override_timeout_sec=override,
        override_setup_timeout_sec=1200,
        env={ep.api_key_env: "${" + ep.api_key_env + "}"},
        kwargs=kwargs,
    )


def build_job_config(settings: EvalSettings) -> Any:
    from harbor.models.environment_type import EnvironmentType
    from harbor.models.job.config import JobConfig, RetryConfig
    from harbor.models.trial.config import EnvironmentConfig

    if settings.k < 1:
        raise ValueError("k must be at least 1")
    return JobConfig(
        job_name=settings.job_name or settings.default_job_name(),
        jobs_dir=settings.jobs_dir,
        n_attempts=settings.k,
        n_concurrent_trials=settings.n_concurrent,
        retry=RetryConfig(max_retries=settings.max_retries,
                          exclude_exceptions=set(NO_RETRY_EXCEPTIONS),
                          min_wait_sec=5, max_wait_sec=60),
        environment=EnvironmentConfig(
            type=EnvironmentType(settings.harbor_env),
            delete=True,
            override_cpus=settings.cpus,
            override_memory_mb=settings.memory_gib * 1024 if settings.memory_gib else None,
        ),
        agents=[agent_config(settings)],
        datasets=[dataset_config(settings.dataset, task_names=settings.task_names,
                                 exclude_task_names=settings.exclude_task_names,
                                 n_tasks=settings.n_tasks)],
    )


def manifest(settings: EvalSettings) -> dict[str, Any]:
    """What was run, written next to the Harbor job so the report can say it."""
    return {
        "benchmark": settings.benchmark,
        "title": settings.spec.title,
        "dataset": settings.dataset,
        "k": settings.k,
        "harbor_env": settings.harbor_env,
        "resources": {"cpus": settings.cpus, "memory_gib": settings.memory_gib},
        "agent": {"name": "pi", "version": settings.pi_version, "tools": settings.tools,
                  "skills": False, "compaction": False,
                  "agent_timeout_sec": settings.agent_timeout_sec},
        "endpoint": settings.endpoint.provenance(),
    }


async def run_eval(settings: EvalSettings) -> Path:
    """Run the evaluation job and return its directory. Costs money: sandboxes and tokens."""
    from harbor.job import Job

    config = build_job_config(settings)
    job = await Job.create(config)
    job_dir = Path(job.job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "aes_manifest.json").write_text(json.dumps(manifest(settings), indent=2) + "\n")
    await job.run()
    return job_dir


def report(job_dirs: list[Path], k: int, *, missing: str = "exclude",
           benchmark: str | None = None) -> dict[str, Any]:
    """Score one or more Harbor job directories (for example a run and its rerun)."""
    attempts = load_attempts(job_dirs)
    result = summarize(attempts, k, missing=missing)  # type: ignore[arg-type]
    manifests = []
    for job_dir in job_dirs:
        path = Path(job_dir) / "aes_manifest.json"
        if path.is_file():
            manifests.append(json.loads(path.read_text()))
    return {
        "benchmark": benchmark or (manifests[0]["benchmark"] if manifests else None),
        "job_dirs": [str(p) for p in job_dirs],
        "manifests": manifests,
        "metric": f"avg@{k}",
        **result.to_dict(),
    }


def format_report(rep: dict[str, Any]) -> str:
    def pct(v: float | None) -> str:
        return "n/a" if v is None else f"{100 * v:.1f}"

    k = rep["k"]
    lines = [
        (f"{rep.get('benchmark') or 'benchmark'}: avg@{k} = {pct(rep['avg_at_k'])}, "
         f"pass@{k} = {pct(rep['pass_at_k'])}"),
        (f"tasks scored {rep['n_tasks_scored']}/{rep['n_tasks']} "
         f"(incomplete {rep['n_tasks_incomplete']}, missing attempts: {rep['missing']})"),
        (f"attempts scored {rep['n_attempts_scored']}; infrastructure errors "
         f"{rep['n_infra_errors']} (retried, never scored); verifier broken "
         f"{rep['n_verifier_broken']}; timed out {rep['n_timed_out']}; context limited "
         f"{rep['n_context_limited']}"),
    ]
    if rep.get("per_attempt_means"):
        runs = ", ".join(pct(v) for v in rep["per_attempt_means"])
        lines.append(f"per-attempt means: {runs}")
    return "\n".join(lines)


def settings_to_dict(settings: EvalSettings) -> dict[str, Any]:
    data = asdict(settings)
    data["jobs_dir"] = str(settings.jobs_dir)
    return data
