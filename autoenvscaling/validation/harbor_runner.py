"""Run Harbor trials for the admission checks and read back per-trial rewards.

Every check that executes a task goes through this module:

* the reference solution runs with Harbor's ``oracle`` agent (it executes ``solution/solve.sh``),
* the empty solution runs with Harbor's ``nop`` agent (it does nothing),
* solver attempts run with Harbor's ``pi`` agent against an OpenAI-compatible endpoint.

How Harbor records rewards: after the agent finishes, Harbor uploads ``tests/`` and runs
``tests/test.sh``, which writes ``/logs/verifier/reward.txt`` (one number) or
``/logs/verifier/reward.json`` (a flat name-to-number mapping). Harbor parses that file into
``verifier_result.rewards`` in the trial's ``result.json``; a ``reward.txt`` becomes
``{"reward": value}``. We read the ``reward`` key. Our tasks score partial credit, so the verifier
is expected to write the fraction of tests that pass (r_S in the paper). When the verifier also
writes a CTRF report (``/logs/verifier/ctrf.json``, the Terminal-Bench convention) or a JUnit XML
report, we read the passed and total test counts from it as well.

Each trial ends in one of three states, and the checks depend on telling them apart:

* scored: the verifier produced a finite reward;
* a task failure: the image did not build, or the verifier crashed or timed out. This is the
  proposer's fault and counts against the task;
* an infrastructure failure: the sandbox, the model endpoint, or the network failed, so the outcome
  is unknown. These are reported as ``infra_error`` and never scored as a task failure.
"""

from __future__ import annotations

import json
import logging
import math
import re
import shutil
import subprocess
import sys
import tomllib
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import tomli_w

logger = logging.getLogger(__name__)

ORACLE_AGENT = "oracle"
NOP_AGENT = "nop"
PI_AGENT = "pi"

# Agent timeouts used by the old host gates (episode_loop.ORACLE_TIMEOUT_SEC and
# SOLVE_AGENT_TIMEOUT_SEC); the reference and empty solutions share the shorter one.
REFERENCE_TIMEOUT_SEC = 600.0
SOLVER_TIMEOUT_SEC = 1200.0
# Build budget applied when a task does not set one (old _stage_harbor_task default).
DEFAULT_BUILD_TIMEOUT_SEC = 1200.0
# Harbor retries trials that raise; the old gates used three retries.
DEFAULT_MAX_RETRIES = 3
VERIFIER_OUTPUT_CHARS = 20_000

# Exceptions that mean the task itself is broken, so Harbor should not retry them.
BUILD_ERROR_TYPES = frozenset({"ImageBuildError", "SandboxBuildFailedError",
                               "EnvironmentStartTimeoutError"})
VERIFIER_TASK_ERROR_TYPES = frozenset({"RewardFileNotFoundError", "RewardFileEmptyError",
                                       "VerifierOutputParseError", "VerifierTimeoutError"})
# Agent exceptions that are genuine solver outcomes: the verifier still scores the container.
AGENT_OUTCOME_ERROR_TYPES = frozenset({"AgentTimeoutError", "ContextWindowExceededError",
                                       "OutputTokenExceededError", "AgentSafetyRefusalError"})
# Agent exceptions that mean the agent never validly ran (endpoint, credentials, setup).
AGENT_INFRA_ERROR_TYPES = frozenset({
    "AgentSetupTimeoutError", "AgentAuthenticationError", "ModelNotFoundError",
    "NetworkConnectionError", "ApiError", "ApiRateLimitError", "ApiUsageLimitError",
    "ApiInternalServerError", "ApiOverloadedError", "ApiConnectionClosedError",
    "ApiResponseStalledError", "UnknownApiError", "ApiProviderResourceNotFoundError",
})
ENVIRONMENT_INFRA_ERROR_TYPES = frozenset({
    "AuthError", "ConnectionError", "InternalError", "ServiceError", "ResourceExhaustedError",
    "InternalFailure", "SandboxTerminatedError", "AddTestsDirError", "DownloadVerifierDirError",
})
# Message signatures of infrastructure failures, ported from calibration_loop._INFRA_PATTERNS and
# _INFRA_TRIAL_SIGNATURES.
INFRA_PATTERN = re.compile(
    r"\b(?:429|500|502|503|504)\b"
    r"|image ?pull|imagepull|manifest unknown|pull ?access"
    r"|sandbox.*(?:create|creation).*(?:timeout|fail)"
    r"|ConnectionError|Connection reset|Connection refused|Name or service not known"
    r"|Temporary failure|EOF occurred|read timed out|ServiceUnavailable|external shut-down"
    r"|No API key found|/login to log into a provider|too many requests|rate limit",
    re.IGNORECASE,
)
_BUILD_HINT = re.compile(r"\bbuild\b|dockerfile", re.IGNORECASE)


@dataclass(frozen=True)
class AgentSpec:
    """Which Harbor agent a trial runs, and how."""

    name: str
    model_name: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    timeout_sec: float | None = None
    extra_allowed_hosts: tuple[str, ...] = ()

    @classmethod
    def reference(cls, timeout_sec: float = REFERENCE_TIMEOUT_SEC) -> "AgentSpec":
        """The proposer's reference solution (Harbor's oracle agent runs solution/solve.sh)."""
        return cls(name=ORACLE_AGENT, timeout_sec=timeout_sec)

    @classmethod
    def empty(cls, timeout_sec: float = REFERENCE_TIMEOUT_SEC) -> "AgentSpec":
        """The empty solution: Harbor's nop agent leaves the container untouched."""
        return cls(name=NOP_AGENT, timeout_sec=timeout_sec)


@dataclass
class TrialOutcome:
    """What one Harbor trial produced."""

    trial_name: str
    trial_dir: Path | None = None
    reward: float | None = None
    rewards: dict[str, float] = field(default_factory=dict)
    tests_passed: int | None = None
    tests_total: int | None = None
    exception_type: str | None = None
    exception_message: str = ""
    failed_phase: str | None = None  # "setup", "environment", "agent_setup", "agent", "verifier"
    infra_error: bool = False
    verifier_output: str = ""

    @property
    def scored(self) -> bool:
        return self.reward is not None and not self.infra_error

    @property
    def build_failed(self) -> bool:
        return self.failed_phase == "environment" and not self.infra_error

    @property
    def error_summary(self) -> str:
        if not self.exception_type:
            return ""
        return f"{self.exception_type}: {self.exception_message}".strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_name": self.trial_name,
            "trial_dir": str(self.trial_dir) if self.trial_dir else None,
            "reward": self.reward,
            "tests_passed": self.tests_passed,
            "tests_total": self.tests_total,
            "exception_type": self.exception_type,
            "exception_message": self.exception_message[-2000:],
            "failed_phase": self.failed_phase,
            "infra_error": self.infra_error,
        }


@dataclass
class RunResult:
    """All trials of one Harbor job, plus where Harbor wrote them."""

    outcomes: list[TrialOutcome]
    job_dir: Path | None = None
    staged_task: Path | None = None

    @property
    def scored(self) -> list[TrialOutcome]:
        return [o for o in self.outcomes if o.scored]

    @property
    def rewards(self) -> list[float]:
        return [o.reward for o in self.scored if o.reward is not None]


class TrialRunner(Protocol):
    """Runs a task directory with one agent n times. HarborRunner is the real implementation."""

    async def run(self, task_dir: Path, agent: AgentSpec, *, n_attempts: int = 1,
                  label: str = "trial", instruction: str | None = None,
                  public_network: bool = False,
                  allowed_hosts: tuple[str, ...] = ()) -> RunResult: ...


class HarborRunner:
    """Runs trials through Harbor's Python API (the same code path as ``harbor run``).

    Each call stages a copy of the task under ``jobs_root/<label>/<run id>/task``, applies the
    requested overrides to the copy, and runs one Harbor job in ``jobs_root/<label>/<run id>``.
    The job directory is kept so every verdict can be inspected afterwards.
    """

    def __init__(self, jobs_root: Path, *, env_type: str = "modal", max_concurrent: int = 8,
                 max_retries: int = DEFAULT_MAX_RETRIES, quiet: bool = True,
                 environment_kwargs: dict[str, Any] | None = None) -> None:
        self.jobs_root = Path(jobs_root)
        self.env_type = env_type
        self.max_concurrent = max(1, max_concurrent)
        self.max_retries = max_retries
        self.quiet = quiet
        self.environment_kwargs = dict(environment_kwargs or {})

    async def run(self, task_dir: Path, agent: AgentSpec, *, n_attempts: int = 1,
                  label: str = "trial", instruction: str | None = None,
                  public_network: bool = False,
                  allowed_hosts: tuple[str, ...] = ()) -> RunResult:
        from harbor.job import Job

        if n_attempts < 1:
            raise ValueError(f"n_attempts must be at least 1, got {n_attempts}")
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        run_dir = self.jobs_root / _safe_name(label) / run_id
        staged = stage_task(Path(task_dir), run_dir / "task", instruction=instruction,
                            public_network=public_network, allowed_hosts=allowed_hosts)
        config = self._job_config(staged, agent, n_attempts=n_attempts, jobs_dir=run_dir,
                                  job_name="job")
        job = await Job.create(config)
        job_result = await job.run()
        outcomes = []
        for trial_result in job_result.trial_results:
            trial_dir = job.job_dir / trial_result.trial_name
            outcomes.append(parse_trial(trial_result.model_dump(mode="json"), trial_dir))
        return RunResult(outcomes=outcomes, job_dir=job.job_dir, staged_task=staged)

    def _job_config(self, task_dir: Path, agent: AgentSpec, *, n_attempts: int, jobs_dir: Path,
                    job_name: str):
        from harbor.models.environment_type import EnvironmentType
        from harbor.models.job.config import JobConfig, RetryConfig
        from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig

        agent_fields: dict[str, Any] = {"name": agent.name, "kwargs": dict(agent.kwargs),
                                        "env": dict(agent.env)}
        if agent.model_name:
            agent_fields["model_name"] = agent.model_name
        if agent.timeout_sec:
            agent_fields["override_timeout_sec"] = float(agent.timeout_sec)
        if agent.extra_allowed_hosts:
            agent_fields["extra_allowed_hosts"] = list(agent.extra_allowed_hosts)
        retry = RetryConfig(max_retries=self.max_retries)
        retry.exclude_exceptions = set(retry.exclude_exceptions or set()) | BUILD_ERROR_TYPES
        return JobConfig(
            job_name=job_name,
            jobs_dir=jobs_dir,
            n_attempts=n_attempts,
            n_concurrent_trials=min(n_attempts, self.max_concurrent),
            quiet=self.quiet,
            retry=retry,
            environment=EnvironmentConfig(type=EnvironmentType(self.env_type),
                                          kwargs=self.environment_kwargs),
            agents=[AgentConfig(**agent_fields)],
            tasks=[TaskConfig(path=task_dir)],
        )


def harbor_cli() -> str:
    """The ``harbor`` executable: the one installed next to this Python if present, else PATH."""
    candidate = Path(sys.executable).parent / "harbor"
    return str(candidate) if candidate.is_file() else "harbor"


def stage_task(task_dir: Path, dest: Path, *, instruction: str | None = None,
               public_network: bool = False, allowed_hosts: tuple[str, ...] = ()) -> Path:
    """Copy a task for one run and apply run-specific overrides to the copy only.

    * ``[task].name`` without an ``org/`` prefix gets ``aes/`` prepended: Harbor requires the
      ``org/name`` form, and the name is run metadata rather than part of the environment.
    * ``[environment].build_timeout_sec`` defaults to 1200 seconds when the task leaves it unset.
    * ``instruction`` replaces instruction.md (used by the hint probe).
    * ``allowed_hosts`` switches the trial to Harbor's allowlist network mode and adds these hosts
      to any the task already allows. Solver attempts use it: Pi runs inside the task container
      and must reach the model endpoint and the hosts it installs from, and nothing else.
    * ``public_network`` opens the network for the whole trial instead (opt-in).
    The reference and empty solutions run with the task's own network settings.
    """
    task_dir = Path(task_dir)
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(task_dir, dest, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    config_path = dest / "task.toml"
    config: dict[str, Any] = tomllib.loads(config_path.read_text()) if config_path.is_file() else {}
    normalize_task_name(config)
    environment = config.setdefault("environment", {})
    environment.setdefault("build_timeout_sec", DEFAULT_BUILD_TIMEOUT_SEC)
    if public_network:
        environment["network_mode"] = "public"
        environment.pop("allowed_hosts", None)
        environment.pop("allow_internet", None)
        agent_section = config.get("agent")
        if isinstance(agent_section, dict):
            agent_section.pop("network_mode", None)
            agent_section.pop("allowed_hosts", None)
    elif allowed_hosts:
        existing = (environment.get("allowed_hosts") or []
                    if environment.get("network_mode") == "allowlist" else [])
        environment["network_mode"] = "allowlist"
        environment["allowed_hosts"] = sorted(set(existing) | set(allowed_hosts))
        environment.pop("allow_internet", None)
    config_path.write_text(tomli_w.dumps(config))
    if instruction is not None:
        (dest / "instruction.md").write_text(instruction)
    return dest


def normalize_task_name(config: dict[str, Any]) -> dict[str, Any]:
    """Prefix ``aes/`` to a ``[task].name`` that lacks Harbor's required ``org/`` part."""
    task_section = config.get("task")
    if isinstance(task_section, dict):
        name = task_section.get("name")
        if isinstance(name, str) and name and "/" not in name:
            task_section["name"] = f"aes/{_safe_name(name)}"
    return config


def parse_trial(data: dict[str, Any], trial_dir: Path | None) -> TrialOutcome:
    """Turn one Harbor ``result.json`` (as a dict) plus its trial directory into a TrialOutcome."""
    exception = data.get("exception_info") or {}
    exception_type = exception.get("exception_type")
    exception_message = str(exception.get("exception_message") or "")
    raw_rewards = (data.get("verifier_result") or {}).get("rewards") or {}
    rewards = {k: float(v) for k, v in raw_rewards.items() if _is_finite_number(v)}
    reward = rewards.get("reward")
    if reward is None and len(rewards) == 1:
        reward = next(iter(rewards.values()))

    passed = total = None
    verifier_output = ""
    if trial_dir is not None:
        passed, total = read_test_counts(trial_dir / "verifier")
        verifier_output = _tail(trial_dir / "verifier" / "test-stdout.txt", VERIFIER_OUTPUT_CHARS)

    phase = _failed_phase(data, exception_type)
    infra = _is_infra(phase, exception_type, exception_message, has_reward=reward is not None)
    if infra:
        reward = None
    return TrialOutcome(
        trial_name=str(data.get("trial_name") or (trial_dir.name if trial_dir else "")),
        trial_dir=trial_dir,
        reward=reward,
        rewards=rewards,
        tests_passed=passed,
        tests_total=total,
        exception_type=exception_type,
        exception_message=exception_message,
        failed_phase=phase,
        infra_error=infra,
        verifier_output=verifier_output,
    )


def _failed_phase(data: dict[str, Any], exception_type: str | None) -> str | None:
    if not exception_type:
        return None
    if exception_type in VERIFIER_TASK_ERROR_TYPES:
        return "verifier"
    if data.get("environment_setup") is None:
        return "setup"
    if data.get("agent_setup") is None:
        return "environment"
    if data.get("agent_execution") is None:
        return "agent_setup"
    return "agent"


def _is_infra(phase: str | None, exception_type: str | None, message: str, *,
              has_reward: bool) -> bool:
    if phase is None or phase == "verifier":
        return False
    if exception_type in AGENT_OUTCOME_ERROR_TYPES:
        return False
    if exception_type in BUILD_ERROR_TYPES:
        # A build that fails or exceeds its budget is the proposer's fault (the old Exp1 gate
        # classified an exhausted build budget the same way); a builder shut down from outside
        # is not.
        return "external shut-down" in message
    if exception_type in AGENT_INFRA_ERROR_TYPES or exception_type in ENVIRONMENT_INFRA_ERROR_TYPES:
        return True
    if INFRA_PATTERN.search(f"{exception_type}: {message}"):
        return True
    if phase == "environment":
        # Anything we cannot recognise as a build failure is an unknown (infrastructure) outcome.
        return not _BUILD_HINT.search(message)
    # An agent that exited abnormally is still a solver outcome if the verifier graded the
    # container. Setup errors, and agent errors that left nothing graded, have an unknown outcome.
    return not (phase == "agent" and has_reward)


_MODAL_IMAGE_LOGS = re.compile(r"modal image logs (im-[A-Za-z0-9]+)")


def build_log_tail(outcome: TrialOutcome, limit: int = 12_000) -> str:
    """Best-effort build log for a failed build: the error, Modal's image log, the trial log.

    Modal reports a failed image build as "Image build for im-... failed" and keeps the log on its
    side, so we fetch it with ``modal image logs <id>`` when the error names an image. The 12,000
    character limit matches the build-log tail the old feedback lane returned to the model.
    """
    parts = [outcome.error_summary]
    match = _MODAL_IMAGE_LOGS.search(outcome.exception_message)
    if match:
        modal_bin = Path(sys.executable).parent / "modal"
        cmd = [str(modal_bin) if modal_bin.is_file() else "modal", "image", "logs", match.group(1)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
            if proc.stdout.strip():
                parts.append("Build log:\n" + proc.stdout[-limit:])
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning("could not fetch Modal build log %s: %s", match.group(1), exc)
    if outcome.trial_dir is not None:
        trial_log = _tail(outcome.trial_dir / "trial.log", 4_000)
        if trial_log.strip():
            parts.append("Trial log (tail):\n" + trial_log)
    return "\n\n".join(p for p in parts if p)[-limit:]


def read_test_counts(verifier_dir: Path) -> tuple[int | None, int | None]:
    """(passed, total) from a CTRF report, else a JUnit XML report, else (None, None).

    Skipped tests do not count as passed, matching the old managed verifier template.
    """
    ctrf = verifier_dir / "ctrf.json"
    if ctrf.is_file():
        try:
            summary = json.loads(ctrf.read_text())["results"]["summary"]
            return int(summary.get("passed", 0)), int(summary.get("tests", 0))
        except (ValueError, KeyError, TypeError):
            pass
    junit = verifier_dir / "junit.xml"
    if junit.is_file():
        try:
            root = ET.parse(junit).getroot()
            suite = root if root.tag == "testsuite" else root.find("testsuite")
            if suite is not None:
                total = int(suite.get("tests", 0))
                bad = sum(int(suite.get(k, 0)) for k in ("failures", "errors", "skipped"))
                return max(total - bad, 0), total
        except (ET.ParseError, ValueError):
            pass
    return None, None


def _is_finite_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _tail(path: Path, limit: int) -> str:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return ""
    return text[-limit:]


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-.") or "task"
