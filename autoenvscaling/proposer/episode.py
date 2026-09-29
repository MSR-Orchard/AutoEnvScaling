"""One proposer episode: assignment in, Harbor task and trajectory out.

An episode (Section 3.1 of the paper):

1. Create the proposer sandbox from ``settings.sandbox`` (image, 16 CPUs, 32 GiB, and a lifetime
   of the 2-hour budget plus a margin for staging and collection).
2. Stage the workspace: upload the host-built workspace (rollouts, past environments, memory,
   tools), copy the skills from the image (or from ``AES_PROPOSER_SKILLS_DIR``), seed ``tools/``
   from the image on the first round, and write the assignment and any host feedback.
3. Build the prompt (the two skills, then the episode instruction, then feedback) and launch the
   harness detached inside the sandbox, under an in-sandbox ``timeout`` of the budget.
4. Poll until the harness exits or the budget runs out (then stop it). There is no turn limit.
5. Collect the native log, ``output/``, ``memory/``, ``tools/`` and ``logs/`` into ``out_dir``,
   parse the log into turns, and return a ``ProposerTrajectory``. Validation, calibration and
   the reward are filled in later by the pipeline.

Files written to ``out_dir``: ``trajectory.json``, ``prompt.md``, ``agent.log`` (native harness
log), ``agent.stderr.txt``, ``episode.json`` (exit status and task candidates), and the
collected ``output/``, ``memory/``, ``tools/`` and ``logs/`` directories.

Environment variables read on the host (candidates for ``Settings``):

    AES_PROPOSER_SECRETS        comma-separated sandbox secret names (Modal Secret names, or
                                host env var names for the Docker backend)
    AES_PROPOSER_BLOCKED_HOSTS  comma-separated hosts to block (default: DEFAULT_BLOCKED_HOSTS)
    AES_BLOCKED_URL_PATTERNS    URL substrings the web tools refuse (forwarded as is)
    AES_PROPOSER_SKILLS_DIR     host directory of skills to stage instead of the image's copy
    AES_PROPOSER_POLL_S         seconds between status polls (default 15)

Credentials are forwarded from the host environment when set: each harness's own
(``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, ``AES_PROPOSER_BASE_URL`` ...) and those the tools
need (``SERPER_API_KEY``, ``AES_SOLVER_*``, ``MODAL_TOKEN_ID``/``MODAL_TOKEN_SECRET`` for child
containers). They are passed with each command rather than baked into the sandbox.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import time
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

from autoenvscaling.config import Settings
from autoenvscaling.proposer import workspace as ws
from autoenvscaling.proposer.assignment import render_assignment
from autoenvscaling.proposer.harnesses import Harness, get_harness
from autoenvscaling.proposer.prompts import PROMPT_SKILLS, WORKSPACE_DESCRIPTION, build_prompt
from autoenvscaling.sandbox.base import Sandbox, SandboxBackend, SandboxSpec, get_backend
from autoenvscaling.tasks import HarborTask
from autoenvscaling.trajectories.schema import Assignment, ProposerTrajectory, Turn

logger = logging.getLogger(__name__)

# Terminal-Bench's site. /etc/hosts has no wildcards, so each name to block is listed.
DEFAULT_BLOCKED_HOSTS = ("tbench.ai", "www.tbench.ai")
# Credentials and settings the in-sandbox tools use, forwarded from the host when set.
TOOL_ENV = ("SERPER_API_KEY", "AES_SOLVER_BASE_URL", "AES_SOLVER_MODEL", "AES_SOLVER_API_KEY",
            "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "MODAL_ENVIRONMENT",
            "AES_BLOCKED_URL_PATTERNS")
STAGING_MARGIN_S = 900        # sandbox lifetime beyond the agent budget
IN_SANDBOX_KILL_AFTER_S = 30  # SIGTERM to SIGKILL grace of the in-sandbox timeout
HOST_GRACE_S = 90             # the host stops the agent itself this long after the budget
DEFAULT_POLL_S = 15.0
MAX_POLL_FAILURES = 8
CONTROL_TIMEOUT_S = 120
LINUX_PATH = "/opt/node/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


class SandboxInfraError(RuntimeError):
    """The sandbox or its control plane failed; not something the proposer did."""


@dataclass(frozen=True)
class EpisodeLayout:
    """Paths inside the proposer sandbox (they match docker/proposer/Dockerfile)."""
    workspace: str = "/workspace"
    run_dir: str = "/tmp/aes-run"
    image_skills: str = "/opt/aes/skills"
    image_tools: str = "/opt/aes/tools"
    base_path: str = LINUX_PATH

    def ws(self, *parts: str) -> str:
        return "/".join((self.workspace.rstrip("/"), *parts))

    def run(self, name: str) -> str:
        return f"{self.run_dir.rstrip('/')}/{name}"


@dataclass
class AgentStatus:
    exit_code: int | None = None
    timed_out: bool = False
    lost: bool = False             # the process vanished without recording an exit code
    elapsed_s: float = 0.0


@dataclass
class EpisodeRecord:
    """Metadata written to out_dir/episode.json."""
    trajectory_id: str
    harness: str
    model: str
    exit_code: int | None
    timed_out: bool
    lost: bool
    harness_completed: bool
    harness_error: str | None
    budget_s: int
    agent_time_s: float
    wall_time_s: float
    task_dir: str | None
    task_candidates: list[str] = field(default_factory=list)
    collection_errors: list[str] = field(default_factory=list)


def _csv(value: str | None) -> list[str]:
    return [p.strip() for p in (value or "").split(",") if p.strip()]


def _q(path: str) -> str:
    return shlex.quote(path)


# ---- shell snippets run inside the sandbox ------------------------------------------------

def start_script(layout: EpisodeLayout, command: str, budget_s: int) -> str:
    """Launch the agent detached in its own session and print its pid.

    Idempotent: a repeated start (for example after a lost control-plane response) prints the
    existing pid instead of launching a second agent.
    """
    run = layout.run_dir
    timed = (f"if command -v timeout >/dev/null 2>&1; then "
             f"exec timeout -s TERM -k {IN_SANDBOX_KILL_AFTER_S} {int(budget_s)} sh -c {_q(command)}; "
             f"else exec sh -c {_q(command)}; fi")
    worker = "\n".join([
        f"cd {_q(layout.workspace)} || exit 97",
        f"sh -c {_q(timed)}",
        "rc=$?",
        f"printf '%s\\n' \"$rc\" > {_q(run + '/rc.tmp')} && mv {_q(run + '/rc.tmp')} {_q(run + '/rc')}",
    ])
    return f"""\
RUN={_q(run)}
mkdir -p "$RUN"
if mkdir "$RUN/started" 2>/dev/null; then
  if command -v setsid >/dev/null 2>&1; then
    setsid sh -c {_q(worker)} > "$RUN/agent.log" 2> "$RUN/agent.err" < /dev/null &
  elif command -v python3 >/dev/null 2>&1; then
    python3 -c 'import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' \\
      sh -c {_q(worker)} > "$RUN/agent.log" 2> "$RUN/agent.err" < /dev/null &
  else
    sh -c {_q(worker)} > "$RUN/agent.log" 2> "$RUN/agent.err" < /dev/null &
  fi
  printf '%s\\n' "$!" > "$RUN/pid.tmp" && mv "$RUN/pid.tmp" "$RUN/pid"
fi
i=0
while [ ! -s "$RUN/pid" ] && [ "$i" -lt 50 ]; do i=$((i + 1)); sleep 0.1; done
cat "$RUN/pid"
"""


def poll_script(layout: EpisodeLayout) -> str:
    return f"""\
RUN={_q(layout.run_dir)}
if [ -s "$RUN/rc" ]; then echo "DONE $(cat "$RUN/rc")"
elif [ -s "$RUN/pid" ] && kill -0 "$(cat "$RUN/pid")" 2>/dev/null; then echo RUNNING
elif [ -s "$RUN/rc" ]; then echo "DONE $(cat "$RUN/rc")"
else echo LOST
fi
"""


def stop_script(layout: EpisodeLayout) -> str:
    return f"""\
PID="$(cat {_q(layout.run('pid'))} 2>/dev/null)"
case "$PID" in ''|*[!0-9]*) exit 0 ;; esac
kill -TERM -- "-$PID" 2>/dev/null || kill -TERM "$PID" 2>/dev/null
sleep 3
kill -KILL -- "-$PID" 2>/dev/null || kill -KILL "$PID" 2>/dev/null
exit 0
"""


def stage_script(layout: EpisodeLayout, *, copy_image_skills: bool) -> str:
    fixed = " ".join(_q(layout.ws(d)) for d in (ws.ASSIGNMENT, ws.FEEDBACK, ws.SKILLS))
    dirs = " ".join(_q(layout.ws(d)) for d in (
        ws.ASSIGNMENT, ws.FEEDBACK, ws.SKILLS, ws.MEMORY, ws.TOOLS, ws.LOGS, ws.OUTPUT_TASKS,
        ws.PAST_ROLLOUTS, ws.PAST_ENVIRONMENTS))
    skills = _q(layout.ws(ws.SKILLS))
    tools = _q(layout.ws(ws.TOOLS))
    lines = [
        "set -e",
        f"chmod -R u+w {fixed} 2>/dev/null || true",
        f"rm -rf {fixed}",
        f"mkdir -p {dirs} {_q(layout.run_dir)}",
    ]
    if copy_image_skills:
        lines.append(f"if [ -d {_q(layout.image_skills)} ]; then "
                     f"cp -R {_q(layout.image_skills)}/. {skills}/; fi")
    lines += [
        # First round: seed the proposer's editable tools from the image.
        (f'if [ -d {_q(layout.image_tools)} ] && [ -z "$(ls -A {tools})" ]; then '
         f"cp -R {_q(layout.image_tools)}/. {tools}/; fi"),
        f"find {tools} -type f ! -name '*.md' -exec chmod a+x {{}} + 2>/dev/null || true",
    ]
    return "\n".join(lines) + "\n"


def seal_script(layout: EpisodeLayout) -> str:
    """Make the fixed inputs read-only. Advisory only: the agent runs as root, but the host
    restages these every episode and never reads them back."""
    fixed = " ".join(_q(layout.ws(d)) for d in (ws.ASSIGNMENT, ws.FEEDBACK, ws.SKILLS))
    return f"chmod -R a-w {fixed} 2>/dev/null || true\n"


# ---- the runner ---------------------------------------------------------------------------

class EpisodeRunner:
    """Runs proposer episodes. Construct once per pipeline and reuse."""

    def __init__(self, settings: Settings, *, backend: SandboxBackend | None = None,
                 layout: EpisodeLayout | None = None, host_env: Mapping[str, str] | None = None,
                 poll_s: float | None = None, skills_dir: Path | None = None,
                 host_grace_s: float = HOST_GRACE_S):
        self.settings = settings
        self.host_grace_s = host_grace_s
        self.layout = layout or EpisodeLayout()
        self.host_env = dict(os.environ if host_env is None else host_env)
        self._backend = backend
        self.poll_s = poll_s or float(self.host_env.get("AES_PROPOSER_POLL_S") or DEFAULT_POLL_S)
        override = skills_dir or self.host_env.get("AES_PROPOSER_SKILLS_DIR")
        self.skills_dir = Path(override) if override else None

    @property
    def backend(self) -> SandboxBackend:
        if self._backend is None:
            self._backend = get_backend(self.settings.sandbox.backend)
        return self._backend

    def blocked_hosts(self) -> list[str]:
        configured = self.host_env.get("AES_PROPOSER_BLOCKED_HOSTS")
        return _csv(configured) if configured is not None else list(DEFAULT_BLOCKED_HOSTS)

    def sandbox_spec(self) -> SandboxSpec:
        sb = self.settings.sandbox
        return SandboxSpec(
            image=sb.proposer_image, cpu=sb.proposer_cpu, memory_gib=sb.proposer_memory_gib,
            timeout_s=sb.proposer_timeout_s + STAGING_MARGIN_S, workdir=self.layout.workspace,
            secrets=_csv(self.host_env.get("AES_PROPOSER_SECRETS")),
            blocked_hosts=self.blocked_hosts())

    def forwarded_env(self, harness: Harness) -> dict[str, str]:
        names = (*harness.credential_env, *TOOL_ENV)
        return {k: self.host_env[k] for k in names if self.host_env.get(k)}

    def tool_env(self) -> dict[str, str]:
        s, lay = self.settings, self.layout
        return {
            "PATH": f"{lay.ws(ws.TOOLS)}:{lay.image_tools}:{lay.base_path}",
            "AES_BLOCKED_HOSTS": ",".join(self.blocked_hosts()),
            "AES_SANDBOX_HARBOR_ENV": s.sandbox.harbor_env,
            "AES_SANDBOX_SOLVER_CPU": str(s.sandbox.solver_cpu),
            "AES_SANDBOX_SOLVER_MEMORY_GIB": str(s.sandbox.solver_memory_gib),
            "AES_VALIDATION_REFERENCE_MIN_REWARD": str(s.validation.reference_min_reward),
            "AES_VALIDATION_EMPTY_MAX_REWARD": str(s.validation.empty_max_reward),
            "AES_CALIBRATION_BAND_LOW": str(s.calibration.band_low),
            "AES_CALIBRATION_BAND_HIGH": str(s.calibration.band_high),
            "AES_CALIBRATION_MIN_REWARD_STD": str(s.calibration.min_reward_std),
            "AES_CALIBRATION_INITIAL_ROLLOUTS": str(s.calibration.initial_rollouts),
        }

    async def run(self, assignment: Assignment, *, harness: str | Harness, model: str,
                  workspace_dir: Path, out_dir: Path, feedback: str | None = None,
                  revision: int = 0, previous_task_dir: Path | None = None) -> ProposerTrajectory:
        """Run one episode (see ``run_episode``).

        ``previous_task_dir``, for a revision, is the task submitted in the previous episode; it
        is staged at ``output/tasks/<name>/`` so the proposer revises it in place. Without it, a
        task already under ``<workspace_dir>/output/tasks/`` plays that role.
        """
        started = time.monotonic()
        lay = self.layout
        impl = harness if isinstance(harness, Harness) else get_harness(harness, model)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        workspace_dir = Path(workspace_dir)
        trajectory_id = f"{assignment.assignment_id}-r{revision}-{uuid.uuid4().hex[:8]}"
        budget_s = int(self.settings.sandbox.proposer_timeout_s)

        env = {**self.forwarded_env(impl), **self.tool_env()}
        prompt_file = lay.run("prompt.md")
        # Prepare before creating the sandbox, so missing credentials fail fast and for free.
        launch = impl.prepare(model=model, prompt_file=prompt_file, run_dir=lay.run_dir,
                              workdir=lay.workspace, env=env)
        exec_env = {**env, **launch.env}

        sandbox = await self.backend.create(self.sandbox_spec())
        status = AgentStatus()
        errors: list[str] = []
        prompt = log = ""
        try:
            await self._stage(sandbox, assignment, workspace_dir, feedback, previous_task_dir)
            skills = await self._read_skills(sandbox)
            previous = (ws.safe_name(Path(previous_task_dir).name) if previous_task_dir
                        else _previous_task(workspace_dir))
            prompt = build_prompt(
                assignment, skills=skills, settings=self.settings, feedback=feedback,
                revision=revision, previous_task=previous if feedback else None,
                web_tools=bool(exec_env.get("SERPER_API_KEY")),
                solver_tool=bool(exec_env.get("AES_SOLVER_BASE_URL")))
            (out_dir / "prompt.md").write_text(prompt)
            await sandbox.write_file(prompt_file, prompt.encode())
            for path, data in launch.files.items():
                await sandbox.write_file(path, data)
            status = await self._run_agent(sandbox, launch.command, exec_env, budget_s)
            log = await self._read_text(sandbox, lay.run("agent.log"), errors)
            stderr = await self._read_text(sandbox, lay.run("agent.err"), errors)
            (out_dir / "agent.log").write_text(impl.compact_log(log))
            (out_dir / "agent.stderr.txt").write_text(stderr[-200_000:])
            for sub in (ws.OUTPUT, ws.MEMORY, ws.TOOLS, ws.LOGS):
                try:
                    await sandbox.download_dir(lay.ws(sub), out_dir / sub)
                except Exception as exc:  # noqa: BLE001  record and keep collecting
                    errors.append(f"download {sub}: {type(exc).__name__}: {exc}")
        finally:
            try:
                await sandbox.terminate()
            except Exception as exc:  # noqa: BLE001  teardown must not mask the result
                logger.warning("could not terminate the proposer sandbox: %s", exc)

        parsed = impl.parse_log(log)
        candidates = ws.find_task_dirs(out_dir / ws.OUTPUT)
        task_dir = candidates[0] if candidates else None
        usage = {**parsed.usage, "exit_code": status.exit_code, "timed_out": status.timed_out,
                 "agent_time_s": round(status.elapsed_s, 1)}
        if parsed.error:
            usage["harness_error"] = parsed.error
        wall = round(time.monotonic() - started, 1)
        trajectory = ProposerTrajectory(
            trajectory_id=trajectory_id, assignment=assignment, model=model, harness=impl.name,
            turns=[Turn(role="user", content=prompt), *parsed.turns],
            task_dir=str(task_dir) if task_dir else None,
            task_digest=HarborTask(task_dir).digest() if task_dir else None,
            revision=revision, wall_time_s=wall, usage=usage)
        record = EpisodeRecord(
            trajectory_id=trajectory_id, harness=impl.name, model=model,
            exit_code=status.exit_code, timed_out=status.timed_out, lost=status.lost,
            harness_completed=parsed.completed, harness_error=parsed.error, budget_s=budget_s,
            agent_time_s=round(status.elapsed_s, 1), wall_time_s=wall,
            task_dir=trajectory.task_dir, task_candidates=[str(c) for c in candidates],
            collection_errors=errors)
        (out_dir / "episode.json").write_text(json.dumps(asdict(record), indent=2))
        (out_dir / "trajectory.json").write_text(trajectory.to_json())
        return trajectory

    # ---- steps ----------------------------------------------------------------------------

    async def _control(self, sandbox: Sandbox, script: str, what: str) -> str:
        result = await sandbox.exec("sh", "-c", script, timeout_s=CONTROL_TIMEOUT_S, workdir="/")
        if not result.ok:
            raise SandboxInfraError(f"{what} failed (rc={result.returncode}): "
                                    f"{(result.stderr or result.stdout).strip()[-1000:]}")
        return result.stdout

    async def _stage(self, sandbox: Sandbox, assignment: Assignment, workspace_dir: Path,
                     feedback: str | None, previous_task_dir: Path | None = None) -> None:
        lay = self.layout
        if workspace_dir.is_dir():
            await sandbox.upload_dir(workspace_dir, lay.workspace)
        await self._control(sandbox, stage_script(lay, copy_image_skills=self.skills_dir is None),
                            "stage workspace")
        if self.skills_dir is not None:
            await sandbox.upload_dir(self.skills_dir, lay.ws(ws.SKILLS))
        if previous_task_dir is not None:
            name = ws.safe_name(Path(previous_task_dir).name)
            await sandbox.upload_dir(Path(previous_task_dir), lay.ws(ws.OUTPUT_TASKS, name))
        await sandbox.write_file(lay.ws(ws.ASSIGNMENT, "assignment.json"),
                                 json.dumps(asdict(assignment), indent=2).encode())
        await sandbox.write_file(lay.ws(ws.ASSIGNMENT, "ASSIGNMENT.md"),
                                 render_assignment(assignment).encode())
        if feedback:
            await sandbox.write_file(lay.ws(ws.FEEDBACK, "FEEDBACK.md"), feedback.encode())
        await sandbox.write_file(lay.ws("WORKSPACE.md"),
                                 f"# Workspace\n\n{WORKSPACE_DESCRIPTION}".encode())
        await self._control(sandbox, seal_script(lay), "seal fixed inputs")

    async def _read_skills(self, sandbox: Sandbox) -> list[tuple[str, str]]:
        skills = []
        for name in PROMPT_SKILLS:
            try:
                text = await sandbox.read_file(self.layout.ws(ws.SKILLS, name, "SKILL.md"))
            except Exception as exc:  # noqa: BLE001  a missing skill is reported, not fatal
                logger.warning("skill %s is not staged in the sandbox: %s", name, exc)
                continue
            skills.append((name, text.decode(errors="replace")))
        return skills

    async def _read_text(self, sandbox: Sandbox, path: str, errors: list[str]) -> str:
        try:
            return (await sandbox.read_file(path)).decode(errors="replace")
        except Exception as exc:  # noqa: BLE001  a missing log is an empty log
            errors.append(f"read {path}: {type(exc).__name__}: {exc}")
            return ""

    async def _run_agent(self, sandbox: Sandbox, command: str, env: dict[str, str],
                         budget_s: int) -> AgentStatus:
        lay = self.layout
        loop = asyncio.get_running_loop()
        result = await sandbox.exec("sh", "-c", start_script(lay, command, budget_s),
                                    timeout_s=CONTROL_TIMEOUT_S, workdir="/", env=env)
        pid = (result.stdout or "").strip().splitlines()[-1:] if result.ok else []
        if not pid or not pid[0].isdigit():
            raise SandboxInfraError(f"could not start the harness (rc={result.returncode}): "
                                    f"{(result.stderr or result.stdout).strip()[-1000:]}")
        t0 = loop.time()
        deadline = t0 + budget_s + self.host_grace_s
        status = AgentStatus()
        failures = 0
        while True:
            try:
                state = (await self._control(sandbox, poll_script(lay), "poll")).strip()
                failures = 0
            except Exception as exc:  # transient control-plane errors are retried
                failures += 1
                if failures > MAX_POLL_FAILURES:
                    raise SandboxInfraError(f"lost contact with the sandbox: {exc}") from exc
                logger.warning("poll failed (%d/%d): %s", failures, MAX_POLL_FAILURES, exc)
                await asyncio.sleep(min(self.poll_s * failures, 120))
                continue
            now = loop.time()
            if state.startswith("DONE"):
                try:
                    status.exit_code = int(state.split()[1])
                except (IndexError, ValueError):
                    status.lost = True
                # 124 (TERM) or 137 (KILL) is the in-sandbox timeout stopping the agent.
                status.timed_out = (status.exit_code in (124, 137)
                                    and now - t0 >= budget_s - self.poll_s)
                break
            if state == "LOST":
                status.lost = True
                break
            if now >= deadline:
                status.timed_out = True
                try:
                    await self._control(sandbox, stop_script(lay), "stop harness")
                except Exception as exc:  # noqa: BLE001  the sandbox is terminated next anyway
                    logger.warning("could not stop the harness cleanly: %s", exc)
                break
            await asyncio.sleep(max(0.05, min(self.poll_s, deadline - now)))
        status.elapsed_s = loop.time() - t0
        return status


def _previous_task(workspace_dir: Path) -> str | None:
    tasks = workspace_dir / ws.OUTPUT_TASKS
    if not tasks.is_dir():
        return None
    names = sorted(p.name for p in tasks.iterdir() if p.is_dir())
    return names[0] if len(names) == 1 else None


async def run_episode(assignment: Assignment, *, harness: str, model: str, settings: Settings,
                      workspace_dir: Path, out_dir: Path, feedback: str | None = None,
                      revision: int = 0,
                      previous_task_dir: Path | None = None) -> ProposerTrajectory:
    """Run the proposer in its sandbox until it submits a task or the time budget runs out.

    `feedback` carries the host's admission-check results when the episode revises an earlier
    submission (at most `settings.calibration.max_revisions` times); stage the earlier task at
    `<workspace_dir>/output/tasks/<name>/` (``ProposerWorkspace.stage_previous_submission``) so
    the proposer revises it in place. `harness` is "claude-code", "codex", "pi", or "auto"
    (chosen from the model name). The returned trajectory has `task_dir` set when a task was
    produced; validation, calibration and reward are filled in later by the pipeline.
    `previous_task_dir` is the task submitted in the previous episode of the same assignment;
    pass it on revisions so the proposer revises its own submission.
    """
    runner = EpisodeRunner(settings)
    return await runner.run(assignment, harness=harness, model=model,
                            workspace_dir=workspace_dir, out_dir=out_dir, feedback=feedback,
                            revision=revision, previous_task_dir=previous_task_dir)
