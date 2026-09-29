"""The proposer's workspace (Section 3.1 and Appendix "Proposer Setup" of the paper).

Layout inside the sandbox (``/workspace``)::

    assignment/            ASSIGNMENT.md and assignment.json for this episode      fixed
    skills/                create-task, auto_env_scaling, harbor_domains            fixed
    past_rollouts/         the solver's rollouts, refreshed from the pool each round
      index.jsonl            one line per environment: rewards, mean, std, pass rate
      failed/<task>/         attempts that did not pass every test, one JSON each
      successful/<task>/     attempts that passed every test
    past_environments/     previously generated environments (complete Harbor tasks),
      <task>/                refreshed from the pool each round
    memory/                the proposer's notes; persists across rounds, editable
    tools/                 the proposer's tools; persists across rounds, editable
    feedback/              the host's admission results when revising              fixed
    logs/                  reports written by aes-validate and solver_model
    output/tasks/<name>/   the new environment the episode produces

Each attempt file holds the solver trajectory and its verifier output::

    {"task_id", "trajectory_id", "model", "outcome": "failed" | "successful",
     "reward", "tests_passed", "tests_total", "verifier_output", "turns": [...]}

The research code kept rollouts under ``past_trajectories/<env_id>/<solver>/NNN.json`` with an
``index.jsonl`` and per-environment ``env.json``; here attempts are split by outcome and keyed by
task id as in the paper, and the per-solver level is dropped because the workspace only ever
holds the current solver's rollouts.

The host builds this tree locally with ``ProposerWorkspace`` and ``run_episode`` uploads it;
fixed inputs (assignment, skills, feedback) are written by ``run_episode`` itself. Memory and
tools come back in the episode's output directory and the pipeline decides which version to
carry into the next round (``save_persistent`` / ``restore_persistent``).
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from pathlib import Path

from autoenvscaling.proposer.assignment import attempt_reward, is_success, reward_summary
from autoenvscaling.tasks import HarborTask
from autoenvscaling.trajectories.schema import SolverTrajectory

ASSIGNMENT = "assignment"
SKILLS = "skills"
PAST_ROLLOUTS = "past_rollouts"
FAILED = "failed"
SUCCESSFUL = "successful"
PAST_ENVIRONMENTS = "past_environments"
MEMORY = "memory"
TOOLS = "tools"
FEEDBACK = "feedback"
LOGS = "logs"
OUTPUT = "output"
OUTPUT_TASKS = f"{OUTPUT}/tasks"

REFRESHED = (PAST_ROLLOUTS, PAST_ENVIRONMENTS)   # rebuilt from the training pool every round
PERSISTENT = (MEMORY, TOOLS)                     # carried across rounds, proposer-editable
FIXED = (ASSIGNMENT, SKILLS, FEEDBACK)           # written by the host, read-only to the proposer

# Environment files larger than this are left out of past_environments/ (noted in the index).
DEFAULT_MAX_FILE_BYTES = 2_000_000


def safe_name(name: str) -> str:
    """A directory-safe version of a task or trajectory id."""
    cleaned = "".join(c if c.isalnum() or c in "-._" else "_" for c in (name or ""))
    cleaned = cleaned.lstrip(".")
    return cleaned or "unnamed"


def _copy_tree(src: Path, dst: Path, max_file_bytes: int | None = None) -> list[str]:
    """Copy src into dst, skipping symlinks and files over max_file_bytes. Returns skipped paths."""
    skipped: list[str] = []
    for path in sorted(src.rglob("*")):
        rel = path.relative_to(src)
        if "__pycache__" in rel.parts or path.is_symlink():
            continue
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            if max_file_bytes is not None and path.stat().st_size > max_file_bytes:
                skipped.append(str(rel))
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    return skipped


def rollout_record(trajectory: SolverTrajectory) -> dict:
    record = asdict(trajectory)
    turns = record.pop("turns")
    return {
        "task_id": trajectory.task_id,
        "trajectory_id": trajectory.trajectory_id,
        "model": trajectory.model,
        "outcome": SUCCESSFUL if is_success(trajectory) else FAILED,
        "reward": attempt_reward(trajectory),
        "tests_passed": trajectory.tests_passed,
        "tests_total": trajectory.tests_total,
        "verifier_output": trajectory.verifier_output,
        "turns": turns,
    }


class ProposerWorkspace:
    """A host-side proposer workspace directory."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    @classmethod
    def build(cls, root: Path, *, rollouts: Iterable[SolverTrajectory] = (),
              environments: Iterable[HarborTask | Path] = (), state_dir: Path | None = None,
              previous_submission: Path | None = None,
              max_file_bytes: int = DEFAULT_MAX_FILE_BYTES) -> ProposerWorkspace:
        """Build a fresh workspace for one round.

        Args:
            root: Directory to create (existing refreshed content is replaced).
            rollouts: The current solver's rollouts on pool environments.
            environments: Previously generated environments to show as exemplars.
            state_dir: Where the previous round's memory/ and tools/ were saved, if any.
            previous_submission: For a revision, the task the proposer submitted last time; it
                is staged at output/tasks/<name>/ so the proposer can revise it in place.
        """
        ws = cls(root)
        ws.root.mkdir(parents=True, exist_ok=True)
        for sub in (*REFRESHED, *PERSISTENT, LOGS, OUTPUT_TASKS):
            ws.path(sub).mkdir(parents=True, exist_ok=True)
        ws.write_environments(environments, max_file_bytes=max_file_bytes)
        ws.write_rollouts(rollouts, categories=ws.environment_categories())
        if state_dir is not None:
            ws.restore_persistent(state_dir)
        if previous_submission is not None:
            ws.stage_previous_submission(previous_submission)
        return ws

    # ---- refreshed inputs ----------------------------------------------------------------

    def write_rollouts(self, rollouts: Iterable[SolverTrajectory],
                       categories: Mapping[str, str] | None = None) -> int:
        """Replace past_rollouts/ with these rollouts. Returns the number of attempts written."""
        base = self.path(PAST_ROLLOUTS)
        if base.exists():
            shutil.rmtree(base)
        (base / FAILED).mkdir(parents=True)
        (base / SUCCESSFUL).mkdir(parents=True)
        by_task: dict[str, list[SolverTrajectory]] = {}
        written = 0
        for trajectory in rollouts:
            by_task.setdefault(trajectory.task_id, []).append(trajectory)
            if attempt_reward(trajectory) is None:
                continue  # an infrastructure error, not solver behaviour
            outcome = SUCCESSFUL if is_success(trajectory) else FAILED
            task_dir = base / outcome / safe_name(trajectory.task_id)
            task_dir.mkdir(parents=True, exist_ok=True)
            out = task_dir / f"{safe_name(trajectory.trajectory_id)}.json"
            out.write_text(json.dumps(rollout_record(trajectory), indent=2, ensure_ascii=False))
            written += 1
        lines = []
        for task_id in sorted(by_task):
            summary = reward_summary(by_task[task_id])
            row = {"task_id": task_id, **summary,
                   "failed_dir": f"{PAST_ROLLOUTS}/{FAILED}/{safe_name(task_id)}",
                   "successful_dir": f"{PAST_ROLLOUTS}/{SUCCESSFUL}/{safe_name(task_id)}"}
            if categories and task_id in categories:
                row["category"] = categories[task_id]
            if self.path(PAST_ENVIRONMENTS, safe_name(task_id)).is_dir():
                row["environment_dir"] = f"{PAST_ENVIRONMENTS}/{safe_name(task_id)}"
            lines.append(json.dumps(row, ensure_ascii=False))
        (base / "index.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""))
        return written

    def write_environments(self, environments: Iterable[HarborTask | Path],
                           max_file_bytes: int = DEFAULT_MAX_FILE_BYTES) -> int:
        """Replace past_environments/ with copies of these Harbor tasks."""
        base = self.path(PAST_ENVIRONMENTS)
        if base.exists():
            shutil.rmtree(base)
        base.mkdir(parents=True)
        rows = []
        for env in environments:
            task = env if isinstance(env, HarborTask) else HarborTask(Path(env))
            if not task.path.is_dir():
                raise FileNotFoundError(f"no such environment: {task.path}")
            dest = base / safe_name(task.task_id)
            skipped = _copy_tree(task.path, dest, max_file_bytes)
            row: dict = {"task_id": task.task_id, "dir": f"{PAST_ENVIRONMENTS}/{dest.name}"}
            category = _category(task)
            if category:
                row["category"] = category
            if skipped:
                row["omitted_large_files"] = skipped
            rows.append(row)
        (base / "index.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        return len(rows)

    def environment_categories(self) -> dict[str, str]:
        index = self.path(PAST_ENVIRONMENTS, "index.jsonl")
        if not index.is_file():
            return {}
        out = {}
        for line in index.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("category"):
                    out[row["task_id"]] = row["category"]
        return out

    # ---- persistent state ----------------------------------------------------------------

    def restore_persistent(self, state_dir: Path) -> None:
        """Replace memory/ and tools/ with the copies saved in state_dir (if present)."""
        for sub in PERSISTENT:
            src = Path(state_dir) / sub
            if not src.is_dir():
                continue
            dst = self.path(sub)
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst, symlinks=False)

    def save_persistent(self, state_dir: Path) -> None:
        save_persistent(self.root, state_dir)

    # ---- revisions -----------------------------------------------------------------------

    def stage_previous_submission(self, task_dir: Path) -> Path:
        task_dir = Path(task_dir)
        dest = self.path(OUTPUT_TASKS, safe_name(task_dir.name))
        if dest.exists():
            shutil.rmtree(dest)
        _copy_tree(task_dir, dest)
        return dest


def save_persistent(source_root: Path, state_dir: Path) -> None:
    """Copy memory/ and tools/ from a workspace or an episode's output directory to state_dir."""
    for sub in PERSISTENT:
        src = Path(source_root) / sub
        if not src.is_dir():
            continue
        dst = Path(state_dir) / sub
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, symlinks=False)


def _category(task: HarborTask) -> str | None:
    try:
        metadata = task.config.get("metadata") or {}
    except (OSError, ValueError):  # unreadable or invalid task.toml
        return None
    category = metadata.get("category")
    return str(category) if category else None


def find_task_dirs(output_dir: Path) -> list[Path]:
    """Complete Harbor tasks under an episode's output directory, most preferred first.

    The proposer is asked to write ``output/tasks/<name>/``. A task nested one level deeper (as
    ``harbor task init <org>/<name>`` creates) or written directly under ``output/`` is accepted
    too, after tasks in the requested place.
    """
    output_dir = Path(output_dir)
    found: list[Path] = []
    tasks_dir = output_dir / "tasks"
    for base in (tasks_dir, output_dir):
        if not base.is_dir():
            continue
        for child in sorted(p for p in base.iterdir() if p.is_dir()):
            if child == tasks_dir:
                continue
            for candidate in (child, *sorted(p for p in child.iterdir() if p.is_dir())):
                if HarborTask(candidate).is_complete() and candidate not in found:
                    found.append(candidate)
                    break
    return found
