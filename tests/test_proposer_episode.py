"""run_episode end to end, with a local stand-in for the sandbox and a scripted fake agent.

The stand-in backend runs every sandbox command on this machine, and the sandbox paths point
into a temporary directory, so the real staging, detached launch, polling, time budget and
collection logic all run. Only the coding agent is fake: a script that reads the workspace,
writes a Harbor task, and prints a Claude Code stream-json log.
"""

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from autoenvscaling.config import Settings
from autoenvscaling.proposer.episode import (
    DEFAULT_BLOCKED_HOSTS,
    STAGING_MARGIN_S,
    EpisodeLayout,
    EpisodeRunner,
)
from autoenvscaling.proposer.harnesses import ClaudeCodeHarness, HarnessLaunch
from autoenvscaling.proposer.workspace import ProposerWorkspace
from autoenvscaling.sandbox.base import ExecResult, SandboxSpec
from autoenvscaling.tasks import HarborTask
from autoenvscaling.trajectories.schema import (
    Assignment,
    ProposerTrajectory,
    SolverTrajectory,
    Turn,
)

REPO = Path(__file__).resolve().parents[1]

FAKE_AGENT = r'''
import json, os, pathlib, shutil, sys, time
prompt = open(sys.argv[1]).read()
if sys.argv[2] == "sleep":
    time.sleep(60)
ws = pathlib.Path.cwd()
feedback = ws / "feedback" / "FEEDBACK.md"
checks = {
    "prompt_starts_with_skill": prompt.startswith("# Skill: create-task"),
    "prompt_has_assignment": "`torch-ckpt`" in prompt,
    "revision_in_prompt": "# Revision 1" in prompt,
    "assignment_file": (ws / "assignment" / "ASSIGNMENT.md").is_file(),
    "rollout_index": (ws / "past_rollouts" / "index.jsonl").is_file(),
    "memory": (ws / "memory" / "lessons.md").read_text() if (ws / "memory" / "lessons.md").exists() else None,
    "tool_on_path": (shutil.which("aes-validate") or "").startswith(str(ws / "tools")),
    "blocked_hosts": os.environ.get("AES_BLOCKED_HOSTS"),
    "serper": os.environ.get("SERPER_API_KEY"),
    "harbor_env": os.environ.get("AES_SANDBOX_HARBOR_ENV"),
    "feedback": feedback.read_text() if feedback.exists() else None,
    "previous": sorted(p.name for p in (ws / "output" / "tasks").iterdir()),
}
task = ws / "output" / "tasks" / (checks["previous"][0] if checks["previous"] else "demo-task")
files = {"task.toml": "[metadata]\ncategory = 'debugging'\n", "instruction.md": "Fix the loader.",
         "environment/Dockerfile": "FROM python:3.12-slim\n", "solution/solve.sh": "#!/bin/sh\n",
         "tests/test.sh": "#!/bin/sh\n"}
for rel, text in files.items():
    (task / rel).parent.mkdir(parents=True, exist_ok=True)
    (task / rel).write_text(text)
(ws / "memory" / "lessons.md").write_text("round 2 notes")
events = [
    {"type": "assistant", "message": {"id": "m1", "content": [
        {"type": "text", "text": "Inspecting the workspace."},
        {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "inspect"}}]}},
    {"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": json.dumps(checks)}]}},
    {"type": "assistant", "message": {"id": "m2", "content": [{"type": "text", "text": "Done."}]}},
    {"type": "result", "subtype": "success", "is_error": False, "num_turns": 2,
     "total_cost_usd": 0.5, "usage": {"input_tokens": 10, "output_tokens": 20}},
]
for event in events:
    print(json.dumps(event), flush=True)
'''


class LocalSandbox:
    def __init__(self, spec: SandboxSpec):
        self.spec, self.terminated, self.execs = spec, False, []

    async def exec(self, *cmd, timeout_s=None, workdir=None, env=None):
        self.execs.append(cmd)
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=workdir if workdir and os.path.isdir(workdir) else None,
            env={**os.environ, **(env or {})}, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout_s)
        except TimeoutError:
            proc.kill()
            await proc.communicate()
            return ExecResult(124, "", "timed out")
        return ExecResult(proc.returncode, out.decode(), err.decode())

    async def write_file(self, remote_path, data):
        Path(remote_path).parent.mkdir(parents=True, exist_ok=True)
        Path(remote_path).write_bytes(data)

    async def read_file(self, remote_path):
        return Path(remote_path).read_bytes()

    async def upload_dir(self, local_dir, remote_dir):
        shutil.copytree(local_dir, remote_dir, dirs_exist_ok=True)

    async def download_dir(self, remote_dir, local_dir):
        if not Path(remote_dir).is_dir():
            raise FileNotFoundError(remote_dir)
        shutil.copytree(remote_dir, local_dir, dirs_exist_ok=True)

    async def terminate(self):
        self.terminated = True


class LocalBackend:
    name = "local"

    def __init__(self):
        self.specs, self.sandboxes = [], []

    async def create(self, spec):
        self.specs.append(spec)
        Path(spec.workdir).mkdir(parents=True, exist_ok=True)
        self.sandboxes.append(LocalSandbox(spec))
        return self.sandboxes[-1]


class FakeAgentHarness(ClaudeCodeHarness):
    """Claude Code's log parser, with a scripted agent instead of the claude CLI."""
    credential_env = ("SERPER_API_KEY",)

    def __init__(self, script: Path, mode: str = "ok"):
        self.script, self.mode = script, mode

    def prepare(self, *, model, prompt_file, run_dir, workdir, env):
        return HarnessLaunch(command=f"{sys.executable} {self.script} {prompt_file} {self.mode}",
                             files={f"{run_dir}/fake/config.json": b"{}"})


@pytest.fixture
def setup(tmp_path):
    image = tmp_path / "image"
    (image / "skills" / "create-task").mkdir(parents=True)
    (image / "skills" / "create-task" / "SKILL.md").write_text(
        "---\nname: create-task\n---\nScaffold with harbor task init.\n")
    shutil.copytree(REPO / "skills" / "auto_env_scaling", image / "skills" / "auto_env_scaling")
    shutil.copytree(REPO / "docker" / "proposer" / "tools", image / "tools")
    layout = EpisodeLayout(workspace=str(tmp_path / "sandbox" / "workspace"),
                           run_dir=str(tmp_path / "sandbox" / "run"),
                           image_skills=str(image / "skills"), image_tools=str(image / "tools"),
                           base_path=os.environ["PATH"])
    script = tmp_path / "fake_agent.py"
    script.write_text(FAKE_AGENT)
    settings = Settings()
    settings.sandbox.proposer_timeout_s = 60
    state = tmp_path / "state"
    (state / "memory").mkdir(parents=True)
    (state / "memory" / "lessons.md").write_text("round 1 notes")
    rollouts = [SolverTrajectory(trajectory_id=f"r{i}", task_id="torch-ckpt", model="solver",
                                 turns=[Turn("assistant", "tried")], tests_passed=i,
                                 tests_total=4) for i in range(3)]
    workspace = ProposerWorkspace.build(tmp_path / "host-ws", rollouts=rollouts, state_dir=state)
    assignment = Assignment(assignment_id="asg-test", parent_task_id="torch-ckpt",
                            solver_pass_rate=0.0, common_failure="buffers loaded as weights",
                            direction="simpler")
    return {"layout": layout, "script": script, "settings": settings, "workspace": workspace,
            "assignment": assignment, "tmp": tmp_path}


def runner(setup, backend, **kwargs):
    return EpisodeRunner(setup["settings"], backend=backend, layout=setup["layout"],
                         host_env={"SERPER_API_KEY": "serper-key", "UNRELATED": "x"},
                         poll_s=0.2, **kwargs)


def checks_from(trajectory: ProposerTrajectory) -> dict:
    tool_turn = next(t for t in trajectory.turns if t.role == "tool")
    return json.loads(tool_turn.content)


async def test_episode_produces_a_task_and_trajectory(setup):
    backend = LocalBackend()
    out = setup["tmp"] / "out"
    trajectory = await runner(setup, backend).run(
        setup["assignment"], harness=FakeAgentHarness(setup["script"]), model="claude-opus-5",
        workspace_dir=setup["workspace"].root, out_dir=out)

    spec = backend.specs[0]
    assert spec.image == setup["settings"].sandbox.proposer_image
    assert (spec.cpu, spec.memory_gib) == (16, 32)
    assert spec.timeout_s == 60 + STAGING_MARGIN_S
    assert spec.blocked_hosts == list(DEFAULT_BLOCKED_HOSTS)
    assert backend.sandboxes[0].terminated

    checks = checks_from(trajectory)
    assert checks["prompt_starts_with_skill"] and checks["prompt_has_assignment"]
    assert checks["assignment_file"] and checks["rollout_index"] and checks["tool_on_path"]
    assert checks["memory"] == "round 1 notes" and checks["serper"] == "serper-key"
    assert checks["blocked_hosts"] == "tbench.ai,www.tbench.ai" and checks["harbor_env"] == "modal"
    assert checks["feedback"] is None and not checks["revision_in_prompt"]

    assert trajectory.harness == "claude-code" and trajectory.model == "claude-opus-5"
    assert trajectory.turns[0].role == "user" and trajectory.turns[0].content.startswith("# Skill")
    assert [t.role for t in trajectory.turns[1:]] == ["assistant", "tool", "assistant"]
    assert trajectory.task_dir == str(out / "output" / "tasks" / "demo-task")
    assert trajectory.task_digest == HarborTask(Path(trajectory.task_dir)).digest()
    assert trajectory.usage["exit_code"] == 0 and not trajectory.usage["timed_out"]
    assert trajectory.usage["output_tokens"] == 20 and trajectory.revision == 0
    assert trajectory.validation is None and trajectory.reward is None

    assert (out / "memory" / "lessons.md").read_text() == "round 2 notes"
    assert (out / "tools" / "aes-validate").is_file()
    assert (out / "prompt.md").read_text() == trajectory.turns[0].content
    record = json.loads((out / "episode.json").read_text())
    assert record["collection_errors"] == [] and record["harness_completed"]
    saved = ProposerTrajectory.from_dict(json.loads((out / "trajectory.json").read_text()))
    assert saved == trajectory


async def test_revision_gets_feedback_and_the_previous_task(setup):
    previous = setup["tmp"] / "prev" / "loader-task"
    for rel in ("task.toml", "instruction.md", "environment/Dockerfile", "solution/solve.sh",
                "tests/test.sh"):
        (previous / rel).parent.mkdir(parents=True, exist_ok=True)
        (previous / rel).write_text("old")
    out = setup["tmp"] / "out-rev"
    trajectory = await runner(setup, LocalBackend()).run(
        setup["assignment"], harness=FakeAgentHarness(setup["script"]), model="claude-opus-5",
        workspace_dir=setup["workspace"].root, out_dir=out,
        feedback="calibration: too_easy (rewards 1, 1, 1, 1)", revision=1,
        previous_task_dir=previous)
    checks = checks_from(trajectory)
    assert checks["feedback"] == "calibration: too_easy (rewards 1, 1, 1, 1)"
    assert checks["revision_in_prompt"] and checks["previous"] == ["loader-task"]
    assert "output/tasks/loader-task/" in trajectory.turns[0].content
    assert trajectory.revision == 1 and trajectory.task_dir.endswith("loader-task")
    # The shared round workspace is not modified by a revision.
    assert not list((setup["workspace"].root / "output" / "tasks").iterdir())


async def test_revision_finds_a_previous_task_staged_in_the_workspace(setup):
    previous = setup["tmp"] / "prev" / "staged-task"
    for rel in ("task.toml", "instruction.md", "environment/Dockerfile", "solution/solve.sh",
                "tests/test.sh"):
        (previous / rel).parent.mkdir(parents=True, exist_ok=True)
        (previous / rel).write_text("old")
    setup["workspace"].stage_previous_submission(previous)
    trajectory = await runner(setup, LocalBackend()).run(
        setup["assignment"], harness=FakeAgentHarness(setup["script"]), model="claude-opus-5",
        workspace_dir=setup["workspace"].root, out_dir=setup["tmp"] / "out-rev2",
        feedback="validation: rubric_review failed", revision=2)
    assert checks_from(trajectory)["previous"] == ["staged-task"]
    assert "output/tasks/staged-task/" in trajectory.turns[0].content


async def test_budget_stops_the_agent(setup):
    setup["settings"].sandbox.proposer_timeout_s = 1
    out = setup["tmp"] / "out-timeout"
    trajectory = await runner(setup, LocalBackend(), host_grace_s=2).run(
        setup["assignment"], harness=FakeAgentHarness(setup["script"], mode="sleep"),
        model="claude-opus-5", workspace_dir=setup["workspace"].root, out_dir=out)
    assert trajectory.usage["timed_out"] and trajectory.task_dir is None
    assert trajectory.usage["agent_time_s"] < 20
    assert len(trajectory.turns) == 1  # only the prompt; the agent produced nothing


async def test_missing_credentials_fail_before_any_sandbox(setup):
    backend = LocalBackend()
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        await EpisodeRunner(setup["settings"], backend=backend, layout=setup["layout"],
                            host_env={}).run(
            setup["assignment"], harness="claude-code", model="claude-opus-5",
            workspace_dir=setup["workspace"].root, out_dir=setup["tmp"] / "out-none")
    assert backend.specs == []
