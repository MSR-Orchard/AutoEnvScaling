"""The proposer workspace layout on the host."""

import json
from pathlib import Path

from autoenvscaling.proposer import workspace as ws
from autoenvscaling.proposer.workspace import ProposerWorkspace, find_task_dirs, save_persistent
from autoenvscaling.tasks import HarborTask
from autoenvscaling.trajectories.schema import SolverTrajectory, Turn


def make_task(root: Path, name: str, category: str | None = None) -> Path:
    task = root / name
    for rel, text in {
        "task.toml": f'[metadata]\ncategory = "{category}"\n' if category else "version = '1'\n",
        "instruction.md": "Do the thing.",
        "environment/Dockerfile": "FROM ubuntu:24.04\n",
        "solution/solve.sh": "#!/bin/sh\necho done\n",
        "tests/test.sh": "#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n",
    }.items():
        (task / rel).parent.mkdir(parents=True, exist_ok=True)
        (task / rel).write_text(text)
    return task


def traj(i, task, reward, verifier="", total=4):
    return SolverTrajectory(trajectory_id=f"run/{i}", task_id=task, model="solver",
                            turns=[Turn("user", "go"), Turn("assistant", "ok")],
                            tests_passed=int((reward or 0) * total), tests_total=total,
                            reward=reward, verifier_output=verifier)


def test_build_writes_the_paper_layout(tmp_path):
    envs = tmp_path / "pool"
    parent = make_task(envs, "parent", category="debugging")
    (parent / "environment" / "big.bin").write_bytes(b"x" * 100)
    rollouts = [traj(0, "parent", 1.0), traj(1, "parent", 0.25, "FAILED t::a"),
                traj(2, "parent", None, total=0)]
    root = tmp_path / "ws"
    workspace = ProposerWorkspace.build(root, rollouts=rollouts, environments=[parent],
                                        max_file_bytes=50)
    for sub in (ws.PAST_ROLLOUTS, ws.PAST_ENVIRONMENTS, ws.MEMORY, ws.TOOLS, ws.LOGS,
                ws.OUTPUT_TASKS):
        assert (root / sub).is_dir(), sub

    ok = root / "past_rollouts/successful/parent/run_0.json"
    bad = root / "past_rollouts/failed/parent/run_1.json"
    assert ok.is_file() and bad.is_file()
    assert not list((root / "past_rollouts").rglob("run_2.json"))  # no verdict, not written
    record = json.loads(bad.read_text())
    assert record["outcome"] == "failed" and record["verifier_output"] == "FAILED t::a"
    assert record["turns"][1] == {"role": "assistant", "content": "ok", "tool_calls": [],
                                  "tool_call_id": None, "name": None}

    index = [json.loads(line) for line in (root / "past_rollouts/index.jsonl").read_text()
             .splitlines()]
    assert index[0]["task_id"] == "parent" and index[0]["rewards"] == [1.0, 0.25]
    assert index[0]["n_errored"] == 1 and index[0]["category"] == "debugging"
    assert index[0]["environment_dir"] == "past_environments/parent"

    copied = root / "past_environments/parent"
    assert HarborTask(copied).is_complete() and not (copied / "environment/big.bin").exists()
    env_index = json.loads((root / "past_environments/index.jsonl").read_text())
    assert env_index["omitted_large_files"] == ["environment/big.bin"]
    assert workspace.environment_categories() == {"parent": "debugging"}


def test_persistent_state_round_trip_and_revision_staging(tmp_path):
    state = tmp_path / "state"
    (state / "memory").mkdir(parents=True)
    (state / "memory" / "lessons.md").write_text("verifiers must be deterministic")
    (state / "tools").mkdir()
    (state / "tools" / "helper.sh").write_text("echo hi")
    previous = make_task(tmp_path / "prev", "my-task")

    root = tmp_path / "ws"
    workspace = ProposerWorkspace.build(root, state_dir=state, previous_submission=previous)
    assert (root / "memory/lessons.md").read_text() == "verifiers must be deterministic"
    assert (root / "tools/helper.sh").is_file()
    assert HarborTask(root / "output/tasks/my-task").is_complete()

    (root / "memory/lessons.md").write_text("updated")
    new_state = tmp_path / "state2"
    workspace.save_persistent(new_state)
    assert (new_state / "memory/lessons.md").read_text() == "updated"
    save_persistent(root, state)  # overwrite in place
    assert (state / "memory/lessons.md").read_text() == "updated"


def test_find_task_dirs_prefers_output_tasks(tmp_path):
    out = tmp_path / "output"
    make_task(out / "tasks", "b-task")
    make_task(out / "tasks" / "a-wrapper", "nested")      # harbor task init <org>/<name>
    (out / "tasks" / "incomplete").mkdir()
    make_task(out, "loose")
    found = [p.relative_to(out).as_posix() for p in find_task_dirs(out)]
    assert found == ["tasks/a-wrapper/nested", "tasks/b-task", "loose"]
    assert find_task_dirs(tmp_path / "missing") == []
