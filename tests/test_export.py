import json

import pytest

from autoenvscaling.tasks import HarborTask
from autoenvscaling.trajectories.export import (
    PROPOSER,
    RL_SAMPLE_SCHEMA,
    SOLVER,
    ConversionError,
    SFTFilter,
    infer_tools,
    normalize_tool_call,
    proposer_rl_groups,
    read_jsonl,
    rl_sample_rows,
    solver_prompt_row,
    solver_rl_groups,
    to_rl_groups,
    to_sft,
    write_rl_samples,
    write_sft,
)
from autoenvscaling.trajectories.schema import (
    Assignment,
    ProposerTrajectory,
    SolverTrajectory,
    Turn,
)


def _turns(n_actions: int = 2) -> list[Turn]:
    turns = [Turn("system", "You build Harbor tasks."), Turn("user", "Build a harder variant.")]
    for i in range(n_actions):
        turns.append(Turn("assistant", f"step {i}", tool_calls=[
            {"id": f"c{i}", "type": "function",
             "function": {"name": "bash", "arguments": json.dumps({"command": f"ls {i}"})}}]))
        turns.append(Turn("tool", f"out {i}", tool_call_id=f"c{i}", name="bash"))
    turns.append(Turn("assistant", "Submitted."))
    return turns


def _ptraj(tid: str, aid: str, reward: float | None, *, revision: int = 0, mean: float = 0.5,
           digest: str | None = None, model: str = "m", turns: list[Turn] | None = None
           ) -> ProposerTrajectory:
    return ProposerTrajectory(
        trajectory_id=tid, assignment=Assignment(aid, "parent", 0.3), model=model, harness="pi",
        turns=_turns() if turns is None else turns, task_digest=digest or tid, reward=reward,
        revision=revision, calibration={"mean": mean, "passed": reward == 1.0},
        validation={"passed": reward is not None and reward > -1})


def _straj(tid: str, task: str, reward: float) -> SolverTrajectory:
    return SolverTrajectory(tid, task, "solver", turns=_turns(1), reward=reward)


def test_normalize_tool_call_shapes():
    openai = normalize_tool_call({"id": "a", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}, 0)
    flat = normalize_tool_call({"name": "bash", "input": {"command": "ls"}}, 3)
    assert openai == {"id": "a", "type": "function", "function": {"name": "bash", "arguments": {"command": "ls"}}}
    assert flat["id"] == "call_3" and flat["function"]["arguments"] == {"command": "ls"}
    as_json = normalize_tool_call({"name": "bash", "arguments": {"command": "ls"}}, 0, "json")
    assert as_json["function"]["arguments"] == '{"command": "ls"}'
    with pytest.raises(ConversionError):
        normalize_tool_call({"name": "bash", "arguments": "not json"}, 0)


def test_infer_tools_collects_argument_types():
    messages = [{"role": "assistant", "content": "", "tool_calls": [
        normalize_tool_call({"name": "edit", "arguments": {"path": "a", "line": 3}}, 0),
        normalize_tool_call({"name": "bash", "arguments": {"command": "ls"}}, 1)]}]
    tools = infer_tools(messages)
    assert [t["function"]["name"] for t in tools] == ["bash", "edit"]
    assert tools[1]["function"]["parameters"]["properties"] == {"path": {"type": "string"},
                                                                "line": {"type": "integer"}}


def test_to_sft_keeps_admitted_deduped_and_ranked(tmp_path):
    trajs = [
        _ptraj("t1", "a1", 1.0, mean=0.7, digest="d1"),
        _ptraj("t2", "a1", 1.0, mean=0.5, digest="d2"),
        _ptraj("t3", "a2", 1.0, mean=0.5, digest="d2"),     # duplicate task
        _ptraj("t4", "a2", -0.25),                          # not admitted
        _ptraj("t5", "a3", 1.0, revision=1, mean=0.4, digest="d5"),
        _ptraj("t6", "a3", 1.0, turns=[Turn("user", "hi")], digest="d6"),  # no actions
    ]
    rows, stats = to_sft(trajs)
    assert [r["metadata"]["trajectory_id"] for r in rows] == ["t2", "t5", "t1"]
    assert stats.skipped == {"not_admitted": 1, "duplicate_task": 1, "too_few_turns": 1}
    first = rows[0]["messages"]
    assert [m["role"] for m in first[:4]] == ["system", "user", "assistant", "tool"]
    assert first[2]["tool_calls"][0]["function"]["arguments"] == {"command": "ls 0"}
    assert first[3]["tool_call_id"] == "c0"
    assert rows[0]["tools"][0]["function"]["name"] == "bash"

    rows, _ = to_sft(trajs, SFTFilter(first_attempt_only=True, limit=1))
    assert [r["metadata"]["trajectory_id"] for r in rows] == ["t2"]

    stats = write_sft(tmp_path / "sft.jsonl", trajs, SFTFilter(max_per_assignment=1),
                      tools=[{"type": "function", "function": {"name": "bash"}}])
    written = read_jsonl(tmp_path / "sft.jsonl")
    assert stats.kept == len(written) == 2
    assert written[0]["tools"] == [{"type": "function", "function": {"name": "bash"}}]


def test_proposer_groups_center_rewards_and_drop_constant_groups():
    trajs = [
        _ptraj("a", "g1", 1.0), _ptraj("b", "g1", -1.0), _ptraj("c", "g1", -0.25),
        _ptraj("d", "g1", 1.0, revision=1),
        _ptraj("e", "g2", -1.0), _ptraj("f", "g2", -1.0),   # constant: dropped
        _ptraj("g", "g3", None),                            # unscored
    ]
    trajs[-1].validation = None
    groups, stats = proposer_rl_groups(trajs)
    assert [g.group_id for g in groups] == ["g1"]
    assert [s.advantage for s in groups[0].samples] == pytest.approx([0.8125, -1.1875, -0.4375, 0.8125])
    assert stats.groups_seen == 2 and stats.groups_constant == 1 and stats.skipped["unscored"] == 1
    assert stats.useful_group_rate == 0.5

    split, _ = proposer_rl_groups(trajs, separate_revisions=True)
    assert [g.group_id for g in split] == ["g1/rev0"]      # g1/rev1 has one member: constant


def test_untrainable_members_still_count_in_the_baseline():
    trajs = [_ptraj("a", "g", 1.0), _ptraj("b", "g", -1.0, turns=[]), _ptraj("c", "g", -1.0)]
    groups, stats = proposer_rl_groups(trajs)
    assert groups[0].rewards == [1.0, -1.0, -1.0]
    assert [s.trajectory_id for s in groups[0].samples] == ["a", "c"]
    assert groups[0].samples[0].advantage == pytest.approx(1.0 - (-1.0 / 3))
    assert stats.skipped["no_assistant_turns"] == 1


def test_reward_derived_from_reports_when_missing():
    t = _ptraj("a", "g", None)
    t.validation, t.calibration = {"passed": True}, {"passed": False}
    u = _ptraj("b", "g", None)
    u.validation, u.calibration = {"passed": False}, None
    groups, _ = proposer_rl_groups([t, u])
    assert groups[0].rewards == [-0.25, -1.0]


def test_solver_groups_and_sample_rows(tmp_path):
    solver = [_straj("s1", "env1", 1.0), _straj("s2", "env1", 0.0), _straj("s3", "env2", 0.5),
              _straj("s4", "env2", 0.5)]
    solver.append(SolverTrajectory("s5", "env3", "solver", turns=_turns(1), tests_passed=1, tests_total=4))
    solver.append(SolverTrajectory("s6", "env3", "solver", turns=_turns(1), tests_passed=4, tests_total=4))
    groups, stats = solver_rl_groups(solver)
    assert [g.group_id for g in groups] == ["env1", "env3"]
    assert [s.advantage for s in groups[1].samples] == pytest.approx([-0.375, 0.375])
    assert stats.groups_constant == 1

    both, merged = to_rl_groups([_ptraj("a", "g", 1.0), _ptraj("b", "g", -1.0)], solver)
    assert [g.role for g in both] == [PROPOSER, SOLVER, SOLVER]
    assert merged.groups_kept == 3
    rows = rl_sample_rows(both)
    assert [r["group_index"] for r in rows] == [0, 0, 1, 1, 2, 2]
    assert [r["index"] for r in rows] == list(range(6))
    assert rows[0]["schema"] == RL_SAMPLE_SCHEMA and rows[0]["advantage"] == 1.0
    assert write_rl_samples(tmp_path / "b.jsonl", both) == 6
    assert read_jsonl(tmp_path / "b.jsonl")[2]["metadata"]["task_id"] == "env1"


def test_solver_prompt_row(tmp_path):
    task_dir = tmp_path / "abc"
    task_dir.mkdir()
    (task_dir / "instruction.md").write_text("Fix the build.")
    row = solver_prompt_row(HarborTask(task_dir), metadata={"image_ref": "img"})
    assert row["prompt"] == [{"role": "user", "content": "Fix the build."}]
    assert row["label"] == "abc"
    assert row["metadata"] == {"task_id": "abc", "task_dir": str(task_dir.resolve()), "image_ref": "img"}


def test_chat_format_supplies_system_prompt_and_tools():
    from autoenvscaling.trajectories.export import ChatFormat

    tools = [{"type": "function", "function": {"name": "bash", "parameters": {}}}]
    solver = [_straj("s1", "env", 1.0), _straj("s2", "env", 0.0)]
    for t in solver:
        t.turns = [turn for turn in t.turns if turn.role != "system"]
    groups, _ = solver_rl_groups(solver, chat=ChatFormat(system_prompt="You are Pi.", tools=tools,
                                                         arguments_format="json"))
    sample = groups[0].samples[0]
    assert sample.messages[0] == {"role": "system", "content": "You are Pi."}
    assert sample.tools == tools
    assert isinstance(sample.messages[2]["tool_calls"][0]["function"]["arguments"], str)
