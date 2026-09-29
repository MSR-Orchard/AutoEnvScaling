"""Turn trajectories into training data.

Three outputs, all JSONL:

* Solver prompt rows (`solver_prompt_row`): one row per pool environment, in the layout Miles
  reads with ``--input-key prompt --label-key label --metadata-key metadata``::

      {"prompt": [{"role": "user", "content": <instruction.md>}],
       "label": <task_id>,
       "metadata": {"task_id": ..., "task_dir": ...}}

* Cold-Start fine-tuning rows (`to_sft`): admitted, high-quality proposer trajectories as chat
  messages with tool calls, in the layout Miles' SFT rollout reads with
  ``--input-key messages --tool-key tools``::

      {"messages": [...], "tools": [...], "metadata": {...}}

* RL samples (`to_rl_groups`, `write_rl_samples`): proposer trajectories grouped by generation
  assignment and scored with r_P, solver trajectories grouped by environment and scored with
  r_S. Advantages are mean-centred within each group (`rewards.group_advantages`) and groups
  whose rewards do not vary are dropped. Each line is one sample::

      {"schema": "aes-rl-sample/v1", "role": "proposer" | "solver", "group_id": ...,
       "group_index": 0, "index": 0, "trajectory_id": ..., "reward": 1.0,
       "advantage": 0.25, "messages": [...], "tools": [...], "metadata": {...}}

  `autoenvscaling.training.miles.rollout` tokenizes these rows into Miles samples.

Every writer replaces its file atomically, because a running Miles job may re-read a prompt
file while it is being rewritten.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autoenvscaling.rewards import (
    PROPOSER_ADMITTED,
    group_advantages,
    proposer_reward,
    solver_reward,
)
from autoenvscaling.tasks import HarborTask
from autoenvscaling.trajectories.schema import ProposerTrajectory, SolverTrajectory, Turn

RL_SAMPLE_SCHEMA = "aes-rl-sample/v1"
PROPOSER = "proposer"
SOLVER = "solver"


# ---------------------------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------------------------
def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Write rows to `path` through a temporary file and a rename. Returns the row count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---------------------------------------------------------------------------------------------
# Solver prompt rows
# ---------------------------------------------------------------------------------------------
def solver_prompt_row(task: HarborTask, *, task_id: str | None = None,
                      metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """One Miles prompt row for an environment. The solver sees only the instruction."""
    task_id = task_id or task.task_id
    meta = {"task_id": task_id, "task_dir": str(Path(task.path).resolve())}
    meta.update(metadata or {})
    return {"prompt": [{"role": "user", "content": task.instruction}], "label": task_id,
            "metadata": meta}


# ---------------------------------------------------------------------------------------------
# Chat messages
# ---------------------------------------------------------------------------------------------
class ConversionError(ValueError):
    """A trajectory cannot be expressed as chat messages."""


def _normalize_arguments(raw: Any, arguments_format: str) -> Any:
    if isinstance(raw, str):
        text = raw.strip() or "{}"
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConversionError(f"tool-call arguments are not JSON: {raw[:80]!r}") from exc
    elif raw is None:
        parsed = {}
    else:
        parsed = raw
    if not isinstance(parsed, dict):
        raise ConversionError(f"tool-call arguments must be an object, got {type(parsed).__name__}")
    return parsed if arguments_format == "dict" else json.dumps(parsed, ensure_ascii=False)


def normalize_tool_call(call: dict[str, Any], index: int, arguments_format: str = "dict") -> dict[str, Any]:
    """Accept the OpenAI shape ({"function": {"name", "arguments"}}) or a flat one
    ({"name", "arguments" | "input"}) and return the OpenAI shape.

    ``arguments_format="dict"`` keeps arguments as an object, which is what the Qwen3.5 and
    Qwen3.6 chat templates iterate over; ``"json"`` encodes them as a string.
    """
    if arguments_format not in ("dict", "json"):
        raise ValueError(f"arguments_format must be 'dict' or 'json', got {arguments_format!r}")
    fn = call.get("function") if isinstance(call.get("function"), dict) else call
    name = fn.get("name")
    if not name:
        raise ConversionError("tool call without a name")
    raw = fn.get("arguments", fn.get("input", fn.get("args")))
    return {"id": call.get("id") or f"call_{index}", "type": "function",
            "function": {"name": name, "arguments": _normalize_arguments(raw, arguments_format)}}


def turn_to_message(turn: Turn, *, arguments_format: str = "dict") -> dict[str, Any]:
    if turn.role not in ("system", "user", "assistant", "tool"):
        raise ConversionError(f"unknown role {turn.role!r}")
    message: dict[str, Any] = {"role": turn.role, "content": turn.content or ""}
    if turn.tool_calls:
        if turn.role != "assistant":
            raise ConversionError(f"{turn.role} turn carries tool calls")
        message["tool_calls"] = [normalize_tool_call(c, i, arguments_format)
                                 for i, c in enumerate(turn.tool_calls)]
    if turn.role == "tool":
        if turn.tool_call_id:
            message["tool_call_id"] = turn.tool_call_id
        if turn.name:
            message["name"] = turn.name
    return message


def turns_to_messages(turns: Sequence[Turn], *, system_prompt: str | None = None,
                      arguments_format: str = "dict") -> list[dict[str, Any]]:
    """Chat messages for a trajectory. `system_prompt` is prepended only if the trajectory has none."""
    messages = [turn_to_message(t, arguments_format=arguments_format) for t in turns]
    if system_prompt and not (messages and messages[0]["role"] == "system"):
        messages.insert(0, {"role": "system", "content": system_prompt})
    return messages


_JSON_TYPES = ((bool, "boolean"), (int, "integer"), (float, "number"), (str, "string"),
               (list, "array"), (dict, "object"))


def _json_type(value: Any) -> str:
    for py_type, name in _JSON_TYPES:
        if isinstance(value, py_type):
            return name
    return "string"


def infer_tools(messages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Minimal tool schemas from the calls a conversation makes.

    Used only when the harness did not record its tool definitions. Each tool gets the argument
    names and JSON types it was actually called with; descriptions are empty.
    """
    seen: dict[str, dict[str, str]] = {}
    for message in messages:
        for call in message.get("tool_calls", []):
            fn = call["function"]
            args = fn["arguments"]
            if isinstance(args, str):
                args = json.loads(args or "{}")
            props = seen.setdefault(fn["name"], {})
            for key, value in args.items():
                props.setdefault(key, _json_type(value))
    return [{"type": "function",
             "function": {"name": name, "description": "",
                          "parameters": {"type": "object",
                                         "properties": {k: {"type": t} for k, t in props.items()}}}}
            for name, props in sorted(seen.items())]


def _assistant_turns(turns: Sequence[Turn]) -> int:
    return sum(1 for t in turns if t.role == "assistant")


# ---------------------------------------------------------------------------------------------
# Cold-Start fine-tuning set
# ---------------------------------------------------------------------------------------------
@dataclass
class SFTFilter:
    """Which proposer trajectories count as high quality. Defaults keep admitted episodes only."""
    admitted_only: bool = True           # r_P == 1: passed validation and calibration
    first_attempt_only: bool = False     # drop revisions made after host feedback
    min_assistant_turns: int = 1
    max_assistant_turns: int | None = None
    models: tuple[str, ...] = ()         # empty means any model
    harnesses: tuple[str, ...] = ()      # empty means any harness
    max_per_assignment: int | None = None
    dedupe_tasks: bool = True            # one trajectory per task digest
    limit: int | None = None             # keep the `limit` closest to the middle of the band


@dataclass
class ExportStats:
    kept: int = 0
    skipped: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        return {"kept": self.kept, "skipped": dict(self.skipped)}


def _calibration_mean(traj: ProposerTrajectory) -> float | None:
    cal = traj.calibration or {}
    if "mean" in cal:
        return float(cal["mean"])
    rewards = cal.get("rewards") or []
    return sum(rewards) / len(rewards) if rewards else None


def _band_distance(traj: ProposerTrajectory) -> float:
    mean = _calibration_mean(traj)
    return abs(mean - 0.5) if mean is not None else 1.0


def select_for_sft(trajectories: Iterable[ProposerTrajectory],
                   filt: SFTFilter | None = None) -> tuple[list[ProposerTrajectory], ExportStats]:
    """Apply `filt`. Selection is deterministic: ties break on trajectory id."""
    filt = filt or SFTFilter()
    stats = ExportStats()
    candidates: list[ProposerTrajectory] = []
    for traj in trajectories:
        n_assistant = _assistant_turns(traj.turns)
        if filt.admitted_only and traj.reward != PROPOSER_ADMITTED:
            stats.skipped["not_admitted"] += 1
        elif filt.first_attempt_only and traj.revision > 0:
            stats.skipped["revision"] += 1
        elif filt.models and traj.model not in filt.models:
            stats.skipped["model"] += 1
        elif filt.harnesses and traj.harness not in filt.harnesses:
            stats.skipped["harness"] += 1
        elif n_assistant < filt.min_assistant_turns:
            stats.skipped["too_few_turns"] += 1
        elif filt.max_assistant_turns is not None and n_assistant > filt.max_assistant_turns:
            stats.skipped["too_many_turns"] += 1
        else:
            candidates.append(traj)

    candidates.sort(key=lambda t: (_band_distance(t), t.trajectory_id))
    selected: list[ProposerTrajectory] = []
    digests: set[str] = set()
    per_assignment: Counter = Counter()
    for traj in candidates:
        if filt.dedupe_tasks and traj.task_digest:
            if traj.task_digest in digests:
                stats.skipped["duplicate_task"] += 1
                continue
            digests.add(traj.task_digest)
        aid = traj.assignment.assignment_id
        if filt.max_per_assignment is not None and per_assignment[aid] >= filt.max_per_assignment:
            stats.skipped["assignment_cap"] += 1
            continue
        per_assignment[aid] += 1
        selected.append(traj)
    if filt.limit is not None and len(selected) > filt.limit:
        stats.skipped["over_limit"] += len(selected) - filt.limit
        selected = selected[: filt.limit]
    return selected, stats


def to_sft(trajectories: Iterable[ProposerTrajectory], filt: SFTFilter | None = None, *,
           tools: list[dict[str, Any]] | None = None, system_prompt: str | None = None,
           arguments_format: str = "dict") -> tuple[list[dict[str, Any]], ExportStats]:
    """Cold-Start rows: the selected trajectories as chat messages with tool calls.

    `tools` should be the proposer harness's tool definitions (OpenAI function format); when it
    is None, minimal schemas are inferred from the calls each trajectory makes.
    """
    selected, stats = select_for_sft(trajectories, filt)
    rows: list[dict[str, Any]] = []
    for traj in selected:
        try:
            messages = turns_to_messages(traj.turns, system_prompt=system_prompt,
                                         arguments_format=arguments_format)
        except ConversionError:
            stats.skipped["unconvertible"] += 1
            continue
        rows.append({
            "messages": messages,
            "tools": tools if tools is not None else infer_tools(messages),
            "metadata": {"trajectory_id": traj.trajectory_id,
                         "assignment_id": traj.assignment.assignment_id,
                         "model": traj.model, "harness": traj.harness,
                         "revision": traj.revision, "reward": traj.reward,
                         "task_digest": traj.task_digest,
                         "calibration_mean": _calibration_mean(traj)},
        })
    stats.kept = len(rows)
    return rows, stats


def write_sft(path: Path, trajectories: Iterable[ProposerTrajectory], filt: SFTFilter | None = None,
              **kwargs: Any) -> ExportStats:
    rows, stats = to_sft(trajectories, filt, **kwargs)
    write_jsonl_atomic(path, rows)
    return stats


# ---------------------------------------------------------------------------------------------
# RL groups
# ---------------------------------------------------------------------------------------------
@dataclass
class ChatFormat:
    """How a harness presents a conversation to the model, for re-tokenizing it for training.

    Set `system_prompt` and `tools` to what the harness actually sent when the recorded turns
    lack them; otherwise the system prompt is omitted and tool schemas are inferred from the
    calls (`infer_tools`), and the trained context differs from the one the model saw.
    """
    system_prompt: str | None = None
    tools: list[dict[str, Any]] | None = None
    arguments_format: str = "dict"


@dataclass
class RLSample:
    trajectory_id: str
    reward: float
    advantage: float
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RLGroup:
    """Trajectories that share a baseline: one assignment (proposer) or one environment (solver).

    `rewards` covers every scored trajectory in the group, including ones with no trainable
    turns (for example an episode that crashed before its first action), so the baseline is the
    mean over the whole group. `samples` holds only the trajectories that can be trained on.
    """
    group_id: str
    role: str
    rewards: list[float]
    samples: list[RLSample]

    @property
    def mean_reward(self) -> float:
        return sum(self.rewards) / len(self.rewards)


@dataclass
class GroupingStats:
    groups_seen: int = 0
    groups_kept: int = 0
    groups_constant: int = 0
    groups_untrainable: int = 0
    samples_kept: int = 0
    skipped: Counter = field(default_factory=Counter)

    @property
    def useful_group_rate(self) -> float:
        return self.groups_kept / self.groups_seen if self.groups_seen else 0.0

    def merge(self, other: GroupingStats) -> GroupingStats:
        return GroupingStats(self.groups_seen + other.groups_seen,
                             self.groups_kept + other.groups_kept,
                             self.groups_constant + other.groups_constant,
                             self.groups_untrainable + other.groups_untrainable,
                             self.samples_kept + other.samples_kept,
                             self.skipped + other.skipped)

    def to_dict(self) -> dict[str, Any]:
        return {"groups_seen": self.groups_seen, "groups_kept": self.groups_kept,
                "groups_constant": self.groups_constant,
                "groups_untrainable": self.groups_untrainable,
                "samples_kept": self.samples_kept, "useful_group_rate": self.useful_group_rate,
                "skipped": dict(self.skipped)}


def scored_proposer_reward(traj: ProposerTrajectory) -> float | None:
    """r_P for a trajectory: the stored reward, or one derived from its reports. None if unscored."""
    if traj.reward is not None:
        return float(traj.reward)
    if traj.validation is None:
        return None
    calibration = None if traj.calibration is None else bool(traj.calibration.get("passed"))
    return proposer_reward(bool(traj.validation.get("passed")), calibration)


def scored_solver_reward(traj: SolverTrajectory) -> float | None:
    if traj.reward is not None:
        return float(traj.reward)
    if traj.tests_total > 0:
        return solver_reward(traj.tests_passed, traj.tests_total)
    return None


def _build_groups(keyed: dict[str, list[tuple[Any, float]]], role: str, chat: ChatFormat | None,
                  meta_fn) -> tuple[list[RLGroup], GroupingStats]:
    chat = chat or ChatFormat()
    stats = GroupingStats()
    groups: list[RLGroup] = []
    for key in sorted(keyed):
        members = keyed[key]
        stats.groups_seen += 1
        rewards = [r for _, r in members]
        advantages = group_advantages(rewards)
        if advantages is None:
            stats.groups_constant += 1
            continue
        samples: list[RLSample] = []
        for (traj, reward), adv in zip(members, advantages):
            if _assistant_turns(traj.turns) == 0:
                stats.skipped["no_assistant_turns"] += 1
                continue
            try:
                messages = turns_to_messages(traj.turns, system_prompt=chat.system_prompt,
                                             arguments_format=chat.arguments_format)
            except ConversionError:
                stats.skipped["unconvertible"] += 1
                continue
            samples.append(RLSample(traj.trajectory_id, reward, adv, messages,
                                    chat.tools if chat.tools is not None else infer_tools(messages),
                                    meta_fn(traj)))
        if not samples:
            stats.groups_untrainable += 1
            continue
        groups.append(RLGroup(key, role, rewards, samples))
        stats.groups_kept += 1
        stats.samples_kept += len(samples)
    return groups, stats


def proposer_rl_groups(trajectories: Iterable[ProposerTrajectory], *, separate_revisions: bool = False,
                       chat: ChatFormat | None = None) -> tuple[list[RLGroup], GroupingStats]:
    """Group proposer trajectories by generation assignment and centre r_P within each group.

    By default revisions of an assignment (episodes that received host feedback) share the
    group with first attempts; `separate_revisions=True` groups by (assignment, revision).
    """
    keyed: dict[str, list[tuple[Any, float]]] = defaultdict(list)
    skipped = Counter()
    for traj in trajectories:
        reward = scored_proposer_reward(traj)
        if reward is None:
            skipped["unscored"] += 1
            continue
        key = traj.assignment.assignment_id
        if separate_revisions:
            key = f"{key}/rev{traj.revision}"
        keyed[key].append((traj, reward))

    def meta(traj: ProposerTrajectory) -> dict[str, Any]:
        return {"assignment_id": traj.assignment.assignment_id, "revision": traj.revision,
                "model": traj.model, "harness": traj.harness, "task_digest": traj.task_digest}

    groups, stats = _build_groups(keyed, PROPOSER, chat, meta)
    stats.skipped += skipped
    return groups, stats


def solver_rl_groups(trajectories: Iterable[SolverTrajectory], *,
                     chat: ChatFormat | None = None) -> tuple[list[RLGroup], GroupingStats]:
    """Group solver attempts by environment and centre r_S within each group."""
    keyed: dict[str, list[tuple[Any, float]]] = defaultdict(list)
    skipped = Counter()
    for traj in trajectories:
        reward = scored_solver_reward(traj)
        if reward is None:
            skipped["unscored"] += 1
            continue
        keyed[traj.task_id].append((traj, reward))

    def meta(traj: SolverTrajectory) -> dict[str, Any]:
        return {"task_id": traj.task_id, "model": traj.model, "tests_passed": traj.tests_passed,
                "tests_total": traj.tests_total}

    groups, stats = _build_groups(keyed, SOLVER, chat, meta)
    stats.skipped += skipped
    return groups, stats


def to_rl_groups(proposer: Iterable[ProposerTrajectory] = (), solver: Iterable[SolverTrajectory] = (),
                 *, separate_revisions: bool = False, proposer_chat: ChatFormat | None = None,
                 solver_chat: ChatFormat | None = None) -> tuple[list[RLGroup], GroupingStats]:
    """Proposer groups followed by solver groups, with advantages computed separately per role."""
    p_groups, p_stats = proposer_rl_groups(proposer, separate_revisions=separate_revisions,
                                           chat=proposer_chat)
    s_groups, s_stats = solver_rl_groups(solver, chat=solver_chat)
    return p_groups + s_groups, p_stats.merge(s_stats)


def rl_sample_rows(groups: Sequence[RLGroup]) -> list[dict[str, Any]]:
    """Flatten groups into sample rows with consecutive `group_index` and `index` values."""
    rows: list[dict[str, Any]] = []
    index = 0
    for group_index, group in enumerate(groups):
        for sample in group.samples:
            rows.append({"schema": RL_SAMPLE_SCHEMA, "role": group.role, "group_id": group.group_id,
                         "group_index": group_index, "index": index,
                         "trajectory_id": sample.trajectory_id, "reward": sample.reward,
                         "advantage": sample.advantage, "group_mean_reward": group.mean_reward,
                         "messages": sample.messages, "tools": sample.tools,
                         "metadata": sample.metadata})
            index += 1
    return rows


def write_rl_samples(path: Path, groups: Sequence[RLGroup]) -> int:
    return write_jsonl_atomic(path, rl_sample_rows(groups))


# ---------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------
def _read_solver_jsonl(path: Path) -> list[SolverTrajectory]:
    rows = read_jsonl(path)
    out = []
    for row in rows:
        row = {k: v for k, v in row.items() if k in SolverTrajectory.__dataclass_fields__}
        row["turns"] = [Turn(**t) for t in row.get("turns", [])]
        out.append(SolverTrajectory(**row))
    return out


def main(argv: list[str] | None = None) -> int:
    """python -m autoenvscaling.trajectories.export {sft,rl} ..."""
    import argparse

    from autoenvscaling.trajectories.schema import read_proposer_jsonl

    parser = argparse.ArgumentParser(prog="python -m autoenvscaling.trajectories.export")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("sft", help="Cold-Start rows from proposer trajectory files")
    p.add_argument("inputs", nargs="+", type=Path, help="proposer_trajectories.jsonl files")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tools", type=Path, help="JSON file with the harness's tool definitions")
    p.add_argument("--system-prompt", type=Path, help="text file prepended when a trajectory has none")
    p.add_argument("--first-attempt-only", action="store_true")
    p.add_argument("--model", action="append", default=[])
    p.add_argument("--harness", action="append", default=[])
    p.add_argument("--min-turns", type=int, default=1)
    p.add_argument("--max-per-assignment", type=int)
    p.add_argument("--limit", type=int)
    r = sub.add_parser("rl", help="RL sample rows from proposer and solver trajectory files")
    r.add_argument("--proposer", nargs="*", type=Path, default=[])
    r.add_argument("--solver", nargs="*", type=Path, default=[])
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--separate-revisions", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "sft":
        trajectories = [t for path in args.inputs for t in read_proposer_jsonl(path)]
        filt = SFTFilter(first_attempt_only=args.first_attempt_only, models=tuple(args.model),
                         harnesses=tuple(args.harness), min_assistant_turns=args.min_turns,
                         max_per_assignment=args.max_per_assignment, limit=args.limit)
        tools = json.loads(args.tools.read_text()) if args.tools else None
        system_prompt = args.system_prompt.read_text() if args.system_prompt else None
        stats = write_sft(args.out, trajectories, filt, tools=tools, system_prompt=system_prompt)
        print(json.dumps(stats.to_dict()))
        return 0

    proposer = [t for path in args.proposer for t in read_proposer_jsonl(path)]
    solver = [t for path in args.solver for t in _read_solver_jsonl(path)]
    groups, stats = to_rl_groups(proposer, solver, separate_revisions=args.separate_revisions)
    write_rl_samples(args.out, groups)
    print(json.dumps(stats.to_dict()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
