"""The solver as seen by calibration and by the proposer's solver_model tool.

A solver attempt is one Harbor trial with Harbor's built-in ``pi`` agent. Pi runs inside the task
container and calls the solver model through an OpenAI-compatible endpoint (for example an SGLang
or vLLM server). Harbor configures Pi for a custom endpoint when the model name has a provider
prefix (``openai/<model>``), ``OPENAI_BASE_URL`` and ``OPENAI_API_KEY`` are set in the agent's
environment, and the ``model_api`` option names the API style (``openai-completions`` for a chat
completions server).

The reward of an attempt is the verifier's reward, which for our tasks is r_S, the fraction of the
environment's tests that pass. An attempt that never validly ran (the endpoint was unreachable,
the sandbox failed) comes back with ``reward=None`` and the error in ``verifier_output``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from autoenvscaling.tasks import HarborTask
from autoenvscaling.trajectories.schema import SolverTrajectory, Turn
from autoenvscaling.validation.harbor_runner import (PI_AGENT, SOLVER_TIMEOUT_SEC, AgentSpec,
                                                     HarborRunner, TrialOutcome, TrialRunner)

logger = logging.getLogger(__name__)

# Pi release used by the paper's solver runs (old pi_agent._PI_VERSION).
PI_VERSION = "0.80.6"
HINTS_HEADER = "## Hints\nHigh-level steps toward a solution:\n"


@dataclass
class SolverEndpoint:
    """An OpenAI-compatible endpoint serving the current solver, driven by the Pi coding agent.

    ``api_key_env`` names the host environment variable that holds the endpoint's key; it is
    passed to Pi as ``OPENAI_API_KEY``. A local server that needs no key can leave it unset.
    ``public_network`` opens the task container's network during solver attempts, because Pi
    installs itself and calls the endpoint from inside the container (the old host did the same);
    set it to False to keep the task's own network policy and allowlist only the endpoint's host.
    """

    base_url: str
    model: str
    api_key_env: str = "AES_SOLVER_API_KEY"
    agent: str = PI_AGENT
    model_api: str = "openai-completions"
    agent_version: str | None = PI_VERSION
    timeout_sec: float = SOLVER_TIMEOUT_SEC
    public_network: bool = True
    agent_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "SolverEndpoint":
        """Read ``AES_SOLVER_BASE_URL`` and ``AES_SOLVER_MODEL``."""
        env = dict(os.environ if env is None else env)
        base_url = env.get("AES_SOLVER_BASE_URL", "").strip()
        model = env.get("AES_SOLVER_MODEL", "").strip()
        if not base_url or not model:
            raise ValueError("set AES_SOLVER_BASE_URL and AES_SOLVER_MODEL for the solver endpoint")
        return cls(base_url=base_url, model=model)

    @property
    def harbor_model_name(self) -> str:
        """``openai/<model>``: Harbor splits off the provider at the first slash, and only the
        ``openai`` provider reads ``OPENAI_BASE_URL``, so a served id such as ``org/model`` must
        keep the prefix too."""
        return self.model if self.model.startswith("openai/") else f"openai/{self.model}"

    def agent_spec(self) -> AgentSpec:
        key = os.environ.get(self.api_key_env, "")
        if not key:
            logger.warning("%s is not set; calling the solver endpoint with a placeholder key",
                           self.api_key_env)
        kwargs = {"model_api": self.model_api, **self.agent_kwargs}
        if self.agent_version:
            kwargs.setdefault("version", self.agent_version)
        host = urlsplit(self.base_url).hostname
        return AgentSpec(
            name=self.agent,
            model_name=self.harbor_model_name,
            kwargs=kwargs,
            env={"OPENAI_BASE_URL": self.base_url, "OPENAI_API_KEY": key or "EMPTY"},
            timeout_sec=self.timeout_sec,
            extra_allowed_hosts=() if self.public_network or not host else (host,),
        )


async def run_solver(task: HarborTask, solver: SolverEndpoint, *, n: int, harbor_env: str,
                     hints: str | None = None, jobs_dir: Path | None = None,
                     runner: TrialRunner | None = None) -> list[SolverTrajectory]:
    """Run n independent solver attempts on the task through Harbor.

    The reward of each attempt is r_S, the fraction of the environment's tests that pass.

    ``hints`` is appended to the instruction under a "Hints" heading (the diagnostic probe).
    Optional: ``jobs_dir`` is where Harbor jobs are kept (default ``<work_dir>/solver/<task>``);
    ``runner`` replaces the Harbor trial runner.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if runner is None:
        if jobs_dir is None:
            from autoenvscaling.config import Settings
            jobs_dir = Settings.load().work_dir / "solver" / f"{task.task_id}-{task.digest()}"
        runner = HarborRunner(Path(jobs_dir), env_type=harbor_env)
    instruction = with_hints(task.instruction, hints) if hints else None
    result = await runner.run(task.path, solver.agent_spec(), n_attempts=n,
                              label="hint_probe" if hints else "solver", instruction=instruction,
                              public_network=solver.public_network)
    shown = instruction if instruction is not None else task.instruction
    return [to_trajectory(task, solver, outcome, shown) for outcome in result.outcomes]


def with_hints(instruction: str, hints: str) -> str:
    return instruction.rstrip() + "\n\n" + HINTS_HEADER + hints.strip() + "\n"


def to_trajectory(task: HarborTask, solver: SolverEndpoint, outcome: TrialOutcome,
                  instruction: str) -> SolverTrajectory:
    turns = [Turn(role="user", content=instruction)]
    if outcome.trial_dir is not None:
        turns += pi_turns(_read(outcome.trial_dir / "agent" / "pi.txt"))
    output = outcome.verifier_output
    if outcome.exception_type:
        output = (output + "\n\n" if output else "") + f"[trial error] {outcome.error_summary}"
    return SolverTrajectory(
        trajectory_id=f"{task.task_id}-{outcome.trial_name or 'trial'}",
        task_id=task.task_id,
        model=solver.model,
        turns=turns,
        tests_passed=outcome.tests_passed or 0,
        tests_total=outcome.tests_total or 0,
        reward=outcome.reward if outcome.scored else None,
        verifier_output=output,
    )


def pi_turns(ndjson: str) -> list[Turn]:
    """Messages from Pi's ``--mode json`` event stream, as chat turns.

    Pi emits one ``message_end`` event per finished message: ``user``, ``assistant`` (content
    blocks of type ``text``, ``thinking`` and ``toolCall``) and ``toolResult``. Thinking blocks are
    kept inside ``<think>`` tags in the assistant content. If a Pi version reports tool results only
    as ``tool_execution_end`` events, those are used instead. The instruction itself is not
    repeated: Pi's first user message is the prompt the caller already recorded.
    """
    turns: list[Turn] = []
    fallback_results: list[Turn] = []
    saw_tool_messages = False
    skipped_prompt = False
    for raw in ndjson.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        kind = event.get("type")
        if kind == "message_end":
            message = event.get("message") or {}
            role = message.get("role")
            if role == "user":
                if not skipped_prompt:
                    skipped_prompt = True
                    continue
                turns.append(Turn(role="user", content=_text(message.get("content"))))
            elif role == "assistant":
                turns.append(_assistant_turn(message))
            elif role == "toolResult":
                saw_tool_messages = True
                turns.append(Turn(role="tool", content=_text(message.get("content")),
                                  tool_call_id=message.get("toolCallId"),
                                  name=message.get("toolName")))
        elif kind == "tool_execution_end":
            result = event.get("result")
            content = _text(result.get("content")) if isinstance(result, dict) else _text(result)
            fallback_results.append(Turn(role="tool", content=content,
                                         tool_call_id=event.get("toolCallId"),
                                         name=event.get("toolName")))
    if saw_tool_messages or not fallback_results:
        return turns
    return _interleave(turns, fallback_results)


def _assistant_turn(message: dict[str, Any]) -> Turn:
    parts: list[str] = []
    calls: list[dict[str, Any]] = []
    content = message.get("content")
    blocks = content if isinstance(content, list) else [{"type": "text", "text": content or ""}]
    for block in blocks:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "thinking":
            thought = block.get("thinking") or block.get("text") or ""
            if thought.strip():
                parts.append(f"<think>\n{thought.strip()}\n</think>")
        elif kind == "text":
            if (block.get("text") or "").strip():
                parts.append(block["text"])
        elif kind == "toolCall":
            calls.append({"id": block.get("id"), "type": "function",
                          "function": {"name": block.get("name"),
                                       "arguments": json.dumps(block.get("arguments") or {})}})
    return Turn(role="assistant", content="\n\n".join(parts), tool_calls=calls)


def _interleave(turns: list[Turn], results: list[Turn]) -> list[Turn]:
    by_id = {r.tool_call_id: r for r in results if r.tool_call_id}
    out: list[Turn] = []
    for turn in turns:
        out.append(turn)
        for call in turn.tool_calls:
            result = by_id.pop(call.get("id"), None)
            if result is not None:
                out.append(result)
    out.extend(r for r in results if r.tool_call_id in by_id)
    return out


def _text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [b.get("text", "") for b in content
                 if isinstance(b, dict) and b.get("type") == "text"]
        if texts:
            return "\n".join(texts)
    return json.dumps(content, ensure_ascii=False)


def _read(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def summarize_solution(solve_sh: str) -> list[str]:
    """High-level steps of a reference solution: each command and its first argument.

    Ported from calibration_loop.summarize_solution. Comments, the shebang, blank lines, shell
    keywords and leading ``VAR=value`` assignments are dropped, and repeated steps are kept once,
    so the hints outline the approach without revealing the literal script.
    """
    steps: list[str] = []
    seen: set[str] = set()
    for raw in solve_sh.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        while tokens and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tokens[0]):
            tokens = tokens[1:]
        if not tokens:
            continue
        command = tokens[0].lstrip("&|;()")
        if not command or command in {"then", "else", "fi", "do", "done", "{", "}"}:
            continue
        step = f"{command} {tokens[1] if len(tokens) > 1 else ''}".strip()
        if step not in seen:
            seen.add(step)
            steps.append(step)
    return steps


def reference_hints(task: HarborTask) -> str | None:
    """Numbered hint steps from the task's reference solution, or None if there are none."""
    solve = task.path / "solution" / "solve.sh"
    if not solve.is_file():
        return None
    steps = summarize_solution(solve.read_text(errors="replace"))
    if not steps:
        return None
    return "\n".join(f"{i}. {step}" for i, step in enumerate(steps, start=1))
