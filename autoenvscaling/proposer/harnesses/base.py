"""What a proposer harness is: a coding-agent CLI run headless inside the proposer sandbox.

A harness does two things. Before the episode, ``prepare`` returns the shell command that runs
the agent on the prompt (its stdout is the agent's native event log), the environment it needs,
and any config files to write first. After the episode, ``parse_log`` turns that native log into
the ``Turn`` list stored in the ``ProposerTrajectory``.

Turn conventions shared by all harnesses:

* The episode prompt is the first turn (``role="user"``); ``run_episode`` adds it, so parsers
  return only what follows.
* An assistant turn carries the visible text in ``content``. Reasoning, when the harness exposes
  it, is prepended as ``<think>...</think>`` so it survives into fine-tuning data.
* Tool calls use the OpenAI chat format:
  ``{"id": ..., "type": "function", "function": {"name": ..., "arguments": "<json>"}}``.
* Each tool result is its own turn: ``role="tool"``, ``tool_call_id`` and ``name`` set.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from autoenvscaling.trajectories.schema import Turn

# Tool outputs longer than this are clipped in the parsed turns (the raw log keeps everything).
MAX_TOOL_OUTPUT_CHARS = 50_000


@dataclass
class HarnessLaunch:
    """What run_episode needs to start the agent inside the sandbox."""
    command: str                                        # shell command; stdout is the native log
    env: dict[str, str] = field(default_factory=dict)   # extra environment for the command
    files: dict[str, bytes] = field(default_factory=dict)  # absolute sandbox path -> content


@dataclass
class ParsedLog:
    turns: list[Turn] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    completed: bool = False        # the harness reported a normal end of run
    error: str | None = None       # the harness reported an error


class Harness(ABC):
    name: ClassVar[str]
    # Host environment variables forwarded into the sandbox when they are set.
    credential_env: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def prepare(self, *, model: str, prompt_file: str, run_dir: str, workdir: str,
                env: Mapping[str, str]) -> HarnessLaunch:
        """Build the launch for one episode.

        Args:
            model: Model name as the harness's provider expects it.
            prompt_file: Absolute sandbox path of the prompt (already written).
            run_dir: Absolute sandbox path for episode-local harness state and config.
            workdir: The proposer's workspace root (the command's working directory).
            env: The environment the episode will forward (host credentials included), for
                harnesses whose config depends on it.
        """

    @abstractmethod
    def parse_log(self, log: str) -> ParsedLog:
        """Turn the harness's native stdout log into turns and usage."""

    def compact_log(self, log: str) -> str:
        """The native log as it should be archived (harnesses may drop redundant events)."""
        return log


def tool_call(call_id: str, name: str, arguments: Any) -> dict[str, Any]:
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def with_reasoning(text: str, reasoning: list[str]) -> str:
    thoughts = "\n\n".join(r.strip() for r in reasoning if r and r.strip())
    if not thoughts:
        return text
    return f"<think>\n{thoughts}\n</think>\n\n{text}" if text else f"<think>\n{thoughts}\n</think>"


def clip(text: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[{len(text) - limit} more characters clipped]"


def iter_json_lines(log: str):
    """Yield each JSON object in a JSON-lines log, skipping blank or malformed lines."""
    for line in log.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            yield value


def token_count(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError, OverflowError):
        return 0


def prompt_arg(prompt_file: str) -> str:
    """Shell word that expands to the prompt text as a single argument."""
    return f'"$(cat {prompt_file})"'
