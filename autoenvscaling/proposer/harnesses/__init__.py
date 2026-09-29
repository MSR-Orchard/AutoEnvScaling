"""Proposer harnesses: Claude Code for Claude models, Codex for GPT models, Pi for all others."""

from __future__ import annotations

from autoenvscaling.proposer.harnesses.base import Harness, HarnessLaunch, ParsedLog
from autoenvscaling.proposer.harnesses.claude_code import ClaudeCodeHarness
from autoenvscaling.proposer.harnesses.codex import CodexHarness
from autoenvscaling.proposer.harnesses.pi import PiHarness

HARNESSES: dict[str, type[Harness]] = {
    ClaudeCodeHarness.name: ClaudeCodeHarness,
    CodexHarness.name: CodexHarness,
    PiHarness.name: PiHarness,
}


def harness_for_model(model: str) -> str:
    """The paper's pairing: Claude Code for Claude, Codex for GPT, Pi for everything else."""
    bare = model.split("/")[-1].lower()
    if bare.startswith("claude"):
        return ClaudeCodeHarness.name
    if bare.startswith("gpt"):
        return CodexHarness.name
    return PiHarness.name


def get_harness(name: str, model: str | None = None) -> Harness:
    """Instantiate a harness by name; "auto" picks one from the model name."""
    if name == "auto":
        if not model:
            raise ValueError("harness 'auto' needs a model name")
        name = harness_for_model(model)
    try:
        return HARNESSES[name]()
    except KeyError:
        raise ValueError(f"unknown harness {name!r}; expected one of {sorted(HARNESSES)} "
                         "or 'auto'") from None


__all__ = ["HARNESSES", "ClaudeCodeHarness", "CodexHarness", "Harness", "HarnessLaunch",
           "ParsedLog", "PiHarness", "get_harness", "harness_for_model"]
