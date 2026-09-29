"""Claude Code (``claude -p``), the harness for Claude models.

Runs headless with ``--output-format stream-json --verbose`` and only the file and shell tools
(Read, Glob, Grep, Write, Edit, Bash). Claude Code's own WebSearch and WebFetch are left off so
that web access goes through the sandbox's ``web_search``/``web_fetch`` tools, which apply the
benchmark blocklist. Authentication: ``ANTHROPIC_API_KEY`` (or ``CLAUDE_CODE_OAUTH_TOKEN``), and
``ANTHROPIC_BASE_URL`` for a compatible gateway.

stream-json events used here: ``assistant`` (content blocks ``text``, ``thinking``, ``tool_use``;
one event per block, grouped by ``message.id``), ``user`` (``tool_result`` blocks keyed by
``tool_use_id``), and ``result`` (final usage and cost).
"""

from __future__ import annotations

import shlex
from collections.abc import Mapping
from typing import Any

from autoenvscaling.proposer.harnesses.base import (
    Harness,
    HarnessLaunch,
    ParsedLog,
    clip,
    iter_json_lines,
    prompt_arg,
    token_count,
    tool_call,
    with_reasoning,
)
from autoenvscaling.trajectories.schema import Turn

ALLOWED_TOOLS = ("Read", "Glob", "Grep", "Write", "Edit", "Bash")


def _result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif block.get("type") == "image":
                parts.append("[image]")
        return "".join(parts)
    return ""


class ClaudeCodeHarness(Harness):
    name = "claude-code"
    credential_env = ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "CLAUDE_CODE_OAUTH_TOKEN")

    def prepare(self, *, model: str, prompt_file: str, run_dir: str, workdir: str,
                env: Mapping[str, str]) -> HarnessLaunch:
        if not (env.get("ANTHROPIC_API_KEY") or env.get("CLAUDE_CODE_OAUTH_TOKEN")):
            raise RuntimeError("claude-code needs ANTHROPIC_API_KEY (or CLAUDE_CODE_OAUTH_TOKEN)")
        argv = ["claude", "-p", "--output-format", "stream-json", "--verbose",
                "--model", model, "--allowedTools", ",".join(ALLOWED_TOOLS),
                "--disable-slash-commands", "--strict-mcp-config"]
        command = f"{shlex.join(argv)} {prompt_arg(prompt_file)}"
        return HarnessLaunch(command=command, env={
            "CLAUDE_CONFIG_DIR": f"{run_dir}/claude",
            "DISABLE_AUTOUPDATER": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        })

    def parse_log(self, log: str) -> ParsedLog:
        parsed = ParsedLog()
        tool_names: dict[str, str] = {}
        current: Turn | None = None
        current_id: str | None = None
        reasoning: list[str] = []
        message_usage: dict[str, dict] = {}

        def flush() -> None:
            nonlocal current, current_id, reasoning
            if current is not None:
                current.content = with_reasoning(current.content, reasoning)
                parsed.turns.append(current)
            current, current_id, reasoning = None, None, []

        for event in iter_json_lines(log):
            etype = event.get("type")
            if etype == "assistant":
                message = event.get("message") or {}
                msg_id = message.get("id")
                if current is None or msg_id is None or msg_id != current_id:
                    flush()
                    current, current_id = Turn(role="assistant", content=""), msg_id
                if msg_id and isinstance(message.get("usage"), dict):
                    message_usage[msg_id] = message["usage"]
                for block in message.get("content") or []:
                    kind = block.get("type")
                    if kind == "text":
                        current.content += block.get("text", "")
                    elif kind == "thinking":
                        reasoning.append(block.get("thinking") or "")
                    elif kind == "tool_use":
                        call_id = block.get("id") or f"call_{len(tool_names)}"
                        tool_names[call_id] = block.get("name") or ""
                        current.tool_calls.append(
                            tool_call(call_id, block.get("name") or "", block.get("input") or {}))
            elif etype == "user":
                message = event.get("message") or {}
                content = message.get("content")
                if not isinstance(content, list):
                    continue
                for block in content:
                    if not isinstance(block, dict) or block.get("type") != "tool_result":
                        continue
                    flush()
                    call_id = block.get("tool_use_id")
                    text = _result_text(block.get("content"))
                    if block.get("is_error"):
                        text = f"[error] {text}"
                    parsed.turns.append(Turn(role="tool", content=clip(text), tool_call_id=call_id,
                                             name=tool_names.get(call_id or "")))
            elif etype == "result":
                flush()
                usage = event.get("usage") or {}
                parsed.completed = True
                parsed.usage = {
                    "input_tokens": token_count(usage.get("input_tokens")),
                    "cache_read_input_tokens": token_count(usage.get("cache_read_input_tokens")),
                    "cache_creation_input_tokens":
                        token_count(usage.get("cache_creation_input_tokens")),
                    "output_tokens": token_count(usage.get("output_tokens")),
                    "cost_usd": event.get("total_cost_usd"),
                    "num_turns": event.get("num_turns"),
                }
                if event.get("is_error"):
                    parsed.error = str(event.get("result") or event.get("subtype") or "error")
        flush()
        if not parsed.usage and message_usage:
            # No final result event (for example the run was stopped at the time budget):
            # fall back to the per-message usage the stream carried.
            parsed.usage = {
                "input_tokens": max(
                    token_count(u.get("input_tokens")) + token_count(u.get("cache_read_input_tokens"))
                    + token_count(u.get("cache_creation_input_tokens"))
                    for u in message_usage.values()),
                "output_tokens": sum(token_count(u.get("output_tokens"))
                                     for u in message_usage.values()),
            }
        return parsed
