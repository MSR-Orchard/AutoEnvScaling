"""Codex (``codex exec --json``), the harness for GPT models.

Codex reads an episode-local ``$CODEX_HOME/config.toml`` that points it at an OpenAI-compatible
Responses endpoint: ``OPENAI_BASE_URL`` if set, otherwise the OpenAI API, authenticated with
``OPENAI_API_KEY``. ``AES_CODEX_WIRE_API=chat`` switches to the chat-completions wire format for
endpoints without the Responses API.

JSONL events used here: ``item.completed`` with item types ``agent_message``, ``reasoning``,
``command_execution``, ``file_change``, ``mcp_tool_call``, ``web_search`` and ``error``;
``turn.completed`` (usage); ``turn.failed`` and ``error`` (failures).
"""

from __future__ import annotations

import json
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

DEFAULT_BASE_URL = "https://api.openai.com/v1"


def render_config(model: str, base_url: str, wire_api: str = "responses") -> str:
    if wire_api not in ("responses", "chat"):
        raise ValueError(f"wire_api must be 'responses' or 'chat', got {wire_api!r}")
    return (
        f"model = {json.dumps(model)}\n"
        'model_provider = "aes"\n'
        "\n"
        "[model_providers.aes]\n"
        'name = "OpenAI-compatible endpoint"\n'
        f"base_url = {json.dumps(base_url)}\n"
        'env_key = "OPENAI_API_KEY"\n'
        f"wire_api = {json.dumps(wire_api)}\n"
    )


class CodexHarness(Harness):
    name = "codex"
    credential_env = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "AES_CODEX_WIRE_API")

    def prepare(self, *, model: str, prompt_file: str, run_dir: str, workdir: str,
                env: Mapping[str, str]) -> HarnessLaunch:
        if not env.get("OPENAI_API_KEY"):
            raise RuntimeError("codex needs OPENAI_API_KEY")
        codex_home = f"{run_dir}/codex"
        config = render_config(model, env.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL,
                               env.get("AES_CODEX_WIRE_API") or "responses")
        argv = ["codex", "exec", "--json", "--skip-git-repo-check",
                "--dangerously-bypass-approvals-and-sandbox", "--model", model]
        return HarnessLaunch(
            command=f"{shlex.join(argv)} {prompt_arg(prompt_file)}",
            env={"CODEX_HOME": codex_home},
            files={f"{codex_home}/config.toml": config.encode()},
        )

    def parse_log(self, log: str) -> ParsedLog:
        parsed = ParsedLog()
        reasoning: list[str] = []
        totals = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
        saw_usage = False

        def call(item_id: str, name: str, arguments: Any, output: str) -> None:
            nonlocal reasoning
            parsed.turns.append(Turn(role="assistant", content=with_reasoning("", reasoning),
                                     tool_calls=[tool_call(item_id, name, arguments)]))
            parsed.turns.append(Turn(role="tool", content=clip(output), tool_call_id=item_id,
                                     name=name))
            reasoning = []

        for event in iter_json_lines(log):
            etype = event.get("type")
            if etype == "item.completed":
                item = event.get("item") or {}
                kind = item.get("type")
                item_id = str(item.get("id") or f"item_{len(parsed.turns)}")
                if kind == "agent_message":
                    parsed.turns.append(Turn(role="assistant",
                                             content=with_reasoning(item.get("text") or "",
                                                                    reasoning)))
                    reasoning = []
                elif kind == "reasoning":
                    reasoning.append(item.get("text") or "")
                elif kind == "command_execution":
                    output = item.get("aggregated_output") or ""
                    if item.get("exit_code") not in (None, 0):
                        output = f"{output}\n[exit code {item.get('exit_code')}]"
                    call(item_id, "shell", {"command": item.get("command")}, output)
                elif kind == "file_change":
                    call(item_id, "apply_patch", {"changes": item.get("changes") or []},
                         str(item.get("status") or "completed"))
                elif kind == "mcp_tool_call":
                    name = ".".join(p for p in (item.get("server"), item.get("tool")) if p)
                    result = item.get("result")
                    output = result if isinstance(result, str) else json.dumps(result)
                    if item.get("error"):
                        output = f"[error] {item.get('error')}"
                    call(item_id, name or "mcp_tool", item.get("arguments") or {}, output)
                elif kind == "web_search":
                    call(item_id, "web_search", {"query": item.get("query")}, "")
                elif kind == "error":
                    parsed.error = str(item.get("message") or "error")
            elif etype == "turn.completed":
                usage = event.get("usage") or {}
                saw_usage = True
                parsed.completed = True
                for key in totals:
                    totals[key] += token_count(usage.get(key))
            elif etype == "turn.failed":
                parsed.error = str((event.get("error") or {}).get("message") or "turn failed")
            elif etype == "error":
                parsed.error = str(event.get("message") or "error")
        if reasoning:
            parsed.turns.append(Turn(role="assistant", content=with_reasoning("", reasoning)))
        if saw_usage:
            parsed.usage = dict(totals)
        return parsed
