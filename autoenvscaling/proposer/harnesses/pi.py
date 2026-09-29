"""The Pi coding agent (``pi -p --mode json``), the harness for all other models.

Pi talks to any OpenAI-compatible chat-completions endpoint. The endpoint comes from the
environment, not from a table of model routes:

    AES_PROPOSER_BASE_URL        endpoint base URL (required), e.g. an SGLang or vLLM server
    AES_PROPOSER_API_KEY         bearer key (set to any placeholder for keyless servers)
    AES_PROPOSER_CONTEXT_WINDOW  prompt budget Pi assumes (default 262144)
    AES_PROPOSER_MAX_TOKENS      per-response output limit (default 32768)
    AES_PROPOSER_REASONING       "1" to declare reasoning output, so Pi keeps and replays the
                                 model's ``reasoning_content`` (default off; some servers reject
                                 the resulting requests)
    AES_PROPOSER_HTTP_IDLE_TIMEOUT_MS   time Pi waits for a response to start (optional)

Configuration lives in an episode-local ``PI_CODING_AGENT_DIR`` so concurrent episodes never
share state. The proposer profile keeps Pi's context compaction on for long authoring runs.

NDJSON events used here: ``message_end`` with an assistant message (content blocks ``thinking``,
``text``, ``toolCall``, plus per-call ``usage``), and tool results from either
``tool_execution_end`` or a ``message_end`` whose message has role ``toolResult``.
``message_update`` events are the token-by-token stream; they are dropped from archived logs.
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

PROVIDER = "aes"
DEFAULT_CONTEXT_WINDOW = 262_144
DEFAULT_MAX_TOKENS = 32_768


def _flag(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def models_json(model: str, base_url: str, *, context_window: int = DEFAULT_CONTEXT_WINDOW,
                max_tokens: int = DEFAULT_MAX_TOKENS, reasoning: bool = False) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": model,
        "name": model,
        "input": ["text"],
        "contextWindow": context_window,
        "maxTokens": max_tokens,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
    }
    if reasoning:
        entry["reasoning"] = True
    return {"providers": {PROVIDER: {
        "baseUrl": base_url,
        "api": "openai-completions",
        "apiKey": "$AES_PROPOSER_API_KEY",
        "authHeader": True,
        "models": [entry],
    }}}


def _result_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict) and isinstance(result.get("content"), list):
        parts = [b.get("text", "") for b in result["content"]
                 if isinstance(b, dict) and b.get("type") == "text"]
        if parts:
            return "".join(parts)
    if isinstance(result, list):
        parts = [b.get("text", "") for b in result if isinstance(b, dict) and b.get("type") == "text"]
        if parts:
            return "".join(parts)
    return json.dumps(result, ensure_ascii=False)


class PiHarness(Harness):
    name = "pi"
    credential_env = ("AES_PROPOSER_BASE_URL", "AES_PROPOSER_API_KEY",
                      "AES_PROPOSER_CONTEXT_WINDOW", "AES_PROPOSER_MAX_TOKENS",
                      "AES_PROPOSER_REASONING", "AES_PROPOSER_HTTP_IDLE_TIMEOUT_MS")

    def prepare(self, *, model: str, prompt_file: str, run_dir: str, workdir: str,
                env: Mapping[str, str]) -> HarnessLaunch:
        base_url = env.get("AES_PROPOSER_BASE_URL")
        if not base_url:
            raise RuntimeError("pi needs AES_PROPOSER_BASE_URL (an OpenAI-compatible endpoint)")
        agent_dir = f"{run_dir}/pi"
        config = models_json(
            model, base_url,
            context_window=int(env.get("AES_PROPOSER_CONTEXT_WINDOW") or DEFAULT_CONTEXT_WINDOW),
            max_tokens=int(env.get("AES_PROPOSER_MAX_TOKENS") or DEFAULT_MAX_TOKENS),
            reasoning=_flag(env.get("AES_PROPOSER_REASONING")))
        settings: dict[str, Any] = {"compaction": {"enabled": True}}
        if env.get("AES_PROPOSER_HTTP_IDLE_TIMEOUT_MS"):
            settings["httpIdleTimeoutMs"] = int(env["AES_PROPOSER_HTTP_IDLE_TIMEOUT_MS"])
        argv = ["pi", "-p", "--no-session", "--mode", "json", "--provider", PROVIDER,
                "--model", model]
        launch_env = {"PI_CODING_AGENT_DIR": agent_dir, "PI_SKIP_VERSION_CHECK": "1",
                      "PI_TELEMETRY": "0"}
        if not env.get("AES_PROPOSER_API_KEY"):
            launch_env["AES_PROPOSER_API_KEY"] = "EMPTY"  # Pi requires a key even if unused
        return HarnessLaunch(
            command=f"{shlex.join(argv)} {prompt_arg(prompt_file)}",
            env=launch_env,
            files={f"{agent_dir}/models.json": json.dumps(config, indent=2).encode(),
                   f"{agent_dir}/settings.json": json.dumps(settings).encode()},
        )

    def compact_log(self, log: str) -> str:
        """Drop the token-by-token ``message_update`` events (they dominate the log's size)."""
        kept = []
        for line in log.splitlines():
            if not line.strip():
                continue
            if "message_update" in line:
                try:
                    if json.loads(line).get("type") == "message_update":
                        continue
                except (json.JSONDecodeError, AttributeError):
                    pass
            kept.append(line)
        return "\n".join(kept)

    def parse_log(self, log: str) -> ParsedLog:
        parsed = ParsedLog()
        names: dict[str, str] = {}
        answered: set[str] = set()
        usage = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
        saw_usage = False

        def add_result(call_id: str | None, name: str | None, text: str, is_error: bool) -> None:
            if call_id and call_id in answered:
                return
            if call_id:
                answered.add(call_id)
            if is_error:
                text = f"[error] {text}"
            parsed.turns.append(Turn(role="tool", content=clip(text), tool_call_id=call_id,
                                     name=name or names.get(call_id or "")))

        for event in iter_json_lines(log):
            etype = event.get("type")
            message = event.get("message") or {}
            if etype == "message_end" and message.get("role") == "assistant":
                text, reasoning, calls = "", [], []
                for block in message.get("content") or []:
                    kind = block.get("type")
                    if kind == "thinking":
                        reasoning.append(block.get("thinking") or block.get("text") or "")
                    elif kind == "text":
                        text += block.get("text", "")
                    elif kind == "toolCall":
                        call_id = block.get("id") or f"call_{len(names)}"
                        names[call_id] = block.get("name") or ""
                        calls.append(tool_call(call_id, block.get("name") or "",
                                               block.get("arguments") or {}))
                parsed.turns.append(Turn(role="assistant", content=with_reasoning(text, reasoning),
                                         tool_calls=calls))
                u = message.get("usage") or {}
                if u:
                    saw_usage = True
                    context = sum(token_count(u.get(k)) for k in ("input", "cacheRead", "cacheWrite"))
                    # Each call re-reads the whole history, so the prompt size is a maximum, not
                    # a sum; outputs and cost add up.
                    usage["input_tokens"] = max(usage["input_tokens"], context)
                    usage["output_tokens"] += token_count(u.get("output"))
                    try:
                        usage["cost_usd"] = round(
                            usage["cost_usd"] + float((u.get("cost") or {}).get("total") or 0), 6)
                    except (TypeError, ValueError):
                        pass
                if message.get("stopReason") == "error":
                    parsed.error = str(message.get("errorMessage") or "model error")
                elif message.get("stopReason") in ("stop", "length"):
                    parsed.completed = True
            elif etype == "message_end" and message.get("role") == "toolResult":
                add_result(message.get("toolCallId"), message.get("toolName"),
                           _result_text(message.get("content")), bool(message.get("isError")))
            elif etype == "tool_execution_end":
                add_result(event.get("toolCallId"), event.get("toolName"),
                           _result_text(event.get("result")), bool(event.get("isError")))
        if saw_usage:
            parsed.usage = usage
        return parsed
