"""The recording proxy: OpenAI-shaped outward, SGLang ``/generate`` inward.

Pi runs inside the task container and speaks ``/v1/chat/completions``. Training needs the
engine's own per-token log-probabilities and token ids, never recomputed, and the OpenAI wire
format has no token id at all (re-tokenizing the returned text is exactly the recomputation the
carry forbids). So this proxy renders the conversation with the policy's chat template, calls
SGLang's native ``/generate`` with ``return_logprob``, records ``(ids, logprobs, loss_mask)``
per episode, and answers Pi in OpenAI shape. The rollout driver reads the recording back
(``GET /episodes/{id}?full=1``) and copies it onto the Miles sample.

Three invariants are enforced as checks, not conventions:

1. every render preserves earlier thinking (``preserve_thinking=True``), otherwise a second
   user turn silently drops the recorded reasoning tokens and the re-render stops extending
   the recording;
2. tool-call ``arguments`` reach the template as dicts;
3. every render must *extend* the recorded prefix. A full re-render legitimately ends one
   newline longer than the recorded stream, so the comparison is prefix-wise.

The park gate
-------------
The colocated flip puts the engines to sleep for the optimizer step. Pi owns its loop and
cannot be told to wait, so instead of killing it the proxy holds its in-flight request until
the engines wake: from Pi's side the call was merely slow, and the conversation is byte for
byte the one an uninterrupted episode would have had. A global gate covers the weight swap;
per-episode gates let the rollout admit resumes one episode at a time. A generation the agent
never read is *provisional*: when the next request is shorter than, or diverges after, the
mark where that generation began, it is rolled back, because a turn with no tool result behind
it must not carry gradient.

Run it with ``python -m autoenvscaling.training.carry proxy`` (needs ``fastapi``, ``uvicorn``,
``transformers`` and, for tool-call parsing, SGLang's parser: the Miles image has all of them).
Configuration is read from ``AES_PROXY_*`` variables; see ``ProxyConfig``.

This is the research run's ``pi_logprob_proxy.py`` with the engine, tokenizer and tool parser
behind small interfaces so the recording and gate logic is testable without a GPU.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator
from dataclasses import dataclass, field, fields
from typing import Any, Protocol

try:   # the web framework is optional: the recording and gate logic needs none of it
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse
except ImportError:  # pragma: no cover - exercised only without the `carry` extra
    FastAPI = HTTPException = Request = JSONResponse = StreamingResponse = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

ENV_PREFIX = "AES_PROXY_"
CARRY_LIFECYCLE_VERSION = 1


class ProxyError(Exception):
    """An HTTP-level error of the proxy; the app layer maps it to a status code."""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


class GenerationAborted(Exception):
    """The engine aborted a request (the rollout cut); its partial output must not reach Pi."""


# ---- configuration ------------------------------------------------------------------------------------
@dataclass
class ProxyConfig:
    hf_dir: str = ""                         # tokenizer of the SERVED model (required)
    sglang_url: str = ""                     # pushed by the driver through POST /config
    sglang_key: str = ""
    served_id: str = "policy"
    tool_parser: str = "qwen3_coder"         # must match the engines' --tool-call-parser
    api_key: str = ""                        # bearer for the public surface; empty = open
    max_context: int = 96000                 # below rollout_max_context_len: the last tool result lands after
    max_turns: int = 128
    min_room: int = 512
    max_connections: int = 1024
    max_park_s: float = 7200.0
    stream_heartbeat_s: float = 10.0
    config_cache: str = ""                   # survives proxy restarts
    backend: str = "sglang"                  # or "mock" (no GPU)
    trace_path: str = ""
    host: str = "0.0.0.0"
    port: int = 30000

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> ProxyConfig:
        env = os.environ if env is None else env
        values: dict[str, Any] = {}
        for f in fields(cls):
            raw = env.get(ENV_PREFIX + f.name.upper())
            if raw is None:
                continue
            default = f.default
            if isinstance(default, bool):
                values[f.name] = raw.lower() in {"1", "true", "yes"}
            elif isinstance(default, int):
                values[f.name] = int(raw)
            elif isinstance(default, float):
                values[f.name] = float(raw)
            else:
                values[f.name] = raw
        return cls(**values)

    def problems(self) -> list[str]:
        out = []
        if self.backend not in ("sglang", "mock"):
            out.append(f"backend must be sglang or mock, got {self.backend!r}")
        if not 0 < self.stream_heartbeat_s < 60:
            out.append("stream_heartbeat_s must be between 0 and 60 seconds")
        if self.max_connections <= 0:
            out.append("max_connections must be positive")
        return out


# ---- the recording ----------------------------------------------------------------------------------
@dataclass
class Episode:
    """Per-episode recording. This is what training consumes."""

    episode_id: str
    tokens: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    loss_mask: list[int] = field(default_factory=list)
    weight_versions: list[Any] = field(default_factory=list)   # one per generation, if the engine reports it
    turns: int = 0
    tool_results_seen: int = 0
    last_turn_start: int = 0      # where the provisional generation began
    rollbacks: int = 0
    prompt_len: int = 0           # length of the FIRST prompt: Miles' prompt/response split
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    closed: bool = False
    error: str | None = None
    stop_reason: str | None = None

    def rollback_uncommitted(self) -> int:
        """Drop the provisional generation. Returns the number of tokens discarded."""
        n = len(self.tokens) - self.last_turn_start
        if n <= 0:
            return 0
        del self.tokens[self.last_turn_start:]
        del self.logprobs[self.last_turn_start:]
        del self.loss_mask[self.last_turn_start:]
        if self.weight_versions:
            self.weight_versions.pop()
        self.turns = max(0, self.turns - 1)
        self.rollbacks += 1
        return n

    def extend_prompt(self, ids: list[int]) -> None:
        """Prompt and injected tokens: recorded but not trained on."""
        if not self.tokens:
            self.prompt_len = len(ids)
        self.tokens.extend(ids)
        self.logprobs.extend([0.0] * len(ids))
        self.loss_mask.extend([0] * len(ids))

    def extend_generated(self, ids: list[int], lps: list[float], weight_version: Any = None) -> None:
        """Model-generated tokens: the ones that carry gradient."""
        if len(ids) != len(lps):
            raise ProxyError(502, "id/logprob length mismatch from the engine")
        self.tokens.extend(ids)
        self.logprobs.extend(lps)
        self.loss_mask.extend([1] * len(ids))
        if weight_version is not None:
            self.weight_versions.append(weight_version)

    def consistent(self) -> bool:
        return len(self.tokens) == len(self.logprobs) == len(self.loss_mask)

    def summary(self, full: bool = False) -> dict[str, Any]:
        out = {"episode_id": self.episode_id, "n_tokens": len(self.tokens),
               "n_trained_tokens": sum(self.loss_mask), "turns": self.turns,
               "tool_results_seen": self.tool_results_seen, "rollbacks": self.rollbacks,
               "prompt_len": self.prompt_len, "error": self.error, "stop_reason": self.stop_reason,
               "weight_versions": list(self.weight_versions)}
        if full:
            if not self.consistent():
                raise ProxyError(500, f"recording inconsistent: tokens={len(self.tokens)} "
                                      f"logprobs={len(self.logprobs)} mask={len(self.loss_mask)}")
            out.update(tokens=self.tokens, logprobs=self.logprobs, loss_mask=self.loss_mask)
        return out


def observe_prompt(ep: Episode, input_ids: list[int], *, max_context: int,
                   decode=None, trace=None) -> None:
    """Record this request's prompt, enforcing invariant 3 and the park rollback rule."""
    if not ep.tokens:
        ep.last_turn_start = 0
        if len(input_ids) <= max_context:
            ep.extend_prompt(input_ids)
        return
    n = min(len(ep.tokens), len(input_ids))
    first = next((i for i in range(n) if input_ids[i] != ep.tokens[i]), n)
    # Shorter but matching: the client has fewer turns than recorded (a park artefact). Left
    # alone, the dangling generation stays and the next one is appended straight after it, a
    # token stream that never existed. Roll back on length, not just on mismatch.
    if first == n and len(input_ids) < len(ep.tokens):
        dropped = ep.rollback_uncommitted()
        if trace:
            trace({"ep": ep.episode_id, "event": "rollback_short", "dropped_tokens": dropped})
        n = min(len(ep.tokens), len(input_ids))
        first = next((i for i in range(n) if input_ids[i] != ep.tokens[i]), n)
    if first < n and first >= ep.last_turn_start:
        # Diverged at or after the provisional generation: the agent never saw that turn.
        dropped = ep.rollback_uncommitted()
        if trace:
            trace({"ep": ep.episode_id, "event": "rollback", "dropped_tokens": dropped})
        n = min(len(ep.tokens), len(input_ids))
        first = next((i for i in range(n) if input_ids[i] != ep.tokens[i]), n)
    if first < n:
        recorded = decode(ep.tokens[first:first + 24]) if decode else "?"
        rendered = decode(input_ids[first:first + 24]) if decode else "?"
        raise ProxyError(409, f"render diverged from recorded tokens at index {first}; refusing "
                              f"to corrupt the sample (recorded={recorded!r} rendered={rendered!r})")
    # The mark sits BEFORE this request's prompt tokens: a park leaves behind both the dangling
    # generation and the generation prompt that preceded it, and rollback must reach both.
    ep.last_turn_start = len(ep.tokens)
    if len(input_ids) <= max_context:
        ep.extend_prompt(input_ids[len(ep.tokens):])


# ---- the gates --------------------------------------------------------------------------------------
class Gates:
    """A global gate (the weight swap) ANDed with per-episode gates (Miles' per-episode admission)."""

    def __init__(self, max_park_s: float = 7200.0):
        self.max_park_s = max_park_s
        self.global_open = asyncio.Event()
        self.global_open.set()
        self.global_since: float | None = None
        self.episode_open: dict[str, asyncio.Event] = {}
        self.episode_since: dict[str, float] = {}
        self.changed = asyncio.Event()
        self.closed_episodes: OrderedDict[str, None] = OrderedDict()
        self.inflight: set[asyncio.Task] = set()

    def episode_gate(self, episode_id: str) -> asyncio.Event:
        if episode_id in self.closed_episodes:
            raise ProxyError(410, "episode already released")
        gate = self.episode_open.get(episode_id)
        if gate is None:
            gate = asyncio.Event()
            gate.set()
            self.episode_open[episode_id] = gate
        return gate

    def park(self, episode_id: str | None = None) -> dict:
        if episode_id:
            gate = self.episode_gate(episode_id)
            if gate.is_set():
                gate.clear()
                self.episode_since[episode_id] = time.time()
                self.changed.set()
            return {"parked": True, "scope": episode_id}
        if self.global_open.is_set():
            self.global_open.clear()
            self.global_since = time.time()
            self.changed.set()
        return {"parked": True, "scope": "global"}

    def unpark(self, episode_id: str | None = None) -> dict:
        if episode_id:
            gate = self.episode_gate(episode_id)
            held = time.time() - self.episode_since.pop(episode_id, time.time())
            gate.set()
            self.changed.set()
            return {"parked": False, "scope": episode_id, "held_sec": round(held, 1)}
        held = time.time() - self.global_since if self.global_since else 0.0
        self.global_open.set()
        self.global_since = None
        self.changed.set()
        return {"parked": False, "scope": "global", "held_sec": round(held, 1)}

    def open_for(self, episode_id: str) -> bool:
        return self.global_open.is_set() and self.episode_gate(episode_id).is_set()

    async def quiesce(self, timeout_s: float = 25.0) -> dict:
        """Acknowledge the cut only after every admitted engine call has settled."""
        if self.global_open.is_set():
            raise ProxyError(409, "close the global gate before quiescing")
        pending = {t for t in self.inflight if not t.done()}
        if pending:
            _, pending = await asyncio.wait(pending, timeout=timeout_s)
        if pending:
            raise ProxyError(503, f"{len(pending)} engine calls still active after abort")
        return {"quiescent": True}

    def state(self) -> dict:
        now = time.time()
        return {"global_parked": not self.global_open.is_set(),
                "global_held_sec": round(now - self.global_since, 1) if self.global_since else 0.0,
                "episodes_parked": [e for e, g in self.episode_open.items() if not g.is_set()],
                "episode_held_sec": {e: round(now - t, 1) for e, t in self.episode_since.items()}}

    def release(self, episode_id: str) -> None:
        self.closed_episodes[episode_id] = None
        self.closed_episodes.move_to_end(episode_id)
        while len(self.closed_episodes) > 8192:
            self.closed_episodes.popitem(last=False)
        self.episode_open.pop(episode_id, None)
        self.episode_since.pop(episode_id, None)
        self.changed.set()

    async def await_open(self, episode_id: str) -> None:
        """Wait until both gates are open, rechecking after every change (including release)."""
        deadline = time.monotonic() + self.max_park_s
        while True:
            if episode_id in self.closed_episodes:
                raise ProxyError(410, "episode released while waiting")
            if self.open_for(episode_id):
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProxyError(503, f"park exceeded {self.max_park_s}s")
            self.changed.clear()
            try:
                await asyncio.wait_for(self.changed.wait(), timeout=remaining)
            except TimeoutError:
                raise ProxyError(503, f"park exceeded {self.max_park_s}s") from None


# ---- engines, rendering, tool parsing ------------------------------------------------------------------
@dataclass
class Generation:
    ids: list[int]
    logprobs: list[float]
    text: str
    weight_version: Any = None


class Engine(Protocol):
    async def generate(self, input_ids: list[int], max_new: int, temperature: float) -> Generation: ...


class Renderer(Protocol):
    def render(self, messages: list[dict], tools: list[dict] | None) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...

    def encode(self, text: str) -> list[int]: ...


class ToolParser(Protocol):
    def parse(self, text: str, tools: list[dict]) -> tuple[str, list[dict]]: ...


def normalise_messages(messages: list[dict]) -> list[dict]:
    """Make Pi's messages safe for the template: split inline ``<think>`` into
    ``reasoning_content`` and coerce tool-call arguments to dicts (invariant 2)."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "assistant":
            content = m.get("content") or ""
            if "reasoning_content" not in m and "</think>" in content:
                think, _, rest = content.partition("</think>")
                m["reasoning_content"] = think.replace("<think>", "").strip()
                m["content"] = rest.strip()
            calls = []
            for tc in m.get("tool_calls") or []:
                tc = dict(tc)
                fn = dict(tc.get("function") or {})
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args) if args.strip() else {}
                    except json.JSONDecodeError as exc:
                        raise ProxyError(400, f"tool arguments are not valid JSON: {exc}") from exc
                tc["function"] = fn
                calls.append(tc)
            if calls:
                m["tool_calls"] = calls
        out.append(m)
    return out


def _as_id_list(out: Any) -> list[int]:
    """Coerce a tokenizer result to a plain list[int] (transformers 4 returns a list, 5 a
    BatchEncoding whose ``Encoding`` objects cannot be serialised)."""
    if isinstance(out, list):
        return [int(x) for x in out]
    ids = getattr(out, "ids", None)
    if ids is None:
        ids = out["input_ids"] if isinstance(out, dict) else getattr(out, "input_ids", None)
    if ids is None:
        raise TypeError(f"cannot coerce tokenizer output {type(out)!r} to list[int]")
    if len(ids) and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(x) for x in ids]


class ChatTemplateRenderer:
    """Renders with the served model's chat template, thinking preserved (invariant 1)."""

    def __init__(self, tokenizer: Any):
        self.tok = tokenizer

    def render(self, messages: list[dict], tools: list[dict] | None) -> list[int]:
        return _as_id_list(self.tok.apply_chat_template(
            normalise_messages(messages), tools=tools or None, add_generation_prompt=True,
            tokenize=True, preserve_thinking=True))

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids)

    def encode(self, text: str) -> list[int]:
        return _as_id_list(self.tok.encode(text, add_special_tokens=False))


class MockEngine:
    """Deterministic generations without a GPU: a short tool call so Pi's parser runs."""

    TEXT = ("planning\n</think>\n\n<tool_call>\n<function=bash>\n<parameter=command>\nls /tmp\n"
            "</parameter>\n</function>\n</tool_call>")

    def __init__(self, renderer: Renderer, text: str | None = None):
        self.renderer = renderer
        self.text = text or self.TEXT

    async def generate(self, input_ids: list[int], max_new: int, temperature: float) -> Generation:
        ids = self.renderer.encode(self.text)[:max_new]
        return Generation(ids, [-0.1] * len(ids), self.renderer.decode(ids), weight_version=0)


class SglangEngine:
    """``POST /generate`` with ``return_logprob``; yields (logprob, token_id) pairs natively."""

    def __init__(self, config: ProxyConfig, client: Any = None):
        self.config = config
        if client is None:
            import httpx
            client = httpx.AsyncClient(
                timeout=httpx.Timeout(600, connect=30),
                limits=httpx.Limits(max_connections=config.max_connections,
                                    max_keepalive_connections=config.max_connections))
        self.client = client

    async def generate(self, input_ids: list[int], max_new: int, temperature: float) -> Generation:
        url = self.config.sglang_url.rstrip("/")
        if not url:
            raise ProxyError(503, "the proxy has no engine address yet; the driver pushes the "
                                  "router address to POST /config on its first episode")
        headers = {"Content-Type": "application/json"}
        if self.config.sglang_key:
            headers["Authorization"] = f"Bearer {self.config.sglang_key}"
        body = {"input_ids": input_ids,
                "sampling_params": {"temperature": temperature, "max_new_tokens": max_new},
                "return_logprob": True}
        r = await self.client.post(f"{url}/generate", json=body, headers=headers)
        r.raise_for_status()
        out = r.json()
        meta = out.get("meta_info", {})
        finish = meta.get("finish_reason")
        kind = finish.get("type") if isinstance(finish, dict) else finish
        if kind == "abort":
            raise GenerationAborted("engine request aborted")
        if kind not in ("stop", "length"):
            raise ProxyError(502, f"unexpected engine finish_reason: {finish!r}")
        otl = meta.get("output_token_logprobs") or []
        if not otl:
            raise ProxyError(502, "engine returned no output_token_logprobs")
        return Generation([e[1] for e in otl], [e[0] for e in otl], out["text"],
                          weight_version=meta.get("weight_version"))


class SglangToolParser:
    """SGLang's own parser, the same family the engines are started with. Cached per tool set."""

    def __init__(self, parser_name: str):
        self.parser_name = parser_name
        self._parsers: dict[str, Any] = {}

    def parse(self, text: str, tools: list[dict]) -> tuple[str, list[dict]]:
        from pydantic import TypeAdapter
        from sglang.srt.entrypoints.openai.protocol import Tool
        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        key = json.dumps(tools, sort_keys=True)
        parser = self._parsers.get(key)
        if parser is None:
            parser = FunctionCallParser(tools=TypeAdapter(list[Tool]).validate_python(tools),
                                        tool_call_parser=self.parser_name)
            self._parsers[key] = parser
        visible, items = parser.parse_non_stream(text)
        return visible, [{"name": item.name, "arguments": item.parameters or "{}"} for item in items]


class NoToolParser:
    def parse(self, text: str, tools: list[dict]) -> tuple[str, list[dict]]:
        return text, []


# ---- the proxy core --------------------------------------------------------------------------------
@dataclass
class ChatResult:
    message: dict[str, Any]
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int


class Proxy:
    def __init__(self, config: ProxyConfig, *, renderer: Renderer, engine: Engine,
                 tool_parser: ToolParser | None = None):
        self.config = config
        self.renderer = renderer
        self.engine = engine
        self.tool_parser = tool_parser or NoToolParser()
        self.episodes: dict[str, Episode] = {}
        self.gates = Gates(config.max_park_s)
        self.parse_failures = 0
        self.stream_requests: set[asyncio.Task] = set()
        self.stream_disconnects = 0

    # ---- configuration -------------------------------------------------------------------------
    def restore_cache(self) -> None:
        if not self.config.config_cache:
            return
        try:
            cached = json.loads(pathlib.Path(self.config.config_cache).read_text())
        except (OSError, ValueError):
            return
        self.config.sglang_url = cached.get("sglang_url") or self.config.sglang_url
        self.config.served_id = cached.get("served_id") or self.config.served_id
        self.config.max_turns = int(cached.get("max_turns", self.config.max_turns))
        self.config.max_context = int(cached.get("max_context", self.config.max_context))

    def set_backend(self, body: dict) -> dict:
        url = (body.get("sglang_url") or "").rstrip("/")
        if url:
            self.config.sglang_url = url
        if body.get("served_id"):
            self.config.served_id = body["served_id"]
        for key in ("max_turns", "max_context"):
            if key in body and int(body[key]) <= 0:
                raise ProxyError(400, f"{key} must be positive")
        self.config.max_turns = int(body.get("max_turns", self.config.max_turns))
        self.config.max_context = int(body.get("max_context", self.config.max_context))
        if self.config.config_cache:
            try:
                pathlib.Path(self.config.config_cache).write_text(json.dumps(
                    {"sglang_url": self.config.sglang_url, "served_id": self.config.served_id,
                     "max_turns": self.config.max_turns, "max_context": self.config.max_context}))
            except OSError as exc:
                logger.warning("config cache write failed: %s", exc)
        return {"sglang_url": self.config.sglang_url, "served_id": self.config.served_id}

    def describe(self) -> dict:
        return {"sglang_url": self.config.sglang_url, "served_id": self.config.served_id,
                "backend": self.config.backend, "carry_lifecycle_version": CARRY_LIFECYCLE_VERSION,
                "max_turns": self.config.max_turns, "max_context": self.config.max_context,
                "inflight_generations": len(self.gates.inflight),
                "episodes": len(self.episodes), "parse_failures": self.parse_failures,
                "stream_heartbeat_seconds": self.config.stream_heartbeat_s,
                "stream_requests": len(self.stream_requests),
                "stream_disconnects": self.stream_disconnects,
                "max_park_seconds": self.config.max_park_s}

    # ---- episodes --------------------------------------------------------------------------------
    def episode(self, episode_id: str, full: bool = False) -> dict:
        ep = self.episodes.get(episode_id)
        if ep is None:
            raise ProxyError(404, "unknown episode")
        return ep.summary(full)

    def drop(self, episode_id: str) -> dict:
        ep = self.episodes.pop(episode_id, None)
        if ep is not None:
            ep.closed = True
        self.gates.release(episode_id)
        return {"dropped": ep is not None}

    def check_auth(self, authorization: str | None) -> None:
        if not self.config.api_key:
            return
        header = authorization or ""
        token = header[7:] if header.lower().startswith("bearer ") else ""
        if token != self.config.api_key:
            raise ProxyError(401, "unauthorized")

    def _trace(self, record: dict) -> None:
        if self.config.trace_path:
            with open(self.config.trace_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")

    async def chat(self, episode_id: str, body: dict) -> ChatResult:
        """One ``/v1/ep/{episode_id}/chat/completions`` request, start to finish."""
        if episode_id in self.gates.closed_episodes:
            raise ProxyError(410, "episode already released")
        ep = self.episodes.setdefault(episode_id, Episode(episode_id))
        async with ep.lock:
            if ep.closed:
                raise ProxyError(410, "episode already released")
            try:
                return await self._chat(episode_id, body, ep)
            except ProxyError as exc:
                ep.error = exc.detail
                raise
            except Exception as exc:
                ep.error = f"{type(exc).__name__}: {exc}"
                raise

    async def _chat(self, episode_id: str, body: dict, ep: Episode) -> ChatResult:
        messages = body.get("messages") or []
        tools = body.get("tools")
        self._trace({"ep": episode_id, "n_messages": len(messages),
                     "roles": [m.get("role") for m in messages][-6:], "n_tools": len(tools or []),
                     "stream": bool(body.get("stream"))})
        ep.tool_results_seen = max(ep.tool_results_seen,
                                   sum(1 for m in messages if m.get("role") == "tool"))
        input_ids = self.renderer.render(messages, tools)
        observe_prompt(ep, input_ids, max_context=self.config.max_context,
                       decode=self.renderer.decode, trace=self._trace)

        # Context budget: a hard engine/trainer limit, not a turn policy. Out of room, tell the
        # agent to stop; everything recorded so far is a valid, merely shorter, trajectory.
        room = self.config.max_context - len(input_ids)
        want = int(body.get("max_tokens") or 2048)
        if room <= self.config.min_room or ep.turns >= self.config.max_turns:
            ep.stop_reason = "max_turns" if ep.turns >= self.config.max_turns else "context_limit"
            return ChatResult({"role": "assistant", "content": "", "reasoning_content": ""},
                              "length", len(input_ids), 0)
        want = min(want, room)

        # Hold here, not earlier: the prompt is recorded and validated, so a released request
        # resumes exactly where it would have been.
        while True:
            await self.gates.await_open(episode_id)

            async def issue():
                if not self.gates.open_for(episode_id):   # queued before a cut, running after it
                    raise GenerationAborted("gate closed before dispatch")
                return await self.engine.generate(input_ids, want, float(body.get("temperature", 1.0)))

            task = asyncio.create_task(issue())
            self.gates.inflight.add(task)
            task.add_done_callback(self.gates.inflight.discard)
            try:
                gen = await task
                break
            except GenerationAborted:
                if self.gates.open_for(episode_id):
                    raise ProxyError(503, "engine aborted outside a coordinated rollout cut") from None
                continue   # retry the SAME prompt after the next admitted resume
        if ep.closed:
            raise ProxyError(410, "episode released during generation")
        ep.extend_generated(gen.ids, gen.logprobs, gen.weight_version)
        ep.turns += 1

        reasoning, _, visible = gen.text.partition("</think>")
        visible = visible.strip() or gen.text
        tool_calls = []
        if tools:
            try:
                visible, items = self.tool_parser.parse(visible, tools)
                tool_calls = [{"id": f"call_{uuid.uuid4().hex[:20]}", "type": "function",
                               "function": {"name": it["name"], "arguments": it["arguments"]}}
                              for it in items]
            except Exception as exc:  # noqa: BLE001 - never fail the turn on a parser edge case
                self.parse_failures += 1
                logger.warning("tool-call parse failed for %s: %s", episode_id, exc)
        message: dict[str, Any] = {"role": "assistant", "content": visible,
                                   "reasoning_content": reasoning.replace("<think>", "").strip()}
        if tool_calls:
            message["tool_calls"] = tool_calls
        return ChatResult(message, "tool_calls" if tool_calls else "stop", len(input_ids), len(gen.ids))


# ---- OpenAI framing ----------------------------------------------------------------------------------
def completion_json(result: ChatResult, served_id: str) -> dict:
    return {"id": f"chatcmpl-{uuid.uuid4().hex[:24]}", "object": "chat.completion",
            "created": int(time.time()), "model": served_id,
            "choices": [{"index": 0, "message": result.message, "finish_reason": result.finish_reason}],
            "usage": {"prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens,
                      "total_tokens": result.prompt_tokens + result.completion_tokens}}


def completion_sse(result: ChatResult, served_id: str) -> list[str]:
    """The whole answer as a few correctly shaped deltas (Pi sets stream:true and cannot read a
    plain JSON body; generation itself is non-streaming because tool calls are parsed whole)."""
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    def chunk(delta: dict, finish: str | None = None) -> str:
        return "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created,
                                      "model": served_id,
                                      "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}) + "\n\n"

    out = [chunk({"role": "assistant"})]
    message = result.message
    if message.get("reasoning_content"):
        out.append(chunk({"reasoning_content": message["reasoning_content"]}))
    if message.get("content"):
        out.append(chunk({"content": message["content"]}))
    for i, tc in enumerate(message.get("tool_calls") or []):
        out.append(chunk({"tool_calls": [{"index": i, **tc}]}))   # streaming tool calls carry an index
    out.append(chunk({}, result.finish_reason))
    out.append("data: [DONE]\n\n")
    return out


# ---- the web app ----------------------------------------------------------------------------------------
def build_app(proxy: Proxy):
    """The FastAPI application (needs the `carry` extra: fastapi and uvicorn)."""
    if FastAPI is None:
        raise RuntimeError("fastapi is not installed; install autoenvscaling[carry]")
    app = FastAPI()

    @app.exception_handler(ProxyError)
    async def _proxy_error(request, exc: ProxyError):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status)

    @app.post("/park")
    async def park(episode_id: str | None = None) -> dict:
        return proxy.gates.park(episode_id)

    @app.post("/unpark")
    async def unpark(episode_id: str | None = None) -> dict:
        return proxy.gates.unpark(episode_id)

    @app.post("/unpark_all")
    async def unpark_all() -> dict:
        raise HTTPException(409, "resume requires explicit episode admission; use /unpark?episode_id=...")

    @app.post("/quiesce")
    async def quiesce() -> dict:
        return await proxy.gates.quiesce()

    @app.get("/park")
    async def park_state() -> dict:
        return proxy.gates.state()

    @app.get("/v1/models")
    async def models() -> dict:
        return {"object": "list", "data": [{"id": proxy.config.served_id, "object": "model"}]}

    @app.post("/config")
    async def set_backend(request: Request) -> dict:
        return proxy.set_backend(await request.json())

    @app.get("/config")
    async def get_backend() -> dict:
        return proxy.describe()

    @app.get("/episodes")
    async def episodes() -> dict:
        return {"episodes": sorted(proxy.episodes)}

    @app.get("/episodes/{episode_id}")
    async def episode_state(episode_id: str, full: int = 0) -> dict:
        return proxy.episode(episode_id, bool(full))

    @app.delete("/episodes/{episode_id}")
    async def drop_episode(episode_id: str) -> dict:
        return proxy.drop(episode_id)

    @app.post("/v1/ep/{episode_id}/chat/completions")
    async def chat_completions(episode_id: str, request: Request):
        proxy.check_auth(request.headers.get("authorization"))
        body = await request.json()
        served = proxy.config.served_id
        if not body.get("stream"):
            return completion_json(await proxy.chat(episode_id, body), served)

        # Return headers immediately and keep the response alive while the gates are closed:
        # waiting for the generation before answering produced ingress timeouts in training.
        # SSE comment lines are ignored by Pi and add no message, token or tool result.
        async def stream() -> AsyncIterator[str]:
            task = asyncio.create_task(proxy.chat(episode_id, body))
            proxy.stream_requests.add(task)

            def settled(done):
                proxy.stream_requests.discard(done)
                if not done.cancelled():
                    done.exception()

            task.add_done_callback(settled)
            wire_done = False
            try:
                yield ": keepalive\n\n"
                while not task.done():
                    done, _ = await asyncio.wait({task}, timeout=proxy.config.stream_heartbeat_s)
                    if not done:
                        yield ": keepalive\n\n"
                try:
                    result = task.result()
                except Exception as exc:  # noqa: BLE001 - headers are out; report on the wire
                    code = exc.status if isinstance(exc, ProxyError) else 502
                    yield "data: " + json.dumps({"error": {"message": f"proxy request failed (HTTP {code})",
                                                           "type": "proxy_error", "code": code}}) + "\n\n"
                    wire_done = True
                    return
                for piece in completion_sse(result, served):
                    yield piece
                wire_done = True
            except (asyncio.CancelledError, GeneratorExit):
                if not wire_done:
                    proxy.stream_disconnects += 1
                    ep = proxy.episodes.get(episode_id)
                    if ep is not None:
                        ep.error = ep.error or "response stream disconnected before completion"
                raise
            # chat() is not cancelled on disconnect: its engine call must stay counted by
            # /quiesce until it settles.

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


def make_proxy(config: ProxyConfig, *, tokenizer: Any = None) -> Proxy:
    """Build the proxy for the configured backend (loads the tokenizer from ``hf_dir`` if none given)."""
    problems = config.problems()
    if problems:
        raise ValueError("; ".join(problems))
    if tokenizer is None:
        if not config.hf_dir:
            raise ValueError("AES_PROXY_HF_DIR must point at the served model's tokenizer")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.hf_dir, trust_remote_code=True)
    renderer = ChatTemplateRenderer(tokenizer)
    engine: Engine = MockEngine(renderer) if config.backend == "mock" else SglangEngine(config)
    parser: ToolParser = SglangToolParser(config.tool_parser)
    proxy = Proxy(config, renderer=renderer, engine=engine, tool_parser=parser)
    proxy.restore_cache()
    return proxy


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    config = ProxyConfig.from_env()
    proxy = make_proxy(config)
    uvicorn.run(build_app(proxy), host=config.host, port=config.port, log_level="info")
    return 0
