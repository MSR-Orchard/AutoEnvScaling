"""The recording proxy: token ids and log-probabilities are the engine's own and extend across
turns and parks; a park holds the request and never changes the conversation; provisional
generations roll back; the web surface is OpenAI-shaped."""

from __future__ import annotations

import asyncio
import json

import pytest

from autoenvscaling.training.carry import proxy as px

GEN = 9000   # the generation-prompt token of the fake template
ROLE = {"system": 9001, "user": 9002, "assistant": 9003, "tool": 9004}


class FakeTokenizer:
    """A chat template that is exact for round trips: an assistant message renders as the
    generation prompt followed by its content's tokens, so a re-render with the model's own
    reply extends the recorded stream."""

    def encode(self, text, add_special_tokens=False):
        return [1000 + (ord(ch) % 500) for ch in text]

    def decode(self, ids):
        return "".join(chr(i - 1000 + 32) if 1000 <= i < 1500 else "?" for i in ids)

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=True, tokenize=True,
                            preserve_thinking=False):
        assert preserve_thinking, "invariant 1: thinking must be preserved on every render"
        ids = []
        if tools:
            ids += [9100]
        for m in messages:
            if m["role"] == "assistant":
                text = (m.get("reasoning_content") or "") + (m.get("content") or "")
                for tc in m.get("tool_calls") or []:
                    assert isinstance(tc["function"]["arguments"], dict), "invariant 2"
                    text += "CALL " + tc["function"]["name"]
                ids += [GEN] + self.encode(text)
            else:
                ids += [ROLE[m["role"]]] + self.encode(m.get("content") or "")
        if add_generation_prompt:
            ids.append(GEN)
        return ids


class ScriptedEngine:
    def __init__(self, renderer, replies, abort_once=False):
        self.renderer = renderer
        self.replies = list(replies)
        self.calls = []
        self.abort_once = abort_once
        self.version = 0

    async def generate(self, input_ids, max_new, temperature):
        self.calls.append((list(input_ids), max_new))
        if self.abort_once:
            self.abort_once = False
            raise px.GenerationAborted("cut")
        text = self.replies.pop(0)
        ids = self.renderer.encode(text)
        return px.Generation(ids, [-0.25] * len(ids), text, weight_version=self.version)


class FakeToolParser:
    def parse(self, text, tools):
        if "CALL bash" in text:
            return text, [{"name": "bash", "arguments": "{\"command\": \"ls\"}"}]
        return text, []


def make_proxy(replies, **cfg):
    config = px.ProxyConfig(backend="mock", max_context=cfg.pop("max_context", 400),
                            max_turns=cfg.pop("max_turns", 10), min_room=cfg.pop("min_room", 8), **cfg)
    renderer = px.ChatTemplateRenderer(FakeTokenizer())
    engine = ScriptedEngine(renderer, replies)
    return px.Proxy(config, renderer=renderer, engine=engine, tool_parser=FakeToolParser()), engine, renderer


def assistant_from(result):
    return {"role": "assistant", "content": result.message["content"],
            "reasoning_content": result.message.get("reasoning_content", ""),
            "tool_calls": [{"id": tc["id"], "type": "function",
                            "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}}
                           for tc in result.message.get("tool_calls", [])]}


# ---- recording across turns ------------------------------------------------------------------------------
async def test_recording_extends_across_turns_with_engine_ids_and_logprobs():
    proxy, engine, renderer = make_proxy(["first reply", "CALL bash", "done"])
    messages = [{"role": "user", "content": "do the task"}]
    tools = [{"type": "function", "function": {"name": "bash"}}]
    r1 = await proxy.chat("e1", {"messages": messages, "tools": tools})
    ep = proxy.episodes["e1"]
    prompt = renderer.render(messages, tools)
    assert ep.prompt_len == len(prompt) and ep.tokens[:len(prompt)] == prompt
    gen1 = renderer.encode("first reply")
    assert ep.tokens[len(prompt):] == gen1 and ep.logprobs[len(prompt):] == [-0.25] * len(gen1)
    assert ep.loss_mask == [0] * len(prompt) + [1] * len(gen1) and ep.turns == 1
    assert ep.weight_versions == [0] and r1.finish_reason == "stop" and r1.completion_tokens == len(gen1)

    messages += [assistant_from(r1), {"role": "user", "content": "go on"}]
    r2 = await proxy.chat("e1", {"messages": messages, "tools": tools})
    assert r2.finish_reason == "tool_calls" and r2.message["tool_calls"][0]["function"]["name"] == "bash"
    assert engine.calls[1][0] == renderer.render(messages, tools)       # the engine saw the extended prompt
    assert engine.calls[1][0][:len(ep.tokens) - len(renderer.encode("CALL bash"))] == \
        ep.tokens[:len(ep.tokens) - len(renderer.encode("CALL bash"))]
    messages += [assistant_from(r2), {"role": "tool", "content": "file.txt", "tool_call_id": "x"}]
    r3 = await proxy.chat("e1", {"messages": messages, "tools": tools})
    assert r3.finish_reason == "stop" and ep.turns == 3 and ep.tool_results_seen == 1 and ep.rollbacks == 0
    full = proxy.episode("e1", full=True)
    assert len(full["tokens"]) == len(full["logprobs"]) == len(full["loss_mask"]) == full["n_tokens"]
    assert full["n_trained_tokens"] == sum(len(renderer.encode(t)) for t in ("first reply", "CALL bash", "done"))
    assert sum(full["loss_mask"][:full["prompt_len"]]) == 0


async def test_park_holds_the_request_and_the_conversation_is_unchanged():
    proxy, engine, renderer = make_proxy(["slow reply"])
    messages = [{"role": "user", "content": "task"}]
    proxy.gates.park()                                       # the weight swap begins
    task = asyncio.create_task(proxy.chat("e1", {"messages": messages}))
    await asyncio.sleep(0.02)
    assert not task.done() and engine.calls == []            # held before dispatch
    assert proxy.gates.state()["global_parked"]
    assert (await proxy.gates.quiesce(timeout_s=0.01))["quiescent"]
    proxy.gates.unpark()
    r = await asyncio.wait_for(task, 2)
    assert r.message["content"] == "slow reply"
    ep = proxy.episodes["e1"]
    assert ep.tokens == renderer.render(messages, None) + renderer.encode("slow reply")

    # Per-episode gate: only that episode waits.
    proxy.engine.replies += ["a", "b"]
    proxy.gates.park("e2")
    held = asyncio.create_task(proxy.chat("e2", {"messages": messages}))
    free = await asyncio.wait_for(proxy.chat("e3", {"messages": messages}), 2)
    assert free.message["content"] == "a" and not held.done()
    proxy.gates.unpark("e2")
    assert (await asyncio.wait_for(held, 2)).message["content"] == "b"


async def test_engine_abort_during_a_cut_retries_the_same_prompt_after_unpark():
    proxy, engine, renderer = make_proxy(["reply"])
    engine.abort_once = True
    messages = [{"role": "user", "content": "task"}]

    async def cut_then_resume():
        await asyncio.sleep(0.01)
        proxy.gates.park()
        await asyncio.sleep(0.02)
        proxy.gates.unpark()

    async def chat():
        # the gate closes while the engine call is in flight; the engine aborts
        task = asyncio.create_task(proxy.chat("e1", {"messages": messages}))
        return await asyncio.wait_for(task, 2)

    # Make the first engine call observe the closed gate: the abort arrives, the gate is closed,
    # the proxy waits for unpark and reissues the identical prompt.
    orig = engine.generate

    async def aborting(input_ids, max_new, temperature):
        if engine.abort_once:
            proxy.gates.park()
            engine.abort_once = False
            asyncio.get_running_loop().call_later(0.02, proxy.gates.unpark)
            raise px.GenerationAborted("cut")
        return await orig(input_ids, max_new, temperature)

    engine.generate = aborting
    r = await chat()
    assert r.message["content"] == "reply" and len(engine.calls) == 1   # the scripted call recorded once
    ep = proxy.episodes["e1"]
    assert ep.rollbacks == 0 and ep.turns == 1 and sum(ep.loss_mask) == len(renderer.encode("reply"))
    # an abort while the gates are open is an error, not a retry
    engine.generate = orig
    engine.abort_once = True
    with pytest.raises(px.ProxyError) as info:
        await proxy.chat("e9", {"messages": messages})
    assert info.value.status == 503


async def test_provisional_generation_rolls_back_when_the_agent_never_saw_it():
    proxy, _engine, renderer = make_proxy(["lost reply", "second reply", "third"])
    messages = [{"role": "user", "content": "task"}]
    await proxy.chat("e1", {"messages": messages})
    ep = proxy.episodes["e1"]
    # Pi re-sends the same prompt (it never received "lost reply"): shorter, matching prefix.
    r = await proxy.chat("e1", {"messages": messages})
    assert ep.rollbacks == 1 and r.message["content"] == "second reply"
    assert ep.tokens == renderer.render(messages, None) + renderer.encode("second reply")
    assert ep.turns == 1 and ep.weight_versions == [0]
    # A divergence after the provisional mark rolls back too; one before it is corruption.
    other = messages + [{"role": "assistant", "content": "something else"}, {"role": "user", "content": "x"}]
    r = await proxy.chat("e1", {"messages": other})
    assert ep.rollbacks == 2 and r.message["content"] == "third"
    # A divergence BEFORE the provisional mark is corruption, not a park artefact: after two
    # completed turns, a changed first message cannot be explained by a lost generation.
    proxy2, _engine2, _renderer2 = make_proxy(["one", "two"])
    msgs = [{"role": "user", "content": "task"}]
    r1 = await proxy2.chat("e2", {"messages": msgs})
    await proxy2.chat("e2", {"messages": msgs + [assistant_from(r1), {"role": "user", "content": "x"}]})
    with pytest.raises(px.ProxyError) as info:
        await proxy2.chat("e2", {"messages": [{"role": "user", "content": "a different task"}]})
    assert info.value.status == 409 and proxy2.episodes["e2"].error.startswith("render diverged")
    # A render shorter than the committed recording (more than one generation lost) is refused:
    # generating from it would record tokens in a context the engine never saw.
    with pytest.raises(px.ProxyError) as info:
        await proxy2.chat("e2", {"messages": msgs})
    assert info.value.status == 409 and "shorter than the committed" in info.value.detail


async def test_context_and_turn_limits_stop_the_agent_without_generating():
    proxy, engine, _renderer = make_proxy(["r"], max_context=40, max_turns=10)
    r = await proxy.chat("e1", {"messages": [{"role": "user", "content": "x" * 60}]})
    assert r.finish_reason == "length" and r.completion_tokens == 0 and engine.calls == []
    assert proxy.episodes["e1"].stop_reason == "context_limit"
    proxy2, _engine2, _ = make_proxy(["r"] * 3, max_turns=1)
    msgs = [{"role": "user", "content": "go"}]
    await proxy2.chat("e2", {"messages": msgs})
    r = await proxy2.chat("e2", {"messages": msgs + [{"role": "assistant", "content": "r"}]})
    assert r.finish_reason == "length" and proxy2.episodes["e2"].stop_reason == "max_turns"


def test_normalise_messages_and_config():
    out = px.normalise_messages([{"role": "assistant", "content": "<think>why</think>  ok",
                                  "tool_calls": [{"function": {"name": "bash", "arguments": "{\"a\": 1}"}}]}])
    assert out[0]["reasoning_content"] == "why" and out[0]["content"] == "ok"
    assert out[0]["tool_calls"][0]["function"]["arguments"] == {"a": 1}
    with pytest.raises(px.ProxyError):
        px.normalise_messages([{"role": "assistant", "content": "", "tool_calls": [{"function": {"arguments": "{"}}]}])
    cfg = px.ProxyConfig.from_env({"AES_PROXY_MAX_CONTEXT": "1234", "AES_PROXY_BACKEND": "mock",
                                   "AES_PROXY_STREAM_HEARTBEAT_S": "5"})
    assert cfg.max_context == 1234 and cfg.backend == "mock" and cfg.stream_heartbeat_s == 5.0 and not cfg.problems()
    assert px.ProxyConfig(backend="vllm").problems()
    assert px._as_id_list({"input_ids": [[1, 2]]}) == [1, 2]


# ---- the web surface -------------------------------------------------------------------------------------
def test_app_endpoints_json_streaming_and_release(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    proxy, _engine, renderer = make_proxy(["hello", "CALL bash", "later"], api_key="secret",
                                         config_cache=str(tmp_path / "cfg.json"))
    client = TestClient(px.build_app(proxy))
    headers = {"Authorization": "Bearer secret"}
    # The whole surface needs the key, not just completions: the control endpoints can change
    # the backend, open the gates and read or delete recordings. Only /healthz is open.
    assert client.get("/healthz").json()["ok"] is True
    for path in ("/config", "/park", "/episodes", "/episodes/e1", "/v1/models"):
        assert client.get(path).status_code == 401, path
    for path in ("/park", "/unpark", "/quiesce", "/config"):
        assert client.post(path, json={}).status_code == 401, path
    assert client.delete("/episodes/e1").status_code == 401
    assert client.get("/config", headers=headers).json()["carry_lifecycle_version"] == 1
    assert client.post("/config", json={"sglang_url": "http://router:30001/", "served_id": "policy",
                                        "max_turns": 7}, headers=headers).json()["sglang_url"] == "http://router:30001"
    assert json.loads((tmp_path / "cfg.json").read_text())["max_turns"] == 7
    assert client.post("/config", json={"max_context": 0}, headers=headers).status_code == 400
    assert client.get("/v1/models", headers=headers).json()["data"][0]["id"] == "policy"
    client.headers.update(headers)

    msgs = [{"role": "user", "content": "task"}]
    tools = [{"type": "function", "function": {"name": "bash"}}]      # Pi sends its tools every request
    assert client.post("/v1/ep/e1/chat/completions", json={"messages": msgs, "tools": tools},
                       headers={"Authorization": "Bearer wrong"}).status_code == 401
    r = client.post("/v1/ep/e1/chat/completions", json={"messages": msgs, "tools": tools}, headers=headers)
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "hello"
    assert r.json()["usage"]["completion_tokens"] == len(renderer.encode("hello"))

    msgs += [{"role": "assistant", "content": "hello"}, {"role": "user", "content": "more"}]
    with client.stream("POST", "/v1/ep/e1/chat/completions", json={"messages": msgs, "tools": tools, "stream": True},
                       headers=headers) as resp:
        assert resp.status_code == 200 and resp.headers["content-type"].startswith("text/event-stream")
        lines = [ln for ln in resp.iter_lines() if ln]
    assert lines[0] == ": keepalive"
    chunks = [json.loads(ln[6:]) for ln in lines if ln.startswith("data: ") and ln != "data: [DONE]"]
    deltas = [c["choices"][0]["delta"] for c in chunks]
    assert any("tool_calls" in d for d in deltas) and chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert deltas[-2]["tool_calls"][0]["index"] == 0 and lines[-1] == "data: [DONE]"

    state = client.get("/episodes/e1?full=1").json()
    assert state["turns"] == 2 and len(state["tokens"]) == state["n_tokens"] and state["prompt_len"] > 0
    assert client.get("/episodes").json()["episodes"] == ["e1"]
    assert client.get("/episodes/none").status_code == 404

    assert client.post("/quiesce").status_code == 409                    # global gate still open
    assert client.post("/park").json() == {"parked": True, "scope": "global"}
    assert client.post("/quiesce").json() == {"quiescent": True}
    assert client.get("/park").json()["global_parked"] is True
    assert client.post("/unpark").json()["scope"] == "global" and client.post("/unpark_all").status_code == 409
    assert client.post("/park?episode_id=e1").json()["scope"] == "e1"
    assert client.get("/park").json()["episodes_parked"] == ["e1"]
    assert client.post("/unpark?episode_id=e1").json()["parked"] is False

    assert client.delete("/episodes/e1").json() == {"dropped": True}
    assert client.delete("/episodes/e1").json() == {"dropped": False}
    r = client.post("/v1/ep/e1/chat/completions", json={"messages": msgs}, headers=headers)
    assert r.status_code == 410
