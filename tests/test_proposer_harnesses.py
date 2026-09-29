"""Harness launch commands and native-log parsing (Claude stream-json, Codex JSONL, Pi NDJSON)."""

import json
import shlex

import pytest

from autoenvscaling.proposer.harnesses import (
    ClaudeCodeHarness,
    CodexHarness,
    PiHarness,
    get_harness,
    harness_for_model,
)
from autoenvscaling.proposer.harnesses.base import MAX_TOOL_OUTPUT_CHARS


def jl(*events) -> str:
    return "\n".join(json.dumps(e) for e in events) + "\n"


def args_of(turn, i=0):
    return json.loads(turn.tool_calls[i]["function"]["arguments"])


def test_harness_selection_follows_the_paper():
    assert harness_for_model("claude-opus-5") == "claude-code"
    assert harness_for_model("gpt-5.6-sol") == "codex"
    assert harness_for_model("openai/gpt-6-astra") == "codex"
    assert harness_for_model("Qwen/Qwen3.6-35B-A3B") == "pi"
    assert isinstance(get_harness("auto", "claude-opus-5"), ClaudeCodeHarness)
    with pytest.raises(ValueError):
        get_harness("unknown-harness")


# ---- Claude Code --------------------------------------------------------------------------

CLAUDE_LOG = jl(
    {"type": "system", "subtype": "init", "model": "claude-opus-5"},
    {"type": "assistant", "message": {"id": "m1", "content": [
        {"type": "thinking", "thinking": "Look at the rollouts first."}],
        "usage": {"input_tokens": 100, "output_tokens": 5}}},
    {"type": "assistant", "message": {"id": "m1", "content": [
        {"type": "text", "text": "Reading the index."},
        {"type": "tool_use", "id": "t1", "name": "Bash",
         "input": {"command": "head past_rollouts/index.jsonl"}}]}},
    {"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "t1",
         "content": [{"type": "text", "text": "{\"task_id\": \"p\"}"}]}]}},
    {"type": "assistant", "message": {"id": "m2", "content": [
        {"type": "tool_use", "id": "t2", "name": "Write",
         "input": {"file_path": "output/tasks/x/instruction.md", "content": "Do it."}}]}},
    {"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "t2", "content": "denied", "is_error": True}]}},
    {"type": "assistant", "message": {"id": "m3", "content": [{"type": "text", "text": "Done."}]}},
    {"type": "result", "subtype": "success", "is_error": False, "num_turns": 3,
     "total_cost_usd": 1.25, "usage": {"input_tokens": 10, "cache_read_input_tokens": 900,
                                       "cache_creation_input_tokens": 50, "output_tokens": 400}},
)


def test_claude_code_parse_groups_blocks_by_message_and_pairs_results():
    parsed = ClaudeCodeHarness().parse_log(CLAUDE_LOG)
    roles = [t.role for t in parsed.turns]
    assert roles == ["assistant", "tool", "assistant", "tool", "assistant"]
    first = parsed.turns[0]
    assert first.content.startswith("<think>\nLook at the rollouts first.\n</think>")
    assert first.content.endswith("Reading the index.")
    assert first.tool_calls[0]["function"]["name"] == "Bash"
    assert args_of(first) == {"command": "head past_rollouts/index.jsonl"}
    assert parsed.turns[1].tool_call_id == "t1" and parsed.turns[1].name == "Bash"
    assert parsed.turns[3].content.startswith("[error]")
    assert parsed.completed and parsed.error is None
    assert parsed.usage["output_tokens"] == 400 and parsed.usage["cost_usd"] == 1.25


def test_claude_code_usage_falls_back_to_messages_when_cut_off():
    cut = "\n".join(CLAUDE_LOG.splitlines()[:4]) + '\n{"type": "assist'  # truncated last line
    parsed = ClaudeCodeHarness().parse_log(cut)
    assert not parsed.completed
    assert parsed.usage == {"input_tokens": 100, "output_tokens": 5}


def test_claude_code_prepare_requires_a_key_and_builds_the_command():
    h = ClaudeCodeHarness()
    with pytest.raises(RuntimeError):
        h.prepare(model="claude-opus-5", prompt_file="/r/p.md", run_dir="/r", workdir="/w", env={})
    launch = h.prepare(model="claude-opus-5", prompt_file="/r/p.md", run_dir="/r", workdir="/w",
                       env={"ANTHROPIC_API_KEY": "k"})
    words = shlex.split(launch.command.replace('"$(cat /r/p.md)"', "PROMPT"))
    assert words[:2] == ["claude", "-p"] and words[-1] == "PROMPT"
    assert "Read,Glob,Grep,Write,Edit,Bash" in words and "WebSearch" not in launch.command
    assert launch.env["CLAUDE_CONFIG_DIR"] == "/r/claude"


# ---- Codex --------------------------------------------------------------------------------

CODEX_LOG = jl(
    {"type": "thread.started", "thread_id": "th"},
    {"type": "turn.started"},
    {"type": "item.started", "item": {"id": "i0", "type": "command_execution", "command": "ls"}},
    {"type": "item.completed", "item": {"id": "i1", "type": "reasoning", "text": "Plan it."}},
    {"type": "item.completed", "item": {"id": "i2", "type": "command_execution", "command": "ls",
                                        "aggregated_output": "memory\n", "exit_code": 0,
                                        "status": "completed"}},
    {"type": "item.completed", "item": {"id": "i3", "type": "command_execution",
                                        "command": "false", "aggregated_output": "",
                                        "exit_code": 1, "status": "failed"}},
    {"type": "item.completed", "item": {"id": "i4", "type": "file_change", "status": "completed",
                                        "changes": [{"path": "a.txt", "kind": "add"}]}},
    {"type": "item.completed", "item": {"id": "i5", "type": "agent_message", "text": "Finished."}},
    {"type": "turn.completed", "usage": {"input_tokens": 1000, "cached_input_tokens": 800,
                                         "output_tokens": 50}},
)


def test_codex_parse():
    parsed = CodexHarness().parse_log(CODEX_LOG)
    roles = [t.role for t in parsed.turns]
    assert roles == ["assistant", "tool", "assistant", "tool", "assistant", "tool", "assistant"]
    assert parsed.turns[0].content == "<think>\nPlan it.\n</think>"
    assert args_of(parsed.turns[0]) == {"command": "ls"}
    assert parsed.turns[1].content == "memory\n" and parsed.turns[1].tool_call_id == "i2"
    assert "[exit code 1]" in parsed.turns[3].content
    assert parsed.turns[4].tool_calls[0]["function"]["name"] == "apply_patch"
    assert parsed.turns[-1].content == "Finished."
    assert parsed.completed and parsed.usage == {"input_tokens": 1000, "cached_input_tokens": 800,
                                                 "output_tokens": 50}


def test_codex_failure_is_reported():
    parsed = CodexHarness().parse_log(jl({"type": "turn.failed", "error": {"message": "quota"}}))
    assert parsed.error == "quota" and not parsed.completed


def test_codex_prepare_writes_an_episode_local_config():
    h = CodexHarness()
    with pytest.raises(RuntimeError):
        h.prepare(model="gpt-5.6-sol", prompt_file="/r/p", run_dir="/r", workdir="/w", env={})
    launch = h.prepare(model="gpt-5.6-sol", prompt_file="/r/p", run_dir="/r", workdir="/w",
                       env={"OPENAI_API_KEY": "k", "OPENAI_BASE_URL": "https://gw.example/v1"})
    config = launch.files["/r/codex/config.toml"].decode()
    assert 'model = "gpt-5.6-sol"' in config and 'base_url = "https://gw.example/v1"' in config
    assert 'env_key = "OPENAI_API_KEY"' in config and 'wire_api = "responses"' in config
    assert launch.env == {"CODEX_HOME": "/r/codex"}
    assert launch.command.startswith("codex exec --json")


# ---- Pi -----------------------------------------------------------------------------------

PI_LOG = jl(
    {"type": "agent_start"},
    {"type": "message_end", "message": {"role": "user", "content": [{"type": "text",
                                                                      "text": "prompt"}]}},
    {"type": "message_update", "message": {"role": "assistant"}},
    {"type": "message_end", "message": {
        "role": "assistant", "stopReason": "toolUse",
        "content": [{"type": "thinking", "thinking": "Check memory."},
                    {"type": "toolCall", "id": "c1", "name": "bash",
                     "arguments": {"command": "cat memory/notes.md"}}],
        "usage": {"input": 500, "output": 20, "cacheRead": 100, "cacheWrite": 0,
                  "cost": {"total": 0.01}}}},
    {"type": "tool_execution_end", "toolCallId": "c1", "toolName": "bash",
     "result": {"content": [{"type": "text", "text": "x" * (MAX_TOOL_OUTPUT_CHARS + 10)}]},
     "isError": False},
    {"type": "message_end", "message": {"role": "toolResult", "toolCallId": "c1",
                                        "toolName": "bash", "content": [{"type": "text",
                                                                         "text": "dup"}]}},
    {"type": "message_end", "message": {
        "role": "assistant", "stopReason": "stop", "content": [{"type": "text", "text": "Done."}],
        "usage": {"input": 900, "output": 30, "cost": {"total": 0.02}}}},
)


def test_pi_parse_dedupes_results_and_accumulates_usage():
    h = PiHarness()
    parsed = h.parse_log(PI_LOG)
    assert [t.role for t in parsed.turns] == ["assistant", "tool", "assistant"]
    assert parsed.turns[0].content == "<think>\nCheck memory.\n</think>"
    assert args_of(parsed.turns[0]) == {"command": "cat memory/notes.md"}
    assert parsed.turns[1].name == "bash" and "more characters clipped" in parsed.turns[1].content
    assert parsed.completed
    assert parsed.usage == {"input_tokens": 900, "output_tokens": 50, "cost_usd": 0.03}
    assert "message_update" not in h.compact_log(PI_LOG)


def test_pi_prepare_uses_the_configured_endpoint():
    h = PiHarness()
    with pytest.raises(RuntimeError, match="AES_PROPOSER_BASE_URL"):
        h.prepare(model="m", prompt_file="/r/p", run_dir="/r", workdir="/w", env={})
    launch = h.prepare(model="qwen3.6-35b-a3b", prompt_file="/r/p", run_dir="/r", workdir="/w",
                       env={"AES_PROPOSER_BASE_URL": "https://llm.example/v1",
                            "AES_PROPOSER_REASONING": "1"})
    models = json.loads(launch.files["/r/pi/models.json"])
    provider = models["providers"]["aes"]
    assert provider["baseUrl"] == "https://llm.example/v1"
    assert provider["apiKey"] == "$AES_PROPOSER_API_KEY"
    assert provider["models"][0]["id"] == "qwen3.6-35b-a3b" and provider["models"][0]["reasoning"]
    assert json.loads(launch.files["/r/pi/settings.json"]) == {"compaction": {"enabled": True}}
    assert launch.env["PI_CODING_AGENT_DIR"] == "/r/pi"
    assert launch.env["AES_PROPOSER_API_KEY"] == "EMPTY"
    assert "--provider aes --model qwen3.6-35b-a3b" in launch.command
