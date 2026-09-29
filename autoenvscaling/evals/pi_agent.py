"""The Pi coding agent as a Harbor agent, running inside the task sandbox.

All Terminal-Bench numbers in the paper come from Pi (https://pi.dev) running in the task
container against an OpenAI-compatible endpoint. This agent pins that setup:

* A pinned Pi and Node.js version, installed in the sandbox if the image lacks them.
* Only Pi's bash tool by default, no skills, and context compaction off, so the transcript is
  append-only.
* Pi sends no decoding parameters of its own and its model catalogue cannot carry them, so a
  request extension (Pi's ``before_provider_request`` hook) adds the sampling vector, adds
  router pinning if configured, and refuses a request for any model other than the configured
  one. Every applied policy is logged to ``request_policy.jsonl`` next to the transcript.
* Pi runs detached under GNU ``timeout`` and the agent polls for its exit marker, so a long
  task never depends on one long-lived exec call. A run that hits the time limit or the context
  window returns normally and the verifier scores the state left behind; provider and network
  failures raise Harbor's API errors so the trial is retried as infrastructure.

Use it through Harbor with ``AgentConfig(import_path=PI_AGENT_IMPORT, model_name=<exact id>,
kwargs={...}, env={<api key env>: "${<api key env>}"})``; ``terminal_bench.agent_config``
builds that for you.
"""

from __future__ import annotations

import asyncio
import json
import math
import shlex
import uuid
from pathlib import PurePosixPath
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.agents.installed.base import (
    ApiInternalServerError,
    ApiOverloadedError,
    ApiRateLimitError,
    NetworkConnectionError,
    NonZeroAgentExitCodeError,
    UnknownApiError,
)
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from autoenvscaling.evals.endpoint import normalize_base_url

PI_AGENT_IMPORT = "autoenvscaling.evals.pi_agent:PiAgent"

NODE_VERSION = "v22.19.0"
PI_VERSION = "0.80.6"
PI_PACKAGE = "@earendil-works/pi-coding-agent"
PROVIDER_NAME = "aes-endpoint"

_PI_DIR = PurePosixPath("/opt/pi-agent")
_SHORT_TIMEOUT_SEC = 30
_POLL_INTERVAL_SEC = 20.0
_TRANSPORT_MARGIN_SEC = 45
# Exit codes of a run ended by the time limit: GNU timeout (124), SIGKILL (137), SIGTERM (143).
TIMEOUT_EXIT_CODES = frozenset({124, 137, 143})

CONTEXT_MARKERS = (
    "context length exceeded",
    "maximum context length",
    "context window exceeded",
    "context_length_exceeded",
    "too many tokens",
    "prompt is too long",
)

# Accepted ranges for the decoding vector. An out-of-range value would make the server reject
# every request of the run, so it fails at configuration time instead.
SAMPLING_RANGES = {
    "temperature": (lambda v: v >= 0.0, "must be >= 0"),
    "top_p": (lambda v: 0.0 < v <= 1.0, "must be in (0, 1]"),
    "top_k": (lambda v: v == -1 or v >= 1, "must be -1 (disabled) or >= 1"),
    "min_p": (lambda v: 0.0 <= v <= 1.0, "must be in [0, 1]"),
    "presence_penalty": (lambda v: -2.0 <= v <= 2.0, "must be in [-2, 2]"),
    "repetition_penalty": (lambda v: 0.0 < v <= 2.0, "must be in (0, 2]"),
}

# Qwen model-card vector for agentic coding. The paper does not state a decoding vector for its
# evaluations, so this is offered as a named preset, not a default.
QWEN_SOLVING_SAMPLING = {
    "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
    "presence_penalty": 0.0, "repetition_penalty": 1.0,
}


def validate_sampling(sampling: dict[str, Any] | None) -> dict[str, float] | None:
    if not sampling:
        return None
    clean: dict[str, float] = {}
    for key, value in sampling.items():
        if key not in SAMPLING_RANGES:
            raise ValueError(f"unknown sampling key {key!r}; expected one of {sorted(SAMPLING_RANGES)}")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"sampling {key}={value!r} is not a finite number")
        check, bound = SAMPLING_RANGES[key]
        if not check(value):
            raise ValueError(f"sampling {key}={value!r} {bound}")
        clean[key] = int(value) if key == "top_k" else float(value)
    return clean


def positive_timeout(value: float | None) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("agent_timeout_sec must be a positive finite number")
    timeout = float(value)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("agent_timeout_sec must be a positive finite number")
    return int(timeout) if timeout.is_integer() else timeout


def install_script(node_version: str = NODE_VERSION, pi_version: str = PI_VERSION) -> str:
    """Shell script that installs the pinned Node.js and Pi unless the image already has them."""
    package = f"{PI_PACKAGE}@{pi_version}"
    return f"""set -e
EXPECTED_NODE={shlex.quote(node_version)}
EXPECTED_PI={shlex.quote(pi_version)}
PI_PACKAGE={shlex.quote(package)}
case "$(uname -m)" in
  x86_64|amd64) NODE_ARCH=x64 ;;
  aarch64|arm64) NODE_ARCH=arm64 ;;
  *) echo "unsupported architecture $(uname -m)" >&2; exit 1 ;;
esac
NODE_DIR=/opt/node
NODE_BIN="$NODE_DIR/bin/node"
NPM_BIN="$NODE_DIR/bin/npm"
NODE_ACTUAL="$([ -x "$NODE_BIN" ] && "$NODE_BIN" --version 2>/dev/null || true)"
if [ "$NODE_ACTUAL" != "$EXPECTED_NODE" ] || [ ! -x "$NPM_BIN" ]; then
  SYSTEM_NODE="$(command -v node 2>/dev/null || true)"
  SYSTEM_NPM="$(command -v npm 2>/dev/null || true)"
  SYSTEM_ACTUAL="$([ -n "$SYSTEM_NODE" ] && "$SYSTEM_NODE" --version 2>/dev/null || true)"
  if [ "$SYSTEM_ACTUAL" = "$EXPECTED_NODE" ] && [ -x "$SYSTEM_NPM" ]; then
    NODE_BIN="$SYSTEM_NODE"
    NPM_BIN="$SYSTEM_NPM"
  else
    if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
      (apt-get update && apt-get install -y --no-install-recommends curl ca-certificates) \\
        >/dev/null 2>&1 || apk add --no-cache curl ca-certificates >/dev/null 2>&1 || true
    fi
    NAME="node-$EXPECTED_NODE-linux-$NODE_ARCH"
    URL="https://nodejs.org/dist/$EXPECTED_NODE/$NAME.tar.gz"
    mkdir -p /opt && cd /opt
    if command -v curl >/dev/null 2>&1; then curl -fsSL -o node.tgz "$URL";
    elif command -v wget >/dev/null 2>&1; then wget -qO node.tgz "$URL";
    else echo "no downloader available to install Node.js" >&2; exit 1; fi
    rm -rf "$NODE_DIR" "$NAME"
    tar -xzf node.tgz && mv "$NAME" "$NODE_DIR" && rm -f node.tgz
    NODE_BIN="$NODE_DIR/bin/node"
    NPM_BIN="$NODE_DIR/bin/npm"
  fi
fi
NODE_BIN_DIR="$(dirname "$NODE_BIN")"
export PATH="$NODE_BIN_DIR:$PATH"
PI_BIN="$(command -v pi 2>/dev/null || true)"
PI_ACTUAL="$([ -n "$PI_BIN" ] && "$PI_BIN" --version 2>/dev/null || true)"
if [ "$PI_ACTUAL" != "$EXPECTED_PI" ]; then
  "$NPM_BIN" install -g --ignore-scripts "$PI_PACKAGE"
  PI_BIN="$NODE_BIN_DIR/pi"
fi
PI_ACTUAL="$([ -x "$PI_BIN" ] && "$PI_BIN" --version 2>/dev/null || true)"
if [ "$PI_ACTUAL" != "$EXPECTED_PI" ]; then
  echo "expected Pi $EXPECTED_PI, found '$PI_ACTUAL'" >&2
  exit 1
fi
mkdir -p /usr/local/bin
ln -sf "$NODE_BIN" /usr/local/bin/node
ln -sf "$NPM_BIN" /usr/local/bin/npm
ln -sf "$PI_BIN" /usr/local/bin/pi
/usr/local/bin/node --version
/usr/local/bin/pi --version"""


def request_extension(model: str, sampling: dict[str, float] | None,
                      provider_routing: dict[str, Any] | None, policy_log: str) -> str:
    """Pi extension that pins the model, adds the decoding vector, and logs what it applied."""
    policy = {"model": model, "sampling": sampling or {}, "provider": provider_routing}
    encoded = json.dumps(policy, separators=(",", ":"))
    log_path = json.dumps(policy_log)
    return f"""const POLICY = {encoded};
import {{ appendFileSync }} from "node:fs";

export default function (pi) {{
  pi.on("before_provider_request", (event) => {{
    const payload = event && event.payload;
    // Chat Completions bodies carry "messages"; other request shapes are left untouched.
    if (!payload || typeof payload !== "object" || !Array.isArray(payload.messages)) return;
    if (payload.model !== POLICY.model) {{
      throw new Error(`model identity drift: requested ${{payload.model}}, pinned ${{POLICY.model}}`);
    }}
    const applied = Object.assign({{}}, payload, POLICY.sampling);
    if (POLICY.provider) applied.provider = POLICY.provider;
    try {{
      appendFileSync({log_path}, JSON.stringify({{
        model: applied.model, provider: POLICY.provider, sampling: POLICY.sampling,
      }}) + "\\n");
    }} catch (err) {{}}
    return applied;
  }});
}}
"""


def models_json(model: str, base_url: str, api: str, api_key_env: str,
                context_window: int, max_output_tokens: int,
                reasoning: bool = False) -> dict[str, Any]:
    return {
        "providers": {
            PROVIDER_NAME: {
                "name": f"OpenAI-compatible endpoint serving {model}",
                "baseUrl": base_url,
                "api": api,
                "apiKey": f"${api_key_env}",
                "authHeader": True,
                "models": [{
                    "id": model,
                    "name": model,
                    "input": ["text"],
                    "reasoning": reasoning,
                    "contextWindow": context_window,
                    "maxTokens": max_output_tokens,
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                }],
            }
        }
    }


def parse_marker(text: str | None, key: str) -> int | None:
    """Last non-negative integer ``KEY=value`` line in ``text``."""
    for line in reversed((text or "").splitlines()):
        left, sep, right = line.strip().partition("=")
        if left != key or not sep:
            continue
        try:
            value = int(right)
        except ValueError:
            continue
        if value >= 0:
            return value
    return None


def is_context_limited(text: str | None) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in CONTEXT_MARKERS)


def api_error(message: str) -> NonZeroAgentExitCodeError:
    """Map a Pi failure message to the Harbor error class that decides retry behaviour."""
    lowered = message.lower()
    if "429" in lowered or "rate limit" in lowered or "too many requests" in lowered:
        return ApiRateLimitError(message)
    if any(m in lowered for m in ("overloaded", "temporarily unavailable", "503")):
        return ApiOverloadedError(message)
    if any(m in lowered for m in ("internal server error", " 500", " 502", " 504")):
        return ApiInternalServerError(message)
    if any(m in lowered for m in ("econnreset", "econnrefused", "enotfound", "network",
                                  "socket hang up", "fetch failed", "tls")):
        return NetworkConnectionError(message)
    if any(m in lowered for m in ("api error", "provider error")):
        return UnknownApiError(message)
    return NonZeroAgentExitCodeError(message)


def usage_script(transcript: PurePosixPath) -> str:
    """Node one-liner that prints token usage from a Pi transcript as integer markers."""
    path = json.dumps(str(transcript))
    return f"""const fs=require('fs');
let input=0, output=0, cache=0, turns=0;
let text=''; try {{ text=fs.readFileSync({path}, 'utf8'); }} catch (e) {{}}
for (const line of text.split(/\\n/)) {{
  if (!line) continue;
  let event; try {{ event=JSON.parse(line); }} catch {{ continue; }}
  if (event.type !== 'message_end' || event.message?.role !== 'assistant') continue;
  const usage=event.message.usage || {{}}; turns++;
  input += Number(usage.input || 0); output += Number(usage.output || 0);
  cache += Number(usage.cacheRead || 0);
}}
console.log(`PI_INPUT=${{Math.round(input)}}`);
console.log(`PI_OUTPUT=${{Math.round(output)}}`);
console.log(`PI_CACHE=${{Math.round(cache)}}`);
console.log(`PI_TURNS=${{turns}}`);"""


class PiAgent(BaseAgent):
    """Pi in the task sandbox against one pinned OpenAI-compatible endpoint."""

    def __init__(
        self,
        logs_dir,
        model_name: str | None = None,
        *,
        base_url: str | None = None,
        base_url_env: str = "AES_EVAL_BASE_URL",
        api_key_env: str = "AES_EVAL_API_KEY",
        api: str = "openai-completions",
        context_window: int = 262_144,
        max_output_tokens: int = 32_768,
        reasoning: bool = False,
        sampling: dict[str, Any] | None = None,
        provider_routing: dict[str, Any] | None = None,
        tools: str | None = "bash",
        skills: bool = False,
        compaction: bool = False,
        agent_timeout_sec: float | None = None,
        pi_version: str = PI_VERSION,
        node_version: str = NODE_VERSION,
        instruction_suffix: str = "",
        extra_env: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir=logs_dir, model_name=model_name, extra_env=extra_env, **kwargs)
        if not model_name:
            raise ValueError("PiAgent needs the exact served model id as model_name")
        self._model = model_name
        self._base_url = base_url
        self._base_url_env = base_url_env
        self._api_key_env = api_key_env
        self._api = api
        self._context_window = int(context_window)
        self._max_output_tokens = int(max_output_tokens)
        self._reasoning = bool(reasoning)
        self._sampling = validate_sampling(sampling)
        self._provider_routing = provider_routing
        self._tools = tools
        self._skills = bool(skills)
        self._compaction = bool(compaction)
        self._agent_timeout_sec = positive_timeout(agent_timeout_sec)
        self._pi_version = pi_version
        self._node_version = node_version
        self._instruction_suffix = instruction_suffix
        self._run_dir = PurePosixPath(f"/tmp/aes-pi-{uuid.uuid4().hex}")

    @staticmethod
    def name() -> str:
        return "aes-pi"

    def version(self) -> str:
        return f"pi-{self._pi_version}"

    def resolved_base_url(self) -> str:
        url = self._base_url or self._get_env(self._base_url_env)
        if not url:
            raise RuntimeError(
                f"no endpoint URL: pass base_url or set {self._base_url_env} on the launching host")
        return normalize_base_url(url)

    def argv(self, instruction: str) -> list[str]:
        """The Pi command line for one instruction (without the timeout wrapper)."""
        argv = ["pi", "-p", "--no-session", "--mode", "json",
                "--provider", PROVIDER_NAME, "--model", self._model]
        if self._tools:
            argv += ["--tools", self._tools]
        if not self._skills:
            argv.append("--no-skills")
        # A leading dash would be parsed as a flag.
        argv.append(instruction if not instruction.startswith("-") else " " + instruction)
        return argv

    async def setup(self, environment: BaseEnvironment) -> None:
        result = await environment.exec(
            command=install_script(self._node_version, self._pi_version),
            user="root", timeout_sec=600)
        if result.return_code != 0:
            detail = (result.stderr or result.stdout or "")[-500:]
            raise RuntimeError(f"Pi install failed rc={result.return_code}: {detail}")

        models = json.dumps(models_json(self._model, self.resolved_base_url(), self._api,
                                        self._api_key_env, self._context_window,
                                        self._max_output_tokens, self._reasoning),
                            separators=(",", ":"))
        settings = json.dumps({"compaction": {"enabled": self._compaction}})
        extension = request_extension(self._model, self._sampling, self._provider_routing,
                                      str(self.environment_logs_dir / "request_policy.jsonl"))
        command = f"""set -e
mkdir -p {_PI_DIR}/extensions
cat > {_PI_DIR}/models.json <<'AESMODELS'
{models}
AESMODELS
cat > {_PI_DIR}/settings.json <<'AESSETTINGS'
{settings}
AESSETTINGS
cat > {_PI_DIR}/extensions/aes_request_policy.js <<'AESEXTENSION'
{extension}
AESEXTENSION
chmod -R a+rX {_PI_DIR}
"""
        result = await environment.exec(command=command, user="root", timeout_sec=60)
        if result.return_code != 0:
            raise RuntimeError(f"Pi config write failed rc={result.return_code}")

    def _runner(self, pi_command: str) -> tuple[str, dict[str, PurePosixPath]]:
        logs = self.environment_logs_dir
        paths = {
            "stream": self._run_dir / "stream.jsonl",
            "transcript": logs / "pi_transcript.jsonl",
            "stderr": logs / "pi_stderr.txt",
            "exit": self._run_dir / "exit",
            "pid": self._run_dir / "pid",
        }
        p = paths
        # Pi emits a "message_update" event with the full partial message on every token.
        # Keeping those and filtering afterwards grows quadratically and can fill the task disk,
        # so a FIFO filters them while Pi runs. Some task images put their own older Node first
        # on PATH, and Pi's shebang uses "env node", so PATH is pinned here as well.
        runner = f"""export PATH=/opt/node/bin:/usr/local/bin:$PATH
grep -v '"type":"message_update"' < {p['stream']} > {p['transcript']}.tmp &
filter_pid=$!
{pi_command} > {p['stream']} 2> {p['stderr']}
pi_exit=$?
wait "$filter_pid"
filter_exit=$?
rm -f {p['stream']}
if [ "$filter_exit" -ne 0 ] && [ "$filter_exit" -ne 1 ]; then
  printf 'transcript filter exited %s\\n' "$filter_exit" >> {p['stderr']}
  if [ "$pi_exit" -eq 0 ]; then pi_exit=74; fi
fi
mv -f {p['transcript']}.tmp {p['transcript']}
printf 'PI_EXIT=%s\\n' "$pi_exit" > {p['exit']}.tmp
mv -f {p['exit']}.tmp {p['exit']}
"""
        return runner, paths

    def _status_command(self, pi_pid: int, p: dict[str, PurePosixPath]) -> str:
        # The model can kill its own runner (e.g. a broad pkill), leaving no process to write the
        # exit marker; the state it left is still a valid outcome, recorded as exit 143. Pi can
        # also keep its event loop alive after its final "agent_end" event; after three quiet
        # minutes the finished child is sent SIGTERM so the runner closes the logs normally.
        return f"""if test -f {p['exit']}; then
  cat {p['exit']}
elif kill -0 {pi_pid} 2>/dev/null; then
  if test -f {p['transcript']}.tmp && tail -n 1 {p['transcript']}.tmp | grep -q '"type":"agent_end"'; then
    age="$(($(date +%s) - $(stat -c %Y {p['transcript']}.tmp)))"
    if test "$age" -ge 180; then
      timeout_pid="$(pgrep -P {pi_pid} -x timeout | head -1)"
      if test -n "$timeout_pid"; then child="$(pgrep -P "$timeout_pid" | head -1)";
      else child="$(pgrep -P {pi_pid} -x node | head -1)"; fi
      test -z "$child" || kill -TERM "$child" 2>/dev/null || true
    fi
  fi
  printf 'PI_RUNNING=1\\n'
elif test -f {p['exit']}; then
  cat {p['exit']}
else
  if test -f {p['transcript']}.tmp && ! test -f {p['transcript']}; then
    mv -f {p['transcript']}.tmp {p['transcript']}
  fi
  printf 'PI_EXIT=143\\n' > {p['exit']}.tmp
  mv -f {p['exit']}.tmp {p['exit']}
  cat {p['exit']}
fi"""

    async def _stop(self, environment: BaseEnvironment, pi_pid: int) -> None:
        """Best effort: terminate the detached runner's session (its process group)."""
        command = (f"kill -TERM -- -{pi_pid} 2>/dev/null; sleep 5; "
                   f"kill -KILL -- -{pi_pid} 2>/dev/null; true")
        try:
            await asyncio.shield(environment.exec(command=command, timeout_sec=_SHORT_TIMEOUT_SEC))
        except BaseException as exc:  # noqa: BLE001  (never mask the cancellation that brought us here)
            self.logger.warning("could not stop Pi session %s: %s", pi_pid, exc)

    async def run(self, instruction: str, environment: BaseEnvironment,
                  context: AgentContext) -> None:
        env = {
            "PI_CODING_AGENT_DIR": str(_PI_DIR),
            "PI_SKIP_VERSION_CHECK": "1",
            "PI_TELEMETRY": "0",
            **self.extra_env,
        }
        if not env.get(self._api_key_env):
            raise RuntimeError(
                f"{self._api_key_env} is not set in the agent env; pass it with "
                f"env={{'{self._api_key_env}': '${{{self._api_key_env}}}'}}")

        argv = self.argv(instruction + self._instruction_suffix)
        if self._agent_timeout_sec is not None:
            argv = ["timeout", "--signal=TERM", "--kill-after=10s",
                    f"{self._agent_timeout_sec}s", *argv]
        runner, paths = self._runner(shlex.join(argv))
        launch = f"""set -e
mkdir -p {self.environment_logs_dir}
if mkdir {self._run_dir} 2>/dev/null; then
  mkfifo {paths['stream']}
  nohup setsid sh -c {shlex.quote(runner)} </dev/null >{self._run_dir}/runner.log 2>&1 &
  printf 'PI_PID=%s\\n' "$!" > {paths['pid']}.tmp
  mv -f {paths['pid']}.tmp {paths['pid']}
fi
cat {paths['pid']}
"""
        launched = await environment.exec(command=launch, env=env, timeout_sec=_SHORT_TIMEOUT_SEC)
        pi_pid = parse_marker(launched.stdout, "PI_PID")
        if launched.return_code != 0 or pi_pid is None:
            raise RuntimeError(
                f"Pi launch failed rc={launched.return_code}: "
                f"{(launched.stderr or launched.stdout or '')[-300:]}")
        self.logger.info("Pi runner launched pid=%s", pi_pid)

        loop = asyncio.get_running_loop()
        deadline = (loop.time() + float(self._agent_timeout_sec) + _TRANSPORT_MARGIN_SEC
                    if self._agent_timeout_sec is not None else None)
        status_command = self._status_command(pi_pid, paths)
        try:
            while True:
                status = await environment.exec(command=status_command,
                                                timeout_sec=_SHORT_TIMEOUT_SEC)
                pi_exit = parse_marker(status.stdout, "PI_EXIT")
                if pi_exit is not None:
                    break
                if deadline is not None and loop.time() >= deadline:
                    raise TimeoutError("Pi did not publish its exit marker after the time limit")
                await asyncio.sleep(_POLL_INTERVAL_SEC)
        except (asyncio.CancelledError, TimeoutError):
            # Harbor cancels run() when the task's time limit passes and then runs the verifier.
            # Pi is detached, so stop its whole session first; otherwise it keeps changing the
            # container while the verifier reads it.
            await self._stop(environment, pi_pid)
            raise

        metadata: dict[str, Any] = {"pi_exit_code": pi_exit, "model_id": self._model,
                                    "context_limited": False}
        usage = await environment.exec(
            command=f"/usr/local/bin/node -e {shlex.quote(usage_script(paths['transcript']))}",
            timeout_sec=_SHORT_TIMEOUT_SEC)
        if usage.return_code == 0:
            input_tokens = parse_marker(usage.stdout, "PI_INPUT") or 0
            cache_tokens = parse_marker(usage.stdout, "PI_CACHE") or 0
            context.n_input_tokens = input_tokens + cache_tokens
            context.n_output_tokens = parse_marker(usage.stdout, "PI_OUTPUT") or 0
            context.n_cache_tokens = cache_tokens
            metadata["pi_turns"] = parse_marker(usage.stdout, "PI_TURNS") or 0
        context.metadata = {**(context.metadata or {}), **metadata}

        if pi_exit == 0 or pi_exit in TIMEOUT_EXIT_CODES:
            return  # the verifier scores whatever state Pi left behind
        tail = await environment.exec(
            command=(f"tail -c 4000 {paths['stderr']} 2>/dev/null; "
                     f"tail -n 3 {paths['transcript']} 2>/dev/null || true"),
            timeout_sec=_SHORT_TIMEOUT_SEC)
        text = (tail.stdout or "") + (tail.stderr or "")
        if is_context_limited(text):
            # Running out of context is the model's outcome, not an infrastructure failure.
            context.metadata = {**context.metadata, "context_limited": True}
            return
        raise api_error(f"Pi exited {pi_exit}: {text[-3000:]}")

    def populate_context_post_run(self, context: AgentContext) -> None:
        transcript = self.logs_dir / "pi_transcript.jsonl"
        if not transcript.is_file():
            return
        input_tokens = output_tokens = cache_tokens = turns = 0
        for line in transcript.read_text(errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            message = event.get("message") or {}
            if event.get("type") != "message_end" or message.get("role") != "assistant":
                continue
            usage = message.get("usage") or {}
            turns += 1
            input_tokens += int(usage.get("input", 0))
            output_tokens += int(usage.get("output", 0))
            cache_tokens += int(usage.get("cacheRead", 0))
        context.n_input_tokens = input_tokens + cache_tokens
        context.n_output_tokens = output_tokens
        context.n_cache_tokens = cache_tokens
        context.metadata = {**(context.metadata or {}), "pi_turns": turns}
