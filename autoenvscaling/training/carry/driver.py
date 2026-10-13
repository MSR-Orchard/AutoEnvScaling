"""The episode driver: Pi in a sandbox of the task's image, talking to the recording proxy.

Each episode gets its own container started from the environment's image (``metadata["image"]``
of the prompt row). Pi is installed into it, pointed at the proxy's per-episode route
``/v1/ep/<episode_id>`` (episode identity comes from the path: sandboxes behind NAT are not
distinguishable by address), and launched once with the instruction on stdin. The driver never
polls the sandbox while Pi runs: it waits on the exec that will return when Pi exits.

The park model
--------------
Pi owns its loop and cannot be resumed without a new message (measured on the research runs:
``--continue`` and ``--session`` load the session and exit without generating), and a new user
turn would be a train/eval mismatch. So a park never kills Pi. The driver closes the episode's
gate at the proxy; Pi blocks inside its in-flight HTTP call while the weights swap; the next
rollout reopens the gate when Miles admits the episode again. ``run_turns`` therefore rejoins
the running Pi on a resume instead of launching a second one.

Verification runs the task's ``tests/test.sh`` inside the same container after Pi finished (the
Harbor convention: the verifier writes ``/logs/verifier/reward.txt`` or ``reward.json``), and
the reward is the fraction of tests that passed, as everywhere else in this repository.

This is the research run's ``pi_driver.py`` on the release's sandbox interface
(``autoenvscaling.sandbox``) instead of its sandbox service.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import shlex
import socket
import threading
import time
import tomllib
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from autoenvscaling.sandbox.base import (
    ExecResult,
    Sandbox,
    SandboxBackend,
    SandboxSpec,
    get_backend,
)
from autoenvscaling.training.carry.episode import (
    EpisodeActiveBudgetExpired,
    TurnOutcome,
    TurnStatus,
)

logger = logging.getLogger(__name__)

# The package Harbor's own Pi agent and docker/proposer install. The old scope stopped at
# 0.73.1 and is deprecated on npm; the paper's 0.80.6 exists only under the new one.
CURRENT_PI_PACKAGE = "@earendil-works/pi-coding-agent"
LEGACY_PI_PACKAGE = "@mariozechner/pi-coding-agent"
PI_PACKAGE_RENAME_VERSION = (0, 74, 0)
PI_CONFIG_DIR = "/tmp/aes-pi-agent"            # PI_CODING_AGENT_DIR, as Harbor's agent sets it
SESSION_DIR = f"{PI_CONFIG_DIR}/sessions"
PI_OUTPUT = "/tmp/pi_out.json"
ABORT_POLL_S = float(os.environ.get("AES_CARRY_ABORT_POLL_S", "5"))
INSTALL_CONCURRENCY = int(os.environ.get("AES_CARRY_INSTALL_CONCURRENCY", "64"))
INSTALL_RETRIES = 3

# Harbor's Node install (nvm, Node 22) followed by Harbor's Pi install flags.
_PI_BOOTSTRAP = (
    "set -euo pipefail; "
    "curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | env -u NODE_VERSION bash >/dev/null 2>&1 && "
    'export NVM_DIR="$HOME/.nvm" && . "$NVM_DIR/nvm.sh" && '
    "nvm install 22 >/dev/null 2>&1 && nvm alias default 22 >/dev/null 2>&1 && "
    "npm install -g --ignore-scripts {package} >/dev/null 2>&1 && pi --version"
)

# Pi's JSON print mode can exit zero after an assistant error. Inspect its final events inside
# the sandbox; only a small JSON summary leaves it.
_PI_COMPLETION_CHECK = r'''
import json, re
last = {}
ended = False
with open('/tmp/pi_out.json') as stream:
    for line in stream:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get('type') == 'agent_start':
            ended = False
        elif event.get('type') == 'agent_end':
            ended = True
        elif event.get('type') == 'message_end':
            message = event.get('message') or {}
            if message.get('role') == 'assistant':
                last = message
error = str(last.get('errorMessage') or '')
status = re.search(r'\b([45][0-9]{2})\b', error)
reason = last.get('stopReason')
print(json.dumps({
    'agent_ended': ended,
    'stop_reason': reason if reason in ('stop', 'length', 'toolUse', 'error', 'aborted') else 'unknown',
    'assistant_error': bool(error),
    'content_blocks': len(last.get('content') or []),
    'http_status': int(status[1]) if status else None,
}))
'''


# ---- the proxy as seen from the driver ---------------------------------------------------------------
class ProxyClient:
    """Control and recording endpoints of ``training.carry.proxy`` (synchronous; call off-loop)."""

    def __init__(self, base_url: str, api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _request(self, method: str, path: str, body: dict | None = None, timeout: float = 60.0) -> Any:
        data = json.dumps(body).encode() if body is not None else (b"{}" if method == "POST" else None)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}{path}", data=data, method=method, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
        return json.loads(raw) if raw else {}

    def configure(self, sglang_url: str, served_id: str, max_turns: int, max_context: int) -> dict:
        return self._request("POST", "/config", {"sglang_url": sglang_url, "served_id": served_id,
                                                 "max_turns": max_turns, "max_context": max_context})

    def gate(self, action: str, episode_id: str) -> None:
        self._request("POST", f"/{action}?episode_id={episode_id}")

    def recording(self, episode_id: str) -> dict | None:
        try:
            return self._request("GET", f"/episodes/{episode_id}?full=1", timeout=120.0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[pi] recording fetch failed for %s: %s", episode_id, exc)
            return None

    def delete(self, episode_id: str) -> None:
        try:
            self._request("DELETE", f"/episodes/{episode_id}", timeout=15.0)
        except urllib.error.HTTPError as exc:
            if exc.code not in (404, 410):
                raise


# ---- instance leases ---------------------------------------------------------------------------------------
class InstanceLease:
    """Liveness record of one rollout process under ``state_dir/instances/<instance_id>.json``.

    Written before the first sandbox is created and refreshed by the heartbeat thread, so a
    later process of the same run can tell a crashed predecessor (stale heartbeat) from a
    concurrent job (fresh heartbeat) before removing anything.
    """

    def __init__(self, state_dir: str, run_id: str, instance_id: str):
        self.enabled = bool(state_dir)
        self.dir = Path(state_dir) / "instances" if state_dir else None
        self.run_id = run_id
        self.instance_id = instance_id
        self.started = time.time()

    def path(self, instance_id: str | None = None) -> Path:
        assert self.dir is not None
        return self.dir / f"{instance_id or self.instance_id}.json"

    def write(self) -> None:
        if not self.enabled:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        record = {"run_id": self.run_id, "instance_id": self.instance_id, "pid": os.getpid(),
                  "host": socket.gethostname(), "started": self.started, "heartbeat": time.time()}
        tmp = self.path().with_suffix(".tmp")
        tmp.write_text(json.dumps(record))
        tmp.replace(self.path())

    def touch(self) -> None:
        self.write()

    def release(self) -> None:
        if self.enabled:
            self.path().unlink(missing_ok=True)

    def remove(self, instance_id: str) -> None:
        if self.enabled:
            self.path(instance_id).unlink(missing_ok=True)

    def read_all(self) -> list[dict]:
        if not self.enabled or not self.dir.is_dir():
            return []
        out = []
        for path in sorted(self.dir.glob("*.json")):
            try:
                record = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if isinstance(record, dict) and record.get("instance_id"):
                out.append(record)
        return out


# ---- one episode's sandbox -------------------------------------------------------------------------------
class CarrySandbox:
    """A sandbox, its episode id, the Pi process that survives a park, and the time budget.

    Active execution is bounded separately from deliberate training pauses: ``pause_execution``
    and ``resume_execution`` (called by the registry at park/resume) stop and restart the
    episode clock, and a finite wall backstop bounds an abandoned parked episode.
    """

    def __init__(self, sandbox: Sandbox, episode_id: str, *, loop: asyncio.AbstractEventLoop,
                 release: Callable[[], None], proxy: ProxyClient, max_paused_s: float,
                 on_closing: Callable[[], None] = lambda: None):
        self.sandbox = sandbox
        self.episode_id = episode_id
        self._on_closing = on_closing
        self.created_at = time.monotonic()
        self.deadline: float | None = None
        self.max_paused_seconds = max_paused_s
        self._loop = loop
        self._release = release
        self._proxy = proxy
        self._budget_lock = threading.RLock()
        self._paused_at: float | None = None
        self._paused_seconds = 0.0
        self.waiter: asyncio.Task | None = None
        self._close_lock = threading.Lock()
        self._close_future: Future | None = None
        self.terminated = False
        self.recording_deleted = False

    # ---- budget --------------------------------------------------------------------------------
    def pause_execution(self) -> None:
        with self._budget_lock:
            if self._paused_at is None:
                self._paused_at = time.monotonic()

    def resume_execution(self) -> None:
        with self._budget_lock:
            if self._paused_at is not None:
                duration = time.monotonic() - self._paused_at
                self._paused_seconds += duration
                if self.deadline is not None:
                    self.deadline += duration
                self._paused_at = None

    def wall_remaining(self, timeout_s: float) -> float:
        return self.created_at + timeout_s + self.max_paused_seconds - time.monotonic()

    def episode_remaining(self, timeout_s: float) -> float:
        with self._budget_lock:
            if self.deadline is None:
                self.deadline = self.created_at + timeout_s + self._paused_seconds
            now = time.monotonic()
            clock = self._paused_at if self._paused_at is not None else now
            paused = self._paused_seconds + (now - clock)
            if paused >= self.max_paused_seconds:
                return 0.0
            return min(self.deadline - clock, self.wall_remaining(timeout_s))

    def episode_expired(self, timeout_s: float) -> bool:
        return self.episode_remaining(timeout_s) <= 0

    def expiry_reason(self, timeout_s: float) -> str | None:
        with self._budget_lock:
            now = time.monotonic()
            if self.created_at + timeout_s + self.max_paused_seconds <= now:
                return "wall"
            clock = self._paused_at if self._paused_at is not None else now
            if self._paused_seconds + now - clock >= self.max_paused_seconds:
                return "paused"
            if self.deadline is not None and self.deadline <= clock:
                return "active"
            return None

    # ---- liveness and cleanup ----------------------------------------------------------------
    def heartbeat(self) -> bool | None:
        """True: alive. False: confirmed gone. None: unknown (keep ownership)."""
        with self._close_lock:
            if self._close_future is not None:
                return None
        try:
            if asyncio.get_running_loop() is self._loop:
                return None   # a blocking wait here would deadlock the rollout loop
        except RuntimeError:
            pass              # not on an event loop: the heartbeat thread
        try:
            future = asyncio.run_coroutine_threadsafe(self.sandbox.exec("true", timeout_s=30), self._loop)
            result = future.result(timeout=60)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[pi] heartbeat unknown for %s: %s", self.episode_id, exc)
            return None
        if result.returncode == 0:
            return True
        text = (result.stderr or "") + (result.stdout or "")
        if re.search(r"No such container|is not running|not found|has been terminated", text, re.IGNORECASE):
            return False
        return None

    def close(self) -> Future:
        """Start idempotent cleanup from any thread; the future resolves once confirmed."""
        with self._close_lock:
            if self._close_future is None:
                self._close_future = Future()
                self._attempt = 0
                self._on_closing()
                waiter = self.waiter
                if waiter is not None and not self._loop.is_closed():
                    self._loop.call_soon_threadsafe(waiter.cancel)
                if not self._loop.is_closed():
                    asyncio.run_coroutine_threadsafe(self._close(), self._loop)
            return self._close_future

    async def _close(self) -> None:
        errors = []
        if not self.terminated:
            try:
                await self.sandbox.terminate()
                self.terminated = True
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))
        if not self.recording_deleted:
            try:
                await asyncio.to_thread(self._proxy.delete, self.episode_id)
                self.recording_deleted = True
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))
        if errors:
            self._attempt += 1
            delay = min(30.0, 2 ** min(self._attempt - 1, 5))
            logger.warning("[pi] cleanup pending for %s (attempt %d): %s", self.episode_id,
                           self._attempt, "; ".join(errors))
            await asyncio.sleep(delay)
            asyncio.ensure_future(self._close())
            return
        self._release()
        self._close_future.set_result(None)


# ---- the driver ---------------------------------------------------------------------------------------
class SandboxEpisodeDriver:
    """``EpisodeDriver`` over ``autoenvscaling.sandbox`` backends."""

    def __init__(self, settings: Any, args: Any, *, backend: SandboxBackend | None = None,
                 proxy: ProxyClient | None = None):
        self.settings = settings
        self.args = args
        self.backend = backend or get_backend(settings.sandbox_backend)
        self.proxy = proxy or ProxyClient(settings.proxy_url,
                                          api_key=os.environ.get(settings.proxy_api_key_env, ""))
        if not settings.proxy_public_url:
            raise RuntimeError("proxy_public_url is unset; Pi inside a sandbox cannot reach the "
                               "trainer's loopback")
        self.public_url = settings.proxy_public_url.rstrip("/")
        self.max_turns = int(settings.max_turns)
        self.pi_package = pi_package_spec(settings.pi_version or _default_pi_version(),
                                          getattr(settings, "pi_package", ""))
        if getattr(settings, "network", "allowlist") != "public" and getattr(self.backend, "name", "") == "docker":
            logger.warning("[pi] the Docker backend cannot enforce the solver allowlist; sandboxes "
                           "have open egress (the Modal backend enforces it)")
        self._live = threading.Semaphore(int(settings.max_live_sandboxes))
        self._install_sem = asyncio.Semaphore(INSTALL_CONCURRENCY)
        # Capacity accounting. A sandbox is ACTIVE from creation until close() starts (its
        # episode is counted by the rollout's in-flight counter, or it is parked in the
        # registry), then CLOSING until the backend confirmed the deletion. The two sets are
        # disjoint so that the rollout's deficit (capacity - in_flight - held) never subtracts
        # an active episode twice: `held` is the closing set only.
        self._active: set[CarrySandbox] = set()
        self._closing: set[CarrySandbox] = set()
        self._state_lock = threading.Lock()
        self._proxy_configured = False
        self.instance_id = uuid.uuid4().hex[:12]
        self.lease = InstanceLease(settings.state_dir, settings.run_id, self.instance_id)
        self.__tokenizer = None
        self.__tokenizer_tried = False

    # ---- accounting ------------------------------------------------------------------------------
    def pending_cleanup_count(self) -> int:
        """Sandboxes whose deletion has started but is not confirmed: they hold capacity that
        neither an in-flight episode nor the registry accounts for."""
        with self._state_lock:
            return len(self._closing)

    def live_count(self) -> int:
        """Sandboxes holding a capacity slot (active or closing)."""
        with self._state_lock:
            return len(self._active) + len(self._closing)

    def _mark_closing(self, sb: CarrySandbox) -> None:
        with self._state_lock:
            self._active.discard(sb)
            self._closing.add(sb)

    async def wait_cleanup(self, timeout_s: float) -> None:
        """Wait for deletions already in progress (never starts new ones)."""
        with self._state_lock:
            closing = list(self._closing)
        futures = [asyncio.wrap_future(sb.close()) for sb in closing]
        if futures:
            await asyncio.wait(futures, timeout=timeout_s)

    async def close_all(self, timeout_s: float) -> None:
        """Shutdown: release every sandbox this process still holds, then wait for confirmation."""
        with self._state_lock:
            held = list(self._active) + list(self._closing)
        futures = [asyncio.wrap_future(sb.close()) for sb in held]
        if futures:
            await asyncio.wait(futures, timeout=timeout_s)

    def _release_slot(self, sb: CarrySandbox) -> None:
        with self._state_lock:
            self._active.discard(sb)
            self._closing.discard(sb)
        self._live.release()

    @property
    def tokenizer(self):
        """Tokenizer for reseeding an empty sample's prompt; lazy and never fatal."""
        if self.__tokenizer is None and not self.__tokenizer_tried:
            self.__tokenizer_tried = True
            path = getattr(self.args, "hf_checkpoint", None)
            try:
                from transformers import AutoTokenizer
                self.__tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[pi] could not load the reseed tokenizer from %s: %s", path, exc)
        return self.__tokenizer

    def sandbox_name(self, episode_id: str) -> str:
        """``<run_id>-<instance_id>-<episode_id>``: the run (one training job, stable across its
        restarts) and the process that owns the sandbox, so cleanup never crosses either."""
        return f"{self.settings.run_id}-{self.instance_id}-{episode_id}"

    async def reap_orphans(self) -> int:
        """Remove sandboxes left by a *dead* previous process of this run.

        Ownership is in the sandbox name (run and instance ids) and liveness in the instance
        lease under ``state_dir`` (written before the first sandbox, refreshed by the heartbeat).
        Only instances of this run whose lease has gone stale are reaped; a live instance, an
        instance of another run, and sandboxes of unknown ownership are left alone. Without a
        state directory nothing can be attributed, so nothing is reaped.
        """
        reaper = getattr(self.backend, "remove_by_prefix", None)
        if reaper is None:
            logger.info("[pi] backend %s cannot list sandboxes; orphans of a previous process "
                        "must be removed by hand (prefix %s-)", getattr(self.backend, "name", "?"),
                        self.settings.run_id)
            return 0
        if not self.lease.enabled:
            logger.warning("[pi] no state_dir: sandboxes of a crashed process cannot be attributed "
                           "and are not reaped")
            return 0
        removed = 0
        stale_s = float(self.settings.lease_stale_s)
        for lease in self.lease.read_all():
            if lease.get("instance_id") == self.instance_id or lease.get("run_id") != self.settings.run_id:
                continue
            age = time.time() - float(lease.get("heartbeat") or 0)
            if age < stale_s:
                logger.info("[pi] instance %s of this run is alive (heartbeat %.0fs ago); leaving "
                            "its sandboxes", lease["instance_id"], age)
                continue
            try:
                n = int(await reaper(f"{self.settings.run_id}-{lease['instance_id']}-"))
            except Exception as exc:  # noqa: BLE001 - keep the record; the next start retries
                logger.warning("[pi] could not remove the sandboxes of dead instance %s (%s); "
                               "keeping its lease so a later start retries", lease["instance_id"], exc)
                continue
            logger.warning("[pi] reaped %d sandboxes of dead instance %s (heartbeat %.0fs ago)",
                           n, lease["instance_id"], age)
            # Only a sweep that completed proves nothing of this instance is left running; the
            # ownership record is what lets a later process find those sandboxes at all.
            self.lease.remove(lease["instance_id"])
            removed += n
        return removed

    # ---- sandbox -----------------------------------------------------------------------------------
    def _meta(self, sample: Any) -> dict:
        return dict(getattr(sample, "metadata", None) or {})

    def _timeout(self) -> float:
        return float(self.settings.episode_timeout_s)

    def _remaining(self, sb: CarrySandbox) -> float:
        remaining = sb.episode_remaining(self._timeout())
        if remaining <= 0:
            reason = sb.expiry_reason(self._timeout())
            if reason == "active":
                raise EpisodeActiveBudgetExpired("Pi active execution budget exhausted")
            raise TimeoutError(f"episode {reason or 'unknown'} lifetime budget exhausted")
        return remaining

    def _point_proxy_at_engines(self) -> None:
        """Tell the proxy which engines serve, once per process: the router address only exists
        once the job runs, and a stale backend would record a different model's log-probabilities."""
        if self._proxy_configured:
            return
        ip = getattr(self.args, "sglang_router_ip", None)
        port = getattr(self.args, "sglang_router_port", None)
        if not ip or not port:
            logger.warning("[pi] no router on args; leaving the proxy backend as it is")
            self._proxy_configured = True
            return
        try:
            info = self.proxy.configure(f"http://{ip}:{port}", self.settings.served_model, self.max_turns,
                                        int(getattr(self.args, "rollout_max_context_len", 98304)))
        except Exception as exc:
            raise RuntimeError(f"could not point the proxy at the engines ({exc}); refusing to run "
                               "episodes against an unknown backend") from exc
        logger.info("[pi] proxy backend set to %s", info)
        self._proxy_configured = True

    async def create_sandbox(self, sample: Any) -> CarrySandbox:
        await asyncio.to_thread(self._point_proxy_at_engines)
        meta = self._meta(sample)
        image = meta.get("image") or ""
        if not image:
            raise ValueError("prompt row carries no image; cannot start a sandbox")
        episode_id = f"ep-{uuid.uuid4().hex[:20]}"
        # Admission on the event loop, never inside an executor thread.
        while not self._live.acquire(blocking=False):
            await asyncio.sleep(0.5)
        loop = asyncio.get_running_loop()
        released = False

        def release_once():
            nonlocal released
            if not released:
                released = True
                self._release_slot(sb_holder[0])

        sb_holder: list[CarrySandbox] = []
        spec = SandboxSpec(image=image, cpu=int(self.settings.sandbox_cpu),
                           memory_gib=int(self.settings.sandbox_memory_gib),
                           timeout_s=int(self._timeout() + self.settings.max_paused_s + 600),
                           workdir=meta.get("workdir") or self.settings.sandbox_workdir,
                           domain_allowlist=self._allowed_hosts(meta),
                           name=self.sandbox_name(episode_id))
        try:
            sandbox = await self.backend.create(spec)
        except BaseException:
            self._live.release()
            raise
        sb = CarrySandbox(sandbox, episode_id, loop=loop, release=release_once, proxy=self.proxy,
                          max_paused_s=float(self.settings.max_paused_s),
                          on_closing=lambda: self._mark_closing(sb_holder[0]))
        sb_holder.append(sb)
        sb.deadline = sb.created_at + self._timeout()
        with self._state_lock:
            self._active.add(sb)
        logger.info("[carry] sandbox episode=%s image=%s", episode_id, image)
        try:
            await asyncio.wait_for(self._install_pi(sb), self._remaining(sb))
        except BaseException:
            try:
                await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(sb.close())), 30)
            except Exception:  # noqa: BLE001
                logger.warning("[pi] release after failed setup did not complete")
            raise
        return sb

    def _allowed_hosts(self, meta: dict) -> list[str] | None:
        """The solver's network, as the repository's Harbor trials apply it: an allowlist of the
        proxy's host, the hosts Pi installs from, and whatever the task itself allows. None opens
        the network (``AES_SOLVER_NETWORK=public``)."""
        if getattr(self.settings, "network", "allowlist") == "public":
            return None
        hosts = [h for h in (urlsplit(self.public_url).hostname,) if h]
        hosts += list(_pi_install_hosts())
        hosts += task_allowed_hosts(meta.get("task_dir") or "")
        return sorted(dict.fromkeys(hosts))

    async def _install_pi(self, sb: CarrySandbox) -> None:
        async with self._install_sem:
            for attempt in range(INSTALL_RETRIES):
                probe = await sb.sandbox.exec(
                    "bash", "-lc",
                    "command -v curl >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq "
                    "curl ca-certificates >/dev/null 2>&1); command -v curl", timeout_s=600)
                if probe.returncode == 0:
                    break
                logger.warning("[pi] curl unavailable (attempt %d/%d) for %s", attempt + 1,
                               INSTALL_RETRIES, sb.episode_id)
                await asyncio.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"pi install failed: curl still missing after {INSTALL_RETRIES} attempts")
            res = await sb.sandbox.exec("bash", "-lc", _PI_BOOTSTRAP.format(package=self.pi_package),
                                        timeout_s=1200)
        if res.returncode != 0:
            detail = (res.stderr or "").strip() or (res.stdout or "").strip()
            raise RuntimeError(f"pi install failed (rc={res.returncode}): {detail[:300] or '<no output>'}")
        models = {"providers": {"openai": {
            "baseUrl": f"{self.public_url}/v1/ep/{sb.episode_id}",
            "api": "openai-completions", "apiKey": "$OPENAI_API_KEY", "authHeader": True,
            "compat": {"supportsStore": False, "supportsDeveloperRole": False,
                       "supportsReasoningEffort": False, "supportsUsageInStreaming": False,
                       "maxTokensField": "max_tokens", "supportsStrictMode": False},
            "models": [{"id": self.settings.served_model, "name": "policy", "api": "openai-completions",
                        "reasoning": True, "input": ["text"],
                        "contextWindow": int(getattr(self.args, "rollout_max_context_len", 98304)),
                        "maxTokens": int(getattr(self.args, "rollout_max_response_len", 16384)),
                        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}}]}}}
        await sb.sandbox.write_file(f"{PI_CONFIG_DIR}/models.json",
                                    (json.dumps(models, indent=2) + "\n").encode("utf-8"))
        # Fail loudly rather than fall back to api.openai.com, which would spend money and
        # record nothing.
        res = await sb.sandbox.exec("bash", "-lc", f'grep -q "/v1/ep/" {PI_CONFIG_DIR}/models.json && echo PLANT_OK',
                                    timeout_s=120)
        if "PLANT_OK" not in (res.stdout or ""):
            raise RuntimeError(f"models.json plant failed: {res.stdout} {res.stderr}")

    # ---- turns ---------------------------------------------------------------------------------------
    async def run_turns(self, *, sample: Any, sandbox: CarrySandbox, turn_state: Any,
                        sampling_params: dict, should_abort: Callable[[], bool],
                        start_turn_index: int) -> TurnOutcome:
        if should_abort():
            return TurnOutcome(TurnStatus.ABORTED, {"turn": start_turn_index}, start_turn_index)
        self._remaining(sandbox)
        if sandbox.waiter is not None:
            # A resume: rejoin the running Pi rather than fork the episode.
            if not sandbox.waiter.done():
                await self._gate("unpark", sandbox.episode_id)
            parked = await self._await_pi(sandbox, should_abort, self._remaining(sandbox), sandbox.waiter)
            return await self._finish(sample, sandbox, parked)

        instruction = self._meta(sample).get("instruction") or _prompt_text(sample)
        prompt_path = f"/tmp/aes_pi_prompt_{sandbox.episode_id}.txt"
        # The task text never appears in a long-lived process's argv (a task's own `pkill -f`
        # could match it); it is staged in a file and read on stdin, Pi's native interface.
        await sandbox.sandbox.write_file(prompt_path, instruction.encode("utf-8"))
        launch = (f"mkdir -p {SESSION_DIR}; . ~/.nvm/nvm.sh; "
                  f"{{ rm -f {shlex.quote(prompt_path)} || exit $?; "
                  f"pi --print --mode json --session-dir {SESSION_DIR} --provider openai "
                  f"--model {shlex.quote(self.settings.served_model)} > {PI_OUTPUT} 2>&1; rc=$?; "
                  f"echo EXIT=$rc; exit $rc; }} < {shlex.quote(prompt_path)}")
        key = os.environ.get(self.settings.proxy_api_key_env, "") or "proxy"
        exec_timeout = int(sandbox.wall_remaining(self._timeout()))
        waiter = asyncio.create_task(sandbox.sandbox.exec(
            "bash", "-lc", launch, timeout_s=exec_timeout,
            env={"OPENAI_API_KEY": key, "PI_CODING_AGENT_DIR": PI_CONFIG_DIR}))
        sandbox.waiter = waiter
        parked = await self._await_pi(sandbox, should_abort, self._remaining(sandbox), waiter)
        return await self._finish(sample, sandbox, parked)

    async def _gate(self, action: str, episode_id: str) -> None:
        """park/unpark THIS episode's gate (the global gate belongs to the rollout cut)."""
        try:
            await asyncio.to_thread(self.proxy.gate, action, episode_id)
        except Exception as exc:
            logger.error("[pi] proxy %s(%s) failed: %s", action, episode_id, exc)
            raise

    async def _await_pi(self, sandbox: CarrySandbox, should_abort: Callable[[], bool], timeout: float,
                        waiter: asyncio.Task) -> bool:
        """Wait for Pi without polling the sandbox. True means parked (Pi still alive)."""
        deadline = min(sandbox.deadline or float("inf"), time.monotonic() + timeout)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._remaining(sandbox)   # raises the budget error with its reason
                raise TimeoutError("Pi wait deadline exhausted")
            done, _ = await asyncio.wait({waiter}, timeout=min(ABORT_POLL_S, remaining))
            if done:
                if waiter.cancelled():
                    raise RuntimeError("Pi waiter was cancelled before completion")
                result: ExecResult = waiter.result()
                if result.timed_out or result.returncode:
                    raise RuntimeError(f"Pi process failed (rc={result.returncode}, "
                                       f"timed_out={result.timed_out}): {result.stderr or result.stdout}")
                return False
            if should_abort():
                await self._gate("park", sandbox.episode_id)
                return True

    async def _finish(self, sample: Any, sandbox: CarrySandbox, parked: bool) -> TurnOutcome:
        rec = await asyncio.to_thread(self.proxy.recording, sandbox.episode_id)
        if parked:
            turns = int((rec or {}).get("turns") or 0)
            if rec and rec.get("n_tokens", 0) > rec.get("prompt_len", 0):
                self._apply_recording(sample, rec)
            return TurnOutcome(TurnStatus.ABORTED, {"turn": turns}, turns)
        if rec and rec.get("error"):
            return TurnOutcome(TurnStatus.FAILED, {}, int(rec.get("turns") or 0),
                               error=f"proxy episode failed: {rec['error']}")
        if rec is None or not rec.get("n_tokens"):
            return TurnOutcome(TurnStatus.FAILED, {"turn": 0}, 0,
                               error="proxy recorded no tokens for this episode")
        terminal = await self._completion_status(sandbox)
        md = getattr(sample, "metadata", None)
        if md is None:
            sample.metadata = md = {}
        md["pi_terminal_status"] = terminal
        bounded_stop = rec.get("stop_reason") in ("context_limit", "max_turns")
        turns = int(rec.get("turns") or 0)
        if (not terminal["agent_ended"] or terminal["assistant_error"]
                or terminal["stop_reason"] not in ("stop", "length")
                or (not terminal["content_blocks"] and not bounded_stop)):
            return TurnOutcome(TurnStatus.FAILED, {}, turns,
                               error=f"Pi terminal status invalid (stop_reason={terminal['stop_reason']}, "
                                     f"agent_ended={terminal['agent_ended']}, http_status={terminal['http_status']})")
        if not rec.get("tool_results_seen"):
            return TurnOutcome(TurnStatus.FAILED, {"turn": turns}, turns,
                               error="episode produced zero tool results")
        self._apply_recording(sample, rec)
        return TurnOutcome(TurnStatus.DONE, {"turn": turns}, turns)

    async def _completion_status(self, sandbox: CarrySandbox) -> dict:
        result = await sandbox.sandbox.exec("python3", "-I", "-c", _PI_COMPLETION_CHECK,
                                            timeout_s=int(min(self._remaining(sandbox), 60.0)), workdir="/tmp")
        if result.returncode or result.timed_out:
            raise RuntimeError("could not validate Pi's final session events")
        status = json.loads(result.stdout)
        if (not isinstance(status.get("agent_ended"), bool)
                or not isinstance(status.get("assistant_error"), bool)
                or not isinstance(status.get("content_blocks"), int)
                or status.get("stop_reason") not in ("stop", "length", "toolUse", "error", "aborted", "unknown")
                or (status.get("http_status") is not None and not isinstance(status["http_status"], int))):
            raise RuntimeError("invalid Pi terminal-status diagnostic")
        return status

    def _apply_recording(self, sample: Any, rec: dict) -> None:
        """Copy the engine's own ids and log-probabilities onto the sample; nothing is recomputed."""
        apply_recording(sample, rec, max_context=int(getattr(self.args, "rollout_max_context_len", 0) or 0))

    # ---- verify ---------------------------------------------------------------------------------------
    async def verify(self, *, sample: Any, sandbox: CarrySandbox, turn_state: Any) -> float:
        """Score the episode, then ALWAYS release the sandbox: verify is the terminal call."""
        try:
            return await self._verify_inner(sample, sandbox)
        finally:
            try:
                await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(sandbox.close())), 30)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[pi] sandbox release pending: %s", exc)

    async def _verify_inner(self, sample: Any, sandbox: CarrySandbox) -> float:
        task_dir = self._meta(sample).get("task_dir") or ""
        tests = Path(task_dir) / "tests"
        if not task_dir or not (tests / "test.sh").is_file():
            raise RuntimeError("prompt row has no task_dir with tests/test.sh; cannot verify")
        started = time.monotonic()
        budget = float(self.settings.verify_timeout_s)

        def remaining() -> float:
            left = min(self._remaining(sandbox), budget - (time.monotonic() - started))
            if left <= 0:
                raise TimeoutError("verifier wall-clock budget exhausted")
            return left

        # Nothing from before this run may score it: the agent could have planted a reward file
        # or a tests directory, and an earlier verifier attempt could have left one behind. Pi
        # has exited by now, so after the reset only test.sh can write the reward.
        fresh = await sandbox.sandbox.exec("bash", "-lc", "rm -rf /tests /logs/verifier && mkdir -p /tests /logs/verifier",
                                           timeout_s=int(min(remaining(), 60)), workdir="/")
        if fresh.returncode or fresh.timed_out:
            raise RuntimeError(f"could not reset the verifier directories: {fresh.stderr or fresh.stdout}")
        await sandbox.sandbox.upload_dir(tests, "/tests")
        res = await sandbox.sandbox.exec(
            "bash", "-lc", "bash /tests/test.sh > /logs/verifier/test-stdout.txt 2>&1; echo RC=$?; "
                           "if [ -s /logs/verifier/reward.txt ]; then echo REWARD_TXT; cat /logs/verifier/reward.txt; "
                           "elif [ -s /logs/verifier/reward.json ]; then echo REWARD_JSON; cat /logs/verifier/reward.json; "
                           "else echo NO_REWARD; tail -c 2000 /logs/verifier/test-stdout.txt; fi",
            timeout_s=int(remaining()), workdir=self.settings.sandbox_workdir)
        if res.timed_out:
            raise RuntimeError("verifier timed out")
        return parse_reward_output(res.stdout or "")


# ---- helpers ------------------------------------------------------------------------------------------------
def apply_recording(sample: Any, rec: dict, *, max_context: int = 0) -> None:
    """Copy a proxy recording onto a Miles sample.

    Miles splits a sample into prompt and response and asserts
    ``len(rollout_log_probs) == response_length``: the response is everything after the initial
    prompt, generated tokens and injected tool results alike, with ``loss_mask`` deciding what
    trains. A recording longer than the context limit is rejected rather than truncated: cutting
    mid-turn would train on a trajectory that never happened.
    """
    tokens = rec.get("tokens") or []
    logprobs = rec.get("logprobs") or []
    loss_mask = rec.get("loss_mask") or []
    if not (len(tokens) == len(logprobs) == len(loss_mask)):
        raise RuntimeError(f"recording is internally inconsistent: tokens={len(tokens)} "
                           f"logprobs={len(logprobs)} mask={len(loss_mask)}")
    prompt_len = int(rec.get("prompt_len") or 0)
    if prompt_len <= 0 or prompt_len >= len(tokens):
        raise RuntimeError(f"recording has an unusable prompt_len={prompt_len} for {len(tokens)} tokens")
    if max_context and len(tokens) > max_context:
        raise RuntimeError(f"recording exceeds the context limit: {len(tokens)} > {max_context} tokens")
    sample.tokens = list(tokens)
    sample.rollout_log_probs = list(logprobs[prompt_len:])
    sample.loss_mask = list(loss_mask[prompt_len:])
    sample.response_length = len(tokens) - prompt_len
    if rec.get("weight_versions") is not None and hasattr(sample, "weight_versions"):
        sample.weight_versions = [str(v) for v in rec["weight_versions"]]
    assert len(sample.rollout_log_probs) == sample.response_length


class VerifierFailed(RuntimeError):
    """The verifier did not produce a usable score; the episode is dropped, never rewarded."""


# The verifier contract. Harbor itself ignores ``test.sh``'s exit status and lets the reward
# file decide, and the admitted tasks were validated under that rule, so a failing test suite
# may legitimately leave a nonzero status (pytest exits 1 when tests fail). The carry therefore
# accepts exit 0 (success) and exit 1 (ordinary failure) together with a fresh, well-formed
# reward in [0, 1]. Everything else means the verifier did not complete and is a failure:
# any other exit status (pytest 2-5, 124 timeout, 126/127 command not found, 128+ signals),
# no reward file, a malformed reward, or exit 1 reporting a full reward (a contradiction).
ORDINARY_EXIT_CODES = (0, 1)


def parse_reward_output(text: str) -> float:
    """Apply the verifier contract to the verify command's output: an ``RC=<status>`` line,
    then ``REWARD_TXT``/``REWARD_JSON`` followed by the file this run wrote, or ``NO_REWARD``."""
    status_line = re.search(r"^RC=(\d+)$", text, re.MULTILINE)
    marker = re.search(r"^(REWARD_TXT|REWARD_JSON|NO_REWARD)$", text, re.MULTILINE)
    if status_line is None or marker is None:
        raise VerifierFailed(f"verifier output has no status or reward marker: {text[-500:]}")
    exit_code = int(status_line.group(1))
    body = text[marker.end():].strip()
    if exit_code not in ORDINARY_EXIT_CODES:
        raise VerifierFailed(f"verifier exited with status {exit_code}; its output is not a score: "
                             f"{body[-300:]}")
    if marker.group(1) == "NO_REWARD" or not body:
        raise VerifierFailed(f"verifier wrote no reward file (exit {exit_code}): {body[-300:]}")
    try:
        parsed = json.loads(body)
    except ValueError:
        parsed = body
    try:
        if isinstance(parsed, dict):
            if "reward" in parsed:
                value = float(parsed["reward"])
            elif parsed and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                for v in parsed.values()):
                value = sum(float(v) for v in parsed.values()) / len(parsed)
            else:
                raise VerifierFailed(f"reward.json has no numeric reward: {body[:200]}")
        elif isinstance(parsed, (bool, list)) or parsed is None:
            raise VerifierFailed(f"malformed reward: {body[:200]}")
        else:
            value = float(str(parsed).strip().split()[0])
    except (TypeError, ValueError) as exc:
        raise VerifierFailed(f"malformed reward: {body[:200]}") from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise VerifierFailed(f"reward {value!r} is not a fraction in [0, 1]")
    if exit_code == 1 and value >= 1.0:
        raise VerifierFailed("verifier exited with status 1 (tests failed) but reported a full "
                             "reward; refusing the contradiction")
    return round(value, 6)


def _prompt_text(sample: Any) -> str:
    prompt = getattr(sample, "prompt", "")
    if isinstance(prompt, list):
        for m in prompt:
            if isinstance(m, dict) and m.get("role") == "user":
                return str(m.get("content", ""))
        return ""
    return str(prompt)


def pi_package_spec(version: str, package: str = "") -> str:
    """``<package>@<version>`` with the npm scope Harbor uses for that version: the package was
    renamed at 0.74.0 and the old scope has no later releases."""
    if not package:
        package = CURRENT_PI_PACKAGE
        try:
            if tuple(int(p) for p in version.split(".")[:3]) < PI_PACKAGE_RENAME_VERSION:
                package = LEGACY_PI_PACKAGE
        except ValueError:
            pass
    return f"{package}@{version or 'latest'}"


def _default_pi_version() -> str:
    try:
        from autoenvscaling.calibration.solver import PI_VERSION
        return PI_VERSION
    except Exception:  # noqa: BLE001
        return "latest"


def _pi_install_hosts() -> tuple[str, ...]:
    try:
        from autoenvscaling.calibration.solver import PI_INSTALL_HOSTS
        return tuple(PI_INSTALL_HOSTS)
    except Exception:  # noqa: BLE001
        return ("raw.githubusercontent.com", "github.com", "nodejs.org", "registry.npmjs.org")


def task_allowed_hosts(task_dir: str) -> list[str]:
    """Hosts a task's ``task.toml`` allows under ``network_mode = "allowlist"`` (as
    ``validation.harbor_runner.stage_task`` merges them). A task that names an ``[agent] user``
    is logged: the sandbox interface runs commands as the image's default user."""
    if not task_dir:
        return []
    path = Path(task_dir) / "task.toml"
    if not path.is_file():
        return []
    try:
        config = tomllib.loads(path.read_text())
    except (OSError, ValueError):
        return []
    agent = config.get("agent")
    if isinstance(agent, dict) and agent.get("user"):
        logger.warning("[pi] task %s sets agent.user=%r; the sandbox backends run as the image's "
                       "default user", path.parent.name, agent["user"])
    environment = config.get("environment") or {}
    if environment.get("network_mode") == "allowlist":
        return [str(h) for h in (environment.get("allowed_hosts") or [])]
    return []
