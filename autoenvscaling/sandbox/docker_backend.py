"""Docker sandbox backend, for local development and tests.

It drives the ``docker`` CLI through subprocesses, so it needs nothing beyond a working Docker
install. Each sandbox is one long-lived container started with ``sleep`` as its command (under
``--init`` so orphaned processes are reaped); commands run through ``docker exec`` and files move
as tar streams through ``docker cp``.

Differences from the Modal backend:

* Secrets: ``SandboxSpec.secrets`` names host environment variables. Their values are passed to
  the container through the docker client's own environment (``-e NAME`` without a value), so
  they never appear on a command line.
* Blocked hosts: enforced the same best-effort way as on Modal, by mapping each host to 0.0.0.0
  in the container's ``/etc/hosts`` (``--add-host``). A process with root inside the container
  can undo this, and it does not stop connections made by IP address.
* Timeouts: ``exec`` wraps the command in the image's ``timeout`` utility (GNU coreutils or
  busybox), so the command itself is stopped inside the container, and also applies a host-side
  deadline as a backstop. A timed-out command returns ``TIMEOUT_RETURNCODE`` (124).
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import shutil
import tarfile
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from autoenvscaling.sandbox.base import ExecResult, SandboxSpec

logger = logging.getLogger(__name__)

TIMEOUT_RETURNCODE = 124
# Extra seconds the host waits beyond an exec's own timeout before killing the docker client.
_HOST_TIMEOUT_GRACE_S = 30
# Seconds between SIGTERM and SIGKILL for the in-container `timeout` wrapper.
_KILL_AFTER_S = 10


@dataclass
class CompletedCommand:
    returncode: int
    stdout: bytes
    stderr: bytes
    timed_out: bool = False


# (argv, stdin bytes, extra env for the docker client, timeout) -> CompletedCommand
Runner = Callable[[Sequence[str], bytes | None, Mapping[str, str] | None, float | None],
                  Awaitable[CompletedCommand]]


async def run_command(argv: Sequence[str], stdin: bytes | None = None,
                      env: Mapping[str, str] | None = None,
                      timeout: float | None = None) -> CompletedCommand:
    """Run a host command without a shell and capture its output."""
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, **env} if env else None,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(stdin), timeout=timeout)
    except TimeoutError:
        proc.kill()
        stdout, stderr = await proc.communicate()
        return CompletedCommand(TIMEOUT_RETURNCODE, stdout or b"", stderr or b"", timed_out=True)
    return CompletedCommand(proc.returncode if proc.returncode is not None else -1,
                            stdout or b"", stderr or b"")


def _check(result: CompletedCommand, what: str) -> None:
    if result.returncode != 0:
        raise RuntimeError(f"docker {what} failed (rc={result.returncode}): "
                           f"{result.stderr.decode(errors='replace').strip()[-2000:]}")


def _tar_dir(local_dir: Path) -> bytes:
    """Tar the contents of local_dir (not the directory itself) into memory."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for child in sorted(local_dir.iterdir()):
            tar.add(child, arcname=child.name)
    return buf.getvalue()


def _tar_single_file(name: str, data: bytes) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo(name)
        info.size = len(data)
        info.mode = 0o644
        info.mtime = int(time.time())
        tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class DockerSandbox:
    """One running container."""

    def __init__(self, container: str, spec: SandboxSpec, *, docker: str = "docker",
                 runner: Runner = run_command):
        self.container = container
        self.spec = spec
        self._docker = docker
        self._run = runner

    async def exec(self, *cmd: str, timeout_s: int | None = None, workdir: str | None = None,
                   env: dict[str, str] | None = None) -> ExecResult:
        if not cmd:
            raise ValueError("exec needs a command")
        argv = [self._docker, "exec", "-w", workdir or self.spec.workdir]
        # Values reach the container through the docker client's environment (`-e NAME`), so
        # they never appear on a command line.
        client_env: dict[str, str] = {}
        for name, value in (env or {}).items():
            argv += ["-e", name]
            client_env[name] = value
        argv.append(self.container)
        command = list(cmd)
        if timeout_s:
            command = ["timeout", "-k", str(_KILL_AFTER_S), str(int(timeout_s)), *command]
        argv += command
        host_timeout = timeout_s + _KILL_AFTER_S + _HOST_TIMEOUT_GRACE_S if timeout_s else None
        started = time.monotonic()
        result = await self._run(argv, None, client_env or None, host_timeout)
        stdout = result.stdout.decode(errors="replace")
        stderr = result.stderr.decode(errors="replace")
        elapsed = time.monotonic() - started
        timed_out = result.timed_out or (
            timeout_s is not None and result.returncode in (124, 137) and elapsed >= timeout_s)
        if timed_out:
            stderr += f"\n[autoenvscaling] command timed out after {timeout_s}s"
            return ExecResult(TIMEOUT_RETURNCODE, stdout, stderr)
        return ExecResult(result.returncode, stdout, stderr)

    async def _mkdir(self, remote_dir: str) -> None:
        result = await self._run([self._docker, "exec", self.container, "mkdir", "-p", remote_dir],
                                 None, None, 120)
        _check(result, f"mkdir {remote_dir}")

    async def write_file(self, remote_path: str, data: bytes) -> None:
        path = PurePosixPath(remote_path)
        if not path.is_absolute():
            raise ValueError(f"remote path must be absolute: {remote_path}")
        await self._mkdir(str(path.parent))
        result = await self._run([self._docker, "cp", "-", f"{self.container}:{path.parent}"],
                                 _tar_single_file(path.name, data), None, 600)
        _check(result, f"cp to {remote_path}")

    async def read_file(self, remote_path: str) -> bytes:
        result = await self._run([self._docker, "cp", f"{self.container}:{remote_path}", "-"],
                                 None, None, 600)
        if result.returncode != 0:
            raise FileNotFoundError(
                f"{remote_path}: {result.stderr.decode(errors='replace').strip()}")
        with tarfile.open(fileobj=io.BytesIO(result.stdout)) as tar:
            member = next((m for m in tar.getmembers() if m.isfile()), None)
            if member is None:
                raise IsADirectoryError(remote_path)
            fh = tar.extractfile(member)
            assert fh is not None
            return fh.read()

    async def upload_dir(self, local_dir: Path, remote_dir: str) -> None:
        local_dir = Path(local_dir)
        if not local_dir.is_dir():
            raise FileNotFoundError(f"no such directory: {local_dir}")
        await self._mkdir(remote_dir)
        if not any(local_dir.iterdir()):
            return
        result = await self._run([self._docker, "cp", "-", f"{self.container}:{remote_dir}"],
                                 _tar_dir(local_dir), None, 1800)
        _check(result, f"cp to {remote_dir}")

    async def download_dir(self, remote_dir: str, local_dir: Path) -> None:
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        # `docker cp C:/dir -` streams a tar whose top-level entry is the directory's basename.
        result = await self._run([self._docker, "cp", f"{self.container}:{remote_dir}", "-"],
                                 None, None, 1800)
        if result.returncode != 0:
            raise FileNotFoundError(
                f"{remote_dir}: {result.stderr.decode(errors='replace').strip()}")
        prefix = PurePosixPath(remote_dir.rstrip("/") or "/").name
        with tarfile.open(fileobj=io.BytesIO(result.stdout)) as tar:
            members = []
            for member in tar.getmembers():
                parts = PurePosixPath(member.name).parts
                if not parts or parts[0] != prefix or len(parts) == 1:
                    continue
                member.name = str(PurePosixPath(*parts[1:]))
                members.append(member)
            tar.extractall(local_dir, members=members, filter="data")

    async def terminate(self) -> None:
        await self._run([self._docker, "rm", "-f", self.container], None, None, 120)


class DockerBackend:
    """Create sandboxes as local Docker containers."""

    name = "docker"

    def __init__(self, *, docker: str = "docker", runner: Runner = run_command,
                 host_env: Mapping[str, str] | None = None, pull: bool = False):
        # Resolve now: an exec's env (which may set PATH for the sandbox) is also the docker
        # client's env, so the client must not be looked up through it.
        self._docker = shutil.which(docker) or docker
        self._run = runner
        self._host_env = os.environ if host_env is None else host_env
        self._pull = pull

    def run_argv(self, spec: SandboxSpec, name: str) -> tuple[list[str], dict[str, str]]:
        """The `docker run` command for a spec, and the env its client process needs."""
        missing = [s for s in spec.secrets if s not in self._host_env]
        if missing:
            raise KeyError(f"sandbox secrets not set in the host environment: {missing}")
        argv = [self._docker, "run", "-d", "--init", "--name", name,
                "--cpus", str(spec.cpu), "--memory", f"{spec.memory_gib}g",
                "-w", spec.workdir, "--entrypoint", "sleep"]
        client_env: dict[str, str] = {}
        for key, value in spec.env.items():
            argv += ["-e", key]
            client_env[key] = value
        for secret in spec.secrets:
            argv += ["-e", secret]
            client_env[secret] = self._host_env[secret]
        for host in spec.blocked_hosts:
            argv += ["--add-host", f"{host}:0.0.0.0"]
        argv += [spec.image, str(int(spec.timeout_s))]
        return argv, client_env

    async def create(self, spec: SandboxSpec) -> DockerSandbox:
        name = spec.name or f"aes-{uuid.uuid4().hex[:12]}"
        argv, client_env = self.run_argv(spec, name)
        if spec.blocked_hosts:
            logger.warning("blocked_hosts are enforced through /etc/hosts only (best effort): %s",
                           ", ".join(spec.blocked_hosts))
        if self._pull:
            _check(await self._run([self._docker, "pull", spec.image], None, None, 3600), "pull")
        result = await self._run(argv, None, client_env, 3600)
        _check(result, "run")
        container = result.stdout.decode().strip() or name
        sandbox = DockerSandbox(container, spec, docker=self._docker, runner=self._run)
        try:
            await sandbox._mkdir(spec.workdir)
        except Exception:
            await sandbox.terminate()
            raise
        return sandbox


__all__ = ["TIMEOUT_RETURNCODE", "CompletedCommand", "DockerBackend", "DockerSandbox", "run_command"]
