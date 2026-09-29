"""Modal sandbox backend (the default).

Each sandbox is a ``modal.Sandbox`` created from a registry image. The mapping from the sandbox
interface to the public Modal SDK (tested against modal 1.6):

    create        modal.App.lookup(app_name, create_if_missing=True)
                  modal.Image.from_registry(spec.image[, secret=registry secret])
                  modal.Sandbox.create(app=, image=, cpu=, memory=MiB, timeout=, workdir=,
                                       env=, secrets=[modal.Secret.from_name(n) ...], name=)
    exec          sandbox.exec(*cmd, timeout=, workdir=, env=), then read stdout/stderr and wait()
    write_file    sandbox.filesystem.write_bytes(data, path)
    read_file     sandbox.filesystem.read_bytes(path)
    upload_dir    local tar.gz -> sandbox.filesystem.copy_from_local -> `tar xzf` in the sandbox
    download_dir  `tar czf` in the sandbox -> sandbox.filesystem.copy_to_local -> local extract
    terminate     sandbox.terminate()

The research code used a private Modal-like client whose ``create_sandbox(image, cpu, memory,
timeout, block_network)``, ``exec``, ``upload_file``, ``download_file`` and ``delete`` calls map
one-to-one onto the calls above. Its heartbeat and checkpoint/roll helpers existed only to work
around that service's hard two-hour TTL; Modal's ``timeout`` sets the sandbox lifetime directly
(up to 24 hours), so they are not ported.

Network restrictions. Modal can block all egress (``block_network``) or allow only listed
domains (``outbound_domain_allowlist``), but it cannot block individual hostnames. The proposer
needs open web access with a few benchmark sites blocked, so ``SandboxSpec.blocked_hosts`` is
enforced best-effort: at startup each host is mapped to 0.0.0.0 and :: in ``/etc/hosts``. This
stops ordinary tools (curl, pip, browsers, our web tools) from resolving those names, but a
process with root inside the sandbox can edit ``/etc/hosts``, use its own resolver, or connect by
IP address. Treat it as a guard against accidental access, not as isolation. For strict egress
control, pass ``domain_allowlist`` to ``ModalBackend``, which switches the sandbox to Modal's
enforced allowlist (all other domains are then unreachable).

Timeouts. Modal reports an exec that exceeded its timeout with return code -1; this backend maps
that to ``TIMEOUT_RETURNCODE`` (124, as GNU ``timeout`` does).
"""

from __future__ import annotations

import asyncio
import logging
import shlex
import tarfile
import tempfile
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from autoenvscaling.sandbox.base import ExecResult, SandboxSpec

logger = logging.getLogger(__name__)

TIMEOUT_RETURNCODE = 124
DEFAULT_APP_NAME = "autoenvscaling"
# Staging area inside the sandbox for directory transfers.
_REMOTE_STAGING = "/tmp"
_TRANSFER_TIMEOUT_S = 1800


def blocked_hosts_script(hosts: Sequence[str]) -> str:
    """Shell script that maps each host to 0.0.0.0 and :: in /etc/hosts (idempotent)."""
    lines = []
    for host in hosts:
        if not host or any(c.isspace() for c in host) or "/" in host:
            raise ValueError(f"invalid blocked host {host!r}")
        for addr in ("0.0.0.0", "::"):
            entry = f"{addr} {host}"
            lines.append(f"grep -qxF {shlex.quote(entry)} /etc/hosts || "
                         f"printf '%s\\n' {shlex.quote(entry)} >> /etc/hosts")
    return "set -e\n" + "\n".join(lines) + "\n"


class ModalSandbox:
    """A running ``modal.Sandbox`` behind the ``Sandbox`` protocol."""

    def __init__(self, sandbox: Any, spec: SandboxSpec):
        self._sb = sandbox
        self.spec = spec

    @property
    def object_id(self) -> str | None:
        return getattr(self._sb, "object_id", None)

    async def exec(self, *cmd: str, timeout_s: int | None = None, workdir: str | None = None,
                   env: dict[str, str] | None = None) -> ExecResult:
        if not cmd:
            raise ValueError("exec needs a command")
        proc = await self._sb.exec.aio(*cmd, timeout=timeout_s, workdir=workdir or self.spec.workdir,
                                       env=env or None, text=True)
        stdout, stderr = await asyncio.gather(proc.stdout.read.aio(), proc.stderr.read.aio())
        returncode = await proc.wait.aio()
        if returncode == -1:
            # Modal's convention for an exec that ran past its timeout.
            stderr = (stderr or "") + f"\n[autoenvscaling] command timed out after {timeout_s}s"
            returncode = TIMEOUT_RETURNCODE
        return ExecResult(returncode, stdout or "", stderr or "")

    async def _check(self, script: str, what: str) -> None:
        result = await self.exec("sh", "-c", script, timeout_s=_TRANSFER_TIMEOUT_S, workdir="/")
        if not result.ok:
            raise RuntimeError(f"{what} failed (rc={result.returncode}): "
                               f"{(result.stderr or result.stdout).strip()[-2000:]}")

    async def write_file(self, remote_path: str, data: bytes) -> None:
        await self._sb.filesystem.write_bytes.aio(data, remote_path)

    async def read_file(self, remote_path: str) -> bytes:
        return await self._sb.filesystem.read_bytes.aio(remote_path)

    async def upload_dir(self, local_dir: Path, remote_dir: str) -> None:
        local_dir = Path(local_dir)
        if not local_dir.is_dir():
            raise FileNotFoundError(f"no such directory: {local_dir}")
        await self._check(f"mkdir -p {shlex.quote(remote_dir)}", f"mkdir {remote_dir}")
        if not any(local_dir.iterdir()):
            return
        remote_tar = f"{_REMOTE_STAGING}/aes-up-{uuid.uuid4().hex}.tar.gz"
        with tempfile.TemporaryDirectory() as tmp:
            local_tar = Path(tmp) / "up.tar.gz"
            with tarfile.open(local_tar, "w:gz") as tar:
                # Add the children, not the directory itself, so extraction never tries to reset
                # the target directory's own owner or mode.
                for child in sorted(local_dir.iterdir()):
                    tar.add(child, arcname=child.name)
            await self._sb.filesystem.copy_from_local.aio(str(local_tar), remote_tar)
        await self._check(
            f"tar xzf {shlex.quote(remote_tar)} -C {shlex.quote(remote_dir)}; rc=$?; "
            f"rm -f {shlex.quote(remote_tar)}; exit $rc",
            f"extract upload into {remote_dir}")

    async def download_dir(self, remote_dir: str, local_dir: Path) -> None:
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        remote_tar = f"{_REMOTE_STAGING}/aes-down-{uuid.uuid4().hex}.tar.gz"
        await self._check(
            f"test -d {shlex.quote(remote_dir)} || {{ echo 'no such directory: {remote_dir}' >&2; "
            f"exit 2; }}; tar czf {shlex.quote(remote_tar)} -C {shlex.quote(remote_dir)} .",
            f"archive {remote_dir}")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                local_tar = Path(tmp) / "down.tar.gz"
                await self._sb.filesystem.copy_to_local.aio(remote_tar, str(local_tar))
                with tarfile.open(local_tar, "r:gz") as tar:
                    tar.extractall(local_dir, filter="data")
        finally:
            try:
                await self.exec("rm", "-f", remote_tar, timeout_s=60, workdir="/")
            except Exception as exc:  # noqa: BLE001  cleanup only; the sandbox is torn down anyway
                logger.debug("could not remove %s: %s", remote_tar, exc)

    async def terminate(self) -> None:
        await self._sb.terminate.aio()


class ModalBackend:
    """Create sandboxes with the Modal SDK. Credentials come from ``modal token new``."""

    name = "modal"

    def __init__(self, *, app_name: str = DEFAULT_APP_NAME, environment_name: str | None = None,
                 registry_secret: str | None = None, domain_allowlist: Sequence[str] | None = None,
                 modal_module: Any = None):
        """
        Args:
            app_name: Modal app the sandboxes are grouped under (created if missing).
            environment_name: Modal environment; defaults to the profile's.
            registry_secret: Name of a Modal Secret with REGISTRY_USERNAME/REGISTRY_PASSWORD for
                pulling a private image.
            domain_allowlist: If given, the sandbox may reach only these domains (enforced by
                Modal). Leave unset for open egress with best-effort ``blocked_hosts``.
            modal_module: The ``modal`` module, or a stand-in for tests.
        """
        self._app_name = app_name
        self._environment_name = environment_name
        self._registry_secret = registry_secret
        self._domain_allowlist = list(domain_allowlist) if domain_allowlist else None
        self._modal = modal_module
        self._app: Any = None

    def _sdk(self) -> Any:
        if self._modal is None:
            import modal
            self._modal = modal
        return self._modal

    async def _get_app(self) -> Any:
        if self._app is None:
            modal = self._sdk()
            kwargs: dict[str, Any] = {"create_if_missing": True}
            if self._environment_name:
                kwargs["environment_name"] = self._environment_name
            self._app = await modal.App.lookup.aio(self._app_name, **kwargs)
        return self._app

    def create_kwargs(self, spec: SandboxSpec, app: Any) -> dict[str, Any]:
        """Keyword arguments for ``modal.Sandbox.create`` (without a command, the sandbox idles
        until its timeout, and all work happens through exec)."""
        modal = self._sdk()
        image_kwargs: dict[str, Any] = {}
        if self._registry_secret:
            image_kwargs["secret"] = modal.Secret.from_name(self._registry_secret)
        kwargs: dict[str, Any] = {
            "app": app,
            "image": modal.Image.from_registry(spec.image, **image_kwargs),
            "cpu": float(spec.cpu),
            "memory": int(spec.memory_gib * 1024),
            "timeout": int(spec.timeout_s),
            "workdir": spec.workdir,
            "secrets": [modal.Secret.from_name(name) for name in spec.secrets],
        }
        if spec.env:
            kwargs["env"] = dict(spec.env)
        if spec.name:
            kwargs["name"] = spec.name
        if self._environment_name:
            kwargs["environment_name"] = self._environment_name
        if self._domain_allowlist:
            kwargs["outbound_domain_allowlist"] = list(self._domain_allowlist)
        return kwargs

    async def create(self, spec: SandboxSpec) -> ModalSandbox:
        modal = self._sdk()
        app = await self._get_app()
        sandbox = ModalSandbox(await modal.Sandbox.create.aio(**self.create_kwargs(spec, app)), spec)
        try:
            await sandbox._check(f"mkdir -p {shlex.quote(spec.workdir)}", "create workdir")
            if spec.blocked_hosts:
                logger.warning(
                    "Modal cannot block hostnames; mapping %s to 0.0.0.0 in /etc/hosts "
                    "(best effort, not enforced isolation)", ", ".join(spec.blocked_hosts))
                await sandbox._check(blocked_hosts_script(spec.blocked_hosts), "block hosts")
        except Exception:
            await sandbox.terminate()
            raise
        return sandbox
