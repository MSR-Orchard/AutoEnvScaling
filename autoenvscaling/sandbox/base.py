"""The sandbox interface the proposer runs in.

The proposer works in an isolated container with a workspace, child containers for building
and testing environments, and restricted web access. Backends implement `SandboxBackend`:
`modal` (default, via the Modal SDK) and `docker` (local development and tests).

Validation and calibration do not use this interface directly: they run Harbor trials, and
Harbor has its own environment backends (``harbor run -e modal``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass
class ExecResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@dataclass
class SandboxSpec:
    image: str
    cpu: int = 16
    memory_gib: int = 32
    timeout_s: int = 7200
    workdir: str = "/workspace"
    env: dict[str, str] = field(default_factory=dict)
    # Names of secrets to inject (for Modal, names of Modal Secrets; for Docker, host env vars).
    secrets: list[str] = field(default_factory=list)
    # Hostnames the sandbox must not reach (benchmark pages). Enforcement is the backend's job;
    # backends that cannot enforce it must say so rather than silently allow everything.
    blocked_hosts: list[str] = field(default_factory=list)
    # If set, the only domains the sandbox may reach (enforced by backends that support it).
    domain_allowlist: list[str] | None = None
    name: str | None = None


@runtime_checkable
class Sandbox(Protocol):
    async def exec(self, *cmd: str, timeout_s: int | None = None, workdir: str | None = None,
                   env: dict[str, str] | None = None) -> ExecResult: ...

    async def write_file(self, remote_path: str, data: bytes) -> None: ...

    async def read_file(self, remote_path: str) -> bytes: ...

    async def upload_dir(self, local_dir: Path, remote_dir: str) -> None: ...

    async def download_dir(self, remote_dir: str, local_dir: Path) -> None: ...

    async def terminate(self) -> None: ...


@runtime_checkable
class SandboxBackend(Protocol):
    name: str

    async def create(self, spec: SandboxSpec) -> Sandbox: ...


def get_backend(name: str) -> SandboxBackend:
    if name == "modal":
        from autoenvscaling.sandbox.modal_backend import ModalBackend
        return ModalBackend()
    if name == "docker":
        from autoenvscaling.sandbox.docker_backend import DockerBackend
        return DockerBackend()
    raise ValueError(f"unknown sandbox backend {name!r} (expected 'modal' or 'docker')")
