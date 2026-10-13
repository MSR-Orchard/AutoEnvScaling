"""Modal backend against a stand-in for the modal SDK that runs commands on this machine."""

import asyncio
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from autoenvscaling.sandbox import modal_backend
from autoenvscaling.sandbox.base import Sandbox, SandboxSpec, get_backend
from autoenvscaling.sandbox.modal_backend import (
    TIMEOUT_RETURNCODE,
    ModalBackend,
    blocked_hosts_script,
)

live = pytest.mark.skipif(os.environ.get("AES_LIVE_TESTS") != "1",
                          reason="needs Modal credentials; set AES_LIVE_TESTS=1")


class Aio:
    """Mimics synchronicity's `.aio` attribute on Modal callables."""

    def __init__(self, fn):
        self.aio = fn


class FakeStream:
    def __init__(self, data):
        self.read = Aio(self._read)
        self._data = data

    async def _read(self):
        return self._data


class FakeProcess:
    def __init__(self, returncode, stdout, stderr):
        self.stdout, self.stderr = FakeStream(stdout), FakeStream(stderr)
        self.wait = Aio(self._wait)
        self._rc = returncode

    async def _wait(self):
        return self._rc


class FakeSandbox:
    """Executes commands locally, the way a sandbox would, and records what it was asked."""

    def __init__(self, hosts_file: Path):
        self.execs, self.terminated = [], False
        self.hosts_file = hosts_file
        self.exec = Aio(self._exec)
        self.terminate = Aio(self._terminate)
        self.filesystem = SimpleNamespace(
            write_bytes=Aio(self._write_bytes), read_bytes=Aio(self._read_bytes),
            copy_from_local=Aio(self._copy_from_local), copy_to_local=Aio(self._copy_to_local))

    async def _exec(self, *cmd, timeout=None, workdir=None, env=None, text=True):
        self.execs.append({"cmd": cmd, "timeout": timeout, "workdir": workdir, "env": env})
        cmd = [c.replace("/etc/hosts", str(self.hosts_file)) for c in cmd]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,  # noqa: ASYNC221
                                  check=False,
                                  cwd=workdir if workdir and os.path.isdir(workdir) else None,
                                  env={**os.environ, **(env or {})})
        except subprocess.TimeoutExpired:
            return FakeProcess(-1, "", "")
        return FakeProcess(proc.returncode, proc.stdout, proc.stderr)

    async def _terminate(self):
        self.terminated = True

    async def _write_bytes(self, data, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(data)

    async def _read_bytes(self, path):
        return Path(path).read_bytes()

    async def _copy_from_local(self, local, remote):
        shutil.copyfile(local, remote)

    async def _copy_to_local(self, remote, local):
        shutil.copyfile(remote, local)


class FakeModal:
    def __init__(self, hosts_file: Path):
        self.created, self.sandboxes, self.hosts_file = [], [], hosts_file
        self.App = SimpleNamespace(lookup=Aio(self._lookup))
        self.Image = SimpleNamespace(
            from_registry=lambda tag, secret=None: ("image", tag, secret))
        self.Secret = SimpleNamespace(from_name=lambda name: ("secret", name))
        self.Sandbox = SimpleNamespace(create=Aio(self._create))

    async def _lookup(self, name, create_if_missing=False, environment_name=None):
        return ("app", name, create_if_missing)

    async def _create(self, **kwargs):
        self.created.append(kwargs)
        sb = FakeSandbox(self.hosts_file)
        self.sandboxes.append(sb)
        return sb


@pytest.fixture
def fake(tmp_path, monkeypatch):
    monkeypatch.setattr(modal_backend, "_REMOTE_STAGING", str(tmp_path / "staging"))
    (tmp_path / "staging").mkdir()
    hosts = tmp_path / "hosts"
    hosts.write_text("127.0.0.1 localhost\n")
    return FakeModal(hosts)


def test_registry_returns_modal_backend():
    assert isinstance(get_backend("modal"), ModalBackend)


async def test_create_maps_spec_to_sandbox_create(fake, tmp_path):
    backend = ModalBackend(modal_module=fake, registry_secret="regcred")
    spec = SandboxSpec(image="ghcr.io/x/proposer:1", cpu=16, memory_gib=32, timeout_s=8100,
                       workdir=str(tmp_path / "ws"), env={"A": "1"}, secrets=["aes-keys"],
                       blocked_hosts=["tbench.ai"], name="ep-1")
    sb = await backend.create(spec)
    assert isinstance(sb, Sandbox)
    kwargs = fake.created[0]
    assert kwargs["app"] == ("app", "autoenvscaling", True)
    assert kwargs["image"] == ("image", "ghcr.io/x/proposer:1", ("secret", "regcred"))
    assert kwargs["cpu"] == 16.0 and kwargs["memory"] == 32 * 1024 and kwargs["timeout"] == 8100
    assert kwargs["secrets"] == [("secret", "aes-keys")] and kwargs["env"] == {"A": "1"}
    assert kwargs["name"] == "ep-1" and "outbound_domain_allowlist" not in kwargs
    # Blocked hosts end up in /etc/hosts (IPv4 and IPv6), idempotently.
    hosts = fake.hosts_file.read_text()
    assert "0.0.0.0 tbench.ai" in hosts and ":: tbench.ai" in hosts
    assert (tmp_path / "ws").is_dir()


def test_domain_allowlist_is_passed_through(fake):
    backend = ModalBackend(modal_module=fake, domain_allowlist=["pypi.org"])
    kwargs = backend.create_kwargs(SandboxSpec(image="img"), app="app")
    assert kwargs["outbound_domain_allowlist"] == ["pypi.org"]


def test_spec_domain_allowlist_overrides_the_backend_default(fake):
    backend = ModalBackend(modal_module=fake, domain_allowlist=["pypi.org"])
    kwargs = backend.create_kwargs(SandboxSpec(image="img", domain_allowlist=["proxy.example"]), app="app")
    assert kwargs["outbound_domain_allowlist"] == ["proxy.example"]
    kwargs = ModalBackend(modal_module=fake).create_kwargs(SandboxSpec(image="img", domain_allowlist=["a.b"]),
                                                            app="app")
    assert kwargs["outbound_domain_allowlist"] == ["a.b"]


def test_blocked_hosts_script_is_idempotent_and_validated(tmp_path):
    hosts = tmp_path / "hosts"
    hosts.write_text("")
    script = blocked_hosts_script(["a.example"]).replace("/etc/hosts", str(hosts))
    for _ in range(2):
        subprocess.run(["sh", "-c", script], check=True)
    assert hosts.read_text().count("0.0.0.0 a.example") == 1
    with pytest.raises(ValueError):
        blocked_hosts_script(["bad host"])


async def test_exec_file_and_dir_roundtrip(fake, tmp_path):
    sb = await ModalBackend(modal_module=fake).create(
        SandboxSpec(image="img", workdir=str(tmp_path / "ws")))
    result = await sb.exec("sh", "-c", "echo out; echo err >&2; exit 3", env={"X": "1"})
    assert (result.returncode, result.stdout.strip(), result.stderr.strip()) == (3, "out", "err")

    remote = tmp_path / "remote"
    await sb.write_file(str(remote / "f.bin"), b"\x00\x01")
    assert await sb.read_file(str(remote / "f.bin")) == b"\x00\x01"

    src = tmp_path / "src"
    (src / "nested").mkdir(parents=True)
    (src / "nested" / "a.txt").write_text("a")
    (src / "top.txt").write_text("t")
    await sb.upload_dir(src, str(remote / "up"))
    assert (remote / "up" / "nested" / "a.txt").read_text() == "a"
    await sb.download_dir(str(remote / "up"), tmp_path / "down")
    assert (tmp_path / "down" / "top.txt").read_text() == "t"
    assert not list((tmp_path / "staging").iterdir())  # staging tarballs cleaned up

    with pytest.raises(RuntimeError):
        await sb.download_dir(str(remote / "missing"), tmp_path / "down2")

    await sb.terminate()
    assert fake.sandboxes[0].terminated


async def test_exec_timeout_maps_to_124(fake, tmp_path):
    sb = await ModalBackend(modal_module=fake).create(
        SandboxSpec(image="img", workdir=str(tmp_path)))
    result = await sb.exec("sleep", "5", timeout_s=1)
    assert result.returncode == TIMEOUT_RETURNCODE and "timed out" in result.stderr


async def test_failed_setup_terminates_the_sandbox(fake, tmp_path, monkeypatch):
    monkeypatch.setattr(modal_backend, "blocked_hosts_script", lambda hosts: "exit 1")
    with pytest.raises(RuntimeError, match="block hosts"):
        await ModalBackend(modal_module=fake).create(
            SandboxSpec(image="img", workdir=str(tmp_path), blocked_hosts=["x.example"]))
    assert fake.sandboxes[0].terminated


@pytest.mark.live
@live
async def test_live_modal_roundtrip(tmp_path):
    backend = ModalBackend(app_name=os.environ.get("AES_TEST_MODAL_APP", "autoenvscaling-tests"))
    sb = await backend.create(SandboxSpec(image="python:3.12-slim", cpu=1, memory_gib=1,
                                          timeout_s=600, blocked_hosts=["tbench.ai"]))
    try:
        assert (await sb.exec("python3", "-c", "print(1 + 1)")).stdout.strip() == "2"
        assert "tbench.ai" in (await sb.read_file("/etc/hosts")).decode()
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.txt").write_text("a")
        await sb.upload_dir(src, "/workspace/up")
        await sb.download_dir("/workspace/up", tmp_path / "down")
        assert (tmp_path / "down" / "a.txt").read_text() == "a"
        assert (await sb.exec("sleep", "30", timeout_s=2)).returncode == TIMEOUT_RETURNCODE
    finally:
        await sb.terminate()
        await asyncio.sleep(0)
