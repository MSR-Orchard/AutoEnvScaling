"""Docker backend: command construction with a fake docker CLI, plus opt-in live tests."""

import io
import os
import tarfile
from pathlib import Path

import pytest

from autoenvscaling.sandbox.base import SandboxBackend, SandboxSpec, get_backend
from autoenvscaling.sandbox.docker_backend import (
    TIMEOUT_RETURNCODE,
    CompletedCommand,
    DockerBackend,
    DockerSandbox,
)

live = pytest.mark.skipif(os.environ.get("AES_LIVE_TESTS") != "1",
                          reason="needs a Docker daemon; set AES_LIVE_TESTS=1")


class FakeDocker:
    """Records docker invocations and answers with canned results."""

    def __init__(self):
        self.calls = []
        self.answers = []  # (predicate, CompletedCommand)

    def answer(self, predicate, result):
        self.answers.append((predicate, result))

    async def __call__(self, argv, stdin=None, env=None, timeout=None):
        self.calls.append({"argv": list(argv), "stdin": stdin, "env": dict(env or {}),
                           "timeout": timeout})
        for predicate, result in self.answers:
            if predicate(list(argv)):
                return result
        return CompletedCommand(0, b"", b"")


def _tar(members: dict[str, bytes], dirs=()) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for d in dirs:
            info = tarfile.TarInfo(d)
            info.type = tarfile.DIRTYPE
            info.mode = 0o755
            tar.addfile(info)
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_backend_registry_and_protocol():
    backend = get_backend("docker")
    assert isinstance(backend, DockerBackend) and isinstance(backend, SandboxBackend)
    with pytest.raises(ValueError):
        get_backend("nope")


def test_run_argv_maps_spec_and_keeps_secret_values_off_the_command_line():
    backend = DockerBackend(host_env={"MY_TOKEN": "s3cret"})
    spec = SandboxSpec(image="img:1", cpu=4, memory_gib=8, timeout_s=600, workdir="/workspace",
                       env={"A": "1"}, secrets=["MY_TOKEN"], blocked_hosts=["tbench.ai"])
    argv, client_env = backend.run_argv(spec, "aes-x")
    joined = " ".join(argv)
    assert "--cpus 4" in joined and "--memory 8g" in joined and "--init" in argv
    assert "--add-host tbench.ai:0.0.0.0" in joined
    assert argv[-2:] == ["img:1", "600"] and "--entrypoint sleep" in joined
    assert "s3cret" not in joined and client_env == {"A": "1", "MY_TOKEN": "s3cret"}


def test_missing_secret_fails_loudly():
    backend = DockerBackend(host_env={})
    with pytest.raises(KeyError, match="MY_TOKEN"):
        backend.run_argv(SandboxSpec(image="img", secrets=["MY_TOKEN"]), "n")


async def test_create_exec_and_timeout_mapping():
    fake = FakeDocker()
    fake.answer(lambda a: a[1] == "run", CompletedCommand(0, b"cid123\n", b""))
    fake.answer(lambda a: "slow" in a, CompletedCommand(124, b"partial", b"", timed_out=False))
    backend = DockerBackend(docker="docker-not-installed", runner=fake, host_env={})
    sb = await backend.create(SandboxSpec(image="img", timeout_s=100))
    assert sb.container == "cid123"

    result = await sb.exec("echo", "hi", env={"KEY": "v"}, workdir="/tmp")
    call = fake.calls[-1]
    assert call["argv"][:6] == ["docker-not-installed", "exec", "-w", "/tmp", "-e", "KEY"]
    assert call["argv"][-2:] == ["echo", "hi"] and call["env"] == {"KEY": "v"}
    assert result.ok

    # A host-side timeout is reported as TIMEOUT_RETURNCODE.
    fake.answer(lambda a: "hang" in a, CompletedCommand(TIMEOUT_RETURNCODE, b"", b"",
                                                        timed_out=True))
    result = await sb.exec("hang", timeout_s=5)
    assert result.returncode == TIMEOUT_RETURNCODE and "timed out" in result.stderr
    assert fake.calls[-1]["argv"][-5:] == ["timeout", "-k", "10", "5", "hang"]


async def test_file_and_directory_transfer(tmp_path: Path):
    fake = FakeDocker()
    fake.answer(lambda a: a[:2] == ["docker", "cp"] and a[2] == "c:/w/a.txt",
                CompletedCommand(0, _tar({"a.txt": b"hello"}), b""))
    fake.answer(lambda a: a[:2] == ["docker", "cp"] and a[2] == "c:/w/out",
                CompletedCommand(0, _tar({"out/x.txt": b"x", "out/sub/y.txt": b"y"},
                                         dirs=("out", "out/sub")), b""))
    sb = DockerSandbox("c", SandboxSpec(image="img"), runner=fake)

    assert await sb.read_file("/w/a.txt") == b"hello"

    await sb.write_file("/w/new/b.txt", b"data")
    mkdir, cp = fake.calls[-2], fake.calls[-1]
    assert mkdir["argv"][-3:] == ["mkdir", "-p", "/w/new"]
    assert cp["argv"] == ["docker", "cp", "-", "c:/w/new"]
    with tarfile.open(fileobj=io.BytesIO(cp["stdin"])) as tar:
        assert tar.extractfile("b.txt").read() == b"data"

    src = tmp_path / "src"
    (src / "d").mkdir(parents=True)
    (src / "d" / "f.txt").write_text("f")
    await sb.upload_dir(src, "/w/up")
    with tarfile.open(fileobj=io.BytesIO(fake.calls[-1]["stdin"])) as tar:
        assert "d/f.txt" in tar.getnames()

    dest = tmp_path / "dest"
    await sb.download_dir("/w/out", dest)
    assert (dest / "x.txt").read_text() == "x" and (dest / "sub" / "y.txt").read_text() == "y"


async def test_missing_remote_file_raises():
    fake = FakeDocker()
    fake.answer(lambda a: a[:2] == ["docker", "cp"], CompletedCommand(1, b"", b"no such file"))
    sb = DockerSandbox("c", SandboxSpec(image="img"), runner=fake)
    with pytest.raises(FileNotFoundError):
        await sb.read_file("/nope")


@pytest.mark.live
@live
async def test_live_docker_roundtrip(tmp_path: Path):
    backend = DockerBackend()
    sb = await backend.create(SandboxSpec(image=os.environ.get("AES_TEST_IMAGE", "alpine:latest"),
                                          cpu=1, memory_gib=1, timeout_s=300,
                                          blocked_hosts=["tbench.ai"]))
    try:
        result = await sb.exec("sh", "-c", "echo $FOO", env={"FOO": "bar"})
        assert result.stdout.strip() == "bar"
        assert "0.0.0.0\ttbench.ai" in (await sb.read_file("/etc/hosts")).decode().replace(" ", "\t")
        await sb.write_file("/workspace/x/y.txt", b"payload")
        assert await sb.read_file("/workspace/x/y.txt") == b"payload"
        src = tmp_path / "src"
        (src / "a").mkdir(parents=True)
        (src / "a" / "b.txt").write_text("b")
        await sb.upload_dir(src, "/workspace/up")
        await sb.download_dir("/workspace/up", tmp_path / "down")
        assert (tmp_path / "down" / "a" / "b.txt").read_text() == "b"
        slow = await sb.exec("sleep", "30", timeout_s=2)
        assert slow.returncode == TIMEOUT_RETURNCODE
    finally:
        await sb.terminate()
