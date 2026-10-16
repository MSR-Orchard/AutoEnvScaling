#!/usr/bin/env python3
"""Run an agent-produced program from a trusted Linux verifier, returning a JSON result.

Run this helper as root, in the verifier container. Each invocation reserves a fresh numeric
UID, removes capabilities, enables no-new-privileges, bounds file output and process count,
and reaps descendants, including children which detach into a new session. This is privilege
separation inside the verifier, not a replacement for its container or network policy.

The caller must keep all expected answers and trusted controller files in protected paths.
Never run this helper from an agent-controlled copy or import agent modules into root pytest.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import pwd
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PR_SET_CHILD_SUBREAPER = 36
PR_SET_DUMPABLE = 4
DEFAULT_PROTECTED = ("/tests", "/logs/verifier")


def protect_directory(path: Path, *, mode: int = 0o700) -> None:
    """Only operate on root-owned directories; refuse symlinks and other file types."""
    path.mkdir(parents=True, exist_ok=True, mode=mode)
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if info.st_uid != 0:
            raise PermissionError(f"trusted directory must be root-owned: {path}")
        os.fchmod(fd, mode)
    finally:
        os.close(fd)


def uid_processes(uid: int) -> list[int]:
    """Find this invocation's processes, including zombies adopted by the subreaper."""
    found = []
    for path in Path("/proc").glob("[0-9]*/status"):
        try:
            line = next(line for line in path.read_text().splitlines() if line.startswith("Uid:"))
            if uid in map(int, line.split()[1:]):
                found.append(int(path.parent.name))
        except (OSError, StopIteration, ValueError):
            continue
    return found


def reserve_uid(runtime: Path) -> tuple[int, Path]:
    """A root-owned reservation prevents concurrent invocations from sharing an identity."""
    for _ in range(100):
        # Stay inside the range supported by common container UID maps and shared filesystems.
        uid = 60_000 + secrets.randbelow(5_000)
        try:
            pwd.getpwuid(uid)
        except KeyError:
            pass
        else:
            continue
        reservation = runtime / f"uid-{uid}"
        try:
            reservation.mkdir(mode=0o711)
        except FileExistsError:
            continue
        if uid_processes(uid):
            reservation.rmdir()
            continue
        return uid, reservation
    raise RuntimeError("could not reserve an unused verifier identity")


def reap(uid: int, *, timeout: float = 3.0) -> None:
    """Kill and reap all processes for the reserved UID, including double-forked daemons."""
    deadline = time.monotonic() + timeout
    while True:
        pids = uid_processes(uid)
        if not pids:
            return
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for pid in pids:
            try:
                os.waitpid(pid, os.WNOHANG)
            except (ChildProcessError, ProcessLookupError):
                pass
        if time.monotonic() >= deadline:
            raise RuntimeError("agent descendants remain after cleanup; do not score this run")
        time.sleep(0.01)


def run(args: argparse.Namespace) -> dict:
    if sys.platform != "linux" or os.geteuid() != 0:
        raise RuntimeError("run_agent.py requires a root-run Linux verifier")
    setpriv = shutil.which("setpriv", path="/usr/sbin:/usr/bin:/sbin:/bin")
    prlimit = shutil.which("prlimit", path="/usr/sbin:/usr/bin:/sbin:/bin")
    if setpriv is None or prlimit is None:
        raise RuntimeError("install util-linux in tests/Dockerfile (setpriv/prlimit are required)")
    if not args.command:
        raise ValueError("provide a program after --")
    if args.timeout <= 0 or args.output_limit <= 0 or args.file_limit <= 0:
        raise ValueError("timeout and output/file limits must be positive")
    if args.process_limit < 1:
        raise ValueError("process limit must be positive")

    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "cannot enable child subreaper")
    if libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "cannot protect verifier process")
    protected = [Path(path).absolute() for path in args.protect or DEFAULT_PROTECTED]
    runtime = Path(args.runtime_dir).absolute()
    if any(runtime.resolve().is_relative_to(path.resolve()) for path in protected):
        raise ValueError("runtime directory must be outside protected grading directories")
    for path in protected:
        protect_directory(Path(path))
    protect_directory(runtime, mode=0o711)
    uid, reservation = reserve_uid(runtime)
    process = None
    clean = False
    try:
        private = reservation / "private"
        private.mkdir(mode=0o700)
        work = reservation / "work"
        work.mkdir(mode=0o700)
        os.chown(work, uid, uid)
        env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8",
               "HOME": str(work), "TMPDIR": str(work), "PYTHONDONTWRITEBYTECODE": "1"}
        # Configuration is the trusted verifier's CLI, not inherited agent environment variables.
        env.update(dict(value.split("=", 1) for value in args.env))
        command = [prlimit, "--core=0:0", f"--fsize={args.file_limit}:{args.file_limit}",
                   f"--nproc={args.process_limit}:{args.process_limit}", "--",
                   setpriv, "--reuid", str(uid), "--regid", str(uid), "--clear-groups",
                   "--bounding-set=-all", "--inh-caps=-all", "--ambient-caps=-all",
                   "--no-new-privs", "--", *args.command]
        paths = [private / name for name in ("stdout", "stderr")]
        with paths[0].open("xb", buffering=0) as stdout, paths[1].open("xb", buffering=0) as stderr:
            process = subprocess.Popen(
                command, cwd=args.cwd or work, env=env, stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr, start_new_session=True, close_fds=True)
            timed_out = False
            try:
                process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=3)
                reap(uid)
                clean = True
        output = []
        truncated = False
        for path in paths:
            with path.open("rb") as stream:
                data = stream.read(args.output_limit + 1)
            truncated |= len(data) > args.output_limit
            output.append(data[:args.output_limit].decode("utf-8", errors="replace"))
        return {"returncode": process.returncode, "timed_out": timed_out,
                "stdout": output[0], "stderr": output[1], "output_truncated": truncated,
                "uid": uid}
    finally:
        if not clean:
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=3)
            reap(uid)
        # A cleanup failure deliberately leaves the reservation in place, preventing UID reuse.
        shutil.rmtree(reservation)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwd", type=Path, help="agent-readable working directory; default: scratch")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--output-limit", type=int, default=1_048_576,
                        help="maximum bytes returned per output stream")
    parser.add_argument("--file-limit", type=int, default=67_108_864,
                        help="hard byte limit per file written by agent code, including output")
    parser.add_argument("--process-limit", type=int, default=64)
    parser.add_argument("--runtime-dir", type=Path,
                        default=Path(tempfile.gettempdir()) / "aes-verifier")
    parser.add_argument("--protect", action="append",
                        help="root-only directory; repeat to replace defaults /tests,/logs/verifier")
    parser.add_argument("--env", action="append", default=[], metavar="NAME=VALUE",
                        help="explicit trusted environment override; no host credentials inherited")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    def interrupted(signum, _frame):
        raise InterruptedError(f"verifier execution interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        result = run(args)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"run_agent: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
