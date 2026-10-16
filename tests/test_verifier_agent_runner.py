"""Exercise real privilege separation and detached-child cleanup; no containers or network."""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

RUNNER = (Path(__file__).resolve().parents[1] / "skills" / "auto_env_scaling" / "run_agent.py")
AGENT_PYTHON = shutil.which("python3", path="/usr/bin:/bin")
pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or os.geteuid() != 0 or not shutil.which("setpriv"),
    reason="requires root Linux with util-linux for real UID/capability separation")


@pytest.fixture
def scratch(tmp_path, tmp_path_factory):
    # pytest creates 0700 parents. Only expose traversal of this test's scratch directories.
    tmp_path_factory.getbasetemp().chmod(0o711)
    tmp_path.chmod(0o711)
    if any(not (parent.stat().st_mode & 0o001) for parent in tmp_path.parents):
        pytest.skip("use a traversable --basetemp for unprivileged execution tests")
    return tmp_path


def invoke(scratch, code, *options, check=True, extra_env=None):
    env = {**os.environ, **(extra_env or {})}
    command = [sys.executable, "-I", str(RUNNER), "--runtime-dir", str(scratch / "runtime"),
               "--protect", str(scratch / "trusted"), *options, "--", AGENT_PYTHON, "-c", code]
    result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=15, check=False)
    if check:
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)
    return result


def test_runs_with_fresh_uid_no_capabilities_and_no_inherited_credentials(scratch):
    code = """
import json,os,pathlib
status = pathlib.Path('/proc/self/status').read_text().splitlines()
print(json.dumps({'uid':os.getuid(), 'gid':os.getgid(), 'groups':os.getgroups(),
                  'secret':os.environ.get('VERIFIER_SECRET'),
                  'status':{line.split(':')[0]:line.split(':')[1].strip() for line in status
                            if line.startswith(('CapEff:', 'CapBnd:', 'NoNewPrivs:'))}}))
pathlib.Path('output.txt').write_text('scratch is writable')
"""
    result = invoke(scratch, code, extra_env={"VERIFIER_SECRET": "must-not-reach-agent"})
    agent = json.loads(result["stdout"])
    assert result["returncode"] == 0 and not result["timed_out"]
    assert agent["uid"] == agent["gid"] == result["uid"] != 0
    assert agent["groups"] == [] and agent["secret"] is None
    assert int(agent["status"]["CapEff"], 16) == int(agent["status"]["CapBnd"], 16) == 0
    assert agent["status"]["NoNewPrivs"] == "1"
    second = invoke(scratch, "print('another identity')")
    assert second["uid"] != result["uid"]
    assert list((scratch / "runtime").iterdir()) == []


@pytest.mark.parametrize("name", ["reward.txt", "ctrf.json", "junit.xml", "expected.json"])
def test_agent_cannot_read_or_forge_grading_files(scratch, name):
    trusted = scratch / "trusted"
    trusted.mkdir()
    report = trusted / name
    report.write_text("root-owned truth")
    code = f"""
from pathlib import Path
p = Path({str(report)!r})
for action in (lambda: p.read_text(), lambda: p.write_text('forged pass')):
    try:
        action()
    except PermissionError:
        print('denied')
    else:
        raise AssertionError('grading file was accessible')
"""
    result = invoke(scratch, code)
    assert result["returncode"] == 0 and result["stdout"] == "denied\ndenied\n"
    assert report.read_text() == "root-owned truth"


def test_reaps_double_forked_child_which_detaches_and_holds_stdout(scratch):
    code = """
import os,time
pid = os.fork()
if pid == 0:
    os.setsid()
    if os.fork():
        os._exit(0)
    print(os.getpid(), flush=True)
    time.sleep(60)
    os._exit(0)
os.waitpid(pid, 0)
time.sleep(.1)
"""
    result = invoke(scratch, code)
    assert result["returncode"] == 0
    grandchild = int(result["stdout"].strip())
    assert not Path(f"/proc/{grandchild}").exists()
    assert list((scratch / "runtime").iterdir()) == []


def test_timeout_kills_agent_and_reports_failure(scratch):
    result = invoke(scratch, "import os,time;print(os.getpid(),flush=True);time.sleep(60)",
                    "--timeout", "0.2")
    assert result["timed_out"] and result["returncode"] != 0
    assert not Path(f"/proc/{int(result['stdout'].strip())}").exists()


def test_interrupting_helper_also_cleans_up_agent(scratch):
    command = [sys.executable, "-I", str(RUNNER), "--runtime-dir", str(scratch / "runtime"),
               "--protect", str(scratch / "trusted"), "--", AGENT_PYTHON, "-c",
               "import os,time;print(os.getpid(),flush=True);time.sleep(60)"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 5
        while True:
            output = list((scratch / "runtime").glob("uid-*/private/stdout"))
            if output and output[0].read_text().strip():
                pid = int(output[0].read_text().strip())
                break
            assert time.monotonic() < deadline, "agent failed to start"
            time.sleep(.02)
        process.terminate()
        process.communicate(timeout=10)
        assert process.returncode == 2
        assert not Path(f"/proc/{pid}").exists()
        assert list((scratch / "runtime").iterdir()) == []
    finally:
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=10)


def test_output_and_file_writes_are_bounded(scratch):
    result = invoke(scratch, "import os;os.write(1,b'x'*4096)",
                    "--output-limit", "32", "--file-limit", "128")
    assert result["stdout"] == "x" * 32 and result["output_truncated"]


def test_refuses_symlinked_trusted_directory(scratch):
    elsewhere = scratch / "elsewhere"
    elsewhere.mkdir(mode=0o755)
    (scratch / "trusted").symlink_to(elsewhere, target_is_directory=True)
    result = invoke(scratch, "raise AssertionError('must not run')", check=False)
    assert result.returncode == 2
    assert (elsewhere.stat().st_mode & 0o777) == 0o755


def test_runtime_cannot_reopen_a_protected_directory(scratch):
    result = invoke(scratch, "raise AssertionError('must not run')", "--runtime-dir",
                    str(scratch / "trusted" / "runtime"), check=False)
    assert result.returncode == 2 and "outside protected" in result.stderr
