"""Live checks against real Harbor sandboxes. They cost money and need credentials.

Run with ``AES_LIVE_TESTS=1 .venv/bin/python -m pytest -q -m live tests/test_validation_live.py``.
The solver test also needs ``AES_SOLVER_BASE_URL`` and ``AES_SOLVER_MODEL``.
"""

import os
from pathlib import Path

import pytest

from autoenvscaling.config import Settings
from autoenvscaling.results import BUILD, EMPTY, REFERENCE
from autoenvscaling.tasks import HarborTask
from autoenvscaling.validation.admission import EXECUTION_CHECKS, validate

pytestmark = [pytest.mark.live,
              pytest.mark.skipif(os.environ.get("AES_LIVE_TESTS") != "1",
                                 reason="live test; set AES_LIVE_TESTS=1")]

TEST_SH = """#!/bin/bash
mkdir -p /logs/verifier
passed=0
[ -f /app/a.txt ] && passed=$((passed + 1))
[ "$(cat /app/b.txt 2>/dev/null)" = "42" ] && passed=$((passed + 1))
python3 -c "print($passed / 2)" > /logs/verifier/reward.txt
"""


def tiny_task(root: Path) -> HarborTask:
    for rel in ("environment", "solution", "tests"):
        (root / rel).mkdir(parents=True, exist_ok=True)
    (root / "task.toml").write_text('schema_version = "1.4"\n\n[verifier]\ntimeout_sec = 120.0\n\n'
                                    '[agent]\ntimeout_sec = 300.0\n')
    (root / "instruction.md").write_text("Create /app/a.txt (any content) and write 42 to "
                                         "/app/b.txt.\n")
    (root / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\nWORKDIR /app\n")
    (root / "solution" / "solve.sh").write_text("#!/bin/bash\ntouch /app/a.txt\n"
                                                "echo 42 > /app/b.txt\n")
    (root / "tests" / "test.sh").write_text(TEST_SH)
    return HarborTask(root)


async def test_live_execution_checks(tmp_path):
    settings = Settings.load()
    settings.work_dir = tmp_path / "work"
    report = await validate(tiny_task(tmp_path / "tiny"), settings, checks=EXECUTION_CHECKS)
    by = {c.name: c for c in report.checks}
    assert by[BUILD].passed, by[BUILD].detail
    assert by[REFERENCE].passed and by[REFERENCE].data["reward"] == 1.0, by[REFERENCE].detail
    assert by[EMPTY].passed and by[EMPTY].data["reward"] == 0.0, by[EMPTY].detail


@pytest.mark.skipif(not os.environ.get("AES_SOLVER_BASE_URL"), reason="no solver endpoint")
async def test_live_solver_attempt(tmp_path):
    from autoenvscaling.calibration.solver import SolverEndpoint, run_solver

    settings = Settings.load()
    trajs = await run_solver(tiny_task(tmp_path / "tiny"), SolverEndpoint.from_env(), n=1,
                             harbor_env=settings.sandbox.harbor_env, jobs_dir=tmp_path / "jobs")
    assert len(trajs) == 1 and trajs[0].reward is not None, trajs[0].verifier_output
