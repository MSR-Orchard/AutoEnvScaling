#!/bin/bash
# AutoEnvScaling fraction-scoring verifier template.
# reward = tests passed / tests total, from pytest's CTRF report (or its JUnit XML as a fallback).
# Install test-only dependencies in tests/Dockerfile, with a separate verifier, for example:
#   RUN pip install --no-cache-dir pytest==8.4.1 pytest-json-ctrf==0.3.5
# Keep pytest and its reports trusted. Tests which execute agent-written code must call it
# through run_agent.py, not import it or run it as root. Data-only checks need no subprocess.
set -euo pipefail
TESTS_DIR="${AES_VERIFIER_TESTS_DIR:-/tests}"
LOG_DIR="${AES_VERIFIER_LOG_DIR:-/logs/verifier}"
[[ "$TESTS_DIR" = /* && "$LOG_DIR" = /* ]] || { echo 'verifier paths must be absolute' >&2; exit 2; }
[[ ! -L "$TESTS_DIR" && ! -L "$LOG_DIR" ]] || { echo 'verifier paths must not be symlinks' >&2; exit 2; }
mkdir -p "$LOG_DIR"
chmod 700 "$TESTS_DIR" "$LOG_DIR"
rm -f "$LOG_DIR/reward.txt" "$LOG_DIR/ctrf.json" "$LOG_DIR/junit.xml"
cd "$TESTS_DIR"
unset PYTEST_ADDOPTS PYTEST_PLUGINS PYTHONPATH PYTHONHOME
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
ctrf_plugin="$(python3 -I - <<'PY'
from importlib.metadata import PackageNotFoundError, distribution
try:
    plugins = distribution('pytest-json-ctrf').entry_points
except PackageNotFoundError:
    pass
else:
    print(next(p.name for p in plugins if p.group == 'pytest11'))
PY
)"
pytest_status=0
pytest_args=(-c /dev/null --rootdir="$TESTS_DIR" --confcutdir="$TESTS_DIR" test_outputs.py -q --tb=short)
if [[ -n "$ctrf_plugin" ]]; then
  python3 -I -m pytest -p "$ctrf_plugin" "${pytest_args[@]}" --ctrf "$LOG_DIR/ctrf.json" || pytest_status=$?
else
  # Without the CTRF plugin, JUnit XML gives the same fraction with any modern pytest.
  python3 -I -m pytest "${pytest_args[@]}" --junitxml="$LOG_DIR/junit.xml" || pytest_status=$?
fi
# Collection, configuration and internal errors are verifier failures, not scoreable trials.
if (( pytest_status > 1 )); then
  echo "verifier failed with pytest status $pytest_status; no reward written" >&2
  exit "$pytest_status"
fi
python3 -I - "$LOG_DIR" "$pytest_status" <<'PY'
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

logs = Path(sys.argv[1])
try:
    summary = json.loads((logs / 'ctrf.json').read_text())["results"]["summary"]
    total, passed = summary["tests"], summary["passed"]
except (OSError, ValueError, KeyError, TypeError):
    try:
        cases = ET.parse(logs / 'junit.xml').getroot().findall('.//testcase')
        total = len(cases)
        passed = sum(not any(c.find(tag) is not None for tag in ('failure', 'error', 'skipped'))
                     for c in cases)
    except (OSError, ET.ParseError) as error:
        raise SystemExit(f'verifier produced no valid report: {error}')
if type(total) is not int or type(passed) is not int or total <= 0 or not 0 <= passed <= total:
    raise SystemExit('verifier report has invalid test counts; no reward written')
if int(sys.argv[2]) == 1 and passed == total:
    raise SystemExit('verifier report contradicts pytest failure; no reward written')
fraction = passed / total
fd = os.open(logs / 'reward.txt', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
with os.fdopen(fd, 'w') as stream:
    stream.write(f'{fraction:.4f}\n')
print("reward:", fraction)
PY
