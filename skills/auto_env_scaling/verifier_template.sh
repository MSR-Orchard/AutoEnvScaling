#!/bin/bash
# AutoEnvScaling fraction-scoring verifier template.
# reward = tests passed / tests total, counted from pytest's individual JUnit test cases.
# Install test-only dependencies in tests/Dockerfile, with a separate verifier, for example:
#   RUN pip install --no-cache-dir pytest==8.4.1
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
pytest_status=0
# pytest-json-ctrf 0.3.5 merges parameterized cases as retries, changing partial credit.
# Use pytest's built-in JUnit writer and derive both reward and CTRF from the same cases.
python3 -I -B -m pytest -c /dev/null --rootdir="$TESTS_DIR" --confcutdir="$TESTS_DIR" \
  test_outputs.py -q --tb=short --junitxml="$LOG_DIR/junit.xml" || pytest_status=$?
# Collection, configuration and internal errors are verifier failures, not scoreable trials.
if (( pytest_status > 1 )); then
  echo "verifier failed with pytest status $pytest_status; no reward written" >&2
  exit "$pytest_status"
fi
python3 -I -B - "$LOG_DIR" "$pytest_status" <<'PY'
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

logs = Path(sys.argv[1])
try:
    cases = ET.parse(logs / 'junit.xml').getroot().findall('.//testcase')
except (OSError, ET.ParseError) as error:
    raise SystemExit(f'verifier produced no valid report: {error}')
tests = []
for case in cases:
    status = ('failed' if any(case.find(tag) is not None for tag in ('failure', 'error'))
              else 'skipped' if case.find('skipped') is not None else 'passed')
    tests.append({'name': case.get('classname', '') + '::' + case.get('name', ''),
                  'status': status, 'duration': float(case.get('time', '0')) * 1000})
total = len(tests)
passed = sum(t['status'] == 'passed' for t in tests)
if type(total) is not int or type(passed) is not int or total <= 0 or not 0 <= passed <= total:
    raise SystemExit('verifier report has invalid test counts; no reward written')
if int(sys.argv[2]) == 1 and passed == total:
    raise SystemExit('verifier report contradicts pytest failure; no reward written')
summary = {'tests': total, 'passed': passed,
           'failed': sum(t['status'] == 'failed' for t in tests),
           'skipped': sum(t['status'] == 'skipped' for t in tests), 'pending': 0, 'other': 0}
report = {'results': {'tool': {'name': 'pytest'}, 'summary': summary, 'tests': tests}}
fd = os.open(logs / 'ctrf.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
with os.fdopen(fd, 'w') as stream:
    json.dump(report, stream)
fraction = passed / total
fd = os.open(logs / 'reward.txt', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
with os.fdopen(fd, 'w') as stream:
    stream.write(f'{fraction:.4f}\n')
print("reward:", fraction)
PY
