#!/bin/bash
# Managed verifier: reward = tests passed / tests total, from /tests/test_outputs.py.
#
# Ported from the research code's managed verifier template (v2). A skipped test is never counted
# as passed, and a run that collects no tests scores 0. The image is expected to contain pytest and
# pytest-json-ctrf, installed at build time, because the verifier may run without network.
#
# Hardening kept from v2:
#  1. The CTRF reporter is detected by asking pytest for its --ctrf option, which does not depend
#     on the plugin's module name.
#  2. Reports and rewards left in /logs/verifier before grading are deleted, so nothing the agent
#     planted there is read back. reward.json is removed because Harbor prefers it over reward.txt.
#  3. pytest's config file, rootdir and confcutdir are pinned to /tests, so a pytest.ini, tox.ini,
#     setup.cfg, pyproject.toml or conftest.py placed above /tests cannot change collection.
set -u
TESTS_DIR=/tests
VDIR=/logs/verifier
mkdir -p "$VDIR"

rm -f "$VDIR/ctrf.json" "$VDIR/junit.xml" "$VDIR/reward.json" "$VDIR/reward.txt"

PIN_INI=""
for cand in pytest.ini tox.ini setup.cfg pyproject.toml; do
  if [ -f "$TESTS_DIR/$cand" ]; then PIN_INI="$TESTS_DIR/$cand"; break; fi
done
if [ -z "$PIN_INI" ]; then
  PIN_INI="$TESTS_DIR/.managed-pytest.ini"
  printf '[pytest]\n' > "$PIN_INI"
fi
PIN=(-c "$PIN_INI" --rootdir "$TESTS_DIR" --confcutdir "$TESTS_DIR" -p no:cacheprovider)

cd "$TESTS_DIR" || exit 1

if python3 -m pytest "${PIN[@]}" --help 2>/dev/null | grep -q -- '--ctrf'; then
  python3 -m pytest "${PIN[@]}" test_outputs.py -q --tb=short --ctrf "$VDIR/ctrf.json"
else
  # Networked environments only: fetch the reporter with uvx. Check that uvx really exists
  # afterwards, because an offline `curl | sh` exits 0 without installing anything.
  if ! command -v uvx >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh 2>/dev/null | sh >/dev/null 2>&1 || true
    export PATH="$HOME/.local/bin:$PATH"
  fi
  if command -v uvx >/dev/null 2>&1; then
    uvx --with pytest==8.4.1 --with pytest-json-ctrf==0.3.5 \
      pytest "${PIN[@]}" --ctrf "$VDIR/ctrf.json" "$TESTS_DIR/test_outputs.py" -q --tb=short
  else
    python3 -m pytest "${PIN[@]}" test_outputs.py -q --tb=short --junitxml="$VDIR/junit.xml"
  fi
fi

python3 - <<'PY'
import json
import xml.etree.ElementTree as ET

# Fails closed: a missing, unreadable or empty report scores 0.0.
frac = 0.0
try:
    s = json.load(open('/logs/verifier/ctrf.json'))["results"]["summary"]
    total = int(s.get("tests", 0))
    passed = int(s.get("passed", 0))
    frac = passed / total if total else 0.0
except Exception:
    try:
        root = ET.parse('/logs/verifier/junit.xml').getroot()
        ts = root if root.tag == 'testsuite' else root.find('testsuite')
        total = int(ts.get('tests', 0))
        passed = total - (int(ts.get('failures', 0)) + int(ts.get('errors', 0))
                          + int(ts.get('skipped', 0)))
        frac = max(0.0, passed / total) if total else 0.0
    except Exception:
        pass
open('/logs/verifier/reward.txt', 'w').write(f"{frac:.4f}\n")
print("reward:", frac)
PY
