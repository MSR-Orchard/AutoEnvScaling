#!/bin/bash
# AutoEnvScaling fraction-scoring verifier template.
# reward = tests passed / tests total, from pytest's CTRF report (or its JUnit XML as a fallback).
# Solving and grading run without network, so install pytest and pytest-json-ctrf in the image
# (environment/Dockerfile), for example:
#   RUN pip install --no-cache-dir pytest==8.4.1 pytest-json-ctrf==0.3.5
mkdir -p /logs/verifier
cd /tests || exit 1
if python3 -c "import pytest_json_ctrf" 2>/dev/null; then
  python3 -m pytest test_outputs.py -q --tb=short --ctrf /logs/verifier/ctrf.json
else
  # Without the CTRF plugin, JUnit XML gives the same fraction with any modern pytest.
  python3 -m pytest test_outputs.py -q --tb=short --junitxml=/logs/verifier/junit.xml
fi
python3 - <<'PY'
import json
import xml.etree.ElementTree as ET

fraction = 0.0
try:
    summary = json.load(open("/logs/verifier/ctrf.json"))["results"]["summary"]
    total, passed = int(summary.get("tests", 0)), int(summary.get("passed", 0))
    fraction = passed / total if total else 0.0
except Exception:
    try:
        root = ET.parse("/logs/verifier/junit.xml").getroot()
        suite = root if root.tag == "testsuite" else root.find("testsuite")
        total = int(suite.get("tests", 0))
        bad = int(suite.get("failures", 0)) + int(suite.get("errors", 0))
        bad += int(suite.get("skipped", 0))
        fraction = (total - bad) / total if total else 0.0
    except Exception:
        pass
open("/logs/verifier/reward.txt", "w").write(f"{fraction:.4f}\n")
print("reward:", fraction)
PY
