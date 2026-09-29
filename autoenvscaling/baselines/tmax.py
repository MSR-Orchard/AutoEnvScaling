"""Rebuild the Tmax static environment pool as Harbor tasks (the baseline of Section 4.2).

Tmax (Ivison et al., 2026) releases TMax-15K as ``tasks.zip`` on the Hugging Face Hub
(``allenai/TMax-15K``). Download it yourself, for example with
``huggingface-cli download allenai/TMax-15K tasks.zip --repo-type dataset``, then run::

    python -m autoenvscaling.baselines.tmax --zip tasks.zip --out pools/tmax15k \\
        [--select selection.json] [--slim] [--template managed|legacy]

Each task directory in the zip becomes a Harbor task:

* ``container.def`` (a Singularity definition bootstrapped from a Docker image) becomes
  ``environment/Dockerfile``: ``From:`` becomes ``FROM``; ``%files`` become ``COPY`` of the
  fixtures shipped in the task directory; ``%post`` becomes one build script run by one ``RUN``;
  ``%environment`` exports become ``ENV`` lines. The build script also installs pytest and
  pytest-json-ctrf so the verifier can run offline.
* ``test_final_state.py`` becomes ``tests/test_outputs.py``; ``tests/test.sh`` is the managed
  verifier (reward = fraction of tests passed) or, with ``--template legacy``, the original JUnit
  scorer.
* ``test_initial_state.py`` runs at build time as a non-fatal sanity check that prints
  ``TMAX_INITIAL_STATE_WARN`` into the build log when it fails.
* ``task.json`` gives ``instruction.md`` (its description) and the task.toml metadata.
* Tmax ships no reference solution, so ``solution/solve.sh`` is a stub that does nothing. Running
  it as the reference solution therefore measures the untouched container's reward.

Ported from the research code's ``tmax2harbor.py``. The research pipeline then built every image
into a private registry (``build_tmax15k.py``); that step is not needed here, because Harbor builds
each task's image from its Dockerfile when a trial starts.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import tomli_w

MANAGED_VERIFIER = Path(__file__).with_name("verifier_template.sh")

LEGACY_TEST_SH = """#!/bin/bash
# Tmax reference verifier: run the final-state pytest suite; reward = fraction of tests passed.
mkdir -p /logs/verifier
python3 -m pytest --version >/dev/null 2>&1 \\
  || python3 -m pip install -q pytest 2>/dev/null || pip3 install -q pytest
cd /tests
python3 -m pytest test_outputs.py -q --tb=short --junitxml=/logs/verifier/junit.xml
python3 - <<'PY'
import xml.etree.ElementTree as ET
frac = 0.0
try:
    root = ET.parse('/logs/verifier/junit.xml').getroot()
    ts = root if root.tag == 'testsuite' else root.find('testsuite')
    total = int(ts.get('tests', 0)); bad = int(ts.get('failures', 0)) + int(ts.get('errors', 0))
    frac = (total - bad) / total if total else 0.0
except Exception:
    pass
open('/logs/verifier/reward.txt', 'w').write(f"{frac:.4f}\\n")
print("reward:", frac)
PY
"""

SOLVE_STUB = """#!/bin/bash
# Tmax ships no executable reference solution. This stub does nothing, so running it as the
# reference solution measures the reward of the untouched container.
echo "tmax reference task: stub solution (no-op)"
"""


def parse_container_def(text: str):
    """Return (from_image, files [(src, dst)], post_lines, env_lines, unhandled_sections)."""
    from_image, files, post, env, unhandled = None, [], [], [], set()
    section = None
    for raw in text.splitlines():
        s = raw.strip()
        if s.lower().startswith("bootstrap:"):
            continue
        if s.lower().startswith("from:") and section is None:
            from_image = s.split(":", 1)[1].strip()
            continue
        if s.startswith("%"):
            section = s.split()[0][1:].lower()
            if section not in ("files", "post", "environment"):
                unhandled.add(section)
            continue
        if section == "files" and s:
            parts = s.split()
            if len(parts) >= 2:
                files.append((parts[0], parts[1]))
        elif section == "post":
            post.append(raw)
        elif section == "environment" and s.startswith("export "):
            env.append(s[len("export "):])
    return from_image, files, post, env, unhandled


def task_toml(tid: str, meta: dict, task_json: dict) -> str:
    domain = str(meta.get("domain", task_json.get("domain", "unknown")))
    tier = str(meta.get("tier", task_json.get("tier", "unknown")))
    skill = str(meta.get("skill_type", task_json.get("skill_type", "")))[:60]
    config = {
        "schema_version": "1.3",
        "task": {"name": f"tmax/{tid}",
                 "description": f"TMax-15K reference task {tid} ({domain}/{tier})",
                 "authors": [{"name": "TMax-15K (allenai), converted by AutoEnvScaling"}],
                 "keywords": ["tmax15k", domain, tier]},
        "metadata": {"difficulty": tier, "category": domain,
                     "tags": [t for t in ("tmax15k", domain, skill, tier) if t]},
        "verifier": {"timeout_sec": 600.0},
        "agent": {"timeout_sec": 1200.0},
        "environment": {"network_mode": "no-network", "build_timeout_sec": 900.0, "os": "linux"},
    }
    return tomli_w.dumps(config)


def convert(task_dir: Path, out_dir: Path, meta: dict | None = None, *, slim: bool = False,
            test_sh: str | None = None) -> dict:
    """Convert one extracted Tmax task directory into ``out_dir/<task id>``.

    ``slim`` writes the text files only (no fixtures), for reading as examples rather than
    building.
    """
    meta = meta or {}
    tid = task_dir.name
    dst = out_dir / tid
    if dst.exists():
        shutil.rmtree(dst)
    envd = dst / "environment"
    envd.mkdir(parents=True)
    (dst / "tests").mkdir()
    (dst / "solution").mkdir()

    from_image, files, post, env_lines, unhandled = parse_container_def(
        (task_dir / "container.def").read_text())
    if not from_image:
        raise ValueError("no From: image in container.def")

    # %files sources are absolute paths on the authors' machines; the files themselves ship in
    # the task directory, so look them up by basename.
    copies = []
    for src, dest in files:
        rel = Path("fixtures") / Path(src).name
        if not slim:
            local = next((c for c in task_dir.rglob(Path(src).name) if c.is_file()), None)
            if local is None:
                raise ValueError(f"fixture {src} not found in task dir")
            (envd / rel).parent.mkdir(exist_ok=True)
            shutil.copy(local, envd / rel)
        copies.append((rel.as_posix(), dest))

    build_sh = ("#!/bin/bash\nset -e\n" + "\n".join(post) +
                "\n# the verifier runs offline, so install its tools at build time\n"
                "python3 -m pip install -q pytest pytest-json-ctrf 2>/dev/null || "
                "pip3 install -q pytest pytest-json-ctrf 2>/dev/null || true\n")
    (envd / "_tmax_build.sh").write_text(build_sh)

    lines = []
    if slim:
        lines.append("# Slim reference copy: fixture files omitted; for reading, not building.")
    lines.append(f"FROM {from_image}")
    lines += [f"ENV {e}" for e in env_lines]
    lines += [f"COPY {rel} {dest}" for rel, dest in copies]
    lines += ["COPY _tmax_build.sh /tmp/_tmax_build.sh",
              "RUN bash /tmp/_tmax_build.sh && rm -f /tmp/_tmax_build.sh"]
    initial = task_dir / "test_initial_state.py"
    if initial.exists():
        shutil.copy(initial, envd / "_tmax_initial_check.py")
        lines += ["COPY _tmax_initial_check.py /opt/_tmax_initial_check.py",
                  ("RUN cd / && python3 -m pytest /opt/_tmax_initial_check.py -q "
                   "|| echo 'TMAX_INITIAL_STATE_WARN'")]
    lines.append("WORKDIR /app")
    (envd / "Dockerfile").write_text("\n".join(lines) + "\n")

    task_json = json.loads((task_dir / "task.json").read_text())
    (dst / "instruction.md").write_text(task_json["description"])
    shutil.copy(task_dir / "test_final_state.py", dst / "tests" / "test_outputs.py")
    (dst / "tests" / "test.sh").write_text(test_sh or MANAGED_VERIFIER.read_text())
    (dst / "solution" / "solve.sh").write_text(SOLVE_STUB)
    for script in (dst / "tests" / "test.sh", dst / "solution" / "solve.sh"):
        script.chmod(0o755)
    (dst / "task.toml").write_text(task_toml(tid, meta, task_json))
    return {"task_id": tid, "from": from_image, "n_fixtures": len(copies),
            "unhandled_sections": sorted(unhandled)}


def convert_zip(zip_path: Path, out: Path, selection: list[dict] | None = None, *,
                slim: bool = False, template: str = "managed") -> list[dict]:
    """Convert the selected tasks (all tasks when ``selection`` is None) from ``tasks.zip``."""
    out.mkdir(parents=True, exist_ok=True)
    test_sh = LEGACY_TEST_SH if template == "legacy" else MANAGED_VERIFIER.read_text()
    extract_root = out.parent / f"{out.name}_extracted"
    results = []
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        if selection is None:
            ids = sorted({n.split("/", 1)[0] for n in names if n.endswith("/task.json")})
            selection = [{"task_id": tid} for tid in ids]
        for meta in selection:
            tid = meta["task_id"]
            for name in names:
                if name.startswith(tid + "/"):
                    zf.extract(name, extract_root)
            try:
                record = convert(extract_root / tid, out, meta, slim=slim, test_sh=test_sh)
                record["ok"] = True
            except Exception as exc:  # one malformed task must not stop the conversion
                record = {"task_id": tid, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
            results.append(record)
    (out / "_conversion_report.json").write_text(json.dumps(results, indent=1))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", type=Path, required=True, help="TMax-15K tasks.zip")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--select", type=Path, default=None,
                        help="JSON list of {task_id, domain, tier, skill_type}; default: all")
    parser.add_argument("--slim", action="store_true", help="text only, no fixture files")
    parser.add_argument("--template", choices=["managed", "legacy"], default="managed")
    args = parser.parse_args(argv)
    selection = json.loads(args.select.read_text()) if args.select else None
    results = convert_zip(args.zip, args.out, selection, slim=args.slim, template=args.template)
    ok = sum(r["ok"] for r in results)
    for r in results:
        if not r["ok"]:
            print(f"FAIL {r['task_id']}: {r['error'][:120]}")
    print(f"converted {ok}/{len(results)} -> {args.out}")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
