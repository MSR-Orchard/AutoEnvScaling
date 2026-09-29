"""Command line for the Section 4.1 baselines.

Generate and score (resumable; finished assignments are skipped)::

    python -m autoenvscaling.baselines.run generate --arm prompt \\
        --assignments assignments.jsonl --model <model> --out runs/baselines/<model>
    python -m autoenvscaling.baselines.run generate --arm feedback \\
        --assignments assignments.jsonl --model <model> --out runs/baselines/<model>

The feedback arm reuses ``<out>/prompt/<id>/result.json`` as its first round and runs the Prompt
arm first where that result is missing. The model is called through the OpenAI-compatible endpoint
in ``AES_PROPOSER_BASE_URL`` (key in ``AES_PROPOSER_API_KEY``). Scoring uses the Harbor environment
in ``Settings.sandbox.harbor_env`` (Modal by default).

Summarise::

    python -m autoenvscaling.baselines.run summarize --out runs/baselines/<model> \\
        [--usd-per-mtok-in 1.25 --usd-per-mtok-out 10]

Assignments are JSON lines (or one JSON list). Each record needs ``assignment_id`` and either a
``description`` (free text) or the fields of ``autoenvscaling.trajectories.schema.Assignment``,
which are rendered into a description.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

from autoenvscaling.baselines.arms import (ArmConfig, ArmResult, run_feedback_arm, run_prompt_arm,
                                           write_result)
from autoenvscaling.baselines.llm import ChatClient
from autoenvscaling.config import Settings
from autoenvscaling.tasks import HarborTask
from autoenvscaling.validation.admission import EXECUTION_CHECKS, validate

logger = logging.getLogger(__name__)


def load_assignments(path: Path) -> list[dict[str, Any]]:
    text = Path(path).read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict) and "assignments" in data:
            records = data["assignments"]
        else:
            records = [data]
    for rec in records:
        if "assignment_id" not in rec:
            raise ValueError(f"assignment without assignment_id: {str(rec)[:200]}")
    return records


def describe(rec: dict[str, Any]) -> str:
    """The task description a baseline receives for one assignment."""
    text = rec.get("description") or (rec.get("extra") or {}).get("description")
    if text:
        return str(text)
    lines = []
    if rec.get("domain"):
        lines.append(f"Domain: {rec['domain']}")
    if rec.get("parent_task_id"):
        lines.append(f"Parent environment: {rec['parent_task_id']}")
    if rec.get("solver_pass_rate") is not None:
        lines.append(f"Solver pass rate on the parent: {float(rec['solver_pass_rate']):.2f}")
    if rec.get("common_failure"):
        lines.append(f"Most common solver failure: {rec['common_failure']}")
    direction = rec.get("direction")
    if direction == "simpler":
        lines.append("Build a simpler variant that exercises the same weakness.")
    elif direction == "harder":
        lines.append("Build a harder variant: a longer horizon or an additional skill.")
    if not lines:
        raise ValueError(f"assignment {rec['assignment_id']} has no description")
    return "\n".join(lines)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Proportion and 95% Wilson interval (as in the research analysis, three_way.py)."""
    if not n:
        return (math.nan, math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def summarize(out: Path, usd_in: float | None = None, usd_out: float | None = None) -> dict:
    """Valid-environment rate per arm, with refusals and infrastructure failures excluded."""
    summary: dict[str, Any] = {}
    for arm in ("prompt", "feedback"):
        results = [ArmResult.from_dict(json.loads(p.read_text()))
                   for p in sorted((Path(out) / arm).glob("*/result.json"))]
        if not results:
            continue
        excluded: dict[str, int] = {}
        by_round: dict[int, int] = {}
        tokens_in = tokens_out = 0
        for r in results:
            if r.excluded:
                excluded[r.excluded] = excluded.get(r.excluded, 0) + 1
                continue
            if r.admitted_round:
                by_round[r.admitted_round] = by_round.get(r.admitted_round, 0) + 1
            # The feedback arm is charged for its shared round 1 as well: every assignment needs
            # one generation call before any repair (research accounting, authoring_cost_feedback).
            for rd in r.rounds:
                for u in rd.usage:
                    tokens_in += int(u.get("prompt_tokens") or u.get("input_tokens") or 0)
                    tokens_out += int(u.get("completion_tokens") or u.get("output_tokens") or 0)
        scored = [r for r in results if not r.excluded]
        admitted = sum(r.admitted for r in scored)
        rate, lo, hi = wilson(admitted, len(scored))
        cell = {"assignments": len(results), "excluded": excluded, "scored": len(scored),
                "admitted": admitted, "valid_rate": rate, "ci95": [lo, hi],
                "admitted_by_round": dict(sorted(by_round.items())),
                "prompt_tokens": tokens_in, "completion_tokens": tokens_out}
        if usd_in is not None and usd_out is not None:
            cost = tokens_in / 1e6 * usd_in + tokens_out / 1e6 * usd_out
            cell["cost_usd"] = cost
            cell["cost_per_valid_env_usd"] = cost / admitted if admitted else None
        summary[arm] = cell
    return summary


async def generate(args: argparse.Namespace) -> int:
    settings = Settings.load(args.config)
    client = ChatClient.from_env(args.model, max_tokens=args.max_tokens,
                                 reasoning_effort=args.reasoning_effort)
    guidance = "\n\n".join(Path(p).read_text() for p in args.guidance_file or [])
    config = ArmConfig(managed_verifier=not args.own_verifier, extra_guidance=guidance,
                       max_repairs=args.max_repairs)
    records = load_assignments(args.assignments)[: args.limit or None]
    out = Path(args.out)
    sem = asyncio.Semaphore(args.concurrency)

    async def score(task_dir: Path):
        return await validate(HarborTask(task_dir), settings, checks=EXECUTION_CHECKS)

    async def prompt_result(aid: str, description: str) -> ArmResult:
        path = out / "prompt" / aid / "result.json"
        if path.is_file():
            return ArmResult.from_dict(json.loads(path.read_text()))
        result = await run_prompt_arm(aid, description, out / "prompt" / aid,
                                      complete=client.complete, score=score, config=config)
        write_result(path, result)
        return result

    async def one(rec: dict[str, Any]) -> None:
        aid = str(rec["assignment_id"])
        async with sem:
            try:
                description = describe(rec)
                seed = await prompt_result(aid, description)
                if args.arm == "feedback":
                    path = out / "feedback" / aid / "result.json"
                    if path.is_file():
                        return
                    result = await run_feedback_arm(aid, description, out / "feedback" / aid,
                                                    seed=seed, seed_dir=out / "prompt" / aid,
                                                    complete=client.complete, score=score,
                                                    config=config)
                    write_result(path, result)
                logger.info("%s done", aid)
            except Exception:
                logger.exception("assignment %s failed", aid)

    await asyncio.gather(*(one(rec) for rec in records))
    print(json.dumps(summarize(out), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="generate and score environments")
    gen.add_argument("--arm", choices=["prompt", "feedback"], required=True)
    gen.add_argument("--assignments", type=Path, required=True)
    gen.add_argument("--model", required=True)
    gen.add_argument("--out", type=Path, required=True)
    gen.add_argument("--config", type=Path, default=None, help="Settings TOML file")
    gen.add_argument("--limit", type=int, default=0)
    gen.add_argument("--concurrency", type=int, default=8)
    gen.add_argument("--max-tokens", type=int, default=64_000)
    gen.add_argument("--max-repairs", type=int, default=2)
    gen.add_argument("--reasoning-effort", default=None)
    gen.add_argument("--guidance-file", action="append",
                     help="authoring guidance to include in the prompt (repeatable)")
    gen.add_argument("--own-verifier", action="store_true",
                     help="let the model write tests/test.sh instead of using the managed one")
    summ = sub.add_parser("summarize", help="valid-environment rate per arm")
    summ.add_argument("--out", type=Path, required=True)
    summ.add_argument("--usd-per-mtok-in", type=float, default=None)
    summ.add_argument("--usd-per-mtok-out", type=float, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "summarize":
        print(json.dumps(summarize(args.out, args.usd_per_mtok_in, args.usd_per_mtok_out),
                         indent=2))
        return 0
    return asyncio.run(generate(args))


if __name__ == "__main__":
    sys.exit(main())
