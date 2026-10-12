"""Command line for pools, generation rounds and training loops.

    # Seed a pool from Harbor task directories (a Tmax pool, or a pool from a frontier proposer).
    python -m autoenvscaling.selfplay seed --pool-dir runs/pool --source tmax tasks/*

    # One generation round with a fixed-weight proposer, parents drawn from a seed pool.
    python -m autoenvscaling.selfplay generate --parents-pool runs/seed --pool-dir runs/pool \\
        --harness claude-code --model claude-opus-5 --assignments 64

    # The active pool as Miles prompt rows (for generate_fn mode).
    python -m autoenvscaling.selfplay prompts --pool-dir runs/pool --out runs/pool/prompts.jsonl

    # When reviews, rounds and proposer updates happen; the total is Miles' --num-rollout.
    python -m autoenvscaling.selfplay schedule --steps 288

    # Train with Miles in bridge mode (start the Miles job with the same --bridge-dir).
    python -m autoenvscaling.selfplay run --mode shared --pool-dir runs/pool \\
        --run-dir runs/rsi --bridge-dir runs/rsi/bridge --steps 288

Settings come from `--config` (TOML) and AES_* environment variables (see config.Settings).
The solver endpoint for `generate` defaults to AES_SOLVER_BASE_URL and AES_SOLVER_MODEL.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path

from autoenvscaling.calibration.solver import SolverEndpoint
from autoenvscaling.config import Settings
from autoenvscaling.pipeline import GenerationPipeline, ProposerSpec, plan_assignments
from autoenvscaling.pool import Pool
from autoenvscaling.selfplay.loop import FRONTIER, MODES, SHARED, LoopConfig, SelfPlayLoop
from autoenvscaling.selfplay.schedule import Schedule
from autoenvscaling.selfplay.trainer import FixedEndpointTrainer
from autoenvscaling.trajectories.export import ChatFormat


def _settings(args: argparse.Namespace) -> Settings:
    return Settings.load(args.config)


def cmd_seed(args: argparse.Namespace) -> int:
    settings = _settings(args)
    pool = Pool.load(args.pool_dir, capacity=args.capacity or settings.selfplay.pool_size)
    entries = pool.seed([Path(p) for p in args.tasks], source=args.source, copy=not args.no_copy)
    pool.save()
    print(json.dumps({"seeded": len(entries), **pool.summary()}))
    return 0


def cmd_prompts(args: argparse.Namespace) -> int:
    pool = Pool.load(args.pool_dir, capacity=args.capacity or _settings(args).selfplay.pool_size)
    images = json.loads(args.images.read_text()) if args.images else None
    n = pool.export_prompt_data(args.out, images=images)
    print(json.dumps({"rows": n, "out": str(args.out)}))
    return 0


def _chat(tools: Path | None, system_prompt: Path | None) -> ChatFormat | None:
    if tools is None and system_prompt is None:
        return None
    return ChatFormat(system_prompt=system_prompt.read_text() if system_prompt else None,
                      tools=json.loads(tools.read_text()) if tools else None)


def cmd_schedule(args: argparse.Namespace) -> int:
    settings = _settings(args)
    sp = settings.selfplay
    schedule = Schedule(args.steps, sp.proposer_interval, sp.pool_review_interval,
                        refresh_pool=args.mode != "static", train_proposer=args.mode == SHARED)
    for step, event in schedule.timeline():
        if event != "solver_update" or args.verbose:
            print(f"after solver step {step:4d}: {event}")
    print(f"planned trainer updates (Miles --num-rollout): {schedule.planned_updates}")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    settings = _settings(args)
    pool = Pool.load(args.pool_dir, capacity=args.capacity or settings.selfplay.pool_size)
    parents = Pool.load(args.parents_pool) if args.parents_pool else pool
    solver = SolverEndpoint(base_url=args.solver_base_url or os.environ.get("AES_SOLVER_BASE_URL", ""),
                            model=args.solver_model or os.environ.get("AES_SOLVER_MODEL", ""))
    if not solver.base_url or not solver.model:
        print("set --solver-base-url and --solver-model (or AES_SOLVER_BASE_URL and AES_SOLVER_MODEL)",
              file=sys.stderr)
        return 2
    rng = random.Random(args.seed)
    round_id = args.round_id or "round-seed"
    assignments = plan_assignments(parents, [], args.assignments, round_id=round_id, rng=rng,
                                   solver_feedback=not args.no_solver_feedback)
    pipeline = GenerationPipeline(settings, proposer=ProposerSpec(args.harness, args.model),
                                  solver=solver, out_dir=Path(args.out_dir), pool=pool,
                                  workspace_pool=parents,
                                  solver_feedback=not args.no_solver_feedback,
                                  max_concurrent_chains=args.concurrency)
    result = asyncio.run(pipeline.run_round(assignments, round_id=round_id,
                                            episodes_per_assignment=args.episodes))
    print(json.dumps({**result.summary(), "pool": pool.summary()}))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    settings = _settings(args)
    pool = Pool.load(args.pool_dir, capacity=settings.selfplay.pool_size)
    config = LoopConfig(total_solver_steps=args.steps, mode=args.mode,
                        train_proposer=not args.no_proposer_training,
                        proposer_harness=args.harness or ("pi" if args.mode == SHARED else ""),
                        proposer_model=args.model or "", solver_feedback=not args.no_solver_feedback,
                        seed=args.seed,
                        solver_chat=_chat(args.solver_tools, args.solver_system_prompt),
                        proposer_chat=_chat(args.proposer_tools, args.proposer_system_prompt))
    if args.fixed_endpoint:
        trainer = FixedEndpointTrainer(SolverEndpoint(args.fixed_endpoint, args.policy_model),
                                       Path(args.run_dir) / "batches")
    else:
        from autoenvscaling.training.miles.bridge import MilesBridgeTrainer
        trainer = MilesBridgeTrainer(Path(args.bridge_dir), model=args.policy_model,
                                     public_base_url=args.public_base_url)
    loop = SelfPlayLoop(settings, config, pool=pool, trainer=trainer, run_dir=Path(args.run_dir))
    print(f"planned trainer updates: {loop.schedule.planned_updates}")
    state = asyncio.run(loop.run())
    print(json.dumps({"solver_steps": state.solver_steps, "proposer_updates": state.proposer_updates}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m autoenvscaling.selfplay",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, help="settings TOML")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("seed", help="admit existing Harbor tasks into a pool")
    p.add_argument("--pool-dir", type=Path, required=True)
    p.add_argument("--capacity", type=int)
    p.add_argument("--source", default="seed")
    p.add_argument("--no-copy", action="store_true", help="reference tasks in place")
    p.add_argument("tasks", nargs="+")
    p.set_defaults(fn=cmd_seed)

    p = sub.add_parser("prompts", help="write the active pool as Miles prompt rows")
    p.add_argument("--pool-dir", type=Path, required=True)
    p.add_argument("--capacity", type=int)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--images", type=Path,
                   help="JSON mapping task ids to built environment images (carry mode)")
    p.set_defaults(fn=cmd_prompts)

    p = sub.add_parser("schedule", help="print the training schedule")
    p.add_argument("--steps", type=int, default=288)
    p.add_argument("--mode", choices=MODES, default=SHARED)
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(fn=cmd_schedule)

    p = sub.add_parser("generate", help="one generation round with a fixed-weight proposer")
    p.add_argument("--pool-dir", type=Path, required=True, help="pool that receives admissions")
    p.add_argument("--parents-pool", type=Path, help="pool that supplies parents (default: --pool-dir)")
    p.add_argument("--capacity", type=int)
    p.add_argument("--out-dir", type=Path, default=Path("runs/generate"))
    p.add_argument("--harness", required=True, choices=("claude-code", "codex", "pi"))
    p.add_argument("--model", required=True)
    p.add_argument("--assignments", type=int, required=True)
    p.add_argument("--episodes", type=int, default=1, help="episodes per assignment")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--solver-base-url")
    p.add_argument("--solver-model")
    p.add_argument("--no-solver-feedback", action="store_true")
    p.add_argument("--round-id")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_generate)

    p = sub.add_parser("run", help="run the training loop")
    p.add_argument("--mode", choices=MODES, default=SHARED)
    p.add_argument("--pool-dir", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=288)
    p.add_argument("--bridge-dir", type=Path, help="Miles bridge control directory")
    p.add_argument("--public-base-url", help="address under which sandboxes reach the Miles router")
    p.add_argument("--policy-model", default="policy", help="served model name of the policy")
    p.add_argument("--fixed-endpoint", help="dry run: use this endpoint and do not train")
    p.add_argument("--harness", help="proposer harness (frontier mode)")
    p.add_argument("--model", help="proposer model (frontier mode)")
    p.add_argument("--no-solver-feedback", action="store_true")
    p.add_argument("--no-proposer-training", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    for role in ("solver", "proposer"):
        p.add_argument(f"--{role}-tools", type=Path,
                       help=f"JSON tool definitions the {role} harness sends (for re-tokenizing)")
        p.add_argument(f"--{role}-system-prompt", type=Path,
                       help=f"text file with the {role} harness's system prompt")
    p.set_defaults(fn=cmd_run)

    args = parser.parse_args(argv)
    if args.command == "run":
        if not args.fixed_endpoint and not args.bridge_dir:
            parser.error("run needs --bridge-dir (Miles) or --fixed-endpoint (dry run)")
        if args.mode == FRONTIER and not (args.harness and args.model):
            parser.error("frontier mode needs --harness and --model")
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
