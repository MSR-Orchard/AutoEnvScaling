"""Print or run a Miles job from a preset.

    python -m autoenvscaling.training.miles plan --preset shared-policy \\
        --hf-checkpoint <hf dir> --ref-load <torch_dist dir> --save <out dir> --bridge-dir <dir>
    python -m autoenvscaling.training.miles run ...        # same arguments, actually launches
    python -m autoenvscaling.training.miles presets

`--set name=value` overrides any MilesConfig field, e.g. `--set rollout_batch_size=8`.
`plan` is the default habit: these jobs cost GPU-days.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

from autoenvscaling.training.miles.config import PRESETS, MilesConfig
from autoenvscaling.training.miles.launcher import ConfigError, MilesLauncher


def _coerce(config: MilesConfig, name: str, raw: str) -> Any:
    kinds = {f.name: f for f in fields(MilesConfig)}
    if name not in kinds:
        raise SystemExit(f"unknown MilesConfig field {name!r}")
    current = getattr(config, name)
    if isinstance(current, bool):
        return raw.lower() in {"1", "true", "yes"}
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, list):
        return raw.split()
    return None if raw.lower() == "none" else raw


def build_config(args: argparse.Namespace) -> MilesConfig:
    paths = {k: v for k, v in {"hf_checkpoint": args.hf_checkpoint, "ref_load": args.ref_load,
                               "save": args.save, "load": args.load, "prompt_data": args.prompt_data,
                               "generate_function_path": args.generate_fn}.items() if v}
    config = PRESETS[args.preset](**paths)
    if args.generate_fn:
        config = config.with_(mode="generate_fn")
    if args.carry:
        config = config.with_(mode="carry")
    if args.num_rollout is not None:
        config = config.with_(num_rollout=args.num_rollout)
    for item in args.set or []:
        name, _, raw = item.partition("=")
        config = config.with_(**{name: _coerce(config, name, raw)})
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m autoenvscaling.training.miles")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("presets")
    for name in ("plan", "run"):
        p = sub.add_parser(name)
        p.add_argument("--preset", choices=sorted(PRESETS), default="shared-policy")
        p.add_argument("--hf-checkpoint")
        p.add_argument("--ref-load")
        p.add_argument("--save")
        p.add_argument("--load")
        p.add_argument("--prompt-data")
        p.add_argument("--generate-fn", help="switch to generate_fn mode with this generate function")
        p.add_argument("--carry", action="store_true",
                       help="switch to carry mode (partial rollout and over-sampling; needs --prompt-data)")
        p.add_argument("--num-rollout", type=int)
        p.add_argument("--bridge-dir", help="bridge control directory (bridge mode)")
        p.add_argument("--set", action="append", metavar="NAME=VALUE")
        p.add_argument("--log", type=Path, help="append the job's output to this file")
    args = parser.parse_args(argv)

    if args.command == "presets":
        for name, make in sorted(PRESETS.items()):
            doc = (make.__doc__ or "").strip()
            print(f"{name}: {doc.splitlines()[0] if doc else ''}")
        return 0

    config = build_config(args)
    launcher = MilesLauncher(bridge_dir=args.bridge_dir)
    problems = launcher.problems(config)
    for problem in problems:
        print(f"problem: {problem}", file=sys.stderr)
    print(launcher.plan(config))
    if args.command == "plan":
        return 1 if problems else 0
    try:
        result = launcher.run(config, log_path=args.log)
    except ConfigError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    print(f"ok={result.ok} returncode={result.returncode} checkpoint={result.checkpoint}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
