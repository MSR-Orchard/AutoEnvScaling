"""Run the recording proxy, or check a carry configuration offline.

    python -m autoenvscaling.training.carry proxy          # AES_PROXY_* configure it
    python -m autoenvscaling.training.carry check          # AES_CARRY_* and the Miles geometry
"""

from __future__ import annotations

import argparse
import sys

from autoenvscaling.training.carry.rollout import CarrySettings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m autoenvscaling.training.carry")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("proxy", help="serve the recording proxy (needs fastapi, uvicorn, transformers)")
    check = sub.add_parser("check", help="validate AES_CARRY_* against the batch geometry")
    check.add_argument("--n-samples-per-prompt", type=int, default=16)
    args = parser.parse_args(argv)

    if args.command == "proxy":
        from autoenvscaling.training.carry.proxy import main as proxy_main
        return proxy_main()

    settings = CarrySettings.from_env()
    problems = settings.problems(args.n_samples_per_prompt)
    for name, value in sorted(settings.to_env().items()):
        print(f"{name}={value}")
    for problem in problems:
        print(f"problem: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
