"""`aes <command> ...` (or `autoenvscaling <command> ...`): one entry point for every part of the pipeline.

    admit       validate (and optionally calibrate) one Harbor task directory
    seed        admit existing Harbor tasks into a pool          (= selfplay seed)
    generate    one generation round with a fixed-weight proposer (= selfplay generate)
    prompts     write the active pool as Miles prompt rows        (= selfplay prompts)
    schedule    print the training schedule                       (= selfplay schedule)
    run         run the self-play training loop                   (= selfplay run)
    selfplay    the same five steps under one command group
    train       plan or launch Miles training jobs
    export      proposer/solver trajectories to SFT rows or RL groups
    baselines   Prompt and Prompt + Env Feedback generation arms
    tmax        convert Tmax tasks to Harbor tasks
    eval        Terminal-Bench 2.1 / 4.0 evaluation
    hilbench    HiL-Bench task generation and evaluation
    solver      run the current solver on a task (the proposer's solver_model tool)

Run `autoenvscaling <command> --help` for the options of each command.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
from pathlib import Path

COMMANDS = {
    "selfplay": "autoenvscaling.selfplay.__main__",
    "train": "autoenvscaling.training.miles.__main__",
    "export": "autoenvscaling.trajectories.export",
    "baselines": "autoenvscaling.baselines.run",
    "tmax": "autoenvscaling.baselines.tmax",
    "eval": "autoenvscaling.evals.cli",
    "hilbench": "autoenvscaling.evals.hilbench.__main__",
    "solver": "autoenvscaling.proposer.solver_tool",
}


# The self-play steps are also top-level commands, so `aes generate ...` works without the
# `selfplay` group name. `selfplay` takes --config before its subcommand; _selfplay_argv moves a
# --config given after the step name into that position.
SELFPLAY_STEPS = ("seed", "generate", "prompts", "schedule", "run")


def _selfplay_argv(step: str, rest: list[str]) -> list[str]:
    config: list[str] = []
    others: list[str] = []
    i = 0
    while i < len(rest):
        arg = rest[i]
        if arg == "--config" and i + 1 < len(rest):
            config = ["--config", rest[i + 1]]
            i += 2
            continue
        if arg.startswith("--config="):
            config = [arg]
        else:
            others.append(arg)
        i += 1
    return [*config, step, *others]


def admit_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="autoenvscaling admit",
        description="Run the host's admission checks on one task: validation, then calibration.")
    parser.add_argument("task_dir", type=Path)
    parser.add_argument("--config", type=Path, help="TOML settings file")
    parser.add_argument("--no-calibrate", action="store_true",
                        help="validation only (no solver endpoint needed)")
    args = parser.parse_args(argv)

    from autoenvscaling.calibration.band import calibrate
    from autoenvscaling.calibration.solver import SolverEndpoint
    from autoenvscaling.config import Settings
    from autoenvscaling.rewards import proposer_reward
    from autoenvscaling.tasks import HarborTask
    from autoenvscaling.validation.admission import validate
    from autoenvscaling.validation.rubric import ReviewerSpec

    settings = Settings.load(args.config)
    task = HarborTask(args.task_dir.resolve())

    async def run():
        validation = await validate(task, settings, reviewer=ReviewerSpec.from_env())
        calibration = None
        if validation.passed and not args.no_calibrate:
            calibration = await calibrate(task, SolverEndpoint.from_env(), settings)
        return validation, calibration

    validation, calibration = asyncio.run(run())
    result = {"validation": validation.to_dict(),
              "calibration": calibration.to_dict() if calibration else None}
    if not args.no_calibrate:
        result["proposer_reward"] = proposer_reward(validation, calibration)
    print(json.dumps(result, indent=2))
    return 0 if validation.passed else 1


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in {"-h", "--help"}:
        print(__doc__)
        return 0
    command, rest = argv[0], argv[1:]
    if command == "admit":
        return admit_main(rest)
    if command in SELFPLAY_STEPS:
        selfplay = importlib.import_module(COMMANDS["selfplay"])
        return selfplay.main(_selfplay_argv(command, rest)) or 0
    if command not in COMMANDS:
        print(f"unknown command {command!r}\n{__doc__}", file=sys.stderr)
        return 2
    module = importlib.import_module(COMMANDS[command])
    return module.main(rest) or 0


if __name__ == "__main__":
    sys.exit(main())
