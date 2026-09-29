"""Command line for the Terminal-Bench evaluations.

    python -m autoenvscaling.evals config  --benchmark tb2.1 --model <id>     # print the Harbor job, run nothing
    python -m autoenvscaling.evals run     --benchmark tb2.1 --model <id> -k 5
    python -m autoenvscaling.evals report  runs/evals/<job> [<rerun job> ...] -k 5
    python -m autoenvscaling.evals preflight --model <id>                   # one billable request

The endpoint URL and key are read from environment variables (``AES_EVAL_BASE_URL`` and
``AES_EVAL_API_KEY`` by default) so they never appear in a config file or a log.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from autoenvscaling.evals.endpoint import EndpointSpec, preflight
from autoenvscaling.evals.pi_agent import QWEN_SOLVING_SAMPLING
from autoenvscaling.evals.terminal_bench import (
    BENCHMARKS,
    EvalSettings,
    build_job_config,
    format_report,
    report,
    run_eval,
)

SAMPLING_PRESETS = {"none": None, "qwen-solving": QWEN_SOLVING_SAMPLING}


def _endpoint_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="exact served model id")
    parser.add_argument("--base-url-env", default="AES_EVAL_BASE_URL")
    parser.add_argument("--api-key-env", default="AES_EVAL_API_KEY")
    parser.add_argument("--api", default="openai-completions",
                        choices=("openai-completions", "openai-responses"))
    parser.add_argument("--context-window", type=int, default=262_144)
    parser.add_argument("--max-output-tokens", type=int, default=32_768)
    parser.add_argument("--sampling", default="none", choices=sorted(SAMPLING_PRESETS),
                        help="decoding vector added to every request (default: server defaults)")
    parser.add_argument("--sampling-json", help="explicit decoding vector as JSON; overrides --sampling")
    parser.add_argument("--provider-routing-json",
                        help='router pinning, e.g. \'{"order": ["X"], "allow_fallbacks": false}\'')


def _endpoint(args: argparse.Namespace) -> EndpointSpec:
    sampling = (json.loads(args.sampling_json) if args.sampling_json
                else SAMPLING_PRESETS[args.sampling])
    routing = json.loads(args.provider_routing_json) if args.provider_routing_json else None
    return EndpointSpec(model=args.model, base_url_env=args.base_url_env,
                        api_key_env=args.api_key_env, api=args.api,
                        context_window=args.context_window,
                        max_output_tokens=args.max_output_tokens,
                        sampling=sampling, provider_routing=routing)


def _run_args(parser: argparse.ArgumentParser) -> None:
    _endpoint_args(parser)
    parser.add_argument("--benchmark", default="tb2.1", choices=sorted(BENCHMARKS))
    parser.add_argument("--dataset", help="override the dataset: a local task dir or a Harbor spec")
    parser.add_argument("-k", type=int, default=5, help="attempts per task (default 5, avg@5)")
    parser.add_argument("--env", default="modal", help="Harbor environment type (default modal)")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--memory-gib", type=int, default=8)
    parser.add_argument("--task-resources", action="store_true",
                        help="use each task's declared resources instead of 4 CPU / 8 GiB")
    parser.add_argument("--agent-timeout-sec", type=float)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--jobs-dir", type=Path, default=Path("runs/evals"))
    parser.add_argument("--job-name")
    parser.add_argument("--task", action="append", dest="task_names", help="glob; repeatable")
    parser.add_argument("--exclude-task", action="append", dest="exclude_task_names")
    parser.add_argument("--n-tasks", type=int)
    parser.add_argument("--tools", default="bash", help="Pi tools (default: bash only)")


def _settings(args: argparse.Namespace) -> EvalSettings:
    return EvalSettings(
        benchmark=args.benchmark, endpoint=_endpoint(args), k=args.k, harbor_env=args.env,
        n_concurrent=args.concurrency,
        cpus=None if args.task_resources else args.cpus,
        memory_gib=None if args.task_resources else args.memory_gib,
        agent_timeout_sec=args.agent_timeout_sec, max_retries=args.max_retries,
        jobs_dir=args.jobs_dir, job_name=args.job_name, task_names=args.task_names,
        exclude_task_names=args.exclude_task_names, n_tasks=args.n_tasks,
        dataset_override=args.dataset, tools=args.tools)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m autoenvscaling.evals",
                                     description="Terminal-Bench 2.1 / 4.0 evaluation (avg@k).")
    sub = parser.add_subparsers(dest="command", required=True)
    _run_args(sub.add_parser("config", help="print the Harbor job config without running it"))
    _run_args(sub.add_parser("run", help="run the evaluation (uses sandboxes and tokens)"))
    rep = sub.add_parser("report", help="score finished Harbor job directories")
    rep.add_argument("job_dirs", nargs="+", type=Path)
    rep.add_argument("-k", type=int, default=5)
    rep.add_argument("--missing", choices=("exclude", "zero"), default="exclude")
    rep.add_argument("--out", type=Path, help="write the JSON report here")
    pre = sub.add_parser("preflight", help="one tiny request to check the endpoint and model id")
    _endpoint_args(pre)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "config":
        config = build_job_config(_settings(args))
        print(config.model_dump_json(indent=2))
        return 0
    if args.command == "run":
        settings = _settings(args)
        job_dir = asyncio.run(run_eval(settings))
        rep = report([job_dir], settings.k, benchmark=settings.benchmark)
        (job_dir / "aes_report.json").write_text(json.dumps(rep, indent=2) + "\n")
        print(format_report(rep))
        print(f"report: {job_dir / 'aes_report.json'}")
        return 0
    if args.command == "report":
        rep = report(args.job_dirs, args.k, missing=args.missing)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(rep, indent=2) + "\n")
        print(format_report(rep))
        return 0
    if args.command == "preflight":
        print(json.dumps(preflight(_endpoint(args)).__dict__, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
