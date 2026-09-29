"""Command line for the HiL-Bench pieces.

    python -m autoenvscaling.evals.hilbench prompt --index 3        # authoring prompt for episode 3
    python -m autoenvscaling.evals.hilbench derive <task_dir> ...     # full_info / ask_human / baseline
    python -m autoenvscaling.evals.hilbench report <episodes.jsonl> ... [--exclude-stop pi_timeout]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from autoenvscaling.evals.hilbench.generation import DEFAULT_OUTPUT_DIR, episode_prompt
from autoenvscaling.evals.hilbench.metrics import HilEpisode, report
from autoenvscaling.evals.hilbench.variants import derive_variants


def unique_names(paths: list[Path]) -> list[str]:
    """Shortest path suffixes that tell the tasks apart (episodes often all end in env_001)."""
    parts = [p.resolve().parts for p in paths]
    for depth in range(1, max((len(x) for x in parts), default=1) + 1):
        names = ["__".join(x[-depth:]) for x in parts]
        if len(set(names)) == len(names):
            return names
    return [str(i) for i in range(len(paths))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m autoenvscaling.evals.hilbench")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prompt", help="print the HiL authoring prompt for one episode")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    d = sub.add_parser("derive", help="derive the three variants of authored HiL tasks")
    d.add_argument("task_dirs", nargs="+", type=Path)
    d.add_argument("--out-root", type=Path,
                   help="write <out-root>/<task name>/ instead of <task_dir>/variants")
    r = sub.add_parser("report", help="task success, Ask-F1 and ask rate from episode records")
    r.add_argument("episodes", nargs="+", type=Path, help="episodes.jsonl files")
    r.add_argument("--model", help="label for records that do not name their model")
    r.add_argument("--exclude-stop", action="append", default=[],
                   help="also exclude episodes with this stop reason (repeatable)")
    args = parser.parse_args(argv)

    if args.command == "prompt":
        print(episode_prompt(args.index, offset=args.offset, output_dir=args.output_dir))
        return 0
    if args.command == "derive":
        failed = 0
        for task, name in zip(args.task_dirs, unique_names(args.task_dirs)):
            out = args.out_root / name if args.out_root else None
            summary = derive_variants(task, out)
            failed += not summary["derived"]
            print(json.dumps(summary))
        return 1 if failed else 0
    if args.command == "report":
        episodes = []
        for path in args.episodes:
            for line in path.read_text().splitlines():
                if line.strip():
                    episodes.append(HilEpisode.from_record(json.loads(line), model=args.model or ""))
        print(json.dumps(report(episodes, exclude_stop_reasons=args.exclude_stop), indent=2))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
