"""Evaluations: Terminal-Bench 2.1 and 4.0 (avg@k with the Pi agent) and HiL-Bench metrics.

Modules:
    scoring          avg@k, pass@k, and the infrastructure / verifier-health accounting
    harbor_results   read Harbor trial results from a job directory
    endpoint         the OpenAI-compatible endpoint under evaluation, and a preflight check
    pi_agent         the Pi coding agent as a Harbor agent, run inside the task sandbox
    terminal_bench   benchmark specs, the Harbor job, and the report
    hilbench         HiL-Bench task generation (blocker registries, variants, ask_human) and
                     metrics (task success, Ask-F1, ask rate)

Command line: ``python -m autoenvscaling.evals --help``.
"""
