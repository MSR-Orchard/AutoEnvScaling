# Experiment 1: does environment design benefit from a terminal?

Section 4.1; Table 2 and Figures 4 and 5 (and the left panel of Figure 1).

Three ways to generate environments, on eight proposer models:

1. **Prompt**: the model writes the environment files from the task description; nothing is
   executed during generation (`autoenvscaling baselines generate --arm prompt`).
2. **Prompt + Env Feedback**: the generated environment is run and the build and verifier logs
   go back to the model for revision (`baselines generate --arm feedback`).
3. **AutoEnvScaling**: the model works in a sandbox with its coding agent and inspects files,
   runs commands, and validates and repairs the environment itself (`selfplay generate`).

An environment is valid if its image builds, its reference solution passes the tests, and an
empty solution fails them. Table 2 reports the valid rate (95% Wilson interval) and the cost per
valid environment.

```bash
export AES_PROPOSER_BASE_URL=... AES_PROPOSER_API_KEY=...   # prompt arms and Pi proposers
export AES_SOLVER_BASE_URL=... AES_SOLVER_MODEL=...         # solver for calibration
ASSIGNMENTS=assignments.jsonl SEED_TASKS="tasks/*" cmd/exp1_authoring/run.sh
DRY_RUN=1 cmd/exp1_authoring/run.sh                          # print the commands only
```

| Proposer | Coding agent (AutoEnvScaling arm) |
|---|---|
| Claude Opus 5 | Claude Code |
| GPT-6-astra, GPT-5.6-sol, GPT-5.6-terra, GPT-5.6-luna | Codex |
| Gemini-3.8-Flash, Grok 4.6, DeepSeek-V4-Flash | Pi |

Proposer sandbox: 16 CPUs, 32 GiB, 2-hour episode budget, no turn limit (Appendix D.2).
Assignments per cell: 400 for the two prompt-based methods and 200 for AutoEnvScaling in the
latest campaigns; the scored counts per cell are in `data/paper/fig_arms_data.json`.

## Commands

| Step | Command |
|---|---|
| Prompt, Prompt + Env Feedback | `autoenvscaling baselines generate --arm prompt\|feedback --model M --assignments A --out D` |
| Their valid rates and costs | `autoenvscaling baselines summarize --out D [--usd-per-mtok-in X --usd-per-mtok-out Y]` |
| Parent pool for the harness arm | `autoenvscaling selfplay seed --pool-dir P --source seed --no-copy <tasks>` |
| AutoEnvScaling | `autoenvscaling selfplay generate --parents-pool P --pool-dir D/pool --harness H --model M --assignments N` |
| Tables and figures | `python analysis/make_all.py` |

## Differences from the paper's runs

- The AutoEnvScaling arm draws its assignments from the parent pool (`SEED_TASKS`), while the
  prompt arms read `ASSIGNMENTS`; in the paper all arms shared one assignment list.
- The pipeline's `valid_rate` counts every validation check (build, reference solution, empty
  solution, contamination, rubric review), which is stricter than Table 2's definition, and it
  also calibrates valid tasks against the solver, which Table 2 did not do.

## Not in this release yet

- Figure 5's solver turns and pass rate: `autoenvscaling solver <task_dir> -n 4` measures one
  task; there is no batch runner over the admitted tasks of all three arms.
- Table 11 (self-check disabled vs enabled): the proposer has no switch to disable its
  self-check.
- The assignment suite and the generated environments behind the reported numbers.
