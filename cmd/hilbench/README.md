# HiL-Bench: learning when to ask for help

Section 4.5 and Appendix B; Tables 4, 5 and 6.

HiL-Bench tasks come in three settings over one environment: `full_info` (complete
instruction), `ask_human` (incomplete instruction plus an `ask_human` tool) and `baseline`
(incomplete instruction, no tool). The official release has only 100 software engineering
training tasks, so Claude Opus 5 generates 500 more with AutoEnvScaling. Qwen3.6-35B-A3B is
first fine-tuned on 75 Claude Opus 5 trajectories that ask about missing requirements, then
trained with RL for up to 150 steps with a shaped clarification reward, on the 100 official
tasks or on the 500 generated ones, and evaluated on held-out HiL-Bench tasks that are disjoint
from the official training set.

```bash
HIL_BENCH_ROOT=<checkout> HIL_TASKS="authored/*" EPISODES="runs/.../episodes.jsonl" \
  cmd/hilbench/run.sh
DRY_RUN=1 cmd/hilbench/run.sh
```

## Commands

| Step | Command |
|---|---|
| Authoring prompt for episode i | `autoenvscaling hilbench prompt --index i` |
| Variants of authored tasks | `autoenvscaling hilbench derive [--out-root D] <task dirs>` |
| Cold-start fine-tuning (given SFT rows) | `autoenvscaling train plan\|run --preset cold-start --prompt-data rows.jsonl ...` |
| Task success, Ask-F1, ask rate | `autoenvscaling hilbench report <episodes.jsonl> ... [--exclude-stop pi_timeout]` |

## Generating tasks (`autoenvscaling.evals.hilbench`)

- `generation`: the authoring prompt. The proposer writes a complete Harbor task first, then
  removes three to five facts that the tests depend on, that cannot be recovered from the
  environment, and that are choices rather than deductions, and records them in
  `ask-human-data/blocker_registry.json`. A domain hint is appended per episode.
- `registry`: the registry schema, structural checks (3-5 blockers, required fields, at least
  three example questions, no description that restates its answer) and a leak pre-filter.
- `variants`: derives `full_info` (instruction plus a `## BLOCKER DETAILS` appendix),
  `ask_human` and `baseline` (byte-identical instructions) from one authored task, in
  HiL-Bench's layout. Answers never enter a variant the solver can read.
- `ask_human`: the tool. Questions are answered by HiL-Bench's own ask-human server (taken
  from a checkout at commit `352d14c`, hash-checked, not redistributed) with a judge model that
  matches questions to blockers. The Pi agent has no MCP client, so it gets an `ask_human
  "<question>"` shell command that calls the same server (`ask_human_client.py`) plus one
  "Available Commands" block in the instruction; `HilPiAgent` installs both.

## Scoring (`autoenvscaling.evals.hilbench.metrics`)

- **Task success**: fraction of episodes whose reward reaches 1 (pass@k over tasks is also
  reported). Infrastructure failures and software engineering episodes whose verifier collected
  no tests are excluded, never scored as 0.
- **Ask-F1**: mean per-episode F1 from the ask-human server (precision = blockers resolved /
  questions, recall = blockers resolved / blockers). An episode with no question counts as 0.
- **Ask rate**: fraction of ask_human episodes with at least one question.
- **Shaped reward** (training): +0.3 per question that targets a blocker, -0.1 per irrelevant or
  repeated question, plus the fraction of blockers discovered if at least one was; no
  task-success term. A "can't answer" reply (judge failure) is not charged.

## Judge: not leaderboard-comparable

HiL-Bench's official judge is Llama-3.3-70B. All Ask-F1 numbers in the paper use
DeepSeek-V4-Flash (`deepseek/deepseek-v4-flash-0731` through OpenRouter) as the judge, so Ask-F1
is **not comparable with the HiL-Bench leaderboard**. Task success does not depend on the judge.
A second deviation: for software engineering tasks the official `ask_human` is a first-class
MCP tool, while Pi reaches the same server through the shell command.

## Not in this release yet

- Task generation with the HiL prompt: the proposer pipeline (`selfplay generate`) uses the
  standard authoring prompt and has no HiL setting.
- A Harbor runner for HiL-Bench episodes (the ask-human sidecar next to the solver's container),
  and with it RL with the shaped reward and the held-out evaluation runs. The research runs used
  a separate harness that started the sidecar inside the task sandbox.
- Exporting asking trajectories from HiL episodes as fine-tuning rows. Given such rows, the
  launcher fine-tunes with the `cold-start` preset (placeholder hyperparameters).
- The 75 cold-start trajectories, the generated tasks and the held-out split.
