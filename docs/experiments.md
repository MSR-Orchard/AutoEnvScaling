# Experiments

How each table and figure in the paper maps to a launcher, a configuration, released data and
an output, and the settings the paper states for each experiment. Figure and table numbers
refer to the arXiv version.

Launchers live in `cmd/<experiment>/run.sh` and call the package CLI (`autoenvscaling
<command> ...`). Each one echoes every command and appends all output to
`runs/<experiment>/<timestamp>/launcher.log` (it prints a `tail -f` line to follow it);
`DRY_RUN=1` prints the commands without running them. A step that this release cannot run
prints "not in this release yet" and the launcher carries on. Figures and tables are rendered
from `data/paper/` by `python analysis/make_all.py` into `outputs/`; this needs the `analysis`
extras and never calls a model or starts a sandbox.

Training runs pair a Miles job in bridge mode (`autoenvscaling train run --preset ...`) with the
self-play loop (`autoenvscaling selfplay run ...`), which runs every solver and proposer episode
and hands Miles finished RL groups through a shared bridge directory. `autoenvscaling selfplay
schedule` gives the number of Miles updates (`--num-rollout`) for a run.

Status legend: **from data** means the item renders from the released numbers; **runs** means
the launcher runs the experiment with this release's commands (it costs sandboxes, tokens and,
for training, GPUs); **partial** means some steps run and the others print "not in this release
yet"; **not released** means neither code nor underlying runs are part of this release.

## Paper items

| Item | Content | Launcher | Config | Data | Output | Status |
|---|---|---|---|---|---|---|
| Figure 1 | Results teaser (three panels) | `cmd/exp1_authoring`, `cmd/exp2_solver_rl`, `cmd/exp3_self_play` | | `fig_arms_data.json`, `fig_teaser_data.json`, `fig_rsi_data.json` | `outputs/figures/fig_teaser.pdf` | from data |
| Figures 2, 3 | System and method diagrams | | | | | illustrations |
| Table 1 | Comparison with prior environment construction | | | | | related work |
| Table 2 | Valid-environment rate and cost, 3 methods x 8 proposers | `cmd/exp1_authoring` | `cmd/exp1_authoring/config.toml` | `fig_arms_data.json`, `astra_results_20260917.json` | `outputs/tables/table2_threeway_rows.tex`, `tables.md` | from data; runs (differences below) |
| Figure 4 | Valid-environment rate across proposers | `cmd/exp1_authoring` | same | `fig_arms_data.json` | `outputs/figures/fig_arms.pdf` | from data; runs |
| Figure 5 | Files, solver turns and pass rate per method | `cmd/exp1_authoring` | same | `fig_harness_artifacts_data.json` | `outputs/figures/fig_harness_artifacts.pdf` | from data; measurement partial |
| Table 3 | Solver RL: AutoEnvScaling vs Tmax, TB2.1 avg@5 | `cmd/exp2_solver_rl`, `cmd/evals` | `cmd/exp2_solver_rl/config.toml` | `paper_tables.json` (transcribed) | `outputs/tables/tables.md` | runs (Tmax baseline differs, below) |
| Figure 6 | Self-play: TB2.1 and TB4.0 during training | `cmd/exp3_self_play`, `cmd/evals` | `cmd/exp3_self_play/config.toml` | `fig_rsi_data.json` | `outputs/figures/fig_rsi.pdf` | from data; runs |
| Figure 7 | Useful-group rate and rollout-collection time | `cmd/exp3_self_play` | same | `qwen36-rollout-metrics-v2.csv`, `fig_training_metrics_summary.json` | `outputs/figures/fig_training_metrics.pdf` | from data; runs, figure data from new runs partial |
| Figure 8 | Proposer traces from one failed run (PyTorch) | | | | | not released (annotated traces) |
| Table 4 | HiL-Bench, base and frontier models | `cmd/hilbench` | `cmd/hilbench/config.toml` | `paper_tables.json` (transcribed) | `outputs/tables/tables.md` | partial (metrics only) |
| Table 5 | HiL-Bench training of Qwen3.6-35B-A3B | `cmd/hilbench` | same | `paper_tables.json` (transcribed) | `outputs/tables/tables.md` | partial (derivation, fine-tuning, metrics) |
| Figure 9, Table 6 | Example generated HiL task and its blockers | `cmd/hilbench` | same | | | example of the HiL authoring prompt's output |
| Table 7 | Cross-domain task types | `cmd/cross_domain` | `cmd/cross_domain/config.toml` | | | descriptive |
| Table 8 | Cross-domain admission and rounds | `cmd/cross_domain` | same | `paper_tables.json` (transcribed); `fig_cross_domain_data.json` (companion figure) | `outputs/tables/tables.md`, `outputs/figures/fig_cross_domain.pdf` | from data; partial (terminal domain only) |
| Figures 10-17, Table 9 | Cross-domain examples and GPU timings | | | | | not released |
| Figures 18, 19 | Proposer workspace and image configuration | | | | | illustrations |
| Table 10 | Sandbox resources | all | package defaults (`autoenvscaling/config.py`) | `paper_tables.json` | | checked by tests |
| Table 11 | Self-check ablation | `cmd/exp1_authoring` | same as Table 2 | `paper_tables.json` (transcribed) | `outputs/tables/tables.md` | from data; the run is not in this release |
| Table 12 | Self-play training settings | `cmd/exp3_self_play` | `cmd/exp3_self_play/config.toml` | `paper_tables.json` | | checked by tests against the package defaults |
| Figure 20 | Proposer validity during self-play | `cmd/exp3_self_play` | same | `fig_proposer_validity_data.json` | `outputs/figures/fig_proposer_validity.pdf` | from data; runs, figure data from new runs partial |
| Figures 21, 22 | Cold-Start admission rate and episode depth | `cmd/cold_start` | `cmd/cold_start/config.toml` | `fig_cold_start_data.json` | `outputs/figures/fig_cold_start_admission.pdf`, `fig_cold_start_depth.pdf` | from data; runs |
| Table 13 | Cold-Start controls (acceptance, TB2.1) | `cmd/cold_start`, `cmd/evals` | same | `paper_tables.json` (transcribed) | `outputs/tables/tables.md` | runs |
| Figures 23, 24 | Authoring-style analyses | | | | | not released |

## Commands per experiment

| Experiment | Commands the launcher calls |
|---|---|
| `cmd/evals` | `eval preflight`, `eval run` (and `eval report`, `eval config` by hand) |
| `cmd/exp1_authoring` | `baselines generate --arm prompt\|feedback`, `baselines summarize`, `selfplay seed`, `selfplay generate` |
| `cmd/exp2_solver_rl` | `tmax`, `selfplay seed`, `selfplay generate`, `selfplay schedule`, `train plan/run --preset exp2-qwen3.5-9b\|exp2-qwen3.5-35b-a3b`, `selfplay run --mode static\|frontier [--no-solver-feedback]` |
| `cmd/exp3_self_play` | `selfplay seed`, `selfplay generate`, `selfplay schedule`, `train plan/run --preset shared-policy`, `selfplay run --mode shared [--no-proposer-training]`, `selfplay run --mode static` |
| `cmd/cold_start` | `selfplay seed`, `selfplay generate`, `export sft`, `train plan/run --preset cold-start` |
| `cmd/cross_domain` | `selfplay seed`, `selfplay generate` (terminal domain) |
| `cmd/hilbench` | `hilbench prompt`, `hilbench derive`, `train plan/run --preset cold-start`, `hilbench report` |

Every launcher ends with `python analysis/make_all.py`. `autoenvscaling admit <task_dir>`
(validation and calibration of one task) and `autoenvscaling solver <task_dir>` (the proposer's
`solver_model` tool) are available for single tasks.

## Settings shared by all experiments

**Coding agents** (Appendix D.2). Each proposer runs in a coding agent: Claude Code for
Claude Opus 5, Codex for the GPT models, and the Pi coding agent for all other models. All
solvers run in Pi. `cmd/common.sh` (`harness_for`) encodes this rule.

**Sandboxes** (Table 10). Proposer: 16 CPUs, 32 GiB, 2-hour budget per episode, no turn limit.
Solver: 4 CPUs, 8 GiB. The default backend is Modal, for both the proposer sandbox and Harbor
trials (`[sandbox] backend` and `harbor_env`).

**Validation** (Section 3.2, Appendix G.4). The environment builds; the reference solution gets
reward 1; an empty (do-nothing) solution gets less than 0.5; no instruction shares a 13-gram
with a held-out task (Terminal-Bench 2.1 and 4.0, sliding window, stride 1); `harbor check`
passes with the Terminal-Bench-derived rubric graded by the proposer's own model. The shipped
rubric omits `binary_reward` because tests give partial credit; the configured
`ValidationSettings.rubric_skip` removes contributor criteria. Its `task_toml_schema` guidance
recognizes Harbor's current network/OS and package fields, and treats `[metadata]` as free-form
descriptive data. These compatibility changes do not waive invalid runtime configuration or
the active verifier hygiene and execution-isolation criteria.

**Calibration** (Section 3.2, Appendix G.4). The current solver attempts the environment 4
times, expanded to 8 when the mean is within 0.15 of a band edge. The environment is kept if the
mean reward is in [0.25, 0.75] and the reward standard deviation is at least 0.1. Environments
below the band get one diagnostic rollout with reference-solution hints. Each assignment allows
up to 2 revisions, and revised environments repeat all checks.

**Rewards** (Section 3.3). Solver: fraction of the environment's tests passed. Proposer: -1 if
validation fails, -0.25 if validation passes but calibration fails, 1 if both pass.
Advantages are mean-centred within each role's groups (proposer groups share one assignment;
solver groups share one environment), without dividing by the standard deviation, and groups
whose rewards do not vary are dropped.

**Evaluation** (Section 4.2, Appendix G.3). Held-out Terminal-Bench 2.1 and Terminal-Bench 4.0,
avg@5, with Pi in 4 CPU / 8 GiB sandboxes; `cmd/evals`. Serving a trained checkpoint as an
OpenAI-compatible endpoint is not part of this package.

## Experiment 1: environment design as a terminal task (`cmd/exp1_authoring`)

Section 4.1, Table 2, Figures 4 and 5, Table 11.

- Methods: Prompt (files written from the task description, nothing executed); Prompt + Env
  Feedback (build and verifier logs returned for revision); AutoEnvScaling (the model works in
  the sandbox with a coding agent).
- Proposers: Claude Opus 5, GPT-6-astra, GPT-5.6-sol, GPT-5.6-terra, GPT-5.6-luna,
  Gemini-3.8-Flash, Grok 4.6, DeepSeek-V4-Flash.
- Valid environment: builds, reference solution passes, empty solution fails. Rates exclude
  recorded infrastructure failures and provider refusals; intervals are 95% Wilson.
- Cost: USD per valid environment; Table 2 prints the AutoEnvScaling-to-Prompt ratio.
- Figure 5: arm means over four author models (Claude Opus 5, GPT-6-astra, Grok 4.6,
  DeepSeek-V4-Flash) of shipped data files, solver turns and solver pass rate.
- Table 11: self-check disabled vs enabled for five proposers, with solver calibration on.
- Differences in this release: the AutoEnvScaling arm (`selfplay generate`) draws assignments
  from a parent pool while the prompt arms read an assignment file; its `valid_rate` counts all
  validation checks (including contamination and rubric review) and valid tasks are also
  calibrated. Not in this release yet: the batch measurement behind Figure 5 and the self-check
  switch behind Table 11.

## Experiment 2: curricula vs static pools (`cmd/exp2_solver_rl`)

Section 4.2, Table 3.

- Solvers: Qwen3.5-9B and Qwen3.5-35B-A3B, trained with RL (Miles presets `exp2-qwen3.5-9b`,
  `exp2-qwen3.5-35b-a3b`).
- Proposers: GPT-5.6-sol and Claude Opus 5, weights fixed, improving only through their harness
  (`selfplay run --mode frontier`).
- Baseline: in the paper, the Tmax pool regenerated with the same proposer following the Tmax
  recipe. This release builds one baseline pool per solver from the released TMax-15K tasks
  (`autoenvscaling tmax`, then `selfplay run --mode static`); Tmax regeneration is not included.
- Ablation (Qwen3.5-35B-A3B): past solver trajectories removed from the proposer's workspace
  (`--no-solver-feedback`).
- Group size 16, 16 environments per batch, maximum sequence length 128k tokens.
- Evaluation: held-out Terminal-Bench 2.1, avg@5, at matched training steps.
- Not stated in the paper: the number of solver updates (the launcher uses the presets' 300) and
  how the initial AutoEnvScaling pool was built (the launcher generates 256 environments from
  Tmax parents).

## Experiment 3: recursive self-improvement (`cmd/exp3_self_play`)

Section 4.3, Appendix G; Figures 6, 7 and 20.

- Qwen3.6-35B-A3B as proposer and solver, starting from the Cold-Start checkpoint (Miles preset
  `shared-policy`, 288 solver updates plus 17 proposer updates).
- Runs: AutoEnvScaling (`selfplay run --mode shared`); AutoEnvScaling without proposer training
  (`--no-proposer-training`: shared weights, proposer loss removed); Tmax (`--mode static`).
- Table 12: learning rate 1e-6; 16 environments x 16 trajectories per solver update; 128
  proposer trajectories per assignment; proposer generation every 16 solver updates; active pool
  of 256 tasks reviewed every 16 updates; DPPO total-variation threshold 0.1; KL and entropy
  coefficients 0.
- Rollouts: groups without reward variance are discarded and more are collected until 16 are
  filled; unfinished trajectories are carried across updates and resume with the updated policy.
  Trained with Miles on 4-8 B200 GPUs.
- Evaluation: TB2.1 and TB4.0 avg@5 every 32 updates, through 288 updates (160 for the ablation,
  which collapsed). Rollout-collection statistics use updates 0-162 with matched hardware and
  rollout concurrency.
- Not in this release yet: converting a run's `metrics.jsonl` into the data files behind
  Figures 7 and 20. The initial pool is one generation round by the Cold-Start policy; the
  paper does not state how its initial pool was built.

## Cold-Start (`cmd/cold_start`)

Section 4.3, Appendix H; Figures 21, 22 and Table 13.

- Fine-tune Qwen3.6-35B-A3B on all 600 admitted proposer trajectories collected from Claude
  Opus 5, DeepSeek-V4-Flash and Kimi-K3 (`selfplay generate` per teacher, `export sft`,
  `train run --preset cold-start`).
- Figures 21 and 22: a shared 136-assignment suite (frontier reference rows from earlier suites).
- Table 13: 60 paired assignments for acceptance; Terminal-Bench 2.1 with five attempts on each
  of 89 tasks for avg@5 and pass@5.
- The `cold-start` preset is slime's standard SFT recipe for Qwen3.5-35B-A3B (3 epochs,
  batch 128, lr 1e-5 cosine to 1e-6, 10% warmup), as used in the paper.

## Cross-domain generation (`cmd/cross_domain`)

Appendix C; Tables 7 and 8.

- GPT-6-astra and GPT-5.6-sol in Codex; Claude Opus 5 in Claude Code.
- Seven domains: terminal, software engineering, tool calling, browser use, computer use, GPU
  kernels (one NVIDIA T4), professional workflows.
- Up to two revision rounds after the initial attempt.
- Admission: initializes; reference solution earns full reward; do-nothing agent earns zero;
  reset reproduces the initial state and a passing reference run. Runs ended by external
  execution failures are excluded.
- With this release only the terminal domain runs (`selfplay generate`, standard admission). Not
  in this release yet: the six other domains' assignments and runtimes, and the
  do-nothing-earns-zero and reset checks.

## HiL-Bench (`cmd/hilbench`)

Section 4.5, Appendix B; Tables 4, 5 and 6.

- Claude Opus 5 generates 500 software engineering tasks in HiL-Bench's format with
  AutoEnvScaling: an underspecified instruction, a blocker registry that `ask_human` answers
  from, and tests that depend on the withheld details. The `full_info` and `baseline` variants
  are derived without further model calls (`hilbench derive`).
- Cold start: fine-tune Qwen3.6-35B-A3B on 75 Claude Opus 5 trajectories that ask about missing
  requirements. RL for up to 150 steps with a shaped clarification reward based on Ask-F1,
  on the 100 official training tasks or on the 500 generated tasks.
- Evaluation: held-out HiL-Bench tasks, disjoint from the official training set, in the
  `ask_human` setting: task success, Ask-F1 and ask rate (`hilbench report`).
- **Judge.** The judge that matches questions to blockers is DeepSeek-V4-Flash
  (`deepseek/deepseek-v4-flash-0731`, through OpenRouter), not HiL-Bench's official
  Llama-3.3-70B. Ask-F1 values are therefore not comparable with the HiL-Bench leaderboard;
  task success does not depend on the judge.
- **Tool surface.** Pi has no MCP client, so the agent reaches HiL-Bench's ask-human server
  through an `ask_human "<question>"` shell command, as HiL-Bench does for its text-to-SQL
  tasks, instead of the first-class MCP tool used for software engineering tasks.
- Not in this release yet: task generation with the HiL prompt (the proposer pipeline has no HiL
  setting), a Harbor runner for HiL-Bench episodes with the ask-human sidecar (and therefore RL
  with the shaped reward and the held-out runs), and exporting asking trajectories as
  fine-tuning rows.

## What this release cannot reproduce yet

Each launcher prints these steps as "not in this release yet".

- Experiment 1: the batch measurement of solver turns and pass rate behind Figure 5, and the
  self-check ablation behind Table 11.
- Experiment 2: Tmax pools regenerated with each proposer (the baseline uses TMax-15K).
- Experiment 3: conversion of run logs into the data behind Figures 7 and 20.
- Cross-domain: all domains except terminal, and the appendix's admission checks.
- HiL-Bench: HiL task generation, the Harbor episode runner with the ask-human sidecar, RL with
  the shaped reward, and cold-start row export. Task derivation, registry checks, the
  `ask_human` command and all metrics are released.
- Everywhere: serving trained checkpoints for evaluation, and the assignment suites, generated
  environments, trajectories and checkpoints behind the reported numbers. Tables 3, 4, 5, 8, 11
  and 13 are transcribed from the paper in `data/paper/`.
- Settings the paper does not state: the matched number of solver updates for Experiment 2, the
  Cold-Start fine-tuning hyperparameters, the agent time limit and decoding parameters for
  Terminal-Bench evaluation, and the HiL-Bench held-out split.
