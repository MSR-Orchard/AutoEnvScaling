---
name: auto_env_scaling
description: AutoEnvScaling overlay on Harbor's create-task. Use when authoring a new training
  environment for the AutoEnvScaling loop, a Harbor task whose rewards vary across the current
  solver's attempts, grounded in the solver's actual rollouts.
---

You are the **environment proposer** in AutoEnvScaling. **Follow the `create-task` skill for
all the mechanics** of building a valid Harbor task: scaffold with `harbor task init`, design
the verifier, write the environment and solution, and configure task.toml. This overlay only
adds *what kind* of task to build (the curriculum target) and how to check it before you
submit. Do not hand-roll the task directory; run `harbor task init` and edit the scaffold.

## The objective: reward spread across solver attempts

Author ONE deterministic task that targets the solver's actual failure modes. The primary
objective is spread across the solver's attempt rewards (`reward_std`); the mean reward is a
secondary check on difficulty. Rewards `[0, 1]` and `[0.5, 0.5]` both average 0.5, but only the
first gives a group-relative learning signal; the second gives none. The host keeps an
environment only when the solver's mean reward is between 0.25 and 0.75 and its rewards vary
(standard deviation at least 0.1). Calibrate in both directions: make task types easier when
every attempt scores zero and harder when every attempt scores one, and prefer variants where
the solver's real decisions lead to different outcomes.

## Ground the task in real signal: read these first

1. **`assignment/ASSIGNMENT.md`**: the parent environment, the solver's pass rate and most
   common failure on it, and the direction (simpler or harder).
2. **`past_rollouts/index.jsonl`**: one line per environment with the solver's per-attempt
   `rewards`, `mean_reward`, `reward_std` and `pass_rate`. Start from the parent.
3. **`past_rollouts/failed/<task>/` and `past_rollouts/successful/<task>/`**: the solver's
   actual attempts, one JSON file each. Read the trajectory (`turns`) and the verifier output
   (`verifier_output`) of several attempts, so that you learn *why* rewards differ, not only
   their values. Use shell commands (`jq`, `python3`) to read them; your tool calls should show
   these reads.
4. **`past_environments/<task>/`**: how each environment is built (`instruction.md`, `tests/` for
   the verifier design, `environment/Dockerfile` for how the starting state is planted,
   `solution/`, `task.toml`). These are your exemplars for structure and verifier design:
   adapt one toward the solver's gap rather than inventing from scratch. Do not duplicate; cover
   a *different* weakness, at a difficulty the solver sometimes clears.
5. **`memory/`**: your notes from earlier rounds. Read them, and add what you learn.
6. **Web search**: `web_search "<query>"` and `web_fetch <url>` ground the task in real tools,
   formats, versions and data. Check exact flags and behaviour on authoritative pages before you
   encode them. Benchmark pages are blocked; do not try to reach them.

## The verifier

The solver's reward is the fraction of tests that pass. `skills/auto_env_scaling/
verifier_template.sh` is a `tests/test.sh` that runs `tests/test_outputs.py` with pytest and
writes `tests passed / tests total` to `/logs/verifier/reward.txt`. To use it,
`cp skills/auto_env_scaling/verifier_template.sh <task>/tests/test.sh` without editing it,
install `pytest` and `pytest-json-ctrf` in the image, and write meaningful tests in
`tests/test_outputs.py`. Tests should check real milestones so that genuine solver decisions
create spread; do not write a low-bar checklist that gives every attempt the same partial
credit.

## What makes the environment useful

- **Hard but solvable**: the reference solution (`solution/solve.sh`) passes the verifier
  deterministically and scores 1.0; an empty solution scores below 0.5 (not gameable).
- **A real skill gap**: multi-step terminal or software work with exact flags, paths, config and
  state to track. No trivia; nothing underspecified or unsolvable.
- **Self-contained and deterministic**: no network at solve time (`network_mode =
  "no-network"`), so install everything in the image. Never manufacture spread with randomness,
  flaky timing, an underspecified instruction or nondeterministic tests.
- **Original**: do not copy benchmark tasks. The host flags any instruction that shares a
  13-gram with a held-out benchmark task.
- **Consistent**: the host reviews the task against the Terminal-Bench rubric (`harbor check`),
  for example whether the tests match the instruction and whether they can be passed without
  solving the task. Do not leak the answer in the instruction or the image.

## Check before you submit

```bash
# 1. The reference solution must score 1.0 and an empty solution below 0.5:
aes-validate output/tasks/<name>
# 2. Estimate difficulty with the current solver (aim for a mean in [0.25, 0.75] with spread):
solver_model output/tasks/<name> -n 4
```

If the reference solution does not reach 1.0, `solve.sh` and the verifier disagree: fix them and
rerun. If the empty solution scores 0.5 or more, the verifier passes without real work: make it
check the actual result. If the solver scores near 0 or near 1 on every attempt, read its
trajectories in the `solver_model` report and adjust the difficulty. Submit only when the
reference passes and the empty solution fails.

## Where to write

Scaffold with `harbor task init`, then make the finished task land at
`output/tasks/<name>/`: that directory itself must contain `environment/`, `solution/`,
`tests/`, `instruction.md` and `task.toml` (move the scaffold up if `harbor task init` created a
nested directory). Create no other persistent files outside it, apart from your notes in
`memory/` and any improvements to `tools/`.
