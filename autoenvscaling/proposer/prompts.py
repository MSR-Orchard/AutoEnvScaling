"""The proposer's episode prompt.

Following Appendix "Proposer Skills" of the paper, the prompt begins with two skills (Harbor's
``create-task`` and our ``auto_env_scaling``), followed by the episode instruction: the
assignment, the workspace, what the host will check, the tools, and where to write the task.
When the episode revises an earlier submission, the host's feedback comes last.

Ported from the research code's generation instruction. Removed: the hard-wired solver names,
the static reference pool and ``provenance`` weighting, the verifier-kind cookbook and taxonomy
requirements, the multi-environment ``env_NNN`` output layout, the A/B prompt variants, and the
batch/live validation modes (in-sandbox validation is always available here, as in the paper).
"""

from __future__ import annotations

from collections.abc import Sequence

from autoenvscaling.config import Settings
from autoenvscaling.proposer import workspace as ws
from autoenvscaling.proposer.assignment import render_assignment
from autoenvscaling.trajectories.schema import Assignment

# Order in which skills open the prompt (other staged skills are listed, not inlined).
PROMPT_SKILLS = ("create-task", "auto_env_scaling")

WORKSPACE_DESCRIPTION = f"""\
Your working directory is the workspace:

- `{ws.ASSIGNMENT}/`: this episode's assignment (ASSIGNMENT.md, assignment.json). Read-only.
- `{ws.SKILLS}/`: the skills above plus `harbor_domains/` for non-terminal domains. Read-only.
- `{ws.PAST_ROLLOUTS}/index.jsonl`: one line per environment in the training pool with the
  solver's per-attempt `rewards`, `mean_reward`, `reward_std` and `pass_rate`.
- `{ws.PAST_ROLLOUTS}/{ws.FAILED}/<task>/` and `{ws.PAST_ROLLOUTS}/{ws.SUCCESSFUL}/<task>/`: the
  solver's attempts, one JSON file each, with the full trajectory (`turns`) and the verifier
  output (`verifier_output`).
- `{ws.PAST_ENVIRONMENTS}/<task>/`: previously generated environments as complete Harbor tasks
  (instruction, environment, solution, tests).
- `{ws.MEMORY}/`: your own notes. They persist across rounds; read them first and keep them useful.
- `{ws.TOOLS}/`: your tools (on PATH). They persist across rounds; you may improve them.
- `{ws.FEEDBACK}/`: the host's admission results when you are revising a submission. Read-only.
- `{ws.LOGS}/`: reports written by the checking tools.
- `{ws.OUTPUT_TASKS}/`: where your new environment goes.

The rollouts and past environments are refreshed from the training pool each round. The
admission checks run on the host, outside this sandbox, and cannot be changed from here.
"""


def _skill_section(name: str, text: str) -> str:
    return f"# Skill: {name}\n\n{text.strip()}\n"


def order_skills(skills: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    """Keep only the skills the prompt inlines, in the paper's order."""
    by_name = dict(skills)
    return [(name, by_name[name]) for name in PROMPT_SKILLS if name in by_name]


def episode_instruction(assignment: Assignment, settings: Settings, *,
                        web_tools: bool = True, solver_tool: bool = True) -> str:
    val, cal = settings.validation, settings.calibration
    hours = settings.sandbox.proposer_timeout_s / 3600
    tools = [
        ("- `aes-validate <task_dir>`: builds the environment in a child container, runs your "
         f"reference solution (it must score at least {val.reference_min_reward:g}) and an empty "
         f"solution (it must score below {val.empty_max_reward:g}). Run it before you submit."),
    ]
    if solver_tool:
        tools.append(
            "- `solver_model <task_dir> [-n N]`: runs the current solver N times on your "
            "environment and reports each attempt's reward, trajectory and verifier output. Use "
            "it to estimate difficulty before you submit.")
    if web_tools:
        tools.append(
            '- `web_search "<query>"` and `web_fetch <url>`: search the web for real problems, '
            "documentation and data, then read the pages you rely on. Benchmark pages are blocked.")
    tools.append("- The `harbor` CLI (`harbor task init`, `harbor run`, `harbor check`) and a "
                 "normal Linux shell.")
    tool_text = "\n".join(tools)
    return f"""\
# Your task

You are the environment proposer in AutoEnvScaling. In this episode you build ONE new training
environment for the solver, as a complete Harbor task: an instruction, an execution environment,
a reference solution, and tests. The solver will see only the instruction and work inside the
environment's container; the reference solution and tests stay hidden from it. Its reward is the
fraction of the tests it passes.

## Assignment

{render_assignment(assignment)}
## Workspace

{WORKSPACE_DESCRIPTION}
## How to work

1. Read `{ws.MEMORY}/`, then the solver's attempts on the parent environment under
   `{ws.PAST_ROLLOUTS}/{ws.FAILED}/{assignment.parent_task_id}/` and
   `{ws.PAST_ROLLOUTS}/{ws.SUCCESSFUL}/{assignment.parent_task_id}/`. Read the trajectories and
   verifier outputs to find what went wrong, or what the solver already does reliably. Look at
   how the parent is built in `{ws.PAST_ENVIRONMENTS}/{assignment.parent_task_id}/`.
2. Decide what to build. Follow the assignment's direction: a simpler variant of the same
   weakness when the solver keeps failing, or a harder one with a longer horizon or an
   additional skill when the solver succeeds. Cover a weakness, not a copy of the parent.
3. Build the environment with the `create-task` skill: scaffold with `harbor task init`, write
   the files, and build and test the environment in child containers. Revise it until the tests
   behave as intended.
4. Check it with the tools below, record what you learned in `{ws.MEMORY}/`, and finish.

You decide what to build and when the environment is finished. There is no turn limit; the
episode has a budget of {hours:g} hours.

## What the host checks after you finish

Validation: the environment builds; the reference solution receives full reward (at least
{val.reference_min_reward:g}); an empty solution receives less than {val.empty_max_reward:g}; no
instruction shares a {val.ngram_n}-gram with a held-out benchmark task; and the task passes
Harbor's task review (`harbor check`) against the Terminal-Bench rubric, which asks, for example,
whether the tests match the instruction and whether they can be passed without solving the task.

Calibration: the current solver attempts the environment several times. It is kept only if the
mean reward is between {cal.band_low:g} and {cal.band_high:g} and the rewards vary across attempts
(standard deviation at least {cal.min_reward_std:g}), so the solver sometimes solves it and
sometimes does not. Rewards [0, 1] have useful spread; [0.5, 0.5] have none.

Environments that fail either check come back to you with the host's feedback for revision.

## Requirements

- Deterministic and self-contained. The image build has network access, so install everything
  the task and the tests need in `environment/Dockerfile`. Solving and grading run without
  network (`network_mode = "no-network"`): nothing may be downloaded then, by the solver or by
  the tests. Never create reward variance with randomness, timing, or an underspecified
  instruction.
- Partial credit through tests that check meaningful milestones, not a checklist every attempt
  clears to the same score.
- Original. Do not copy tasks from Terminal-Bench or other benchmarks.
- In the Dockerfile, create files with COPY from the build context rather than shell heredocs.

## Tools

{tool_text}

## Where to write

Write the finished task to `{ws.OUTPUT_TASKS}/<task-name>/`, with `task.toml`,
`instruction.md`, `environment/Dockerfile`, `solution/solve.sh` and `tests/test.sh` directly
inside it. If `harbor task init` creates a nested directory, move its contents up. Produce
exactly one task. Do not modify `{ws.ASSIGNMENT}/`, `{ws.SKILLS}/`, `{ws.PAST_ROLLOUTS}/` or
`{ws.PAST_ENVIRONMENTS}/`.
"""


def revision_section(feedback: str, revision: int, max_revisions: int,
                     previous_task: str | None = None) -> str:
    where = (f"Your previous submission is at `{ws.OUTPUT_TASKS}/{previous_task}/`; revise it in "
             "place and keep it the only task under that directory.\n\n"
             if previous_task else "")
    return (f"# Revision {revision} of {max_revisions}\n\n"
            "The host ran its admission checks on your previous submission. Their results "
            f"follow (also saved in `{ws.FEEDBACK}/FEEDBACK.md`). Fix what failed, check the "
            "environment again, and submit.\n\n"
            f"{where}{feedback.strip()}\n")


def build_prompt(assignment: Assignment, *, skills: Sequence[tuple[str, str]],
                 settings: Settings, feedback: str | None = None, revision: int = 0,
                 previous_task: str | None = None, web_tools: bool = True,
                 solver_tool: bool = True) -> str:
    """The full prompt: skills first, then the episode instruction, then any host feedback.

    The result always starts with a Markdown heading, never with ``-``, because it is passed to
    the harness CLIs as a command-line argument.
    """
    parts = [_skill_section(name, text) for name, text in order_skills(skills)]
    parts.append(episode_instruction(assignment, settings, web_tools=web_tools,
                                     solver_tool=solver_tool))
    if feedback:
        parts.append(revision_section(feedback, revision or 1, settings.calibration.max_revisions,
                                      previous_task))
    return "\n---\n\n".join(parts)
