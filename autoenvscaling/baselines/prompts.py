"""Prompts for the Section 4.1 baselines (Prompt, and Prompt + Env Feedback).

The research runs built these prompts from files that are not in the source archive
(``neutral_prompt.py``, ``flat_parse.py`` and ``run_feedback_lane.py``). What survives, in
``exp1_authoring/refresh_worker.py`` and ``refresh_common.py``, fixes their structure, which we
keep: a tool-free notice, the task contract and the assignment, the file emission format, a
continuation request when the model stops before emitting files, and a repair prompt that
contains the original prompt, the model's previous answer and the environment's feedback. The
wording below is a reconstruction.

``extra_guidance`` lets a run include the same authoring guidance the harness proposer receives
(for example Harbor's ``create-task`` skill), so the arms differ only in how they generate.
"""

from __future__ import annotations

from autoenvscaling.baselines.emission import FORMAT_INSTRUCTIONS
from autoenvscaling.results import BUILD, EMPTY, REFERENCE, ValidationReport

NO_TOOLS_NOTICE = ("You have no tools in this conversation: you cannot run commands, build images, "
                   "or test anything. Write every file of the task in full in your answer.")

CONTRACT = """Design one training environment for a terminal agent, as a complete Harbor task
(https://www.harborframework.com/docs/tasks). The task directory contains:

- `task.toml`: the Harbor task configuration ([metadata], [verifier] timeout_sec,
  [agent] timeout_sec, [environment] build_timeout_sec, cpus, memory_mb).
- `instruction.md`: what the solving agent sees. State everything the tests check, and use
  absolute paths.
- `environment/Dockerfile`, plus any files it copies from `environment/`: the starting state of
  the container. Install every tool, package and fixture the task and the tests need at build
  time; the agent and the tests may run without network access.
- `solution/solve.sh`: a reference solution that makes every test pass. It is hidden from the
  solving agent.
- `tests/`: the verifier, hidden from the solving agent. {verifier_contract}

The environment is valid only if its image builds, the reference solution receives full reward
(1.0), and an empty solution (the untouched container) receives less than 0.5."""

MANAGED_VERIFIER_CONTRACT = """Write the tests as pytest functions in `tests/test_outputs.py`,
  one test per independent requirement. The host supplies `tests/test.sh`, which runs them and
  writes the fraction of tests that pass as the reward, so the Dockerfile must install `pytest`
  and `pytest-json-ctrf` (and `python3` if the base image lacks it)."""

OWN_VERIFIER_CONTRACT = """`tests/test.sh` runs the tests and writes the fraction of tests that
  pass (a number in [0, 1]) to `/logs/verifier/reward.txt`."""

CONTINUATION = ("You stopped before emitting the task files. Emit them NOW, in full, in the "
                "required format. Output ONLY the files; do not restate the plan.")


def build_prompt(description: str, *, managed_verifier: bool = True,
                 extra_guidance: str = "") -> str:
    contract = CONTRACT.format(verifier_contract=MANAGED_VERIFIER_CONTRACT if managed_verifier
                               else OWN_VERIFIER_CONTRACT)
    parts = [NO_TOOLS_NOTICE, contract]
    if extra_guidance.strip():
        parts.append("# Authoring guidance\n\n" + extra_guidance.strip())
    parts.append("# Task to build\n\n" + description.strip())
    parts.append("# Output format\n\n" + FORMAT_INSTRUCTIONS)
    return "\n\n".join(parts) + "\n"


def continuation_prompt(prompt: str, reply: str) -> str:
    """The follow-up the Prompt arm sends once when a finished reply contains no files."""
    return f"{prompt}\n\n{reply}\n\n---\n\n{CONTINUATION}"


def repair_prompt(base: str, previous_reply: str, feedback: str) -> str:
    """One Prompt + Env Feedback round: the original prompt, the last answer and its feedback."""
    return (f"{base}\n\n---\n\n# Your previous answer\n\n{previous_reply.strip()}\n\n---\n\n"
            f"# Feedback from running your environment\n\n{feedback.strip()}\n\n---\n\n"
            "Fix the environment. Emit the corrected files in the same format; files you do not "
            "emit keep their previous content.")


def env_feedback(report: ValidationReport, log_chars: int = 12_000) -> str:
    """Build and verifier logs for the first failed check (Exp1 feedback lane wording)."""
    by_name = {c.name: c for c in report.checks}
    build = by_name.get(BUILD)
    if build is not None and not build.passed:
        if build.data.get("stage") == "structure":
            return (f"VALIDATION FAILED at the structural check.\n{build.detail}\n\nThe task "
                    "must be a canonical Harbor task: task.toml, instruction.md, "
                    "environment/Dockerfile, solution/solve.sh, tests/.")
        log = str(build.data.get("build_log") or build.detail)[-log_chars:]
        return (f"VALIDATION FAILED while building the Docker image. Build log tail:\n\n{log}\n\n"
                "Fix the Dockerfile so the image builds. Keep the task itself unchanged.")
    ref = by_name.get(REFERENCE)
    if ref is not None and not ref.passed:
        log = str(ref.data.get("verifier_output") or "")[-log_chars:]
        return ("VALIDATION FAILED: your own reference solution did not satisfy your own tests.\n"
                f"{ref.detail}\n\nVerifier log tail:\n\n{log}\n\nEither solution/solve.sh does not "
                "accomplish the task, or tests/ checks something the instruction never asked for. "
                "Make them agree.")
    empty = by_name.get(EMPTY)
    if empty is not None and not empty.passed:
        log = str(empty.data.get("verifier_output") or "")[-log_chars:]
        return ("VALIDATION FAILED: a do-nothing solution scored highly enough to pass.\n"
                f"{empty.detail}\n\nVerifier log tail:\n\n{log}\n\nThe tests do not "
                "distinguish real work from an empty submission. Make them check the task's "
                "actual effect on the environment.")
    failed = [c for c in report.checks if not c.passed]
    detail = failed[0].detail if failed else "unknown failure"
    return f"VALIDATION FAILED: {detail}. Fix the environment."
