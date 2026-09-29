"""The HiL task-generation setting used to build the 500 generated training tasks (Section 4.5).

A setting replaces the proposer's default authoring prompt. The HiL prompt asks the proposer to
write a complete Harbor task first, then pick three to five facts that the tests depend on, that
cannot be recovered from the environment, and that are choices rather than deductions; remove
them from the instruction; and record them in ``ask-human-data/blocker_registry.json``. A domain
hint is appended per episode because episodes started from an empty memory otherwise collapse
onto one subject.

The prompt text in ``generation_instruction.md`` is kept as it was used, except that the output
location is a parameter and an infrastructure-specific reference was removed. After an episode,
``variants.derive_variants`` builds the full_info, ask_human and baseline variants.
"""

from __future__ import annotations

from importlib import resources

DEFAULT_OUTPUT_DIR = "output/env_001"

_SCENARIO = (
    "Author this task in the following domain: {domain}.\n\n"
    "This constrains the SUBJECT MATTER only. Every requirement of the authoring contract above "
    "still applies unchanged -- in particular the blocker rules and the no-network, stdlib-only "
    "environment.\n"
)


def _read(name: str) -> str:
    return resources.files(__package__).joinpath(name).read_text()


def generation_instruction(output_dir: str = DEFAULT_OUTPUT_DIR) -> str:
    """The authoring prompt, with the task's output directory filled in (keep it under output/)."""
    return _read("generation_instruction.md").replace("{output_dir}", output_dir.rstrip("/"))


def domains() -> list[str]:
    lines = (line.strip() for line in _read("domains.txt").splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def domain_for_episode(index: int, offset: int = 0) -> str:
    """The domain hint for episode ``index`` of a batch (round-robin over the domain list)."""
    items = domains()
    return items[(index + offset) % len(items)]


def scenario(domain: str) -> str:
    """Text appended to the authoring prompt to steer the subject matter of one episode."""
    return _SCENARIO.format(domain=domain)


def episode_prompt(index: int, *, offset: int = 0, output_dir: str = DEFAULT_OUTPUT_DIR) -> str:
    """Authoring prompt plus domain hint for one episode, in the order the pipeline joins them."""
    return generation_instruction(output_dir) + "\n\n" + scenario(domain_for_episode(index, offset))
