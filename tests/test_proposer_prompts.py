"""The episode prompt and the shipped skills."""

import re
from pathlib import Path

from autoenvscaling.config import Settings
from autoenvscaling.proposer.prompts import build_prompt
from autoenvscaling.trajectories.schema import Assignment

REPO = Path(__file__).resolve().parents[1]
ASSIGNMENT = Assignment(assignment_id="asg-1", parent_task_id="torch-ckpt", solver_pass_rate=0.0,
                        common_failure="failing tests: test_weights", direction="simpler")
SKILLS = [("auto_env_scaling", "overlay text"), ("create-task", "---\nname: create-task\n---\n"),
          ("harbor_domains", "not inlined")]


def test_prompt_starts_with_the_two_skills_in_order():
    prompt = build_prompt(ASSIGNMENT, skills=SKILLS, settings=Settings())
    assert prompt.startswith("# Skill: create-task")
    assert prompt.index("# Skill: create-task") < prompt.index("# Skill: auto_env_scaling") \
        < prompt.index("# Your task")
    assert "not inlined" not in prompt


def test_prompt_carries_assignment_settings_and_layout():
    settings = Settings()
    settings.calibration.band_low = 0.3
    prompt = build_prompt(ASSIGNMENT, skills=SKILLS, settings=settings)
    for text in ("`torch-ckpt`", "failing tests: test_weights", "Direction: **simpler**",
                 "between 0.3 and 0.75", "13-gram", "harbor check", "past_rollouts/failed/torch-ckpt/",
                 "output/tasks/<task-name>/", "aes-validate", "solver_model", "web_search",
                 "budget of 2 hours"):
        assert text in prompt, text
    assert "Revision" not in prompt


def test_optional_tools_and_revision_feedback():
    prompt = build_prompt(ASSIGNMENT, skills=SKILLS, settings=Settings(), web_tools=False,
                          solver_tool=False, feedback="rubric_review failed: tests leak answer",
                          revision=1, previous_task="my-task")
    assert "web_search" not in prompt
    assert "solver_model" not in prompt
    assert prompt.rstrip().endswith("rubric_review failed: tests leak answer")
    assert "# Revision 1 of 2" in prompt and "output/tasks/my-task/" in prompt


def test_no_research_code_identifiers_or_em_dashes():
    texts = [build_prompt(ASSIGNMENT, skills=SKILLS, settings=Settings())]
    texts += [p.read_text() for p in (REPO / "skills").rglob("*.md")]
    texts += [p.read_text() for p in (REPO / "docker" / "proposer").rglob("*")
              if p.is_file() and "__pycache__" not in p.parts]
    texts += [p.read_text() for p in (REPO / "autoenvscaling" / "proposer").rglob("*.py")]
    forbidden = re.compile(r"qwen3\.5-9b|terminal-rsi|trsi|aks.modal|azurecr|trycloudflare|"
                           r"frp|/mnt/pvc|/data/users|copilot|t-simonyu|msr-spare|dobby",
                           re.IGNORECASE)
    for text in texts:
        assert not forbidden.search(text), forbidden.search(text).group(0)
        assert "\u2014" not in text  # no em dashes


def test_auto_env_scaling_skill_matches_the_release_layout():
    skill = (REPO / "skills" / "auto_env_scaling" / "SKILL.md").read_text()
    assert skill.startswith("---\nname: auto_env_scaling\n")
    for text in ("past_rollouts/index.jsonl", "past_rollouts/failed/<task>/", "aes-validate",
                 "solver_model", "output/tasks/<name>/", "verifier_template.sh"):
        assert text in skill, text
