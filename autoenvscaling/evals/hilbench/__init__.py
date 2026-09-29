"""HiL-Bench pieces for Section 4.5: generating HiL-style training tasks and scoring episodes.

Generation (training tasks):
    generation   the authoring prompt and per-episode domain hints
    registry     blocker registry schema, structural checks, and a leak pre-filter
    variants     derive full_info / ask_human / baseline variants from one authored task
    ask_human    the ask_human tool: HiL-Bench's server (pinned), its judge settings, and the
                 ``ask_human`` shell command the Pi agent calls (``ask_human_client.py``)
    pi_agent     Pi with the ask_human command installed

Evaluation:
    metrics      task success, pass@k, Ask-F1, ask rate, and the shaped clarification reward

The judge that matches questions to blockers is DeepSeek-V4-Flash in the paper, not HiL-Bench's
official Llama-3.3-70B, so Ask-F1 is not comparable with the HiL-Bench leaderboard.
"""
