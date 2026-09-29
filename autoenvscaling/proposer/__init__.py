"""The proposer: a coding agent that builds new training environments (Section 3.1 of the paper).

Modules:
    assignment    assignments (parent environment, pass rate, common failure, direction)
    workspace     the proposer's workspace layout and how the host builds it
    prompts       the episode prompt (skills, instruction, host feedback)
    harnesses     Claude Code, Codex and Pi: launch commands and native-log parsers
    episode       run_episode: one episode in the proposer sandbox
    solver_tool   the in-sandbox solver_model tool
"""
