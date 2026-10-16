# Release status (private preview, 29/09/2026)

This page records what the preview has been checked against and what still needs a live run or
an author decision before the public release.

## Checked

* Unit tests for every module (`pytest -q`), with fakes for Modal, Harbor trials, model
  endpoints, and Miles.
* Every paper figure re-renders from `data/paper/` with the same content as the paper; Table 2's
  LaTeX rows match the paper byte for byte; the transcribed Tables 3, 4, 5, 8, 11, and 13 match
  number for number.
* The Miles patch applies cleanly to upstream `radixark/miles` at `5e392cf`.
* The Terminal-Bench datasets resolve in Harbor's registry: `terminal-bench/terminal-bench-2-1`
  (Terminal-Bench 2.1, 89 tasks) and `terminal-bench/terminal-bench@4.0.0` (66 tasks).
* No internal paths, IP addresses, private registries or endpoints, personal identifiers, or
  credentials in the tree.

## Not yet run on real infrastructure

* The Modal sandbox backend and the proposer image (`docker/proposer/`), including the CLI flags
  of Claude Code, Codex, and Pi carried over from the research code.
* Harbor trials on Modal for validation, calibration, and evaluation.
* `harbor check` with the Terminal-Bench rubric.
* Miles training, in particular the bridge between the self-play loop and a running Miles job,
  and carry mode (`autoenvscaling/training/carry/`, added 12/10/2026), which restores the
  paper's partial rollout and over-sampling with a recording proxy in front of the router.
  Carry mode is tested offline only (fakes for Miles, the engines and the sandboxes); see
  `docs/partial-rollout.md` for what was verified against the final run and what still needs a
  live run.

## Author decisions (recorded 29/09/2026)

* Rubric review: `binary_reward` and the contributor-metadata criteria (`task_readme`,
  `expert_time_estimate`, `category_and_tags`, `task_name`, the three `*_explanation_quality`
  criteria, `ctrf_reporting`, `separate_verifier_configured`) are skipped, as in the paper's runs
  (`ValidationSettings.rubric_skip`).
* Schema compatibility: the shipped rubric follows Harbor's free-form `[metadata]` and valid
  `[task]` package fields instead of the Terminal-Bench contributor-metadata whitelist. Current
  network/OS fields remain recognized. Other active criteria, including verifier hygiene and
  execution isolation, remain in effect.
* Solver network during calibration: allowlist. Solver attempts reach only the endpoint's host
  and the hosts Pi installs from (`PI_INSTALL_HOSTS`), plus anything the task already allows;
  `AES_SOLVER_NETWORK=public` opens it.
* Blocked pages for the proposer: Terminal-Bench (`tbench.ai`) and its GitHub repositories. The
  web tools refuse those URLs and the proposer image makes `git clone` of the repositories fail.
* Exp 2 runs 300 solver updates.
* Cold-Start: standard SFT with slime's recipe (the `cold-start` preset), on all 600 admitted
  proposer trajectories that were collected; no further selection.
* Self-play: one assignment per proposer round with 128 episodes; revisions join the first
  attempt's group.
* Licence: Apache License 2.0, as in MSR-Orchard/execcritic (`LICENSE`,
  `THIRD_PARTY_NOTICES.md`).

## Still open

1. The fraction-of-tests verifier template: inject it into every generated task in all three
   generation arms, or only in the baselines?
2. The copyright line in `LICENSE` is left as the Apache template; fill it in (for example,
   Microsoft Corporation) before the public release.
