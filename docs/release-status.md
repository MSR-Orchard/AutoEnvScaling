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
* Miles training, in particular the bridge between the self-play loop and a running Miles job.
  Both roles currently train on re-tokenized conversations; exact token capture needs a proxy in
  front of the rollout router, which is not in this release.

## Decisions needed from the authors

1. Rubric scope: the official Terminal-Bench rubric includes contributor-metadata criteria
   (`task_readme`, `expert_time_estimate`, `category_and_tags`, `task_name`, the
   `*_explanation_quality` criteria, `ctrf_reporting`, `separate_verifier_configured`). The paper
   applies every criterion except `binary_reward`. Were these also skipped in the paper's runs?
2. Solver network during calibration: public, as in the research code (Pi installs itself and
   calls the model endpoint from inside the task container), or an allowlist?
3. The fraction-of-tests verifier template: inject it into every generated task in all three
   generation arms, or only in the baselines?
4. Web access: which benchmark pages besides Terminal-Bench were blocked for the proposer?
5. Self-play details not stated in the paper: assignments per proposer round (default 1 x 128
   episodes), whether revisions join the first attempt's group, and which episode's memory and
   tools persist when episodes run concurrently.
6. Exp 2 matched step count and learning rate; Cold-Start fine-tuning hyperparameters and the
   "high-quality" selection rule.
7. Licence for the public release.
