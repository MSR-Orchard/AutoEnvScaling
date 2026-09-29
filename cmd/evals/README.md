# Evaluations: Terminal-Bench 2.1 and 4.0

Every Terminal-Bench number in the paper (Tables 3 and 13, Figures 1 and 6) is avg@5: each task
is attempted five times and the score is the mean verifier reward over all task-attempts. The
solver runs in the Pi coding agent inside the task container, in a sandbox with 4 CPUs and
8 GiB of memory (Appendix G.3).

```bash
export AES_EVAL_BASE_URL=https://<your OpenAI-compatible endpoint>/v1
export AES_EVAL_API_KEY=...
cmd/evals/run.sh <served model id> both          # or tb2.1 / tb4
```

The launcher first sends one tiny request to check that the endpoint serves the exact model id,
then runs one Harbor job per benchmark (default environment `modal`; set `HARBOR_ENV`), and
writes `aes_report.json` next to each job. The endpoint must be reachable from the sandboxes,
because Pi calls the model from inside the task container.

Rescore or merge finished jobs (for example after rerunning tasks that ended in infrastructure
errors) without running anything:

```bash
autoenvscaling eval report runs/evals/<job> runs/evals/<rerun job> -k 5
```

Print the Harbor job configuration without running it:

```bash
autoenvscaling eval config --benchmark tb2.1 --model <id>
```

## What the numbers mean

- Infrastructure errors (sandbox, network, provider API) are retried by Harbor and never scored
  as reward 0; a retry does not count as an attempt. A trial whose verifier reported no reward is
  counted as verifier-broken, not as a failure of the model. Agent timeouts and context-window
  exhaustion are the model's outcome: the verifier scores the state left behind.
- Tasks with fewer than five scored attempts are reported as incomplete. By default the score
  averages over the attempts a task has; `--missing zero` counts missing attempts as 0.
- `per_attempt_means` gives the five pass@1 scores; their spread is the run-to-run variation.

## Settings

| Setting | Value | Source |
|---|---|---|
| Benchmarks | `terminal-bench@2.1`, `terminal-bench/terminal-bench@4.0.0` | Section 4 |
| Attempts per task | 5 (avg@5) | Section 4.2, Appendix G.3 |
| Agent | Pi 0.80.6, bash tool only, no skills, compaction off | evaluation harness |
| Sandbox | 4 CPUs, 8 GiB | Appendix G.3 |
| Agent time limit | each task's declared timeout | not stated in the paper |
| Decoding | endpoint defaults unless `--sampling` is given | not stated in the paper |

Terminal-Bench 4.0 contains tasks that need Docker Compose or GPUs; the Harbor environment must
support them, otherwise those tasks end as infrastructure errors and are listed as incomplete.
