# Cross-domain environment generation

Appendix C; Table 8 (and a companion figure that is not printed in the paper).

GPT-6-astra and GPT-5.6-sol in Codex, and Claude Opus 5 in Claude Code, each write complete
Harbor environments (instruction, initialization code, reference solution, verifier) in seven
domains, then build and test them, with up to two revision rounds after the initial attempt.
This study measures generation and admission only; no solver is trained.

| Domain | Environment | Verification |
|---|---|---|
| Terminal | files, manifests, CLI tools | parsed outputs and exact totals |
| Software engineering | repository, issue, terminal | regression tests and private boundary cases |
| Tool calling | seeded service, typed MCP calls | service state and decision policy |
| Browser use | local web application | saved fields and state transitions |
| Computer use | LibreOffice desktop | saved formulas, values, formatting |
| GPU kernels | PyTorch/Triton on one NVIDIA T4 | numerical correctness and timing against eager |
| Professional workflows | read-only inputs, workbook, PDF | calculations and cross-deliverable consistency |

Admission in the paper: the environment initializes, its reference solution earns full reward,
a do-nothing agent earns zero, and a reset reproduces the initial state and a passing reference
run. Acceptance is measured over completed, evaluable runs (1,061 of 1,489 overall, 71.3%);
rounds are averaged over accepted tasks.

```bash
export AES_SOLVER_BASE_URL=... AES_SOLVER_MODEL=...
SEED_TASKS="tasks/*" cmd/cross_domain/run.sh
DRY_RUN=1 cmd/cross_domain/run.sh
```

## What runs

- **Terminal domain**: `aes generate --harness codex|claude-code --model M`
  from `SEED_TASKS` parents. Admission is the standard pipeline's (empty solution below 0.5
  rather than exactly zero, no reset check, plus contamination, rubric review and calibration),
  so it is stricter in some checks and looser in others than the appendix protocol.
- **Table 8** as transcribed and the companion figure from the released lineage counts:
  `python analysis/make_all.py`. The companion figure keeps infrastructure failures in the
  denominator (1,061 of 1,629), so its rates are lower than Table 8's.

## Not in this release yet

- The six non-terminal domains: their assignments and runtimes (repositories, MCP services,
  local web applications, a graphical desktop, a T4 GPU sandbox, office documents). The
  launcher prints a notice for each.
- The appendix's admission checks that a do-nothing agent earns exactly zero and that a reset
  reproduces the initial state.
