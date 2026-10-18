# Cold-Start

Section 4.3 and Appendix H; Figures 21 and 22 and Table 13.

Base Qwen3.6-35B-A3B produces valid environments only 32.3% of the time, too rarely to start
self-play. Cold-Start fine-tunes it on the 600 admitted proposer trajectories collected from Claude Opus 5,
DeepSeek-V4-Flash and Kimi-K3 (all of them; no further selection), which raises the
valid-environment rate to 86.8%.

```bash
export BASE_HF=... BASE_REF=... MILES_DIR=... MEGATRON_DIR=...
export AES_PROPOSER_BASE_URL=... AES_PROPOSER_API_KEY=... AES_SOLVER_BASE_URL=... AES_SOLVER_MODEL=...
SEED_TASKS="tasks/*" cmd/cold_start/run.sh
DRY_RUN=1 cmd/cold_start/run.sh
```

## Commands

| Step | Command |
|---|---|
| Teacher episodes | `aes generate --parents-pool P --pool-dir Q --harness H --model <teacher> --assignments N` |
| Fine-tuning rows | `autoenvscaling export sft <round>/proposer_trajectories.jsonl ... --out sft.jsonl` |
| Fine-tuning | `autoenvscaling train plan\|run --preset cold-start --prompt-data sft.jsonl --hf-checkpoint ... --ref-load ... --save ...` |
| Generation before and after | `aes generate --harness pi --model <served checkpoint> --assignments 60\|136 --seed S` |
| Solving before and after | `cmd/evals/run.sh <served checkpoint> tb2.1` |

`export sft` keeps admitted episodes (r_P = 1), one per task. The paper trained on every such
trajectory it had (600); `--limit` exists for smaller runs but was not used.

Evaluations:

- **Figures 21 and 22**: admission rate (95% Wilson intervals) and turns per episode on a shared
  136-assignment suite for the base and Cold-Start models and GPT-5.6-sol; the frontier
  reference rows come from earlier suites. Admission requires a successful image build, a
  passing reference solution, and a failing do-nothing solution.
- **Table 13**: a paired control on 60 assignments (33.3% vs 76.7% acceptance), and solving on
  Terminal-Bench 2.1 with five attempts on each of 89 tasks (avg@5 40.0 vs 41.1, pass@5 59.6 vs
  66.3).

Teachers run in their coding agents (Claude Opus 5 in Claude Code, the others in Pi) in the
16 CPU / 32 GiB / 2-hour proposer sandbox.

## Gaps

- The `cold-start` preset uses slime's standard SFT recipe for Qwen3.5-35B-A3B
  (`scripts/run-qwen3.5-35B-A3B-sft.sh`: 3 epochs, batch 128, lr 1e-5 cosine to 1e-6 with 10%
  warmup), which is what the paper's standard SFT used.
- The paper's 136- and 60-assignment suites are not released; the launcher draws the same number
  of assignments from `SEED_TASKS` with a fixed seed, so both models see the same ones.
- Serving the checkpoints as OpenAI-compatible endpoints is not part of this package.
