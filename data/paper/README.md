# Paper data

The numbers behind the paper's figures and tables. The scripts in `analysis/` read these files
and write to `outputs/` (`python analysis/make_all.py`). Figure and table numbers refer to the
arXiv version.

| File | Backs | Script |
|---|---|---|
| `fig_arms_data.json` | Figure 4, Table 2, Figure 1 (left) | `plot_arms.py`, `make_tables.py`, `plot_teaser.py` |
| `astra_results_20260917.json` | Table 2, GPT-6-astra row (rates and costs of the stopped campaign) | none; supporting record |
| `fig_harness_artifacts_data.json` | Figure 5 (and the paired effects quoted in Section 4.1) | `plot_harness_artifacts.py` |
| `fig_teaser_data.json` | Figure 1 (middle); the GPT-5.6-sol column of Table 3 | `plot_teaser.py` |
| `fig_rsi_data.json` | Figure 6, Figure 1 (right) | `plot_rsi.py`, `plot_teaser.py` |
| `qwen36-rollout-metrics-v2.csv` | Figure 7 | `plot_training_metrics.py` |
| `fig_training_metrics_summary.json` | Figure 7 pooled statistics (3.1x useful-group rate, 42% less collection time) | `plot_training_metrics.py` checks against it |
| `fig_proposer_validity_data.json` | Figure 20 | `plot_proposer_validity.py` |
| `fig_cold_start_data.json` | Figures 21 and 22 | `plot_cold_start.py` |
| `fig_cross_domain_data.json` | Companion figure for Appendix C (not printed in the paper) | `plot_cross_domain.py` |
| `paper_tables.json` | Tables 3, 4, 5, 8, 10, 11, 12, 13 (transcribed) | `make_tables.py` |

## Conventions

- Rates are accepted divided by scored. Recorded infrastructure failures and provider refusals
  are excluded; model rejections stay in the denominator. Intervals are two-sided 95% Wilson
  score intervals (Figure 1, left, uses a t-interval across the eight models).
- Costs are USD per valid environment. The Table 2 subscript is the cost ratio of
  AutoEnvScaling to Prompt. For DeepSeek-V4-Flash the paper prints 0.25x, the ratio of the
  rounded costs (0.02 / 0.08); `cost_ratio_label` carries that value next to the stored
  one-decimal `cost_ratio_vs_prompt`.
- In `qwen36-rollout-metrics-v2.csv`, Tmax steps 163-200 are a log-linear extrapolation fitted
  to steps 93-162 (`source = extrapolated`). Pooled statistics use steps 0-162, which both runs
  measured.
- `fig_rsi_data.json` and `fig_proposer_validity_data.json` hold only measured evaluation points;
  nothing is interpolated.
- `fig_cross_domain_data.json` keeps infrastructure statuses in the denominator (1,061 accepted
  of 1,629 judged lineages). Table 8 measures acceptance over completed, evaluable runs (1,061 of
  1,489), so the file cannot reproduce Table 8; its values are in `paper_tables.json`.
- `paper_tables.json` is transcribed from the paper's LaTeX. The runs behind Tables 3, 4, 5, 8,
  11 and 13 are not part of this release.

## What was changed from the authors' working copies

Numbers are unchanged. For release, provenance fields that pointed into the authors' internal
research repository (repository names, commits, blob hashes, internal paths and run-lane
names), the operational shutdown notes of the stopped GPT-6-astra campaign, and the log-file
column of the rollout-metrics CSV were removed, and personal names in "supplied by" notes were
replaced by "the authors". Two working files were not copied: the raw cost-audit ledger behind
the Gemini and Grok cost cells (its per-cell results are in `fig_arms_data.json`) and an earlier
version of the rollout-metrics CSV superseded by `qwen36-rollout-metrics-v2.csv`.
