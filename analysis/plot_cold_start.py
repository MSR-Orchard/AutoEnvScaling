"""Figures 21 and 22: admission rate and proposer episode depth, before and after Cold-Start.

Figure 21 (fig_cold_start_admission): admission rate with 95% Wilson intervals for frontier
models, open models, and Qwen3.6-35B-A3B before and after Cold-Start. Figure 22
(fig_cold_start_depth): turns per proposer episode (boxes are the interquartile range and
median, whiskers reach 1.5 IQR, outliers omitted).

Data: data/paper/fig_cold_start_data.json (admission counts and per-episode turns).
Output: outputs/figures/fig_cold_start_admission.{pdf,png} and fig_cold_start_depth.{pdf,png}.

    python analysis/plot_cold_start.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import load_json, parse_args, reset_style, save_figure
from matplotlib.patches import Patch

KEYS = ["Claude Opus 5", "DeepSeek V4-Flash", "Kimi K3", "GPT-5.6-sol",
        "Qwen3.6-35B-A3B distilled", "Qwen3.6-35B-A3B", "Nemotron-3-Ultra 550B",
        "Nemotron-3-Super 120B"]
LABELS = ["Claude Opus 5", "DeepSeek V4-Flash", "Kimi K3", "GPT-5.6-sol",
          "Qwen3.6-35B-A3B\n(Cold-Start)", "Qwen3.6-35B-A3B\n(Base)", "Nemotron-3-Ultra 550B",
          "Nemotron-3-Super 120B"]
DEPTH = [0, 3, 4, 5, 6, 7]  # indices of KEYS shown in the depth figure
COLORS = {"ours": "#32006E", "frontier": "#7A55B5", "open": "#B9A8D4"}


def style(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E1DBEB", linewidth=.7)
    ax.tick_params(axis="x", length=0)


def legend(fig, fontsize=10.5):
    groups = [("frontier", "Frontier"), ("ours", "Ours (Qwen3.6-35B-A3B)"),
              ("open", "Open-source")]
    fig.legend(handles=[Patch(facecolor=COLORS[g], label=label) for g, label in groups],
               loc="upper center", ncol=3, frameon=False, fontsize=fontsize,
               columnspacing=1.1, handlelength=1.1)


def wilson_errors(admits, n, z=1.96):
    """Asymmetric error bar lengths (below, above) of a Wilson interval, in percent."""
    p = admits / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** .5) / d
    return p * 100, max(0, (p - c + h) * 100), max(0, (c + h - p) * 100)


def main(argv=None) -> int:
    reset_style()
    args = parse_args(__doc__.splitlines()[0], argv)
    by = {r["label"]: r for r in load_json(args.data_dir, "fig_cold_start_data.json")}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    paths = []

    fig, ax = plt.subplots(figsize=(6.8, 5.3))
    rates, lo, hi = zip(*(wilson_errors(by[k]["admits"], by[k]["n"]) for k in KEYS))
    ax.bar(range(len(KEYS)), rates, color=[COLORS[by[k]["group"]] for k in KEYS],
           yerr=[lo, hi], capsize=2, error_kw={"lw": 1})
    for i, v in enumerate(rates):
        ax.text(i, v + hi[i] + 2, f"{v:.1f}", ha="center", fontsize=16)
    ax.set_xticks(range(len(KEYS)))
    ax.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=12)
    ax.set_ylim(0, 116)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Admission rate (%)", fontsize=13)
    style(ax)
    legend(fig, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, .91])
    paths += save_figure(fig, args.out_dir, "fig_cold_start_admission", dpi=200,
                         bbox_inches="tight")
    plt.close(fig)

    keys = [KEYS[i] for i in DEPTH]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bp = ax.boxplot([by[k]["turns"] for k in keys], patch_artist=True, showfliers=False,
                    medianprops={"color": "white", "lw": 1.8})
    for box, k in zip(bp["boxes"], keys):
        box.set_facecolor(COLORS[by[k]["group"]])
    ax.set_xticks(range(1, len(keys) + 1))
    ax.set_xticklabels([LABELS[i] for i in DEPTH], fontsize=11, rotation=32, ha="right")
    ax.set_ylabel("Turns per episode")
    style(ax)
    legend(fig)
    fig.tight_layout(rect=[0, 0, 1, .9])
    paths += save_figure(fig, args.out_dir, "fig_cold_start_depth", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", *paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
