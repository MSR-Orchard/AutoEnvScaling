"""Companion figure for Appendix C (not printed in the paper): the 3 x 7 cross-domain matrix.

Left: acceptance rate with 95% Wilson intervals per domain and proposer. Right: mean authoring
rounds (initial session plus repair sessions, 1 to 3) among accepted lineages.

This figure keeps infrastructure statuses in the denominator (1,061 accepted of 1,629 judged
lineages; 30 unfinished lineages excluded). Table 8 in the paper instead measures acceptance
over completed, evaluable runs (1,061 of 1,489), so its rates are higher; the table's values
are in data/paper/paper_tables.json and are rendered by make_tables.py.

GPT models used Codex and Claude Opus 5 used Claude Code, with different time limits, so these
are pipeline outcomes rather than a controlled comparison of models. Acceptance means passing
the technical admission checks; domain review and solver calibration were not run.

Data: data/paper/fig_cross_domain_data.json.
Output: outputs/figures/fig_cross_domain.pdf and .png.

    python analysis/plot_cross_domain.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from _common import load_json, parse_args, reset_style, save_figure, wilson
from matplotlib.patches import Patch

# Muted categorical colours distinguish models from the purple authoring-method scale.
COLORS = ["#A493C6", "#89A9C9", "#D3A0B0"]
INK = "#3A3145"
MODELS = ["gpt-6-astra", "gpt-5.6-sol", "claude-opus-5"]
MODEL_LABELS = ["GPT-6-astra", "GPT-5.6-sol", "Claude Opus-5"]
DOMAINS = ["terminal", "swe", "tool", "browser", "desktop-cua", "kernelbench",
           "professional-workflow"]
DOMAIN_LABELS = ["Terminal", "SWE", "Tool calling", "Browser use", "Computer use",
                 "GPU kernels", "Professional"]


def checked_cells(data):
    cells = {(row["model"], row["domain"]): row for row in data["cells"]}
    assert set(cells) == {(m, d) for m in MODELS for d in DOMAINS}
    for cell in cells.values():
        assert 0 < cell["accepted"] <= cell["judged"]
        counts = cell["accepted_attempt_counts"]
        assert set(counts) == {"1", "2", "3"}
        assert all(isinstance(n, int) and n >= 0 for n in counts.values())
        assert sum(counts.values()) == cell["accepted"]
        mean = sum(int(k) * n for k, n in counts.items()) / cell["accepted"]
        assert abs(mean - cell["mean_authoring_attempts"]) < 1e-12
    return cells


def main(argv=None) -> int:
    reset_style()
    args = parse_args(__doc__.splitlines()[0], argv)
    cells = checked_cells(load_json(args.data_dir, "fig_cross_domain_data.json"))
    accepted = sum(c["accepted"] for c in cells.values())
    judged = sum(c["judged"] for c in cells.values())
    unfinished = sum(c["unfinished"] for c in cells.values())
    sns.set_theme(style="whitegrid", context="paper", font="DejaVu Sans", rc={
        "font.size": 16, "axes.titlesize": 19, "axes.titleweight": "bold",
        "axes.labelsize": 18, "xtick.labelsize": 18, "ytick.labelsize": 16,
        "legend.fontsize": 17, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK,
        "ytick.color": INK, "axes.edgecolor": "#C3B9D6", "axes.linewidth": 0.8,
        "grid.color": "#E2DEEA", "grid.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.7))
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.28, top=0.82, wspace=0.22)
    positions = np.arange(len(DOMAINS))
    width = 0.235
    for model_index, model in enumerate(MODELS):
        x = positions + (model_index - 1) * width
        selected = [cells[model, domain] for domain in DOMAINS]
        rates = np.array([100 * c["accepted"] / c["judged"] for c in selected])
        intervals = np.array([wilson(c["accepted"], c["judged"]) for c in selected]) * 100
        # Rounds include the initial session and any repair sessions.
        rounds = [sum(int(k) * n for k, n in c["accepted_attempt_counts"].items()) / c["accepted"]
                  for c in selected]
        axes[0].bar(x, rates, width=width, color=COLORS[model_index], edgecolor=INK,
                    linewidth=0.6, zorder=3,
                    yerr=np.vstack([rates - intervals[:, 0], intervals[:, 1] - rates]),
                    error_kw={"ecolor": INK, "elinewidth": 1.0, "capsize": 3.0, "capthick": 1.0})
        axes[1].bar(x, rounds, width=width, color=COLORS[model_index], edgecolor=INK,
                    linewidth=0.6, zorder=3)
    axes[0].set(title="(a) Acceptance rate", ylabel="Accepted (%)", ylim=(0, 105))
    axes[0].set_yticks([0, 25, 50, 75, 100])
    axes[1].set(title="(b) Authoring rounds", ylabel="Rounds per\naccepted task", ylim=(0, 3.05))
    axes[1].set_yticks([0, 1, 2, 3])
    for ax in axes:
        ax.set_xlim(-0.6, len(DOMAINS) - 0.4)
        ax.set_xticks(positions, DOMAIN_LABELS)
        ax.tick_params(axis="x", pad=9)
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="y", length=3, width=0.8, left=True)
        title = ax.get_title()
        ax.set_title("")
        ax.set_title(title, loc="left", pad=13)
        sns.despine(ax=ax)
    handles = [Patch(facecolor=c, edgecolor=INK, linewidth=0.6, label=label)
               for c, label in zip(COLORS, MODEL_LABELS)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.525, 1.0), ncol=3,
               frameon=False, columnspacing=2.4, handlelength=1.7)
    paths = save_figure(fig, args.out_dir, "fig_cross_domain", dpi=200,
                        metadata={"Title": "Environment authoring across seven domains"})
    plt.close(fig)
    print(f"{len(cells)} cells: {accepted:,} accepted / {judged:,} judged; "
          f"{unfinished} unfinished excluded")
    print("wrote", *paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
