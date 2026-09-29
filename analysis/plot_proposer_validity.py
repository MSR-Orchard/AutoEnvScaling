"""Figure 20: valid-environment rate of Qwen3.6-35B-A3B acting as proposer during self-play.

Both runs start from the Cold-Start checkpoint (open diamond). Without proposer training the
shared weights still change through solver updates and the rate falls until training collapses
at step 160. Only measured points are drawn.

Data: data/paper/fig_proposer_validity_data.json.
Output: outputs/figures/fig_proposer_validity.pdf and .png.

    python analysis/plot_proposer_validity.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from _common import load_json, parse_args, reset_style, save_figure
from matplotlib.lines import Line2D

FULL = "AutoEnvScaling"
ABLATION = "AutoEnvScaling (w/o Proposer Training)"
COLORS = {FULL: "#32006E", ABLATION: "#D07832"}
MARKERS = {FULL: "s", ABLATION: "o"}
TEXT = "#3A3145"


def main(argv=None) -> int:
    reset_style()
    args = parse_args(__doc__.splitlines()[0], argv)
    data = load_json(args.data_dir, "fig_proposer_validity_data.json")
    for run in data["runs"]:
        assert len(run["steps"]) == len(run["valid_rate"]) and run["steps"][0] == 0
    sns.set_theme(style="whitegrid", context="paper", font="DejaVu Sans", rc={
        "font.size": 12.5, "axes.labelsize": 16, "xtick.labelsize": 15, "ytick.labelsize": 15,
        "axes.labelcolor": "#231A33", "text.color": TEXT, "xtick.color": TEXT,
        "ytick.color": TEXT, "axes.edgecolor": "#C3B9D6", "axes.linewidth": 0.8,
        "grid.color": "#DED8EC", "grid.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.16, top=0.87)
    # The ablation is drawn first so the full system sits on top.
    for run in sorted(data["runs"], key=lambda r: r["name"] == FULL):
        full = run["name"] == FULL
        ax.plot(run["steps"], run["valid_rate"], color=COLORS[run["name"]],
                marker=MARKERS[run["name"]], markersize=8, markeredgecolor=TEXT,
                markeredgewidth=0.8, linewidth=3.0 if full else 2.2, zorder=4 if full else 3)
    start = data["runs"][0]["valid_rate"][0]
    ax.scatter([0], [start], marker="D", s=110, facecolor="white", edgecolor=COLORS[FULL],
               linewidth=2, zorder=6)
    ax.set(xlabel="RL training step", ylabel="Valid-environment rate (%)", xlim=(-7, 296),
           ylim=(15, 100))
    ax.set_xticks(range(0, 301, 50))
    ax.set_yticks(range(20, 101, 20))
    ax.tick_params(axis="both", length=3, width=0.8, bottom=True, left=True)
    sns.despine(ax=ax)
    handles = [Line2D([], [], color=COLORS[name], marker=MARKERS[name], markersize=8,
                      markeredgecolor=TEXT, markeredgewidth=0.8,
                      linewidth=3.0 if name == FULL else 2.2, label=name)
               for name in (FULL, ABLATION)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 1.0), ncol=2,
               frameon=False, fontsize=14.5, columnspacing=1.5, handlelength=2.0)
    paths = save_figure(fig, args.out_dir, "fig_proposer_validity", metadata={
        "Title": "Proposer valid-environment rate during shared-policy training"})
    plt.close(fig)
    print("wrote", *paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
