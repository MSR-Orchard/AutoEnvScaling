"""Figure 6: Qwen3.6-35B-A3B on Terminal-Bench 2.1 and 4.0 during self-play training.

Three runs from the same Cold-Start checkpoint: AutoEnvScaling, AutoEnvScaling without proposer
training (ends at step 160, when training collapsed), and static Tmax. Dashed and dotted lines
mark the base model and the Cold-Start checkpoint. Only measured evaluation points are drawn.

Data: data/paper/fig_rsi_data.json (the same file backs the right panel of Figure 1).
Output: outputs/figures/fig_rsi.pdf and .png.

    python analysis/plot_rsi.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from _common import load_json, parse_args, reset_style, save_figure
from matplotlib.lines import Line2D

# Distinct baseline hues; deep purple for the full system.
COLORS = ["#218C8D", "#D07832", "#32006E"]
MARKERS = ["^", "o", "s"]
TEXT = "#3A3145"
REFERENCE = "#8C8FC2"
PANELS = [
    ("TB2.1", "Terminal-Bench 2.1", (38.8, 54.2), [40, 42, 44, 46, 48, 50, 52, 54]),
    ("TB4", "Terminal-Bench 4.0", (0.8, 3.45), [0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2]),
]


def check(data) -> None:
    for run in data["runs"]:
        for metric, *_ in PANELS:
            assert len(run["steps"]) == len(run[metric]), (run["name"], metric)
        assert run["steps"] == sorted(run["steps"])


def main(argv=None) -> int:
    reset_style()
    args = parse_args(__doc__.splitlines()[0], argv)
    data = load_json(args.data_dir, "fig_rsi_data.json")
    check(data)
    sns.set_theme(style="whitegrid", context="paper", font="DejaVu Sans", rc={
        "font.size": 12.5, "axes.titlesize": 16, "axes.titleweight": "bold",
        "axes.labelsize": 16, "xtick.labelsize": 16, "ytick.labelsize": 16,
        "axes.labelcolor": "#231A33", "text.color": TEXT, "xtick.color": TEXT,
        "ytick.color": TEXT, "axes.edgecolor": "#C3B9D6", "axes.linewidth": 0.8,
        "grid.color": "#DED8EC", "grid.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))
    fig.subplots_adjust(left=0.075, right=0.97, bottom=0.17, top=0.76, wspace=0.27)
    for ax, (metric, title, limits, ticks) in zip(axes, PANELS):
        base = data["base"][metric]
        initial = data["cold_start"][metric]
        ax.axhline(base, color=REFERENCE, linewidth=1.5, linestyle=(0, (6, 4)), zorder=1)
        ax.axhline(initial, color=REFERENCE, linewidth=1.5, linestyle=(0, (1.5, 2.5)), zorder=1)
        for i, run in enumerate(data["runs"]):
            ax.plot([0, *run["steps"]], [initial, *run[metric]],
                    color=COLORS[i], marker=MARKERS[i], markersize=9.5,
                    markeredgecolor=TEXT, markeredgewidth=0.8,
                    linewidth=3.0 if i == 2 else 2.2, zorder=3 + i)
        # One open diamond marks the shared initialization at step zero.
        ax.scatter([0], [initial], marker="D", s=125, facecolor="white",
                   edgecolor=COLORS[2], linewidth=2, zorder=7)
        ax.set(title=title, xlabel="RL training step", ylabel="Score", xlim=(-7, 300), ylim=limits)
        ax.set_xticks(range(0, 301, 50))
        ax.set_yticks(ticks)
        ax.tick_params(axis="both", length=3, width=0.8, bottom=True, left=True)
        sns.despine(ax=ax)

    handles = [
        Line2D([], [], color=COLORS[i], marker=MARKERS[i], markersize=9.5,
               markeredgecolor=TEXT, markeredgewidth=0.8, linewidth=3 if i == 2 else 2.2,
               label=run["name"].replace(" (w/o", "\n(w/o"))
        for i, run in enumerate(data["runs"])
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 1.0),
               ncol=3, frameon=False, fontsize=17.25, columnspacing=1.5, handlelength=2.0)
    paths = save_figure(fig, args.out_dir, "fig_rsi",
                        metadata={"Title": "Recursive self-improvement training results"})
    plt.close(fig)
    print("wrote", *paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
