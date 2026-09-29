"""Figure 1: AutoEnvScaling across three settings (the page-one results teaser).

Left: mean valid-environment rate over the eight proposers for each generation method, with a
95% t-interval across models. Middle: Terminal-Bench 2.1 after solver RL on environments from
GPT-5.6-sol, against the static Tmax pool, with the gain over Tmax. Right: Qwen3.6-35B-A3B as
both proposer and solver, with the gain over the Cold-Start checkpoint. The figure is drawn at
its printed size (6.5 in wide), so font sizes are final. ``--variant mixed`` swaps the left
panel for a per-model slope chart (not used in the paper).

Data: data/paper/fig_arms_data.json (left), fig_teaser_data.json (middle, the GPT-5.6-sol
column of Table 3), fig_rsi_data.json (right).
Output: outputs/figures/fig_teaser.pdf and .png.

    python analysis/plot_teaser.py [--variant bars|mixed]
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from _common import DATA_DIR, FIGURES_DIR, load_json, read_arms, reset_style, save_figure
from matplotlib.patches import Patch
from scipy.stats import t

# Colours follow the body figures: the arms figure for methods, the self-play figure for runs.
PROMPT, FEEDBACK, OURS = "#CDBCEA", "#7A55B5", "#32006E"
TMAX = "#218C8D"
BASE = "#B9B4C7"
TEXT = "#3A3145"
REFERENCE = "#8C8FC2"

WIDTH, HEIGHT = 6.5, 2.05
FONT = 7.0
TITLE = 7.6


def style():
    sns.set_theme(style="whitegrid", context="paper", font="DejaVu Sans", rc={
        "font.size": FONT, "axes.titlesize": TITLE, "axes.titleweight": "bold",
        "axes.titlepad": 5, "axes.labelsize": FONT, "xtick.labelsize": FONT - 0.4,
        "ytick.labelsize": FONT - 0.4, "legend.fontsize": FONT - 0.6,
        "axes.labelcolor": "#231A33", "text.color": TEXT, "xtick.color": TEXT,
        "ytick.color": TEXT, "axes.edgecolor": "#C3B9D6", "axes.linewidth": 0.6,
        "grid.color": "#E4DFF0", "grid.linewidth": 0.5, "xtick.major.pad": 2,
        "ytick.major.pad": 2, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })


def generation_slope(ax, arms):
    """Valid-environment rate per proposer, one line per model across the three methods."""
    methods = [("prompt", "Prompt"), ("feedback", "Prompt +\nEnv Feedback"),
               ("harness", "AutoEnv-\nScaling")]
    xs = range(len(methods))
    for row in arms.values():
        rates = [100 * row["arms"][m]["rate"] for m, _ in methods]
        ax.plot(xs, rates, color="#CFC6E0", linewidth=0.8, zorder=2)
        ax.scatter(xs, rates, s=9, color=[PROMPT, FEEDBACK, OURS], edgecolor="white",
                   linewidth=0.4, zorder=3)
    means = [sum(100 * r["arms"][m]["rate"] for r in arms.values()) / len(arms)
             for m, _ in methods]
    ax.plot(xs, means, color=OURS, linewidth=1.8, zorder=4)
    ax.scatter(xs, means, s=28, color=[PROMPT, FEEDBACK, OURS], edgecolor=OURS, linewidth=0.9,
               zorder=5)
    for x, mean, dy in zip(xs, means, (-9, -11, -12)):
        ax.annotate(f"{mean:.0f}", (x, mean), xytext=(0, dy), textcoords="offset points",
                    ha="center", va="top", fontsize=FONT, fontweight="bold", color=OURS,
                    bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none",
                          "alpha": 0.85}, zorder=6)
    ds = arms["dsv4"]["arms"]
    before, after = 100 * ds["prompt"]["rate"], 100 * ds["harness"]["rate"]
    ax.annotate(f"DeepSeek-V4-Flash\n{before:.1f} → {after:.1f} ({after / before:.0f}×)",
                xy=(0, before), xytext=(0.18, 6), fontsize=FONT - 1, color=TEXT, va="center",
                arrowprops={"arrowstyle": "-", "color": "#9C92B5", "linewidth": 0.5,
                            "shrinkA": 0, "shrinkB": 2})
    ax.set_xticks(list(xs))
    ax.set_xticklabels([label for _, label in methods])
    ax.set_xlim(-0.35, 2.35)
    ax.set_ylim(0, 102)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Valid-environment rate (%)")
    ax.set_title("Environment generation")
    ax.grid(axis="x", visible=False)


def generation_bars(ax, arms):
    """Mean valid-environment rate over the proposers, with a 95% t-interval across models."""
    methods = [("prompt", "Prompt", PROMPT), ("feedback", "Prompt +\nEnv Feedback", FEEDBACK),
               ("harness", "AutoEnv-\nScaling", OURS)]
    for x, (key, _, colour) in enumerate(methods):
        rates = [100 * r["arms"][key]["rate"] for r in arms.values()]
        mean = statistics.mean(rates)
        half = t.ppf(0.975, len(rates) - 1) * statistics.stdev(rates) / len(rates) ** 0.5
        ax.bar(x, mean, width=0.62, color=colour, edgecolor=TEXT, linewidth=0.4, zorder=2)
        # A wider white error bar underneath keeps it visible where it crosses the dark bar.
        for colour_, width, cap in (("white", 2.2, 3.1), (TEXT, 0.8, 2.5)):
            ax.errorbar(x, mean, yerr=half, color=colour_, linewidth=width, capsize=cap,
                        capthick=width, zorder=3)
        ax.text(x, mean + half + 2.5, f"{mean:.0f}%", ha="center", va="bottom", fontsize=FONT,
                fontweight="bold", color=OURS if key == "harness" else TEXT)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([label for _, label, _ in methods])
    ax.set_ylim(0, 106)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Valid-environment rate (%)")
    ax.set_title("Environment generation")
    ax.grid(axis="x", visible=False)


def solver_bars(ax, training):
    """Terminal-Bench 2.1 after RL on environments from a frontier proposer."""
    runs = [("base", "Base", BASE), ("tmax", "Tmax", TMAX), ("ours", "AutoEnvScaling", OURS)]
    width = 0.26
    for g, solver in enumerate(training["solvers"]):
        for i, (key, _, colour) in enumerate(runs):
            x = g + (i - 1) * width
            value = solver[key]
            ax.bar(x, value, width=width * 0.94, color=colour, edgecolor=TEXT, linewidth=0.4,
                   zorder=2)
            ax.text(x, value + 0.8, f"{value:.1f}", ha="center", va="bottom", fontsize=FONT - 1,
                    fontweight="bold" if key == "ours" else "normal",
                    color=OURS if key == "ours" else TEXT)
        gain = solver["ours"] - solver["tmax"]
        # Bracket legs start above the value labels of the two bars being compared.
        top = solver["ours"] + 7.2
        ax.plot([g, g, g + width, g + width],
                [solver["tmax"] + 4.6, top, top, solver["ours"] + 4.6], color=OURS, linewidth=0.6)
        ax.text(g + width / 2, top + 0.5, f"+{gain:.1f}", ha="center", va="bottom",
                fontsize=FONT, fontweight="bold", color=OURS)
    ax.set_xticks(range(len(training["solvers"])))
    ax.set_xticklabels([s["name"] for s in training["solvers"]])
    ax.set_ylim(0, 60)
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.set_ylabel("Terminal-Bench 2.1")
    ax.set_title("Solver RL on generated environments")
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[Patch(facecolor=c, edgecolor=TEXT, linewidth=0.4, label=label)
                       for _, label, c in runs],
              loc="upper center", ncol=3, frameon=False, handlelength=0.9, handleheight=0.8,
              handletextpad=0.35, columnspacing=0.7, borderaxespad=0.1)


def rsi_curve(ax, rsi):
    """Terminal-Bench 2.1 while Qwen3.6-35B-A3B designs and solves its own environments."""
    start = rsi["cold_start"]["TB2.1"]
    runs = {r["name"]: r for r in rsi["runs"]}
    shown = [("Tmax", "Tmax", TMAX, "^"), ("AutoEnvScaling", "AutoEnvScaling", OURS, "s")]
    ax.axhline(start, color=REFERENCE, linewidth=0.8, linestyle=(0, (1.5, 2)), zorder=1)
    for name, label, colour, marker in shown:
        run = runs[name]
        ax.plot([0, *run["steps"]], [start, *run["TB2.1"]], color=colour, marker=marker,
                markersize=3.2, markeredgecolor="white", markeredgewidth=0.4,
                linewidth=1.8 if name == "AutoEnvScaling" else 1.2, label=label, zorder=3)
        end = run["TB2.1"][-1]
        above = colour == OURS
        ax.annotate(f"{end:.1f}", (run["steps"][-1], end), xytext=(-2, 5 if above else -5),
                    textcoords="offset points", ha="center", va="bottom" if above else "top",
                    fontsize=FONT, fontweight="bold" if colour == OURS else "normal",
                    color=colour)
    ax.annotate(f"Cold-Start {start:.1f}", (296, start), xytext=(0, -2.5),
                textcoords="offset points", ha="right", va="top", fontsize=FONT - 1,
                color="#6F6A8F")
    end = runs["AutoEnvScaling"]["TB2.1"][-1]
    ax.annotate("", xy=(316, end), xytext=(316, start),
                arrowprops={"arrowstyle": "-|>", "color": OURS, "linewidth": 0.8,
                            "mutation_scale": 6, "shrinkA": 0, "shrinkB": 0})
    ax.text(322, (start + end) / 2, f"+{end - start:.1f}", rotation=90, ha="left", va="center",
            fontsize=FONT, fontweight="bold", color=OURS)
    ax.set_xlim(-8, 336)
    ax.set_ylim(39, 56.5)
    ax.set_xticks([0, 100, 200, 300])
    ax.set_xlabel("RL training step", labelpad=1)
    ax.set_ylabel("Terminal-Bench 2.1")
    ax.set_title("Self-improvement")
    ax.legend(loc="upper left", frameon=False, handlelength=1.4, borderaxespad=0.2,
              labelspacing=0.25)


def draw(variant, arms, training, rsi):
    style()
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, HEIGHT),
                             gridspec_kw={"width_ratios": [1.0, 1.0, 1.1]})
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.2, top=0.8, wspace=0.4)
    (generation_slope if variant == "mixed" else generation_bars)(axes[0], arms)
    solver_bars(axes[1], training)
    rsi_curve(axes[2], rsi)
    for ax in axes:
        ax.tick_params(axis="both", length=2, width=0.5, bottom=True, left=True)
        sns.despine(ax=ax)
    return fig


def main(argv=None) -> int:
    reset_style()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--variant", choices=["bars", "mixed"], default="bars")
    args = parser.parse_args(argv)
    arms = {row["key"]: row for row in read_arms(args.data_dir)}
    training = load_json(args.data_dir, "fig_teaser_data.json")
    rsi = load_json(args.data_dir, "fig_rsi_data.json")
    fig = draw(args.variant, arms, training, rsi)
    name = "fig_teaser" if args.variant == "bars" else "fig_teaser_mixed"
    paths = save_figure(fig, args.out_dir, name, bbox_inches="tight", pad_inches=0.02,
                        metadata={"Title": "AutoEnvScaling results teaser"})
    plt.close(fig)
    print("wrote", *paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
