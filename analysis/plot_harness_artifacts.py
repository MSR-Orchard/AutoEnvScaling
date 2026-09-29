"""Figure 5: files, solver turns and solver pass rate of environments from each method.

The lollipops show arm-level means over four author models. The badge inside the harness stem
compares arm means with Prompt + Env Feedback; it reads larger than the paired effects the text
quotes, because a few outsized packages pull the means (DeepSeek-V4-Flash ships 49 data files on
average against a median of 2). The paired effects are kept in the data file and checked here.

Data: data/paper/fig_harness_artifacts_data.json.
Output: outputs/figures/fig_harness_artifacts.pdf and .png.

    python analysis/plot_harness_artifacts.py [--check]
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from _common import load_json, parse_args, reset_style, save_figure
from matplotlib.lines import Line2D

COLORS = {"prompt": "#CDBCEA", "feedback": "#7A55B5", "harness": "#32006E"}
EDGE = "#231A33"
GRID = "#DED8EC"
SPINE = "#C3B9D6"
TICK = "#3A3145"
NEUTRAL = "#7A55B5"
ADVERSE = "#218C8D"

RATIO_EFFECTS = ["files_ratio", "data_files_ratio", "verifier_tokens_ratio"]
DELTA_EFFECTS = ["solver_pass1_delta", "learnable_share_delta"]
FOREST_ORDER = RATIO_EFFECTS + DELTA_EFFECTS


def read_data(data_dir):
    document = load_json(data_dir, "fig_harness_artifacts_data.json")
    arms = [key for key, _ in document["arms"]]
    for panel in document["panels"]:
        assert set(panel["mean"]) == set(arms), panel["key"]
        for value in panel["mean"].values():
            assert math.isfinite(value) and value >= 0, panel["key"]
        # Not every panel has a paired effect extracted yet; those that claim one must name
        # a real entry rather than a stale key.
        assert panel.get("paired", next(iter(document["paired_effects"]))) \
            in document["paired_effects"], panel["key"]
        spec = panel["badge"]
        assert spec["kind"] in {"ratio", "complement_ratio"}, panel["key"]
        if spec["kind"] == "complement_ratio":
            # Complements are only meaningful if every arm sits inside the scale.
            assert all(0 <= v <= spec["total"] for v in panel["mean"].values()), panel["key"]
    for key in FOREST_ORDER:
        effect = document["paired_effects"][key]
        low, high = effect["ci"]
        assert low <= effect["value"] <= high, key
        assert effect["kind"] in {"ratio", "pp"}, key
        # A ratio interval straddling 1, or a pp interval straddling 0, is not a resolved
        # effect; the caption must not claim one.
        null = 1.0 if effect["kind"] == "ratio" else 0.0
        effect["resolved"] = high < null or low > null
    return document


def badge(panel):
    """The label inside the harness stem, comparing it with Prompt + Env Feedback.

    These are ratios of arm means, not the paired ratios quoted in the text; the means are
    pulled by a few outsized packages, so the badge reads larger than the paired effect.
    Where a lower value is the claim (pass rate), the badge compares complements so it still
    reads in the claim's direction: a 51.0% pass rate against 61.0% is a 49/39 = 1.3x failure
    ratio, printed as "1.3x harder".
    """
    spec = panel["badge"]
    harness, baseline = panel["mean"]["harness"], panel["mean"]["feedback"]
    if spec["kind"] == "complement_ratio":
        total = spec["total"]
        harness, baseline = total - harness, total - baseline
    return f'{harness / baseline:.1f}{spec["suffix"]}'


def style(ax):
    ax.grid(axis="x", color=GRID, linewidth=.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)
    ax.tick_params(labelsize=16, colors=TICK, length=3)


# Per-panel x-axis floor overrides; panels not listed start at 0.
X_FLOOR = {"solver_turns": 25, "solver_pass1": 30}


def lollipop(ax, panel, arms, show_arm_labels):
    keys = [key for key, _ in arms]
    values = [panel["mean"][k] for k in keys]
    ys = list(range(len(keys)))
    x_floor = X_FLOOR.get(panel["key"], 0)
    for y, key, value in zip(ys, keys, values):
        ax.plot([x_floor, value], [y, y], color=COLORS[key], linewidth=3.4,
                solid_capstyle="round", zorder=2)
        ax.scatter([value], [y], s=190 if key == "harness" else 130,
                   color=COLORS[key], edgecolor=EDGE,
                   linewidth=1.1 if key == "harness" else .6, zorder=4)
        ax.annotate(panel["value_format"].format(value), (value, y),
                    textcoords="offset points", xytext=(14, 0),
                    va="center", ha="left", fontsize=19, color=EDGE,
                    fontweight="bold" if key == "harness" else "normal")
    harness = panel["mean"]["harness"]
    mid = x_floor + (harness - x_floor) * 0.5
    ax.annotate(badge(panel), (mid, ys[keys.index("harness")]),
                textcoords="offset points", xytext=(0, 2.5), ha="center", va="bottom",
                fontsize=19, color=COLORS["harness"], fontweight="bold", zorder=6)
    ax.set_yticks(ys)
    if show_arm_labels:
        labels = [label.replace(" + ", " +\n ") for _, label in arms]
        ax.set_yticklabels(labels, fontsize=17)
    else:
        ax.set_yticklabels([])
    ax.set_ylim(len(keys) - .40, -.60)
    ax.set_xlim(x_floor, x_floor + (max(values) - x_floor) * 1.40)
    ax.set_title(panel["title"].replace("\\%", "%"), fontsize=19, pad=11,
                 fontweight="bold", color=EDGE)
    style(ax)


def main(argv=None) -> int:
    reset_style()
    args = parse_args(__doc__.splitlines()[0], argv, check=True)
    document = read_data(args.data_dir)
    if args.check:
        print(f'{len(document["panels"])} panels and {len(FOREST_ORDER)} paired effects agree.')
        return 0

    matplotlib.rcParams["pdf.fonttype"] = 42
    sns.set_style("white")
    panels = document["panels"]
    # Equal cells, so every panel's plot area is the same width. The arm labels on the first
    # panel overflow into the figure's left margin rather than eating its plot area.
    fig, axes = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 3.9),
                             constrained_layout=True)
    fig.get_layout_engine().set(w_pad=.07, wspace=.03)
    for column, (ax, panel) in enumerate(zip(axes, panels)):
        lollipop(ax, panel, document["arms"], show_arm_labels=column == 0)

    handles = [Line2D([], [], marker="o", linestyle="", markersize=13,
                      markerfacecolor=COLORS[key], markeredgecolor=EDGE,
                      markeredgewidth=1.1 if key == "harness" else .6, label=label)
               for key, label in document["arms"]]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False, fontsize=22)
    paths = save_figure(fig, args.out_dir, "fig_harness_artifacts", metadata={
        "Title": "What the coding harness changes about authored environments"})
    plt.close(fig)
    for panel in panels:
        means = " / ".join(panel["value_format"].format(panel["mean"][k])
                           for k, _ in document["arms"])
        print(f'  {panel["title"]}: {means}')
    print("wrote", *paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
