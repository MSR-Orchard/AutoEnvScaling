"""Figure 4: valid-environment rate of eight proposers under the three generation methods.

Data: data/paper/fig_arms_data.json (the same file backs Table 2 and the left panel of Figure 1).
Output: outputs/figures/fig_arms.pdf and .png.

    python analysis/plot_arms.py [--check]
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import ARMS, parse_args, read_arms, reset_style, save_figure
from matplotlib.patches import Patch

LEFT = ["opus", "astra", "sol", "grok", "gemini38", "dsv4"]
RIGHT = ["sol", "terra", "luna"]
COLORS = {"prompt": "#CDBCEA", "feedback": "#7A55B5", "harness": "#32006E"}
EDGE = {"prompt": 0.6, "feedback": 0.6, "harness": 1.1}
BAR_ANNOTATION_SCALE = 1.5
LINE_ANNOTATION_SCALE = 1.7
LEGEND_SCALE = 1.5
SHORT = {
    "opus": "Claude\nOpus 5", "sol": "GPT-5.6\nsol", "terra": "GPT-5.6\nterra",
    "luna": "GPT-5.6\nluna", "dsv4": "DeepSeek\nV4-Flash", "grok": "Grok\n4.6",
    "gemini38": "Gemini-3.8\nFlash", "astra": "GPT-6\nastra",
}


def plot_data(rows):
    return {
        row["key"]: {
            "label": row["label"],
            **{arm: row["arms"][arm]["rate"] for arm, _ in ARMS},
            "ci": {arm: row["arms"][arm]["ci"] for arm, _ in ARMS},
        }
        for row in rows
    }


def _axes_style(ax, keys, title):
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([SHORT.get(k, k) for k in keys], fontsize=11)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title(title, fontsize=9.6, pad=8, loc="center", fontweight="bold")
    ax.grid(axis="y", color="#DED8EC", linewidth=.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#C3B9D6")
    ax.tick_params(labelsize=11, colors="#3A3145", length=3)


def panel(ax, keys, data, title):
    """Grouped bars with 95% Wilson intervals, one group per proposer."""
    w = 0.26
    for i, (arm, _) in enumerate(ARMS):
        for j, k in enumerate(keys):
            d = data.get(k, {})
            v = d.get(arm)
            if v is None:
                continue
            x = j + (i - 1) * w
            y = v * 100
            ci = (d.get("ci") or {}).get(arm)
            yerr = None
            if ci and ci[0] is not None:
                lo, hi = ci[0] * 100, ci[1] * 100
                yerr = [[max(0.0, y - lo)], [max(0.0, hi - y)]]
            ax.bar(x, y, width=w, color=COLORS[arm], edgecolor="#231A33",
                   linewidth=EDGE[arm], zorder=3,
                   yerr=yerr, error_kw={"ecolor": "#231A33", "elinewidth": .9, "capsize": 2.4,
                                        "capthick": .9, "zorder": 5})
            top = y if yerr is None else y + yerr[1][0]
            ax.text(x, top + 1.8, f"{y:.0f}", ha="center", va="bottom",
                    fontsize=7.4 * BAR_ANNOTATION_SCALE, color="#231A33", zorder=6)
    _axes_style(ax, keys, title)
    ax.set_ylabel("Valid-environment rate (\\%)" if matplotlib.rcParams["text.usetex"]
                  else "Valid-environment rate (%)", fontsize=12)


def line_panel(ax, keys, data, title):
    """One line per method across proposers ordered strongest to weakest.

    Lines rather than a second bar group because the claim is about a trend along an ordered
    axis: the harness line stays flat while the two prompt-based lines fall away under it.
    """
    xs = list(range(len(keys)))
    for arm, lab in ARMS:
        ys, los, his = [], [], []
        for k in keys:
            d = data.get(k, {})
            v = d.get(arm)
            ys.append(None if v is None else v * 100)
            ci = (d.get("ci") or {}).get(arm)
            if ci and ci[0] is not None and v is not None:
                los.append(v * 100 - ci[0] * 100)
                his.append(ci[1] * 100 - v * 100)
            else:
                los.append(0.0)
                his.append(0.0)
        ax.errorbar(xs, ys, yerr=[los, his], label=lab,
                    color=COLORS[arm], marker="o", markersize=5.2,
                    linewidth=2.6 if arm == "harness" else 1.7,
                    markeredgecolor="#231A33", markeredgewidth=.6,
                    capsize=2.6, capthick=.9, elinewidth=.9, zorder=4 if arm == "harness" else 3)
        for x, y in zip(xs, ys):
            if y is None:
                continue
            ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                        xytext=(0, 9 if arm == "harness" else -13), ha="center",
                        fontsize=7.4 * LINE_ANNOTATION_SCALE, color="#231A33")
    _axes_style(ax, keys, title)
    ax.set_xlim(-.35, len(keys) - .65)
    ax.set_ylabel("Valid-environment rate (%)", fontsize=12)


def main(argv=None) -> int:
    reset_style()
    args = parse_args(__doc__.splitlines()[0], argv, check=True)
    rows = read_arms(args.data_dir)
    if args.check:
        print(f"All {len(rows) * len(ARMS)} cells and Wilson intervals agree with the data.")
        return 0
    data = plot_data(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3), gridspec_kw={"width_ratios": [1.55, 1]})
    panel(axes[0], LEFT, data, "Across model families")
    line_panel(axes[1], RIGHT, data, "Within the GPT-5.6 family, strongest to weakest")
    handles = [Patch(facecolor=COLORS[a], edgecolor="#231A33", linewidth=.6, label=lab)
               for a, lab in ARMS]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
               fontsize=10 * LEGEND_SCALE, bbox_to_anchor=(.5, 1.005))
    fig.tight_layout(rect=(0, 0, 1, .88))
    paths = save_figure(fig, args.out_dir, "fig_arms",
                        metadata={"Title": "Environment authoring with eight frontier proposers"})
    plt.close(fig)
    for row in rows:
        print(row["label"] + ": " + " / ".join(f'{100 * row["arms"][a]["rate"]:.1f}'
                                              for a, _ in ARMS))
    print("wrote", *paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
