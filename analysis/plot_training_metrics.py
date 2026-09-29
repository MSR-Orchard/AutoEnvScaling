"""Figure 7: useful rollout-group rate and rollout-collection time, AutoEnvScaling vs Tmax.

Qwen3.6-35B-A3B self-play. Every training step keeps 16 groups with varying rewards; the useful
rate is accepted groups over collected groups. Faint lines are per-step values, solid lines
trailing means (3 steps for the rate, 5 for time). Tmax steps 163-200 are a log-linear
extrapolation fitted to steps 93-162 and are marked "extrapolated" in the data; pooled
statistics (the 3.1x rate ratio and 42% time reduction in the text) use steps 0-162, which both
runs measured.

Data: data/paper/qwen36-rollout-metrics-v2.csv; the pooled summary is checked against
data/paper/fig_training_metrics_summary.json.
Output: outputs/figures/fig_training_metrics.{pdf,png} and fig_training_metrics_summary.json.

    python analysis/plot_training_metrics.py [--check]
"""

from __future__ import annotations

import csv
import json
import math
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import load_json, parse_args, reset_style, save_figure
from matplotlib.lines import Line2D

WINDOWS = {"useful_ratio": 3, "rollout_collection_minutes": 5}
METHODS = [("tmax", "Tmax", "#218C8D", "s"),
           ("AutoEnvScaling", "AutoEnvScaling", "#32006E", "o")]
MATCHED_LAST_STEP = 162


def read_data(data_dir):
    with open(data_dir / "qwen36-rollout-metrics-v2.csv", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 402
    runs = {key: [] for key, *_ in METHODS}
    for source in rows:
        row = dict(source)
        for name in ("step", "groups_per_training_step", "accepted_groups", "rejected_groups"):
            row[name] = int(row[name])
        for name in ("useful_ratio", "rollout_collection_minutes"):
            row[name] = float(row[name])
            assert math.isfinite(row[name])
        assert row["accepted_groups"] == 16 and row["rejected_groups"] >= 0
        assert row["groups_per_training_step"] == row["accepted_groups"] + row["rejected_groups"]
        assert math.isclose(row["useful_ratio"],
                            row["accepted_groups"] / row["groups_per_training_step"],
                            abs_tol=1e-12)
        assert row["rollout_collection_minutes"] > 0
        assert row["source"] in ("measured", "extrapolated")
        assert row["source"] == "measured" or row["method"] == "tmax"
        runs[row["method"]].append(row)
    for method_rows in runs.values():
        method_rows.sort(key=lambda r: r["step"])
        assert [r["step"] for r in method_rows] == list(range(201))
    return runs


def summarize(runs, reference):
    """Pooled acceptance and mean collection time; text fields are carried from the reference."""
    result = {"source": reference["source"], "definitions": reference["definitions"], "runs": {}}
    for key, rows in runs.items():
        rows = [r for r in rows if r["source"] == "measured"]
        accepted = sum(r["accepted_groups"] for r in rows)
        collected = sum(r["groups_per_training_step"] for r in rows)
        result["runs"][key] = {
            "steps": len(rows), "first_step": rows[0]["step"], "last_step": rows[-1]["step"],
            "accepted_groups": accepted, "collected_groups": collected,
            "accepted_fraction": accepted / collected,
            "mean_groups_per_step": collected / len(rows),
            "mean_collection_minutes": statistics.mean(r["rollout_collection_minutes"]
                                                       for r in rows),
        }
    matched = {}
    for key, rows in runs.items():
        rows = [r for r in rows if r["step"] <= MATCHED_LAST_STEP]
        accepted = sum(r["accepted_groups"] for r in rows)
        collected = sum(r["groups_per_training_step"] for r in rows)
        matched[key] = {"accepted_fraction": accepted / collected,
                        "mean_collection_minutes": statistics.mean(
                            r["rollout_collection_minutes"] for r in rows)}
    result["matched_steps_0_162"] = {
        **matched,
        "useful_rate_ratio": (matched["AutoEnvScaling"]["accepted_fraction"]
                              / matched["tmax"]["accepted_fraction"]),
        "collection_time_reduction": 1 - (matched["AutoEnvScaling"]["mean_collection_minutes"]
                                          / matched["tmax"]["mean_collection_minutes"]),
    }
    return result


def trailing_mean(values, window):
    return [sum(values[i - window + 1:i + 1]) / window for i in range(window - 1, len(values))]


def main(argv=None) -> int:
    reset_style()
    args = parse_args(__doc__.splitlines()[0], argv, check=True)
    runs = read_data(args.data_dir)
    reference = load_json(args.data_dir, "fig_training_metrics_summary.json")
    summary = summarize(runs, reference)
    assert summary == reference, "pooled statistics differ from fig_training_metrics_summary.json"
    ratio = summary["matched_steps_0_162"]
    print(f"useful-group ratio {ratio['useful_rate_ratio']:.2f}x, collection time "
          f"-{100 * ratio['collection_time_reduction']:.0f}% (steps 0-{MATCHED_LAST_STEP})")
    if args.check:
        print("All 402 rows, group accounting, and pooled summaries agree.")
        return 0

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 15,
        "axes.labelsize": 16, "axes.titlesize": 16, "axes.titleweight": "bold",
        "xtick.labelsize": 15, "ytick.labelsize": 15,
        "text.color": "#3A3145", "axes.labelcolor": "#3A3145",
        "axes.edgecolor": "#C3B9D6", "axes.linewidth": .8,
        "xtick.color": "#3A3145", "ytick.color": "#3A3145",
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.25))
    fig.subplots_adjust(left=.075, right=.98, bottom=.17, top=.75, wspace=.30)
    panels = [
        ("useful_ratio", 100, "Useful rollout rate", "Useful groups (%)", (0, 52)),
        ("rollout_collection_minutes", 1, "Rollout collection time", "Minutes per training step",
         (0, 420)),
    ]
    for ax, (field, scale, title, ylabel, limits) in zip(axes, panels):
        for key, label, color, marker in METHODS:
            rows = runs[key]
            xs = [r["step"] for r in rows]
            values = [scale * r[field] for r in rows]
            ax.plot(xs, values, color=color, alpha=.20, linewidth=.9, zorder=2)
            sx = xs[WINDOWS[field] - 1:]
            sy = trailing_mean(values, WINDOWS[field])
            ax.plot(sx, sy, color=color, linewidth=2.5, zorder=3)
            marks = [i for i, step in enumerate(sx) if step % 32 == 0]
            ax.scatter([sx[i] for i in marks], [sy[i] for i in marks], marker=marker, s=58,
                       facecolor=color, edgecolor="#3A3145", linewidth=.6, zorder=4)
        ax.set(title=title, xlabel="RL training step", ylabel=ylabel, xlim=(-3, 203), ylim=limits)
        ax.set_xticks(range(0, 201, 50))
        ax.grid(axis="y", color="#DED8EC", linewidth=.7, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(length=3)
    handles = [Line2D([], [], color=c, marker=m, markersize=8, linewidth=2.5, label=label)
               for _, label, c, m in METHODS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.52, .99), ncol=2,
               frameon=False, fontsize=17, columnspacing=2.2)
    paths = save_figure(fig, args.out_dir, "fig_training_metrics", dpi=250, metadata={
        "Title": "Qwen3.6 rollout-group acceptance and collection time"})
    plt.close(fig)
    summary_path = args.out_dir / "fig_training_metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print("wrote", *paths, summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
