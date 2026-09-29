"""Render every paper figure and table from data/paper/ into outputs/.

    python analysis/make_all.py [--data-dir DIR] [--out-dir DIR]

Figures go to <out-dir>/figures and tables to <out-dir>/tables (default: outputs/).
Needs the analysis extras: matplotlib, seaborn, numpy, pandas, scipy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import make_tables
import plot_arms
import plot_cold_start
import plot_cross_domain
import plot_harness_artifacts
import plot_proposer_validity
import plot_rsi
import plot_teaser
import plot_training_metrics
from _common import DATA_DIR, REPO

FIGURE_SCRIPTS = [plot_teaser, plot_arms, plot_harness_artifacts, plot_rsi, plot_training_metrics,
                  plot_proposer_validity, plot_cold_start, plot_cross_domain]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=REPO / "outputs")
    args = parser.parse_args(argv)
    common = ["--data-dir", str(args.data_dir)]
    for script in FIGURE_SCRIPTS:
        script.main(common + ["--out-dir", str(args.out_dir / "figures")])
    make_tables.main(common + ["--out-dir", str(args.out_dir / "tables")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
