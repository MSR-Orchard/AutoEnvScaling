"""Shared paths and helpers for the paper's figure and table scripts.

Every script reads its numbers from ``data/paper/`` and writes to ``outputs/figures/`` (figures,
as PDF plus a PNG preview) or ``outputs/tables/``. Both directories can be changed with
``--data-dir`` and ``--out-dir``. No script calls a model or starts a sandbox.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data" / "paper"
FIGURES_DIR = REPO / "outputs" / "figures"
TABLES_DIR = REPO / "outputs" / "tables"
Z95 = 1.959963984540054


def parse_args(description: str, argv: list[str] | None = None, *, default_out: Path = FIGURES_DIR,
               check: bool = False) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=default_out)
    if check:
        parser.add_argument("--check", action="store_true",
                            help="validate the data file and render nothing")
    return parser.parse_args(argv)


def reset_style() -> None:
    """Start from matplotlib's defaults, so styles set by one script never leak into the next."""
    import matplotlib

    matplotlib.rcdefaults()
    matplotlib.use("Agg")


def load_json(data_dir: Path, name: str) -> Any:
    return json.loads((Path(data_dir) / name).read_text())


def wilson(accepted: int, n: int, z: float = Z95) -> list[float]:
    """Two-sided Wilson score interval for accepted / n."""
    rate = accepted / n
    scale = 1 + z * z / n
    center = (rate + z * z / (2 * n)) / scale
    radius = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / scale
    return [center - radius, center + radius]


ARMS = [("prompt", "Prompt"), ("feedback", "Prompt + Env Feedback"),
        ("harness", "AutoEnvScaling")]
ARM_KEYS = ["opus", "astra", "sol", "terra", "gemini38", "luna", "dsv4", "grok"]


def read_arms(data_dir: Path) -> list[dict[str, Any]]:
    """Rows of fig_arms_data.json (Figure 4, Table 2, Figure 1 left), with consistency checks."""
    rows = load_json(data_dir, "fig_arms_data.json")["rows"]
    assert {r["key"] for r in rows} == set(ARM_KEYS) and len(rows) == len(ARM_KEYS)
    for row in rows:
        assert set(row["arms"]) == {arm for arm, _ in ARMS}
        for cell in row["arms"].values():
            assert 0 <= cell["accepted"] <= cell["scored"] and cell["scored"] > 0
            assert math.isclose(cell["rate"], cell["accepted"] / cell["scored"], abs_tol=1e-12)
            # Earlier exports used z = 1.96; their stored intervals agree to 1e-5.
            assert all(abs(a - b) < 1e-5
                       for a, b in zip(cell["ci"], wilson(cell["accepted"], cell["scored"])))
            assert cell["cost_usd"] is None or cell["cost_usd"] > 0
            if "terminal" in cell:
                assert cell["terminal"] == (cell["scored"] + cell["infrastructure_failure"]
                                            + cell.get("provider_refusal", 0))
                assert cell["terminal"] <= cell["target"]
                assert cell["scored"] == cell["accepted"] + cell["model_rejection"]
    return rows


def save_figure(fig: Any, out_dir: Path, name: str, *, dpi: int = 300,
                metadata: dict[str, str] | None = None, **kwargs: Any) -> list[Path]:
    """Write ``<name>.pdf`` and a ``<name>.png`` preview; return both paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf, png = out_dir / f"{name}.pdf", out_dir / f"{name}.png"
    fig.savefig(pdf, metadata=metadata or {}, **kwargs)
    fig.savefig(png, dpi=dpi, **kwargs)
    return [pdf, png]
