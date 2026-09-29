import importlib
import json
import re
import sys
from pathlib import Path

import pytest

from autoenvscaling.config import Settings

REPO = Path(__file__).resolve().parents[1]
ANALYSIS = REPO / "analysis"
DATA = REPO / "data" / "paper"


@pytest.fixture(scope="module")
def scripts():
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    pytest.importorskip("scipy")
    sys.path.insert(0, str(ANALYSIS))
    try:
        yield lambda name: importlib.import_module(name)
    finally:
        sys.path.remove(str(ANALYSIS))


def test_every_data_file_is_documented():
    readme = (DATA / "README.md").read_text()
    for path in DATA.iterdir():
        if path.name != "README.md":
            assert f"`{path.name}`" in readme, path.name


def test_released_data_has_no_internal_identifiers():
    # Absolute machine paths, IP addresses and links into source repositories.
    pattern = re.compile(r"/Users/|/mnt/|/root/|/home/|/data/|github\.com/|"
                         r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")
    for path in DATA.iterdir():
        text = path.read_text()
        assert not pattern.search(text), (path.name, pattern.search(text).group(0))


def test_checks_pass_on_released_data(scripts, capsys):
    for name in ("plot_arms", "plot_harness_artifacts", "plot_training_metrics"):
        assert scripts(name).main(["--check"]) == 0


def test_table2_rows_match_paper_format(scripts, tmp_path):
    assert scripts("make_tables").main(["--out-dir", str(tmp_path)]) == 0
    rows = (tmp_path / "table2_threeway_rows.tex").read_text().splitlines()
    assert rows[0] == (r"Claude Opus 5 & 74.6 \ci{70.0}{78.7} & 89.5 \ci{86.0}{92.2} & "
                       r"\textbf{94.0} \ci{91.2}{95.9} & 0.72 & 0.70 & "
                       r"1.18$_{\textcolor{CostDear}{1.6\times\,\uparrow}}$ \\")
    deepseek = next(r for r in rows if r.startswith("DeepSeek-V4-Flash"))
    assert r"{0.25\times\,\downarrow}" in deepseek and "3.5 \\ci{2.1}{5.8}" in deepseek
    assert len(rows) == 8
    md = (tmp_path / "tables.md").read_text()
    for number in (2, 3, 4, 5, 8, 10, 11, 12, 13):
        assert f"## Table {number}\n" in md


def test_every_figure_renders(scripts, tmp_path):
    assert scripts("make_all").main(["--out-dir", str(tmp_path)]) == 0
    expected = ["fig_teaser", "fig_arms", "fig_harness_artifacts", "fig_rsi",
                "fig_training_metrics", "fig_proposer_validity", "fig_cold_start_admission",
                "fig_cold_start_depth", "fig_cross_domain"]
    for name in expected:
        for suffix in ("pdf", "png"):
            path = tmp_path / "figures" / f"{name}.{suffix}"
            assert path.is_file() and path.stat().st_size > 5_000, path
    summary = json.loads((tmp_path / "figures" / "fig_training_metrics_summary.json").read_text())
    matched = summary["matched_steps_0_162"]
    assert round(matched["useful_rate_ratio"], 1) == 3.1           # "3.1x" in Section 4.3
    assert round(100 * matched["collection_time_reduction"]) == 42  # "42% less time"


def test_paper_numbers_quoted_in_text_follow_from_data():
    arms = {r["key"]: r for r in json.loads((DATA / "fig_arms_data.json").read_text())["rows"]}
    ds = arms["dsv4"]["arms"]
    assert round(100 * ds["prompt"]["rate"], 1) == 3.5 and round(100 * ds["harness"]["rate"], 1) == 89.0
    rsi = json.loads((DATA / "fig_rsi_data.json").read_text())
    full = next(r for r in rsi["runs"] if r["name"] == "AutoEnvScaling")
    tmax = next(r for r in rsi["runs"] if r["name"] == "Tmax")
    assert round(full["TB2.1"][-1] - rsi["cold_start"]["TB2.1"], 1) == 12.2
    assert round(full["TB2.1"][-1] - tmax["TB2.1"][-1], 1) == 6.1
    assert full["TB4"][-1] == 3.2 and rsi["cold_start"]["TB4"] == 1.6
    validity = json.loads((DATA / "fig_proposer_validity_data.json").read_text())
    ablation = next(r for r in validity["runs"] if "w/o" in r["name"])
    assert ablation["steps"][-1] == 160 and ablation["valid_rate"][-1] == 22.4
    cold = {r["label"]: r for r in json.loads((DATA / "fig_cold_start_data.json").read_text())}
    assert round(100 * cold["Qwen3.6-35B-A3B distilled"]["admits"]
                 / cold["Qwen3.6-35B-A3B distilled"]["n"], 1) == 86.8
    assert round(100 * cold["Qwen3.6-35B-A3B"]["admits"] / cold["Qwen3.6-35B-A3B"]["n"], 1) == 32.3


def test_package_defaults_match_paper_settings_tables():
    tables = json.loads((DATA / "paper_tables.json").read_text())
    settings = Settings()
    t12 = tables["tab:training-settings"]
    sp = settings.selfplay
    assert (sp.learning_rate, sp.solver_envs_per_batch, sp.solver_rollouts_per_env,
            sp.proposer_group_size, sp.proposer_interval, sp.pool_size, sp.pool_review_interval,
            sp.dppo_tv_threshold) == (
        t12["learning_rate"], t12["solver_envs_per_batch"], t12["solver_rollouts_per_env"],
        t12["proposer_group_size"], t12["proposer_interval"], t12["pool_size"],
        t12["pool_review_interval"], t12["dppo_tv_threshold"])
    t10 = tables["tab:sandbox-resources"]
    sb = settings.sandbox
    assert (sb.proposer_cpu, sb.proposer_memory_gib, sb.proposer_timeout_s) == (
        t10["proposer"]["cpus"], t10["proposer"]["memory_gib"],
        t10["proposer"]["time_budget_hours"] * 3600)
    assert (sb.solver_cpu, sb.solver_memory_gib) == (t10["solver"]["cpus"],
                                                    t10["solver"]["memory_gib"])
    from autoenvscaling.evals.terminal_bench import EvalSettings
    ev = EvalSettings()
    assert (ev.cpus, ev.memory_gib, ev.k) == (t10["solver"]["cpus"], t10["solver"]["memory_gib"], 5)
