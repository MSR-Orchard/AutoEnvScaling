"""The paper's tables, rendered from data/paper/.

Table 2 (valid-environment rate and cost per valid environment) is computed from
fig_arms_data.json: rates are accepted / scored with 95% Wilson intervals, costs are USD per
valid environment, and the subscript is the cost ratio of AutoEnvScaling to Prompt. Its LaTeX
rows are written in exactly the form the paper prints. The other tables (3, 4, 5, 8, 10, 11, 12,
13) have no underlying released runs; their values are transcribed in paper_tables.json and
rendered here for reference.

Output: outputs/tables/table2_threeway_rows.tex and outputs/tables/tables.md.

    python analysis/make_tables.py
"""

from __future__ import annotations

from _common import ARMS, TABLES_DIR, load_json, parse_args, read_arms


def table2_latex_rows(rows) -> str:
    """Body rows of Table 2 in the paper's LaTeX form."""
    lines = []
    for row in rows:
        rates, costs = [], []
        for arm, _ in ARMS:
            cell = row["arms"][arm]
            value = f'{100 * cell["rate"]:.1f}'
            if arm == "harness":
                value = r"\textbf{" + value + "}"
            lo, hi = cell["ci"]
            rates.append(value + f" \\ci{{{100 * lo:.1f}}}{{{100 * hi:.1f}}}")
            cost = cell["cost_usd"]
            rendered = "--" if cost is None else f"{cost:.2f}"
            ratio = cell.get("cost_ratio_vs_prompt")
            if arm == "harness" and cost is not None and ratio is not None:
                label = cell.get("cost_ratio_label") or f"{ratio:.1f}"
                color = "CostCheap" if ratio <= 1 else "CostDear"
                arrow = r"\downarrow" if ratio <= 1 else r"\uparrow"
                rendered += (r"$_{\textcolor{" + color + "}{" + label + r"\times\," + arrow
                             + r"}}$")
            costs.append(rendered)
        lines.append(" & ".join([row["label"], *rates, *costs]) + r" \\")
    return "\n".join(lines) + "\n"


def _md(header: list[str], body: list[list[object]]) -> str:
    def cell(v: object) -> str:
        return "--" if v is None else str(v)

    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(cell(v) for v in row) + " |" for row in body]
    return "\n".join(out)


def table2_markdown(rows) -> str:
    body = []
    for row in rows:
        cells = [row["label"]]
        for arm, _ in ARMS:
            c = row["arms"][arm]
            cells.append(f'{100 * c["rate"]:.1f} [{100 * c["ci"][0]:.1f}, {100 * c["ci"][1]:.1f}]')
        for arm, _ in ARMS:
            c = row["arms"][arm]
            cost = "--" if c["cost_usd"] is None else f'{c["cost_usd"]:.2f}'
            if arm == "harness" and c.get("cost_ratio_vs_prompt") is not None:
                cost += f' ({c.get("cost_ratio_label") or format(c["cost_ratio_vs_prompt"], ".1f")}x)'
            cells.append(cost)
        body.append(cells)
    header = ["Proposer", "Prompt (%)", "Prompt + Env Feedback (%)", "AutoEnvScaling (%)",
              "Prompt ($)", "Prompt + Env Feedback ($)", "AutoEnvScaling ($)"]
    return _md(header, body)


def transcribed_markdown(tables: dict) -> list[str]:
    parts = []
    t = tables["tab:training"]
    body = []
    for r in t["rows"]:
        name = {"base": r["solver"], "tmax": "+ Tmax", "autoenvscaling": "+ AutoEnvScaling",
                "autoenvscaling_without_solver_feedback":
                    "+ AutoEnvScaling (w/o solver feedback)"}[r["setting"]]
        vals = ([f'{r["shared"][0]} ± {r["shared"][1]}'] * 2 if "shared" in r
                else [f"{v} ± {e}" for v, e in r["values"]])
        body.append([name, *vals])
    parts.append(f"## Table {t['number']}\n\n{t['caption']}\n\n"
                 + _md(["Model / setting", *t["proposers"]], body))
    for key, first in (("tab:hil-frontier", "Model"), ("tab:hil-training", "Setting"),
                       ("tab:selfvalidate", "Proposer"), ("tab:initialization-controls", "Model")):
        t = tables[key]
        name_key = {"Model": "model", "Setting": "setting", "Proposer": "proposer"}[first]
        body = [[r[name_key], *r["values"]] for r in t["rows"]]
        parts.append(f"## Table {t['number']}\n\n{t['caption']}\n\n"
                     + _md([first, *t["columns"]], body))
    t = tables["tab:cross-domain-results"]
    body = [[r["domain"], r["proposer"], f'{r["values"][0]:.1f}', f'{r["values"][1]:.2f}']
            for r in t["rows"]]
    parts.append(f"## Table {t['number']}\n\n{t['caption']} {t['note']}\n\n"
                 + _md(["Domain", "Proposer", "Acceptance (%)", "Rounds"], body))
    t = tables["tab:sandbox-resources"]
    keys = ["cpus", "memory_gib", "time_budget_hours", "turn_limit", "coding_agent"]
    parts.append(f"## Table {t['number']}\n\n{t['caption']}\n\n"
                 + _md(["", "Proposer", "Solver"],
                       [[k, t["proposer"][k], t["solver"][k]] for k in keys]))
    t = tables["tab:training-settings"]
    parts.append(f"## Table {t['number']}\n\n{t['caption']}\n\n"
                 + _md(["Setting", "Value"],
                       [[k, v] for k, v in t.items() if k not in ("number", "caption")]))
    return parts


def main(argv=None) -> int:
    args = parse_args(__doc__.splitlines()[0], argv, default_out=TABLES_DIR)
    rows = read_arms(args.data_dir)
    tables = load_json(args.data_dir, "paper_tables.json")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tex = args.out_dir / "table2_threeway_rows.tex"
    tex.write_text(table2_latex_rows(rows))
    md = args.out_dir / "tables.md"
    md.write_text("# Paper tables\n\n## Table 2\n\nValid-environment rate (95% Wilson interval) "
                  "and cost per valid environment (USD).\n\n" + table2_markdown(rows) + "\n\n"
                  + "\n\n".join(transcribed_markdown(tables)) + "\n")
    print("wrote", tex, md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
