"""Stacked phase bars for SQL–ML pipeline comparison."""
from __future__ import annotations

import json
from pathlib import Path
_REPO = Path(__file__).resolve().parents[2]

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(_REPO)
SRC = ROOT / "results" / "sql_ml_pipeline.json"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SYSTEMS = [
    "postgres_external",
    "duckdb_preloaded",
    "duckdb_load_from_pg",
    "pgml",
    "madlib",
    "postgresml",
]
LABELS = {
    "postgres_external": "PG → Python",
    "duckdb_preloaded": "DuckDB warm",
    "duckdb_load_from_pg": "DuckDB←PG load",
    "pgml": "pgml",
    "madlib": "MADlib",
    "postgresml": "PostgresML",
}
# phase colors
PHASE_COLORS = {
    "load_total_ms": "#8a6d1d",
    "materialize_ms": "#c4a035",
    "sql_ms": "#1f4e79",
    "transfer_ms": "#5c4d7a",
    "convert_ms": "#7a8a9a",
    "train_ms": "#1b7f4a",
    "predict_ms": "#2a9d5c",
}
PHASE_ORDER = ["load_total_ms", "materialize_ms", "sql_ms", "transfer_ms", "convert_ms", "train_ms", "predict_ms"]
PHASE_LABELS = {
    "load_total_ms": "load from PG",
    "materialize_ms": "DuckDB materialize",
    "sql_ms": "SQL",
    "transfer_ms": "transfer",
    "convert_ms": "convert",
    "train_ms": "train",
    "predict_ms": "predict",
}


def load():
    return json.loads(SRC.read_text(encoding="utf-8"))


def plot(data):
    pts = [p for p in data["points"] if "phases" in p]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), layout="constrained")

    for ax, task, title in (
        (axes[0], "regression", "A  Regression (linreg)"),
        (axes[1], "classification", "B  Classification (tree)"),
    ):
        # focus n=50k
        sub = [p for p in pts if p["task"] == task and p["n"] == 50000]
        by = {p["system"]: p for p in sub}
        x = np.arange(len(SYSTEMS))
        bottoms = np.zeros(len(SYSTEMS))
        # collect used phases
        for phase in PHASE_ORDER:
            vals = []
            for sys in SYSTEMS:
                ph = by.get(sys, {}).get("phases", {})
                if phase == "load_total_ms":
                    vals.append(ph.get("load_total_ms", 0.0) or 0.0)
                elif phase == "materialize_ms":
                    # only count materialize for warm duckdb (not double-count with load)
                    vals.append(ph.get("materialize_ms", 0.0) if sys == "duckdb_preloaded" else 0.0)
                else:
                    vals.append(ph.get(phase, 0.0) or 0.0)
            vals = np.asarray(vals, dtype=float)
            if vals.sum() <= 0:
                continue
            ax.bar(x, vals, bottom=bottoms, color=PHASE_COLORS[phase],
                   edgecolor="#222", linewidth=0.4, label=PHASE_LABELS[phase])
            bottoms += vals
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[s] for s in SYSTEMS], rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("time (ms)")
        ax.set_title(title + r" at $n{=}5{\cdot}10^4$")
        ax.grid(True, axis="y", ls=":", alpha=0.45)
        for i, sys in enumerate(SYSTEMS):
            tot = by.get(sys, {}).get("phases", {}).get("total_ms")
            if tot is not None:
                ax.text(i, bottoms[i] * 1.01 + 5, f"{tot:.0f}", ha="center", va="bottom", fontsize=7)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=7, fontsize=8, framealpha=0.95,
               bbox_to_anchor=(0.5, 1.08))
    fig.suptitle(
        f"SQL–ML pipelines (DuckDB {data['duckdb_version']}): phases of full cycle",
        fontsize=10, y=1.12,
    )
    png = OUT / "sql_ml_pipeline.png"
    pdf = OUT / "sql_ml_pipeline.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}")
    print(f"wrote {pdf}")

    # scaling line plot for regression totals
    fig2, ax = plt.subplots(figsize=(7.8, 4.0), layout="constrained")
    for sys in SYSTEMS:
        xs, ys = [], []
        for n in data["grid"]["n"]:
            hit = next((p for p in pts if p["system"] == sys and p["n"] == n and p["task"] == "regression"), None)
            if hit:
                xs.append(n)
                ys.append(hit["phases"]["total_ms"])
        ax.plot(xs, ys, "o-", label=LABELS[sys], linewidth=1.5, markersize=6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"rows $n$")
    ax.set_ylabel("full-cycle time (ms)")
    ax.set_title("Regression full-cycle scaling")
    ax.grid(True, which="both", ls=":", alpha=0.45)
    ax.legend(fontsize=8, loc="best")
    png2 = OUT / "sql_ml_pipeline_scaling.png"
    pdf2 = OUT / "sql_ml_pipeline_scaling.pdf"
    fig2.savefig(png2, dpi=180, bbox_inches="tight")
    fig2.savefig(pdf2, bbox_inches="tight")
    plt.close(fig2)
    print(f"wrote {png2}")
    print(f"wrote {pdf2}")


if __name__ == "__main__":
    plot(load())
