#!/usr/bin/env python3
"""Publication figure: pgml performance overview (three panels).

Reads audited JSON results under results/ and writes:
  figures/performance_overview.{pdf,svg,png}
  figures/performance_overview_data.csv
  figures/performance_overview_caption.tex
  figures/performance_overview_audit.md

No hand-entered timings — all values come from the result files.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "figures"
PAPER_OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
PAPER_OUT.mkdir(parents=True, exist_ok=True)

# Colorblind-friendly (Okabe–Ito), pgml fixed
C_PGML = "#0072B2"
C_SKLEARN = "#E69F00"
C_EXTERNAL = "#D55E00"
C_PACK = "#56B4E9"
C_TRAIN = "#0072B2"
C_PREDICT = "#009E73"
C_TRANSFER = "#CC79A7"
C_CONVERT = "#F0E442"
C_SQL = "#999999"
C_MISSING = "#F2F2F2"
C_FAIL = "#000000"

NS_MAP = [1000, 10000, 50000, 100000, 200000]
DS_MAP = [8, 32, 128]
TREE_FAST_N = 30000


def load_json(name: str):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def setup_style():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "grid.linewidth": 0.4,
            "grid.alpha": 0.35,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "mathtext.fontset": "cm",
        }
    )


def select_regime_tree_cells(rm: dict):
    """Binary classification tree cells for sparse n×d map.

    Prefer scale_task for (n, d=32); dimensionality for other d at n=10000.
    Coverage (not speedup) motivates choosing tree: most measured (n,d) pairs
    among single-algorithm train-mode cells in regime_map.
    """
    cells = {}
    meta = {}
    for p in rm["points"]:
        c = p["config"]
        if c.get("task") != "binary_classification" or c.get("model") != "tree":
            continue
        axis = c.get("axis")
        if axis not in ("scale_task", "dimensionality", "memory_probe"):
            continue
        n, d = int(c["n"]), int(c["d"])
        key = (n, d)
        rec = {
            "id": p.get("id"),
            "axis": axis,
            "n": n,
            "d": d,
            "batch_predict": p.get("batch"),
            "pgml_train_ms": (p.get("pgml_ms") or {}).get("train"),
            "pgml_predict_ms": (p.get("pgml_ms") or {}).get("predict"),
            "pgml_materialize_ms": (p.get("pgml_ms") or {}).get("materialize"),
            "pgml_total_ms": (p.get("pgml_ms") or {}).get("total"),
            "sklearn_fit_ms": (p.get("sklearn_ms") or {}).get("fit"),
            "sklearn_predict_ms": (p.get("sklearn_ms") or {}).get("predict"),
            "sklearn_total_ms": (p.get("sklearn_ms") or {}).get("total"),
            "speedup_vs_sklearn": p.get("speedup_vs_sklearn"),
            "verdict": p.get("verdict"),
            "failure_stage": p.get("failure_stage"),
            "oom_confirmed": p.get("oom_confirmed"),
            "algorithm_variant": (
                "tree_fast" if n >= TREE_FAST_N and p.get("verdict") != "materialization_failure" else "tree_exact"
            ),
            "timing_scope": p.get("timing_scope") or rm.get("timing_scope_global"),
            "source": "results/regime_map.json",
        }
        # Prefer scale_task over duplicate dimensionality at (10000,32)
        if key not in cells or (cells[key]["axis"] == "dimensionality" and axis == "scale_task"):
            cells[key] = rec
        meta[key] = rec
    return cells


def select_regime_scaling(rm: dict):
    """d=32 binary tree scale_task series (compatible settings)."""
    rows = []
    for p in rm["points"]:
        c = p["config"]
        if (
            c.get("axis") == "scale_task"
            and c.get("task") == "binary_classification"
            and c.get("model") == "tree"
            and int(c.get("d", -1)) == 32
        ):
            rows.append(
                {
                    "n": int(c["n"]),
                    "d": 32,
                    "pgml_total_ms": p["pgml_ms"]["total"],
                    "sklearn_total_ms": p["sklearn_ms"]["total"],
                    "pgml_train_ms": p["pgml_ms"]["train"],
                    "sklearn_fit_ms": p["sklearn_ms"]["fit"],
                    "batch_predict": p.get("batch"),
                    "algorithm_variant": "tree_fast" if int(c["n"]) >= TREE_FAST_N else "tree_exact",
                    "speedup_vs_sklearn": p.get("speedup_vs_sklearn"),
                    "source": "results/regime_map.json",
                }
            )
    rows.sort(key=lambda r: r["n"])
    return rows


def select_pipeline_breakdown(dr: dict):
    """Shared-holdout database-resident regression phases (same starting table)."""
    out = []
    for p in dr["points"]:
        if p.get("task") != "regression":
            continue
        if p.get("system") not in ("pgml", "postgres_external"):
            continue
        ph = p["phases"]
        if p["system"] == "pgml":
            stages = {
                "sql_pack_ms": ph["sql_pack_ms"],
                "sql_ms": ph["sql_ms"],
                "train_ms": ph["train_ms"],
                "predict_ms": ph["predict_ms"],
                "transfer_ms": 0.0,
                "convert_ms": 0.0,
            }
            total = ph["full_pipeline_from_table_ms"]
            total_key = "full_pipeline_from_table_ms"
            pred_note = ",".join(ph.get("predict_ms_includes") or [])
        else:
            stages = {
                "sql_pack_ms": 0.0,
                "sql_ms": ph["sql_ms"],
                "train_ms": ph["train_ms"],
                "predict_ms": ph["predict_ms"],
                "transfer_ms": ph["transfer_ms"],
                "convert_ms": ph["convert_ms"],
            }
            total = ph["pipeline_from_table_ms"]
            total_key = "pipeline_from_table_ms"
            pred_note = "numpy_predict"
        stage_sum = sum(stages.values())
        out.append(
            {
                "system": p["system"],
                "n": int(p["n"]),
                "d": int(p["d"]),
                "task": "regression",
                "stages": stages,
                "total_ms": total,
                "total_key": total_key,
                "stage_sum_ms": stage_sum,
                "quality_r2": (p.get("quality") or {}).get("r2"),
                "predict_ms_includes": pred_note,
                "initial_state": p.get("initial_state"),
                "pack_scenario": p.get("pack_scenario"),
                "repeats": p.get("repeats"),
                "source": "results/sql_ml_database_resident.json",
            }
        )
    out.sort(key=lambda r: (r["n"], r["system"]))
    return out


def collect_csv_rows(cells, scale_rows, pipe_rows, pack_repeats, env):
    rows = []
    for (n, d), c in sorted(cells.items()):
        rows.append(
            {
                "panel": "a_performance_map",
                "source": c["source"],
                "system_a": "pgml",
                "system_b": "sklearn_inprocess",
                "algorithm": "decision_tree",
                "algorithm_variant": c["algorithm_variant"],
                "n_train": n,
                "n_test_or_batch": c.get("batch_predict"),
                "d": d,
                "threads": 1,
                "status": c["verdict"],
                "failure_stage": c.get("failure_stage") or "",
                "oom_confirmed": c.get("oom_confirmed"),
                "time_a_ms": c.get("pgml_total_ms"),
                "time_b_ms": c.get("sklearn_total_ms"),
                "time_a_train_ms": c.get("pgml_train_ms"),
                "time_a_predict_ms": c.get("pgml_predict_ms"),
                "time_a_materialize_ms": c.get("pgml_materialize_ms"),
                "time_b_fit_ms": c.get("sklearn_fit_ms"),
                "time_b_predict_ms": c.get("sklearn_predict_ms"),
                "ratio_b_over_a": c.get("speedup_vs_sklearn"),
                "log2_ratio": (
                    float(np.log2(c["speedup_vs_sklearn"]))
                    if c.get("speedup_vs_sklearn") not in (None, 0)
                    else ""
                ),
                "timing_boundary": "train+predict; excludes float4[][] materialization",
                "repeats": "single_run",
                "warmup": 0,
                "point_id": c.get("id"),
                "axis": c.get("axis"),
                "env_note": env.get("threads_note", ""),
            }
        )
    for r in scale_rows:
        for sys, tkey, tval in (
            ("pgml", "pgml_total_ms", r["pgml_total_ms"]),
            ("sklearn_inprocess", "sklearn_total_ms", r["sklearn_total_ms"]),
        ):
            rows.append(
                {
                    "panel": "b_scaling",
                    "source": r["source"],
                    "system_a": sys,
                    "system_b": "",
                    "algorithm": "decision_tree",
                    "algorithm_variant": r["algorithm_variant"],
                    "n_train": r["n"],
                    "n_test_or_batch": r.get("batch_predict"),
                    "d": r["d"],
                    "threads": 1,
                    "status": "ok",
                    "failure_stage": "",
                    "oom_confirmed": "",
                    "time_a_ms": tval,
                    "time_b_ms": "",
                    "time_a_train_ms": r["pgml_train_ms"] if sys == "pgml" else r["sklearn_fit_ms"],
                    "time_a_predict_ms": "",
                    "time_a_materialize_ms": "",
                    "time_b_fit_ms": "",
                    "time_b_predict_ms": "",
                    "ratio_b_over_a": r["speedup_vs_sklearn"] if sys == "pgml" else "",
                    "log2_ratio": "",
                    "timing_boundary": "train+predict; excludes materialization; predict on first min(n,10000) rows",
                    "repeats": "single_run",
                    "warmup": 0,
                    "point_id": "",
                    "axis": "scale_task",
                    "env_note": env.get("threads_note", ""),
                }
            )
    for r in pipe_rows:
        for stage, val in r["stages"].items():
            rows.append(
                {
                    "panel": "c_pipeline_breakdown",
                    "source": r["source"],
                    "system_a": r["system"],
                    "system_b": "",
                    "algorithm": "linreg",
                    "algorithm_variant": "linreg",
                    "n_train": int(0.75 * r["n"]),
                    "n_test_or_batch": int(0.25 * r["n"]),
                    "d": r["d"],
                    "threads": 1,
                    "status": "ok",
                    "failure_stage": "",
                    "oom_confirmed": "",
                    "time_a_ms": val,
                    "time_b_ms": "",
                    "time_a_train_ms": r["stages"]["train_ms"] if stage == "train_ms" else "",
                    "time_a_predict_ms": r["stages"]["predict_ms"] if stage == "predict_ms" else "",
                    "time_a_materialize_ms": "",
                    "time_b_fit_ms": "",
                    "time_b_predict_ms": "",
                    "ratio_b_over_a": "",
                    "log2_ratio": "",
                    "timing_boundary": f"stage={stage}; total_key={r['total_key']}; sum_check={r['stage_sum_ms']:.4f} vs {r['total_ms']:.4f}; predict={r['predict_ms_includes']}",
                    "repeats": r["repeats"],
                    "warmup": 0,
                    "point_id": stage,
                    "axis": "database_resident",
                    "env_note": r.get("initial_state", ""),
                }
            )
        rows.append(
            {
                "panel": "c_pipeline_breakdown",
                "source": r["source"],
                "system_a": r["system"],
                "system_b": "",
                "algorithm": "linreg",
                "algorithm_variant": "linreg",
                "n_train": int(0.75 * r["n"]),
                "n_test_or_batch": int(0.25 * r["n"]),
                "d": r["d"],
                "threads": 1,
                "status": "ok",
                "failure_stage": "",
                "oom_confirmed": "",
                "time_a_ms": r["total_ms"],
                "time_b_ms": "",
                "time_a_train_ms": "",
                "time_a_predict_ms": "",
                "time_a_materialize_ms": "",
                "time_b_fit_ms": "",
                "time_b_predict_ms": "",
                "ratio_b_over_a": "",
                "log2_ratio": "",
                "timing_boundary": f"TOTAL {r['total_key']}; starting_state=pipe_obs in PostgreSQL",
                "repeats": r["repeats"],
                "warmup": 0,
                "point_id": "total",
                "axis": "database_resident",
                "env_note": f"r2={r['quality_r2']}",
            }
        )
    # Pack repeats (median evidence for caption; individual runs preserved)
    if pack_repeats:
        for i, t in enumerate(pack_repeats.get("database_resident_pack_ms") or []):
            rows.append(
                {
                    "panel": "c_pack_repeats",
                    "source": "results/sql_ml_database_resident.json",
                    "system_a": "pgml_sql_pack",
                    "system_b": "",
                    "algorithm": "pack",
                    "algorithm_variant": "database_resident_sql_array",
                    "n_train": pack_repeats["n"],
                    "n_test_or_batch": "",
                    "d": 20,
                    "threads": 1,
                    "status": "ok",
                    "failure_stage": "",
                    "oom_confirmed": "",
                    "time_a_ms": t,
                    "time_b_ms": "",
                    "time_a_train_ms": "",
                    "time_a_predict_ms": "",
                    "time_a_materialize_ms": "",
                    "time_b_fit_ms": "",
                    "time_b_predict_ms": "",
                    "ratio_b_over_a": "",
                    "log2_ratio": "",
                    "timing_boundary": "SQL ARRAY pack of features already in PostgreSQL",
                    "repeats": f"repeat_{i+1}_of_{pack_repeats['n_repeats']}",
                    "warmup": pack_repeats.get("warmup", 0),
                    "point_id": f"sql_pack_rep{i+1}",
                    "axis": "pack_repeats",
                    "env_note": "",
                }
            )
        for i, t in enumerate(pack_repeats.get("client_upload_pack_ms") or []):
            rows.append(
                {
                    "panel": "c_pack_repeats",
                    "source": "results/sql_ml_database_resident.json",
                    "system_a": "client_upload_pack",
                    "system_b": "",
                    "algorithm": "pack",
                    "algorithm_variant": "client_upload",
                    "n_train": pack_repeats["n"],
                    "n_test_or_batch": "",
                    "d": 20,
                    "threads": 1,
                    "status": "ok",
                    "failure_stage": "",
                    "oom_confirmed": "",
                    "time_a_ms": t,
                    "time_b_ms": "",
                    "time_a_train_ms": "",
                    "time_a_predict_ms": "",
                    "time_a_materialize_ms": "",
                    "time_b_fit_ms": "",
                    "time_b_predict_ms": "",
                    "ratio_b_over_a": "",
                    "log2_ratio": "",
                    "timing_boundary": "Python tolist + INSERT float4[][]; NumPy stack outside timer",
                    "repeats": f"repeat_{i+1}_of_{pack_repeats['n_repeats']}",
                    "warmup": pack_repeats.get("warmup", 0),
                    "point_id": f"client_pack_rep{i+1}",
                    "axis": "pack_repeats",
                    "env_note": "",
                }
            )
    return rows


def plot_panel_a(ax, cells):
    ns = NS_MAP
    ds = DS_MAP
    Z = np.full((len(ds), len(ns)), np.nan)
    status = [[None] * len(ns) for _ in ds]
    for i, d in enumerate(ds):
        for j, n in enumerate(ns):
            rec = cells.get((n, d))
            if rec is None:
                status[i][j] = "missing"
                continue
            if rec["verdict"] == "materialization_failure":
                status[i][j] = "mat_fail"
                continue
            if rec.get("speedup_vs_sklearn") is None:
                status[i][j] = "error"
                continue
            Z[i, j] = np.log2(rec["speedup_vs_sklearn"])
            status[i][j] = "ok"

    vmax = 2.0
    finite = Z[np.isfinite(Z)]
    if finite.size:
        vmax = max(2.0, float(np.ceil(np.max(np.abs(finite)))))
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(C_MISSING)

    # Draw missing background
    for i in range(len(ds)):
        for j in range(len(ns)):
            if status[i][j] == "missing":
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        facecolor=C_MISSING,
                        edgecolor="#CCCCCC",
                        linewidth=0.5,
                        hatch="////",
                        zorder=0,
                    )
                )

    im = ax.imshow(Z, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest", zorder=1)
    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels([r"$10^{3}$", r"$10^{4}$", r"$5{\cdot}10^{4}$", r"$10^{5}$", r"$2{\cdot}10^{5}$"])
    ax.set_yticks(range(len(ds)))
    ax.set_yticklabels([str(d) for d in ds])
    ax.set_xlabel(r"training rows $n$")
    ax.set_ylabel(r"features $d$")
    ax.text(
        -0.12,
        1.08,
        "(a)",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="right",
    )

    for i in range(len(ds)):
        for j in range(len(ns)):
            st = status[i][j]
            if st == "missing":
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=6.5, color="#666666", zorder=3)
            elif st == "mat_fail":
                ax.plot(j, i, marker="X", markersize=9, color=C_FAIL, markeredgewidth=1.2, zorder=4)
                # Label kept short; meaning in legend ("mat. OOM")
                pass
            elif st == "ok":
                val = Z[i, j]
                # readable label: multiplicative factor
                factor = 2 ** val
                if factor >= 1:
                    lab = f"{factor:.2f}×"
                else:
                    lab = f"{factor:.2f}×"
                color = "#111111" if abs(val) < 0.85 * vmax else "#ffffff"
                ax.text(j, i, lab, ha="center", va="center", fontsize=6.8, color=color, zorder=3)
                # mark fast-tree cells
                n = ns[j]
                if n >= TREE_FAST_N:
                    ax.plot(
                        j + 0.38,
                        i - 0.38,
                        marker="^",
                        markersize=4.5,
                        color="#333333",
                        zorder=4,
                    )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # Tick labels as multiplicative speedup T_ref/T_pgml
    tick_powers = [-2, -1, 0, 1, 2]
    tick_powers = [t for t in tick_powers if abs(t) <= vmax + 1e-9]
    cbar.set_ticks(tick_powers)
    cbar.set_ticklabels([{ -2: "0.25×", -1: "0.5×", 0: "1×", 1: "2×", 2: "4×"}[t] for t in tick_powers])
    cbar.set_label(r"$\log_2(T_{\mathrm{sk}}/T_{\mathrm{pgml}})$ (+ = pgml faster)", fontsize=6.5)
    cbar.ax.tick_params(labelsize=6.5)

    ax.set_xlim(-0.5, len(ns) - 0.5)
    ax.set_ylim(len(ds) - 0.5, -0.5)
    return im


def plot_panel_b(ax, scale_rows):
    ns = [r["n"] for r in scale_rows]
    pg = [r["pgml_total_ms"] / 1000.0 for r in scale_rows]
    sk = [r["sklearn_total_ms"] / 1000.0 for r in scale_rows]

    # Split series at algorithm switch
    exact_ns, exact_pg, exact_sk = [], [], []
    fast_ns, fast_pg, fast_sk = [], [], []
    for r, pgt, skt in zip(scale_rows, pg, sk):
        if r["algorithm_variant"] == "tree_exact":
            exact_ns.append(r["n"])
            exact_pg.append(pgt)
            exact_sk.append(skt)
        else:
            fast_ns.append(r["n"])
            fast_pg.append(pgt)
            fast_sk.append(skt)

    # Connect within variant only; bridge with dotted visual cue optional — do not mix variants
    if exact_ns:
        ax.plot(
            exact_ns,
            exact_pg,
            color=C_PGML,
            marker="o",
            markersize=5.5,
            linewidth=1.4,
            label="pgml (exact tree)",
            zorder=3,
        )
    if fast_ns:
        ax.plot(
            fast_ns,
            fast_pg,
            color=C_PGML,
            marker="^",
            markersize=6,
            linewidth=1.4,
            linestyle="--",
            label="pgml (histogram tree)",
            zorder=3,
        )
    ax.plot(
        ns,
        sk,
        color=C_SKLEARN,
        marker="s",
        markersize=5,
        linewidth=1.4,
        linestyle="-.",
        label="sklearn (in-process)",
        zorder=2,
    )

    ax.axvline(TREE_FAST_N, color="#888888", ls=":", lw=0.8, zorder=1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"training rows $n$ ($d{=}32$)")
    ax.set_ylabel("train + predict time (s)")
    ax.grid(True, which="both", ls=":", alpha=0.35)
    # Switch label near the top, just right of the threshold (avoids curve overlap)
    ax.text(
        TREE_FAST_N * 1.08,
        max(max(pg), max(sk)) * 0.92,
        "exact→hist.",
        fontsize=5.5,
        color="#555555",
        ha="left",
        va="top",
    )
    ax.text(
        -0.12,
        1.08,
        "(b)",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="right",
    )
    ax.text(
        0.02,
        0.02,
        "single run (no IQR)",
        transform=ax.transAxes,
        fontsize=6,
        color="#555555",
        va="bottom",
        ha="left",
    )


def plot_panel_c(ax, pipe_rows):
    # Groups: n=5k pgml | n=5k external | n=20k pgml | n=20k external
    order = [
        ("pgml", 5000),
        ("postgres_external", 5000),
        ("pgml", 20000),
        ("postgres_external", 20000),
    ]
    by = {(r["system"], r["n"]): r for r in pipe_rows}
    x = np.arange(len(order))
    width = 0.62

    # Stage stacking order (non-overlapping)
    # pgml: sql_pack, sql, train, predict
    # external: sql, transfer, convert, train, predict
    stage_specs = [
        ("sql_pack_ms", "SQL pack", C_PACK),
        ("sql_ms", "SQL select", C_SQL),
        ("transfer_ms", "transfer", C_TRANSFER),
        ("convert_ms", "convert", C_CONVERT),
        ("train_ms", "train", C_TRAIN),
        ("predict_ms", "predict", C_PREDICT),
    ]

    bottoms = np.zeros(len(order))
    legend_handles = []
    used = set()
    for key, label, color in stage_specs:
        vals = []
        for sys, n in order:
            vals.append(by[(sys, n)]["stages"].get(key, 0.0) / 1000.0)
        vals = np.asarray(vals, dtype=float)
        if vals.sum() <= 0:
            continue
        bars = ax.bar(
            x,
            vals,
            bottom=bottoms,
            width=width,
            color=color,
            edgecolor="#333333",
            linewidth=0.35,
            label=label,
        )
        if label not in used:
            legend_handles.append(bars)
            used.add(label)
        bottoms += vals

    # Total annotations
    for i, (sys, n) in enumerate(order):
        tot = by[(sys, n)]["total_ms"] / 1000.0
        ax.text(i, bottoms[i] + 0.008, f"{tot:.2f}s", ha="center", va="bottom", fontsize=6)

    labels = []
    for sys, n in order:
        short = "pgml" if sys == "pgml" else "PG→sklearn"
        labels.append(f"{short}\n$n={n // 1000}$k")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("pipeline time (s)")
    ax.set_ylim(0, max(bottoms) * 1.42)
    ax.grid(True, axis="y", ls=":", alpha=0.35)
    ax.text(
        -0.12,
        1.08,
        "(c)",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def write_audit(path: Path):
    text = """# Performance overview figure — data audit

Date: 2026-11-01. Sources under `results/` after protocol fix
(`results/protocol_fix_report.md`, backups in
`results/_backup_pre_protocol_20261101_113247/`). Pre-fix backups were **not**
merged into plotted values.

## Selected for the figure

### Panel (a) — `results/regime_map.json` (binary tree)
- **Task / algorithm:** binary classification, decision tree (`pgml_train_tree` /
  `pgml_train_tree_fast` vs `sklearn.tree.DecisionTreeClassifier`,
  `max_depth=8`, `min_samples_split=2`).
- **Why this algorithm:** among single-algorithm train-mode cells, binary tree
  has the densest measured $(n,d)$ coverage (scale_task × dimensionality cross),
  not because it maximises speedup.
- **Grid:** $n \\in \\{10^3,10^4,5\\cdot10^4,10^5\\}$, $d \\in \\{8,32,128\\}$ plus
  memory probe $(2\\cdot10^5,128)$. **Not** a full Cartesian product; missing
  cells marked `n/a` (no interpolation).
- **Duplicate $(10^4,32)$:** both `scale_task` and `dimensionality` exist;
  **scale_task** retained for consistency with other $d{=}32$ points.
- **Timing boundary:** `total = train + predict` on the first
  $\\min(n,10^4)$ rows. **Excludes** `float4[][]` materialization
  (`timing_scope_global.is_full_end_to_end = false`).
- **Threads:** pinned to 1 (see `results/environment.json`).
- **Repeats:** single run; no IQR/CI.
- **Algorithm switch:** $n \\ge 30000$ uses `pgml_train_tree_fast` (histogram);
  marked with triangles in (a)/(b).
- **Failure:** $(2\\cdot10^5,128)$ is `materialization_failure` during
  `INSERT INTO rm_pack` with **oom_confirmed=true** (cgroup `oom_kill=1` + log).
  Not labelled as training OOM.

### Panel (b) — same regime_map scale_task series, $d{=}32$
- Absolute train+predict times for pgml vs in-process sklearn.
- Exact and histogram pgml variants plotted as **separate** line styles;
  points are not joined across the switch.
- Single run (no error bars).

### Panel (c) — `results/sql_ml_database_resident.json` (regression)
- **Starting state:** features/labels already in PostgreSQL (`pipe_obs` /
  `pipe_obs_feat`). Initial table load **not** timed.
- **Systems:** `pgml` vs `postgres_external` (sklearn), shared 25% hold-out
  (matching R² ≈ 0.988).
- **Sizes:** $n \\in \\{5000,20000\\}$, $d{=}20$ (only sizes present).
- **pgml stages (non-overlapping):** `sql_pack_ms` + `sql_ms` + `train_ms` +
  `predict_ms` = `full_pipeline_from_table_ms`.
- **external stages:** `sql_ms` + `transfer_ms` + `convert_ms` + `train_ms` +
  `predict_ms` = `pipeline_from_table_ms`.
- **Hybrid caveat:** pgml `predict_ms` includes `Xte.tolist` + SQL predict +
  fetch — **not** a fully database-resident end-to-end path.
- **Pack repeats** (n=5000, 1 warmup + 5): SQL pack ≈ 17–19 ms vs client_upload
  ≈ 343–370 ms; stored in CSV panel `c_pack_repeats`, not as fabricated bars.

## Excluded (with reason)

| Result | Reason |
|---|---|
| `sql_ml_pipeline.json` legacy totals for quality claims | Quality protocols differ (pgml in-sample vs external hold-out); pack is **client_upload**, not DB-resident. Timing usable only with explicit scenario labels; panel (c) uses the remasured resident file instead. |
| `sql_ml_pipeline.json` for panel (c) stacked bars | Would mis-state pack composition; resident remasure supersedes. |
| Pre-protocol backup JSON | Superseded; do not mix with patched/remeasured files. |
| `kmeans_acceleration.json` | pgml not in `exact_acceleration_shared_init` (init not injectable); would not be exact shared-center acceleration. |
| `pca_solvers.json` / `pca_transform_stages.json` | Full $n \\times d$ train grid exists, but transform timings are asymmetric (client serialization vs NumPy-ready); reserved for the dedicated PCA figure. |
| `madlib_comparison.json` | Single $(n,d)$ point; little scaling content. |
| `postgresml_comparison.json` / tree alignment | Quality/API-wall layers differ; tree ≠ identical CART without `max_features` alignment; not used in this overview. |
| `hist_gbm_comparison.json` | Separate quality–time figure; memory metrics not cross-comparable (backend contexts vs RSS). |
| `categorical_comparison.json` | Encoding/quality figure; not this overview. |
| `public_benchmarks.json`, `dbinfer_tpcds.json` | Different tasks/protocols; soft-leakage / AUROC caveats. |
| Cross-system peak RSS comparisons | `pg_backend_memory_contexts` ≠ process RSS. |

## Environment (from `results/environment.json`)
- Host: Windows 10, Intel CPU, ~32 GiB RAM; PostgreSQL 18.6 (Debian container).
- Python 3.11; sklearn 1.9.1; threads generally pinned to 1.
- Repeats policy: single_run unless a file documents warmup+5 (pack / PCA stages).
"""
    path.write_text(text, encoding="utf-8")


def write_caption(path: Path):
    caption = r"""% Written by scripts/plot_performance_overview.py
% Figure files: figures/performance_overview.pdf (mirrored to paper/figures/).
% Classification-quality tables are unchanged; this is a timing figure only.

\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{paper/figures/performance_overview.pdf}
  \caption{Performance overview for native \textsc{pgml} versus reference pipelines.
  \textbf{(a)}~Sparse regime map for binary decision trees
  ($\mathrm{max\_depth}{=}8$): colour shows
  $\log_2(T_{\mathrm{sklearn}}/T_{\mathrm{pgml}})$ for train$+$predict on the first
  $\min(n,10^{4})$ rows (materialization excluded; missing $(n,d)$ cells are unmarked
  measurements, not interpolated). Positive values favour \textsc{pgml}.
  Triangles mark the switch to histogram training for $n\ge 3\cdot10^{4}$;
  $\times$ denotes a confirmed OOM during input materialization (not training).
  \textbf{(b)}~Absolute train$+$predict time versus $n$ at $d{=}32$ for the same
  protocol (single run; exact and histogram \textsc{pgml} variants shown as separate
  series).
  \textbf{(c)}~Pipeline phase breakdown for linear regression starting from a
  row-wise PostgreSQL table ($d{=}20$, shared 25\% hold-out): SQL-side array pack /
  select, client transfer/convert (external only), train, and predict.
  \textsc{pgml} predict includes client serialization of the hold-out matrix and is
  therefore a hybrid path, not fully database-resident.
  Worker threads pinned to one; hardware and library versions as in the environment
  snapshot.}
  \label{fig:performance-overview}
\end{figure*}
"""
    path.write_text(caption, encoding="utf-8")


def main():
    setup_style()
    rm = load_json("regime_map.json")
    dr = load_json("sql_ml_database_resident.json")
    env = load_json("environment.json")

    cells = select_regime_tree_cells(rm)
    scale_rows = select_regime_scaling(rm)
    pipe_rows = select_pipeline_breakdown(dr)
    pack_repeats = dr.get("pack_repeats")

    # Sanity: stage sums
    for r in pipe_rows:
        if abs(r["stage_sum_ms"] - r["total_ms"]) > 1.0:
            raise SystemExit(
                f"stage sum mismatch for {r['system']} n={r['n']}: "
                f"{r['stage_sum_ms']} vs {r['total_ms']}"
            )

    csv_rows = collect_csv_rows(cells, scale_rows, pipe_rows, pack_repeats, env)
    csv_path = OUT / "performance_overview_data.csv"
    fieldnames = list(csv_rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(csv_rows)

    write_audit(OUT / "performance_overview_audit.md")
    write_caption(OUT / "performance_overview_caption.tex")

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.0, 2.85),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.15, 1.0, 1.05]},
    )
    plot_panel_a(axes[0], cells)
    plot_panel_b(axes[1], scale_rows)
    plot_panel_c(axes[2], pipe_rows)

    axes[1].legend(
        loc="lower right",
        frameon=True,
        fancybox=False,
        edgecolor="#CCCCCC",
        framealpha=0.95,
        fontsize=6,
        borderpad=0.3,
        handlelength=1.6,
    )

    axes[2].legend(
        loc="upper left",
        frameon=True,
        fancybox=False,
        edgecolor="#CCCCCC",
        framealpha=0.92,
        fontsize=5.5,
        ncol=1,
        borderpad=0.25,
        labelspacing=0.25,
        handlelength=1.0,
        bbox_to_anchor=(0.02, 0.98),
    )

    a_handles = [
        mpatches.Patch(facecolor=C_MISSING, edgecolor="#CCCCCC", hatch="////", label="not measured"),
        Line2D(
            [0],
            [0],
            marker="X",
            color="none",
            markerfacecolor=C_FAIL,
            markeredgecolor=C_FAIL,
            markersize=7,
            label="mat. OOM",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor="#333333",
            markersize=5,
            label="hist. tree",
        ),
    ]
    axes[0].legend(
        handles=a_handles,
        loc="lower left",
        fontsize=5.5,
        frameon=True,
        fancybox=False,
        edgecolor="#CCCCCC",
        framealpha=0.95,
        handlelength=1.2,
        borderpad=0.3,
        bbox_to_anchor=(0.0, -0.02),
        bbox_transform=axes[0].transAxes,
    )

    out_pdf = OUT / "performance_overview.pdf"
    out_svg = OUT / "performance_overview.svg"
    out_png = OUT / "performance_overview.png"
    for dest in (out_pdf, out_svg):
        fig.savefig(dest, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Mirror into paper/figures for existing paper layout convention
    for src in (
        out_pdf,
        out_svg,
        out_png,
        csv_path,
        OUT / "performance_overview_audit.md",
        OUT / "performance_overview_caption.tex",
    ):
        target = PAPER_OUT / src.name
        target.write_bytes(src.read_bytes())

    print(f"wrote {out_pdf}")
    print(f"wrote {out_svg}")
    print(f"wrote {out_png}")
    print(f"wrote {csv_path}")
    print(f"wrote {OUT / 'performance_overview_audit.md'}")
    print(f"wrote {OUT / 'performance_overview_caption.tex'}")
    print(f"mirrored into {PAPER_OUT}")


if __name__ == "__main__":
    main()
