#!/usr/bin/env python3
"""Aggregate clustering_quality_v1 results and emit LaTeX tables."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from adapters import CFG, load_canonical
from metrics import agreement_dbscan

ROOT = Path(CFG["paths"]["results"])
DATA = Path(CFG["paths"]["data"])
RAW = ROOT / "raw_results.jsonl"
PRED = ROOT / "predictions"

REAL = ["iris", "wine", "wdbc", "digits", "seeds", "penguins"]
DBSCAN_REAL = ["iris_2d", "seeds_2d", "penguins_2d"]
SYN_FAM = ["blobs", "aniso", "moons", "circles"]
IMPL_LABEL = {
    "pgml": "pgml",
    "sklearn": "scikit-learn",
    "mlpack": "mlpack",
    "julia": "Clustering.jl",
}
ALGO_LABEL = {"kmeans": "k-means", "agglomerative": "Agglomerative (complete)"}
DS_LABEL = {
    "iris": "Iris",
    "wine": "Wine",
    "wdbc": "WDBC",
    "digits": "Digits",
    "seeds": "Seeds",
    "penguins": "Penguins",
    "iris_2d": "Iris-2D",
    "seeds_2d": "Seeds-2D",
    "penguins_2d": "Penguins-2D",
    "blobs": "Gaussian blobs",
    "aniso": "Anisotropic blobs",
    "moons": "Two moons",
    "circles": "Concentric circles",
}


def load_raw():
    rows = []
    with RAW.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    # Prefer successful records when duplicates exist (e.g. transient PG failures).
    order = {"ok": 3, "undefined_metric": 2, "unavailable": 1, "failed": 0}
    best = {}
    for r in rows:
        k = r.get("run_key")
        if not k:
            p = r.get("params") or {}
            k = json.dumps(
                [r.get("dataset"), r.get("dataset_seed"), r.get("algorithm"),
                 r.get("implementation"), r.get("run_id"), p],
                sort_keys=True,
            )
        prev = best.get(k)
        if prev is None or order.get(r.get("status"), -1) > order.get(prev.get("status"), -1):
            best[k] = r
        elif order.get(r.get("status"), -1) == order.get(prev.get("status"), -1):
            best[k] = r
    return list(best.values())


def mean_std(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return None, None, 0
    arr = np.asarray(vals, dtype=float)
    if len(arr) == 1:
        return float(arr[0]), 0.0, 1
    return float(arr.mean()), float(arr.std(ddof=1)), len(arr)


def fmt_pm(mean, std, digits=3):
    if mean is None:
        return "--"
    if std is None or std == 0:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def fmt_num(v, digits=3):
    if v is None:
        return "--"
    return f"{float(v):.{digits}f}"


def write_summary_csv(rows):
    path = ROOT / "summary.csv"
    fields = [
        "dataset", "algorithm", "implementation", "agg_kind", "n_runs",
        "ARI_mean", "ARI_std", "AMI_mean", "AMI_std",
        "silhouette_mean", "silhouette_std", "silhouette_n_valid",
        "K_found_mean", "K_found_std", "noise_pct_mean", "noise_pct_std", "inertia_mean",
        "n_ok", "n_failed", "n_unavailable",
    ]
    # kmeans real: mean±std over runs
    groups = defaultdict(list)
    for r in rows:
        if r["algorithm"] == "kmeans" and r["dataset"] in REAL:
            groups[("kmeans_real", r["dataset"], r["implementation"])].append(r)
        if r["algorithm"] == "agglomerative" and r["dataset"] in REAL:
            groups[("agg_real", r["dataset"], r["implementation"])].append(r)
        # dbscan primary
        if r["algorithm"] == "dbscan":
            p = r.get("params") or {}
            if abs(float(p.get("eps", -1)) - 0.3) < 1e-12 and int(p.get("min_samples", -1)) == 5:
                fam = None
                for f in SYN_FAM:
                    if r["dataset"].startswith(f + "_seed"):
                        fam = f
                        break
                key_ds = fam if fam else r["dataset"]
                groups[("dbscan_primary", key_ds, r["implementation"])].append(r)

    out_rows = []
    for (kind, ds, impl), rs in sorted(groups.items()):
        ok = [r for r in rs if r.get("status") == "ok"]
        failed = [r for r in rs if r.get("status") == "failed"]
        unavail = [r for r in rs if r.get("status") == "unavailable"]
        ari_m, ari_s, _ = mean_std([r.get("ARI") for r in ok])
        ami_m, ami_s, _ = mean_std([r.get("AMI") for r in ok])
        sil_vals = [
            r.get("silhouette")
            for r in ok
            if r.get("silhouette_status") != "undefined_metric" and r.get("silhouette") is not None
        ]
        sil_m, sil_s, sil_n = mean_std(sil_vals)
        k_m, k_s, _ = mean_std([r.get("K_found") for r in ok])
        n_m, n_s, _ = mean_std([r.get("noise_pct") for r in ok])
        in_m, _, _ = mean_std([r.get("inertia") for r in ok])
        out_rows.append({
            "dataset": ds,
            "algorithm": kind.split("_")[0] if kind != "agg_real" else "agglomerative",
            "implementation": impl,
            "agg_kind": kind,
            "n_runs": len(ok),
            "ARI_mean": ari_m,
            "ARI_std": ari_s,
            "AMI_mean": ami_m,
            "AMI_std": ami_s,
            "silhouette_mean": sil_m,
            "silhouette_std": sil_s,
            "silhouette_n_valid": sil_n,
            "K_found_mean": k_m,
            "K_found_std": k_s,
            "noise_pct_mean": n_m,
            "noise_pct_std": n_s,
            "inertia_mean": in_m,
            "n_ok": len(ok),
            "n_failed": len(failed),
            "n_unavailable": len(unavail),
        })
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    return out_rows


def write_clustering_real_tex(summary):
    path = ROOT / "clustering_real.tex"
    by = {(r["dataset"], r["agg_kind"], r["implementation"]): r for r in summary}
    lines = [
        "% Written by experiments/clustering/clustering_quality/aggregate.py — do not edit by hand",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Clustering quality on real datasets. For $k$-means, $K$ equals the number of "
        "reference classes and values are mean $\\pm$ sample standard deviation over 10 runs "
        "with native $k$-means++ initializations (identical integer seeds do \\emph{not} imply "
        "identical starting centers across libraries; $n_{\\mathrm{init}}=1$, $\\mathrm{max\\_iter}=300$). "
        "Agglomerative clustering uses complete linkage and Euclidean distance with a single run "
        "per dataset. External class labels are descriptors for ARI/AMI, not guaranteed natural clusters.}",
        "\\label{tab:clustering-real}",
        "\\begin{tabular}{lllccc}",
        "\\toprule",
        "Dataset & Method & Implementation & ARI $\\uparrow$ & AMI $\\uparrow$ & Silhouette $\\uparrow$ \\\\",
        "\\midrule",
    ]
    notes = []
    for ds in REAL:
        first = True
        ds_name = DS_LABEL[ds]
        for impl in ("pgml", "sklearn", "mlpack", "julia"):
            r = by.get((ds, "kmeans_real", impl))
            if r is None or r["n_ok"] == 0:
                cell = ("--", "--", "--")
                notes.append(f"{ds_name}/k-means/{impl}: missing or failed")
            else:
                cell = (
                    fmt_pm(r["ARI_mean"], r["ARI_std"]),
                    fmt_pm(r["AMI_mean"], r["AMI_std"]),
                    fmt_pm(r["silhouette_mean"], r["silhouette_std"]),
                )
            ds_cell = ds_name if first else ""
            first = False
            lines.append(
                f"{ds_cell} & $k$-means & {IMPL_LABEL[impl]} & {cell[0]} & {cell[1]} & {cell[2]} \\\\"
            )
        for impl in ("pgml", "sklearn", "julia"):
            r = by.get((ds, "agg_real", impl))
            if r is None or r["n_ok"] == 0:
                cell = ("--", "--", "--")
                notes.append(f"{ds_name}/agg/{impl}: missing or failed")
            else:
                cell = (
                    fmt_num(r["ARI_mean"]),
                    fmt_num(r["AMI_mean"]),
                    fmt_num(r["silhouette_mean"]) if r["silhouette_mean"] is not None else "--",
                )
            lines.append(
                f" & Agglomerative (complete) & {IMPL_LABEL[impl]} & {cell[0]} & {cell[1]} & {cell[2]} \\\\"
            )
        lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    if notes:
        lines.append("% NOTES:")
        for n in notes:
            lines.append(f"% {n}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dbscan_quality_tex(summary):
    path = ROOT / "dbscan_quality.tex"
    by = {(r["dataset"], r["implementation"]): r for r in summary if r["agg_kind"] == "dbscan_primary"}
    order = DBSCAN_REAL + SYN_FAM
    lines = [
        "% Written by experiments/clustering/clustering_quality/aggregate.py — DBSCAN primary (eps=0.30, min_samples=5)",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{DBSCAN quality at fixed $\\varepsilon=0.30$, $\\mathrm{min\\_samples}=5$ "
        "(self included). Silhouette uses non-noise points only and is undefined unless "
        "$2 \\le K \\le n_{\\mathrm{non\\text{-}noise}}-1$ (shown as --). "
        "For synthetic datasets, $K$ is the mean number of non-noise clusters and all "
        "mean $\\pm$ sample std values are over data-generation seeds $0,\ldots,9$ "
        "(not algorithm restarts). Silhouette on Concentric circles is averaged over the "
        "8/10 seeds where it is defined; on Seeds-2D it is undefined ($K=1$). "
        "Silhouette favors compact clusters and is a poor sole criterion for moons/circles.}",
        "\\label{tab:dbscan-quality}",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Dataset & Implementation & ARI $\\uparrow$ & AMI $\\uparrow$ & Silhouette $\\uparrow$ "
        "& Mean $K$ & Noise (\\%) \\\\",
        "\\midrule",
    ]
    footnotes = []
    sil_note_done = set()
    for ds in order:
        first = True
        ds_name = DS_LABEL[ds]
        for impl in ("pgml", "sklearn", "mlpack", "julia"):
            r = by.get((ds, impl))
            ds_cell = ds_name if first else ""
            first = False
            if r is None or r["n_ok"] == 0:
                lines.append(
                    f"{ds_cell} & {IMPL_LABEL[impl]} & -- & -- & -- & -- & -- \\\\"
                )
                footnotes.append(f"{ds_name}/{IMPL_LABEL[impl]}: unavailable or failed")
                continue
            is_syn = ds in SYN_FAM
            if is_syn:
                ari = fmt_pm(r["ARI_mean"], r["ARI_std"])
                ami = fmt_pm(r["AMI_mean"], r["AMI_std"])
                if r["silhouette_mean"] is None:
                    sil = "--"
                else:
                    sil = fmt_pm(r["silhouette_mean"], r["silhouette_std"])
                kcell = fmt_pm(r["K_found_mean"], r["K_found_std"], digits=1)
                ncell = (
                    fmt_pm(r["noise_pct_mean"], r["noise_pct_std"], digits=1)
                    if r["noise_pct_mean"] is not None
                    else "--"
                )
            else:
                ari = fmt_num(r["ARI_mean"])
                ami = fmt_num(r["AMI_mean"])
                sil = fmt_num(r["silhouette_mean"]) if r["silhouette_mean"] is not None else "--"
                kcell = fmt_num(r["K_found_mean"], 0)
                ncell = fmt_num(r["noise_pct_mean"], 1) if r["noise_pct_mean"] is not None else "--"
            if r["silhouette_mean"] is None:
                key = f"{ds}:undefined"
                if key not in sil_note_done:
                    sil_note_done.add(key)
                    footnotes.append(
                        f"{ds_name}: Silhouette undefined "
                        f"(valid={r['silhouette_n_valid']}/{r['n_ok']}; typically $K=1$)."
                    )
            elif is_syn and r["silhouette_n_valid"] < r["n_ok"]:
                key = f"{ds}:partial"
                if key not in sil_note_done:
                    sil_note_done.add(key)
                    footnotes.append(
                        f"{ds_name}: Silhouette mean over "
                        f"{r['silhouette_n_valid']}/{r['n_ok']} seeds where defined."
                    )
            lines.append(
                f"{ds_cell} & {IMPL_LABEL[impl]} & {ari} & {ami} & {sil} & {kcell} & {ncell} \\\\"
            )
        lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    if footnotes:
        # Visible note under the table (no extra packages).
        note = " \\quad ".join(footnotes)
        lines.append(f"\\par\\smallskip\\footnotesize {note}")
    lines.append("\\end{table*}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dbscan_sweep_and_agreement(rows):
    # sweep csv
    sweep_path = ROOT / "dbscan_parameter_sweep.csv"
    fields = [
        "dataset", "dataset_seed", "implementation", "eps", "min_samples",
        "ARI", "AMI", "silhouette", "silhouette_status", "K_found", "noise_pct", "status",
    ]
    with sweep_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            if r["algorithm"] != "dbscan":
                continue
            p = r.get("params") or {}
            w.writerow({
                "dataset": r["dataset"],
                "dataset_seed": r.get("dataset_seed"),
                "implementation": r["implementation"],
                "eps": p.get("eps"),
                "min_samples": p.get("min_samples"),
                "ARI": r.get("ARI"),
                "AMI": r.get("AMI"),
                "silhouette": r.get("silhouette"),
                "silhouette_status": r.get("silhouette_status"),
                "K_found": r.get("K_found"),
                "noise_pct": r.get("noise_pct"),
                "status": r.get("status"),
            })

    # agreement pgml vs others for each dataset/param
    agr_path = ROOT / "dbscan_agreement.csv"
    by = defaultdict(dict)
    for r in rows:
        if r["algorithm"] != "dbscan" or r.get("status") != "ok":
            continue
        p = r.get("params") or {}
        key = (r["dataset"], float(p["eps"]), int(p["min_samples"]))
        by[key][r["implementation"]] = r

    agr_fields = [
        "dataset", "eps", "min_samples", "other_impl",
        "ari_partition", "noise_mask_agreement", "K_pgml", "K_other",
        "noise_pct_pgml", "noise_pct_other",
    ]
    with agr_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=agr_fields)
        w.writeheader()
        for (ds, eps, ms), impls in sorted(by.items()):
            if "pgml" not in impls:
                continue
            pg = impls["pgml"]
            if not pg.get("prediction_path"):
                continue
            lab_pg = np.load(pg["prediction_path"])["labels"]
            for other in ("sklearn", "mlpack", "julia"):
                if other not in impls or not impls[other].get("prediction_path"):
                    continue
                lab_o = np.load(impls[other]["prediction_path"])["labels"]
                agr = agreement_dbscan(lab_pg, lab_o)
                w.writerow({
                    "dataset": ds,
                    "eps": eps,
                    "min_samples": ms,
                    "other_impl": other,
                    "ari_partition": agr["ari_partition"],
                    "noise_mask_agreement": agr["noise_mask_agreement"],
                    "K_pgml": agr["K_a"],
                    "K_other": agr["K_b"],
                    "noise_pct_pgml": agr["noise_pct_a"],
                    "noise_pct_other": agr["noise_pct_b"],
                })


def write_report(summary, rows):
    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    n_fail = sum(1 for r in rows if r.get("status") == "failed")
    n_un = sum(1 for r in rows if r.get("status") == "unavailable")
    path = ROOT / "experiment_report.md"
    text = f"""# Clustering quality experiment report (`clustering_quality_v1`)

## Completed
- Raw runs: {len(rows)} (ok={n_ok}, failed={n_fail}, unavailable={n_un})
- Real k-means: 6 datasets × 4 implementations × 10 seeds
- Real agglomerative complete: 6 × 3 implementations
- DBSCAN: real 2D (3) + synthetic families (4×10 seeds) × parameter grid × 4 implementations

## Blocked / unavailable
- None expected if Julia+mlpack+pgml reachable. Check `errors.log` for per-run failures.
- mlpack agglomerative: intentionally unavailable (no API).

## Non-comparabilities
- See `compatibility_report.md`.
- k-means: native initializations only (pgml cannot inject centers).
- DBSCAN Silhouette on non-noise subsets; may be undefined; not a sole ranking criterion for moons/circles.
- Class labels are external descriptors, not guaranteed cluster ground truth.

## Artifacts
- `clustering_real.tex`
- `dbscan_quality.tex`
- `summary.csv`, `raw_results.jsonl`, `dbscan_parameter_sweep.csv`, `dbscan_agreement.csv`
- `predictions/`, `dataset_manifest.csv`, `environment.json`

## Reproduce
```
python experiments/clustering/clustering_quality/prepare_data.py
python experiments/clustering/clustering_quality/run_all.py
python experiments/clustering/clustering_quality/aggregate.py
```
"""
    path.write_text(text, encoding="utf-8")


def main():
    rows = load_raw()
    summary = write_summary_csv(rows)
    write_clustering_real_tex(summary)
    write_dbscan_quality_tex(summary)
    write_dbscan_sweep_and_agreement(rows)
    write_report(summary, rows)
    print(f"aggregated {len(rows)} runs -> {ROOT}", flush=True)


if __name__ == "__main__":
    main()
