#!/usr/bin/env python3
"""Compare pgml_dbscan_fast (grid) vs exhaustive pgml_dbscan on all 774 configs."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

from adapters import CFG, connect_pg, load_canonical
from metrics import agreement_dbscan
from run_all import dbscan_targets

ROOT = Path(CFG["paths"]["results"])
DATA = Path(CFG["paths"]["data"])
OUT = ROOT / "dbscan_fast_vs_brute"
PRED = OUT / "predictions"
EPS_GRID = CFG["dbscan"]["eps_grid"]
MS_GRID = CFG["dbscan"]["min_samples_grid"]


def call_dbscan(cur, fn: str, X32: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    cur.execute(
        f"SELECT {fn}(%s::float4[][], %s::float4, %s)",
        (X32.tolist(), float(eps), int(min_samples)),
    )
    return np.asarray(cur.fetchone()[0], dtype=np.int64)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    PRED.mkdir(parents=True, exist_ok=True)
    conn, cur = connect_pg()
    # confirm both functions exist
    cur.execute(
        "SELECT proname FROM pg_proc WHERE proname IN ('pgml_dbscan','pgml_dbscan_fast') ORDER BY 1"
    )
    print("functions:", [r[0] for r in cur.fetchall()], flush=True)

    summary_rows = []
    diffs = []
    n = 0
    t0 = time.time()
    for ds_name, dseed in dbscan_targets():
        ds = load_canonical(DATA / ds_name)
        X32 = ds["X32"]
        assert X32.shape[1] == 2, f"{ds_name} not 2D"
        for eps in EPS_GRID:
            for ms in MS_GRID:
                n += 1
                lab_b = call_dbscan(cur, "pgml_dbscan", X32, eps, ms)
                lab_f = call_dbscan(cur, "pgml_dbscan_fast", X32, eps, ms)
                agr = agreement_dbscan(lab_b, lab_f)
                identical = bool(np.array_equal(lab_b, lab_f))
                row = {
                    "dataset": ds_name,
                    "dataset_seed": dseed,
                    "eps": eps,
                    "min_samples": ms,
                    "n": int(X32.shape[0]),
                    "ari_partition": agr["ari_partition"],
                    "noise_mask_agreement": agr["noise_mask_agreement"],
                    "K_brute": agr["K_a"],
                    "K_fast": agr["K_b"],
                    "noise_pct_brute": agr["noise_pct_a"],
                    "noise_pct_fast": agr["noise_pct_b"],
                    "labels_identical": identical,
                }
                summary_rows.append(row)
                if (
                    agr["ari_partition"] < 1.0 - 1e-15
                    or agr["noise_mask_agreement"] < 1.0 - 1e-15
                    or not identical
                ):
                    # save disagreeing indices
                    only_b = np.where((lab_b == -1) & (lab_f != -1))[0]
                    only_f = np.where((lab_b != -1) & (lab_f == -1))[0]
                    label_diff = np.where(lab_b != lab_f)[0]
                    diff = {
                        **row,
                        "only_brute_noise_idx": only_b.tolist(),
                        "only_fast_noise_idx": only_f.tolist(),
                        "n_label_diff": int(label_diff.size),
                        "label_diff_idx_head": label_diff[:50].tolist(),
                    }
                    diffs.append(diff)
                    stem = f"{ds_name}__e{eps}_m{ms}".replace(".", "p")
                    np.savez_compressed(
                        PRED / f"{stem}.npz",
                        row_id=ds["row_id"],
                        labels_brute=lab_b,
                        labels_fast=lab_f,
                    )
                if n % 50 == 0:
                    print(f"  {n}/774 elapsed={time.time()-t0:.1f}s diffs={len(diffs)}", flush=True)

    fields = list(summary_rows[0].keys())
    with (OUT / "dbscan_fast_vs_brute.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary_rows)
    with (OUT / "dbscan_fast_vs_brute_diffs.json").open("w", encoding="utf-8") as f:
        json.dump(diffs, f, indent=2)

    n_ari = sum(1 for r in summary_rows if r["ari_partition"] < 1 - 1e-15)
    n_noise = sum(1 for r in summary_rows if r["noise_mask_agreement"] < 1 - 1e-15)
    n_lab = sum(1 for r in summary_rows if not r["labels_identical"])
    report = f"""# pgml_dbscan_fast vs exhaustive pgml_dbscan

Configs: **{len(summary_rows)}** (expected 774 = 43 datasets × 18 param cells).
Functions: `pgml_dbscan` (brute) vs `pgml_dbscan_fast` (2D grid; falls back to brute if d≠2).
Same canonical float32 matrices; parameters unchanged.

| Check | Count |
|---|---|
| ARI(partition) < 1 | {n_ari} |
| noise-mask agreement < 1 | {n_noise} |
| labels not bit-identical | {n_lab} |

Artifacts: `dbscan_fast_vs_brute.csv`, `dbscan_fast_vs_brute_diffs.json`, `predictions/`.
"""
    (OUT / "dbscan_fast_vs_brute_report.md").write_text(report, encoding="utf-8")
    print(report, flush=True)
    try:
        conn.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
