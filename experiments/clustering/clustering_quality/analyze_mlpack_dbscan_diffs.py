#!/usr/bin/env python3
"""Explain all pgml vs mlpack DBSCAN disagreements from the quality suite."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

from adapters import CFG, load_canonical

ROOT = Path(CFG["paths"]["results"])
DATA = Path(CFG["paths"]["data"])
PRED = ROOT / "predictions"
OUT = ROOT / "verification"
AGR = ROOT / "dbscan_agreement.csv"


def load_pred(path: Path) -> np.ndarray:
    z = np.load(path, allow_pickle=True)
    return np.asarray(z["labels"], dtype=np.int64)[np.argsort(z["row_id"])]


def find_pred(impl: str, dataset: str, eps: float, ms: int) -> Path:
    # filenames like dbscan__pgml__blobs_seed2__run0__eps0p1_ms20.npz
    eps_s = str(eps).replace(".", "p")
    if eps_s.endswith("p0"):
        eps_s = eps_s[:-2] + "p0"  # keep
    # try common patterns from runner
    candidates = list(PRED.glob(f"dbscan__{impl}__{dataset}__*eps*_ms{ms}.npz"))
    # filter eps
    hit = []
    for c in candidates:
        name = c.name
        # eps0p1 / eps0p10 / eps0p3 / eps1p0
        if f"_ms{ms}." in name or name.endswith(f"_ms{ms}.npz"):
            # parse eps from name
            if "eps" in name:
                part = name.split("eps")[1].split("_ms")[0]
                val = float(part.replace("p", "."))
                if abs(val - eps) < 1e-12:
                    hit.append(c)
    if not hit:
        raise FileNotFoundError(f"no pred for {impl} {dataset} eps={eps} ms={ms}")
    return hit[0]


def neighborhood(X: np.ndarray, i: int, eps: float):
    d = np.sqrt(((X - X[i]) ** 2).sum(axis=1))
    return np.where(d <= eps)[0], d


def analyze_one(dataset: str, eps: float, ms: int, kind: str):
    ds = load_canonical(DATA / dataset)
    X = ds["X32"].astype(np.float32)
    X64 = ds["X64"]
    pg = load_pred(find_pred("pgml", dataset, eps, ms))
    ml = load_pred(find_pred("mlpack", dataset, eps, ms))
    sk = load_pred(find_pred("sklearn", dataset, eps, ms))

    # geometric cores on float32 with <= eps, including self
    dmat = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    nn = (dmat <= eps).sum(axis=1)
    cores = nn >= ms

    only_pg_noise = np.where((pg == -1) & (ml != -1))[0]
    only_ml_noise = np.where((pg != -1) & (ml == -1))[0]
    noise_agree = float(np.mean((pg == -1) == (ml == -1)))

    # cluster size histograms (non-noise)
    def sizes(lab):
        return dict(sorted(Counter(lab[lab >= 0].tolist()).items()))

    # points that change noise status: classify relative to cores/borders
    def classify(idxs, lab_assigned):
        rows = []
        for i in idxs:
            neigh, _ = neighborhood(X, i, eps)
            near_cores = [j for j in neigh if cores[j]]
            near_pg_border = [j for j in neigh if (pg[j] >= 0) and (not cores[j])]
            rows.append(
                {
                    "idx": int(i),
                    "nn_incl_self": int(nn[i]),
                    "is_geom_core": bool(cores[i]),
                    "pg_label": int(pg[i]),
                    "ml_label": int(ml[i]),
                    "sk_label": int(sk[i]),
                    "n_near_cores": len(near_cores),
                    "n_near_pg_borders": len(near_pg_border),
                    "pg_cluster_size_if_any": int((pg == pg[i]).sum()) if pg[i] >= 0 else None,
                    "ml_cluster_size_if_any": int((ml == ml[i]).sum()) if ml[i] >= 0 else None,
                }
            )
        return rows

    # Also check float64 neighbor counts for disagreeing points
    d64 = np.sqrt(((X64[:, None, :] - X64[None, :, :]) ** 2).sum(-1))
    nn64 = (d64 <= eps).sum(axis=1)

    # mlpack min_size semantics probe: clusters in pgml with size < ms
    pg_sizes = sizes(pg)
    small_pg_clusters = {k: v for k, v in pg_sizes.items() if v < ms}
    ml_sizes = sizes(ml)

    # points belonging to small pg clusters
    small_pts = []
    for cid, sz in small_pg_clusters.items():
        pts = np.where(pg == cid)[0]
        small_pts.extend(pts.tolist())
    small_pts = np.array(sorted(set(small_pts)), dtype=int)
    small_become_noise_in_ml = (
        int(np.sum(ml[small_pts] == -1)) if small_pts.size else 0
    )

    out = {
        "dataset": dataset,
        "eps": eps,
        "min_samples": ms,
        "kind": kind,
        "ari_pg_ml": float(adjusted_rand_score(pg, ml)),
        "ari_pg_sk": float(adjusted_rand_score(pg, sk)),
        "ari_ml_sk": float(adjusted_rand_score(ml, sk)),
        "noise_mask_agreement": noise_agree,
        "K_pg": int(len(set(pg.tolist()) - {-1})),
        "K_ml": int(len(set(ml.tolist()) - {-1})),
        "K_sk": int(len(set(sk.tolist()) - {-1})),
        "noise_pct_pg": float(100.0 * np.mean(pg == -1)),
        "noise_pct_ml": float(100.0 * np.mean(ml == -1)),
        "pg_cluster_sizes": pg_sizes,
        "ml_cluster_sizes": ml_sizes,
        "pg_clusters_smaller_than_min_samples": small_pg_clusters,
        "n_points_in_small_pg_clusters": int(small_pts.size),
        "small_pg_points_noise_in_ml": small_become_noise_in_ml,
        "only_pg_noise": classify(only_pg_noise, pg),
        "only_ml_noise": classify(only_ml_noise, ml),
        "nn_f32_vs_f64_on_noise_diffs": [
            {
                "idx": int(i),
                "nn32": int(nn[i]),
                "nn64": int(nn64[i]),
                "core32": bool(cores[i]),
                "core64": bool(nn64[i] >= ms),
            }
            for i in sorted(set(only_pg_noise.tolist()) | set(only_ml_noise.tolist()))
        ],
    }
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(AGR.open(encoding="utf-8")))
    ml = [r for r in rows if r["other_impl"] == "mlpack"]
    disagree = [
        r
        for r in ml
        if float(r["ari_partition"]) < 1 - 1e-12
        or float(r["noise_mask_agreement"]) < 1 - 1e-12
    ]
    noise_diff = [r for r in disagree if float(r["noise_mask_agreement"]) < 1 - 1e-12]
    ari_only = [r for r in disagree if float(r["noise_mask_agreement"]) >= 1 - 1e-12]

    detailed = []
    for r in noise_diff:
        detailed.append(
            analyze_one(r["dataset"], float(r["eps"]), int(r["min_samples"]), "noise_mask")
        )
    # sample a few ari-only for border ownership
    for r in ari_only[:8]:
        detailed.append(
            analyze_one(r["dataset"], float(r["eps"]), int(r["min_samples"]), "ari_only")
        )

    # summary stats for all 44
    summary = {
        "n_mlpack_comparisons": len(ml),
        "n_disagree": len(disagree),
        "n_noise_mask_disagree": len(noise_diff),
        "n_ari_only_identical_noise": len(ari_only),
        "noise_mask_cases": [
            {
                "dataset": r["dataset"],
                "eps": float(r["eps"]),
                "min_samples": int(r["min_samples"]),
                "ari": float(r["ari_partition"]),
                "noise_agree": float(r["noise_mask_agreement"]),
                "K_pgml": int(r["K_pgml"]),
                "K_mlpack": int(r["K_other"]),
                "noise_pct_pgml": float(r["noise_pct_pgml"]),
                "noise_pct_mlpack": float(r["noise_pct_other"]),
            }
            for r in noise_diff
        ],
        "ari_only_cases": [
            {
                "dataset": r["dataset"],
                "eps": float(r["eps"]),
                "min_samples": int(r["min_samples"]),
                "ari": float(r["ari_partition"]),
                "K": int(r["K_pgml"]),
            }
            for r in ari_only
        ],
        "detailed": detailed,
        "mlpack_api_actual": {
            "version": "4.8.0",
            "call": "mlpack.dbscan(input_=Xf, epsilon=eps, min_size=min_samples, naive=True)",
            "naive_accepted": True,
            "doc_min_size": "Minimum number of points for a cluster",
        },
    }
    (OUT / "mlpack_dbscan_diffs.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Markdown report
    lines = [
        "# pgml vs mlpack DBSCAN disagreements (774 comparisons)",
        "",
        f"- Total disagreements (ARI<1 or noise-mask<1): **{len(disagree)}**",
        f"- Noise-mask disagreements: **{len(noise_diff)}**",
        f"- Partition ARI<1 with identical noise mask: **{len(ari_only)}**",
        "",
        "## Actual mlpack 4.8.0 call used in the suite",
        "",
        "```python",
        "mlpack.dbscan(input_=Xf, epsilon=float(eps), min_size=int(min_samples), naive=True)",
        "```",
        "",
        "`naive=True` is accepted by this build (brute-force range search). "
        "No fallback occurred. `min_size` is documented as "
        "*Minimum number of points for a cluster*.",
        "",
        "## 7 noise-mask disagreements",
        "",
    ]
    for d in [x for x in detailed if x["kind"] == "noise_mask"]:
        lines.append(
            f"### {d['dataset']} eps={d['eps']} min_samples={d['min_samples']}"
        )
        lines.append(
            f"- K pgml/mlpack/sklearn = {d['K_pg']}/{d['K_ml']}/{d['K_sk']}; "
            f"noise% {d['noise_pct_pg']:.3f}/{d['noise_pct_ml']:.3f}; "
            f"ARI(pg,ml)={d['ari_pg_ml']:.6f}; ARI(pg,sk)={d['ari_pg_sk']:.6f}"
        )
        lines.append(f"- pgml cluster sizes: `{d['pg_cluster_sizes']}`")
        lines.append(f"- mlpack cluster sizes: `{d['ml_cluster_sizes']}`")
        lines.append(
            f"- pgml clusters with size < min_samples: `{d['pg_clusters_smaller_than_min_samples']}` "
            f"({d['n_points_in_small_pg_clusters']} pts; "
            f"{d['small_pg_points_noise_in_ml']} of them are noise in mlpack)"
        )
        lines.append(f"- only_pg_noise detail: `{d['only_pg_noise']}`")
        lines.append(f"- only_ml_noise detail: `{d['only_ml_noise']}`")
        lines.append(f"- nn f32 vs f64 on diffs: `{d['nn_f32_vs_f64_on_noise_diffs']}`")
        lines.append("")
    lines.append("## Interpretation notes")
    lines.append("")
    lines.append(
        "Border-point *ownership* (which cluster ID a shared border receives) "
        "cannot by itself flip noise↔non-noise. Noise-mask changes require a point "
        "to be density-reachable from a core in one implementation and not the other, "
        "or a whole component to be discarded by a cluster-size rule, or a core-test "
        "/ distance discrepancy."
    )
    lines.append("")
    (OUT / "mlpack_dbscan_diffs.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT/'mlpack_dbscan_diffs.md'} disagree={len(disagree)} noise={len(noise_diff)}")


if __name__ == "__main__":
    main()
