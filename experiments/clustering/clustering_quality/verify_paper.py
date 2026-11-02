#!/usr/bin/env python3
"""Verification analyses for clustering_quality_v1 (no parameter retuning)."""
from __future__ import annotations

import collections
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

ROOT = Path(__file__).resolve().parents[3] / "results" / "clustering_quality_v1"
PRED = ROOT / "predictions"
DATA = ROOT / "data"
OUT = ROOT / "verification"


def load_rows():
    rows = []
    with open(ROOT / "raw_results.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    order = {"ok": 3, "undefined_metric": 2, "unavailable": 1, "failed": 0}
    by = {}
    for r in rows:
        key = r.get("run_key")
        if not key:
            p = r.get("params") or {}
            key = (
                r.get("dataset"),
                r.get("dataset_seed"),
                r.get("algorithm"),
                r.get("implementation"),
                r.get("run_id"),
                json.dumps(p, sort_keys=True),
            )
        prev = by.get(key)
        if prev is None or order.get(r.get("status"), -1) > order.get(prev.get("status"), -1):
            by[key] = r
        elif order.get(r.get("status"), -1) == order.get(prev.get("status"), -1):
            by[key] = r
    return list(by.values()), len(rows)


def fam(ds: str) -> str:
    return ds.split("_seed")[0] if "_seed" in ds else ds


def counts(uniq):
    ctr = collections.Counter()
    ok = collections.Counter()
    for r in uniq:
        p = r.get("params") or {}
        if r["algorithm"] == "dbscan":
            kind = f"dbscan eps={p.get('eps')} ms={p.get('min_samples')}"
            ckey = (fam(r["dataset"]), kind, r["implementation"])
        else:
            ckey = (r["dataset"], r["algorithm"], r["implementation"])
        ctr[ckey] += 1
        if r.get("status") == "ok":
            ok[ckey] += 1
    return ctr, ok


def load_pred(path_rel: str) -> np.ndarray:
    path = Path(path_rel)
    if not path.is_absolute():
        path = ROOT / path_rel
    if not path.exists():
        path = PRED / Path(path_rel).name
    if path.suffix == ".npz":
        z = np.load(path, allow_pickle=True)
        row_id = np.asarray(z["row_id"])
        labels = np.asarray(z["labels"], dtype=np.int64)
        order = np.argsort(row_id)
        return labels[order]
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    rows.sort(key=lambda x: int(x["row_id"]))
    return np.array([int(x["label"]) for x in rows], dtype=np.int64)


def find_row(uniq, dataset, algorithm, implementation, run_id=0, eps=None, ms=None):
    for r in uniq:
        if r.get("dataset") != dataset:
            continue
        if r.get("algorithm") != algorithm:
            continue
        if r.get("implementation") != implementation:
            continue
        if int(r.get("run_id") or 0) != run_id:
            continue
        p = r.get("params") or {}
        if eps is not None and abs(float(p.get("eps", -1)) - eps) > 1e-12:
            continue
        if ms is not None and int(p.get("min_samples", -1)) != ms:
            continue
        if r.get("status") != "ok":
            continue
        return r
    return None


def agglomerative_pairwise(uniq):
    out = []
    for ds in ["iris", "wine", "wdbc", "digits", "seeds", "penguins"]:
        labs = {}
        for impl in ["pgml", "sklearn", "julia"]:
            r = find_row(uniq, ds, "agglomerative", impl)
            if r is None:
                continue
            labs[impl] = load_pred(r["prediction_path"])
        impls = list(labs.keys())
        for i, a in enumerate(impls):
            for b in impls[i + 1 :]:
                ari = adjusted_rand_score(labs[a], labs[b])
                identical = bool(np.array_equal(labs[a], labs[b]))
                # also check up to label permutation via ARI==1
                out.append(
                    {
                        "dataset": ds,
                        "a": a,
                        "b": b,
                        "ari_between": ari,
                        "identical_labels": identical,
                        "sizes_a": dict(sorted(collections.Counter(labs[a].tolist()).items())),
                        "sizes_b": dict(sorted(collections.Counter(labs[b].tolist()).items())),
                    }
                )
        # cluster sizes detail for wdbc/digits
        if ds in ("wdbc", "digits") and "pgml" in labs:
            z = np.load(DATA / ds / "canonical.npz", allow_pickle=True)
            y = np.asarray(z["y"])
            X = np.asarray(z["X_f32"], dtype=np.float32)
            print(f"\n=== {ds} agglomerative cluster sizes vs classes ===")
            print(f"  X shape={X.shape} row order check row_id monotonic={np.all(np.diff(z['row_id'])>0) or np.all(np.diff(z['row_id'])>=0)}")
            for impl, lab in labs.items():
                sizes = collections.Counter(lab.tolist())
                print(f"  {impl} sizes={dict(sorted(sizes.items()))} n={len(lab)}")
                print(f"  {impl} ARI(vs class)={adjusted_rand_score(y, lab):.6f}")
                for c in sorted(sizes):
                    mask = lab == c
                    maj = collections.Counter(y[mask].tolist()).most_common(1)[0]
                    print(f"    cluster {c}: n={sizes[c]} maj_class={maj[0]} maj_count={maj[1]}")
            # orientation sanity: pairwise complete on rows should match sklearn sizes
            from sklearn.cluster import AgglomerativeClustering
            sk = AgglomerativeClustering(n_clusters=len(set(y.tolist())), linkage="complete", metric="euclidean")
            sk_lab = sk.fit_predict(X.astype(np.float64))
            print(f"  recompute sklearn sizes={dict(sorted(collections.Counter(sk_lab.tolist()).items()))}")
            print(f"  ARI(pgml, recompute_sk)={adjusted_rand_score(labs['pgml'], sk_lab):.10f}")
    return out


def kmeans_digits_penguins(uniq):
    print("\n=== k-means Digits per-run ARI ===")
    for impl in ["pgml", "sklearn", "mlpack", "julia"]:
        vals = []
        inertias = []
        for run_id in range(10):
            r = find_row(uniq, "digits", "kmeans", impl, run_id=run_id)
            vals.append(r["ARI"])
            inertias.append(r.get("inertia"))
        print(
            f"  {impl}: ARI mean={np.mean(vals):.4f} std={np.std(vals, ddof=1):.4f} "
            f"min={min(vals):.4f} max={max(vals):.4f} "
            f"inertia mean={np.mean(inertias):.1f}"
        )
        # pairwise ARI between runs (partition diversity)
        labs = []
        for run_id in range(10):
            r = find_row(uniq, "digits", "kmeans", impl, run_id=run_id)
            labs.append(load_pred(r["prediction_path"]))
        aris = []
        for i in range(10):
            for j in range(i + 1, 10):
                aris.append(adjusted_rand_score(labs[i], labs[j]))
        print(f"    between-run partition ARI mean={np.mean(aris):.4f} min={min(aris):.4f}")

    print("\n=== k-means Penguins per-run ARI ===")
    for impl in ["pgml", "sklearn", "mlpack", "julia"]:
        vals = []
        for run_id in range(10):
            r = find_row(uniq, "penguins", "kmeans", impl, run_id=run_id)
            vals.append((run_id, r["ARI"], r.get("inertia"), r.get("K_found")))
        print(f"  {impl}:")
        for run_id, ari, inertia, kf in vals:
            print(f"    run{run_id}: ARI={ari:.4f} inertia={inertia:.2f} K={kf}")
        aris = [v[1] for v in vals]
        print(f"    mean={np.mean(aris):.4f} std={np.std(aris, ddof=1):.4f}")


def dbscan_julia_diff(uniq):
    print("\n=== DBSCAN Julia vs pgml primary (eps=0.3, ms=5) ===")
    for ds in ["iris_2d", "penguins_2d"]:
        rpg = find_row(uniq, ds, "dbscan", "pgml", eps=0.3, ms=5)
        rsk = find_row(uniq, ds, "dbscan", "sklearn", eps=0.3, ms=5)
        rjl = find_row(uniq, ds, "dbscan", "julia", eps=0.3, ms=5)
        rml = find_row(uniq, ds, "dbscan", "mlpack", eps=0.3, ms=5)
        labs = {
            "pgml": load_pred(rpg["prediction_path"]),
            "sklearn": load_pred(rsk["prediction_path"]),
            "mlpack": load_pred(rml["prediction_path"]),
            "julia": load_pred(rjl["prediction_path"]),
        }
        X = np.asarray(np.load(DATA / ds / "canonical.npz")["X_f32"], dtype=np.float32)
        print(f"\n{ds}: n={len(X)} d={X.shape[1]}")
        for name, lab in labs.items():
            noise = lab == -1
            print(
                f"  {name}: K={len(set(lab.tolist()) - {-1})} noise_n={noise.sum()} "
                f"noise%={100*noise.mean():.3f} ARI_vs_y={find_row(uniq, ds, 'dbscan', name, eps=0.3, ms=5)['ARI']:.4f}"
            )
        # disagreement masks
        for other in ["sklearn", "mlpack", "julia"]:
            a, b = labs["pgml"], labs[other]
            noise_agree = float(np.mean((a == -1) == (b == -1)))
            only_pg = np.where((a == -1) & (b != -1))[0]
            only_ot = np.where((a != -1) & (b == -1))[0]
            print(
                f"  pgml vs {other}: partition_ARI={adjusted_rand_score(a,b):.6f} "
                f"noise_agree={noise_agree:.6f} only_pg_noise={only_pg.tolist()} only_{other}_noise={only_ot.tolist()}"
            )
            # for differing points, compute neighbor counts at eps=0.3 in float32 and float64
            eps = 0.3
            idxs = sorted(set(only_pg.tolist()) | set(only_ot.tolist()))
            if not idxs:
                continue
            X32 = X.astype(np.float32)
            X64 = X32.astype(np.float64)  # expand rounded f32
            for i in idxs[:20]:
                d32 = np.sqrt(((X32 - X32[i]) ** 2).sum(axis=1))
                d64 = np.sqrt(((X64 - X64[i]) ** 2).sum(axis=1))
                nn32_le = int((d32 <= eps).sum())
                nn32_lt = int((d32 < eps).sum())
                nn64_le = int((d64 <= eps).sum())
                # exclude self neighbor count
                nn32_le_ex = nn32_le - 1
                print(
                    f"    pt {i}: labels pgml={a[i]} {other}={b[i]} "
                    f"nn<=eps f32={nn32_le} (ex_self={nn32_le_ex}) nn<eps f32={nn32_lt} nn<=eps f64={nn64_le}"
                )

    # anisotropic primary AMI tiny diffs
    print("\n=== aniso primary AMI per seed (pgml vs julia) ===")
    for seed in range(10):
        ds = f"aniso_seed{seed}"
        rpg = find_row(uniq, ds, "dbscan", "pgml", eps=0.3, ms=5)
        rjl = find_row(uniq, ds, "dbscan", "julia", eps=0.3, ms=5)
        rsk = find_row(uniq, ds, "dbscan", "sklearn", eps=0.3, ms=5)
        rml = find_row(uniq, ds, "dbscan", "mlpack", eps=0.3, ms=5)
        if not all([rpg, rjl, rsk, rml]):
            print(" missing", ds)
            continue
        labs = {k: load_pred(v["prediction_path"]) for k, v in [("pgml", rpg), ("sklearn", rsk), ("mlpack", rml), ("julia", rjl)]}
        print(
            f"  seed{seed}: AMI pgml={rpg['AMI']:.6f} sk={rsk['AMI']:.6f} ml={rml['AMI']:.6f} jl={rjl['AMI']:.6f} "
            f"ARI_pg_jl={adjusted_rand_score(labs['pgml'], labs['julia']):.6f} "
            f"noise_agree={np.mean((labs['pgml']==-1)==(labs['julia']==-1)):.6f} "
            f"K={rpg['K_found']}/{rjl['K_found']} noise%={rpg['noise_pct']:.3f}/{rjl['noise_pct']:.3f}"
        )


def agreement_primary(uniq):
    print("\n=== dbscan_agreement primary cells (eps=0.3, ms=5) ===")
    path = ROOT / "dbscan_agreement.csv"
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if abs(float(row["eps"]) - 0.3) > 1e-12 or int(row["min_samples"]) != 5:
                continue
            ds = row["dataset"]
            if ds in ("iris_2d", "penguins_2d", "seeds_2d") or ds.startswith("aniso_seed") or ds.startswith("circles_seed"):
                if row["other_impl"] == "julia" or ds in ("iris_2d", "penguins_2d"):
                    print(
                        f"  {ds} vs {row['other_impl']}: ari={float(row['ari_partition']):.6f} "
                        f"noise={float(row['noise_mask_agreement']):.6f} "
                        f"K={row['K_pgml']}/{row['K_other']} "
                        f"noise%={float(row['noise_pct_pgml']):.3f}/{float(row['noise_pct_other']):.3f}"
                    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    uniq, n_raw = load_rows()
    print(f"raw_lines={n_raw} unique={len(uniq)} status={dict(collections.Counter(r.get('status') for r in uniq))}")
    ctr, ok = counts(uniq)

    lines = ["# Run counts (unique keys, after dedupe)", ""]
    lines.append("## k-means (expect 10 ok each)")
    for ds in ["iris", "wine", "wdbc", "digits", "seeds", "penguins"]:
        for impl in ["pgml", "sklearn", "mlpack", "julia"]:
            lines.append(f"- {ds}/kmeans/{impl}: {ok[(ds,'kmeans',impl)]}/{ctr[(ds,'kmeans',impl)]}")
    lines.append("")
    lines.append("## agglomerative (expect 1 ok each)")
    for ds in ["iris", "wine", "wdbc", "digits", "seeds", "penguins"]:
        for impl in ["pgml", "sklearn", "julia"]:
            lines.append(f"- {ds}/agglomerative/{impl}: {ok[(ds,'agglomerative',impl)]}/{ctr[(ds,'agglomerative',impl)]}")
    lines.append("")
    lines.append("## DBSCAN primary eps=0.3 min_samples=5")
    for ds in ["iris_2d", "seeds_2d", "penguins_2d", "blobs", "aniso", "moons", "circles"]:
        for impl in ["pgml", "sklearn", "mlpack", "julia"]:
            k = (ds, "dbscan eps=0.3 ms=5", impl)
            lines.append(f"- {ds}/{impl}: {ok[k]}/{ctr[k]}")
    lines.append("")
    lines.append("## DBSCAN full grid (6 eps × 3 min_samples; real=18, synthetic=180)")
    for ds in ["iris_2d", "seeds_2d", "penguins_2d", "blobs", "aniso", "moons", "circles"]:
        for impl in ["pgml", "sklearn", "mlpack", "julia"]:
            tot_ok = sum(v for (d, kind, i), v in ok.items() if d == ds and i == impl and kind.startswith("dbscan"))
            tot = sum(v for (d, kind, i), v in ctr.items() if d == ds and i == impl and kind.startswith("dbscan"))
            lines.append(f"- {ds}/{impl}: {tot_ok}/{tot}")
    (OUT / "run_counts.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    pair = agglomerative_pairwise(uniq)
    with open(OUT / "agglomerative_pairwise.json", "w", encoding="utf-8") as f:
        json.dump(pair, f, indent=2)
    print("\n=== agglomerative pairwise ARI ===")
    for p in pair:
        print(f"  {p['dataset']} {p['a']}-{p['b']}: ARI={p['ari_between']:.10f} identical={p['identical_labels']}")

    kmeans_digits_penguins(uniq)
    dbscan_julia_diff(uniq)
    agreement_primary(uniq)
    print("\nWrote", OUT)


if __name__ == "__main__":
    main()
