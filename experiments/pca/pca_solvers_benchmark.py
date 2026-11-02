#!/usr/bin/env python3
"""PCA: pgml Jacobi vs sklearn full / covariance_eigh / randomized.

Grid: n_rows × dimensionality × n_components.
Reports train time, memory, relative reconstruction error, and Grassmann
subspace distance to a reference (sklearn full). Randomized is marked
approximate.
"""
from __future__ import annotations

import json
import struct
import time
import traceback
from pathlib import Path
_REPO = Path(__file__).resolve().parents[2]

import numpy as np
import psutil
import psycopg2
from sklearn.decomposition import PCA

OUT = Path(_REPO.joinpath('results', 'pca_solvers.json'))
PG = dict(host="localhost", port=5433, dbname="ml_test", user="postgres", password="postgres")

N_GRID = [1_000, 5_000, 20_000]
D_GRID = [16, 64, 128]
K_GRID = [2, 8, 32]
SEED = 0

SYSTEMS = [
    "pgml_jacobi",
    "sklearn_full",
    "sklearn_covariance_eigh",
    "sklearn_randomized",
]


def connect():
    conn = psycopg2.connect(**PG)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SET statement_timeout = '300s'")
    return conn, cur


def clock_ms(cur, fn):
    cur.execute("SELECT clock_timestamp()")
    t0 = cur.fetchone()[0]
    out = fn()
    cur.execute("SELECT extract(epoch FROM (clock_timestamp() - %s)) * 1000", (t0,))
    return float(cur.fetchone()[0]), out


def pg_ctx_mb(cur):
    cur.execute(
        "SELECT coalesce(sum(total_bytes),0)/(1024.0*1024.0) FROM pg_backend_memory_contexts"
    )
    return float(cur.fetchone()[0])


def peak_rss(fn):
    proc = psutil.Process()
    base = proc.memory_info().rss
    t0 = time.perf_counter()
    out = fn()
    ms = (time.perf_counter() - t0) * 1000
    after = proc.memory_info().rss
    peak = max(base, after)
    return out, ms, peak / (1024 * 1024), (peak - base) / (1024 * 1024)


def make_X(n, d, rng):
    # low-rank signal + noise so top-k subspace is meaningful
    rank = min(12, d)
    A = rng.normal(size=(n, rank)).astype(np.float64)
    B = rng.normal(size=(rank, d)).astype(np.float64)
    X = (A @ B + 0.15 * rng.normal(size=(n, d))).astype(np.float32)
    return X


def deser_pgml_pca(model: bytes):
    """Match ser_pca in pgpca.c: d,k, mean[d], evals[d], evecs[d*k] (col=component)."""
    raw = model
    # psycopg may return memoryview; skip PostgreSQL varlena header if present
    if isinstance(raw, memoryview):
        raw = raw.tobytes()
    # bytea from psycopg Binary fetch is payload without header when using memoryview
    # of bytea - actually fetchone returns bytes of the datum without 4-byte header
    # when using memoryview... In practice with psycopg2, bytea comes as memoryview
    # of user data (without varlena header).
    off = 0
    d, k = struct.unpack_from("<ii", raw, off)
    off += 8
    mean = np.frombuffer(raw, dtype="<f4", count=d, offset=off).copy()
    off += 4 * d
    evals = np.frombuffer(raw, dtype="<f4", count=d, offset=off).copy()
    off += 4 * d
    evecs = np.frombuffer(raw, dtype="<f4", count=d * k, offset=off).reshape(d, k).copy()
    return {"d": d, "k": k, "mean": mean, "evals": evals, "components": evecs.T}  # (k,d)


def orthonormalize_rows(W):
    """Orthonormal basis for rowspan of W (k,d) via thin QR on W.T."""
    Q, _ = np.linalg.qr(W.T, mode="reduced")  # (d,k)
    return Q.T  # (k,d)


def grassmann_distance(W1, W2):
    """Principal-angle / Grassmann distance between subspaces spanned by rows."""
    k = min(W1.shape[0], W2.shape[0])
    A = orthonormalize_rows(W1[:k])
    B = orthonormalize_rows(W2[:k])
    # cosines of principal angles = singular values of A @ B.T
    s = np.linalg.svd(A @ B.T, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    theta = np.arccos(s)
    return float(np.linalg.norm(theta))


def reconstruction_error(X, mean, components):
    """Relative Frobenius error of rank-k PCA reconstruction."""
    Xc = X.astype(np.float64) - mean.astype(np.float64)
    W = components.astype(np.float64)  # (k,d)
    Z = Xc @ W.T
    Xrec = Z @ W + mean.astype(np.float64)
    num = np.linalg.norm(X.astype(np.float64) - Xrec, ord="fro")
    den = np.linalg.norm(X.astype(np.float64), ord="fro")
    return float(num / max(den, 1e-30))


def run_pgml(cur, X, k):
    cur.execute("DROP TABLE IF EXISTS pca_pack")
    cur.execute("CREATE TABLE pca_pack (x float4[][])")
    cur.execute("INSERT INTO pca_pack VALUES (%s::float4[][])", (X.tolist(),))
    base = pg_ctx_mb(cur)
    train_ms, model = clock_ms(
        cur, lambda: cur.execute("SELECT pgml_pca_fit(x, %s) FROM pca_pack", (k,)) or cur.fetchone()[0]
    )
    peak = max(base, pg_ctx_mb(cur))
    t0 = time.perf_counter()
    cur.execute(
        "SELECT pgml_pca_transform(%s, %s::float4[][])",
        (psycopg2.Binary(model), X.tolist()),
    )
    Z = np.asarray(cur.fetchone()[0], dtype=np.float64)
    transform_ms = (time.perf_counter() - t0) * 1000
    parsed = deser_pgml_pca(bytes(model))
    # sanity: use deserialized components for metrics
    reco = reconstruction_error(X, parsed["mean"], parsed["components"])
    return {
        "system": "pgml_jacobi",
        "approximate": False,
        "train_ms": train_ms,
        "transform_ms": transform_ms,
        "peak_mem_mb": peak,
        "mem_delta_mb": peak - base,
        "model_bytes": len(model),
        "reconstruction_rel_frob": reco,
        "components": parsed["components"],
        "mean": parsed["mean"],
        "explained_variance": parsed["evals"][:k].tolist(),
        "Z_shape": list(Z.shape),
        "notes": "covariance + sequential Jacobi eigendecomposition",
    }


def run_sklearn(X, k, solver: str):
    name = {
        "full": "sklearn_full",
        "covariance_eigh": "sklearn_covariance_eigh",
        "randomized": "sklearn_randomized",
    }[solver]
    kwargs = dict(n_components=k, svd_solver=solver, random_state=SEED if solver == "randomized" else None)
    if solver == "randomized":
        kwargs["random_state"] = SEED
    pca = PCA(**{kk: vv for kk, vv in kwargs.items() if vv is not None})

    def fit():
        pca.fit(X)

    _, train_ms, peak, delta = peak_rss(fit)
    t0 = time.perf_counter()
    _ = pca.transform(X)
    transform_ms = (time.perf_counter() - t0) * 1000
    reco = reconstruction_error(X, pca.mean_, pca.components_)
    return {
        "system": name,
        "approximate": solver == "randomized",
        "train_ms": train_ms,
        "transform_ms": transform_ms,
        "peak_mem_mb": peak,
        "mem_delta_mb": delta,
        "model_bytes": int(pca.components_.nbytes + pca.mean_.nbytes + pca.explained_variance_.nbytes),
        "reconstruction_rel_frob": reco,
        "components": pca.components_.copy(),
        "mean": pca.mean_.copy(),
        "explained_variance": pca.explained_variance_.tolist(),
        "notes": f'sklearn PCA svd_solver="{solver}"'
        + (" (approximate)" if solver == "randomized" else ""),
    }


def configs():
    for n in N_GRID:
        for d in D_GRID:
            for k in K_GRID:
                if k >= d:
                    continue
                if k > n:
                    continue
                yield n, d, k


def main():
    conn, cur = connect()
    cells = list(configs())
    points = []
    print(f"cells={len(cells)} systems={len(SYSTEMS)}", flush=True)

    for i, (n, d, k) in enumerate(cells, 1):
        rng = np.random.default_rng(SEED + n + 31 * d + 17 * k)
        X = make_X(n, d, rng)
        print(f"[{i}/{len(cells)}] n={n} d={d} k={k}", flush=True)
        ref_components = None
        for sys in SYSTEMS:
            try:
                if sys == "pgml_jacobi":
                    row = run_pgml(cur, X, k)
                elif sys == "sklearn_full":
                    row = run_sklearn(X, k, "full")
                    ref_components = row["components"]
                elif sys == "sklearn_covariance_eigh":
                    row = run_sklearn(X, k, "covariance_eigh")
                else:
                    row = run_sklearn(X, k, "randomized")

                # subspace distance vs sklearn full (computed after full is available)
                comps = row.pop("components")
                mean = row.pop("mean")
                point = {"n": n, "d": d, "k": k, **row}
                if ref_components is not None:
                    point["subspace_dist_vs_sklearn_full"] = grassmann_distance(comps, ref_components)
                else:
                    point["subspace_dist_vs_sklearn_full"] = None  # filled in second pass for pgml
                # stash for deferred pgml distance
                point["_comps"] = comps
                points.append(point)
                print(
                    f"  {row['system']:28s} train={row['train_ms']:.1f}ms "
                    f"reco={row['reconstruction_rel_frob']:.4e} "
                    f"approx={row['approximate']}",
                    flush=True,
                )
            except Exception as e:
                print(f"  ! {sys}: {e}", flush=True)
                points.append({
                    "n": n, "d": d, "k": k, "system": sys,
                    "error": str(e), "trace": traceback.format_exc()[-500:],
                })
                try:
                    conn.close()
                except Exception:
                    pass
                time.sleep(1)
                conn, cur = connect()

        # fill pgml / others subspace dist once full exists
        full = next((p for p in points if p.get("n") == n and p.get("d") == d and p.get("k") == k
                     and p.get("system") == "sklearn_full" and "_comps" in p), None)
        if full is not None:
            for p in points:
                if p.get("n") == n and p.get("d") == d and p.get("k") == k and "_comps" in p:
                    if p["system"] == "sklearn_full":
                        p["subspace_dist_vs_sklearn_full"] = 0.0
                    else:
                        p["subspace_dist_vs_sklearn_full"] = grassmann_distance(
                            p["_comps"], full["_comps"]
                        )
                    print(
                        f"  subspace {p['system']:20s} dist={p['subspace_dist_vs_sklearn_full']:.4e}",
                        flush=True,
                    )

    # strip numpy stashes
    clean = []
    for p in points:
        p = dict(p)
        p.pop("_comps", None)
        clean.append(p)

    payload = {
        "hyperparams": {"seed": SEED, "signal_rank": 12, "noise": 0.15},
        "grid": {"n": N_GRID, "d": D_GRID, "k": K_GRID},
        "systems": SYSTEMS,
        "metrics": [
            "train_ms",
            "transform_ms",
            "mem_delta_mb",
            "reconstruction_rel_frob",
            "subspace_dist_vs_sklearn_full",
        ],
        "notes": {
            "sklearn_randomized": "approximate truncated SVD; not an exact eigendecomposition",
            "reference_subspace": "sklearn svd_solver=full",
            "pgml": "dense covariance + Jacobi (exact for the dense eigenproblem up to tolerance)",
        },
        "points": clean,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {OUT} ({len(clean)} points)", flush=True)
    conn.close()


if __name__ == "__main__":
    main()
