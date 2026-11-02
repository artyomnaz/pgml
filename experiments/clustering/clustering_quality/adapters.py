#!/usr/bin/env python3
"""Library adapters for clustering_quality_v1."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import psycopg2
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans

CFG = json.loads(Path(__file__).with_name("config.json").read_text(encoding="utf-8"))
_REPO = Path(__file__).resolve().parents[3]
for _key, _raw in list(CFG["paths"].items()):
    if _key == "julia":
        _env = os.environ.get("JULIA")
        if _env:
            CFG["paths"][_key] = _env
        else:
            _found = shutil.which(str(_raw)) or shutil.which("julia")
            CFG["paths"][_key] = _found or str(_raw)
        continue
    _path = Path(_raw)
    CFG["paths"][_key] = str(_path if _path.is_absolute() else _REPO / _path)
JULIA = Path(CFG["paths"]["julia"])
JULIA_SCRIPT = Path(__file__).with_name("julia_cluster.jl")
JULIA_PROJECT = Path(__file__).resolve().parent


def connect_pg():
    pg = CFG["pg"]
    conn = psycopg2.connect(**pg)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SET statement_timeout = 0")
    return conn, cur


def load_canonical(dataset_dir: Path):
    z = np.load(dataset_dir / "canonical.npz", allow_pickle=True)
    y = z["y"]
    if y.dtype == object or y.shape == ():
        y = None
    else:
        y = np.asarray(y)
        if y.dtype == np.dtype("O") or (y.size == 1 and y.item() is None):
            y = None
    return {
        "X32": z["X_f32"].astype(np.float32),
        "X64": z["X_f64_from_f32"].astype(np.float64),
        "y": y,
        "row_id": z["row_id"].astype(np.int64),
        "meta": json.loads((dataset_dir / "meta.json").read_text(encoding="utf-8")),
    }


# -------------------- pgml --------------------

def pgml_kmeans(cur, X32: np.ndarray, k: int, max_iter: int, seed: int):
    cur.execute(
        "SELECT pgml_kmeans(%s::float4[][], %s, %s, %s)",
        (X32.tolist(), int(k), int(max_iter), int(seed)),
    )
    model = cur.fetchone()[0]
    cur.execute(
        "SELECT pgml_kmeans_predict_batch(%s, %s::float4[][])",
        (psycopg2.Binary(model), X32.tolist()),
    )
    labels = np.asarray(cur.fetchone()[0], dtype=np.int64)
    return labels, {"pgml_fn": "pgml_kmeans+predict_batch", "init": "kmeans++ LCG seed"}


def pgml_agg(cur, X32: np.ndarray, k: int):
    cur.execute(
        "SELECT pgml_hierarchical(%s::float4[][], %s, %s)",
        (X32.tolist(), int(k), 1),  # complete
    )
    labels = np.asarray(cur.fetchone()[0], dtype=np.int64)
    return labels, {"pgml_fn": "pgml_hierarchical", "linkage": "complete=1"}


def pgml_dbscan(cur, X32: np.ndarray, eps: float, min_samples: int):
    cur.execute(
        "SELECT pgml_dbscan(%s::float4[][], %s::float4, %s)",
        (X32.tolist(), float(eps), int(min_samples)),
    )
    labels = np.asarray(cur.fetchone()[0], dtype=np.int64)
    return labels, {"pgml_fn": "pgml_dbscan", "predicate": "sqdist<=eps^2", "min_includes_self": True}


# -------------------- sklearn --------------------

def sklearn_kmeans(X64, k, max_iter, seed):
    km = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=1,
        max_iter=max_iter,
        random_state=seed,
    )
    labels = km.fit_predict(X64)
    return np.asarray(labels, dtype=np.int64), {"api": "sklearn.cluster.KMeans", "n_init": 1}


def sklearn_agg(X64, k):
    model = AgglomerativeClustering(n_clusters=k, linkage="complete", metric="euclidean")
    labels = model.fit_predict(X64)
    return np.asarray(labels, dtype=np.int64), {"api": "AgglomerativeClustering complete"}


def sklearn_dbscan(X64, eps, min_samples):
    model = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    labels = model.fit_predict(X64)
    return np.asarray(labels, dtype=np.int64), {"api": "sklearn.cluster.DBSCAN", "min_includes_self": True}


# -------------------- mlpack --------------------

def mlpack_kmeans(X64, k, max_iter, seed):
    import mlpack

    # mlpack 4.8.0: algorithm="naive" is accepted (Lloyd-style).
    Xf = np.asfortranarray(X64)
    out = mlpack.kmeans(
        clusters=int(k),
        input_=Xf,
        max_iterations=int(max_iter),
        kmeans_plus_plus=True,
        allow_empty_clusters=True,
        seed=int(seed),
        algorithm="naive",
    )
    if "output" in out:
        assign = np.asarray(out["output"])
        labels = assign[:, -1] if assign.ndim == 2 else assign
    else:
        labels = np.asarray(out["assignments"]).ravel()
    return np.asarray(labels, dtype=np.int64).ravel(), {
        "api": "mlpack.kmeans",
        "version": "4.8.0",
        "kmeans_plus_plus": True,
        "algorithm": "naive",
        "allow_empty_clusters": True,
        "max_iterations": int(max_iter),
        "seed": int(seed),
    }


def mlpack_dbscan(X64, eps, min_samples):
    import mlpack

    # mlpack 4.8.0: naive=True accepted (brute-force range search).
    # min_size -> C++ minPoints: core threshold AND post-filter
    # (components with size < minPoints become noise).
    Xf = np.asfortranarray(X64)
    out = mlpack.dbscan(
        input_=Xf,
        epsilon=float(eps),
        min_size=int(min_samples),
        naive=True,
    )
    labels = np.asarray(out["assignments"], dtype=np.int64).ravel()
    return labels, {
        "api": "mlpack.dbscan",
        "version": "4.8.0",
        "epsilon": float(eps),
        "min_size": int(min_samples),
        "naive": True,
    }


# -------------------- Julia --------------------

JULIA_BATCH = Path(__file__).with_name("julia_batch.jl")


def julia_run(algo: str, X32: np.ndarray, **params):
    """Single-job wrapper (used in smoke). Prefer julia_run_batch for grids."""
    results = julia_run_batch([
        {"id": "one", "algo": algo, "X32": X32, **params}
    ])
    return results["one"]


def julia_run_batch(jobs: list[dict], work_dir: Path | None = None):
    """Run many Clustering.jl jobs in one Julia process.

    Each job: id, algo, X32, optional k/seed/eps/min_samples.
    Returns dict id -> (labels|None, info_dict)
    """
    if not JULIA.exists():
        return {
            j["id"]: (None, {"status": "unavailable", "error": f"julia not found at {JULIA}"})
            for j in jobs
        }
    own = work_dir is None
    if own:
        td_ctx = tempfile.TemporaryDirectory()
        td = Path(td_ctx.__enter__())
    else:
        td_ctx = None
        td = Path(work_dir)
        td.mkdir(parents=True, exist_ok=True)
    try:
        job_specs = []
        for j in jobs:
            x_path = td / f"{j['id']}_X.csv"
            out_path = td / f"{j['id']}_out.json"
            np.savetxt(x_path, np.asarray(j["X32"], dtype=np.float64), delimiter=",")
            spec = {
                "id": j["id"],
                "algo": j["algo"],
                "X": str(x_path),
                "out": str(out_path),
            }
            for key in ("k", "seed", "eps", "min_samples"):
                if key in j and j[key] is not None:
                    spec[key] = j[key]
            job_specs.append(spec)
        jobs_path = td / "jobs.json"
        jobs_path.write_text(json.dumps(job_specs), encoding="utf-8")
        cmd = [
            str(JULIA),
            f"--project={JULIA_PROJECT}",
            str(JULIA_BATCH),
            str(jobs_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        out = {}
        for spec in job_specs:
            out_path = Path(spec["out"])
            if not out_path.exists():
                out[spec["id"]] = (
                    None,
                    {
                        "status": "failed",
                        "error": f"julia batch missing output; exit={proc.returncode}; stderr={proc.stderr[-400:]}",
                    },
                )
                continue
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            if payload.get("status") != "ok" or payload.get("labels") is None:
                out[spec["id"]] = (
                    None,
                    {"status": payload.get("status", "failed"), "error": payload.get("error"), "julia": payload},
                )
            else:
                labels = np.asarray(payload["labels"], dtype=np.int64)
                out[spec["id"]] = (labels, {"api": f"Clustering.jl/{spec['algo']}", "julia": payload})
        return out
    finally:
        if td_ctx is not None:
            td_ctx.__exit__(None, None, None)
