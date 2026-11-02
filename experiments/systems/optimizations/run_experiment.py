#!/usr/bin/env python3
"""pgml_optimizations_v2: remasure histogram / DBSCAN-grid / parallel k-means.

Protocol: 1 warmup + 5 measured repeats; alternate variant order; SQL-side
kernel timing only (pack outside timer); statement_timeout=15min for fits.
Incremental JSONL persistence under results/pgml_optimizations_v2/.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
_REPO = Path(__file__).resolve().parents[3]

import numpy as np
import psutil
import psycopg2
from sklearn.datasets import make_blobs
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(_REPO)
OUT = ROOT / "results" / "pgml_optimizations_v2"
PRED = OUT / "predictions"
LAB = OUT / "labels"
DATA = OUT / "data"
RUNS = OUT / "runs.jsonl"

PG = dict(host="localhost", port=5433, dbname="ml_test", user="postgres", password="postgres")

# --- Panel (a): same binary GBM workload as hist_gbm_comparison ---
N_TREES = 50
MAX_DEPTH = 3
LR = 0.1
HIST_BINS = 32  # fixed in pgml_train_gbm_fast / train_gbm_hist
SEED = 0
# User grid is n_train; test is 1/3 of train (≈75/25 of total).
N_TRAIN_GRID = [5000, 10000, 25000, 50000]
D_GRID = [16, 64]
FIT_TIMEOUT_MS = 15 * 60 * 1000  # 15 minutes

# --- Panel (b): same generator as rerun.timing.dbscan ---
DBSCAN_NS = [2000, 4000, 8000, 16000, 32000]
DBSCAN_EPS = 0.5
DBSCAN_MIN_PTS = 5
DBSCAN_STD = 0.35
DBSCAN_CENTERS = 4
DBSCAN_SEED = 0

# --- Panel (c) ---
KM_CONFIGS = [
    {"n": 100_000, "d": 20, "k": 8, "iters": 20},
    {"n": 500_000, "d": 20, "k": 8, "iters": 20},
]
KM_SEED = 0
N_WARMUP = 1
N_REPEATS = 5


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def connect(timeout_ms: int | None = None):
    conn = psycopg2.connect(**PG)
    conn.autocommit = True
    cur = conn.cursor()
    if timeout_ms is None:
        cur.execute("SET statement_timeout = 0")
    else:
        cur.execute("SET statement_timeout = %s", (timeout_ms,))
    cur.execute("SELECT pg_backend_pid()")
    pid = int(cur.fetchone()[0])
    return conn, cur, pid


def sha256_arr(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def append_run(row: dict):
    OUT.mkdir(parents=True, exist_ok=True)
    with RUNS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_done_keys() -> set[str]:
    keys = set()
    if not RUNS.exists():
        return keys
    for line in RUNS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("phase") == "measured" and r.get("status") in ("ok", "timeout", "error"):
            keys.add(r["run_key"])
    return keys


def clock_sql_ms(cur, sql: str, params=None):
    """SQL-side wall time around a single statement; returns (ms, fetchone_result)."""
    cur.execute("SELECT clock_timestamp()")
    t0 = cur.fetchone()[0]
    cur.execute(sql, params)
    out = cur.fetchone()[0]
    cur.execute("SELECT extract(epoch FROM (clock_timestamp() - %s)) * 1000", (t0,))
    ms = float(cur.fetchone()[0])
    return ms, out


def store_pack(cur, table: str, X, y=None, y_kind: str | None = None):
    cur.execute(f"DROP TABLE IF EXISTS {table}")
    if y is None:
        cur.execute(f"CREATE TABLE {table} (x float4[][])")
        cur.execute(f"INSERT INTO {table} VALUES (%s::float4[][])", (np.asarray(X, np.float32).tolist(),))
    elif y_kind == "int":
        cur.execute(f"CREATE TABLE {table} (x float4[][], y int[])")
        cur.execute(
            f"INSERT INTO {table} VALUES (%s::float4[][], %s::int[])",
            (np.asarray(X, np.float32).tolist(), np.asarray(y, np.int64).tolist()),
        )
    else:
        raise ValueError(y_kind)


def collect_environment(cur) -> dict:
    cur.execute("SELECT version()")
    pg_ver = cur.fetchone()[0]
    cur.execute("SHOW shared_buffers")
    shared_buffers = cur.fetchone()[0]
    cur.execute("SHOW max_parallel_workers")
    mpw = cur.fetchone()[0]
    cur.execute(
        """
        SELECT extname, extversion FROM pg_extension WHERE extname LIKE '%pgml%' OR extname='plpython3u'
        ORDER BY 1
        """
    )
    exts = cur.fetchall()
    # container cgroup if present
    cgroup = {}
    try:
        # probe via docker stats is heavy; record host view
        pass
    except Exception:
        pass
    return {
        "collected_at": utc_now(),
        "host": {
            "os": platform.platform(),
            "python": platform.python_version(),
            "cpu": platform.processor() or platform.machine(),
            "cpu_logical": psutil.cpu_count(logical=True),
            "cpu_physical": psutil.cpu_count(logical=False),
            "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        },
        "postgresql": {
            "version": pg_ver,
            "shared_buffers": shared_buffers,
            "max_parallel_workers": mpw,
            "dsn": {k: v for k, v in PG.items() if k != "password"},
            "extensions": [{"name": a, "version": b} for a, b in exts],
        },
        "threads_note": "Panel (c) requests p in {1,2,4,8}; host has >=8 logical CPUs.",
        "fit_timeout_ms": FIT_TIMEOUT_MS,
        "repeats": {"warmup": N_WARMUP, "measured": N_REPEATS},
    }


def write_protocol(env: dict):
    protocol = {
        "experiment": "pgml_optimizations_v2",
        "created_at": utc_now(),
        "panels": {
            "a_histogram": {
                "task": "binary_gbm",
                "exact_sql": "SELECT pgml_train_gbm(x,y,n_trees,depth,lr) FROM pack",
                "hist_sql": "SELECT pgml_train_gbm_fast(x,y,n_trees,depth,lr) FROM pack",
                "exact_path_note": "pgml_train_gbm → train_gbm (sort-based); no n-based switch to hist",
                "hist_bins_fixed": HIST_BINS,
                "n_trees": N_TREES,
                "max_depth": MAX_DEPTH,
                "learning_rate": LR,
                "seed": SEED,
                "n_train_grid": N_TRAIN_GRID,
                "d_grid": D_GRID,
                "split": "generate n_train + n_test with n_test=n_train//3; same rng stream as make_data",
                "timing_boundary": "SQL clock_timestamp around train only; pack outside",
                "quality": ["accuracy", "balanced_accuracy", "f1_macro"],
                "legacy_note": (
                    "hist_gbm_comparison.json used n as TOTAL rows before 75/25 split "
                    "(n=5000 → n_train=3750). Those points are legacy/additional, not this grid."
                ),
            },
            "b_dbscan": {
                "brute_sql": "SELECT pgml_dbscan(x, eps, min_pts) FROM pack",
                "grid_sql": "SELECT pgml_dbscan_fast(x, eps, min_pts) FROM pack",
                "data": f"make_blobs(n, centers={DBSCAN_CENTERS}, cluster_std={DBSCAN_STD}, random_state={DBSCAN_SEED})",
                "eps": DBSCAN_EPS,
                "min_samples": DBSCAN_MIN_PTS,
                "metric": "euclidean (squared L2 threshold eps^2); point counts itself as neighbor",
                "n_grid": DBSCAN_NS,
                "timing_boundary": "SQL clock around clustering only; pack outside",
                "agreement": "ARI(brute,grid) and noise-mask agreement; not ground-truth quality",
            },
            "c_kmeans": {
                "sql": "SELECT pgml_kmeans_parallel(x, k, iters, seed, n_threads) FROM pack",
                "configs": KM_CONFIGS,
                "seed": KM_SEED,
                "init": "k-means++ with rng_seed(seed); centers not injectable via public API",
                "threads": [1, 2, 4, 8],
                "timing_boundary": "SQL clock around train; float4[][] pre-materialized",
                "stop_rule": "fixed iteration count (no early exit in C implementation)",
            },
        },
        "common": {
            "warmup": N_WARMUP,
            "measured_repeats": N_REPEATS,
            "alternation": "measured repeats alternate which variant runs first",
            "sequential": True,
            "environment": "see environment.json",
        },
        "environment_snapshot_keys": list(env.keys()),
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    (OUT / "environment.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")


# -------------------- Panel A --------------------

def make_gbm_data(n_train: int, d: int, seed: int):
    n_test = max(1, n_train // 3)
    n_total = n_train + n_test
    rng = np.random.default_rng(seed + n_train + 17 * d)
    X = rng.normal(size=(n_total, d)).astype(np.float32)
    z = X[:, 0] + 0.5 * X[:, 1] * X[:, min(2, d - 1)] - 0.3 * X[:, min(3, d - 1)] ** 2
    p = 1.0 / (1.0 + np.exp(-z))
    y = (rng.random(n_total) < p).astype(np.int32)
    # Deterministic split preserving sizes exactly via indices
    idx = np.arange(n_total)
    rng2 = np.random.default_rng(SEED)
    rng2.shuffle(idx)
    # stratify-like: keep class balance approximately by shuffle then cut
    tr_idx, te_idx = idx[:n_train], idx[n_train:]
    # Prefer sklearn stratified split when possible for quality fairness
    try:
        Xtr, Xte, ytr, yte, id_tr, id_te = train_test_split(
            X,
            y,
            idx,
            train_size=n_train,
            test_size=n_test,
            random_state=SEED,
            stratify=y,
        )
    except ValueError:
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]
        id_tr, id_te = tr_idx, te_idx
    assert len(Xtr) == n_train and len(Xte) == n_test
    meta = {
        "n_train": int(n_train),
        "n_test": int(n_test),
        "n_total": int(n_total),
        "d": int(d),
        "seed_data": int(seed + n_train + 17 * d),
        "seed_split": SEED,
        "X_train_sha256": sha256_arr(Xtr),
        "y_train_sha256": sha256_arr(ytr),
        "X_test_sha256": sha256_arr(Xte),
        "y_test_sha256": sha256_arr(yte),
        "test_ids_sha256": sha256_arr(np.asarray(id_te, np.int64)),
        "test_ids_sum": int(np.asarray(id_te, np.int64).sum()),
    }
    return Xtr, ytr, Xte, yte, np.asarray(id_te, np.int64), meta


def quality_binary(y_true, pred):
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
    }


def run_gbm_once(cur, variant: str, Xtr, ytr, Xte, yte, id_te, meta, phase: str, rep: int, order_pos: int):
    run_key = f"a|{meta['n_train']}|{meta['d']}|{variant}|{phase}|{rep}"
    table = "opt_a_pack"
    t_pack0 = time.perf_counter()
    store_pack(cur, table, Xtr, ytr, y_kind="int")
    pack_ms = (time.perf_counter() - t_pack0) * 1000
    if variant == "exact":
        sql = f"SELECT pgml_train_gbm(x, y, %s, %s, %s) FROM {table}"
        path_name = "train_gbm"
    else:
        sql = f"SELECT pgml_train_gbm_fast(x, y, %s, %s, %s) FROM {table}"
        path_name = "train_gbm_hist"
    row = {
        "panel": "a_histogram",
        "run_key": run_key,
        "variant": variant,
        "phase": phase,
        "repeat": rep,
        "order_pos": order_pos,
        "n_train": meta["n_train"],
        "n_test": meta["n_test"],
        "d": meta["d"],
        "n_trees": N_TREES,
        "max_depth": MAX_DEPTH,
        "learning_rate": LR,
        "hist_bins": HIST_BINS if variant == "histogram" else None,
        "c_path": path_name,
        "pack_ms_outside_timer": pack_ms,
        "timing_boundary": "SQL train only",
        "data": meta,
        "started_at": utc_now(),
    }
    try:
        ms, model = clock_sql_ms(cur, sql, (N_TREES, MAX_DEPTH, LR))
        cur.execute(
            "SELECT pgml_predict_gbm_batch(%s, %s::float4[][])",
            (psycopg2.Binary(model), np.asarray(Xte, np.float32).tolist()),
        )
        pred = np.asarray(cur.fetchone()[0], dtype=np.int32)
        q = quality_binary(yte, pred)
        row.update(
            {
                "status": "ok",
                "train_ms": ms,
                "quality": q,
                "finished_at": utc_now(),
            }
        )
        if phase == "measured":
            pred_path = PRED / f"a_n{meta['n_train']}_d{meta['d']}_{variant}_rep{rep}.npz"
            np.savez_compressed(
                pred_path,
                y_true=yte,
                y_pred=pred,
                test_ids=id_te,
            )
            row["predictions_file"] = str(pred_path.relative_to(OUT)).replace("\\", "/")
    except Exception as e:
        msg = str(e).lower()
        is_timeout = "timeout" in msg or "canceling statement" in msg
        row.update(
            {
                "status": "timeout" if is_timeout else "error",
                "error": str(e),
                "traceback": traceback.format_exc(limit=8),
                "finished_at": utc_now(),
            }
        )
        append_run(row)
        raise
    append_run(row)
    return row


def panel_a(done: set[str]):
    print("== Panel A: exact vs histogram GBM ==", flush=True)
    DATA.mkdir(parents=True, exist_ok=True)
    PRED.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for n_train in N_TRAIN_GRID:
        for d in D_GRID:
            print(f"\n--- A n_train={n_train} d={d} ---", flush=True)
            Xtr, ytr, Xte, yte, id_te, meta = make_gbm_data(n_train, d, SEED)
            hashes[f"n{n_train}_d{d}"] = meta
            (DATA / "a_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")

            # Warmup: exact then hist (once)
            for variant in ("exact", "histogram"):
                key = f"a|{n_train}|{d}|{variant}|warmup|0"
                if key in done:
                    print(f"  skip {key}", flush=True)
                    continue
                conn, cur, _ = connect(FIT_TIMEOUT_MS)
                try:
                    print(f"  warmup {variant}...", flush=True)
                    run_gbm_once(cur, variant, Xtr, ytr, Xte, yte, id_te, meta, "warmup", 0, 0)
                    done.add(key)
                except Exception as e:
                    print(f"  warmup {variant} failed: {e}", flush=True)
                    # still recorded inside run_gbm_once before re-raise? on raise before append for connect issues
                    try:
                        append_run(
                            {
                                "panel": "a_histogram",
                                "run_key": key,
                                "variant": variant,
                                "phase": "warmup",
                                "repeat": 0,
                                "n_train": n_train,
                                "d": d,
                                "status": "timeout" if "timeout" in str(e).lower() else "error",
                                "error": str(e),
                                "finished_at": utc_now(),
                            }
                        )
                        done.add(key)
                    except Exception:
                        pass
                finally:
                    try:
                        conn.close()
                    except Exception:
                        pass

            # Measured repeats with alternating order
            for rep in range(N_REPEATS):
                order = ("exact", "histogram") if rep % 2 == 0 else ("histogram", "exact")
                print(f"  measured rep={rep} order={order}", flush=True)
                for pos, variant in enumerate(order):
                    key = f"a|{n_train}|{d}|{variant}|measured|{rep}"
                    if key in done:
                        print(f"    skip {key}", flush=True)
                        continue
                    conn, cur, _ = connect(FIT_TIMEOUT_MS)
                    try:
                        row = run_gbm_once(
                            cur, variant, Xtr, ytr, Xte, yte, id_te, meta, "measured", rep, pos
                        )
                        done.add(key)
                        st = row.get("status")
                        tm = row.get("train_ms")
                        print(f"    {variant}: {st} {tm if tm is not None else ''} ms", flush=True)
                    except Exception as e:
                        print(f"    {variant} failed: {e}", flush=True)
                        done.add(key)  # failure already appended inside run_gbm_once
                    finally:
                        try:
                            conn.close()
                        except Exception:
                            pass


# -------------------- Panel B --------------------

def noise_mask_agreement(a: np.ndarray, b: np.ndarray) -> float:
    ma = a < 0
    mb = b < 0
    return float(np.mean(ma == mb))


def run_dbscan_once(cur, variant: str, X, y_true, n, phase: str, rep: int, order_pos: int):
    run_key = f"b|{n}|{variant}|{phase}|{rep}"
    table = "opt_b_pack"
    t_pack0 = time.perf_counter()
    store_pack(cur, table, X)
    pack_ms = (time.perf_counter() - t_pack0) * 1000
    sql = (
        f"SELECT pgml_dbscan(x, %s, %s) FROM {table}"
        if variant == "brute"
        else f"SELECT pgml_dbscan_fast(x, %s, %s) FROM {table}"
    )
    row = {
        "panel": "b_dbscan",
        "run_key": run_key,
        "variant": variant,
        "phase": phase,
        "repeat": rep,
        "order_pos": order_pos,
        "n": n,
        "d": 2,
        "eps": DBSCAN_EPS,
        "min_samples": DBSCAN_MIN_PTS,
        "pack_ms_outside_timer": pack_ms,
        "timing_boundary": "SQL DBSCAN only",
        "started_at": utc_now(),
    }
    try:
        ms, labels = clock_sql_ms(cur, sql, (DBSCAN_EPS, DBSCAN_MIN_PTS))
        lab = np.asarray(labels, dtype=np.int32)
        n_clusters = int(len(set(lab.tolist()) - {-1}))
        noise_frac = float(np.mean(lab < 0))
        row.update(
            {
                "status": "ok",
                "train_ms": ms,
                "n_clusters": n_clusters,
                "noise_fraction": noise_frac,
                "finished_at": utc_now(),
            }
        )
        if phase == "measured":
            path = LAB / f"b_n{n}_{variant}_rep{rep}.npz"
            np.savez_compressed(path, labels=lab, y_true=y_true)
            row["labels_file"] = str(path.relative_to(OUT)).replace("\\", "/")
            # ground-truth metrics stored separately from inter-impl agreement
            row["quality_vs_truth"] = {
                "ari": float(adjusted_rand_score(y_true, lab)),
                "note": "vs make_blobs labels; not used as inter-implementation agreement",
            }
        append_run(row)
        return row, lab
    except Exception as e:
        row.update({"status": "error", "error": str(e), "finished_at": utc_now()})
        append_run(row)
        raise


def panel_b(done: set[str]):
    print("== Panel B: DBSCAN brute vs grid ==", flush=True)
    LAB.mkdir(parents=True, exist_ok=True)
    # No statement timeout needed for these (should be seconds), but set generous
    for n in DBSCAN_NS:
        print(f"\n--- B n={n} ---", flush=True)
        X, y = make_blobs(
            n_samples=n,
            centers=DBSCAN_CENTERS,
            cluster_std=DBSCAN_STD,
            random_state=DBSCAN_SEED,
        )
        X = X.astype(np.float32)
        meta = {
            "n": n,
            "X_sha256": sha256_arr(X),
            "y_sha256": sha256_arr(np.asarray(y, np.int64)),
            "generator": "sklearn.datasets.make_blobs",
            "centers": DBSCAN_CENTERS,
            "cluster_std": DBSCAN_STD,
            "random_state": DBSCAN_SEED,
            "eps": DBSCAN_EPS,
            "min_samples": DBSCAN_MIN_PTS,
        }
        (DATA / f"b_n{n}_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

        for variant in ("brute", "grid"):
            key = f"b|{n}|{variant}|warmup|0"
            if key in done:
                continue
            conn, cur, _ = connect(None)
            try:
                print(f"  warmup {variant}", flush=True)
                run_dbscan_once(cur, variant, X, y, n, "warmup", 0, 0)
                done.add(key)
            finally:
                conn.close()

        for rep in range(N_REPEATS):
            order = ("brute", "grid") if rep % 2 == 0 else ("grid", "brute")
            labs = {}
            print(f"  measured rep={rep} order={order}", flush=True)
            for pos, variant in enumerate(order):
                key = f"b|{n}|{variant}|measured|{rep}"
                if key in done:
                    # still need label for agreement if file exists
                    path = LAB / f"b_n{n}_{variant}_rep{rep}.npz"
                    if path.exists():
                        labs[variant] = np.load(path)["labels"]
                    print(f"    skip {key}", flush=True)
                    continue
                conn, cur, _ = connect(None)
                try:
                    row, lab = run_dbscan_once(cur, variant, X, y, n, "measured", rep, pos)
                    labs[variant] = lab
                    done.add(key)
                    print(f"    {variant}: {row['train_ms']:.1f} ms clusters={row['n_clusters']}", flush=True)
                finally:
                    conn.close()
            if "brute" in labs and "grid" in labs:
                ari = float(adjusted_rand_score(labs["brute"], labs["grid"]))
                noise_agr = noise_mask_agreement(labs["brute"], labs["grid"])
                agree_key = f"b|{n}|agreement|measured|{rep}"
                if agree_key not in done:
                    append_run(
                        {
                            "panel": "b_dbscan_agreement",
                            "run_key": agree_key,
                            "phase": "measured",
                            "repeat": rep,
                            "n": n,
                            "ari_between_implementations": ari,
                            "noise_mask_agreement": noise_agr,
                            "labels_identical": bool(np.array_equal(labs["brute"], labs["grid"])),
                            "status": "ok",
                            "note": "ARI is between brute and grid, not vs ground truth",
                            "finished_at": utc_now(),
                        }
                    )
                    done.add(agree_key)
                    print(f"    agreement ARI={ari:.4f} noise={noise_agr:.4f}", flush=True)
                    if ari < 1.0 - 1e-12 or noise_agr < 1.0 - 1e-12:
                        diag = LAB / f"b_n{n}_diag_rep{rep}.npz"
                        np.savez_compressed(
                            diag,
                            brute=labs["brute"],
                            grid=labs["grid"],
                            X=X,
                            y_true=y,
                        )
                        print(f"    DIAGNOSTIC saved {diag}", flush=True)


# -------------------- Panel C --------------------

def inertia_from_labels(X: np.ndarray, labels: np.ndarray, k: int) -> float:
    inert = 0.0
    for c in range(k):
        mask = labels == c
        if not np.any(mask):
            continue
        cen = X[mask].mean(axis=0)
        diff = X[mask] - cen
        inert += float(np.sum(diff * diff))
    return inert


def run_kmeans_once(cur, X, n, d, k, iters, threads, phase: str, rep: int, order_pos: int):
    run_key = f"c|{n}|p{threads}|{phase}|{rep}"
    table = "opt_c_pack"
    # pack is prepared once outside; here assume already present — still allow insert if needed
    sql = f"SELECT pgml_kmeans_parallel(x, %s, %s, %s, %s) FROM {table}"
    row = {
        "panel": "c_kmeans",
        "run_key": run_key,
        "variant": f"p{threads}",
        "threads": threads,
        "phase": phase,
        "repeat": rep,
        "order_pos": order_pos,
        "n": n,
        "d": d,
        "k": k,
        "max_iter": iters,
        "seed": KM_SEED,
        "timing_boundary": "SQL kmeans_parallel only; array pre-materialized",
        "init_note": "k-means++ via rng_seed(seed); centers not injectable",
        "started_at": utc_now(),
    }
    try:
        ms, model = clock_sql_ms(cur, sql, (k, iters, KM_SEED, threads))
        cur.execute(
            "SELECT pgml_kmeans_predict_batch(%s, %s::float4[][])",
            (psycopg2.Binary(model), np.asarray(X, np.float32).tolist()),
        )
        lab = np.asarray(cur.fetchone()[0], dtype=np.int32)
        inert = inertia_from_labels(X, lab, k)
        row.update(
            {
                "status": "ok",
                "train_ms": ms,
                "n_iter_executed": iters,  # C always runs fixed iters
                "inertia_partition": inert,
                "finished_at": utc_now(),
            }
        )
        if phase == "measured":
            path = LAB / f"c_n{n}_p{threads}_rep{rep}.npz"
            np.savez_compressed(path, labels=lab)
            row["labels_file"] = str(path.relative_to(OUT)).replace("\\", "/")
        append_run(row)
        return row, lab
    except Exception as e:
        row.update({"status": "error", "error": str(e), "finished_at": utc_now()})
        append_run(row)
        raise


def panel_c(done: set[str]):
    print("== Panel C: parallel k-means ==", flush=True)
    LAB.mkdir(parents=True, exist_ok=True)
    logical = psutil.cpu_count(logical=True) or 1
    thread_grid = [p for p in (1, 2, 4, 8) if p <= logical]
    print(f"thread_grid={thread_grid} (logical_cpus={logical})", flush=True)

    for cfg in KM_CONFIGS:
        n, d, k, iters = cfg["n"], cfg["d"], cfg["k"], cfg["iters"]
        print(f"\n--- C n={n} d={d} k={k} iters={iters} ---", flush=True)
        # memory estimate
        bytes_est = n * d * 4
        avail = psutil.virtual_memory().available
        if bytes_est > avail * 0.5:
            print(f"  SKIP n={n}: estimated {bytes_est/1e9:.2f}GB > 50% available RAM", flush=True)
            append_run(
                {
                    "panel": "c_kmeans",
                    "run_key": f"c|{n}|skip|config|0",
                    "status": "skipped",
                    "reason": "memory_guard",
                    "n": n,
                    "bytes_est": bytes_est,
                    "available_bytes": avail,
                    "finished_at": utc_now(),
                }
            )
            continue

        rng = np.random.default_rng(KM_SEED + n)
        X = rng.normal(size=(n, d)).astype(np.float32)
        meta = {
            "n": n,
            "d": d,
            "k": k,
            "iters": iters,
            "seed": KM_SEED,
            "X_sha256": sha256_arr(X),
        }
        (DATA / f"c_n{n}_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

        conn, cur, _ = connect(None)
        try:
            print("  packing array (outside timers)...", flush=True)
            t0 = time.perf_counter()
            store_pack(cur, "opt_c_pack", X)
            pack_ms = (time.perf_counter() - t0) * 1000
            append_run(
                {
                    "panel": "c_kmeans_pack",
                    "run_key": f"c|{n}|pack|once|0",
                    "status": "ok",
                    "pack_ms": pack_ms,
                    "n": n,
                    "finished_at": utc_now(),
                }
            )
            print(f"  pack_ms={pack_ms:.1f}", flush=True)

            # Warmup each p ascending
            for threads in thread_grid:
                key = f"c|{n}|p{threads}|warmup|0"
                if key in done:
                    continue
                print(f"  warmup p={threads}", flush=True)
                run_kmeans_once(cur, X, n, d, k, iters, threads, "warmup", 0, 0)
                done.add(key)

            for rep in range(N_REPEATS):
                order = list(thread_grid) if rep % 2 == 0 else list(reversed(thread_grid))
                labs = {}
                print(f"  measured rep={rep} order={order}", flush=True)
                for pos, threads in enumerate(order):
                    key = f"c|{n}|p{threads}|measured|{rep}"
                    if key in done:
                        path = LAB / f"c_n{n}_p{threads}_rep{rep}.npz"
                        if path.exists():
                            labs[threads] = np.load(path)["labels"]
                        print(f"    skip {key}", flush=True)
                        continue
                    row, lab = run_kmeans_once(cur, X, n, d, k, iters, threads, "measured", rep, pos)
                    labs[threads] = lab
                    done.add(key)
                    print(f"    p={threads}: {row['train_ms']:.1f} ms inertia={row['inertia_partition']:.1f}", flush=True)

                # agreement vs p=1
                if 1 in labs:
                    for threads, lab in labs.items():
                        ak = f"c|{n}|agree_p{threads}|measured|{rep}"
                        if ak in done:
                            continue
                        ari = float(adjusted_rand_score(labs[1], lab))
                        append_run(
                            {
                                "panel": "c_kmeans_agreement",
                                "run_key": ak,
                                "phase": "measured",
                                "repeat": rep,
                                "n": n,
                                "threads": threads,
                                "ari_vs_p1": ari,
                                "labels_identical_vs_p1": bool(np.array_equal(labs[1], lab)),
                                "status": "ok",
                                "finished_at": utc_now(),
                            }
                        )
                        done.add(ak)
        finally:
            conn.close()


def summarize():
    """Build summary.csv and report.md from runs.jsonl."""
    rows = []
    if RUNS.exists():
        for line in RUNS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))

    # Panel A summary
    import csv
    from collections import defaultdict

    summary_rows = []

    # A: group by n_train,d,variant
    a_times = defaultdict(list)
    a_acc = defaultdict(list)
    a_status = defaultdict(list)
    for r in rows:
        if r.get("panel") != "a_histogram" or r.get("phase") != "measured":
            continue
        key = (r["n_train"], r["d"], r["variant"])
        a_status[key].append(r.get("status"))
        if r.get("status") == "ok" and "train_ms" in r:
            a_times[key].append(r["train_ms"])
            a_acc[key].append(r["quality"]["accuracy"])

    for (n_train, d) in [(n, dd) for n in N_TRAIN_GRID for dd in D_GRID]:
        for variant in ("exact", "histogram"):
            key = (n_train, d, variant)
            ts = a_times.get(key, [])
            accs = a_acc.get(key, [])
            sts = a_status.get(key, [])
            summary_rows.append(
                {
                    "panel": "a",
                    "n_train": n_train,
                    "d": d,
                    "variant": variant,
                    "n_ok": len(ts),
                    "n_timeout": sum(1 for s in sts if s == "timeout"),
                    "n_error": sum(1 for s in sts if s == "error"),
                    "median_ms": float(np.median(ts)) if ts else "",
                    "q1_ms": float(np.percentile(ts, 25)) if ts else "",
                    "q3_ms": float(np.percentile(ts, 75)) if ts else "",
                    "median_accuracy": float(np.median(accs)) if accs else "",
                }
            )
        # speedup / delta acc
        te = a_times.get((n_train, d, "exact"), [])
        th = a_times.get((n_train, d, "histogram"), [])
        ae = a_acc.get((n_train, d, "exact"), [])
        ah = a_acc.get((n_train, d, "histogram"), [])
        n_to = sum(1 for s in a_status.get((n_train, d, "exact"), []) if s == "timeout")
        if te and th:
            spd = float(np.median(te) / np.median(th))
            spd_note = "median_ratio"
        elif (not te) and th and n_to > 0:
            spd = float(FIT_TIMEOUT_MS / np.median(th))
            spd_note = "lower_bound_via_timeout"
        else:
            spd, spd_note = "", ""
        dacc = ""
        if ae and ah:
            dacc = 100.0 * (float(np.median(ah)) - float(np.median(ae)))
        summary_rows.append(
            {
                "panel": "a_pair",
                "n_train": n_train,
                "d": d,
                "variant": "speedup_exact_over_hist",
                "n_ok": "",
                "n_timeout": n_to,
                "n_error": "",
                "median_ms": "",
                "q1_ms": "",
                "q3_ms": "",
                "median_accuracy": dacc,
                "speedup": spd,
                "speedup_note": spd_note,
                "delta_acc_pp": dacc,
            }
        )

    # B
    b_times = defaultdict(list)
    for r in rows:
        if r.get("panel") != "b_dbscan" or r.get("phase") != "measured" or r.get("status") != "ok":
            continue
        b_times[(r["n"], r["variant"])].append(r["train_ms"])
    for n in DBSCAN_NS:
        for variant in ("brute", "grid"):
            ts = b_times.get((n, variant), [])
            summary_rows.append(
                {
                    "panel": "b",
                    "n": n,
                    "variant": variant,
                    "n_ok": len(ts),
                    "median_ms": float(np.median(ts)) if ts else "",
                    "q1_ms": float(np.percentile(ts, 25)) if ts else "",
                    "q3_ms": float(np.percentile(ts, 75)) if ts else "",
                }
            )
        tb = b_times.get((n, "brute"), [])
        tg = b_times.get((n, "grid"), [])
        if tb and tg:
            summary_rows.append(
                {
                    "panel": "b_pair",
                    "n": n,
                    "variant": "speedup_brute_over_grid",
                    "speedup": float(np.median(tb) / np.median(tg)),
                }
            )

    # C
    c_times = defaultdict(list)
    for r in rows:
        if r.get("panel") != "c_kmeans" or r.get("phase") != "measured" or r.get("status") != "ok":
            continue
        c_times[(r["n"], r["threads"])].append(r["train_ms"])
    for cfg in KM_CONFIGS:
        n = cfg["n"]
        t1 = c_times.get((n, 1), [])
        for p in (1, 2, 4, 8):
            ts = c_times.get((n, p), [])
            spd = float(np.median(t1) / np.median(ts)) if (t1 and ts) else ""
            summary_rows.append(
                {
                    "panel": "c",
                    "n": n,
                    "threads": p,
                    "n_ok": len(ts),
                    "median_ms": float(np.median(ts)) if ts else "",
                    "q1_ms": float(np.percentile(ts, 25)) if ts else "",
                    "q3_ms": float(np.percentile(ts, 75)) if ts else "",
                    "speedup_vs_p1": spd,
                }
            )

    # write flexible CSV
    keys = sorted({k for r in summary_rows for k in r})
    with (OUT / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary_rows)

    # report
    lines = [
        "# pgml_optimizations_v2 report",
        "",
        f"Recorded: {utc_now()}",
        "",
        "## Counts",
        f"- total run records: {len(rows)}",
        f"- measured ok: {sum(1 for r in rows if r.get('phase')=='measured' and r.get('status')=='ok')}",
        f"- timeouts: {sum(1 for r in rows if r.get('status')=='timeout')}",
        f"- errors: {sum(1 for r in rows if r.get('status')=='error')}",
        "",
        "## Panel A speedups / Δacc",
    ]
    for r in summary_rows:
        if r.get("panel") == "a_pair":
            lines.append(
                f"- n_train={r['n_train']} d={r['d']}: speedup={r.get('speedup')} ({r.get('speedup_note')}), "
                f"delta_acc_pp={r.get('delta_acc_pp')}, exact_timeouts={r.get('n_timeout')}"
            )
    lines += ["", "## Panel B speedups"]
    for r in summary_rows:
        if r.get("panel") == "b_pair":
            lines.append(f"- n={r['n']}: speedup={r.get('speedup')}")
    lines += ["", "## Panel C speedups"]
    for r in summary_rows:
        if r.get("panel") == "c" and r.get("threads"):
            lines.append(
                f"- n={r['n']} p={r['threads']}: median_ms={r.get('median_ms')} speedup={r.get('speedup_vs_p1')} n_ok={r.get('n_ok')}"
            )
    lines += [
        "",
        "## Reproduce",
        "```",
        "python experiments/systems/optimizations/run_experiment.py",
        "python scripts/plot_pgml_optimizations.py",
        "```",
        "",
        "## Limitations",
        "- Exact GBM has 15-minute statement_timeout; timeouts yield lower-bound speedup only.",
        "- k-means centers not injectable; same seed used for k-means++ init across thread counts.",
        "- k-means always executes the configured iteration count (no early stop).",
        "- Legacy hist_gbm_comparison n is total rows before split; this grid uses n_train explicitly.",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote summary.csv and report.md", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    PRED.mkdir(parents=True, exist_ok=True)
    LAB.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    conn, cur, _ = connect(None)
    env = collect_environment(cur)
    conn.close()
    write_protocol(env)

    done = load_done_keys()
    print(f"resuming with {len(done)} completed measured/warmup keys", flush=True)

    # Pilot estimate for panel A
    print("\n== Pilot: exact GBM n_train=10000 d=16 (1 timed fit) ==", flush=True)
    pilot_key = "a|pilot|10000|16|exact"
    if not any(json.loads(l).get("run_key") == pilot_key for l in (RUNS.read_text(encoding='utf-8').splitlines() if RUNS.exists() else [])):
        Xtr, ytr, Xte, yte, id_te, meta = make_gbm_data(10000, 16, SEED)
        conn, cur, _ = connect(FIT_TIMEOUT_MS)
        try:
            t0 = time.perf_counter()
            store_pack(cur, "opt_a_pack", Xtr, ytr, y_kind="int")
            ms, _ = clock_sql_ms(
                cur, "SELECT pgml_train_gbm(x, y, %s, %s, %s) FROM opt_a_pack", (N_TREES, MAX_DEPTH, LR)
            )
            append_run(
                {
                    "panel": "a_pilot",
                    "run_key": pilot_key,
                    "status": "ok",
                    "train_ms": ms,
                    "n_train": 10000,
                    "d": 16,
                    "note": "single pilot fit for cost estimation",
                    "finished_at": utc_now(),
                }
            )
            print(f"pilot exact n_train=10000 d=16: {ms/1000:.1f}s", flush=True)
        except Exception as e:
            append_run(
                {
                    "panel": "a_pilot",
                    "run_key": pilot_key,
                    "status": "timeout" if "timeout" in str(e).lower() else "error",
                    "error": str(e),
                    "finished_at": utc_now(),
                }
            )
            print(f"pilot failed: {e}", flush=True)
        finally:
            conn.close()

    panel_b(done)  # start with faster panels so partial figure is useful
    panel_c(done)
    panel_a(done)
    summarize()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
