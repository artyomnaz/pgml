#!/usr/bin/env python3
"""Histogram GBM comparison: pgml vs sklearn / HistGBM / XGBoost hist / LightGBM.

Grid: n_rows × n_features × n_bins (bins apply to histogram implementations;
pgml_hist is fixed at HIST_BINS=32; exact sklearn/pgml have no bins axis).

Reports train time, peak RSS delta, and holdout quality. Primary plot
(quality vs train time) is produced by plot_hist_gbm.py.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
_REPO = Path(__file__).resolve().parents[2]

import lightgbm as lgb
import numpy as np
import psutil
import psycopg2
import xgboost as xgb
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

OUT = Path(_REPO.joinpath('results', 'hist_gbm_comparison.json'))
PG = dict(host="localhost", port=5433, dbname="ml_test", user="postgres", password="postgres")

# Shared boosting hyperparams — same tree *budget*, not same expressiveness.
N_TREES = 50
MAX_DEPTH = 3
LR = 0.1
SEED = 0

N_GRID = [5_000, 20_000, 50_000]
D_GRID = [16, 64]
BINS_GRID = [32, 64, 255]  # common hist defaults: pgml=32, xgb≈256, sklearn Hist=255

PGML_FIXED_BINS = 32


def connect():
    conn = psycopg2.connect(**PG)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SET statement_timeout = '600s'")
    cur.execute("SELECT pg_backend_pid()")
    pid = int(cur.fetchone()[0])
    return conn, cur, pid


def clock_ms(cur, fn):
    cur.execute("SELECT clock_timestamp()")
    t0 = cur.fetchone()[0]
    out = fn()
    cur.execute("SELECT extract(epoch FROM (clock_timestamp() - %s)) * 1000", (t0,))
    return float(cur.fetchone()[0]), out


def rss_mb(pid: int) -> float:
    try:
        return psutil.Process(pid).memory_info().rss / (1024 * 1024)
    except (psutil.Error, ProcessLookupError):
        return float("nan")


def pg_ctx_mb(cur) -> float:
    """Backend allocator footprint (usable when Postgres runs in Docker)."""
    cur.execute(
        "SELECT coalesce(sum(total_bytes), 0) / (1024.0 * 1024.0) FROM pg_backend_memory_contexts"
    )
    return float(cur.fetchone()[0])


def peak_during(pid: int, fn, sample_hz: float = 40.0):
    """Run fn; return (result, elapsed_ms, peak_rss_mb, delta_mb) via process RSS."""
    proc = psutil.Process(pid)
    base = proc.memory_info().rss
    t0 = time.perf_counter()
    out = fn()
    after = proc.memory_info().rss
    peak = max(base, after)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return out, elapsed_ms, peak / (1024 * 1024), (peak - base) / (1024 * 1024)


def make_data(task: str, n: int, d: int, rng: np.random.Generator):
    X = rng.normal(size=(n, d)).astype(np.float32)
    # mild nonlinear signal so boosting has something to fit
    z = X[:, 0] + 0.5 * X[:, 1] * X[:, min(2, d - 1)] - 0.3 * X[:, min(3, d - 1)] ** 2
    if task == "binary":
        p = 1.0 / (1.0 + np.exp(-z))
        y = (rng.random(n) < p).astype(np.int32)
    else:
        y = (z + 0.4 * rng.normal(size=n)).astype(np.float32)
    return X, y


def quality(task, y_true, pred, proba=None):
    if task == "binary":
        acc = float(accuracy_score(y_true, pred))
        auc = None
        if proba is not None:
            try:
                auc = float(roc_auc_score(y_true, proba))
            except ValueError:
                auc = None
        # Accuracy is primary so pgml (hard labels only) is comparable on the plot.
        return {"accuracy": acc, "auc": auc, "primary": acc}
    rmse = float(np.sqrt(mean_squared_error(y_true, pred)))
    r2 = float(r2_score(y_true, pred))
    return {"rmse": rmse, "r2": r2, "primary": r2}


def store_pack(cur, X, y, task):
    cur.execute("DROP TABLE IF EXISTS hgb_pack")
    if task == "binary":
        cur.execute("CREATE TABLE hgb_pack (x float4[][], y int[])")
        cur.execute(
            "INSERT INTO hgb_pack VALUES (%s::float4[][], %s::int[])",
            (np.asarray(X, np.float32).tolist(), np.asarray(y, np.int64).tolist()),
        )
    else:
        cur.execute("CREATE TABLE hgb_pack (x float4[][], y float4[])")
        cur.execute(
            "INSERT INTO hgb_pack VALUES (%s::float4[][], %s::float4[])",
            (np.asarray(X, np.float32).tolist(), np.asarray(y, np.float32).tolist()),
        )


def predict_pgml(cur, model, Xte, task):
    if task == "binary":
        cur.execute(
            "SELECT pgml_predict_gbm_batch(%s, %s::float4[][])",
            (psycopg2.Binary(model), np.asarray(Xte, np.float32).tolist()),
        )
        return np.asarray(cur.fetchone()[0], dtype=np.int32)
    cur.execute(
        "SELECT pgml_predict_gbm_regression_batch(%s, %s::float4[][])",
        (psycopg2.Binary(model), np.asarray(Xte, np.float32).tolist()),
    )
    return np.asarray(cur.fetchone()[0], dtype=np.float32)


def run_pgml(cur, pg_pid, task, Xtr, ytr, Xte, yte, path: str):
    store_pack(cur, Xtr, ytr, task)
    if task == "binary":
        sql = (
            "SELECT pgml_train_gbm_fast(x, y, %s, %s, %s) FROM hgb_pack"
            if path == "hist"
            else "SELECT pgml_train_gbm(x, y, %s, %s, %s) FROM hgb_pack"
        )
    else:
        # regression: only ordinary path exposed; no separate hist SQL entry
        if path == "hist":
            return None  # no pgml_gbm_regression_fast
        sql = "SELECT pgml_gbm_regression(x, y, %s, %s, %s) FROM hgb_pack"

    base = pg_ctx_mb(cur)
    ms, model = clock_ms(cur, lambda: cur.execute(sql, (N_TREES, MAX_DEPTH, LR)) or cur.fetchone()[0])
    peak = max(base, pg_ctx_mb(cur))
    pred = predict_pgml(cur, model, Xte, task)
    q = quality(task, yte, pred)
    return {
        "system": f"pgml_{path}",
        "bins": PGML_FIXED_BINS if path == "hist" else None,
        "train_ms": ms,
        "backend_memory_contexts_mb": peak,
        "backend_contexts_delta_mb": peak - base,
        "memory_metric": "pg_backend_memory_contexts_MB",
        "memory_note": "Not process RSS; do not compare to external peak_rss_mb",
        "quality": q,
        "notes": "SQL-side train after matrix already in hgb_pack; bins fixed at HIST_BINS=32"
        if path == "hist"
        else "sort-based exact split search",
    }


def run_sklearn_exact(task, Xtr, ytr, Xte, yte):
    pid = os_pid()
    if task == "binary":
        clf = GradientBoostingClassifier(
            n_estimators=N_TREES, max_depth=MAX_DEPTH, learning_rate=LR, random_state=SEED
        )
        _, ms, peak, delta = peak_during(pid, lambda: clf.fit(Xtr, ytr))
        pred = clf.predict(Xte)
        proba = clf.predict_proba(Xte)[:, 1]
        q = quality(task, yte, pred, proba)
    else:
        reg = GradientBoostingRegressor(
            n_estimators=N_TREES, max_depth=MAX_DEPTH, learning_rate=LR, random_state=SEED
        )
        _, ms, peak, delta = peak_during(pid, lambda: reg.fit(Xtr, ytr))
        pred = reg.predict(Xte)
        q = quality(task, yte, pred)
    return {
        "system": "sklearn_gbm",
        "bins": None,
        "train_ms": ms,
        "peak_rss_mb": peak,
        "rss_delta_mb": delta,
        "memory_metric": "process_rss_MB",
        "memory_note": "Host-process RSS; not comparable to pg_backend_memory_contexts",
        "quality": q,
        "notes": "exact greedy splits; no bins axis",
    }


def run_sklearn_hist(task, Xtr, ytr, Xte, yte, bins: int):
    pid = os_pid()
    if task == "binary":
        clf = HistGradientBoostingClassifier(
            max_iter=N_TREES,
            max_depth=MAX_DEPTH,
            learning_rate=LR,
            max_bins=bins,
            random_state=SEED,
        )
        _, ms, peak, delta = peak_during(pid, lambda: clf.fit(Xtr, ytr))
        pred = clf.predict(Xte)
        proba = clf.predict_proba(Xte)[:, 1]
        q = quality(task, yte, pred, proba)
    else:
        reg = HistGradientBoostingRegressor(
            max_iter=N_TREES,
            max_depth=MAX_DEPTH,
            learning_rate=LR,
            max_bins=bins,
            random_state=SEED,
        )
        _, ms, peak, delta = peak_during(pid, lambda: reg.fit(Xtr, ytr))
        pred = reg.predict(Xte)
        q = quality(task, yte, pred)
    return {
        "system": "sklearn_hist_gbm",
        "bins": bins,
        "train_ms": ms,
        "peak_rss_mb": peak,
        "rss_delta_mb": delta,
        "memory_metric": "process_rss_MB",
        "memory_note": "Host-process RSS; not comparable to pg_backend_memory_contexts",
        "quality": q,
        "notes": "HistGradientBoosting; max_bins",
    }


def run_xgboost(task, Xtr, ytr, Xte, yte, bins: int):
    pid = os_pid()
    params = dict(
        n_estimators=N_TREES,
        max_depth=MAX_DEPTH,
        learning_rate=LR,
        tree_method="hist",
        max_bin=bins,
        n_jobs=1,
        random_state=SEED,
    )
    if task == "binary":
        clf = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **params)
        _, ms, peak, delta = peak_during(pid, lambda: clf.fit(Xtr, ytr))
        pred = clf.predict(Xte)
        proba = clf.predict_proba(Xte)[:, 1]
        q = quality(task, yte, pred, proba)
    else:
        reg = xgb.XGBRegressor(objective="reg:squarederror", **params)
        _, ms, peak, delta = peak_during(pid, lambda: reg.fit(Xtr, ytr))
        pred = reg.predict(Xte)
        q = quality(task, yte, pred)
    return {
        "system": "xgboost_hist",
        "bins": bins,
        "train_ms": ms,
        "peak_rss_mb": peak,
        "rss_delta_mb": delta,
        "memory_metric": "process_rss_MB",
        "memory_note": "Host-process RSS; not comparable to pg_backend_memory_contexts",
        "quality": q,
        "notes": 'XGBoost CPU tree_method="hist"',
    }


def run_lightgbm(task, Xtr, ytr, Xte, yte, bins: int):
    pid = os_pid()
    # LightGBM prefers num_leaves; cap by 2^depth for depth-limited trees.
    params = dict(
        n_estimators=N_TREES,
        max_depth=MAX_DEPTH,
        num_leaves=min(31, 2**MAX_DEPTH),
        learning_rate=LR,
        max_bin=bins,
        n_jobs=1,
        random_state=SEED,
        verbosity=-1,
    )
    if task == "binary":
        clf = lgb.LGBMClassifier(objective="binary", **params)
        _, ms, peak, delta = peak_during(pid, lambda: clf.fit(Xtr, ytr))
        pred = clf.predict(Xte)
        proba = clf.predict_proba(Xte)[:, 1]
        q = quality(task, yte, pred, proba)
    else:
        reg = lgb.LGBMRegressor(objective="regression", **params)
        _, ms, peak, delta = peak_during(pid, lambda: reg.fit(Xtr, ytr))
        pred = reg.predict(Xte)
        q = quality(task, yte, pred)
    return {
        "system": "lightgbm",
        "bins": bins,
        "train_ms": ms,
        "peak_rss_mb": peak,
        "rss_delta_mb": delta,
        "memory_metric": "process_rss_MB",
        "memory_note": "Host-process RSS; not comparable to pg_backend_memory_contexts",
        "quality": q,
        "notes": "LightGBM CPU histogram",
    }


def os_pid():
    return psutil.Process().pid


def configs():
    """Yield experiment cells. Exact methods: one cell per (task,n,d).
    Hist methods: full (task,n,d,bins). pgml_hist only when bins==32.
    """
    for task in ("binary", "regression"):
        for n in N_GRID:
            for d in D_GRID:
                yield {"task": task, "n": n, "d": d, "bins": None, "family": "exact"}
                for bins in BINS_GRID:
                    yield {"task": task, "n": n, "d": d, "bins": bins, "family": "hist"}


def run_cell(cur, pg_pid, cfg, Xtr, ytr, Xte, yte):
    task = cfg["task"]
    rows = []
    if cfg["family"] == "exact":
        # Exact sort-based paths are O(n log n · d · trees); keep them on the
        # small end of the grid so the hist sweep stays tractable.
        run_pgml_exact = cfg["n"] <= 5_000
        run_sk_exact = cfg["n"] <= 20_000
        if run_pgml_exact:
            try:
                r = run_pgml(cur, pg_pid, task, Xtr, ytr, Xte, yte, "exact")
                if r:
                    rows.append(r)
            except Exception as e:
                rows.append({"system": "pgml_exact", "error": str(e), "trace": traceback.format_exc()[-500:]})
        else:
            rows.append({
                "system": "pgml_exact",
                "skipped": True,
                "reason": f"exact path skipped for n={cfg['n']} (hist is the target comparison)",
            })
        if run_sk_exact:
            try:
                rows.append(run_sklearn_exact(task, Xtr, ytr, Xte, yte))
            except Exception as e:
                rows.append({"system": "sklearn_gbm", "error": str(e)})
        else:
            rows.append({
                "system": "sklearn_gbm",
                "skipped": True,
                "reason": f"exact sklearn GBM skipped for n={cfg['n']}",
            })
        return rows

    bins = cfg["bins"]
    # pgml hist only at fixed bins and binary (no reg hist entry point)
    if bins == PGML_FIXED_BINS and task == "binary":
        try:
            r = run_pgml(cur, pg_pid, task, Xtr, ytr, Xte, yte, "hist")
            if r:
                rows.append(r)
        except Exception as e:
            rows.append({"system": "pgml_hist", "bins": bins, "error": str(e),
                         "trace": traceback.format_exc()[-500:]})
    elif bins == PGML_FIXED_BINS and task == "regression":
        rows.append({
            "system": "pgml_hist",
            "bins": bins,
            "skipped": True,
            "reason": "no pgml_gbm_regression_fast / hist entry point",
        })

    for runner, name in (
        (lambda: run_sklearn_hist(task, Xtr, ytr, Xte, yte, bins), "sklearn_hist_gbm"),
        (lambda: run_xgboost(task, Xtr, ytr, Xte, yte, bins), "xgboost_hist"),
        (lambda: run_lightgbm(task, Xtr, ytr, Xte, yte, bins), "lightgbm"),
    ):
        try:
            rows.append(runner())
        except Exception as e:
            rows.append({"system": name, "bins": bins, "error": str(e),
                         "trace": traceback.format_exc()[-400:]})
    return rows


def main():
    conn, cur, pg_pid = connect()
    points = []
    cells = list(configs())
    print(f"cells={len(cells)} trees={N_TREES} depth={MAX_DEPTH} lr={LR}")

    cache_xy = {}
    for i, cfg in enumerate(cells, 1):
        key = (cfg["task"], cfg["n"], cfg["d"])
        if key not in cache_xy:
            rng = np.random.default_rng(SEED + cfg["n"] + 17 * cfg["d"] + (0 if cfg["task"] == "binary" else 99))
            X, y = make_data(cfg["task"], cfg["n"], cfg["d"], rng)
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.25, random_state=SEED,
                stratify=y if cfg["task"] == "binary" else None,
            )
            cache_xy[key] = (Xtr, Xte, ytr, yte)
        Xtr, Xte, ytr, yte = cache_xy[key]
        print(f"[{i}/{len(cells)}] {cfg}", flush=True)
        try:
            rows = run_cell(cur, pg_pid, cfg, Xtr, ytr, Xte, yte)
        except Exception as e:
            # reconnect after server abort
            print("  cell failed:", e)
            try:
                conn.close()
            except Exception:
                pass
            time.sleep(2)
            conn, cur, pg_pid = connect()
            rows = [{"system": "cell", "error": str(e)}]
        for r in rows:
            point = {
                "task": cfg["task"],
                "n": cfg["n"],
                "d": cfg["d"],
                "bins": cfg["bins"] if cfg["bins"] is not None else r.get("bins"),
                "family": cfg["family"],
                **r,
            }
            points.append(point)
            if "error" in r:
                print(f"  ! {r['system']}: {r['error'][:120]}")
            elif r.get("skipped"):
                print(f"  - {r['system']}: skipped ({r.get('reason')})")
            else:
                q = r["quality"]["primary"]
                print(
                    f"  {r['system']:18s} bins={point['bins']!s:>4}  "
                    f"train={r['train_ms']:.1f}ms  "
                    f"dMem={r.get('rss_delta_mb', r.get('backend_contexts_delta_mb', float('nan'))):.1f}MB  "
                    f"Q={q:.4f}"
                )

    payload = {
        "hyperparams": {
            "n_trees": N_TREES,
            "max_depth": MAX_DEPTH,
            "learning_rate": LR,
            "seed": SEED,
            "note": "Equal tree budget; quality-vs-time is the primary comparison",
        },
        "grid": {"n": N_GRID, "d": D_GRID, "bins": BINS_GRID},
        "pgml_hist_bins_fixed": PGML_FIXED_BINS,
        "systems": [
            "pgml_exact",
            "pgml_hist",
            "sklearn_gbm",
            "sklearn_hist_gbm",
            "xgboost_hist",
            "lightgbm",
        ],
        "points": points,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {OUT} ({len(points)} points)")
    conn.close()


if __name__ == "__main__":
    main()
