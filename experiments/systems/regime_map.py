#!/usr/bin/env python3
"""Regime map: where native pgml wins, ties, loses, or OOMs.

Sparse grid over task / n / d / model / batch / feature type / structure /
serving mode / hard cases. Each point times SQL-side pgml against in-process
scikit-learn on the same arrays.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
_REPO = Path(__file__).resolve().parents[2]

import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier

OUT = Path(_REPO.joinpath('results', 'regime_map.json'))
PG = dict(host="localhost", port=5433, dbname="ml_test", user="postgres", password="postgres")

COVERAGE = {
    "tasks": ["binary_classification", "multiclass", "regression", "clustering", "pca"],
    "n_rows": [1000, 10000, 50000, 100000],
    "n_features": [8, 32, 128],
    "feature_types": ["numeric", "mixed_categorical"],
    "structure": ["single_table", "join_aggregates"],
    "mode": ["train", "batch_inference", "repeated_serving"],
    "hard_cases": ["imbalance", "high_cardinality", "collinearity", "uneven_density"],
}


def connect():
    conn = psycopg2.connect(**PG)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SET statement_timeout = '180s'")
    return conn, cur


def clock_ms(cur, fn):
    cur.execute("SELECT clock_timestamp()")
    t0 = cur.fetchone()[0]
    out = fn()
    cur.execute("SELECT extract(epoch FROM (clock_timestamp() - %s)) * 1000", (t0,))
    return float(cur.fetchone()[0]), out


def store_pack(cur, name, X, y=None, y_kind=None):
    cur.execute(f"DROP TABLE IF EXISTS {name}")
    if y is None:
        cur.execute(f"CREATE TABLE {name} (x float4[][])")
        cur.execute(f"INSERT INTO {name} VALUES (%s::float4[][])", (np.asarray(X, np.float32).tolist(),))
    elif y_kind == "int":
        cur.execute(f"CREATE TABLE {name} (x float4[][], y int[])")
        cur.execute(
            f"INSERT INTO {name} VALUES (%s::float4[][], %s::int[])",
            (np.asarray(X, np.float32).tolist(), np.asarray(y, np.int64).tolist()),
        )
    else:
        cur.execute(f"CREATE TABLE {name} (x float4[][], y float4[])")
        cur.execute(
            f"INSERT INTO {name} VALUES (%s::float4[][], %s::float4[])",
            (np.asarray(X, np.float32).tolist(), np.asarray(y, np.float32).tolist()),
        )


def verdict(pg_ms, sk_ms):
    if pg_ms is None or sk_ms is None or pg_ms <= 0 or sk_ms <= 0:
        return "error"
    ratio = pg_ms / sk_ms
    if ratio < 1 / 1.25:
        return "faster"
    if ratio > 1.25:
        return "slower"
    return "comparable"


def make_xy(task, n, d, kind="numeric", hard=None, rng=None):
    rng = rng or np.random.default_rng(0)
    X = rng.normal(size=(n, d)).astype(np.float32)
    meta = {"task": task, "n": n, "d": d, "kind": kind, "hard": hard}

    if kind == "mixed_categorical":
        n_cat = max(1, d // 2)
        for j in range(d - n_cat, d):
            X[:, j] = rng.integers(0, 8, size=n).astype(np.float32)
        meta["n_categorical"] = n_cat

    if hard == "imbalance":
        y = np.zeros(n, dtype=np.int64)
        y[rng.choice(n, size=max(1, n // 20), replace=False)] = 1
        return X, y, meta
    if hard == "high_cardinality":
        card = min(5000, max(50, n // 2))
        X[:, -1] = rng.integers(0, card, size=n).astype(np.float32)
        y = (X[:, 0] + 0.01 * X[:, -1] > 0).astype(np.int64)
        meta["card_last"] = int(card)
        return X, y, meta
    if hard == "collinearity":
        X[:, 1] = X[:, 0] + rng.normal(0, 1e-4, size=n).astype(np.float32)
        if d > 2:
            X[:, 2] = (2 * X[:, 0] - X[:, 1]).astype(np.float32)
        y = (2 * X[:, 0] + 3 * X[:, 1] + rng.normal(0, 0.5, n)).astype(np.float32)
        return X, y, meta
    if hard == "uneven_density":
        n1 = n // 3
        X = rng.uniform(-8, 8, size=(n, d)).astype(np.float32)
        X[:n1] = rng.normal(0, 0.05, size=(n1, d)).astype(np.float32)
        X[n1 : 2 * n1] = rng.normal(5, 0.05, size=(n1, d)).astype(np.float32)
        y = np.zeros(n, dtype=np.int64)
        y[n1 : 2 * n1] = 1
        y[2 * n1 :] = 2
        return X, y, meta

    if task == "binary_classification":
        y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
    elif task == "multiclass":
        y = (np.argmax(X[:, : min(4, d)], axis=1) % 4).astype(np.int64)
    elif task == "regression":
        y = (2 * X[:, 0] + 3 * X[:, 1] + rng.normal(0, 0.5, n)).astype(np.float32)
    else:
        y = None
    return X, y, meta


def cls_train_sql(model, n):
    fast = n >= 30000
    if model == "tree":
        return (
            "SELECT pgml_train_tree_fast(x, y, 8, 2) FROM rm_pack" if fast
            else "SELECT pgml_train_tree(x, y, 8, 2) FROM rm_pack"
        ), "pgml_predict_tree_batch", DecisionTreeClassifier(max_depth=8, min_samples_split=2, random_state=0)
    if model == "forest":
        return (
            "SELECT pgml_train_forest_fast(x, y, 15, 8, 2) FROM rm_pack" if fast
            else "SELECT pgml_train_forest(x, y, 15, 8, 2) FROM rm_pack"
        ), "pgml_predict_forest_batch", RandomForestClassifier(
            n_estimators=15, max_depth=8, min_samples_split=2, random_state=0
        )
    if model == "gbm":
        return (
            "SELECT pgml_train_gbm_fast(x, y, 30, 3, 0.1) FROM rm_pack",
            "pgml_predict_gbm_batch",
            GradientBoostingClassifier(n_estimators=30, max_depth=3, learning_rate=0.1, random_state=0),
        )
    if model == "svm":
        return (
            "SELECT pgml_train_svm(x, y, 0.001, 20) FROM rm_pack",
            "pgml_predict_svm_batch",
            LogisticRegression(max_iter=2000),
        )
    raise ValueError(model)


def run_point(cur, point_id, cfg):
    task = cfg["task"]
    n, d = cfg["n"], cfg["d"]
    kind = cfg.get("kind", "numeric")
    hard = cfg.get("hard")
    mode = cfg.get("mode", "train")
    model = cfg.get("model", "tree")
    batch = cfg.get("batch")
    structure = cfg.get("structure", "single_table")
    row = {"id": point_id, "config": cfg, "pgml_ms": {}, "sklearn_ms": {}, "quality": {}}

    try:
        X, y, meta = make_xy(task if hard != "uneven_density" else "multiclass", n, d, kind=kind, hard=hard)
        if hard == "uneven_density":
            task = "multiclass"
        if hard == "collinearity":
            task = "regression"
        if hard in ("imbalance", "high_cardinality"):
            task = "binary_classification"
        row["meta"] = meta

        feature_sql_ms = 0.0
        if structure == "join_aggregates":
            cur.execute("DROP TABLE IF EXISTS rm_obs CASCADE")
            cur.execute("CREATE TABLE rm_obs (id int, grp int, v0 float8, v1 float8, label int)")
            grp = np.arange(n) % max(10, n // 100)
            labels = y if y is not None else np.zeros(n, dtype=int)
            execute_values(
                cur,
                "INSERT INTO rm_obs VALUES %s",
                list(zip(range(n), grp.tolist(), X[:, 0].tolist(), X[:, 1].tolist(), labels.astype(int).tolist())),
                page_size=5000,
            )
            cur.execute("CREATE INDEX ON rm_obs(grp)")
            t0 = time.perf_counter()
            cur.execute("DROP TABLE IF EXISTS rm_feat")
            cur.execute(
                """
                CREATE TABLE rm_feat AS
                SELECT o.id, o.v0, o.v1, o.label,
                       g.cnt::float8 AS grp_cnt, g.avg0, g.avg1
                FROM rm_obs o
                JOIN (
                  SELECT grp, COUNT(*) AS cnt, AVG(v0) AS avg0, AVG(v1) AS avg1
                  FROM rm_obs GROUP BY grp
                ) g USING (grp)
                """
            )
            feature_sql_ms = (time.perf_counter() - t0) * 1000
            cur.execute("SELECT v0, v1, grp_cnt, avg0, avg1, label FROM rm_feat ORDER BY id")
            data = np.asarray(cur.fetchall(), dtype=np.float32)
            X, y = data[:, :5], data[:, 5].astype(np.int64)
            d = X.shape[1]
            row["meta"]["d_after_join"] = int(d)
            cur.execute("DROP TABLE IF EXISTS rm_feat")
        row["pgml_ms"]["feature_sql"] = round(feature_sql_ms, 1)

        t0 = time.perf_counter()
        if task in ("clustering", "pca"):
            store_pack(cur, "rm_pack", X)
        elif task == "regression":
            store_pack(cur, "rm_pack", X, y, y_kind="float")
        else:
            store_pack(cur, "rm_pack", X, y, y_kind="int")
        row["pgml_ms"]["materialize"] = round((time.perf_counter() - t0) * 1000, 1)

        model_blob = None
        pred_fn = None
        sk = None

        if task in ("binary_classification", "multiclass"):
            if task == "multiclass":
                model = "tree"
            train_sql, pred_fn, sk = cls_train_sql(model, n)
            pg_train, model_blob = clock_ms(cur, lambda: cur.execute(train_sql) or cur.fetchone()[0])
            t0 = time.perf_counter()
            sk.fit(X, y)
            sk_fit = (time.perf_counter() - t0) * 1000
            row["pgml_ms"]["train"] = round(pg_train, 1)
            row["sklearn_ms"]["fit"] = round(sk_fit, 1)

        elif task == "regression":
            pred_fn = "pgml_predict_linreg_batch"
            pg_train, model_blob = clock_ms(
                cur, lambda: cur.execute("SELECT pgml_linreg(x, y) FROM rm_pack") or cur.fetchone()[0]
            )
            sk = LinearRegression()
            t0 = time.perf_counter()
            sk.fit(X, y)
            sk_fit = (time.perf_counter() - t0) * 1000
            row["pgml_ms"]["train"] = round(pg_train, 1)
            row["sklearn_ms"]["fit"] = round(sk_fit, 1)

        elif task == "clustering":
            pred_fn = "pgml_kmeans_predict_batch"
            pg_train, model_blob = clock_ms(
                cur, lambda: cur.execute("SELECT pgml_kmeans(x, 8, 20, 0) FROM rm_pack") or cur.fetchone()[0]
            )
            sk = KMeans(n_clusters=8, n_init=1, max_iter=20, random_state=0)
            t0 = time.perf_counter()
            sk.fit(X)
            sk_fit = (time.perf_counter() - t0) * 1000
            row["pgml_ms"]["train"] = round(pg_train, 1)
            row["sklearn_ms"]["fit"] = round(sk_fit, 1)

        elif task == "pca":
            pred_fn = None
            pg_train, model_blob = clock_ms(
                cur, lambda: cur.execute("SELECT pgml_pca_fit(x, 5) FROM rm_pack") or cur.fetchone()[0]
            )
            sk = PCA(n_components=min(5, d, n))
            t0 = time.perf_counter()
            sk.fit(X)
            sk_fit = (time.perf_counter() - t0) * 1000
            row["pgml_ms"]["train"] = round(pg_train, 1)
            row["sklearn_ms"]["fit"] = round(sk_fit, 1)

        # Inference
        b = int(batch or min(n, 10000))
        b = min(b, n)
        Xb = X[:b]
        if mode == "repeated_serving":
            reps = 20
            b = int(batch or 64)
            Xb = X[:b]

            def pg_rep():
                total = 0.0
                for _ in range(reps):
                    ms, _ = clock_ms(
                        cur,
                        lambda: cur.execute(
                            f"SELECT {pred_fn}(%s, %s::float4[][])",
                            (model_blob, Xb.tolist()),
                        ) or cur.fetchone()[0],
                    )
                    total += ms
                return total

            if pred_fn is None:
                raise RuntimeError("repeated_serving requires a predict function")
            pg_rep_ms = pg_rep()
            t0 = time.perf_counter()
            for _ in range(reps):
                sk.predict(Xb)
            sk_rep_ms = (time.perf_counter() - t0) * 1000
            row["pgml_ms"]["repeated_predict"] = round(pg_rep_ms, 1)
            row["sklearn_ms"]["repeated_predict"] = round(sk_rep_ms, 1)
            row["batch"] = b
            row["reps"] = reps
            # Serving map: verdict on repeated predict only.
            pg_total = pg_rep_ms
            sk_total = sk_rep_ms
            row["pgml_ms"]["pipeline_total"] = round(feature_sql_ms + row["pgml_ms"]["train"] + pg_rep_ms, 1)
            row["sklearn_ms"]["pipeline_total"] = round(sk_fit + sk_rep_ms, 1)
        else:
            if pred_fn is not None:
                pg_pred, pred = clock_ms(
                    cur,
                    lambda: cur.execute(
                        f"SELECT {pred_fn}(%s, %s::float4[][])",
                        (model_blob, Xb.tolist()),
                    ) or cur.fetchone()[0],
                )
                t0 = time.perf_counter()
                sk_pred_out = sk.predict(Xb) if hasattr(sk, "predict") else sk.transform(Xb)
                sk_pred = (time.perf_counter() - t0) * 1000
                row["pgml_ms"]["predict"] = round(pg_pred, 1)
                row["sklearn_ms"]["predict"] = round(sk_pred, 1)
                row["batch"] = b
                if task in ("binary_classification", "multiclass"):
                    row["quality"]["pgml_acc_head"] = round(float(accuracy_score(y[:b], np.asarray(pred))), 4)
                    row["quality"]["sklearn_acc_head"] = round(float(accuracy_score(y[:b], sk_pred_out)), 4)
                elif task == "regression":
                    row["quality"]["pgml_r2_head"] = round(float(r2_score(y[:b], np.asarray(pred, dtype=float))), 4)
                    row["quality"]["sklearn_r2_head"] = round(float(r2_score(y[:b], sk_pred_out)), 4)
            else:
                # PCA transform attempt
                try:
                    pg_pred, _ = clock_ms(
                        cur,
                        lambda: cur.execute(
                            "SELECT pgml_pca_transform_batch(%s, %s::float4[][])",
                            (model_blob, Xb.tolist()),
                        ) or cur.fetchone()[0],
                    )
                    row["pgml_ms"]["predict"] = round(pg_pred, 1)
                except Exception as exc:
                    row["pgml_ms"]["predict"] = 0.0
                    row["pgml_ms"]["predict_note"] = f"no transform_batch: {exc}"
                t0 = time.perf_counter()
                sk.transform(Xb)
                row["sklearn_ms"]["predict"] = round((time.perf_counter() - t0) * 1000, 1)
                row["batch"] = b

            # Batch-inference axis: compare predict only (train recorded separately).
            if mode == "batch_inference":
                pg_total = row["pgml_ms"].get("predict", 0)
                sk_total = row["sklearn_ms"].get("predict", 0)
                row["pgml_ms"]["pipeline_total"] = round(
                    feature_sql_ms + row["pgml_ms"].get("train", 0) + row["pgml_ms"].get("predict", 0), 1
                )
                row["sklearn_ms"]["pipeline_total"] = round(
                    row["sklearn_ms"].get("fit", 0) + row["sklearn_ms"].get("predict", 0), 1
                )
            else:
                pg_total = feature_sql_ms + row["pgml_ms"].get("train", 0) + row["pgml_ms"].get("predict", 0)
                sk_total = row["sklearn_ms"].get("fit", 0) + row["sklearn_ms"].get("predict", 0)

        row["pgml_ms"]["total"] = round(pg_total, 1)
        row["sklearn_ms"]["total"] = round(sk_total, 1)
        row["speedup_vs_sklearn"] = round(sk_total / pg_total, 3) if pg_total > 0 else None
        row["verdict"] = verdict(pg_total, sk_total)
        return row
    except MemoryError as exc:
        stage = "store_pack" if "materialize" not in row.get("pgml_ms", {}) else "unknown"
        # materialize key is set only after store_pack returns
        row["verdict"] = "materialization_failure"
        row["failure_stage"] = stage
        row["failure_before_train"] = True
        row["error"] = str(exc)
        return row
    except Exception as exc:
        msg = str(exc).lower()
        tb = traceback.format_exc(limit=6)
        in_store = "store_pack" in tb
        materialized = "materialize" in row.get("pgml_ms", {})
        if (
            "memory" in msg
            or "out of memory" in msg
            or "server closed the connection" in msg
            or "terminat" in msg
        ):
            if in_store or not materialized:
                row["verdict"] = "materialization_failure"
                row["failure_stage"] = "input_materialization"
                row["failure_before_train"] = True
                row["oom_confirmed"] = None  # set only after cgroup/log evidence
                row["failure_cause"] = "unknown_sigkill_pending_evidence"
                row["oom_note"] = (
                    "Backend abort during input pack. Signal 9 alone does not prove OOM; "
                    "set oom_confirmed from cgroup memory.events / OOM killer evidence. "
                    "Not a demonstrated memory limit of the training algorithm."
                )
            else:
                row["verdict"] = "oom"
                row["failure_stage"] = "train_or_predict"
                row["failure_before_train"] = False
        else:
            row["verdict"] = "error"
        row["error"] = str(exc)
        row["traceback"] = tb
        return row


def build_grid():
    points = []
    for n in (1000, 10000, 50000, 100000):
        points.append({"task": "binary_classification", "n": n, "d": 32, "model": "tree", "mode": "train", "axis": "scale_task"})
        points.append({"task": "regression", "n": n, "d": 32, "model": "linreg", "mode": "train", "axis": "scale_task"})
        if n <= 50000:
            points.append({"task": "clustering", "n": n, "d": 32, "model": "kmeans", "mode": "train", "axis": "scale_task"})
            points.append({"task": "pca", "n": n, "d": 32, "model": "pca", "mode": "train", "axis": "scale_task"})
            points.append({"task": "multiclass", "n": n, "d": 32, "model": "tree", "mode": "train", "axis": "scale_task"})
    for d in (8, 32, 128):
        points.append({"task": "binary_classification", "n": 10000, "d": d, "model": "tree", "mode": "train", "axis": "dimensionality"})
        points.append({"task": "regression", "n": 10000, "d": d, "model": "linreg", "mode": "train", "axis": "dimensionality"})
        points.append({"task": "pca", "n": 10000, "d": d, "model": "pca", "mode": "train", "axis": "dimensionality"})
    for model in ("tree", "forest", "gbm", "svm"):
        points.append({"task": "binary_classification", "n": 10000, "d": 32, "model": model, "mode": "train", "axis": "model_complexity"})
    for batch in (1, 10, 100, 1000, 10000, 50000):
        points.append({
            "task": "binary_classification", "n": 50000, "d": 32, "model": "tree",
            "mode": "batch_inference", "batch": batch, "axis": "batch_inference",
        })
    points.append({"task": "binary_classification", "n": 10000, "d": 32, "model": "tree", "kind": "numeric", "mode": "train", "axis": "feature_types"})
    points.append({"task": "binary_classification", "n": 10000, "d": 32, "model": "tree", "kind": "mixed_categorical", "mode": "train", "axis": "feature_types"})
    points.append({"task": "binary_classification", "n": 10000, "d": 8, "model": "tree", "structure": "single_table", "mode": "train", "axis": "structure"})
    points.append({"task": "binary_classification", "n": 10000, "d": 8, "model": "tree", "structure": "join_aggregates", "mode": "train", "axis": "structure"})
    points.append({"task": "binary_classification", "n": 10000, "d": 32, "model": "tree", "mode": "repeated_serving", "batch": 64, "axis": "mode"})
    points.append({"task": "binary_classification", "n": 20000, "d": 32, "model": "tree", "hard": "imbalance", "mode": "train", "axis": "hard_cases"})
    points.append({"task": "binary_classification", "n": 20000, "d": 16, "model": "tree", "hard": "high_cardinality", "mode": "train", "axis": "hard_cases"})
    points.append({"task": "regression", "n": 20000, "d": 16, "model": "linreg", "hard": "collinearity", "mode": "train", "axis": "hard_cases"})
    points.append({"task": "multiclass", "n": 20000, "d": 16, "model": "tree", "hard": "uneven_density", "mode": "train", "axis": "hard_cases"})
    points.append({"task": "binary_classification", "n": 200000, "d": 128, "model": "tree", "mode": "train", "axis": "memory_probe"})
    return points


def summarize(points):
    by_verdict, by_axis = {}, {}
    map_rows = []
    for p in points:
        v = p.get("verdict", "error")
        by_verdict[v] = by_verdict.get(v, 0) + 1
        axis = p.get("config", {}).get("axis", "?")
        by_axis.setdefault(axis, {})
        by_axis[axis][v] = by_axis[axis].get(v, 0) + 1
        cfg = p["config"]
        map_rows.append({
            "axis": cfg.get("axis"),
            "task": cfg.get("task"),
            "n": cfg.get("n"),
            "d": cfg.get("d"),
            "model": cfg.get("model"),
            "mode": cfg.get("mode"),
            "batch": p.get("batch", cfg.get("batch")),
            "kind": cfg.get("kind", "numeric"),
            "structure": cfg.get("structure", "single_table"),
            "hard": cfg.get("hard"),
            "pgml_total_ms": p.get("pgml_ms", {}).get("total"),
            "sklearn_total_ms": p.get("sklearn_ms", {}).get("total"),
            "speedup_vs_sklearn": p.get("speedup_vs_sklearn"),
            "verdict": v,
        })
    return {"by_verdict": by_verdict, "by_axis": by_axis, "map_rows": map_rows}


def main():
    conn, cur = connect()
    grid = build_grid()
    results = {
        "coverage_matrix": COVERAGE,
        "comparison": "pgml SQL-side train/predict vs scikit-learn in-process on the same arrays",
        "verdict_rule": "faster if pgml_total < sklearn/1.25; slower if >1.25x; else comparable",
        "points": [],
        "skipped_cells": [
            "No full Cartesian product across all axes",
            "No MADlib/PostgresML in this map (separate head-to-heads)",
            "repeated_serving is sequential batches in one backend, not multi-connection",
            "Large multi-table RDBs covered in RelBench/4DBInfer sections",
            "Memory probe is a single n=200k d=128 point",
        ],
    }
    for i, cfg in enumerate(grid):
        print(f"[{i+1}/{len(grid)}] {cfg}", flush=True)
        try:
            cur.execute("SET statement_timeout = '180s'")
        except Exception:
            conn, cur = connect()
        row = run_point(cur, f"p{i:03d}", cfg)
        results["points"].append(row)
        print(
            f"  -> {row.get('verdict')} pgml={row.get('pgml_ms', {}).get('total')} "
            f"sk={row.get('sklearn_ms', {}).get('total')} {row.get('error', '')}",
            flush=True,
        )
    results["summary"] = summarize(results["points"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(f"wrote {OUT}", flush=True)
    print("verdicts:", results["summary"]["by_verdict"], flush=True)
    conn.close()


if __name__ == "__main__":
    main()
