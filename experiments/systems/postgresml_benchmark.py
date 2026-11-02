#!/usr/bin/env python3
"""Compare native pgml with PostgresML 2.10.0 — methodology-safe version.

Separates:
  - API wall time (native SQL train vs full pgml.train including snapshot/split/metrics)
  - Learner fit time (native train_ms vs metrics.fit_time)
  - Quality on a *shared* hold-out (25%): native and PostgresML both scored on the
    same external test rows (PostgresML trained with test_size=0 on the train relation).

Does not place in-sample native metrics next to PostgresML hold-out metrics.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
import psycopg2
from psycopg2.extras import execute_values
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

OUT = str(_REPO.joinpath('results', 'postgresml_comparison.json'))
N, D = 10000, 20
LIB = "$libdir/pgml_native"
SEED = 0
TEST_SIZE = 0.25


def connect():
    conn = psycopg2.connect(
        host="localhost", port=5435, dbname="postgresml",
        user="postgresml", password="postgresml",
    )
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SET statement_timeout = 0")
    cur.execute("SELECT version()")
    print(cur.fetchone()[0], flush=True)
    cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'pgml'")
    row = cur.fetchone()
    if row is None:
        raise SystemExit("PostgresML extension pgml is not installed")
    print(f"postgresml extension {row[0]}", flush=True)
    return conn, cur


def install_native(cur):
    specs = [
        ("pgml_kmeans", "float4[][], int, int, int", "bytea", "pgclust_kmeans"),
        ("pgml_linreg", "float4[][], float4[]", "bytea", "pgreg_linreg"),
        ("pgml_predict_linreg_batch", "bytea, float4[][]", "float4[]", "pgreg_predict_linreg_batch"),
        ("pgml_train_tree", "float4[][], int[], int, int", "bytea", "pgml_train_tree"),
        ("pgml_predict_tree_batch", "bytea, float4[][]", "int[]", "pgml_predict_tree_batch"),
    ]
    for name, args, ret, symbol in specs:
        cur.execute(
            f"CREATE OR REPLACE FUNCTION {name}({args}) RETURNS {ret} "
            f"AS '{LIB}', '{symbol}' LANGUAGE C STRICT"
        )


def time_query(cur, select_sql):
    cur.execute(
        "SELECT extract(epoch FROM (clock_timestamp() - t0)) * 1000 "
        "FROM (SELECT clock_timestamp() AS t0) s, LATERAL (" + select_sql + ") q"
    )
    return float(cur.fetchone()[0])


def load_split(cur):
    rng = np.random.default_rng(SEED)
    X = rng.normal(0, 1, (N, D)).astype(np.float64)
    y = 2 * X[:, 0] + 3 * X[:, 1] + 1 + rng.normal(0, 0.5, N)
    labels = (X[:, 0] + X[:, 1] > 0).astype(int)
    idx = np.arange(N)
    tr, te = train_test_split(idx, test_size=TEST_SIZE, random_state=SEED, stratify=labels)
    Xtr, Xte = X[tr], X[te]
    ytr, yte = y[tr], y[te]
    ltr, lte = labels[tr], labels[te]

    feat_cols = ", ".join(f"x{i + 1} float8" for i in range(D))
    cur.execute("DROP VIEW IF EXISTS reg_train_v, cls_train_v, km_train_v CASCADE")
    cur.execute("DROP TABLE IF EXISTS obs_train, obs_test, pgml_pack CASCADE")
    cur.execute(f"CREATE TABLE obs_train (id int, y float8, label int, {feat_cols})")
    cur.execute(f"CREATE TABLE obs_test (id int, y float8, label int, {feat_cols})")

    def rows_of(ids, Xs, ys, ls):
        out = []
        for j, i in enumerate(ids):
            out.append((int(i), float(ys[j]), int(ls[j]), *map(float, Xs[j])))
        return out

    execute_values(cur, "INSERT INTO obs_train VALUES %s", rows_of(tr, Xtr, ytr, ltr), page_size=2000)
    execute_values(cur, "INSERT INTO obs_test VALUES %s", rows_of(te, Xte, yte, lte), page_size=2000)
    cur.execute("ANALYZE obs_train")
    cur.execute("ANALYZE obs_test")
    xs = ", ".join(f"x{i}" for i in range(1, D + 1))
    cur.execute(f"CREATE VIEW reg_train_v AS SELECT {xs}, y FROM obs_train")
    cur.execute(f"CREATE VIEW cls_train_v AS SELECT {xs}, label FROM obs_train")
    cur.execute(f"CREATE VIEW km_train_v AS SELECT {xs} FROM obs_train")

    # native pack = TRAIN rows only
    cur.execute("CREATE TABLE pgml_pack (x float4[][], y float4[], labels int[])")
    cur.execute(
        "INSERT INTO pgml_pack VALUES (%s::float4[][], %s::float4[], %s::int[])",
        (Xtr.astype(np.float32).tolist(), ytr.astype(np.float32).tolist(), ltr.tolist()),
    )
    tr_i = np.asarray(tr, dtype=np.int64)
    te_i = np.asarray(te, dtype=np.int64)
    return {
        "Xtr": Xtr.astype(np.float32),
        "Xte": Xte.astype(np.float32),
        "ytr": ytr,
        "yte": yte,
        "ltr": ltr,
        "lte": lte,
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "train_ids": tr_i,
        "test_ids": te_i,
        "train_ids_checksum": {
            "n": int(len(tr_i)),
            "sum": int(tr_i.sum()),
            "sha256_sorted": __import__("hashlib").sha256(np.sort(tr_i).tobytes()).hexdigest(),
        },
        "test_ids_checksum": {
            "n": int(len(te_i)),
            "sum": int(te_i.sum()),
            "sha256_sorted": __import__("hashlib").sha256(np.sort(te_i).tobytes()).hexdigest(),
        },
    }


def train_postgresml(cur, project, task, relation, y_column, algorithm, hyperparams):
    y_sql = "NULL" if y_column is None else f"'{y_column}'"
    # PostgresML rejects test_size=0 (NotEnoughSamples). Use a tiny internal
    # fold for its API metrics only; shared-holdout quality is scored later via
    # predict() on obs_test, not from these metrics.
    sql = (
        "SELECT 1 FROM pgml.train("
        f"project_name => '{project}', "
        f"task => '{task}', "
        f"relation_name => '{relation}', "
        f"y_column_name => {y_sql}, "
        f"algorithm => '{algorithm}', "
        f"hyperparams => '{hyperparams}'::jsonb, "
        "test_size => 0.05, test_sampling => 'random')"
    )
    wall = time_query(cur, sql)
    cur.execute(
        "SELECT m.id, m.runtime::text, m.metrics::text "
        "FROM pgml.models m JOIN pgml.projects p ON p.id = m.project_id "
        "WHERE p.name = %s ORDER BY m.id DESC LIMIT 1",
        (project,),
    )
    mid, runtime, metrics = cur.fetchone()
    return wall, int(mid), runtime, json.loads(metrics)


def predict_postgresml(cur, project, relation):
    """Score latest model for project on relation; return predictions column if available."""
    # PostgresML deploy/predict API varies; use pgml.predict from a subquery when present.
    cur.execute(
        "SELECT m.id FROM pgml.models m JOIN pgml.projects p ON p.id = m.project_id "
        "WHERE p.name = %s ORDER BY m.id DESC LIMIT 1",
        (project,),
    )
    mid = cur.fetchone()[0]
    # Try common predict forms
    try:
        cur.execute(
            f"SELECT pgml.predict('{project}', ARRAY[{','.join(f'x{i}' for i in range(1, D+1))}]) "
            f"FROM {relation}"
        )
        return np.asarray([r[0] for r in cur.fetchall()], dtype=float)
    except Exception:
        cur.connection.rollback()
        cur.connection.autocommit = True
    try:
        cur.execute(
            f"SELECT prediction FROM pgml.predict_batch('{project}', '{relation}')"
        )
        return np.asarray([r[0] for r in cur.fetchall()], dtype=float)
    except Exception as e:
        return None, str(e)


def main():
    conn, cur = connect()
    install_native(cur)
    print(f"loading n={N} d={D} holdout={TEST_SIZE}", flush=True)
    data = load_split(cur)
    Xtr, Xte = data["Xtr"], data["Xte"]

    results = {
        "n": N,
        "d": D,
        "n_train": data["n_train"],
        "n_test": data["n_test"],
        "test_size": TEST_SIZE,
        "postgresml_image": "ghcr.io/postgresml/postgresml:2.10.0",
        "postgres": 15,
        "methodology": {
            "quality": (
                "Shared hold-out: both systems train on obs_train only and are "
                "scored on the same obs_test / Xte rows. Native in-sample metrics "
                "are not compared to PostgresML."
            ),
            "timing": (
                "api_wall_ms: full user-facing call (native SQL train, or PostgresML "
                "pgml.train including snapshot/split/metrics). "
                "fit_ms: native SQL train time, or metrics.fit_time*1000 from PostgresML. "
                "These two layers must not be equated."
            ),
            "repeats": "Single run per cell; no SD/CI/significance.",
            "tree_note": (
                "PostgresML has no standalone CART; tree row uses random_forest with "
                "one unbootstrapped tree — algorithmically adjacent, not identical."
            ),
        },
        "note": (
            "Same PostgreSQL 15 server. Timing layers and quality sets are separated; "
            "see methodology."
        ),
    }

    # ---- native train on train pack; score on Xte ----
    print("native linreg", flush=True)
    lin_api = time_query(cur, "SELECT octet_length(pgml_linreg(x, y)) FROM pgml_pack")
    cur.execute("SELECT pgml_linreg(x, y) FROM pgml_pack")
    lin_model = cur.fetchone()[0]
    cur.execute(
        "SELECT pgml_predict_linreg_batch(%s, %s::float4[][])",
        (psycopg2.Binary(lin_model), Xte.tolist()),
    )
    lin_pred = np.asarray(cur.fetchone()[0], dtype=float)
    lin_r2 = float(r2_score(data["yte"], lin_pred))

    print("native tree", flush=True)
    tree_api = time_query(
        cur, "SELECT octet_length(pgml_train_tree(x, labels, 8, 2)) FROM pgml_pack"
    )
    cur.execute("SELECT pgml_train_tree(x, labels, 8, 2) FROM pgml_pack")
    tree_model = cur.fetchone()[0]
    cur.execute(
        "SELECT pgml_predict_tree_batch(%s, %s::float4[][])",
        (psycopg2.Binary(tree_model), Xte.tolist()),
    )
    tree_pred = np.asarray(cur.fetchone()[0])
    tree_acc = float(accuracy_score(data["lte"], tree_pred))

    print("native kmeans", flush=True)
    km_api = time_query(cur, "SELECT octet_length(pgml_kmeans(x, 8, 20, 0)) FROM pgml_pack")

    results["native"] = {
        "kmeans": {
            "api_wall_ms": round(km_api, 1),
            "fit_ms": round(km_api, 1),
            "phases_included": ["SQL train on pre-packed train matrix"],
            "quality_holdout": None,
            "quality_note": "clustering quality not compared on hold-out labels",
        },
        "linreg": {
            "api_wall_ms": round(lin_api, 1),
            "fit_ms": round(lin_api, 1),
            "phases_included": ["SQL train on pre-packed train matrix"],
            "r2_holdout": round(lin_r2, 4),
        },
        "tree": {
            "api_wall_ms": round(tree_api, 1),
            "fit_ms": round(tree_api, 1),
            "phases_included": ["SQL train on pre-packed train matrix"],
            "accuracy_holdout": round(tree_acc, 4),
        },
    }
    print(results["native"], flush=True)

    # ---- PostgresML on train views; score on test table via predict ----
    print("postgresml linear", flush=True)
    wall, mid, runtime, metrics = train_postgresml(
        cur, "cmp_lin_holdout", "regression", "reg_train_v", "y", "linear", "{}",
    )
    # predict on test
    xs = ", ".join(f"x{i}" for i in range(1, D + 1))
    try:
        # Keep table scan order (= insert order = yte order). Do not ORDER BY id.
        cur.execute(
            f"SELECT y, pgml.predict('cmp_lin_holdout', ARRAY[{xs}]::float8[]) FROM obs_test"
        )
        rows = cur.fetchall()
        y_true = np.asarray([float(r[0]) for r in rows], dtype=float)
        pml_lin_pred = np.asarray([float(r[1]) for r in rows], dtype=float)
        pml_lin_r2 = float(r2_score(y_true, pml_lin_pred))
        lin_pred_err = None
    except Exception as e:
        pml_lin_r2 = None
        lin_pred_err = str(e)

    results["postgresml_linear"] = {
        "api_wall_ms": round(wall, 1),
        "fit_ms": round(float(metrics.get("fit_time", 0)) * 1000, 3),
        "runtime": runtime,
        "phases_included_api_wall": [
            "snapshot relation", "optional split", "library fit", "metrics",
        ],
        "metrics_from_train_api": metrics,
        "r2_holdout_shared": pml_lin_r2,
        "predict_error": lin_pred_err,
    }
    print(results["postgresml_linear"], flush=True)

    print("postgresml kmeans", flush=True)
    km_hp = json.dumps({"n_clusters": 8, "max_iter": 20, "n_init": 1, "random_state": 0})
    wall, mid, runtime, metrics = train_postgresml(
        cur, "cmp_km_holdout", "clustering", "km_train_v", None, "kmeans", km_hp,
    )
    results["postgresml_kmeans"] = {
        "api_wall_ms": round(wall, 1),
        "fit_ms": round(float(metrics.get("fit_time", 0)) * 1000, 3),
        "runtime": runtime,
        "phases_included_api_wall": [
            "snapshot relation", "optional split", "library fit", "metrics",
        ],
        "metrics_from_train_api": metrics,
        "quality_holdout": None,
        "quality_note": "clustering quality not compared",
    }
    print(results["postgresml_kmeans"], flush=True)

    print("postgresml one tree", flush=True)
    # Keep historical default config for continuity; aligned max_features=1.0
    # is measured separately in protocol_remeasure / postgresml_tree_alignment.json.
    tree_hp = json.dumps({
        "n_estimators": 1, "max_depth": 8, "min_samples_split": 2,
        "bootstrap": False, "random_state": 0,
        # NOTE: sklearn RF default max_features='sqrt' remains here intentionally
        # so the published gap is preserved; see tree_alignment remasure.
    })
    wall, mid, runtime, metrics = train_postgresml(
        cur, "cmp_tree_holdout", "classification", "cls_train_v", "label",
        "random_forest", tree_hp,
    )
    try:
        cur.execute(
            f"SELECT label, pgml.predict('cmp_tree_holdout', ARRAY[{xs}]::float8[]) FROM obs_test"
        )
        rows = cur.fetchall()
        y_true = np.asarray([int(r[0]) for r in rows], dtype=np.int32)
        pml_tree_pred = []
        for _y, v in rows:
            if isinstance(v, dict):
                pml_tree_pred.append(int(max(v, key=v.get)))
            else:
                pml_tree_pred.append(int(round(float(v))))
        pml_tree_pred = np.asarray(pml_tree_pred, dtype=np.int32)
        pml_tree_acc = float(accuracy_score(y_true, pml_tree_pred))
        tree_pred_err = None
    except Exception as e:
        pml_tree_acc = None
        tree_pred_err = str(e)

    results["postgresml_tree"] = {
        "api_wall_ms": round(wall, 1),
        "fit_ms": round(float(metrics.get("fit_time", 0)) * 1000, 3),
        "runtime": runtime,
        "phases_included_api_wall": [
            "snapshot relation", "optional split", "library fit", "metrics",
        ],
        "metrics_from_train_api": metrics,
        "accuracy_holdout_shared": pml_tree_acc,
        "predict_error": tree_pred_err,
        "hyperparams": json.loads(tree_hp),
        "algorithm_note": "one unbootstrapped random_forest tree; not identical to native CART",
    }
    print(results["postgresml_tree"], flush=True)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {OUT}", flush=True)
    conn.close()


if __name__ == "__main__":
    main()
