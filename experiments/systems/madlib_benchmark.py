#!/usr/bin/env python3
"""Head-to-head pgml vs MADlib 2.1.0.

MADlib 2.1.0 supports PostgreSQL 15, so this runs against the pgml-madlib
container (localhost:5434), where both extensions are installed. The three
shared algorithms are k-means, linear regression, and a classification tree.

Both sides train on data that is already stored in the database. pgml sees one
float4 matrix; MADlib sees one row per observation. The clock is
clock_timestamp() inside the backend, so client transfer is not included.
"""
import json
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
import psycopg2
from psycopg2.extras import execute_values
from sklearn.metrics import r2_score

OUT = str(_REPO.joinpath('results', 'madlib_comparison.json'))
N, D = 10000, 20
K, ITERS = 8, 20
TREE_DEPTH, TREE_MIN = 8, 2


def connect():
    conn = psycopg2.connect(
        host="localhost", port=5434, dbname="ml_test",
        user="postgres", password="postgres",
    )
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SET statement_timeout = 0")
    cur.execute("SELECT extname FROM pg_extension WHERE extname = 'pgml'")
    if cur.fetchone() is None:
        raise SystemExit("pgml is not installed on port 5434")
    cur.execute("SELECT nspname FROM pg_namespace WHERE nspname = 'madlib'")
    if cur.fetchone() is None:
        raise SystemExit("MADlib schema is not installed on port 5434")
    return conn, cur


def time_query(cur, select_sql):
    cur.execute(
        "SELECT extract(epoch FROM (clock_timestamp() - t0)) * 1000 "
        "FROM (SELECT clock_timestamp() AS t0) s, LATERAL (" + select_sql + ") q"
    )
    return float(cur.fetchone()[0])


def load(cur):
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (N, D))
    y = (2 * X[:, 0] + 3 * X[:, 1] + 1 + rng.normal(0, 0.5, N))
    labels = (X[:, 0] + X[:, 1] > 0).astype(int)
    feat_cols = ", ".join(f"x{i + 1} float8" for i in range(D))
    cur.execute("DROP TABLE IF EXISTS obs CASCADE")
    cur.execute(f"CREATE TABLE obs (id int, x float8[], y float8, label int, {feat_cols})")
    rows = []
    for i in range(N):
        row = [i, X[i].tolist(), float(y[i]), int(labels[i])]
        row.extend(float(v) for v in X[i])
        rows.append(tuple(row))
    execute_values(cur, "INSERT INTO obs VALUES %s", rows, page_size=2000)
    cur.execute("ANALYZE obs")
    cur.execute("DROP TABLE IF EXISTS pgml_pack")
    cur.execute("CREATE TABLE pgml_pack (x float4[][], y float4[], labels int[])")
    cur.execute(
        "INSERT INTO pgml_pack VALUES (%s::float4[][], %s::float4[], %s::int[])",
        (X.astype(np.float32).tolist(), y.astype(np.float32).tolist(), labels.tolist()),
    )
    return X, y, labels


def main():
    conn, cur = connect()
    print(f"loading n={N} d={D}", flush=True)
    X, y, labels = load(cur)
    feats = ", ".join(f"x{i}" for i in range(1, D + 1))
    results = {
        "n": N, "d": D, "k": K, "iters": ITERS,
        "tree_depth": TREE_DEPTH, "tree_min_samples": TREE_MIN,
        "madlib": "2.1.0", "postgres": 15,
        "note": "SQL-side clock after the tables are loaded. MADlib bins continuous tree features into 20 bins. Both k-means runs are asked for 20 iterations.",
    }

    print("k-means", flush=True)
    pg_km = time_query(cur, "SELECT octet_length(pgml_kmeans(x, 8, 20, 0)) FROM pgml_pack")
    cur.execute("DROP TABLE IF EXISTS mad_km")
    cur.execute(
        "CREATE TABLE mad_km AS "
        "SELECT q.*, extract(epoch FROM (clock_timestamp() - t0)) * 1000 AS bench_ms "
        "FROM (SELECT clock_timestamp() AS t0) s, LATERAL ("
        "SELECT * FROM madlib.kmeanspp('obs', 'x', 8, "
        "'madlib.squared_dist_norm2', 'madlib.avg', 20, 0.001)"
        ") q"
    )
    cur.execute("SELECT bench_ms, num_iterations, objective_fn FROM mad_km")
    mad_ms, mad_iters, mad_obj = cur.fetchone()
    results["kmeans_ms"] = {"pgml": round(pg_km, 1), "madlib": round(float(mad_ms), 1)}
    results["kmeans_madlib_iterations"] = int(mad_iters)
    results["kmeans_madlib_objective"] = float(mad_obj)
    print(f"  pgml {pg_km:.0f} ms | madlib {float(mad_ms):.0f} ms ({int(mad_iters)} iters)", flush=True)

    print("linear regression", flush=True)
    pg_lr = time_query(cur, "SELECT octet_length(pgml_linreg(x, y)) FROM pgml_pack")
    cur.execute("DROP TABLE IF EXISTS mad_lin, mad_lin_summary CASCADE")
    mad_lr = time_query(
        cur,
        "SELECT madlib.linregr_train('obs', 'mad_lin', 'y', 'ARRAY[1.0]::float8[] || x')",
    )
    cur.execute("SELECT coef FROM mad_lin")
    coef = np.asarray(cur.fetchone()[0], dtype=float)
    design = np.column_stack([np.ones(N), X])
    r2_mad = float(r2_score(y, design @ coef))
    cur.execute("SELECT pgml_predict_linreg_batch(pgml_linreg(x, y), x) FROM pgml_pack")
    pred = np.asarray(cur.fetchone()[0], dtype=float)
    r2_pg = float(r2_score(y, pred))
    results["linreg_ms"] = {"pgml": round(pg_lr, 1), "madlib": round(mad_lr, 1)}
    results["linreg_r2"] = {"pgml": round(r2_pg, 4), "madlib": round(r2_mad, 4)}
    print(f"  pgml {pg_lr:.0f} ms R2 {r2_pg:.4f} | madlib {mad_lr:.0f} ms R2 {r2_mad:.4f}", flush=True)

    print("decision tree", flush=True)
    pg_tree = time_query(cur, "SELECT octet_length(pgml_train_tree(x, labels, 8, 2)) FROM pgml_pack")
    cur.execute("SELECT pgml_predict_tree_batch(pgml_train_tree(x, labels, 8, 2), x) FROM pgml_pack")
    pg_pred = np.asarray(cur.fetchone()[0])
    pg_acc = float(np.mean(pg_pred == labels))
    cur.execute("DROP TABLE IF EXISTS mad_tree, mad_tree_summary CASCADE")
    mad_tree = time_query(
        cur,
        "SELECT madlib.tree_train('obs', 'mad_tree', 'id', 'label', "
        f"'{feats}', NULL, 'gini', NULL, NULL, 8, 2, 1, 20)",
    )
    cur.execute("DROP TABLE IF EXISTS mad_tree_pred")
    cur.execute("SELECT madlib.tree_predict('mad_tree', 'obs', 'mad_tree_pred', 'response')")
    cur.execute(
        "SELECT avg((p.estimated_label::int = o.label)::int) "
        "FROM mad_tree_pred p JOIN obs o USING (id)"
    )
    mad_acc = float(cur.fetchone()[0])
    results["tree_ms"] = {"pgml": round(pg_tree, 1), "madlib": round(mad_tree, 1)}
    results["tree_train_accuracy"] = {"pgml": round(pg_acc, 4), "madlib": round(mad_acc, 4)}
    print(f"  pgml {pg_tree:.0f} ms acc {pg_acc:.3f} | madlib {mad_tree:.0f} ms acc {mad_acc:.3f}", flush=True)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {OUT}", flush=True)
    conn.close()


if __name__ == "__main__":
    main()
