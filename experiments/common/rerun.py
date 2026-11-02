#!/usr/bin/env python3
"""Repeat the pgml headline experiments against the running database.

Quality checks follow the published figures: hold-out accuracy on WDBC
(and Ionosphere/Sonar when the UCI files are reachable), XOR, Iris
multiclass, diabetes R^2, Iris/blobs clustering ARI, WDBC PCA, and the
categorical-split check. Timing checks are clocked inside SQL after the
array is already stored, matching the original SQL-side protocol.
"""
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
import time
import urllib.request

import numpy as np
import psycopg2
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, make_blobs
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import accuracy_score, adjusted_rand_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

OUT = str(_REPO.joinpath('results', 'rerun.json'))
EXT = str(_REPO.joinpath('results', 'extended_features.json'))


def connect():
    conn = psycopg2.connect(
        host="localhost", port=5433, dbname="ml_test",
        user="postgres", password="postgres",
    )
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SET statement_timeout = 0")
    return conn, cur


def acc(y, pred):
    return float(accuracy_score(np.asarray(y), np.asarray(pred)))


def train_predict(cur, train_sql, pred_sql, Xtr, ytr, Xte, y_cast="int[]"):
    cur.execute(train_sql, (Xtr.astype(np.float32).tolist(), np.asarray(ytr).tolist()))
    model = cur.fetchone()[0]
    cur.execute(pred_sql, (model, Xte.astype(np.float32).tolist()))
    return cur.fetchone()[0]


def cls_suite(cur, name, X, y):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    scaler = StandardScaler().fit(Xtr)
    Xtr, Xte = scaler.transform(Xtr).astype(np.float32), scaler.transform(Xte).astype(np.float32)
    specs = {
        "tree": (
            "SELECT pgml_train_tree(%s::float4[][], %s::int[], 8, 2)",
            "SELECT pgml_predict_tree_batch(%s, %s::float4[][])",
            DecisionTreeClassifier(max_depth=8, min_samples_split=2, random_state=0),
        ),
        "forest": (
            "SELECT pgml_train_forest(%s::float4[][], %s::int[], 15, 8, 2)",
            "SELECT pgml_predict_forest_batch(%s, %s::float4[][])",
            RandomForestClassifier(n_estimators=15, max_depth=8, min_samples_split=2, random_state=0),
        ),
        "gbm": (
            "SELECT pgml_train_gbm(%s::float4[][], %s::int[], 30, 3, 0.1)",
            "SELECT pgml_predict_gbm_batch(%s, %s::float4[][])",
            GradientBoostingClassifier(n_estimators=30, max_depth=3, learning_rate=0.1, random_state=0),
        ),
        "adaboost": (
            "SELECT pgml_train_adaboost(%s::float4[][], %s::int[], 30)",
            "SELECT pgml_predict_adaboost_batch(%s, %s::float4[][])",
            AdaBoostClassifier(n_estimators=30, random_state=0),
        ),
        "svm": (
            "SELECT pgml_train_svm(%s::float4[][], %s::int[], 0.001, 20)",
            "SELECT pgml_predict_svm_batch(%s, %s::float4[][])",
            LinearSVC(C=1.0, dual=False, random_state=0, max_iter=5000),
        ),
    }
    out = {}
    for model, (tr, pr, sk) in specs.items():
        if model in ("adaboost", "svm") and len(np.unique(y)) > 2:
            continue
        pred = train_predict(cur, tr, pr, Xtr, ytr, Xte)
        sk.fit(Xtr, ytr)
        out[model] = {"pgml": round(acc(yte, pred), 4), "sklearn": round(float(sk.score(Xte, yte)), 4)}
        print(f"  {name:12} {model:9} pgml {out[model]['pgml']:.3f}  sklearn {out[model]['sklearn']:.3f}", flush=True)
    return out


def load_uci_csv(url, label_last=True):
    with urllib.request.urlopen(url, timeout=20) as resp:
        raw = resp.read().decode("utf-8", errors="replace").strip().splitlines()
    rows = [line.split(",") for line in raw if line and not line.startswith("@")]
    X, y = [], []
    labels = {}
    for row in rows:
        if label_last:
            feats, lab = row[:-1], row[-1]
        else:
            feats, lab = row[1:], row[0]
        if any(v in ("?", "") for v in feats):
            continue
        X.append([float(v) for v in feats])
        if lab not in labels:
            labels[lab] = len(labels)
        y.append(labels[lab])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def xor_suite(cur):
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (800, 2)).astype(np.float32)
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(np.int64)
    y[rng.random(len(y)) < 0.08] ^= 1
    return cls_suite(cur, "xor", X, y)


def iris_multiclass(cur):
    iris = load_iris()
    Xtr, Xte, ytr, yte = train_test_split(
        iris.data.astype(np.float32), iris.target.astype(np.int64),
        test_size=0.2, random_state=0, stratify=iris.target,
    )
    out = {}
    specs = {
        "tree": ("SELECT pgml_train_tree(%s::float4[][], %s::int[], 6, 2)",
                 "SELECT pgml_predict_tree_batch(%s, %s::float4[][])"),
        "forest": ("SELECT pgml_train_forest(%s::float4[][], %s::int[], 15, 6, 2)",
                   "SELECT pgml_predict_forest_batch(%s, %s::float4[][])"),
    }
    for name, (tr, pr) in specs.items():
        pred = train_predict(cur, tr, pr, Xtr, ytr, Xte)
        out[name] = round(acc(yte, pred), 4)
        print(f"  iris         {name:9} {out[name]:.3f}", flush=True)
    cur.execute(
        "SELECT pgml_predict_svm_ovr_batch(pgml_train_svm_ovr(%s::float4[][], %s::int[], 3, 0.01, 30), %s::float4[][])",
        (Xtr.tolist(), ytr.tolist(), Xte.tolist()),
    )
    out["svm_ovr"] = round(acc(yte, cur.fetchone()[0]), 4)
    print(f"  iris         svm_ovr   {out['svm_ovr']:.3f}", flush=True)
    return out


def regression_suite(cur):
    X, y = load_diabetes(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    Xtr, Xte = Xtr.astype(np.float32), Xte.astype(np.float32)
    ytr, yte = ytr.astype(np.float32), yte.astype(np.float32)
    specs = {
        "linear": (
            "SELECT pgml_linreg(%s::float4[][], %s::float4[])",
            "SELECT pgml_predict_linreg_batch(%s, %s::float4[][])",
            LinearRegression(),
        ),
        "ridge": (
            "SELECT pgml_ridge(%s::float4[][], %s::float4[], 1.0)",
            "SELECT pgml_predict_linreg_batch(%s, %s::float4[][])",
            Ridge(alpha=1.0),
        ),
        "tree": (
            "SELECT pgml_regression_tree(%s::float4[][], %s::float4[], 4, 5)",
            "SELECT pgml_predict_regression_tree_batch(%s, %s::float4[][])",
            DecisionTreeRegressor(max_depth=4, min_samples_split=5, random_state=0),
        ),
        "gbm": (
            "SELECT pgml_gbm_regression(%s::float4[][], %s::float4[], 40, 2, 0.1)",
            "SELECT pgml_predict_gbm_regression_batch(%s, %s::float4[][])",
            GradientBoostingRegressor(n_estimators=40, max_depth=2, learning_rate=0.1, random_state=0),
        ),
    }
    out = {}
    for name, (tr, pr, sk) in specs.items():
        pred = np.asarray(train_predict(cur, tr, pr, Xtr, ytr, Xte), dtype=float)
        sk.fit(Xtr, ytr)
        out[name] = {
            "pgml": round(float(r2_score(yte, pred)), 4),
            "sklearn": round(float(sk.score(Xte, yte)), 4),
        }
        print(f"  diabetes     {name:9} pgml {out[name]['pgml']:.4f}  sklearn {out[name]['sklearn']:.4f}", flush=True)
    return out


def clustering_suite(cur):
    iris = load_iris()
    X = iris.data.astype(np.float32)
    y = iris.target
    out = {}
    cur.execute("SELECT pgml_kmeans_predict_batch(pgml_kmeans(%s::float4[][], 3, 50, 0), %s::float4[][])",
                (X.tolist(), X.tolist()))
    pg = np.asarray(cur.fetchone()[0])
    sk = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(X)
    out["kmeans_iris"] = {"pgml": round(float(adjusted_rand_score(y, pg)), 4),
                          "sklearn": round(float(adjusted_rand_score(y, sk)), 4)}
    print(f"  kmeans iris  pgml {out['kmeans_iris']['pgml']:.3f}  sklearn {out['kmeans_iris']['sklearn']:.3f}", flush=True)

    cur.execute("SELECT pgml_kmeans_predict_batch(pgml_kmeans_random(%s::float4[][], 3, 50, 0), %s::float4[][])",
                (X.tolist(), X.tolist()))
    rnd = np.asarray(cur.fetchone()[0])
    out["kmeans_init_iris"] = {
        "kmeans++_ari": out["kmeans_iris"]["pgml"],
        "random_ari": round(float(adjusted_rand_score(y, rnd)), 4),
    }
    print(f"  kmeans init  ++ {out['kmeans_init_iris']['kmeans++_ari']:.3f}  random {out['kmeans_init_iris']['random_ari']:.3f}", flush=True)

    for linkage, sk_link in ((0, "single"), (1, "complete")):
        cur.execute("SELECT pgml_hierarchical(%s::float4[][], 3, %s)", (X.tolist(), linkage))
        pg = np.asarray(cur.fetchone()[0])
        sk = AgglomerativeClustering(n_clusters=3, linkage=sk_link).fit_predict(X)
        out[f"hierarchical_{sk_link}"] = {
            "pgml": round(float(adjusted_rand_score(y, pg)), 4),
            "sklearn": round(float(adjusted_rand_score(y, sk)), 4),
        }
        print(f"  hier {sk_link:8} pgml {out[f'hierarchical_{sk_link}']['pgml']:.3f}  sklearn {out[f'hierarchical_{sk_link}']['sklearn']:.3f}", flush=True)

    Xb, yb = make_blobs(n_samples=400, centers=3, cluster_std=0.40, random_state=0)
    Xb = Xb.astype(np.float32)
    chosen = None
    for eps in (0.4, 0.6, 0.8, 1.0, 1.2):
        sk = DBSCAN(eps=eps, min_samples=5).fit_predict(Xb)
        ari = float(adjusted_rand_score(yb, sk))
        if ari >= 0.999:
            chosen = eps
            break
    if chosen is None:
        chosen = 0.8
    cur.execute("SELECT pgml_dbscan(%s::float4[][], %s, 5)", (Xb.tolist(), chosen))
    pg = np.asarray(cur.fetchone()[0])
    sk = DBSCAN(eps=chosen, min_samples=5).fit_predict(Xb)
    out["dbscan_blobs"] = {
        "eps": chosen,
        "pgml": round(float(adjusted_rand_score(yb, pg)), 4),
        "sklearn": round(float(adjusted_rand_score(yb, sk)), 4),
    }
    print(f"  dbscan blobs pgml {out['dbscan_blobs']['pgml']:.3f}  sklearn {out['dbscan_blobs']['sklearn']:.3f}  eps {chosen}", flush=True)
    return out


def pca_suite(cur):
    X, _ = load_breast_cancer(return_X_y=True)
    Xs = StandardScaler().fit_transform(X).astype(np.float32)
    cur.execute("SELECT pgml_pca_eigenvalues(pgml_pca_fit(%s::float4[][], 5))", (Xs.tolist(),))
    ev = np.clip(np.asarray(cur.fetchone()[0], dtype=float), 0, None)
    ratio = ev / ev.sum()
    sk = PCA(n_components=5).fit(Xs).explained_variance_ratio_
    out = {
        "pgml": [round(float(v), 4) for v in ratio[:5]],
        "sklearn": [round(float(v), 4) for v in sk],
        "max_abs_diff": round(float(np.max(np.abs(ratio[:5] - sk))), 4),
    }
    print(f"  pca wdbc     pgml {out['pgml']}  sklearn {out['sklearn']}  max|.| {out['max_abs_diff']}", flush=True)
    return out


def categorical_suite(cur):
    rng = np.random.default_rng(0)
    n = 300
    x0 = rng.normal(size=n).astype(np.float32)
    x1 = rng.normal(size=n).astype(np.float32)
    cat = rng.integers(0, 3, size=n).astype(np.float32)
    X = np.column_stack([x0, x1, cat])
    y = (cat == 0).astype(np.int64)
    cur.execute(
        "SELECT pgml_predict_tree_batch(pgml_train_tree_cat(%s::float4[][], %s::int[], 4, 2, %s::int[]), %s::float4[][])",
        (X.tolist(), y.tolist(), [0, 0, 1], X.tolist()),
    )
    cat_acc = acc(y, cur.fetchone()[0])
    cur.execute(
        "SELECT pgml_predict_tree_batch(pgml_train_tree(%s::float4[][], %s::int[], 4, 2), %s::float4[][])",
        (X.tolist(), y.tolist(), X.tolist()),
    )
    num_acc = acc(y, cur.fetchone()[0])
    out = {"categorical_tree": round(cat_acc, 4), "numeric_tree": round(num_acc, 4)}
    print(f"  categorical  cat {out['categorical_tree']:.3f}  numeric {out['numeric_tree']:.3f}", flush=True)
    return out


def store_xy(cur, table, X, y=None, y_kind="int"):
    cur.execute(f"DROP TABLE IF EXISTS {table}")
    if y is None:
        cur.execute(f"CREATE TABLE {table} (x float4[][])")
        cur.execute(f"INSERT INTO {table} VALUES (%s::float4[][])", (np.asarray(X, dtype=np.float32).tolist(),))
    elif y_kind == "int":
        cur.execute(f"CREATE TABLE {table} (x float4[][], y int[])")
        cur.execute(
            f"INSERT INTO {table} VALUES (%s::float4[][], %s::int[])",
            (np.asarray(X, dtype=np.float32).tolist(), np.asarray(y).tolist()),
        )
    else:
        cur.execute(f"CREATE TABLE {table} (x float4[][], y float4[])")
        cur.execute(
            f"INSERT INTO {table} VALUES (%s::float4[][], %s::float4[])",
            (np.asarray(X, dtype=np.float32).tolist(), np.asarray(y, dtype=np.float32).tolist()),
        )


def time_call(cur, sql):
    cur.execute(f"SELECT extract(epoch FROM (clock_timestamp() - t0)) * 1000 FROM (SELECT clock_timestamp() AS t0, ({sql}) AS payload) s")
    return float(cur.fetchone()[0])


def timing_suite(cur):
    rng = np.random.default_rng(0)
    out = {}

    print("  hist speedup n=5000 d=20 ...", flush=True)
    X = rng.normal(size=(5000, 20)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
    store_xy(cur, "bench_cls", X, y)
    pairs = {
        "tree": ("pgml_train_tree(x, y, 6, 2)", "pgml_train_tree_fast(x, y, 6, 2)"),
        "forest": ("pgml_train_forest(x, y, 8, 6, 2)", "pgml_train_forest_fast(x, y, 8, 6, 2)"),
        "gbm": ("pgml_train_gbm(x, y, 10, 3, 0.1)", "pgml_train_gbm_fast(x, y, 10, 3, 0.1)"),
    }
    hist = {}
    for name, (base, fast) in pairs.items():
        b = time_call(cur, f"SELECT {base} FROM bench_cls")
        f = time_call(cur, f"SELECT {fast} FROM bench_cls")
        hist[name] = {"baseline_ms": round(b, 1), "histogram_ms": round(f, 1), "speedup": round(b / f, 2)}
        print(f"    {name:8} {b:.0f} ms -> {f:.0f} ms  ({hist[name]['speedup']:.2f}x)", flush=True)
    out["histogram"] = hist

    print("  dbscan grid vs brute ...", flush=True)
    db = {}
    for n in (2000, 4000, 8000):
        blobs, _ = make_blobs(n_samples=n, centers=4, cluster_std=0.35, random_state=0)
        store_xy(cur, "bench_db", blobs.astype(np.float32))
        brute = time_call(cur, "SELECT pgml_dbscan(x, 0.5, 5) FROM bench_db")
        grid = time_call(cur, "SELECT pgml_dbscan_fast(x, 0.5, 5) FROM bench_db")
        db[str(n)] = {"brute_ms": round(brute, 1), "grid_ms": round(grid, 1), "speedup": round(brute / grid, 2)}
        print(f"    n={n:<5} brute {brute:.0f} ms  grid {grid:.0f} ms  ({db[str(n)]['speedup']:.2f}x)", flush=True)
    out["dbscan"] = db

    print("  k-means threads n=100000 k=8 ...", flush=True)
    X = rng.normal(size=(100000, 20)).astype(np.float32)
    store_xy(cur, "bench_km", X)
    km = {}
    for threads in (1, 4, 8):
        ms = time_call(cur, f"SELECT pgml_kmeans_parallel(x, 8, 20, 0, {threads}) FROM bench_km")
        km[str(threads)] = round(ms, 1)
        print(f"    {threads} thread(s)  {ms:.0f} ms", flush=True)
    out["kmeans_parallel_ms"] = km
    out["kmeans_speedup_4x"] = round(km["1"] / km["4"], 2)
    out["kmeans_speedup_8x"] = round(km["1"] / km["8"], 2)

    print("  large-scale hist/svm n=100000 d=20 ...", flush=True)
    y = (X[:, 0] > 0).astype(np.int64)
    store_xy(cur, "bench_large", X, y)
    large = {}
    for name, sql in (
        ("tree_hist", "pgml_train_tree_fast(x, y, 6, 20)"),
        ("forest_hist", "pgml_train_forest_fast(x, y, 8, 6, 20)"),
        ("gbm_hist", "pgml_train_gbm_fast(x, y, 10, 3, 0.1)"),
        ("svm", "pgml_train_svm(x, y, 0.001, 5)"),
    ):
        ms = time_call(cur, f"SELECT {sql} FROM bench_large")
        large[name] = round(ms, 1)
        print(f"    {name:12} {ms/1000:.2f} s", flush=True)
    out["large_scale_ms"] = large

    print("  linreg / pca scale ...", flush=True)
    scale = {}
    for n in (10000, 100000):
        Xr = rng.normal(size=(n, 20)).astype(np.float32)
        yr = (Xr[:, 0] * 2 + rng.normal(size=n)).astype(np.float32)
        store_xy(cur, "bench_reg", Xr, yr, y_kind="float")
        ms = time_call(cur, "SELECT pgml_linreg(x, y) FROM bench_reg")
        scale[f"linreg_n{n}_d20_ms"] = round(ms, 1)
        print(f"    linreg n={n}  {ms/1000:.2f} s", flush=True)
    for n, d in ((5000, 30), (20000, 50)):
        Xp = rng.normal(size=(n, d)).astype(np.float32)
        store_xy(cur, "bench_pca", Xp)
        ms = time_call(cur, "SELECT pgml_pca_fit(x, 5) FROM bench_pca")
        scale[f"pca_n{n}_d{d}_ms"] = round(ms, 1)
        print(f"    pca n={n} d={d}  {ms/1000:.2f} s", flush=True)
    out["scale_ms"] = scale
    return out


def main():
    conn, cur = connect()
    results = {}
    print("== classification ==", flush=True)
    X, y = load_breast_cancer(return_X_y=True)
    results["wdbc"] = cls_suite(cur, "wdbc", X, y)
    results["xor"] = xor_suite(cur)
    try:
        Xi, yi = load_uci_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data")
        results["ionosphere"] = cls_suite(cur, "ionosphere", Xi, yi)
    except Exception as exc:
        results["ionosphere"] = {"error": str(exc)}
        print(f"  ionosphere skipped: {exc}", flush=True)
    try:
        Xs, ys = load_uci_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data")
        results["sonar"] = cls_suite(cur, "sonar", Xs, ys)
    except Exception as exc:
        results["sonar"] = {"error": str(exc)}
        print(f"  sonar skipped: {exc}", flush=True)

    print("== iris multiclass ==", flush=True)
    results["iris_multiclass"] = iris_multiclass(cur)
    print("== regression ==", flush=True)
    results["diabetes"] = regression_suite(cur)
    print("== clustering ==", flush=True)
    results["clustering"] = clustering_suite(cur)
    print("== pca ==", flush=True)
    results["pca"] = pca_suite(cur)
    print("== categorical ==", flush=True)
    results["categorical"] = categorical_suite(cur)
    print("== timing ==", flush=True)
    t0 = time.perf_counter()
    results["timing"] = timing_suite(cur)
    results["timing_wall_s"] = round(time.perf_counter() - t0, 1)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    km = results["timing"]["kmeans_parallel_ms"]
    extended = {
        "multithreading": {
            "note": (
                "Clean SQL-side timing (array pre-materialized, no client serialization). "
                "PostgreSQL 18. n=100000, d=20, k=8, 20 iterations."
            ),
            "kmeans_parallel_ms": {
                "1_thread": km["1"],
                "4_threads": km["4"],
                "8_threads": km["8"],
            },
            "kmeans_speedup_4x": results["timing"]["kmeans_speedup_4x"],
            "kmeans_speedup_8x": results["timing"]["kmeans_speedup_8x"],
        },
        "categorical_features": {
            "test": "3 features, feature 2 categorical {0,1,2}, label = (category==0)",
            "categorical_tree_accuracy": results["categorical"]["categorical_tree"],
            "numeric_tree_accuracy": results["categorical"]["numeric_tree"],
        },
    }
    with open(EXT, "w", encoding="utf-8") as f:
        json.dump(extended, f, indent=2)
        f.write("\n")
    print(f"wrote {OUT}", flush=True)
    print(f"wrote {EXT}", flush=True)
    conn.close()


if __name__ == "__main__":
    main()
