#!/usr/bin/env python3
"""Categorical-feature comparison: pgml cat splits vs codes vs one-hot vs CatBoost/LightGBM.

Grid: category cardinality × fraction of rare categories.
Reports quality, train (incl. encoding), inference, model size, memory.
"""
from __future__ import annotations

import json
import pickle
import tempfile
import time
import traceback
from pathlib import Path
_REPO = Path(__file__).resolve().parents[2]

import lightgbm as lgb
import numpy as np
import psutil
import psycopg2
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

OUT = Path(_REPO.joinpath('results', 'categorical_comparison.json'))
PG = dict(host="localhost", port=5433, dbname="ml_test", user="postgres", password="postgres")

N_ROWS = 20_000
N_NUM = 4
N_CAT = 4
MAX_DEPTH = 6
MIN_SAMPLES = 5
N_EST = 50  # for boosting competitors
SEED = 0

CARDINALITIES = [8, 32, 128, 512]
RARE_FRACTIONS = [0.0, 0.25, 0.5, 0.75]  # share of category ids marked rare


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
    elapsed = (time.perf_counter() - t0) * 1000
    after = proc.memory_info().rss
    peak = max(base, after)
    return out, elapsed, peak / (1024 * 1024), (peak - base) / (1024 * 1024)


def make_dataset(n, cardinality, rare_frac, rng):
    """Synthetic binary task where labels depend on categorical membership.

    For each categorical feature, a subset of values is 'positive'. Rare
    categories share a small probability mass so high rare_frac + high K
    stresses encoding and split search.
    """
    X_num = rng.normal(size=(n, N_NUM)).astype(np.float32)
    X_cat = np.zeros((n, N_CAT), dtype=np.int32)

    n_rare = int(round(rare_frac * cardinality))
    n_freq = max(1, cardinality - n_rare)
    # frequent cats share 90% mass; rare cats share 10% (or uniform if no rare)
    if n_rare == 0:
        probs = np.ones(cardinality) / cardinality
        rare_ids = np.array([], dtype=int)
        freq_ids = np.arange(cardinality)
    else:
        probs = np.zeros(cardinality)
        freq_ids = np.arange(n_freq)
        rare_ids = np.arange(n_freq, cardinality)
        probs[freq_ids] = 0.90 / n_freq
        probs[rare_ids] = 0.10 / n_rare

    for j in range(N_CAT):
        X_cat[:, j] = rng.choice(cardinality, size=n, p=probs)

    # Label hinges on equality to a few concrete category ids — the regime
    # where one-category-vs-rest splits are expressive. Discriminative ids are
    # drawn from the frequent set so rare_frac controls mass imbalance, not
    # whether the signal category is observed.
    key0 = int(freq_ids[0])
    key1 = int(freq_ids[min(1, len(freq_ids) - 1)])
    key2 = int(freq_ids[min(2, len(freq_ids) - 1)])
    score = (
        1.4 * (X_cat[:, 0] == key0).astype(np.float64)
        + 1.1 * (X_cat[:, 1] == key1).astype(np.float64)
        + 0.7 * (X_cat[:, 2] == key2).astype(np.float64)
        + 0.2 * X_num[:, 0]
        - 0.15 * X_num[:, 1]
    )
    # small noise so accuracy is not trivially 1.0
    p = 1.0 / (1.0 + np.exp(-(score - 0.9)))
    y = (rng.random(n) < p).astype(np.int32)

    meta = {
        "n": n,
        "cardinality": cardinality,
        "rare_frac": rare_frac,
        "n_rare": int(n_rare),
        "n_freq": int(n_freq),
        "n_num": N_NUM,
        "n_cat": N_CAT,
        "key_cats": [key0, key1, key2],
    }
    return X_num, X_cat, y, meta


def quality(y_true, pred, proba=None):
    acc = float(accuracy_score(y_true, pred))
    auc = None
    if proba is not None:
        try:
            auc = float(roc_auc_score(y_true, proba))
        except ValueError:
            auc = None
    return {"accuracy": acc, "auc": auc, "primary": acc}


def pack_xy(cur, X, y, ftypes=None):
    cur.execute("DROP TABLE IF EXISTS cat_pack")
    if ftypes is None:
        cur.execute("CREATE TABLE cat_pack (x float4[][], y int[])")
        cur.execute(
            "INSERT INTO cat_pack VALUES (%s::float4[][], %s::int[])",
            (np.asarray(X, np.float32).tolist(), np.asarray(y, np.int64).tolist()),
        )
    else:
        cur.execute("CREATE TABLE cat_pack (x float4[][], y int[], ft int[])")
        cur.execute(
            "INSERT INTO cat_pack VALUES (%s::float4[][], %s::int[], %s::int[])",
            (
                np.asarray(X, np.float32).tolist(),
                np.asarray(y, np.int64).tolist(),
                list(map(int, ftypes)),
            ),
        )


def run_pgml_cat(cur, X_num, X_cat, ytr, Xte_num, Xte_cat, yte):
    Xtr = np.hstack([X_num, X_cat.astype(np.float32)])
    Xte = np.hstack([Xte_num, Xte_cat.astype(np.float32)])
    ftypes = [0] * N_NUM + [1] * N_CAT
    enc_ms = 0.0  # codes already integers; no one-hot
    pack_xy(cur, Xtr, ytr, ftypes)
    base = pg_ctx_mb(cur)
    train_ms, model = clock_ms(
        cur,
        lambda: cur.execute(
            "SELECT pgml_train_tree_cat(x, y, %s, %s, ft) FROM cat_pack",
            (MAX_DEPTH, MIN_SAMPLES),
        )
        or cur.fetchone()[0],
    )
    peak = max(base, pg_ctx_mb(cur))
    t0 = time.perf_counter()
    cur.execute(
        "SELECT pgml_predict_tree_batch(%s, %s::float4[][])",
        (psycopg2.Binary(model), Xte.tolist()),
    )
    pred = np.asarray(cur.fetchone()[0], dtype=np.int32)
    infer_ms = (time.perf_counter() - t0) * 1000
    encode_train_ms = enc_ms + train_ms
    total_pipeline_ms = enc_ms + train_ms + infer_ms
    return {
        "system": "pgml_cat",
        "encode_ms": enc_ms,
        "train_ms": train_ms,
        "infer_ms": infer_ms,
        "encode_train_ms": encode_train_ms,
        "total_pipeline_ms": total_pipeline_ms,
        "total_pipeline_definition": "encode_ms+train_ms+infer_ms; excludes pack_xy",
        "is_full_end_to_end": False,
        "model_bytes": len(model),
        "peak_mem_mb": peak,
        "mem_delta_mb": peak - base,
        **{
            "memory_metric": "pg_backend_memory_contexts_sum_total_bytes",
            "memory_scope": "backend",
            "memory_measurement_method": "pre/post pg_backend_memory_contexts snapshot around train",
            "memory_is_peak": False,
            "memory_comparable_across_systems": False,
        },
        "encoded_width": int(Xtr.shape[1]),
        "quality": quality(yte, pred),
        "notes": "native one-category-vs-rest splits via feature_types",
        "arithmetic_check": {
            "encode_train_ms_ok": abs(encode_train_ms - (enc_ms + train_ms)) < 1e-9,
            "total_pipeline_ms_ok": abs(total_pipeline_ms - (enc_ms + train_ms + infer_ms)) < 1e-9,
        },
    }


def run_pgml_codes(cur, X_num, X_cat, ytr, Xte_num, Xte_cat, yte):
    Xtr = np.hstack([X_num, X_cat.astype(np.float32)])
    Xte = np.hstack([Xte_num, Xte_cat.astype(np.float32)])
    pack_xy(cur, Xtr, ytr)
    base = pg_ctx_mb(cur)
    train_ms, model = clock_ms(
        cur,
        lambda: cur.execute(
            "SELECT pgml_train_tree(x, y, %s, %s) FROM cat_pack",
            (MAX_DEPTH, MIN_SAMPLES),
        )
        or cur.fetchone()[0],
    )
    peak = max(base, pg_ctx_mb(cur))
    t0 = time.perf_counter()
    cur.execute(
        "SELECT pgml_predict_tree_batch(%s, %s::float4[][])",
        (psycopg2.Binary(model), Xte.tolist()),
    )
    pred = np.asarray(cur.fetchone()[0], dtype=np.int32)
    infer_ms = (time.perf_counter() - t0) * 1000
    encode_train_ms = 0.0 + train_ms
    total_pipeline_ms = train_ms + infer_ms
    return {
        "system": "pgml_codes",
        "encode_ms": 0.0,
        "train_ms": train_ms,
        "infer_ms": infer_ms,
        "encode_train_ms": encode_train_ms,
        "total_pipeline_ms": total_pipeline_ms,
        "total_pipeline_definition": "encode_ms+train_ms+infer_ms; excludes pack_xy",
        "is_full_end_to_end": False,
        "model_bytes": len(model),
        "peak_mem_mb": peak,
        "mem_delta_mb": peak - base,
        **{
            "memory_metric": "pg_backend_memory_contexts_sum_total_bytes",
            "memory_scope": "backend",
            "memory_measurement_method": "pre/post pg_backend_memory_contexts snapshot around train",
            "memory_is_peak": False,
            "memory_comparable_across_systems": False,
        },
        "encoded_width": int(Xtr.shape[1]),
        "quality": quality(yte, pred),
        "notes": "integer category codes treated as ordered numeric thresholds",
        "arithmetic_check": {
            "encode_train_ms_ok": True,
            "total_pipeline_ms_ok": abs(total_pipeline_ms - (train_ms + infer_ms)) < 1e-9,
        },
    }


def run_onehot_tree(X_num, X_cat, ytr, Xte_num, Xte_cat, yte):
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    t0 = time.perf_counter()
    Xtr_oh = enc.fit_transform(X_cat)
    encode_ms = (time.perf_counter() - t0) * 1000
    Xte_oh = enc.transform(Xte_cat)
    Xtr = np.hstack([X_num, Xtr_oh])
    Xte = np.hstack([Xte_num, Xte_oh])
    clf = DecisionTreeClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_SAMPLES, random_state=SEED)

    def fit():
        clf.fit(Xtr, ytr)

    _, train_ms, peak, delta = peak_rss(fit)
    t1 = time.perf_counter()
    pred = clf.predict(Xte)
    proba = clf.predict_proba(Xte)[:, 1]
    infer_ms = (time.perf_counter() - t1) * 1000
    blob = pickle.dumps(clf, protocol=pickle.HIGHEST_PROTOCOL)
    encode_train_ms = encode_ms + train_ms
    total_pipeline_ms = encode_ms + train_ms + infer_ms
    return {
        "system": "onehot_sklearn_tree",
        "encode_ms": encode_ms,
        "train_ms": train_ms,
        "infer_ms": infer_ms,
        "encode_train_ms": encode_train_ms,
        "total_pipeline_ms": total_pipeline_ms,
        "total_pipeline_definition": (
            "encode_ms+train_ms+infer_ms; test OneHot transform+hstack outside timers"
        ),
        "is_full_end_to_end": False,
        "model_bytes": len(blob),
        "peak_mem_mb": peak,
        "mem_delta_mb": delta,
        **{
            "memory_metric": "process_rss_delta_around_fit",
            "memory_scope": "process",
            "memory_measurement_method": "psutil RSS before/after fit (not true peak sampler)",
            "memory_is_peak": False,
            "memory_comparable_across_systems": False,
        },
        "encoded_width": int(Xtr.shape[1]),
        "onehot_bytes": int(Xtr_oh.nbytes + Xte_oh.nbytes),
        "quality": quality(yte, pred, proba),
        "notes": "OneHotEncoder + sklearn DecisionTree",
        "arithmetic_check": {
            "encode_train_ms_ok": abs(encode_train_ms - (encode_ms + train_ms)) < 1e-9,
            "total_pipeline_ms_ok": abs(total_pipeline_ms - (encode_ms + train_ms + infer_ms)) < 1e-9,
        },
    }


def run_onehot_histgbm(X_num, X_cat, ytr, Xte_num, Xte_cat, yte):
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    t0 = time.perf_counter()
    Xtr_oh = enc.fit_transform(X_cat)
    encode_ms = (time.perf_counter() - t0) * 1000
    Xte_oh = enc.transform(Xte_cat)
    Xtr = np.hstack([X_num, Xtr_oh])
    Xte = np.hstack([Xte_num, Xte_oh])
    # HistGBM max_bins < 256; one-hot is binary so fine
    clf = HistGradientBoostingClassifier(
        max_iter=N_EST, max_depth=MAX_DEPTH, learning_rate=0.1, random_state=SEED
    )

    def fit():
        clf.fit(Xtr, ytr)

    _, train_ms, peak, delta = peak_rss(fit)
    t1 = time.perf_counter()
    pred = clf.predict(Xte)
    proba = clf.predict_proba(Xte)[:, 1]
    infer_ms = (time.perf_counter() - t1) * 1000
    blob = pickle.dumps(clf, protocol=pickle.HIGHEST_PROTOCOL)
    encode_train_ms = encode_ms + train_ms
    total_pipeline_ms = encode_ms + train_ms + infer_ms
    return {
        "system": "onehot_sklearn_histgbm",
        "encode_ms": encode_ms,
        "train_ms": train_ms,
        "infer_ms": infer_ms,
        "encode_train_ms": encode_train_ms,
        "total_pipeline_ms": total_pipeline_ms,
        "total_pipeline_definition": (
            "encode_ms+train_ms+infer_ms; test OneHot transform+hstack outside timers"
        ),
        "is_full_end_to_end": False,
        "model_bytes": len(blob),
        "peak_mem_mb": peak,
        "mem_delta_mb": delta,
        **{
            "memory_metric": "process_rss_delta_around_fit",
            "memory_scope": "process",
            "memory_measurement_method": "psutil RSS before/after fit (not true peak sampler)",
            "memory_is_peak": False,
            "memory_comparable_across_systems": False,
        },
        "encoded_width": int(Xtr.shape[1]),
        "onehot_bytes": int(Xtr_oh.nbytes + Xte_oh.nbytes),
        "quality": quality(yte, pred, proba),
        "notes": "OneHotEncoder + sklearn HistGradientBoosting",
        "arithmetic_check": {
            "encode_train_ms_ok": abs(encode_train_ms - (encode_ms + train_ms)) < 1e-9,
            "total_pipeline_ms_ok": abs(total_pipeline_ms - (encode_ms + train_ms + infer_ms)) < 1e-9,
        },
    }


def run_catboost(X_num, X_cat, ytr, Xte_num, Xte_cat, yte):
    # CatBoost rejects float arrays for cat columns — keep cats as int.
    Xtr = np.hstack([X_num.astype(np.float64), X_cat.astype(np.int32)])
    Xte = np.hstack([Xte_num.astype(np.float64), Xte_cat.astype(np.int32)])
    # object array so dtypes are not unified to float
    Xtr_obj = np.empty((Xtr.shape[0], Xtr.shape[1]), dtype=object)
    Xte_obj = np.empty((Xte.shape[0], Xte.shape[1]), dtype=object)
    for j in range(N_NUM):
        Xtr_obj[:, j] = Xtr[:, j]
        Xte_obj[:, j] = Xte[:, j]
    for j in range(N_CAT):
        Xtr_obj[:, N_NUM + j] = X_cat[:, j].astype(np.int32)
        Xte_obj[:, N_NUM + j] = Xte_cat[:, j].astype(np.int32)
    cat_idx = list(range(N_NUM, N_NUM + N_CAT))
    train_pool = Pool(Xtr_obj, ytr, cat_features=cat_idx)
    test_pool = Pool(Xte_obj, cat_features=cat_idx)
    clf = CatBoostClassifier(
        iterations=N_EST,
        depth=MAX_DEPTH,
        learning_rate=0.1,
        loss_function="Logloss",
        random_seed=SEED,
        verbose=False,
        allow_writing_files=False,
        thread_count=1,
    )

    def fit():
        clf.fit(train_pool)

    _, train_ms, peak, delta = peak_rss(fit)
    t1 = time.perf_counter()
    pred = clf.predict(test_pool).astype(np.int32).ravel()
    proba = clf.predict_proba(test_pool)[:, 1]
    infer_ms = (time.perf_counter() - t1) * 1000
    with tempfile.NamedTemporaryFile(suffix=".cbm", delete=False) as tmp:
        path = tmp.name
    try:
        clf.save_model(path, format="cbm")
        model_bytes = Path(path).stat().st_size
    finally:
        Path(path).unlink(missing_ok=True)
    total_pipeline_ms = train_ms + infer_ms
    return {
        "system": "catboost_native",
        "encode_ms": 0.0,
        "train_ms": train_ms,
        "infer_ms": infer_ms,
        "encode_train_ms": train_ms,
        "total_pipeline_ms": total_pipeline_ms,
        "total_pipeline_definition": "encode_ms+train_ms+infer_ms; Pool build outside timers",
        "is_full_end_to_end": False,
        "model_bytes": model_bytes,
        "peak_mem_mb": peak,
        "mem_delta_mb": delta,
        **{
            "memory_metric": "process_rss_delta_around_fit",
            "memory_scope": "process",
            "memory_measurement_method": "psutil RSS before/after fit (not true peak sampler)",
            "memory_is_peak": False,
            "memory_comparable_across_systems": False,
        },
        "encoded_width": int(Xtr.shape[1]),
        "quality": quality(yte, pred, proba),
        "notes": "CatBoost ordered target stats / native cat handling",
        "arithmetic_check": {
            "encode_train_ms_ok": True,
            "total_pipeline_ms_ok": abs(total_pipeline_ms - (train_ms + infer_ms)) < 1e-9,
        },
    }


def run_lightgbm_cat(X_num, X_cat, ytr, Xte_num, Xte_cat, yte):
    Xtr = np.hstack([X_num, X_cat.astype(np.float32)])
    Xte = np.hstack([Xte_num, Xte_cat.astype(np.float32)])
    cat_idx = list(range(N_NUM, N_NUM + N_CAT))
    clf = lgb.LGBMClassifier(
        n_estimators=N_EST,
        max_depth=MAX_DEPTH,
        num_leaves=min(63, 2**MAX_DEPTH),
        learning_rate=0.1,
        random_state=SEED,
        n_jobs=1,
        verbosity=-1,
    )

    def fit():
        clf.fit(Xtr, ytr, categorical_feature=cat_idx)

    _, train_ms, peak, delta = peak_rss(fit)
    t1 = time.perf_counter()
    pred = clf.predict(Xte).astype(np.int32)
    proba = clf.predict_proba(Xte)[:, 1]
    infer_ms = (time.perf_counter() - t1) * 1000
    blob = pickle.dumps(clf, protocol=pickle.HIGHEST_PROTOCOL)
    total_pipeline_ms = train_ms + infer_ms
    return {
        "system": "lightgbm_native",
        "encode_ms": 0.0,
        "train_ms": train_ms,
        "infer_ms": infer_ms,
        "encode_train_ms": train_ms,
        "total_pipeline_ms": total_pipeline_ms,
        "total_pipeline_definition": "encode_ms+train_ms+infer_ms",
        "is_full_end_to_end": False,
        "model_bytes": len(blob),
        "peak_mem_mb": peak,
        "mem_delta_mb": delta,
        **{
            "memory_metric": "process_rss_delta_around_fit",
            "memory_scope": "process",
            "memory_measurement_method": "psutil RSS before/after fit (not true peak sampler)",
            "memory_is_peak": False,
            "memory_comparable_across_systems": False,
        },
        "encoded_width": int(Xtr.shape[1]),
        "quality": quality(yte, pred, proba),
        "notes": "LightGBM categorical_feature (exclusive feature bundling / cat splits)",
        "arithmetic_check": {
            "encode_train_ms_ok": True,
            "total_pipeline_ms_ok": abs(total_pipeline_ms - (train_ms + infer_ms)) < 1e-9,
        },
    }


RUNNERS = [
    ("pgml_cat", "pg"),
    ("pgml_codes", "pg"),
    ("onehot_sklearn_tree", "py"),
    ("onehot_sklearn_histgbm", "py"),
    ("catboost_native", "py"),
    ("lightgbm_native", "py"),
]


def run_one(cur, name, Xn, Xc, ytr, Xnte, Xcte, yte):
    if name == "pgml_cat":
        return run_pgml_cat(cur, Xn, Xc, ytr, Xnte, Xcte, yte)
    if name == "pgml_codes":
        return run_pgml_codes(cur, Xn, Xc, ytr, Xnte, Xcte, yte)
    if name == "onehot_sklearn_tree":
        return run_onehot_tree(Xn, Xc, ytr, Xnte, Xcte, yte)
    if name == "onehot_sklearn_histgbm":
        return run_onehot_histgbm(Xn, Xc, ytr, Xnte, Xcte, yte)
    if name == "catboost_native":
        return run_catboost(Xn, Xc, ytr, Xnte, Xcte, yte)
    if name == "lightgbm_native":
        return run_lightgbm_cat(Xn, Xc, ytr, Xnte, Xcte, yte)
    raise ValueError(name)


def main():
    conn, cur = connect()
    points = []
    cells = [(k, r) for k in CARDINALITIES for r in RARE_FRACTIONS]
    print(f"cells={len(cells)} systems={len(RUNNERS)} n={N_ROWS}", flush=True)

    for i, (card, rare) in enumerate(cells, 1):
        rng = np.random.default_rng(SEED + card * 17 + int(rare * 100))
        X_num, X_cat, y, meta = make_dataset(N_ROWS, card, rare, rng)
        Xn_tr, Xn_te, Xc_tr, Xc_te, ytr, yte = train_test_split(
            X_num, X_cat, y, test_size=0.25, random_state=SEED, stratify=y
        )
        print(f"[{i}/{len(cells)}] K={card} rare={rare}", flush=True)
        for name, _ in RUNNERS:
            try:
                row = run_one(cur, name, Xn_tr, Xc_tr, ytr, Xn_te, Xc_te, yte)
                point = {**meta, **row}
                points.append(point)
                q = row["quality"]["primary"]
                print(
                    f"  {name:24s} train={row['train_ms']:.1f}ms "
                    f"enc={row['encode_ms']:.1f}ms infer={row['infer_ms']:.1f}ms "
                    f"acc={q:.4f} model={row['model_bytes']}B "
                    f"width={row['encoded_width']}",
                    flush=True,
                )
            except Exception as e:
                print(f"  ! {name}: {e}", flush=True)
                points.append({
                    **meta,
                    "system": name,
                    "error": str(e),
                    "trace": traceback.format_exc()[-500:],
                })
                try:
                    conn.close()
                except Exception:
                    pass
                time.sleep(1)
                conn, cur = connect()

    payload = {
        "hyperparams": {
            "n_rows": N_ROWS,
            "n_num": N_NUM,
            "n_cat": N_CAT,
            "max_depth": MAX_DEPTH,
            "min_samples": MIN_SAMPLES,
            "n_estimators_boosting": N_EST,
            "seed": SEED,
            "note": "Comparison of categorical handling mechanisms, not equivalent algorithms",
        },
        "grid": {"cardinality": CARDINALITIES, "rare_fraction": RARE_FRACTIONS},
        "systems": [s for s, _ in RUNNERS],
        "points": points,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {OUT} ({len(points)} points)", flush=True)
    conn.close()


if __name__ == "__main__":
    main()
