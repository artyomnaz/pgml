#!/usr/bin/env python3
"""Public-benchmark suite for the native pgml extension.

Priority order requested by the paper:
  1. RelBench  — multi-table JOIN/aggregate features inside PostgreSQL
  2. TPC-H    — controlled scale factor, SQL feature cost before training
  3. TabArena — curated tabular splits via BeyondArena / data-foundry when available
  +. OpenML-CC18 — fixed classification suite as a simpler tabular baseline

All model training for pgml goes through localhost:5433. SQL-side clocks use
clock_timestamp() after the feature matrix is already stored.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
_REPO = Path(__file__).resolve().parents[2]

import numpy as np
import openml
import psycopg2
from psycopg2.extras import execute_values
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

OUT = Path(_REPO.joinpath('results', 'public_benchmarks.json'))
PG = dict(host="localhost", port=5433, dbname="ml_test", user="postgres", password="postgres")


def connect():
    conn = psycopg2.connect(**PG)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SET statement_timeout = 0")
    return conn, cur


def time_sql(cur, sql):
    cur.execute(
        "SELECT extract(epoch FROM (clock_timestamp() - t0)) * 1000 "
        "FROM (SELECT clock_timestamp() AS t0) s, LATERAL (" + sql + ") q"
    )
    return float(cur.fetchone()[0])


def store_pack(cur, table, X, y=None, y_kind="float"):
    cur.execute(f"DROP TABLE IF EXISTS {table}")
    if y is None:
        cur.execute(f"CREATE TABLE {table} (x float4[][])")
        cur.execute(
            f"INSERT INTO {table} VALUES (%s::float4[][])",
            (np.asarray(X, dtype=np.float32).tolist(),),
        )
        return
    if y_kind == "int":
        cur.execute(f"CREATE TABLE {table} (x float4[][], y int[])")
        cur.execute(
            f"INSERT INTO {table} VALUES (%s::float4[][], %s::int[])",
            (np.asarray(X, dtype=np.float32).tolist(), np.asarray(y, dtype=np.int64).tolist()),
        )
    else:
        cur.execute(f"CREATE TABLE {table} (x float4[][], y float4[])")
        cur.execute(
            f"INSERT INTO {table} VALUES (%s::float4[][], %s::float4[])",
            (np.asarray(X, dtype=np.float32).tolist(), np.asarray(y, dtype=np.float32).tolist()),
        )


def encode_matrix(train_df, test_df, feature_cols):
    """Numeric + categorical -> float32 matrix. Categoricals are integer-coded on train."""
    Xtr_parts, Xte_parts = [], []
    meta = []
    for col in feature_cols:
        tr = train_df[col]
        te = test_df[col]
        kind = str(tr.dtype)
        numeric = kind.startswith(("float", "Float", "int", "Int", "UInt", "bool", "boolean", "Boolean"))
        if numeric:
            trv = tr.astype(float).fillna(0.0).to_numpy()
            tev = te.astype(float).fillna(0.0).to_numpy()
            Xtr_parts.append(trv)
            Xte_parts.append(tev)
            meta.append({"col": col, "kind": "numeric"})
        else:
            enc = LabelEncoder()
            vals = tr.astype(str).fillna("__NA__").to_numpy()
            enc.fit(np.unique(np.concatenate([vals, te.astype(str).fillna("__NA__").to_numpy()])))
            Xtr_parts.append(enc.transform(vals).astype(float))
            Xte_parts.append(enc.transform(te.astype(str).fillna("__NA__").to_numpy()).astype(float))
            meta.append({"col": col, "kind": "categorical", "n_levels": int(len(enc.classes_))})
    Xtr = np.column_stack(Xtr_parts).astype(np.float32) if Xtr_parts else np.zeros((len(train_df), 0), np.float32)
    Xte = np.column_stack(Xte_parts).astype(np.float32) if Xte_parts else np.zeros((len(test_df), 0), np.float32)
    return Xtr, Xte, meta


def pandas_to_postgres(cur, schema, table, df, page_size=2000):
    cols = list(df.columns)
    coldefs = []
    for c in cols:
        s = df[c]
        kind = str(s.dtype)
        if "datetime" in kind or "timestamp" in kind:
            coldefs.append(f'"{c}" timestamptz')
            df[c] = s.astype("datetime64[ns]").astype(object).where(s.notna(), None)
        elif kind.startswith(("Int", "UInt")) or kind in ("int64", "int32", "int16", "int8"):
            coldefs.append(f'"{c}" bigint')
            df[c] = s.astype(object).where(s.notna(), None)
        elif kind.startswith("Float") or kind in ("float64", "float32"):
            coldefs.append(f'"{c}" double precision')
            df[c] = s.astype(object).where(s.notna(), None)
        elif kind in ("bool", "boolean", "Boolean"):
            coldefs.append(f'"{c}" boolean')
            df[c] = s.astype(object).where(s.notna(), None)
        else:
            coldefs.append(f'"{c}" text')
            df[c] = s.astype(object).where(s.notna(), None)
    cur.execute(f'DROP TABLE IF EXISTS {schema}."{table}" CASCADE')
    cur.execute(f'CREATE TABLE {schema}."{table}" ({", ".join(coldefs)})')
    def cell(v):
        if v is None:
            return None
        try:
            if isinstance(v, float) and np.isnan(v):
                return None
        except Exception:
            pass
        if hasattr(v, "item"):
            try:
                return v.item()
            except Exception:
                return v
        return v
    rows = [tuple(cell(v) for v in row) for row in df.itertuples(index=False, name=None)]
    if rows:
        execute_values(
            cur,
            f'INSERT INTO {schema}."{table}" VALUES %s',
            rows,
            page_size=page_size,
        )


# ---------------------------------------------------------------------------
# 1. RelBench
# ---------------------------------------------------------------------------

def relbench_suite(cur):
    import pandas as pd
    import relbench
    from relbench.base import EntityTask, TaskType

    print("== RelBench rel-f1 ==", flush=True)
    dataset = relbench.load_dataset("rel-f1")
    db = dataset.get_db()
    cur.execute("DROP SCHEMA IF EXISTS relbench_f1 CASCADE")
    cur.execute("CREATE SCHEMA relbench_f1")
    load_t0 = time.perf_counter()
    for tname, table in db.table_dict.items():
        pandas_to_postgres(cur, "relbench_f1", tname, table.df.copy())
        print(f"  loaded {tname}: {len(table.df)} rows", flush=True)
    load_s = round(time.perf_counter() - load_t0, 2)

    out = {"dataset": "rel-f1", "load_tables_s": load_s, "tasks": {}}
    task_names = ["driver-dnf", "driver-position", "driver-top3"]
    for task_name in task_names:
        task = dataset.load_task(task_name)
        if not isinstance(task, EntityTask):
            continue
        train_table = task.get_table("train")
        test_table = task.get_table("test", mask_input_cols=False)
        ecol = task.entity_col
        entity = task.entity_table
        target = task.target_col

        # Stage label tables.
        pandas_to_postgres(cur, "relbench_f1", f"labels_{task_name.replace('-', '_')}_train", train_table.df.copy())
        pandas_to_postgres(cur, "relbench_f1", f"labels_{task_name.replace('-', '_')}_test", test_table.df.copy())
        train_lbl = f'relbench_f1."labels_{task_name.replace("-", "_")}_train"'
        test_lbl = f'relbench_f1."labels_{task_name.replace("-", "_")}_test"'

        # Multi-table features: entity attributes + history aggregates from results/races
        # strictly before the prediction date (no future leakage).
        def feat_query(label_rel):
            return f"""
            SELECT
              l."{ecol}" AS entity_id,
              l.date AS asof,
              EXTRACT(EPOCH FROM l.date)::float8 AS ts_epoch,
              EXTRACT(YEAR FROM d.dob)::float8 AS birth_year,
              d.nationality AS nationality,
              COALESCE(h.hist_races, 0) AS hist_races,
              COALESCE(h.hist_avg_points, 0) AS hist_avg_points,
              COALESCE(h.hist_avg_pos, 0) AS hist_avg_pos,
              COALESCE(h.hist_avg_grid, 0) AS hist_avg_grid,
              COALESCE(h.hist_points_finishes, 0) AS hist_points_finishes,
              l."{target}" AS target
            FROM {label_rel} l
            JOIN relbench_f1.drivers d ON d."driverId" = l."{ecol}"
            LEFT JOIN LATERAL (
              SELECT
                COUNT(*)::float8 AS hist_races,
                AVG(r.points)::float8 AS hist_avg_points,
                AVG(r."positionOrder")::float8 AS hist_avg_pos,
                AVG(r.grid)::float8 AS hist_avg_grid,
                SUM(CASE WHEN r.points > 0 THEN 1 ELSE 0 END)::float8 AS hist_points_finishes
              FROM relbench_f1.results r
              JOIN relbench_f1.races ra ON ra."raceId" = r."raceId"
              WHERE r."driverId" = l."{ecol}" AND ra.date < l.date
            ) h ON TRUE
            """

        # Time SQL feature materialization for train+test.
        cur.execute(f"DROP TABLE IF EXISTS relbench_f1.feat_{task_name.replace('-', '_')}_train")
        cur.execute(f"DROP TABLE IF EXISTS relbench_f1.feat_{task_name.replace('-', '_')}_test")
        t_feat = time.perf_counter()
        cur.execute(
            f"CREATE TABLE relbench_f1.feat_{task_name.replace('-', '_')}_train AS {feat_query(train_lbl)}"
        )
        cur.execute(
            f"CREATE TABLE relbench_f1.feat_{task_name.replace('-', '_')}_test AS {feat_query(test_lbl)}"
        )
        feat_ms = round((time.perf_counter() - t_feat) * 1000, 1)

        train_df = pd.read_sql_query(
            f'SELECT * FROM relbench_f1.feat_{task_name.replace("-", "_")}_train',
            psycopg2.connect(**PG),
        )
        test_df = pd.read_sql_query(
            f'SELECT * FROM relbench_f1.feat_{task_name.replace("-", "_")}_test',
            psycopg2.connect(**PG),
        )
        feature_cols = [
            c for c in train_df.columns
            if c not in ("entity_id", "asof", "target")
        ]
        Xtr, Xte, meta = encode_matrix(train_df, test_df, feature_cols)
        ytr = train_df["target"].to_numpy()
        yte = test_df["target"].to_numpy()

        task_out = {
            "task_type": str(task.task_type),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "n_features": int(Xtr.shape[1]),
            "feature_sql_ms": feat_ms,
            "feature_meta": meta,
            "models": {},
        }

        if task.task_type == TaskType.BINARY_CLASSIFICATION:
            ytr_i = ytr.astype(np.int64)
            yte_i = yte.astype(np.int64)
            store_pack(cur, "rb_pack", Xtr, ytr_i, y_kind="int")
            specs = {
                "tree": ("SELECT pgml_train_tree(x, y, 8, 2) FROM rb_pack",
                         DecisionTreeClassifier(max_depth=8, min_samples_split=2, random_state=0)),
                "forest": ("SELECT pgml_train_forest(x, y, 15, 8, 2) FROM rb_pack",
                           RandomForestClassifier(n_estimators=15, max_depth=8, min_samples_split=2, random_state=0)),
                "gbm": ("SELECT pgml_train_gbm(x, y, 30, 3, 0.1) FROM rb_pack",
                        GradientBoostingClassifier(n_estimators=30, max_depth=3, learning_rate=0.1, random_state=0)),
                "svm": ("SELECT pgml_train_svm(x, y, 0.001, 20) FROM rb_pack",
                        LogisticRegression(max_iter=2000)),
            }
            scaler = StandardScaler().fit(Xtr)
            Xtr_s = scaler.transform(Xtr).astype(np.float32)
            Xte_s = scaler.transform(Xte).astype(np.float32)
            store_pack(cur, "rb_pack_s", Xtr_s, ytr_i, y_kind="int")
            for name, (train_sql, sk) in specs.items():
                use = "rb_pack_s" if name == "svm" else "rb_pack"
                Xte_use = Xte_s if name == "svm" else Xte
                Xtr_use = Xtr_s if name == "svm" else Xtr
                ms = time_sql(cur, train_sql.replace("rb_pack", use))
                cur.execute(train_sql.replace("rb_pack", use))
                model = cur.fetchone()[0]
                pred_fn = {
                    "tree": "pgml_predict_tree_batch",
                    "forest": "pgml_predict_forest_batch",
                    "gbm": "pgml_predict_gbm_batch",
                    "svm": "pgml_predict_svm_batch",
                }[name]
                cur.execute(
                    f"SELECT {pred_fn}(%s, %s::float4[][])",
                    (model, Xte_use.tolist()),
                )
                pred = np.asarray(cur.fetchone()[0])
                sk.fit(Xtr_use, ytr_i)
                sk_pred = sk.predict(Xte_use)
                row = {
                    "pgml_train_ms": round(ms, 1),
                    "pgml_accuracy": round(float(accuracy_score(yte_i, pred)), 4),
                    "sklearn_accuracy": round(float(accuracy_score(yte_i, sk_pred)), 4),
                }
                try:
                    # NOTE: pred/sk_pred are hard class labels, not probabilities.
                    row["pgml_auroc"] = round(float(roc_auc_score(yte_i, pred)), 4)
                    row["sklearn_auroc"] = round(float(roc_auc_score(yte_i, sk_pred)), 4)
                    row["auroc_score_source"] = "hard_class_labels"
                    row["auroc_limitation"] = (
                        "AUROC computed on discrete predicted classes, not continuous scores"
                    )
                except ValueError:
                    pass
                # Official RelBench metric when labels are binary 0/1 predictions.
                try:
                    row["relbench_evaluate"] = {
                        k: float(v) for k, v in task.evaluate(pred.astype(float), test_table).items()
                    }
                    row["relbench_evaluate_input"] = "hard_class_labels"
                except Exception as exc:
                    row["relbench_evaluate_error"] = str(exc)
                task_out["models"][name] = row
                print(
                    f"  {task_name:16} {name:7} pgml acc {row['pgml_accuracy']:.3f} "
                    f"sk {row['sklearn_accuracy']:.3f}  train {ms:.0f} ms",
                    flush=True,
                )
        else:
            ytr_f = ytr.astype(np.float32)
            yte_f = yte.astype(np.float32)
            store_pack(cur, "rb_pack", Xtr, ytr_f, y_kind="float")
            specs = {
                "linreg": ("SELECT pgml_linreg(x, y) FROM rb_pack", LinearRegression()),
                "ridge": ("SELECT pgml_ridge(x, y, 1.0) FROM rb_pack", Ridge(alpha=1.0)),
                "tree": ("SELECT pgml_regression_tree(x, y, 6, 5) FROM rb_pack",
                         DecisionTreeRegressor(max_depth=6, min_samples_split=5, random_state=0)),
                "gbm": ("SELECT pgml_gbm_regression(x, y, 40, 2, 0.1) FROM rb_pack",
                        GradientBoostingRegressor(n_estimators=40, max_depth=2, learning_rate=0.1, random_state=0)),
            }
            for name, (train_sql, sk) in specs.items():
                ms = time_sql(cur, train_sql)
                cur.execute(train_sql)
                model = cur.fetchone()[0]
                pred_fn = {
                    "linreg": "pgml_predict_linreg_batch",
                    "ridge": "pgml_predict_linreg_batch",
                    "tree": "pgml_predict_regression_tree_batch",
                    "gbm": "pgml_predict_gbm_regression_batch",
                }[name]
                cur.execute(f"SELECT {pred_fn}(%s, %s::float4[][])", (model, Xte.tolist()))
                pred = np.asarray(cur.fetchone()[0], dtype=float)
                sk.fit(Xtr, ytr_f)
                sk_pred = sk.predict(Xte)
                row = {
                    "pgml_train_ms": round(ms, 1),
                    "pgml_mae": round(float(mean_absolute_error(yte_f, pred)), 4),
                    "sklearn_mae": round(float(mean_absolute_error(yte_f, sk_pred)), 4),
                    "pgml_r2": round(float(r2_score(yte_f, pred)), 4),
                    "sklearn_r2": round(float(r2_score(yte_f, sk_pred)), 4),
                }
                try:
                    row["relbench_evaluate"] = {
                        k: float(v) for k, v in task.evaluate(pred.astype(float), test_table).items()
                    }
                except Exception as exc:
                    row["relbench_evaluate_error"] = str(exc)
                task_out["models"][name] = row
                print(
                    f"  {task_name:16} {name:7} pgml MAE {row['pgml_mae']:.3f} "
                    f"sk {row['sklearn_mae']:.3f}  train {ms:.0f} ms",
                    flush=True,
                )
        out["tasks"][task_name] = task_out
    return out


# ---------------------------------------------------------------------------
# 2. TPC-H
# ---------------------------------------------------------------------------

def tpch_suite(cur):
    import duckdb
    import pandas as pd

    print("== TPC-H ==", flush=True)
    out = {"scale_factors": {}}
    for sf in (0.01, 0.1):
        print(f"  SF={sf}", flush=True)
        con = duckdb.connect()
        con.execute("INSTALL tpch; LOAD tpch;")
        con.execute(f"CALL dbgen(sf={sf});")
        schema = f"tpch_sf{str(sf).replace('.', '_')}"
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        cur.execute(f"CREATE SCHEMA {schema}")
        tables = ["customer", "orders", "lineitem", "nation", "region"]
        load_t0 = time.perf_counter()
        for t in tables:
            df = con.execute(f"SELECT * FROM {t}").fetchdf()
            pandas_to_postgres(cur, schema, t, df)
        load_s = round(time.perf_counter() - load_t0, 2)

        # Predict whether an order is in the top half of revenue.
        # Features come only from customer/nation and pre-order lineitem-free
        # attributes available at order creation time, plus order aggregates
        # that a warehouse query would compute before training.
        feat_sql = f"""
        CREATE TABLE {schema}.order_ml AS
        SELECT
          o.o_orderkey,
          o.o_orderdate,
          o.o_totalprice,
          (o.o_totalprice > (SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY o2.o_totalprice)
                             FROM {schema}.orders o2))::int AS high_value,
          c.c_acctbal,
          c.c_mktsegment,
          n.n_name AS nation,
          r.r_name AS region,
          COALESCE(agg.n_lines, 0) AS n_lines,
          COALESCE(agg.sum_qty, 0) AS sum_qty,
          COALESCE(agg.avg_price, 0) AS avg_price,
          COALESCE(agg.max_discount, 0) AS max_discount
        FROM {schema}.orders o
        JOIN {schema}.customer c ON c.c_custkey = o.o_custkey
        JOIN {schema}.nation n ON n.n_nationkey = c.c_nationkey
        JOIN {schema}.region r ON r.r_regionkey = n.n_regionkey
        LEFT JOIN (
          SELECT l_orderkey,
                 COUNT(*)::float8 AS n_lines,
                 SUM(l_quantity)::float8 AS sum_qty,
                 AVG(l_extendedprice)::float8 AS avg_price,
                 MAX(l_discount)::float8 AS max_discount
          FROM {schema}.lineitem
          GROUP BY l_orderkey
        ) agg ON agg.l_orderkey = o.o_orderkey
        """
        t0 = time.perf_counter()
        cur.execute(f"DROP TABLE IF EXISTS {schema}.order_ml")
        cur.execute(feat_sql)
        feat_ms = round((time.perf_counter() - t0) * 1000, 1)
        df = pd.read_sql_query(f"SELECT * FROM {schema}.order_ml", psycopg2.connect(**PG))
        # Temporal-ish split by order date: first 80% train, last 20% test.
        df = df.sort_values("o_orderdate").reset_index(drop=True)
        cut = int(len(df) * 0.8)
        train_df, test_df = df.iloc[:cut].copy(), df.iloc[cut:].copy()
        feature_cols = [
            "c_acctbal", "c_mktsegment", "nation", "region",
            "n_lines", "sum_qty", "avg_price", "max_discount",
        ]
        Xtr, Xte, meta = encode_matrix(train_df, test_df, feature_cols)
        ytr = train_df["high_value"].to_numpy(dtype=np.int64)
        yte = test_df["high_value"].to_numpy(dtype=np.int64)
        store_pack(cur, "tpch_pack", Xtr, ytr, y_kind="int")
        models = {}
        # Histogram paths for SF>=0.1: exact GBM on ~120k rows is too slow for a
        # systems sweep; the paper already isolates histogram vs sort separately.
        use_fast = sf >= 0.1
        model_specs = [
            ("tree",
             "SELECT pgml_train_tree_fast(x, y, 8, 2) FROM tpch_pack" if use_fast
             else "SELECT pgml_train_tree(x, y, 8, 2) FROM tpch_pack",
             "pgml_predict_tree_batch",
             DecisionTreeClassifier(max_depth=8, min_samples_split=2, random_state=0)),
            ("forest",
             "SELECT pgml_train_forest_fast(x, y, 15, 8, 2) FROM tpch_pack" if use_fast
             else "SELECT pgml_train_forest(x, y, 15, 8, 2) FROM tpch_pack",
             "pgml_predict_forest_batch",
             RandomForestClassifier(n_estimators=15, max_depth=8, min_samples_split=2, random_state=0)),
            ("gbm",
             "SELECT pgml_train_gbm_fast(x, y, 30, 3, 0.1) FROM tpch_pack" if use_fast
             else "SELECT pgml_train_gbm(x, y, 30, 3, 0.1) FROM tpch_pack",
             "pgml_predict_gbm_batch",
             GradientBoostingClassifier(n_estimators=30, max_depth=3, learning_rate=0.1, random_state=0)),
        ]
        for name, train_sql, pred_sql, sk in model_specs:
            ms = time_sql(cur, train_sql)
            cur.execute(train_sql)
            model = cur.fetchone()[0]
            cur.execute(f"SELECT {pred_sql}(%s, %s::float4[][])", (model, Xte.tolist()))
            pred = np.asarray(cur.fetchone()[0])
            sk.fit(Xtr, ytr)
            sk_pred = sk.predict(Xte)
            models[name] = {
                "pgml_train_ms": round(ms, 1),
                "pgml_accuracy": round(float(accuracy_score(yte, pred)), 4),
                "sklearn_accuracy": round(float(accuracy_score(yte, sk_pred)), 4),
            }
            print(
                f"  SF={sf} {name:7} pgml {models[name]['pgml_accuracy']:.3f} "
                f"sk {models[name]['sklearn_accuracy']:.3f}  "
                f"feat {feat_ms:.0f} ms train {ms:.0f} ms",
                flush=True,
            )
        out["scale_factors"][str(sf)] = {
            "n_orders": int(len(df)),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "n_features": int(Xtr.shape[1]),
            "load_tables_s": load_s,
            "feature_sql_ms": feat_ms,
            "feature_meta": meta,
            "pgml_histogram_path": bool(use_fast),
            "models": models,
        }
        con.close()
    return out


# ---------------------------------------------------------------------------
# 3. TabArena / BeyondArena
# ---------------------------------------------------------------------------

def tabarena_suite(cur):
    print("== TabArena / BeyondArena ==", flush=True)
    try:
        from data_foundry.collections import BEYOND_ARENA
    except Exception as exc:
        return {"available": False, "error": f"data_foundry import failed: {exc}"}

    # Small, numeric-friendly datasets; skip text-heavy ones.
    names = [
        "airfoil_self_noise",
        "concrete_compressive_strength",
        "wine_quality",
        "early_stage_diabetes_risk_prediction",
        "bank_customer_churn",
    ]
    out = {"available": True, "datasets": {}}
    for name in names:
        try:
            container = BEYOND_ARENA.get_dataset(name)
        except Exception as exc:
            out["datasets"][name] = {"error": str(exc)}
            print(f"  {name}: skip ({exc})", flush=True)
            continue
        df = container.dataset.copy()
        target = container.task_metadata.target_column_name
        problem = getattr(container.task_metadata, "problem_type", None) or getattr(
            container.task_metadata, "task_type", "unknown"
        )
        # First outer split only, for a bounded paper table.
        splits = container.experiment_metadata.splits
        repeat_id = next(iter(splits))
        fold_id, (train_idx, test_idx) = next(iter(splits[repeat_id].items()))
        train_df = df.iloc[list(train_idx)].copy()
        test_df = df.iloc[list(test_idx)].copy()
        feature_cols = [c for c in df.columns if c != target]
        # Drop pure text object columns with huge cardinality / non-scalar.
        keep = []
        for c in feature_cols:
            if df[c].dtype == object:
                sample = df[c].dropna().astype(str).head(20)
                if sample.map(lambda s: len(s) > 64).mean() > 0.5:
                    continue
            keep.append(c)
        Xtr, Xte, meta = encode_matrix(train_df, test_df, keep)
        ytr = train_df[target].to_numpy()
        yte = test_df[target].to_numpy()
        entry = {
            "problem_type": str(problem),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "n_features": int(Xtr.shape[1]),
            "split": {"repeat": str(repeat_id), "fold": str(fold_id)},
            "models": {},
        }
        is_cls = str(problem).lower() in ("binary", "multiclass", "classification") or (
            not np.issubdtype(ytr.dtype, np.floating)
        )
        # Heuristic: few unique values => classification.
        if len(np.unique(ytr)) <= 20 and not np.issubdtype(np.asarray(ytr).dtype, np.floating):
            is_cls = True
        if is_cls or (np.issubdtype(np.asarray(ytr).dtype, np.integer) and len(np.unique(ytr)) <= 20):
            # map labels to 0..K-1
            enc = LabelEncoder()
            ytr_i = enc.fit_transform(ytr.astype(str))
            yte_i = enc.transform(yte.astype(str))
            entry["executed_as"] = (
                "multiclass_classification" if len(enc.classes_) > 2 else "binary_classification"
            )
            if str(problem).lower() == "regression" and len(enc.classes_) > 2:
                entry["problem_type_metadata"] = str(problem)
                entry["problem_type"] = "multiclass_classification"
                entry["target_handling"] = (
                    "Metadata said regression; run path discretized/treated target as multiclass "
                    f"({len(enc.classes_)} classes) and reported accuracy."
                )
            if len(enc.classes_) > 2:
                entry["note"] = "multiclass; svm path skipped"
                store_pack(cur, "ta_pack", Xtr.astype(np.float32), ytr_i, y_kind="int")
                train_sql = "SELECT pgml_train_tree(x, y, 8, 2) FROM ta_pack"
                ms = time_sql(cur, train_sql)
                cur.execute(train_sql)
                model = cur.fetchone()[0]
                cur.execute("SELECT pgml_predict_tree_batch(%s, %s::float4[][])", (model, Xte.tolist()))
                pred = np.asarray(cur.fetchone()[0])
                sk = DecisionTreeClassifier(max_depth=8, min_samples_split=2, random_state=0).fit(Xtr, ytr_i)
                entry["models"]["tree"] = {
                    "pgml_train_ms": round(ms, 1),
                    "pgml_accuracy": round(float(accuracy_score(yte_i, pred)), 4),
                    "sklearn_accuracy": round(float(sk.score(Xte, yte_i)), 4),
                }
            else:
                store_pack(cur, "ta_pack", Xtr.astype(np.float32), ytr_i.astype(np.int64), y_kind="int")
                for mname, tsql, pfn, sk in (
                    ("tree", "SELECT pgml_train_tree(x, y, 8, 2) FROM ta_pack",
                     "pgml_predict_tree_batch",
                     DecisionTreeClassifier(max_depth=8, min_samples_split=2, random_state=0)),
                    ("gbm", "SELECT pgml_train_gbm(x, y, 30, 3, 0.1) FROM ta_pack",
                     "pgml_predict_gbm_batch",
                     GradientBoostingClassifier(n_estimators=30, max_depth=3, learning_rate=0.1, random_state=0)),
                ):
                    ms = time_sql(cur, tsql)
                    cur.execute(tsql)
                    model = cur.fetchone()[0]
                    cur.execute(f"SELECT {pfn}(%s, %s::float4[][])", (model, Xte.tolist()))
                    pred = np.asarray(cur.fetchone()[0])
                    sk.fit(Xtr, ytr_i)
                    entry["models"][mname] = {
                        "pgml_train_ms": round(ms, 1),
                        "pgml_accuracy": round(float(accuracy_score(yte_i, pred)), 4),
                        "sklearn_accuracy": round(float(sk.score(Xte, yte_i)), 4),
                    }
        else:
            ytr_f = np.asarray(ytr, dtype=np.float32)
            yte_f = np.asarray(yte, dtype=np.float32)
            store_pack(cur, "ta_pack", Xtr.astype(np.float32), ytr_f, y_kind="float")
            for mname, tsql, pfn, sk in (
                ("linreg", "SELECT pgml_linreg(x, y) FROM ta_pack",
                 "pgml_predict_linreg_batch", LinearRegression()),
                ("gbm", "SELECT pgml_gbm_regression(x, y, 40, 2, 0.1) FROM ta_pack",
                 "pgml_predict_gbm_regression_batch",
                 GradientBoostingRegressor(n_estimators=40, max_depth=2, learning_rate=0.1, random_state=0)),
            ):
                ms = time_sql(cur, tsql)
                cur.execute(tsql)
                model = cur.fetchone()[0]
                cur.execute(f"SELECT {pfn}(%s, %s::float4[][])", (model, Xte.tolist()))
                pred = np.asarray(cur.fetchone()[0], dtype=float)
                sk.fit(Xtr, ytr_f)
                entry["models"][mname] = {
                    "pgml_train_ms": round(ms, 1),
                    "pgml_r2": round(float(r2_score(yte_f, pred)), 4),
                    "sklearn_r2": round(float(sk.score(Xte, yte_f)), 4),
                    "pgml_mae": round(float(mean_absolute_error(yte_f, pred)), 4),
                    "sklearn_mae": round(float(mean_absolute_error(yte_f, sk.predict(Xte))), 4),
                }
        out["datasets"][name] = entry
        print(f"  {name}: {entry['models']}", flush=True)
    return out


# ---------------------------------------------------------------------------
# 4. OpenML-CC18
# ---------------------------------------------------------------------------

def openml_cc18_suite(cur, max_datasets=12, max_rows=8000, max_features=80):
    print("== OpenML-CC18 ==", flush=True)
    suite = openml.study.get_suite(99)
    out = {"suite_id": 99, "datasets": {}}
    count = 0
    for task_id in suite.tasks:
        if count >= max_datasets:
            break
        try:
            task = openml.tasks.get_task(task_id)
            dataset = task.get_dataset()
            X, y, categorical, attribute_names = dataset.get_data(
                target=dataset.default_target_attribute, dataset_format="dataframe"
            )
        except Exception as exc:
            print(f"  task {task_id}: skip load ({exc})", flush=True)
            continue
        if X.shape[0] > max_rows or X.shape[1] > max_features:
            continue
        # Use OpenML's first official split when available.
        try:
            train_idx, test_idx = task.get_train_test_split_indices(repeat=0, fold=0, sample=0)
        except Exception:
            from sklearn.model_selection import train_test_split
            idx = np.arange(len(X))
            train_idx, test_idx = train_test_split(
                idx, test_size=0.2, random_state=0,
                stratify=y if len(np.unique(y)) > 1 else None,
            )
        train_df = X.iloc[list(train_idx)].copy()
        test_df = X.iloc[list(test_idx)].copy()
        ytr_raw = np.asarray(y)[list(train_idx)]
        yte_raw = np.asarray(y)[list(test_idx)]
        feature_cols = list(train_df.columns)
        Xtr, Xte, meta = encode_matrix(train_df, test_df, feature_cols)
        enc = LabelEncoder()
        ytr = enc.fit_transform(ytr_raw.astype(str))
        yte = enc.transform(yte_raw.astype(str))
        if len(enc.classes_) > 64:
            continue
        name = dataset.name
        store_pack(cur, "cc18_pack", Xtr, ytr.astype(np.int64), y_kind="int")
        entry = {
            "openml_task_id": int(task_id),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_features": int(Xtr.shape[1]),
            "n_classes": int(len(enc.classes_)),
            "models": {},
        }
        if len(enc.classes_) == 2:
            model_list = (
                ("tree", "SELECT pgml_train_tree(x, y, 8, 2) FROM cc18_pack",
                 "pgml_predict_tree_batch",
                 DecisionTreeClassifier(max_depth=8, min_samples_split=2, random_state=0)),
                ("gbm", "SELECT pgml_train_gbm(x, y, 30, 3, 0.1) FROM cc18_pack",
                 "pgml_predict_gbm_batch",
                 GradientBoostingClassifier(n_estimators=30, max_depth=3, learning_rate=0.1, random_state=0)),
            )
        else:
            model_list = (
                ("tree", "SELECT pgml_train_tree(x, y, 8, 2) FROM cc18_pack",
                 "pgml_predict_tree_batch",
                 DecisionTreeClassifier(max_depth=8, min_samples_split=2, random_state=0)),
                ("forest", "SELECT pgml_train_forest(x, y, 15, 8, 2) FROM cc18_pack",
                 "pgml_predict_forest_batch",
                 RandomForestClassifier(n_estimators=15, max_depth=8, min_samples_split=2, random_state=0)),
            )
        for mname, tsql, pfn, sk in model_list:
            try:
                ms = time_sql(cur, tsql)
                cur.execute(tsql)
                model = cur.fetchone()[0]
                cur.execute(f"SELECT {pfn}(%s, %s::float4[][])", (model, Xte.tolist()))
                pred = np.asarray(cur.fetchone()[0])
                sk.fit(Xtr, ytr)
                entry["models"][mname] = {
                    "pgml_train_ms": round(ms, 1),
                    "pgml_accuracy": round(float(accuracy_score(yte, pred)), 4),
                    "sklearn_accuracy": round(float(sk.score(Xte, yte)), 4),
                }
            except Exception as exc:
                entry["models"][mname] = {"error": str(exc)}
        out["datasets"][name] = entry
        count += 1
        print(f"  {name}: {entry['models']}", flush=True)
    out["n_datasets_run"] = count
    return out


def main():
    conn, cur = connect()
    results = {
        "note": (
            "Public benchmarks for native pgml. RelBench features are built with "
            "SQL JOIN/LATERAL aggregates inside PostgreSQL. TPC-H comes from "
            "DuckDB dbgen and loaded into PostgreSQL before feature SQL. TabArena "
            "uses BeyondArena containers when data-foundry is installed. OpenML-CC18 "
            "uses official task split fold 0 where available."
        ),
    }
    t0 = time.perf_counter()
    results["relbench"] = relbench_suite(cur)
    results["tpch"] = tpch_suite(cur)
    results["tabarena"] = tabarena_suite(cur)
    results["openml_cc18"] = openml_cc18_suite(cur)
    results["wall_s"] = round(time.perf_counter() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(f"wrote {OUT}", flush=True)
    conn.close()


if __name__ == "__main__":
    main()
