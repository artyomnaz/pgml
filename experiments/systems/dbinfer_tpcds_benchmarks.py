#!/usr/bin/env python3
"""4DBInfer (DBInfer-Bench) and TPC-DS experiments for native pgml.

4DBInfer: load multi-table RDB into PostgreSQL, build JOIN/aggregate features
in SQL, train pgml, compare with scikit-learn.

TPC-DS: DuckDB dsdgen -> PostgreSQL, time complex analytical SQL, then an
order/sale-level prediction task on warehouse joins.

Avoids importing dbinfer_bench.__init__ (pulls broken DGL graphbolt on this host)
by loading rdb_dataset modules directly.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
import types
from pathlib import Path
_REPO = Path(__file__).resolve().parents[2]

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

OUT = Path(_REPO.joinpath('results', 'dbinfer_tpcds.json'))
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


def store_pack(cur, table, X, y):
    cur.execute(f"DROP TABLE IF EXISTS {table}")
    cur.execute(f"CREATE TABLE {table} (x float4[][], y int[])")
    cur.execute(
        f"INSERT INTO {table} VALUES (%s::float4[][], %s::int[])",
        (np.asarray(X, dtype=np.float32).tolist(), np.asarray(y, dtype=np.int64).tolist()),
    )


def encode_matrix(train_df, test_df, feature_cols):
    Xtr_parts, Xte_parts, meta = [], [], []
    for col in feature_cols:
        tr, te = train_df[col], test_df[col]
        kind = str(tr.dtype)
        numeric = kind.startswith(("float", "Float", "int", "Int", "UInt", "bool", "boolean", "Boolean"))
        if numeric:
            Xtr_parts.append(tr.astype(float).fillna(0.0).to_numpy())
            Xte_parts.append(te.astype(float).fillna(0.0).to_numpy())
            meta.append({"col": col, "kind": "numeric"})
        else:
            enc = LabelEncoder()
            trv = tr.astype(str).fillna("__NA__").to_numpy()
            tev = te.astype(str).fillna("__NA__").to_numpy()
            enc.fit(np.unique(np.concatenate([trv, tev])))
            Xtr_parts.append(enc.transform(trv).astype(float))
            Xte_parts.append(enc.transform(tev).astype(float))
            meta.append({"col": col, "kind": "categorical", "n_levels": int(len(enc.classes_))})
    return (
        np.column_stack(Xtr_parts).astype(np.float32),
        np.column_stack(Xte_parts).astype(np.float32),
        meta,
    )


def pandas_to_postgres(cur, schema, table, df, page_size=5000):
    df = df.copy()
    cols = list(df.columns)
    coldefs = []
    for c in cols:
        kind = str(df[c].dtype)
        if "datetime" in kind or "timestamp" in kind:
            coldefs.append(f'"{c}" timestamptz')
            df[c] = df[c].astype("datetime64[ns]").astype(object).where(df[c].notna(), None)
        elif kind.startswith(("Int", "UInt")) or kind in ("int64", "int32", "int16", "int8"):
            coldefs.append(f'"{c}" bigint')
            df[c] = df[c].astype(object).where(df[c].notna(), None)
        elif kind.startswith("Float") or kind in ("float64", "float32"):
            coldefs.append(f'"{c}" double precision')
            df[c] = df[c].astype(object).where(df[c].notna(), None)
        else:
            coldefs.append(f'"{c}" text')
            df[c] = df[c].astype(object).where(df[c].notna(), None)

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

    cur.execute(f'DROP TABLE IF EXISTS {schema}."{table}" CASCADE')
    cur.execute(f'CREATE TABLE {schema}."{table}" ({", ".join(coldefs)})')
    rows = [tuple(cell(v) for v in row) for row in df.itertuples(index=False, name=None)]
    if rows:
        execute_values(cur, f'INSERT INTO {schema}."{table}" VALUES %s', rows, page_size=page_size)


def load_dbinfer(name):
    root = Path(r"C:\Program Files\Python311\Lib\site-packages\dbinfer_bench")
    if "dbinfer_bench" not in sys.modules:
        pkg = types.ModuleType("dbinfer_bench")
        pkg.__path__ = [str(root)]
        sys.modules["dbinfer_bench"] = pkg
    for mod_name in ("version", "dataset_meta", "download", "rdb_dataset"):
        full = f"dbinfer_bench.{mod_name}"
        if full in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(full, root / f"{mod_name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
    from dbinfer_bench.rdb_dataset import load_rdb_data
    return load_rdb_data(name)


def dict_table_to_df(table_dict):
    return pd.DataFrame({k: v for k, v in table_dict.items()})


def task_split_to_df(split_dict):
    return pd.DataFrame({k: v for k, v in split_dict.items()})


# ---------------------------------------------------------------------------
# 4DBInfer / DBInfer-Bench
# ---------------------------------------------------------------------------

def dbinfer_suite(cur):
    print("== 4DBInfer / DBInfer-Bench ==", flush=True)
    out = {"datasets": {}}
    # seznam is compact; diginetica exercises a larger click graph.
    # diginetica.QueryResult is ~90M rows and avs is a multi-GB download —
    # too large for this host sweep. seznam covers the multi-table 4DBInfer path.
    for ds_name, task_name, max_train, max_test in (
        ("seznam", "charge", 25000, 10000),
    ):
        print(f"  dataset {ds_name} task {task_name}", flush=True)
        try:
            ds = load_dbinfer(ds_name)
        except Exception as exc:
            out["datasets"][ds_name] = {"error": str(exc)}
            print(f"    load failed: {exc}", flush=True)
            continue

        schema = f"dbinfer_{ds_name.replace('-', '_')}"
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        cur.execute(f"CREATE SCHEMA {schema}")
        load_t0 = time.perf_counter()
        for tname, tdict in ds.tables.items():
            df = dict_table_to_df(tdict)
            pandas_to_postgres(cur, schema, tname, df)
            print(f"    loaded {tname}: {len(df)} rows", flush=True)
        load_s = round(time.perf_counter() - load_t0, 2)

        task = next(t for t in ds.tasks if t.metadata.name == task_name)
        target = task.metadata.target_column
        train_df = task_split_to_df(task.train_set)
        test_df = task_split_to_df(task.test_set)
        rng = np.random.default_rng(0)
        if len(train_df) > max_train:
            train_df = train_df.iloc[rng.choice(len(train_df), size=max_train, replace=False)].copy()
        if len(test_df) > max_test:
            test_df = test_df.iloc[rng.choice(len(test_df), size=max_test, replace=False)].copy()

        pandas_to_postgres(cur, schema, f"labels_{task_name}_train", train_df)
        pandas_to_postgres(cur, schema, f"labels_{task_name}_test", test_df)

        # Dataset-specific multi-table feature SQL.
        if ds_name == "seznam":
            # Indexes make as-of history aggregates practical on ~1.5M event rows.
            cur.execute(f'CREATE INDEX IF NOT EXISTS {schema}_client_pk ON {schema}."Client"(client_id)')
            cur.execute(f'CREATE INDEX IF NOT EXISTS {schema}_dobito_cd ON {schema}."Dobito"(client_id, date)')
            cur.execute(f'CREATE INDEX IF NOT EXISTS {schema}_probe_cd ON {schema}."Probehnuto"(client_id, date)')
            cur.execute(f'ANALYZE {schema}."Client"')
            cur.execute(f'ANALYZE {schema}."Dobito"')
            cur.execute(f'ANALYZE {schema}."Probehnuto"')

            def feat_sql(label):
                # Hash-aggregate join instead of per-row LATERAL nested loops.
                return f"""
                WITH lbl AS (
                  SELECT client_id, date AS asof, "{target}" AS target
                  FROM {schema}."labels_{task_name}_{label}"
                ),
                charge_hist AS (
                  SELECT l.client_id, l.asof,
                         COUNT(*)::float8 AS n_charge,
                         SUM(d.kc_dobito)::float8 AS sum_charge,
                         AVG(d.kc_dobito)::float8 AS avg_charge
                  FROM lbl l
                  JOIN {schema}."Dobito" d
                    ON d.client_id = l.client_id AND d.date < l.asof
                  GROUP BY l.client_id, l.asof
                ),
                prepay_hist AS (
                  SELECT l.client_id, l.asof,
                         COUNT(*)::float8 AS n_prepay,
                         SUM(pr.kc_proklikano)::float8 AS sum_prepay
                  FROM lbl l
                  JOIN {schema}."Probehnuto" pr
                    ON pr.client_id = l.client_id AND pr.date < l.asof
                  GROUP BY l.client_id, l.asof
                )
                SELECT
                  l.client_id,
                  l.asof,
                  EXTRACT(EPOCH FROM l.asof)::float8 AS ts_epoch,
                  c.kraj,
                  c.obor,
                  COALESCE(h.n_charge, 0) AS hist_n_charge,
                  COALESCE(h.sum_charge, 0) AS hist_sum_charge,
                  COALESCE(h.avg_charge, 0) AS hist_avg_charge,
                  COALESCE(p.n_prepay, 0) AS hist_n_prepay,
                  COALESCE(p.sum_prepay, 0) AS hist_sum_prepay,
                  l.target
                FROM lbl l
                JOIN {schema}."Client" c ON c.client_id = l.client_id
                LEFT JOIN charge_hist h ON h.client_id = l.client_id AND h.asof = l.asof
                LEFT JOIN prepay_hist p ON p.client_id = l.client_id AND p.asof = l.asof
                """
        else:
            out["datasets"][ds_name] = {
                "task": task_name,
                "load_tables_s": load_s,
                "skipped": "no feature SQL template for this dataset in the current sweep",
            }
            continue

        cur.execute(f'DROP TABLE IF EXISTS {schema}.feat_train')
        cur.execute(f'DROP TABLE IF EXISTS {schema}.feat_test')
        t0 = time.perf_counter()
        try:
            cur.execute(f"CREATE TABLE {schema}.feat_train AS {feat_sql('train')}")
            cur.execute(f"CREATE TABLE {schema}.feat_test AS {feat_sql('test')}")
        except Exception as exc:
            out["datasets"][ds_name] = {
                "task": task_name,
                "load_tables_s": load_s,
                "feature_sql_error": str(exc),
            }
            print(f"    feature SQL failed: {exc}", flush=True)
            continue
        feat_ms = round((time.perf_counter() - t0) * 1000, 1)

        ftr = pd.read_sql_query(f"SELECT * FROM {schema}.feat_train", psycopg2.connect(**PG))
        fte = pd.read_sql_query(f"SELECT * FROM {schema}.feat_test", psycopg2.connect(**PG))
        feature_cols = [c for c in ftr.columns if c not in ("target", "client_id", "asof")]
        Xtr, Xte, meta = encode_matrix(ftr, fte, feature_cols)
        enc = LabelEncoder()
        ytr = enc.fit_transform(ftr["target"].astype(str))
        yte = enc.transform(fte["target"].astype(str))
        n_classes = len(enc.classes_)
        store_pack(cur, "dbi_pack", Xtr, ytr)

        models = {}
        specs = [
            ("tree", "SELECT pgml_train_tree(x, y, 8, 2) FROM dbi_pack",
             "pgml_predict_tree_batch",
             DecisionTreeClassifier(max_depth=8, min_samples_split=2, random_state=0)),
            ("forest", "SELECT pgml_train_forest(x, y, 15, 8, 2) FROM dbi_pack",
             "pgml_predict_forest_batch",
             RandomForestClassifier(n_estimators=15, max_depth=8, min_samples_split=2, random_state=0)),
        ]
        if n_classes == 2:
            specs.append(
                ("gbm", "SELECT pgml_train_gbm_fast(x, y, 30, 3, 0.1) FROM dbi_pack",
                 "pgml_predict_gbm_batch",
                 GradientBoostingClassifier(n_estimators=30, max_depth=3, learning_rate=0.1, random_state=0))
            )
        for mname, tsql, pfn, sk in specs:
            ms = time_sql(cur, tsql)
            cur.execute(tsql)
            model = cur.fetchone()[0]
            cur.execute(f"SELECT {pfn}(%s, %s::float4[][])", (model, Xte.tolist()))
            pred = np.asarray(cur.fetchone()[0])
            sk.fit(Xtr, ytr)
            models[mname] = {
                "pgml_train_ms": round(ms, 1),
                "pgml_accuracy": round(float(accuracy_score(yte, pred)), 4),
                "sklearn_accuracy": round(float(sk.score(Xte, yte)), 4),
            }
            print(
                f"    {mname:7} pgml {models[mname]['pgml_accuracy']:.3f} "
                f"sk {models[mname]['sklearn_accuracy']:.3f}  train {ms:.0f} ms",
                flush=True,
            )

        out["datasets"][ds_name] = {
            "task": task_name,
            "task_type": task.metadata.task_type,
            "metric": task.metadata.evaluation_metric,
            "n_classes": int(n_classes),
            "n_train_used": int(len(ftr)),
            "n_test_used": int(len(fte)),
            "n_features": int(Xtr.shape[1]),
            "load_tables_s": load_s,
            "feature_sql_ms": feat_ms,
            "feature_meta": meta,
            "models": models,
            "note": "Train/test subsampled with seed 0 when larger than caps; history aggregates use date < asof.",
        }
    return out


# ---------------------------------------------------------------------------
# TPC-DS
# ---------------------------------------------------------------------------

def tpcds_suite(cur):
    import duckdb

    print("== TPC-DS ==", flush=True)
    out = {"scale_factors": {}}
    # Representative analytical queries (warehouse-style joins/aggs).
    analytic_templates = {
        "store_sales_by_month": """
            SELECT d.d_year, d.d_moy, SUM(ss.ss_net_paid) AS revenue, COUNT(*) AS n
            FROM {s}.store_sales ss
            JOIN {s}.date_dim d ON d.d_date_sk = ss.ss_sold_date_sk
            GROUP BY d.d_year, d.d_moy
            ORDER BY 1, 2
        """,
        "customer_store_spend": """
            SELECT c.c_customer_sk, SUM(ss.ss_net_paid) AS spend, COUNT(*) AS n_sales
            FROM {s}.store_sales ss
            JOIN {s}.customer c ON c.c_customer_sk = ss.ss_customer_sk
            GROUP BY c.c_customer_sk
            HAVING SUM(ss.ss_net_paid) IS NOT NULL
        """,
        "item_class_returns_ratio": """
            SELECT i.i_class,
                   SUM(ss.ss_quantity)::float8 AS sold_qty,
                   COALESCE(SUM(sr.sr_return_quantity), 0)::float8 AS ret_qty
            FROM {s}.store_sales ss
            JOIN {s}.item i ON i.i_item_sk = ss.ss_item_sk
            LEFT JOIN {s}.store_returns sr
              ON sr.sr_item_sk = ss.ss_item_sk
             AND sr.sr_ticket_number = ss.ss_ticket_number
            GROUP BY i.i_class
        """,
    }

    for sf in (0.01, 0.1):
        print(f"  SF={sf}", flush=True)
        con = duckdb.connect()
        con.execute("INSTALL tpcds; LOAD tpcds;")
        con.execute(f"CALL dsdgen(sf={sf});")
        schema = f"tpcds_sf{str(sf).replace('.', '_')}"
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        cur.execute(f"CREATE SCHEMA {schema}")
        tables = [
            "store_sales", "store_returns", "customer", "item",
            "date_dim", "store", "customer_demographics",
        ]
        load_t0 = time.perf_counter()
        for t in tables:
            df = con.execute(f"SELECT * FROM {t}").fetchdf()
            pandas_to_postgres(cur, schema, t, df)
            print(f"    loaded {t}: {len(df)}", flush=True)
        load_s = round(time.perf_counter() - load_t0, 2)

        analytic = {}
        for qname, tmpl in analytic_templates.items():
            sql = tmpl.format(s=schema)
            t0 = time.perf_counter()
            cur.execute(f"CREATE TEMP TABLE _qdump AS {sql}")
            n = cur.rowcount if cur.rowcount is not None and cur.rowcount >= 0 else None
            cur.execute("SELECT COUNT(*) FROM _qdump")
            nrows = cur.fetchone()[0]
            cur.execute("DROP TABLE _qdump")
            analytic[qname] = {"ms": round((time.perf_counter() - t0) * 1000, 1), "n_rows": int(nrows)}
            print(f"    analytic {qname}: {analytic[qname]['ms']:.0f} ms ({nrows} rows)", flush=True)

        # ML task: predict high-value store sale using customer/item/store dims + date.
        feat_sql = f"""
        CREATE TABLE {schema}.sale_ml AS
        SELECT
          ss.ss_ticket_number,
          ss.ss_item_sk,
          ss.ss_sold_date_sk,
          (ss.ss_net_paid > (SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY s2.ss_net_paid)
                             FROM {schema}.store_sales s2
                             WHERE s2.ss_net_paid IS NOT NULL))::int AS high_value,
          COALESCE(ss.ss_quantity, 0)::float8 AS qty,
          COALESCE(ss.ss_wholesale_cost, 0)::float8 AS wholesale,
          COALESCE(ss.ss_list_price, 0)::float8 AS list_price,
          COALESCE(ss.ss_sales_price, 0)::float8 AS sales_price,
          COALESCE(ss.ss_ext_discount_amt, 0)::float8 AS discount,
          COALESCE(c.c_birth_year, 0)::float8 AS birth_year,
          c.c_preferred_cust_flag AS preferred,
          c.c_email_address IS NOT NULL AS has_email,
          i.i_category AS category,
          i.i_class AS item_class,
          i.i_brand AS brand,
          COALESCE(i.i_current_price, 0)::float8 AS item_price,
          st.s_city AS store_city,
          st.s_state AS store_state,
          d.d_moy AS month,
          d.d_qoy AS quarter,
          d.d_day_name AS day_name
        FROM {schema}.store_sales ss
        LEFT JOIN {schema}.customer c ON c.c_customer_sk = ss.ss_customer_sk
        LEFT JOIN {schema}.item i ON i.i_item_sk = ss.ss_item_sk
        LEFT JOIN {schema}.store st ON st.s_store_sk = ss.ss_store_sk
        LEFT JOIN {schema}.date_dim d ON d.d_date_sk = ss.ss_sold_date_sk
        WHERE ss.ss_net_paid IS NOT NULL
        """
        t0 = time.perf_counter()
        cur.execute(f"DROP TABLE IF EXISTS {schema}.sale_ml")
        cur.execute(feat_sql)
        feat_ms = round((time.perf_counter() - t0) * 1000, 1)

        df = pd.read_sql_query(f"SELECT * FROM {schema}.sale_ml", psycopg2.connect(**PG))
        df = df.sort_values("ss_sold_date_sk").reset_index(drop=True)
        # Cap rows for training cost at large SF while keeping temporal split.
        if len(df) > 120000:
            df = df.iloc[:120000].copy()
        cut = int(len(df) * 0.8)
        train_df, test_df = df.iloc[:cut].copy(), df.iloc[cut:].copy()
        feature_cols = [
            "qty", "wholesale", "list_price", "sales_price", "discount",
            "birth_year", "preferred", "has_email", "category", "item_class",
            "brand", "item_price", "store_city", "store_state", "month",
            "quarter", "day_name",
        ]
        Xtr, Xte, meta = encode_matrix(train_df, test_df, feature_cols)
        ytr = train_df["high_value"].to_numpy(dtype=np.int64)
        yte = test_df["high_value"].to_numpy(dtype=np.int64)
        store_pack(cur, "tpcds_pack", Xtr, ytr)
        use_fast = sf >= 0.1 or len(train_df) > 30000
        models = {}
        specs = [
            ("tree",
             "SELECT pgml_train_tree_fast(x, y, 8, 2) FROM tpcds_pack" if use_fast
             else "SELECT pgml_train_tree(x, y, 8, 2) FROM tpcds_pack",
             "pgml_predict_tree_batch",
             DecisionTreeClassifier(max_depth=8, min_samples_split=2, random_state=0)),
            ("forest",
             "SELECT pgml_train_forest_fast(x, y, 15, 8, 2) FROM tpcds_pack" if use_fast
             else "SELECT pgml_train_forest(x, y, 15, 8, 2) FROM tpcds_pack",
             "pgml_predict_forest_batch",
             RandomForestClassifier(n_estimators=15, max_depth=8, min_samples_split=2, random_state=0)),
            ("gbm",
             "SELECT pgml_train_gbm_fast(x, y, 30, 3, 0.1) FROM tpcds_pack",
             "pgml_predict_gbm_batch",
             GradientBoostingClassifier(n_estimators=30, max_depth=3, learning_rate=0.1, random_state=0)),
        ]
        for mname, tsql, pfn, sk in specs:
            ms = time_sql(cur, tsql)
            cur.execute(tsql)
            model = cur.fetchone()[0]
            cur.execute(f"SELECT {pfn}(%s, %s::float4[][])", (model, Xte.tolist()))
            pred = np.asarray(cur.fetchone()[0])
            # Same training rows as pgml (no subsample) so quality is comparable.
            sk.fit(Xtr, ytr)
            models[mname] = {
                "pgml_train_ms": round(ms, 1),
                "pgml_accuracy": round(float(accuracy_score(yte, pred)), 4),
                "sklearn_accuracy": round(float(sk.score(Xte, yte)), 4),
                "sklearn_fit_rows": int(len(Xtr)),
                "train_rows_equal": True,
            }
            print(
                f"    SF={sf} {mname:7} pgml {models[mname]['pgml_accuracy']:.3f} "
                f"sk {models[mname]['sklearn_accuracy']:.3f}  "
                f"feat {feat_ms:.0f} ms train {ms:.0f} ms",
                flush=True,
            )
        out["scale_factors"][str(sf)] = {
            "n_sales_ml": int(len(df)),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "n_features": int(Xtr.shape[1]),
            "load_tables_s": load_s,
            "feature_sql_ms": feat_ms,
            "analytic_queries": analytic,
            "pgml_histogram_path": bool(use_fast),
            "feature_meta": meta,
            "models": models,
        }
        con.close()
    return out


def main():
    conn, cur = connect()
    results = {
        "note": (
            "4DBInfer via dbinfer-bench RDB loader (DGL graph path bypassed). "
            "TPC-DS via DuckDB dsdgen, loaded into PostgreSQL. "
            "Analytic queries measure warehouse SQL cost; sale_ml is a "
            "high-value classification task on joined dimensions."
        ),
    }
    t0 = time.perf_counter()
    results["dbinfer"] = dbinfer_suite(cur)
    results["tpcds"] = tpcds_suite(cur)
    results["wall_s"] = round(time.perf_counter() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(f"wrote {OUT}", flush=True)
    conn.close()


if __name__ == "__main__":
    main()
