#!/usr/bin/env python3
"""Full SQL–ML pipeline comparison.

Systems:
  - postgres_external: PostgreSQL SQL → client fetch → NumPy → sklearn
  - duckdb_preloaded: DuckDB (data already local) → fetchnumpy → same sklearn
  - duckdb_load_from_pg: Postgres ATTACH/copy into DuckDB + same path (cold load)
  - pgml: in-database native train/predict
  - postgresml: hosted sklearn via PostgresML train()
  - madlib: in-database MADlib

Phases timed separately: sql, transfer/convert, train, predict, total.
Shared learner for external paths: sklearn DecisionTreeClassifier / LinearRegression.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
_REPO = Path(__file__).resolve().parents[2]

import duckdb
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

OUT = Path(_REPO.joinpath('results', 'sql_ml_pipeline.json'))

PG_NATIVE = dict(host="localhost", port=5433, dbname="ml_test", user="postgres", password="postgres")
PG_MADLIB = dict(host="localhost", port=5434, dbname="ml_test", user="postgres", password="postgres")
PG_PML = dict(host="localhost", port=5435, dbname="postgresml", user="postgresml", password="postgresml")

N_GRID = [5_000, 20_000, 50_000]
D = 20
SEED = 0
TREE_DEPTH = 8
TREE_MIN = 2


def connect(cfg):
    conn = psycopg2.connect(**cfg)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SET statement_timeout = '300s'")
    return conn, cur


def now_ms():
    return time.perf_counter() * 1000


def clock_sql(cur, fn):
    cur.execute("SELECT clock_timestamp()")
    t0 = cur.fetchone()[0]
    out = fn()
    cur.execute("SELECT extract(epoch FROM (clock_timestamp() - %s)) * 1000", (t0,))
    return float(cur.fetchone()[0]), out


def make_data(n, d, rng):
    X = rng.normal(size=(n, d)).astype(np.float64)
    y_reg = (2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + rng.normal(0, 0.4, n)).astype(np.float64)
    y_cls = (X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] > 0).astype(np.int32)
    return X, y_reg, y_cls


def ensure_pg_tables(cur, X, y_reg, y_cls, table="pipe_obs"):
    """Row-wise observation table + a light feature SQL view.

    Returns (n_features, pack_ms) where pack_ms is the cost of building
    pipe_pack for pgml (counted separately from post-prepare train/predict).
    """
    n, d = X.shape
    cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
    feat = ", ".join(f"x{i+1} float8" for i in range(d))
    cur.execute(f"CREATE TABLE {table} (id int PRIMARY KEY, y float8, label int, {feat})")
    rows = []
    for i in range(n):
        rows.append((i, float(y_reg[i]), int(y_cls[i]), *map(float, X[i])))
    execute_values(
        cur,
        f"INSERT INTO {table} VALUES %s",
        rows,
        page_size=5000,
    )
    cur.execute("ANALYZE " + table)
    # Feature SQL: numeric cols + one derived interaction (forces SQL work)
    cols = ", ".join(f"x{i+1}" for i in range(d))
    cur.execute(f"DROP VIEW IF EXISTS {table}_feat")
    cur.execute(
        f"""
        CREATE VIEW {table}_feat AS
        SELECT id, y, label, {cols},
               (x1 * x2) AS x_inter
        FROM {table}
        """
    )
    # client_upload pack: NumPy already holds X; time tolist+INSERT only.
    # Does NOT include reading PG rows or SQL-side array aggregation.
    # See protocol_remeasure.py for database_resident packing.
    cur.execute("DROP TABLE IF EXISTS pipe_pack")
    cur.execute("CREATE TABLE pipe_pack (x float4[][], y float4[], labels int[])")
    Xp = np.column_stack([X, X[:, 0] * X[:, 1]]).astype(np.float32)  # outside pack timer
    t_pack = now_ms()
    cur.execute(
        "INSERT INTO pipe_pack VALUES (%s::float4[][], %s::float4[], %s::int[])",
        (Xp.tolist(), y_reg.astype(np.float32).tolist(), y_cls.tolist()),
    )
    pack_ms = now_ms() - t_pack  # tolist + serialize/transfer + INSERT
    return Xp.shape[1], pack_ms


def feature_sql_select(table="pipe_obs"):
    return f"SELECT x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x_inter, y, label FROM {table}_feat"


# -------------------- paths --------------------

def run_postgres_external(cur, task: str):
    """PG SQL → fetch → numpy → sklearn."""
    phases = {}
    t0 = now_ms()
    sql = feature_sql_select()
    cur.execute(sql)
    phases["sql_ms"] = now_ms() - t0

    t1 = now_ms()
    rows = cur.fetchall()
    phases["transfer_ms"] = now_ms() - t1

    t2 = now_ms()
    arr = np.asarray(rows, dtype=np.float64)
    X = arr[:, :-2]
    y = arr[:, -2] if task == "regression" else arr[:, -1].astype(np.int32)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED)
    phases["convert_ms"] = now_ms() - t2

    if task == "regression":
        model = LinearRegression()
    else:
        model = DecisionTreeClassifier(max_depth=TREE_DEPTH, min_samples_leaf=TREE_MIN, random_state=SEED)

    t3 = now_ms()
    model.fit(Xtr, ytr)
    phases["train_ms"] = now_ms() - t3

    t4 = now_ms()
    pred = model.predict(Xte)
    phases["predict_ms"] = now_ms() - t4

    if task == "regression":
        q = {"r2": float(r2_score(yte, pred)), "primary": float(r2_score(yte, pred))}
    else:
        q = {"accuracy": float(accuracy_score(yte, pred)), "primary": float(accuracy_score(yte, pred))}

    phases["total_ms"] = sum(phases[k] for k in ("sql_ms", "transfer_ms", "convert_ms", "train_ms", "predict_ms"))
    return {
        "system": "postgres_external",
        "task": task,
        "phases": phases,
        "quality": q,
        "notes": "PostgreSQL → psycopg2 fetch → NumPy → sklearn",
    }


def duckdb_load_from_postgres(n_hint: int):
    """Cold path: ATTACH Postgres, CREATE TABLE AS SELECT features."""
    con = duckdb.connect()
    con.execute("LOAD postgres")
    attach = (
        "ATTACH 'dbname=ml_test user=postgres host=localhost port=5433 "
        "password=postgres' AS pgdb (TYPE POSTGRES, READ_ONLY)"
    )
    t0 = now_ms()
    con.execute(attach)
    attach_ms = now_ms() - t0

    t1 = now_ms()
    con.execute("DROP TABLE IF EXISTS local_feat")
    con.execute(
        """
        CREATE TABLE local_feat AS
        SELECT x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,
               (x1 * x2) AS x_inter, y, label
        FROM pgdb.pipe_obs
        """
    )
    load_ms = now_ms() - t1
    return con, {"attach_ms": attach_ms, "load_from_pg_ms": load_ms, "load_total_ms": attach_ms + load_ms}


def duckdb_preload_from_numpy(X, y_reg, y_cls):
    """Warm path: register/create table already in DuckDB (no PG round-trip)."""
    con = duckdb.connect()
    n, d = X.shape
    # build dataframe-like via duckdb values from numpy registration
    cols = {f"x{i+1}": X[:, i] for i in range(d)}
    cols["y"] = y_reg
    cols["label"] = y_cls.astype(np.int32)
    # register as relation
    con.register("raw_np", cols)
    t0 = now_ms()
    con.execute("DROP TABLE IF EXISTS local_feat")
    con.execute(
        """
        CREATE TABLE local_feat AS
        SELECT x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,
               (x1 * x2) AS x_inter, y, label
        FROM raw_np
        """
    )
    materialize_ms = now_ms() - t0
    return con, {"materialize_ms": materialize_ms}


def run_duckdb_sklearn(con, task: str, extra_phases=None, system_name="duckdb_preloaded"):
    phases = dict(extra_phases or {})
    t0 = now_ms()
    rel = con.execute(
        """
        SELECT x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,
               x_inter, y, label FROM local_feat
        """
    )
    phases["sql_ms"] = now_ms() - t0

    t1 = now_ms()
    npmap = rel.fetchnumpy()
    phases["transfer_ms"] = now_ms() - t1  # includes Arrow/NumPy conversion inside fetchnumpy

    t2 = now_ms()
    feat_keys = [f"x{i}" for i in range(1, 21)] + ["x_inter"]
    X = np.column_stack([np.asarray(npmap[k], dtype=np.float64) for k in feat_keys])
    y = np.asarray(npmap["y"] if task == "regression" else npmap["label"], dtype=np.float64)
    if task != "regression":
        y = y.astype(np.int32)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED)
    phases["convert_ms"] = now_ms() - t2

    model = (
        LinearRegression()
        if task == "regression"
        else DecisionTreeClassifier(max_depth=TREE_DEPTH, min_samples_leaf=TREE_MIN, random_state=SEED)
    )
    t3 = now_ms()
    model.fit(Xtr, ytr)
    phases["train_ms"] = now_ms() - t3
    t4 = now_ms()
    pred = model.predict(Xte)
    phases["predict_ms"] = now_ms() - t4

    if task == "regression":
        q = {"r2": float(r2_score(yte, pred)), "primary": float(r2_score(yte, pred))}
    else:
        q = {"accuracy": float(accuracy_score(yte, pred)), "primary": float(accuracy_score(yte, pred))}

    core = ("sql_ms", "transfer_ms", "convert_ms", "train_ms", "predict_ms")
    phases["pipeline_ms"] = sum(phases[k] for k in core)
    # full cycle including cold load if present
    load_keys = [k for k in ("attach_ms", "load_from_pg_ms", "load_total_ms", "materialize_ms") if k in phases]
    phases["total_ms"] = phases["pipeline_ms"] + phases.get("load_total_ms", phases.get("materialize_ms", 0.0))

    return {
        "system": system_name,
        "task": task,
        "phases": phases,
        "quality": q,
        "notes": (
            "DuckDB fetchnumpy → same sklearn learner; "
            + ("includes Postgres ATTACH+CTAS load" if "load_from_pg_ms" in phases else "data preloaded in DuckDB")
        ),
    }


def run_pgml(cur, task: str, pack_ms: float = 0.0):
    phases = {"pack_ms": float(pack_ms)}
    # feature SQL already baked into pipe_pack at load; time a recount query as SQL phase
    t0 = now_ms()
    cur.execute("SELECT count(*) FROM pipe_obs_feat")
    cur.fetchone()
    phases["sql_ms"] = now_ms() - t0  # view touch
    phases["transfer_ms"] = 0.0
    phases["convert_ms"] = 0.0

    if task == "regression":
        train_ms, model = clock_sql(
            cur, lambda: cur.execute("SELECT pgml_linreg(x, y) FROM pipe_pack") or cur.fetchone()[0]
        )
        phases["train_ms"] = train_ms
        pred_ms, pred = clock_sql(
            cur,
            lambda: cur.execute(
                "SELECT pgml_predict_linreg_batch(%s, x) FROM pipe_pack",
                (psycopg2.Binary(model),),
            )
            or cur.fetchone()[0],
        )
        phases["predict_ms"] = pred_ms
        cur.execute("SELECT y FROM pipe_pack")
        y = np.asarray(cur.fetchone()[0], dtype=np.float64)
        q = {"r2": float(r2_score(y, np.asarray(pred, dtype=np.float64))), "primary": float(r2_score(y, np.asarray(pred)))}
    else:
        train_ms, model = clock_sql(
            cur,
            lambda: cur.execute(
                "SELECT pgml_train_tree(x, labels, %s, %s) FROM pipe_pack",
                (TREE_DEPTH, TREE_MIN),
            )
            or cur.fetchone()[0],
        )
        phases["train_ms"] = train_ms
        pred_ms, pred = clock_sql(
            cur,
            lambda: cur.execute(
                "SELECT pgml_predict_tree_batch(%s, x) FROM pipe_pack",
                (psycopg2.Binary(model),),
            )
            or cur.fetchone()[0],
        )
        phases["predict_ms"] = pred_ms
        cur.execute("SELECT labels FROM pipe_pack")
        y = np.asarray(cur.fetchone()[0], dtype=np.int32)
        q = {
            "accuracy": float(accuracy_score(y, np.asarray(pred))),
            "primary": float(accuracy_score(y, np.asarray(pred))),
        }

    # Post-prepare: SQL touch + train + predict (matrix already packed).
    phases["total_after_prepare_ms"] = phases["sql_ms"] + phases["train_ms"] + phases["predict_ms"]
    phases["total_ms"] = phases["total_after_prepare_ms"]  # kept for plotters; see also e2e
    phases["total_e2e_including_pack_ms"] = phases["pack_ms"] + phases["total_after_prepare_ms"]
    return {
        "system": "pgml",
        "task": task,
        "phases": phases,
        "quality": q,
        "notes": (
            "total_after_prepare_ms excludes float4[][] packing; "
            "total_e2e_including_pack_ms adds pack_ms. "
            "Do not treat total_after_prepare_ms as full end-to-end vs export paths."
        ),
    }


def run_madlib(cur, task: str, d: int):
    phases = {}
    t0 = now_ms()
    cur.execute("SELECT count(*) FROM pipe_obs_feat")
    cur.fetchone()
    phases["sql_ms"] = now_ms() - t0
    phases["transfer_ms"] = 0.0
    phases["convert_ms"] = 0.0

    if task == "regression":
        cur.execute("DROP TABLE IF EXISTS mad_lin, mad_lin_summary CASCADE")
        train_ms, _ = clock_sql(
            cur,
            lambda: cur.execute(
                "SELECT madlib.linregr_train('pipe_obs', 'mad_lin', 'y', "
                "'ARRAY[1.0]::float8[] || ARRAY[x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,"
                "x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x1*x2]')"
            ),
        )
        phases["train_ms"] = train_ms
        t1 = now_ms()
        cur.execute("SELECT coef FROM mad_lin")
        coef = np.asarray(cur.fetchone()[0], dtype=float)
        cur.execute(
            "SELECT y, ARRAY[1.0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x1*x2] "
            "FROM pipe_obs"
        )
        ys, preds = [], []
        for y, design in cur.fetchall():
            ys.append(y)
            preds.append(float(np.dot(coef, design)))
        phases["predict_ms"] = now_ms() - t1
        q = {"r2": float(r2_score(ys, preds)), "primary": float(r2_score(ys, preds))}
    else:
        # MADlib tree on features — use tree_train if available
        cur.execute("DROP TABLE IF EXISTS mad_tree, mad_tree_summary CASCADE")
        feats = ",".join(f"x{i}" for i in range(1, d + 1))
        train_ms, _ = clock_sql(
            cur,
            lambda: cur.execute(
                f"SELECT madlib.tree_train('pipe_obs', 'mad_tree', 'id', 'label', "
                f"'{feats}', NULL, 'gini', NULL, NULL, {TREE_DEPTH}, {TREE_MIN}, 1, 10)"
            ),
        )
        phases["train_ms"] = train_ms
        cur.execute("DROP TABLE IF EXISTS mad_tree_pred")
        t1 = now_ms()
        cur.execute(
            "SELECT madlib.tree_predict('mad_tree', 'pipe_obs', 'mad_tree_pred', 'id')"
        )
        cur.execute(
            "SELECT o.label, p.estimated_prob_1 IS NOT NULL "
            "FROM pipe_obs o JOIN mad_tree_pred p ON o.id = p.id LIMIT 1"
        )
        # get predictions properly
        cur.execute(
            """
            SELECT o.label,
                   CASE WHEN p.estimated_prob_1 >= 0.5 THEN 1 ELSE 0 END
            FROM pipe_obs o
            JOIN mad_tree_pred p ON o.id = p.id
            """
        )
        try:
            rows = cur.fetchall()
            y = np.asarray([r[0] for r in rows], dtype=np.int32)
            pred = np.asarray([r[1] for r in rows], dtype=np.int32)
            phases["predict_ms"] = now_ms() - t1
            q = {"accuracy": float(accuracy_score(y, pred)), "primary": float(accuracy_score(y, pred))}
        except Exception:
            # fallback column name variants
            cur.execute("SELECT * FROM mad_tree_pred LIMIT 1")
            colnames = [d[0] for d in cur.description]
            phases["predict_ms"] = now_ms() - t1
            q = {"accuracy": None, "primary": None, "pred_columns": colnames}

    phases["total_ms"] = phases["sql_ms"] + phases["train_ms"] + phases["predict_ms"]
    return {
        "system": "madlib",
        "task": task,
        "phases": phases,
        "quality": q,
        "notes": "MADlib in-DB on Postgres 15 (:5434); data preloaded",
    }


def run_postgresml(cur, task: str):
    phases = {}
    # ensure views for PostgresML
    cur.execute("DROP VIEW IF EXISTS pipe_reg_v CASCADE")
    cur.execute("DROP VIEW IF EXISTS pipe_cls_v CASCADE")
    feats = ", ".join(f"x{i}" for i in range(1, D + 1))
    cur.execute(
        f"CREATE VIEW pipe_reg_v AS SELECT {feats}, (x1*x2) AS x_inter, y FROM pipe_obs"
    )
    cur.execute(
        f"CREATE VIEW pipe_cls_v AS SELECT {feats}, (x1*x2) AS x_inter, label FROM pipe_obs"
    )

    t0 = now_ms()
    cur.execute("SELECT count(*) FROM pipe_obs")
    cur.fetchone()
    phases["sql_ms"] = now_ms() - t0
    phases["transfer_ms"] = 0.0
    phases["convert_ms"] = 0.0

    if task == "regression":
        project = f"pipe_lin_{int(time.time()*1000)%100000}"
        sql = (
            "SELECT 1 FROM pgml.train("
            f"project_name => '{project}', task => 'regression', "
            "relation_name => 'pipe_reg_v', y_column_name => 'y', "
            "algorithm => 'linear', hyperparams => '{}'::jsonb, "
            "test_size => 0.25, test_sampling => 'random')"
        )
    else:
        project = f"pipe_tree_{int(time.time()*1000)%100000}"
        hp = json.dumps({
            "n_estimators": 1, "max_depth": TREE_DEPTH,
            "min_samples_split": TREE_MIN, "bootstrap": False, "random_state": SEED,
        })
        sql = (
            "SELECT 1 FROM pgml.train("
            f"project_name => '{project}', task => 'classification', "
            "relation_name => 'pipe_cls_v', y_column_name => 'label', "
            "algorithm => 'random_forest', "
            f"hyperparams => '{hp}'::jsonb, "
            "test_size => 0.25, test_sampling => 'random')"
        )

    train_ms, _ = clock_sql(cur, lambda: cur.execute(sql))
    phases["train_ms"] = train_ms  # includes snapshot/fit/metrics
    phases["predict_ms"] = 0.0  # folded into train metrics path
    cur.execute(
        "SELECT m.metrics::text FROM pgml.models m "
        "JOIN pgml.projects p ON p.id = m.project_id "
        "WHERE p.name = %s ORDER BY m.id DESC LIMIT 1",
        (project,),
    )
    metrics = json.loads(cur.fetchone()[0])
    if task == "regression":
        primary = metrics.get("r2") or metrics.get("r2_score")
        q = {"metrics": metrics, "primary": primary}
    else:
        primary = metrics.get("accuracy") or metrics.get("f1")
        q = {"metrics": metrics, "primary": primary}

    phases["total_ms"] = phases["sql_ms"] + phases["train_ms"]
    return {
        "system": "postgresml",
        "task": task,
        "phases": phases,
        "quality": q,
        "notes": "PostgresML pgml.train wall (snapshot+fit+metrics); predict folded into train API",
    }


def sync_obs_to(cur, X, y_reg, y_cls):
    """Load tables; return pack_ms for the pgml float4[][] materialization."""
    _nf, pack_ms = ensure_pg_tables(cur, X, y_reg, y_cls)
    return pack_ms


def main():
    print("connecting...", flush=True)
    conn_n, cur_n = connect(PG_NATIVE)
    conn_m, cur_m = connect(PG_MADLIB)
    conn_p, cur_p = connect(PG_PML)

    points = []
    cells = [(n, task) for n in N_GRID for task in ("regression", "classification")]
    print(f"cells={len(cells)} duckdb={duckdb.__version__}", flush=True)

    for i, (n, task) in enumerate(cells, 1):
        rng = np.random.default_rng(SEED + n)
        X, y_reg, y_cls = make_data(n, D, rng)
        print(f"[{i}/{len(cells)}] n={n} task={task} — loading tables", flush=True)
        pack_ms_native = sync_obs_to(cur_n, X, y_reg, y_cls)
        sync_obs_to(cur_m, X, y_reg, y_cls)
        sync_obs_to(cur_p, X, y_reg, y_cls)

        runners = []

        # 1. postgres → python
        try:
            row = run_postgres_external(cur_n, task)
            row.update({"n": n, "d": D})
            points.append(row)
            print(f"  postgres_external  total={row['phases']['total_ms']:.1f}ms Q={row['quality']['primary']}", flush=True)
        except Exception as e:
            print(f"  ! postgres_external: {e}", flush=True)
            points.append({"n": n, "d": D, "task": task, "system": "postgres_external", "error": str(e)})

        # 2. duckdb preloaded
        try:
            con, extra = duckdb_preload_from_numpy(X, y_reg, y_cls)
            row = run_duckdb_sklearn(con, task, extra, "duckdb_preloaded")
            row.update({"n": n, "d": D})
            points.append(row)
            print(f"  duckdb_preloaded   total={row['phases']['total_ms']:.1f}ms pipe={row['phases']['pipeline_ms']:.1f}ms", flush=True)
            con.close()
        except Exception as e:
            print(f"  ! duckdb_preloaded: {e}", flush=True)
            points.append({"n": n, "d": D, "task": task, "system": "duckdb_preloaded", "error": str(e),
                           "trace": traceback.format_exc()[-400:]})

        # 3. duckdb cold load from PG
        try:
            con, extra = duckdb_load_from_postgres(n)
            row = run_duckdb_sklearn(con, task, extra, "duckdb_load_from_pg")
            row.update({"n": n, "d": D})
            points.append(row)
            print(
                f"  duckdb_load_from_pg total={row['phases']['total_ms']:.1f}ms "
                f"load={row['phases']['load_total_ms']:.1f}ms pipe={row['phases']['pipeline_ms']:.1f}ms",
                flush=True,
            )
            con.close()
        except Exception as e:
            print(f"  ! duckdb_load_from_pg: {e}", flush=True)
            points.append({"n": n, "d": D, "task": task, "system": "duckdb_load_from_pg", "error": str(e),
                           "trace": traceback.format_exc()[-500:]})

        # 4. pgml
        try:
            row = run_pgml(cur_n, task, pack_ms=pack_ms_native)
            row.update({"n": n, "d": D})
            points.append(row)
            print(
                f"  pgml               after_prep={row['phases']['total_after_prepare_ms']:.1f}ms "
                f"e2e+pack={row['phases']['total_e2e_including_pack_ms']:.1f}ms "
                f"pack={row['phases']['pack_ms']:.1f}ms Q={row['quality']['primary']}",
                flush=True,
            )
        except Exception as e:
            print(f"  ! pgml: {e}", flush=True)
            points.append({"n": n, "d": D, "task": task, "system": "pgml", "error": str(e)})

        # 5. madlib
        try:
            row = run_madlib(cur_m, task, D)
            row.update({"n": n, "d": D})
            points.append(row)
            print(f"  madlib             total={row['phases']['total_ms']:.1f}ms Q={row['quality'].get('primary')}", flush=True)
        except Exception as e:
            print(f"  ! madlib: {e}", flush=True)
            try:
                conn_m.close()
            except Exception:
                pass
            time.sleep(1)
            conn_m, cur_m = connect(PG_MADLIB)
            sync_obs_to(cur_m, X, y_reg, y_cls)
            points.append({"n": n, "d": D, "task": task, "system": "madlib", "error": str(e),
                           "trace": traceback.format_exc()[-500:]})

        # 6. postgresml
        try:
            row = run_postgresml(cur_p, task)
            row.update({"n": n, "d": D})
            points.append(row)
            print(f"  postgresml         total={row['phases']['total_ms']:.1f}ms Q={row['quality'].get('primary')}", flush=True)
        except Exception as e:
            print(f"  ! postgresml: {e}", flush=True)
            points.append({"n": n, "d": D, "task": task, "system": "postgresml", "error": str(e),
                           "trace": traceback.format_exc()[-400:]})

    payload = {
        "duckdb_version": duckdb.__version__,
        "grid": {"n": N_GRID, "d": D, "tasks": ["regression", "classification"]},
        "systems": [
            "postgres_external",
            "duckdb_preloaded",
            "duckdb_load_from_pg",
            "pgml",
            "madlib",
            "postgresml",
        ],
        "phase_keys": ["sql_ms", "transfer_ms", "convert_ms", "train_ms", "predict_ms", "total_ms"],
        "notes": {
            "external_learner": "sklearn LinearRegression / DecisionTreeClassifier",
            "duckdb_preloaded": "low exchange cost control (same process NumPy)",
            "duckdb_load_from_pg": "includes ATTACH + CTAS from PostgreSQL source table",
            "in_db": "pgml / madlib / postgresml train with data already loaded",
            "pack_ms": (
                "client_upload: Xp.tolist + INSERT float4[][]; excludes NumPy stack and "
                "PG row read. database_resident scenario is in sql_ml_database_resident.json."
            ),
            "quality": (
                "Legacy points: postgres_external/duckdb use 25% hold-out; pgml scores "
                "in-sample on full pack; postgresml uses train-API metrics. Not comparable."
            ),
            "pgml_total_ms": "alias of total_after_prepare_ms (excludes pack_ms)",
        },
        "points": points,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {OUT} ({len(points)} points)", flush=True)
    conn_n.close()
    conn_m.close()
    conn_p.close()


if __name__ == "__main__":
    main()
