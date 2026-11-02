"""Measure pgml training stage breakdown for the stacked-bar figure.

Stages (end-to-end training from row-wise SQL tables):
  1) SQL materialization — ARRAY aggregation into float4[][] (+ labels)
  2) Array conversion   — SQL arrays -> C buffers (inside extension)
  3) Native computation — learner body
  4) Model serialization — bytea / result array construction

C stages are reported via NOTICE lines emitted by instrumented wrappers:
  PGML_STAGE_TIMING convert_ms=... native_ms=... serialize_ms=...
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "train_stage_breakdown"
OUT_JSON = OUT_DIR / "stages.json"
OUT_CSV = OUT_DIR / "stages.csv"

PG = dict(host="localhost", port=5433, dbname="ml_test", user="postgres", password="postgres")

N = 10000
D = 20
SEED = 0
WARMUPS = 1
REPEATS = 3

# DBSCAN is O(n^2); keep same d but smaller n so wall time stays practical.
DBSCAN_N = 4000

NOTICE_RE = re.compile(
    r"PGML_STAGE_TIMING convert_ms=([0-9.]+) native_ms=([0-9.]+) serialize_ms=([0-9.]+)"
)


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def make_data(n: int, d: int, seed: int = SEED):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    # mildly separable binary labels
    y_cls = (X[:, 0] + 0.3 * X[:, 1] + rng.normal(scale=0.5, size=n) > 0).astype(np.int32)
    y_reg = (X[:, :3].sum(axis=1) + rng.normal(scale=0.1, size=n)).astype(np.float32)
    return X, y_cls, y_reg


def load_obs(cur, table: str, X: np.ndarray, y_cls: np.ndarray, y_reg: np.ndarray):
    n, d = X.shape
    cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
    feat = ", ".join(f"x{i} float4" for i in range(d))
    cur.execute(
        f"CREATE TABLE {table} (id int PRIMARY KEY, y float4, label int, {feat})"
    )
    rows = [
        (i, float(y_reg[i]), int(y_cls[i]), *map(float, X[i]))
        for i in range(n)
    ]
    execute_values(cur, f"INSERT INTO {table} VALUES %s", rows, page_size=5000)
    cur.execute(f"ANALYZE {table}")


def sql_pack(cur, obs: str, pack: str, kind: str, d: int) -> float:
    """Build float4[][] (+ labels) inside PostgreSQL. Returns wall ms."""
    arr_elems = ", ".join(f"x{i}" for i in range(d))
    cur.execute(f"DROP TABLE IF EXISTS {pack}")
    if kind == "cls":
        cur.execute(f"CREATE TABLE {pack} (x float4[][], y int[])")
        sql = f"""
            INSERT INTO {pack}
            SELECT
              ARRAY(SELECT ARRAY[{arr_elems}]::float4[] FROM {obs} ORDER BY id),
              ARRAY(SELECT label FROM {obs} ORDER BY id)
        """
    elif kind == "reg":
        cur.execute(f"CREATE TABLE {pack} (x float4[][], y float4[])")
        sql = f"""
            INSERT INTO {pack}
            SELECT
              ARRAY(SELECT ARRAY[{arr_elems}]::float4[] FROM {obs} ORDER BY id),
              ARRAY(SELECT y FROM {obs} ORDER BY id)
        """
    else:  # unsupervised
        cur.execute(f"CREATE TABLE {pack} (x float4[][])")
        sql = f"""
            INSERT INTO {pack}
            SELECT ARRAY(SELECT ARRAY[{arr_elems}]::float4[] FROM {obs} ORDER BY id)
        """
    t0 = now_ms()
    cur.execute(sql)
    return now_ms() - t0


def parse_stage_notice(conn) -> dict:
    convert = native = serialize = None
    for notice in list(conn.notices):
        m = NOTICE_RE.search(notice)
        if m:
            convert = float(m.group(1))
            native = float(m.group(2))
            serialize = float(m.group(3))
    conn.notices.clear()
    if convert is None:
        raise RuntimeError("missing PGML_STAGE_TIMING NOTICE — rebuild instrumented pgml.so?")
    return {
        "convert_ms": convert,
        "native_ms": native,
        "serialize_ms": serialize,
    }


ALGORITHMS = [
    {
        "key": "decision_tree",
        "label": "Decision tree",
        "kind": "cls",
        "n": N,
        "sql": "SELECT pgml_train_tree(x, y, 8, 2) FROM {pack}",
        "params": {},
    },
    {
        "key": "gradient_boosting",
        "label": "Gradient boosting",
        "kind": "cls",
        "n": N,
        # Histogram GBM: exact CART boosting is minutes-scale at n=10k and
        # would dominate the panel scale; matches hist path used elsewhere.
        "sql": "SELECT pgml_train_gbm_fast(x, y, 30, 3, 0.1::float4) FROM {pack}",
        "params": {"variant": "pgml_train_gbm_fast"},
    },
    {
        "key": "linear_svm",
        "label": "Linear SVM",
        "kind": "cls",
        "n": N,
        "sql": "SELECT pgml_train_svm(x, y, 0.001::float4, 20) FROM {pack}",
        "params": {},
    },
    {
        "key": "kmeans",
        "label": "k-means",
        "kind": "unsup",
        "n": N,
        "sql": "SELECT pgml_kmeans(x, 8, 20, 0) FROM {pack}",
        "params": {},
    },
    {
        "key": "dbscan",
        "label": "DBSCAN",
        "kind": "unsup",
        "n": DBSCAN_N,
        "sql": "SELECT pgml_dbscan(x, 1.5::float4, 5) FROM {pack}",
        "params": {"note": "smaller n due to O(n^2) brute-force DBSCAN"},
    },
    {
        "key": "linear_regression",
        "label": "Linear regression",
        "kind": "reg",
        "n": N,
        "sql": "SELECT pgml_linreg(x, y) FROM {pack}",
        "params": {},
    },
    {
        "key": "pca",
        "label": "PCA",
        "kind": "unsup",
        "n": N,
        "sql": "SELECT pgml_pca_fit(x, 5) FROM {pack}",
        "params": {},
    },
]


def mean_std(xs):
    a = np.asarray(xs, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def run_algorithm(conn, cur, algo: dict) -> dict:
    n = algo["n"]
    X, y_cls, y_reg = make_data(n, D, seed=SEED)
    obs = f"tsb_obs_{algo['key']}"
    pack = f"tsb_pack_{algo['key']}"
    load_obs(cur, obs, X, y_cls, y_reg)
    conn.commit()

    # materialization repeats (independent packs)
    mat_runs = []
    for _ in range(WARMUPS + REPEATS):
        ms = sql_pack(cur, obs, pack, algo["kind"], D)
        conn.commit()
        mat_runs.append(ms)
    mat_ms = mat_runs[WARMUPS:]

    # ensure pack exists for train
    sql_pack(cur, obs, pack, algo["kind"], D)
    conn.commit()

    train_runs = []
    for i in range(WARMUPS + REPEATS):
        conn.notices.clear()
        t0 = now_ms()
        cur.execute(algo["sql"].format(pack=pack))
        cur.fetchone()
        wall_ms = now_ms() - t0
        stages = parse_stage_notice(conn)
        stages["train_wall_ms"] = wall_ms
        conn.commit()
        train_runs.append(stages)

    measured = train_runs[WARMUPS:]
    convert = [r["convert_ms"] for r in measured]
    native = [r["native_ms"] for r in measured]
    serialize = [r["serialize_ms"] for r in measured]
    wall = [r["train_wall_ms"] for r in measured]

    mat_mean, mat_std = mean_std(mat_ms)
    conv_mean, conv_std = mean_std(convert)
    nat_mean, nat_std = mean_std(native)
    ser_mean, ser_std = mean_std(serialize)
    wall_mean, wall_std = mean_std(wall)

    total = mat_mean + conv_mean + nat_mean + ser_mean
    row = {
        "algorithm": algo["key"],
        "label": algo["label"],
        "n": n,
        "d": D,
        "kind": algo["kind"],
        "params": algo["params"],
        "warmups": WARMUPS,
        "repeats": REPEATS,
        "sql_materialization_ms": {"mean": mat_mean, "std": mat_std, "runs": mat_ms},
        "array_conversion_ms": {"mean": conv_mean, "std": conv_std, "runs": convert},
        "native_computation_ms": {"mean": nat_mean, "std": nat_std, "runs": native},
        "model_serialization_ms": {"mean": ser_mean, "std": ser_std, "runs": serialize},
        "train_wall_ms": {"mean": wall_mean, "std": wall_std, "runs": wall},
        "total_ms": total,
        "total_s": total / 1000.0,
        "share_pct": {
            "sql_materialization": 100.0 * mat_mean / total,
            "array_conversion": 100.0 * conv_mean / total,
            "native_computation": 100.0 * nat_mean / total,
            "model_serialization": 100.0 * ser_mean / total,
        },
    }
    cur.execute(f"DROP TABLE IF EXISTS {obs} CASCADE")
    cur.execute(f"DROP TABLE IF EXISTS {pack} CASCADE")
    conn.commit()
    return row


def write_csv(rows):
    cols = [
        "algorithm",
        "label",
        "n",
        "d",
        "sql_materialization_ms",
        "array_conversion_ms",
        "native_computation_ms",
        "model_serialization_ms",
        "total_ms",
        "total_s",
        "pct_sql",
        "pct_convert",
        "pct_native",
        "pct_serialize",
    ]
    lines = [",".join(cols)]
    for r in rows:
        lines.append(
            ",".join(
                map(
                    str,
                    [
                        r["algorithm"],
                        r["label"].replace(",", ";"),
                        r["n"],
                        r["d"],
                        f"{r['sql_materialization_ms']['mean']:.4f}",
                        f"{r['array_conversion_ms']['mean']:.4f}",
                        f"{r['native_computation_ms']['mean']:.4f}",
                        f"{r['model_serialization_ms']['mean']:.4f}",
                        f"{r['total_ms']:.4f}",
                        f"{r['total_s']:.6f}",
                        f"{r['share_pct']['sql_materialization']:.4f}",
                        f"{r['share_pct']['array_conversion']:.4f}",
                        f"{r['share_pct']['native_computation']:.4f}",
                        f"{r['share_pct']['model_serialization']:.4f}",
                    ],
                )
            )
        )
    OUT_CSV.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = psycopg2.connect(**PG)
    conn.autocommit = False
    # Capture NOTICE messages
    conn.set_session(readonly=False)
    cur = conn.cursor()
    # Ensure client min messages includes NOTICE
    cur.execute("SET client_min_messages TO NOTICE")
    conn.commit()

    rows = []
    for algo in ALGORITHMS:
        print(f"=== {algo['label']} (n={algo['n']}, d={D}) ===", flush=True)
        row = run_algorithm(conn, cur, algo)
        rows.append(row)
        sp = row["share_pct"]
        print(
            f"  total={row['total_s']:.3f}s  "
            f"sql={sp['sql_materialization']:.1f}%  "
            f"conv={sp['array_conversion']:.1f}%  "
            f"native={sp['native_computation']:.1f}%  "
            f"ser={sp['model_serialization']:.1f}%",
            flush=True,
        )

    payload = {
        "experiment": "train_stage_breakdown",
        "n_default": N,
        "dbscan_n": DBSCAN_N,
        "d": D,
        "seed": SEED,
        "warmups": WARMUPS,
        "repeats": REPEATS,
        "timing_boundary": {
            "sql_materialization": "SQL ARRAY aggregation from row table into float4[][] (+labels)",
            "array_conversion": "deconstruct_array + copy into C buffers (NOTICE)",
            "native_computation": "learner body (NOTICE)",
            "model_serialization": "bytea / result-array construction (NOTICE)",
        },
        "points": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(rows)
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_CSV}")
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
