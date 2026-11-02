#!/usr/bin/env python3
"""Regression multilibrary: pgml | sklearn | mlpack | R/caret.

Protocol: regression_multilibrary_v1
- Shared 80/20 splits (seeds 0..4), StandardScaler on train only
- Learners aligned with experiments/common/rerun.py::regression_suite hyperparameters
- Aggregate mean ± sample std (ddof=1) of held-out R^2
- Does not overwrite classification multilibrary or results/rerun.json
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import time
import traceback
from collections import defaultdict
from pathlib import Path
_REPO = Path(__file__).resolve().parents[3]

import numpy as np
import psycopg2
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(_REPO.joinpath('results', 'regression_multilibrary_v1'))
DATA = ROOT / "data"
TMP = ROOT / "tmp"
PRED_DIR = ROOT / "predictions"
_rscript = os.environ.get("RSCRIPT") or shutil.which("Rscript") or "Rscript"
RSCRIPT = Path(_rscript)
CARET_R = Path(_REPO.joinpath('experiments', 'regression_multilibrary', 'run_caret_reg.R'))
PG = dict(host="localhost", port=5433, dbname="ml_test", user="postgres", password="postgres")

DATASETS = ["diabetes", "airfoil", "concrete"]
LEARNERS = [
    "linear_regression",
    "ridge_regression",
    "regression_tree",
    "gradient_boosting",
]
IMPLS = ["pgml", "sklearn", "mlpack", "r_caret"]
SPLIT_SEEDS = [0, 1, 2, 3, 4]
MODEL_SEED = 42

# Match experiments/common/rerun.py::regression_suite
PARAMS = {
    "linear_regression": {
        "pgml_sql": "SELECT pgml_linreg(%s::float4[][], %s::float4[])",
        "pgml_pred": "SELECT pgml_predict_linreg_batch(%s, %s::float4[][])",
        "pgml_path": "exact",
        "ridge_lambda": None,
    },
    "ridge_regression": {
        "pgml_sql": "SELECT pgml_ridge(%s::float4[][], %s::float4[], 1.0)",
        "pgml_pred": "SELECT pgml_predict_linreg_batch(%s, %s::float4[][])",
        "pgml_path": "exact",
        "ridge_lambda": 1.0,
        "note": "pgml ridge penalizes intercept; sklearn Ridge does not",
    },
    "regression_tree": {
        "max_depth": 4,
        "min_samples_split": 5,
        "pgml_sql": "SELECT pgml_regression_tree(%s::float4[][], %s::float4[], 4, 5)",
        "pgml_pred": "SELECT pgml_predict_regression_tree_batch(%s, %s::float4[][])",
        "pgml_path": "exact",
    },
    "gradient_boosting": {
        "n_estimators": 40,
        "max_depth": 2,
        "learning_rate": 0.1,
        "pgml_sql": "SELECT pgml_gbm_regression(%s::float4[][], %s::float4[], 40, 2, 0.1)",
        "pgml_pred": "SELECT pgml_predict_gbm_regression_batch(%s, %s::float4[][])",
        "pgml_path": "exact",
    },
}

LEARNER_LABEL = {
    "linear_regression": "Linear regression",
    "ridge_regression": "Ridge regression",
    "regression_tree": "Regression tree",
    "gradient_boosting": "Gradient boosting",
}
DATASET_LABEL = {
    "diabetes": "Diabetes",
    "airfoil": "Airfoil Self-Noise",
    "concrete": "Concrete Compressive Strength",
}
IMPL_LABEL = {
    "pgml": "pgml",
    "sklearn": "scikit-learn",
    "mlpack": "mlpack",
    "r_caret": "R/caret",
}


def now_ms():
    return time.perf_counter() * 1000.0


def connect_pg():
    conn = psycopg2.connect(**PG)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SET statement_timeout = 0")
    return conn, cur


def load_split(dataset: str, split_seed: int):
    d = DATA / dataset / f"seed_{split_seed}"
    arr = np.load(d / "arrays.npz")
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    return {
        "X_train": arr["X_train"].astype(np.float64),
        "X_test": arr["X_test"].astype(np.float64),
        "y_train": arr["y_train"].astype(np.float64),
        "y_test": arr["y_test"].astype(np.float64),
        "train_row_ids": arr["train_row_ids"].astype(np.int64),
        "test_row_ids": arr["test_row_ids"].astype(np.int64),
        "meta": meta,
    }


def metrics_from_pred(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def base_result(dataset, learner, impl, split_seed, model_seed, split):
    return {
        "dataset": dataset,
        "learner": learner,
        "implementation": impl,
        "split_seed": split_seed,
        "model_seed": model_seed,
        "status": "ok",
        "error": None,
        "train_ids_sha256": split["meta"]["train_ids_sha256"],
        "test_ids_sha256": split["meta"]["test_ids_sha256"],
        "n_train": int(split["meta"]["n_train"]),
        "n_test": int(split["meta"]["n_test"]),
    }


def run_sklearn(split, learner, model_seed):
    Xtr, Xte = split["X_train"], split["X_test"]
    ytr = split["y_train"]
    if learner == "linear_regression":
        model = LinearRegression()
        hyper = {"fit_intercept": True}
    elif learner == "ridge_regression":
        model = Ridge(alpha=1.0, random_state=model_seed)
        hyper = {"alpha": 1.0, "fit_intercept": True, "random_state": model_seed}
    elif learner == "regression_tree":
        model = DecisionTreeRegressor(
            max_depth=4, min_samples_split=5, random_state=model_seed
        )
        hyper = {
            "max_depth": 4,
            "min_samples_split": 5,
            "random_state": model_seed,
            "criterion": "squared_error",
        }
    elif learner == "gradient_boosting":
        model = GradientBoostingRegressor(
            n_estimators=40,
            max_depth=2,
            learning_rate=0.1,
            random_state=model_seed,
        )
        hyper = {
            "n_estimators": 40,
            "max_depth": 2,
            "learning_rate": 0.1,
            "random_state": model_seed,
        }
    else:
        raise ValueError(learner)

    t1 = now_ms()
    model.fit(Xtr, ytr)
    train_ms = now_ms() - t1
    t2 = now_ms()
    pred = model.predict(Xte)
    pred_ms = now_ms() - t2
    return {
        "status": "ok",
        "y_pred": np.asarray(pred, dtype=float),
        "hyperparameters": hyper,
        "timings_ms": {"train_ms": train_ms, "predict_ms": pred_ms},
        "notes": [],
    }


def run_pgml(cur, split, learner, model_seed):
    Xtr = split["X_train"].astype(np.float32)
    Xte = split["X_test"].astype(np.float32)
    ytr = split["y_train"].astype(np.float32)
    spec = PARAMS[learner]
    t0 = now_ms()
    xtr_list = Xtr.tolist()
    ytr_list = ytr.tolist()
    xte_list = Xte.tolist()
    prep = now_ms() - t0
    t1 = now_ms()
    cur.execute(spec["pgml_sql"], (xtr_list, ytr_list))
    model = cur.fetchone()[0]
    train_ms = now_ms() - t1
    t2 = now_ms()
    cur.execute(spec["pgml_pred"], (psycopg2.Binary(model), xte_list))
    pred = np.asarray(cur.fetchone()[0], dtype=float)
    pred_ms = now_ms() - t2
    return {
        "status": "ok",
        "y_pred": pred,
        "hyperparameters": {
            k: v
            for k, v in spec.items()
            if k not in ("pgml_sql", "pgml_pred", "note")
        },
        "timings_ms": {
            "prepare_input_ms": prep,
            "train_ms": train_ms,
            "predict_ms": pred_ms,
            "pgml_path": spec["pgml_path"],
        },
        "notes": [spec.get("note")] if spec.get("note") else [],
        "seed_control": {
            "model_seed": model_seed,
            "applied": False,
            "note": "deterministic given data order",
        },
    }


def run_mlpack(split, learner, model_seed):
    import mlpack
    from mlpack import linear_regression_predict, linear_regression_train

    mlpack_ver = getattr(mlpack, "__version__", "unknown")
    Xtr = np.asfortranarray(split["X_train"].astype(np.float64))
    Xte = np.asfortranarray(split["X_test"].astype(np.float64))
    ytr = np.asfortranarray(split["y_train"].astype(np.float64))

    if learner in ("regression_tree", "gradient_boosting"):
        return {
            "status": "unsupported",
            "error": (
                f"mlpack {mlpack_ver}: no regression decision-tree / gradient-boosting "
                "API in the installed Python bindings (DecisionTree is classification-only; "
                "no GBM regressor binding)"
            ),
            "y_pred": None,
            "hyperparameters": {},
            "timings_ms": {},
            "notes": [],
        }

    lam = 0.0 if learner == "linear_regression" else 1.0
    t1 = now_ms()
    out = linear_regression_train(
        training=Xtr, training_responses=ytr, lambda_=lam
    )
    train_ms = now_ms() - t1
    t2 = now_ms()
    pred_out = linear_regression_predict(
        input_model=out["output_model"], test=Xte
    )
    pred = np.asarray(pred_out["output_predictions"], dtype=float).ravel()
    pred_ms = now_ms() - t2
    if pred.size == 0 or any(v is None for v in pred.tolist()):
        raise RuntimeError(f"mlpack returned empty/None predictions (lambda_={lam})")
    return {
        "status": "ok",
        "y_pred": pred,
        "hyperparameters": {
            "lambda_": lam,
            "api": "linear_regression_train/predict",
            "note": "lambda_=0 => OLS; lambda_=1 => Tikhonov ridge",
        },
        "timings_ms": {"train_ms": train_ms, "predict_ms": pred_ms},
        "notes": [
            "mlpack LinearRegression; intercept handling follows mlpack defaults"
        ],
        "seed_control": {"model_seed": model_seed, "applied": False},
    }


def run_caret(split, learner, model_seed, tag: str):
    TMP.mkdir(parents=True, exist_ok=True)
    train_x = TMP / f"{tag}_train_x.csv"
    train_y = TMP / f"{tag}_train_y.csv"
    test_x = TMP / f"{tag}_test_x.csv"
    test_y = TMP / f"{tag}_test_y.csv"
    out_json = TMP / f"{tag}_caret.json"
    np.savetxt(train_x, split["X_train"], delimiter=",")
    np.savetxt(train_y, split["y_train"], delimiter=",")
    np.savetxt(test_x, split["X_test"], delimiter=",")
    np.savetxt(test_y, split["y_test"], delimiter=",")
    cmd = [
        str(RSCRIPT),
        str(CARET_R),
        "--train_x",
        str(train_x),
        "--train_y",
        str(train_y),
        "--test_x",
        str(test_x),
        "--test_y",
        str(test_y),
        "--learner",
        learner,
        "--out",
        str(out_json),
        "--seed",
        str(model_seed),
        "--n_train",
        str(int(split["meta"]["n_train"])),
    ]
    t0 = now_ms()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = now_ms() - t0
    if proc.returncode != 0 and not out_json.exists():
        return {
            "status": "failed",
            "error": f"Rscript exit {proc.returncode}: {proc.stderr[-2000:]}",
            "y_pred": None,
            "hyperparameters": {},
            "timings_ms": {"adapter_wall_ms": wall},
            "notes": [proc.stdout[-1000:] if proc.stdout else ""],
        }
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    if payload.get("status") != "ok":
        return {
            "status": "failed",
            "error": payload.get("error") or "caret failed",
            "y_pred": None,
            "hyperparameters": {
                "method": payload.get("method"),
                "tune": payload.get("tune"),
            },
            "timings_ms": {"adapter_wall_ms": wall},
            "notes": [proc.stderr[-500:] if proc.stderr else ""],
        }
    return {
        "status": "ok",
        "y_pred": np.asarray(payload["y_pred"], dtype=float),
        "hyperparameters": {
            "method": payload.get("method"),
            "tune": payload.get("tune"),
            "backend_args": payload.get("backend_args"),
        },
        "timings_ms": {
            "train_ms": payload.get("train_ms"),
            "predict_ms": payload.get("predict_ms"),
            "adapter_wall_ms": wall,
        },
        "notes": [],
        "r_reported_r2": payload.get("r2"),
    }


def save_prediction(row, y_pred, y_true):
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    name = (
        f"{row['dataset']}__{row['learner']}__{row['implementation']}"
        f"__split{row['split_seed']}__model{row['model_seed']}.json"
    )
    path = PRED_DIR / name
    payload = {
        **{k: row[k] for k in (
            "dataset", "learner", "implementation", "split_seed", "model_seed", "status"
        )},
        "y_pred": None if y_pred is None else np.asarray(y_pred, dtype=float).tolist(),
        "y_true": np.asarray(y_true, dtype=float).tolist(),
        "metrics": row.get("metrics"),
        "error": row.get("error"),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def run_one(cur, dataset, learner, impl, split_seed, model_seed):
    split = load_split(dataset, split_seed)
    row = base_result(dataset, learner, impl, split_seed, model_seed, split)
    tag = f"{dataset}_{learner}_split{split_seed}_model{model_seed}"
    try:
        if impl == "sklearn":
            out = run_sklearn(split, learner, model_seed)
        elif impl == "pgml":
            out = run_pgml(cur, split, learner, model_seed)
        elif impl == "mlpack":
            out = run_mlpack(split, learner, model_seed)
        elif impl == "r_caret":
            out = run_caret(split, learner, model_seed, tag)
        else:
            raise ValueError(impl)
        row["status"] = out["status"]
        row["error"] = out.get("error")
        row["hyperparameters"] = out.get("hyperparameters")
        row["timings_ms"] = out.get("timings_ms")
        row["notes"] = out.get("notes", [])
        y_pred = out.get("y_pred")
        if row["status"] == "ok" and y_pred is not None:
            row["metrics"] = metrics_from_pred(split["y_test"], y_pred)
        else:
            row["metrics"] = None
        row["prediction_path"] = save_prediction(row, y_pred, split["y_test"])
    except Exception as e:
        row["status"] = "failed"
        row["error"] = f"{type(e).__name__}: {e}"
        row["traceback"] = traceback.format_exc()[-2000:]
        row["metrics"] = None
        row["prediction_path"] = save_prediction(row, None, split["y_test"])
    return row


def fmt_mean_std(vals, digits=4):
    vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return "N/A"
    if len(vals) == 1:
        return f"{vals[0]:.{digits}f}"
    m = float(np.mean(vals))
    s = float(np.std(vals, ddof=1))
    return f"{m:.{digits}f} $\\pm$ {s:.{digits}f}"


def aggregate_r2(runs):
    """dataset -> learner -> impl -> list of r2."""
    bucket = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    status = defaultdict(lambda: defaultdict(dict))
    for r in runs:
        key = (r["dataset"], r["learner"], r["implementation"])
        if r["status"] == "unsupported":
            status[r["dataset"]][r["learner"]][r["implementation"]] = "unsupported"
            continue
        if r["status"] != "ok" or not r.get("metrics"):
            status[r["dataset"]][r["learner"]][r["implementation"]] = "failed"
            continue
        bucket[r["dataset"]][r["learner"]][r["implementation"]].append(
            r["metrics"]["r2"]
        )
        status[r["dataset"]][r["learner"]][r["implementation"]] = "ok"
    return bucket, status


def write_latex(bucket, status):
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Regression quality on public datasets, measured by",
        r"held-out $R^2$ (higher is better). Values are reported as",
        r"mean $\pm$ standard deviation over five shared 80/20",
        r"train/test splits. Preprocessing is fitted on the training",
        r"partition only.}",
        r"\label{tab:regression_quality}",
        r"\small",
        r"\setlength{\tabcolsep}{8pt}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Dataset & Learner & \textit{pgml} & scikit-learn",
        r"        & mlpack & R/caret \\",
        r"\midrule",
    ]
    for di, dataset in enumerate(DATASETS):
        if di > 0:
            lines.append(r"\midrule")
        for li, learner in enumerate(LEARNERS):
            cells = []
            for impl in IMPLS:
                st = status[dataset][learner].get(impl)
                if st == "unsupported":
                    cells.append("N/A")
                elif st != "ok":
                    cells.append("---")
                else:
                    cells.append(fmt_mean_std(bucket[dataset][learner][impl]))
            ds_cell = DATASET_LABEL[dataset] if li == 0 else ""
            lines.append(
                f"{ds_cell} & {LEARNER_LABEL[learner]} & "
                + " & ".join(cells)
                + r" \\"
            )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ]
    text = "\n".join(lines)
    (ROOT / "r2.tex").write_text(text, encoding="utf-8")
    return text


def write_markdown(bucket, status):
    header = (
        "| Dataset | Learner | pgml | scikit-learn | mlpack | R/caret |\n"
        "|---|---|---:|---:|---:|---:|\n"
    )
    rows = []
    for dataset in DATASETS:
        for learner in LEARNERS:
            cells = []
            for impl in IMPLS:
                st = status[dataset][learner].get(impl)
                if st == "unsupported":
                    cells.append("N/A")
                elif st != "ok":
                    cells.append("---")
                else:
                    cells.append(
                        fmt_mean_std(bucket[dataset][learner][impl]).replace(
                            "$\\pm$", "±"
                        )
                    )
            rows.append(
                f"| {DATASET_LABEL[dataset]} | {LEARNER_LABEL[learner]} | "
                + " | ".join(cells)
                + " |"
            )
    text = header + "\n".join(rows) + "\n"
    (ROOT / "report.md").write_text(
        "# regression_multilibrary_v1\n\nHeld-out $R^2$ mean ± sample std (ddof=1) "
        "over five shared 80/20 splits.\n\n"
        + text
        + "\nN/A = unsupported API in that library.\n",
        encoding="utf-8",
    )
    return text


def write_summary_csv(bucket, status):
    path = ROOT / "summary.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "dataset",
                "learner",
                "implementation",
                "status",
                "r2_mean",
                "r2_std",
                "n_splits",
            ]
        )
        for dataset in DATASETS:
            for learner in LEARNERS:
                for impl in IMPLS:
                    st = status[dataset][learner].get(impl, "missing")
                    vals = bucket[dataset][learner].get(impl, [])
                    if st == "ok" and vals:
                        w.writerow(
                            [
                                dataset,
                                learner,
                                impl,
                                st,
                                float(np.mean(vals)),
                                float(np.std(vals, ddof=1)),
                                len(vals),
                            ]
                        )
                    else:
                        w.writerow([dataset, learner, impl, st, "", "", len(vals)])
    return path


def write_protocol(runs):
    catalog = json.loads((ROOT / "splits_catalog.json").read_text(encoding="utf-8"))
    protocol = {
        "version": "regression_multilibrary_v1",
        "inherits_hyperparameters_from": "experiments/common/rerun.py::regression_suite",
        "metric": "held-out R^2 (sklearn.metrics.r2_score)",
        "aggregate": "mean ± sample std (ddof=1) over split_seeds 0..4",
        "split": catalog,
        "model_seed": MODEL_SEED,
        "learners": PARAMS,
        "parameter_alignment": {
            "linear_regression": {
                "pgml": "pgml_linreg OLS normal equations",
                "sklearn": "LinearRegression()",
                "mlpack": "linear_regression_train lambda_=0",
                "r_caret": "caret method=lm",
            },
            "ridge_regression": {
                "pgml": "pgml_ridge lambda=1.0 (penalizes intercept)",
                "sklearn": "Ridge(alpha=1.0)",
                "mlpack": "linear_regression_train lambda_=1.0",
                "r_caret": "caret glmnet alpha=0, lambda=1/(2*n_train), standardize=FALSE",
                "remaining_difference": "intercept penalty and exact ridge scaling differ",
            },
            "regression_tree": {
                "pgml": "pgml_regression_tree depth=4 min_samples=5",
                "sklearn": "DecisionTreeRegressor(max_depth=4, min_samples_split=5)",
                "mlpack": "UNSUPPORTED",
                "r_caret": "caret rpart2 maxdepth=4, minsplit=5, cp=1e-8",
            },
            "gradient_boosting": {
                "pgml": "pgml_gbm_regression 40 rounds depth 2 lr 0.1",
                "sklearn": "GradientBoostingRegressor(n_estimators=40, max_depth=2, learning_rate=0.1)",
                "mlpack": "UNSUPPORTED",
                "r_caret": "caret gbm n.trees=40 interaction.depth=2 shrinkage=0.1 bag.fraction=1",
            },
        },
        "mlpack_coverage": {
            "linear_regression": "supported",
            "ridge_regression": "supported",
            "regression_tree": "unsupported",
            "gradient_boosting": "unsupported",
        },
        "n_runs": len(runs),
        "planned_cells": len(DATASETS) * len(LEARNERS) * len(IMPLS) * len(SPLIT_SEEDS),
    }
    (ROOT / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    if not (ROOT / "splits_catalog.json").exists():
        raise SystemExit("Run prepare_splits.py first")

    conn, cur = connect_pg()
    runs = []
    try:
        for dataset in DATASETS:
            for learner in LEARNERS:
                for impl in IMPLS:
                    for seed in SPLIT_SEEDS:
                        row = run_one(cur, dataset, learner, impl, seed, MODEL_SEED)
                        runs.append(row)
                        m = row.get("metrics") or {}
                        r2s = f"{m['r2']:.4f}" if m else row["status"]
                        print(
                            f"{dataset:10} {learner:20} {impl:8} "
                            f"split={seed} {r2s}",
                            flush=True,
                        )
    finally:
        cur.close()
        conn.close()

    (ROOT / "runs.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    bucket, status = aggregate_r2(runs)
    tex = write_latex(bucket, status)
    md = write_markdown(bucket, status)
    write_summary_csv(bucket, status)
    write_protocol(runs)
    print("\n=== R^2 table ===\n")
    print(md)
    print(f"\nwrote {ROOT / 'r2.tex'}")
    print(tex)


if __name__ == "__main__":
    main()
