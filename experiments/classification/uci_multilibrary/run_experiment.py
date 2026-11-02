#!/usr/bin/env python3
"""UCI multilibrary classification: pgml | sklearn | mlpack | R/caret.

Protocol: uci_multilibrary_v2
- Shared 80/20 stratified splits (split_seed = 0..4)
- Explicit model_seed (default 42), separate from split_seed
- Seeded pgml_train_forest(..., seed)
- Aggregate mean ± sample std (ddof=1) over five splits
Does not overwrite results/uci_multilibrary_v1 or results/rerun.json.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
_REPO = Path(__file__).resolve().parents[3]

import numpy as np
import psycopg2
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(_REPO.joinpath('results', 'uci_multilibrary_v2'))
V1_ROOT = Path(_REPO.joinpath('results', 'uci_multilibrary_v1'))
DATA = ROOT / "data"
TMP = ROOT / "tmp"
PRED_DIR = ROOT / "predictions"
_rscript = os.environ.get("RSCRIPT") or shutil.which("Rscript")
RSCRIPT = Path(_rscript) if _rscript else None
CARET_R = Path(_REPO.joinpath('experiments', 'uci_multilibrary', 'run_caret.R'))
PG = dict(host="localhost", port=5433, dbname="ml_test", user="postgres", password="postgres")

DATASETS = ["wdbc", "ionosphere", "sonar"]
LEARNERS = [
    "decision_tree",
    "bagged_trees",
    "gradient_boosting",
    "adaboost",
    "linear_svm",
]
IMPLS = ["pgml", "sklearn", "mlpack", "r_caret"]
SPLIT_SEEDS = [0, 1, 2, 3, 4]
MODEL_SEED = 42

# Align with experiments/common/rerun.py / v1 where possible; bagged_trees uses true bagging
# (all features) to match pgml_train_forest, not sklearn RandomForest(sqrt).
PARAMS = {
    "decision_tree": {
        "max_depth": 8,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "pgml_sql": "SELECT pgml_train_tree(%s::float4[][], %s::int[], 8, 2)",
        "pgml_pred": "SELECT pgml_predict_tree_batch(%s, %s::float4[][])",
        "pgml_path": "exact",
        "pgml_seed_control": "none (deterministic CART)",
    },
    "bagged_trees": {
        "n_estimators": 15,
        "max_depth": 8,
        "min_samples_split": 2,
        "bootstrap_fraction": 1.0,
        "max_features": 1.0,
        "pgml_sql": "SELECT pgml_train_forest(%s::float4[][], %s::int[], 15, 8, 2, %s)",
        "pgml_pred": "SELECT pgml_predict_forest_batch(%s, %s::float4[][])",
        "pgml_path": "exact",
        "pgml_seed_control": "explicit 6-arg seed (local LCG)",
        "note": "pgml forest = bootstrap + all features (bagging). sklearn uses BaggingClassifier.",
    },
    "gradient_boosting": {
        "n_estimators": 30,
        "max_depth": 3,
        "learning_rate": 0.1,
        "pgml_sql": "SELECT pgml_train_gbm(%s::float4[][], %s::int[], 30, 3, 0.1)",
        "pgml_pred": "SELECT pgml_predict_gbm_batch(%s, %s::float4[][])",
        "pgml_path": "exact",
        "pgml_seed_control": "none (deterministic given residuals)",
    },
    "adaboost": {
        "n_estimators": 30,
        "pgml_sql": "SELECT pgml_train_adaboost(%s::float4[][], %s::int[], 30)",
        "pgml_pred": "SELECT pgml_predict_adaboost_batch(%s, %s::float4[][])",
        "pgml_path": "exact",
        "weak_learner": "decision_stump",
        "pgml_seed_control": "none (deterministic weighted stumps)",
    },
    "linear_svm": {
        "pgml_lambda": 0.001,
        "pgml_iters": 20,
        "pgml_sql": "SELECT pgml_train_svm(%s::float4[][], %s::int[], 0.001, 20)",
        "pgml_pred": "SELECT pgml_predict_svm_batch(%s, %s::float4[][])",
        "sklearn_C": 1.0,
        "pgml_path": "exact",
        "pgml_seed_control": "none (deterministic SGD over fixed order)",
    },
}

LEARNER_LABEL = {
    "decision_tree": "Decision tree",
    "bagged_trees": "Bagged trees",
    "gradient_boosting": "Gradient boosting",
    "adaboost": "AdaBoost",
    "linear_svm": "Linear SVM",
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
        "y_train": arr["y_train"].astype(np.int64),
        "y_test": arr["y_test"].astype(np.int64),
        "train_row_ids": arr["train_row_ids"].astype(np.int64),
        "test_row_ids": arr["test_row_ids"].astype(np.int64),
        "meta": meta,
    }


def metrics_from_pred(y_true, y_pred, score=None, positive_class=1):
    """Centralized metric computation. AUROC only with continuous score."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "auroc": None,
        "auroc_note": None,
    }
    if score is not None:
        score_arr = np.asarray(score, dtype=float)
        if score_arr.ndim != 1 or len(score_arr) != len(y_true):
            out["auroc_note"] = "score shape invalid; AUROC skipped"
        elif np.allclose(score_arr, score_arr.astype(int)):
            out["auroc_note"] = "score looks discrete; AUROC not computed from hard labels"
        else:
            try:
                out["auroc"] = float(roc_auc_score(y_true, score_arr))
                out["auroc_note"] = (
                    f"continuous score; positive_class={positive_class} "
                    "(sklearn roc_auc_score orientation: higher score => class 1)"
                )
            except Exception as e:
                out["auroc"] = None
                out["auroc_note"] = f"auroc failed: {e}"
    else:
        out["auroc_note"] = "score unavailable; AUROC not computed from hard labels"
    return out


def base_result(dataset, learner, impl, split_seed, model_seed, split):
    return {
        "dataset": dataset,
        "learner": learner,
        "implementation": impl,
        "split_seed": split_seed,
        "model_seed": model_seed,
        "seed": split_seed,  # backward-compatible alias
        "status": "ok",
        "error": None,
        "train_ids_sha256": split["meta"]["train_ids_sha256"],
        "test_ids_sha256": split["meta"]["test_ids_sha256"],
        "n_train": split["meta"]["n_train"],
        "n_test": split["meta"]["n_test"],
        "n_features": int(split["X_train"].shape[1]),
        "feature_names": split["meta"]["feature_names"],
        "label_encoding": split["meta"]["label_encoding"],
        "positive_class": split["meta"]["positive_class"],
        "scaler_mean": split["meta"].get("scaler_mean"),
        "scaler_scale": split["meta"].get("scaler_scale"),
        "hyperparameters": {},
        "seed_control": {},
        "timings_ms": {},
        "metrics": {},
        "test_row_ids": split["test_row_ids"].tolist(),
        "y_true": split["y_test"].tolist(),
        "y_pred": None,
        "score": None,
        "notes": [],
        "alternatives": [],
        "reused_from": None,
    }


# -------------------- implementations --------------------

def run_sklearn(split, learner, model_seed):
    Xtr, Xte = split["X_train"], split["X_test"]
    ytr, yte = split["y_train"], split["y_test"]
    t0 = now_ms()
    prep = now_ms() - t0
    hyper = {}
    seed_control = {"model_seed": model_seed, "random_state": model_seed}
    if learner == "decision_tree":
        model = DecisionTreeClassifier(
            max_depth=8, min_samples_split=2, min_samples_leaf=1, random_state=model_seed
        )
        hyper = {
            "max_depth": 8,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "criterion": "gini",
            "random_state": model_seed,
        }
    elif learner == "bagged_trees":
        base = DecisionTreeClassifier(
            max_depth=8, min_samples_split=2, min_samples_leaf=1, random_state=model_seed
        )
        model = BaggingClassifier(
            estimator=base,
            n_estimators=15,
            max_features=1.0,
            max_samples=1.0,
            bootstrap=True,
            random_state=model_seed,
            n_jobs=1,
        )
        hyper = {
            "estimator": "DecisionTreeClassifier(max_depth=8, min_samples_split=2, min_samples_leaf=1)",
            "n_estimators": 15,
            "max_features": 1.0,
            "max_samples": 1.0,
            "bootstrap": True,
            "random_state": model_seed,
            "n_jobs": 1,
            "not_RandomForest": True,
        }
    elif learner == "gradient_boosting":
        model = GradientBoostingClassifier(
            n_estimators=30, max_depth=3, learning_rate=0.1, random_state=model_seed
        )
        hyper = {
            "n_estimators": 30,
            "max_depth": 3,
            "learning_rate": 0.1,
            "random_state": model_seed,
        }
    elif learner == "adaboost":
        # SAMME (algorithm default / explicit where available)
        kwargs = dict(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=30,
            random_state=model_seed,
        )
        try:
            model = AdaBoostClassifier(**kwargs, algorithm="SAMME")
            hyper_algo = "SAMME"
        except TypeError:
            model = AdaBoostClassifier(**kwargs)
            hyper_algo = "library_default"
        hyper = {
            "estimator": "DecisionTreeClassifier(max_depth=1)",
            "n_estimators": 30,
            "algorithm": hyper_algo,
            "random_state": model_seed,
        }
    elif learner == "linear_svm":
        model = LinearSVC(C=1.0, dual=False, random_state=model_seed, max_iter=5000)
        hyper = {
            "C": 1.0,
            "dual": False,
            "max_iter": 5000,
            "random_state": model_seed,
            "solver": "liblinear (LinearSVC)",
        }
    else:
        raise ValueError(learner)

    t1 = now_ms()
    model.fit(Xtr, ytr)
    train_ms = now_ms() - t1
    t2 = now_ms()
    pred = model.predict(Xte)
    score = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(Xte)
            classes = list(model.classes_)
            if 1 in classes:
                score = proba[:, classes.index(1)]
            else:
                score = proba[:, -1]
        except Exception:
            score = None
    elif hasattr(model, "decision_function"):
        try:
            score = np.asarray(model.decision_function(Xte), dtype=float)
        except Exception:
            score = None
    pred_ms = now_ms() - t2
    return {
        "status": "ok",
        "y_pred": np.asarray(pred, dtype=np.int64),
        "score": None if score is None else np.asarray(score, dtype=float),
        "hyperparameters": hyper,
        "seed_control": seed_control,
        "timings_ms": {
            "prepare_input_ms": prep,
            "train_ms": train_ms,
            "predict_ms": pred_ms,
            "adapter_wall_ms": prep + train_ms + pred_ms,
            "includes": [
                "NumPy matrices already in process",
                "library fit + predict",
                "no subprocess",
            ],
            "threads": "n_jobs=1 where applicable",
        },
        "notes": [],
    }


def run_pgml(cur, split, learner, model_seed):
    Xtr = split["X_train"].astype(np.float32)
    Xte = split["X_test"].astype(np.float32)
    ytr = split["y_train"].astype(np.int64)
    spec = PARAMS[learner]
    t0 = now_ms()
    xtr_list = Xtr.tolist()
    ytr_list = ytr.tolist()
    xte_list = Xte.tolist()
    prep = now_ms() - t0
    t1 = now_ms()
    if learner == "bagged_trees":
        cur.execute(spec["pgml_sql"], (xtr_list, ytr_list, int(model_seed)))
        seed_control = {
            "model_seed": model_seed,
            "sql": "pgml_train_forest(..., ntrees, max_depth, min_samples, seed)",
            "rng": "local LCG (1664525/1013904223); does not call libc rand/srand",
        }
        hyper = {
            "n_estimators": 15,
            "max_depth": 8,
            "min_samples": 2,
            "bootstrap": "with replacement, size n",
            "features_per_split": "all",
            "seed": model_seed,
        }
    else:
        cur.execute(spec["pgml_sql"], (xtr_list, ytr_list))
        seed_control = {
            "model_seed": model_seed,
            "applied": False,
            "note": spec.get("pgml_seed_control", "none"),
        }
        hyper = {k: v for k, v in spec.items() if k not in ("pgml_sql", "pgml_pred", "note")}
    model = cur.fetchone()[0]
    train_ms = now_ms() - t1
    t2 = now_ms()
    cur.execute(spec["pgml_pred"], (psycopg2.Binary(model), xte_list))
    pred = np.asarray(cur.fetchone()[0], dtype=np.int64)
    pred_ms = now_ms() - t2
    return {
        "status": "ok",
        "y_pred": pred,
        "score": None,
        "hyperparameters": hyper,
        "seed_control": seed_control,
        "timings_ms": {
            "prepare_input_ms": prep,
            "train_ms": train_ms,
            "predict_ms": pred_ms,
            "adapter_wall_ms": prep + train_ms + pred_ms,
            "includes": [
                "client float32 tolist serialization",
                "SQL train API wall",
                "SQL predict + fetch",
            ],
            "excludes": ["PostgreSQL process start", "extension load"],
            "pgml_path": spec["pgml_path"],
            "not_comparable_to_pure_library_fit": True,
        },
        "notes": [f"pgml path={spec['pgml_path']}"],
    }


def run_mlpack(split, learner, model_seed):
    import mlpack
    from mlpack import Adaboost, DecisionTree, RandomForest

    Xtr = np.asfortranarray(split["X_train"].astype(np.float64))
    Xte = np.asfortranarray(split["X_test"].astype(np.float64))
    ytr = split["y_train"].astype(np.int64)
    n_features = Xtr.shape[1]
    notes = []
    alternatives = []
    mlpack_ver = getattr(mlpack, "__version__", "unknown")

    if learner == "gradient_boosting":
        return {
            "status": "unsupported",
            "error": (
                f"mlpack {mlpack_ver}: no gradient boosting / GradientBoosting classifier "
                "in the Python bindings / C++ API used here (checked Adaboost, DecisionTree, "
                "RandomForest, SoftmaxRegression, LinearSVM only)"
            ),
            "y_pred": None,
            "score": None,
            "hyperparameters": {},
            "seed_control": {"model_seed": model_seed, "applied": False},
            "timings_ms": {},
            "notes": notes,
            "alternatives": alternatives,
        }

    if learner == "bagged_trees":
        try:
            t0 = now_ms()
            rf = RandomForest(
                num_trees=15,
                maximum_depth=8,
                minimum_leaf_size=1,
                subspace_dim=n_features,
                seed=model_seed,
            )
            rf.fit(training=Xtr, labels=ytr)
            alt_pred = np.asarray(rf.predict(Xte), dtype=np.int64).ravel()
            alternatives.append({
                "name": "mlpack_random_forest_full_subspace",
                "note": (
                    "RandomForest with subspace_dim=n_features; closer to bagging but still "
                    "RF implementation — NOT used as primary bagged_trees score"
                ),
                "accuracy": float(accuracy_score(split["y_test"], alt_pred)),
                "y_pred": alt_pred.tolist(),
                "wall_ms": now_ms() - t0,
                "seed": model_seed,
            })
        except Exception as e:
            alternatives.append({"name": "mlpack_random_forest_full_subspace", "error": str(e)})
        return {
            "status": "unsupported",
            "error": (
                f"mlpack {mlpack_ver}: provides RandomForest (feature subspace sampling), "
                "not bootstrap bagging with all features as the primary algorithm; "
                "primary bagged_trees unsupported (RF alternative stored separately)"
            ),
            "y_pred": None,
            "score": None,
            "hyperparameters": {},
            "seed_control": {"model_seed": model_seed, "applied_to_alternative_rf": True},
            "timings_ms": {},
            "notes": notes,
            "alternatives": alternatives,
        }

    t_prep0 = now_ms()
    prep = now_ms() - t_prep0
    score = None
    try:
        t_train0 = now_ms()
        if learner == "decision_tree":
            model = DecisionTree(maximum_depth=8, minimum_leaf_size=1)
            model.fit(training=Xtr, labels=ytr)
            train_ms = now_ms() - t_train0
            t_pred0 = now_ms()
            pred = np.asarray(model.predict(Xte), dtype=np.int64).ravel()
            try:
                probs = np.asarray(model.predict_proba(Xte), dtype=float)
                if probs.ndim == 2 and probs.shape[1] >= 2:
                    score = probs[:, 1]
            except Exception:
                score = None
            pred_ms = now_ms() - t_pred0
            notes.append("mlpack DecisionTree class API; minimum_gain_split default")
            hyper = {"maximum_depth": 8, "minimum_leaf_size": 1}
            seed_control = {
                "model_seed": model_seed,
                "applied": False,
                "note": "DecisionTree API has no seed parameter",
            }
        elif learner == "adaboost":
            model = Adaboost(iterations=30, weak_learner="decision_stump")
            model.fit(training=Xtr, labels=ytr)
            train_ms = now_ms() - t_train0
            t_pred0 = now_ms()
            pred = np.asarray(model.predict(Xte), dtype=np.int64).ravel()
            pred_ms = now_ms() - t_pred0
            notes.append("mlpack AdaBoost.MH with decision_stump weak learner")
            hyper = {"iterations": 30, "weak_learner": "decision_stump", "variant": "AdaBoost.MH"}
            seed_control = {
                "model_seed": model_seed,
                "applied": False,
                "note": "Adaboost API has no seed in this binding",
            }
        elif learner == "linear_svm":
            out = mlpack.linear_svm(
                training=Xtr,
                labels=ytr.reshape(-1, 1),
                lambda_=0.001,
                max_iterations=5000,
                num_classes=2,
                optimizer="lbfgs",
                seed=model_seed,
                test=Xte,
            )
            pred = np.asarray(out["predictions"], dtype=np.int64).ravel()
            train_and_predict_ms = now_ms() - t_train0
            notes.append(
                "mlpack linear_svm L2 lambda_=0.001 optimizer=lbfgs; not identical to pgml SGD hinge"
            )
            return {
                "status": "ok",
                "y_pred": pred,
                "score": None,
                "hyperparameters": {
                    "lambda_": 0.001,
                    "max_iterations": 5000,
                    "optimizer": "lbfgs",
                    "seed": model_seed,
                },
                "seed_control": {"model_seed": model_seed, "applied": True},
                "timings_ms": {
                    "prepare_input_ms": prep,
                    "train_ms": None,
                    "predict_ms": None,
                    "train_and_predict_ms": train_and_predict_ms,
                    "adapter_wall_ms": prep + train_and_predict_ms,
                    "includes": ["mlpack.linear_svm training+test in one call"],
                },
                "notes": notes,
                "alternatives": alternatives,
            }
        else:
            raise ValueError(learner)
        return {
            "status": "ok",
            "y_pred": pred,
            "score": None if score is None else np.asarray(score, dtype=float),
            "hyperparameters": hyper,
            "seed_control": seed_control,
            "timings_ms": {
                "prepare_input_ms": prep,
                "train_ms": train_ms,
                "predict_ms": pred_ms,
                "adapter_wall_ms": prep + train_ms + pred_ms,
                "includes": ["mlpack class fit + predict"],
            },
            "notes": notes,
            "alternatives": alternatives,
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "y_pred": None,
            "score": None,
            "hyperparameters": {},
            "seed_control": {"model_seed": model_seed},
            "timings_ms": {},
            "notes": notes,
            "alternatives": alternatives,
            "trace": traceback.format_exc()[-500:],
        }


def run_r_caret(split, learner, split_seed, model_seed):
    if RSCRIPT is None:
        return {
            "status": "failed",
            "error": f"Rscript not found at {RSCRIPT}",
            "y_pred": None,
            "score": None,
            "hyperparameters": {},
            "seed_control": {},
            "timings_ms": {},
            "notes": [],
        }
    TMP.mkdir(parents=True, exist_ok=True)
    tag = f"{split['meta'].get('dataset','ds')}_{learner}_split{split_seed}_model{model_seed}"
    train_x = TMP / f"{tag}_train_x.csv"
    train_y = TMP / f"{tag}_train_y.csv"
    test_x = TMP / f"{tag}_test_x.csv"
    test_y = TMP / f"{tag}_test_y.csv"
    out_json = TMP / f"{tag}_caret.json"

    t0 = now_ms()
    np.savetxt(train_x, split["X_train"], delimiter=",")
    np.savetxt(test_x, split["X_test"], delimiter=",")
    np.savetxt(train_y, split["y_train"], delimiter=",", fmt="%d")
    np.savetxt(test_y, split["y_test"], delimiter=",", fmt="%d")
    write_ms = now_ms() - t0

    cmd = [
        str(RSCRIPT),
        str(CARET_R),
        "--train_x", str(train_x),
        "--train_y", str(train_y),
        "--test_x", str(test_x),
        "--test_y", str(test_y),
        "--learner", learner,
        "--out", str(out_json),
        "--seed", str(model_seed),
        "--n_features", str(split["X_train"].shape[1]),
    ]
    t1 = now_ms()
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    wall = now_ms() - t1
    if proc.returncode != 0 and not out_json.exists():
        return {
            "status": "failed",
            "error": f"Rscript exit {proc.returncode}: {proc.stderr[-500:]}",
            "y_pred": None,
            "score": None,
            "hyperparameters": {},
            "seed_control": {"model_seed": model_seed, "set.seed": model_seed},
            "timings_ms": {"adapter_wall_ms": write_ms + wall, "csv_write_ms": write_ms, "r_process_ms": wall},
            "notes": ["includes R process start + package load inside script"],
            "stdout": proc.stdout[-300:],
        }
    try:
        payload = json.loads(out_json.read_text(encoding="utf-8"))
    except Exception as e:
        return {
            "status": "failed",
            "error": f"cannot parse caret JSON: {e}; stderr={proc.stderr[-400:]}",
            "y_pred": None,
            "score": None,
            "hyperparameters": {},
            "seed_control": {"model_seed": model_seed},
            "timings_ms": {"adapter_wall_ms": write_ms + wall},
            "notes": [],
        }
    if payload.get("status") != "ok":
        return {
            "status": "failed",
            "error": payload.get("error") or "caret status != ok",
            "y_pred": None,
            "score": None,
            "hyperparameters": {},
            "seed_control": {"model_seed": model_seed, "set.seed": model_seed},
            "timings_ms": {
                "csv_write_ms": write_ms,
                "r_process_ms": wall,
                "adapter_wall_ms": write_ms + wall,
                **(payload.get("timings_ms") or {}),
            },
            "notes": ["R caret failed"],
        }
    pred = np.asarray(payload["y_pred"], dtype=np.int64)
    score = payload.get("score")
    score = None if score is None else np.asarray(score, dtype=float)
    tm = payload.get("timings_ms") or {}
    method = payload.get("method")
    hyper = {"caret_method": method, "model_seed": model_seed}
    if learner == "decision_tree":
        hyper.update({"maxdepth": 8, "minsplit": 2, "minbucket": 1, "cp": 1e-8, "package": "rpart"})
    elif learner == "bagged_trees":
        hyper.update({"nbagg": 15, "maxdepth": 8, "package": "ipred/rpart via caret treebag"})
    elif learner == "gradient_boosting":
        hyper.update({
            "n.trees": 30, "interaction.depth": 3, "shrinkage": 0.1,
            "bag.fraction": 1.0, "train.fraction": 1.0, "package": "gbm",
        })
    elif learner == "adaboost":
        hyper.update({
            "mfinal": 30, "maxdepth": 1, "coeflearn": "Breiman",
            "variant": "AdaBoost.M1", "package": "adabag",
        })
    elif learner == "linear_svm":
        hyper.update({"C": 1.0, "scaled": False, "package": "kernlab", "kernel": "vanilladot"})
    return {
        "status": "ok",
        "y_pred": pred,
        "score": score,
        "hyperparameters": hyper,
        "seed_control": {"model_seed": model_seed, "set.seed": model_seed, "split_seed_not_used_for_model": split_seed},
        "timings_ms": {
            "csv_write_ms": write_ms,
            "r_process_ms": wall,
            "prepare_input_ms": tm.get("prepare_input_ms"),
            "train_ms": tm.get("train_ms"),
            "predict_ms": tm.get("predict_ms"),
            "adapter_wall_ms": write_ms + wall,
            "includes": [
                "Python CSV write",
                "Rscript process start",
                "caret package load inside R process",
                "train + predict",
            ],
            "r_method": method,
        },
        "notes": [f"caret method={method}"],
    }


def execute_one(cur, dataset, learner, impl, split_seed, model_seed, split):
    row = base_result(dataset, learner, impl, split_seed, model_seed, split)
    try:
        if impl == "sklearn":
            out = run_sklearn(split, learner, model_seed)
        elif impl == "pgml":
            out = run_pgml(cur, split, learner, model_seed)
        elif impl == "mlpack":
            out = run_mlpack(split, learner, model_seed)
        elif impl == "r_caret":
            out = run_r_caret(split, learner, split_seed, model_seed)
        else:
            raise ValueError(impl)
    except Exception as e:
        row["status"] = "failed"
        row["error"] = str(e)
        row["trace"] = traceback.format_exc()[-600:]
        return row

    row["status"] = out.get("status", "ok")
    row["error"] = out.get("error")
    row["timings_ms"] = out.get("timings_ms") or {}
    row["notes"] = out.get("notes") or []
    row["alternatives"] = out.get("alternatives") or []
    row["hyperparameters"] = out.get("hyperparameters") or {}
    row["seed_control"] = out.get("seed_control") or {}
    if out.get("trace"):
        row["trace"] = out["trace"]

    if row["status"] == "ok" and out.get("y_pred") is not None:
        y_pred = np.asarray(out["y_pred"], dtype=np.int64)
        if len(y_pred) != len(split["y_test"]):
            row["status"] = "failed"
            row["error"] = f"pred length {len(y_pred)} != n_test {len(split['y_test'])}"
            return row
        # Label sanity: predictions must be in known label set
        known = set(int(x) for x in np.unique(np.concatenate([split["y_train"], split["y_test"]])))
        if not set(int(x) for x in np.unique(y_pred)).issubset(known):
            row["status"] = "failed"
            row["error"] = f"y_pred contains labels outside encoding {known}"
            return row
        score = out.get("score")
        row["y_pred"] = y_pred.tolist()
        row["score"] = None if score is None else np.asarray(score, dtype=float).tolist()
        row["metrics"] = metrics_from_pred(
            split["y_test"], y_pred, score, positive_class=split["meta"]["positive_class"]
        )
    elif row["status"] == "unsupported":
        row["metrics"] = {
            "accuracy": None,
            "balanced_accuracy": None,
            "macro_f1": None,
            "confusion_matrix": None,
            "auroc": None,
            "auroc_note": "unsupported",
        }
    return row


def save_prediction_file(row):
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    name = (
        f"{row['dataset']}__{row['learner']}__{row['implementation']}"
        f"__split{row['split_seed']}__model{row['model_seed']}.json"
    )
    payload = {
        "dataset": row["dataset"],
        "learner": row["learner"],
        "implementation": row["implementation"],
        "split_seed": row["split_seed"],
        "model_seed": row["model_seed"],
        "status": row["status"],
        "error": row.get("error"),
        "test_row_ids": row.get("test_row_ids"),
        "y_true": row.get("y_true"),
        "y_pred": row.get("y_pred"),
        "score": row.get("score"),
        "train_ids_sha256": row.get("train_ids_sha256"),
        "test_ids_sha256": row.get("test_ids_sha256"),
        "label_encoding": row.get("label_encoding"),
        "metrics": row.get("metrics"),
    }
    (PRED_DIR / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def smoke_test(cur, model_seed=MODEL_SEED):
    print("== smoke test: wdbc split_seed=0 all learners/impls ==", flush=True)
    split = load_split("wdbc", 0)
    split["meta"]["dataset"] = "wdbc"
    results = []
    for learner in LEARNERS:
        for impl in IMPLS:
            print(f"  smoke {learner} / {impl} ...", flush=True)
            row = execute_one(cur, "wdbc", learner, impl, 0, model_seed, split)
            results.append(row)
            acc = (row.get("metrics") or {}).get("accuracy")
            print(f"    -> {row['status']} acc={acc}", flush=True)
    return results


def run_matrix(cur, split_seeds, model_seed=MODEL_SEED):
    runs = []
    total = len(DATASETS) * len(LEARNERS) * len(IMPLS) * len(split_seeds)
    i = 0
    for split_seed in split_seeds:
        for dataset in DATASETS:
            split = load_split(dataset, split_seed)
            split["meta"]["dataset"] = dataset
            for learner in LEARNERS:
                for impl in IMPLS:
                    i += 1
                    print(
                        f"[{i}/{total}] {dataset} split={split_seed} model={model_seed} "
                        f"{learner} {impl}",
                        flush=True,
                    )
                    row = execute_one(cur, dataset, learner, impl, split_seed, model_seed, split)
                    save_prediction_file(row)
                    runs.append(row)
                    acc = (row.get("metrics") or {}).get("accuracy")
                    print(f"  -> {row['status']} acc={acc}", flush=True)
    return runs


def write_predictions_csv(runs, path: Path):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "dataset", "learner", "implementation", "split_seed", "model_seed", "row_id",
            "y_true", "y_pred", "score", "status",
        ])
        for r in runs:
            if r["status"] != "ok" or r.get("y_pred") is None:
                continue
            y_true = r["y_true"]
            row_ids = r["test_row_ids"]
            scores = r.get("score")
            for j, rid in enumerate(row_ids):
                sc = "" if scores is None else scores[j]
                w.writerow([
                    r["dataset"], r["learner"], r["implementation"],
                    r["split_seed"], r["model_seed"],
                    int(rid), int(y_true[j]), int(r["y_pred"][j]), sc, r["status"],
                ])


def write_summary(runs, path: Path):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "dataset", "learner", "implementation", "split_seed", "model_seed", "status",
            "accuracy", "balanced_accuracy", "macro_f1", "auroc", "error",
            "train_ids_sha256", "test_ids_sha256", "reused_from",
        ])
        for r in runs:
            m = r.get("metrics") or {}
            w.writerow([
                r["dataset"], r["learner"], r["implementation"],
                r["split_seed"], r["model_seed"], r["status"],
                m.get("accuracy"), m.get("balanced_accuracy"), m.get("macro_f1"), m.get("auroc"),
                r.get("error"),
                r.get("train_ids_sha256"), r.get("test_ids_sha256"), r.get("reused_from"),
            ])


def aggregate_metric(runs, metric_name):
    """Return dict[(dataset, learner, impl)] -> cell info for mean±std over splits."""
    groups = defaultdict(list)
    meta = {}
    for r in runs:
        key = (r["dataset"], r["learner"], r["implementation"])
        groups[key].append(r)
        meta[key] = r
    out = {}
    for key, rows in groups.items():
        by_split = {r["split_seed"]: r for r in rows}
        missing = [s for s in SPLIT_SEEDS if s not in by_split]
        present = [by_split[s] for s in SPLIT_SEEDS if s in by_split]
        unsupported = [r for r in present if r["status"] == "unsupported"]
        failed = [r for r in present if r["status"] == "failed"]
        ok = [r for r in present if r["status"] == "ok"]
        if unsupported and len(unsupported) == len(present) and not missing:
            out[key] = {
                "kind": "unsupported",
                "text": "N/A",
                "n": len(unsupported),
                "reason": unsupported[0].get("error"),
            }
        elif len(ok) == len(SPLIT_SEEDS) and not missing:
            vals = [float(r["metrics"][metric_name]) for r in ok]
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1))
            out[key] = {
                "kind": "ok",
                "text": f"{mean:.4f} ± {std:.4f}",
                "tex": f"{mean:.4f} $\\pm$ {std:.4f}",
                "mean": mean,
                "std": std,
                "n": 5,
                "values": vals,
            }
        else:
            reasons = []
            if missing:
                reasons.append(f"missing splits {missing}")
            if failed:
                reasons.append(
                    "failed: " + "; ".join(
                        f"split{r['split_seed']}:{r.get('error')}" for r in failed[:3]
                    )
                )
            if unsupported and ok:
                reasons.append(f"mixed unsupported n={len(unsupported)}")
            out[key] = {
                "kind": "incomplete",
                "text": f"n={len(ok)}/5",
                "n": len(ok),
                "reason": "; ".join(reasons) if reasons else "incomplete",
            }
    return out


def write_metric_tex(runs, metric_name, path: Path, caption: str):
    agg = aggregate_metric(runs, metric_name)
    lines = [
        r"% Written by experiments/classification/uci_multilibrary/run_experiment.py — do not edit by hand",
        r"\begin{table*}",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{tab:uci-multilib-{metric_name.replace('_', '-')}}}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Dataset & Learner & pgml & scikit-learn & mlpack & R/caret \\",
        r"\midrule",
    ]
    for ds in DATASETS:
        first = True
        for learner in LEARNERS:
            ds_cell = ds if first else ""
            first = False
            cells = []
            for impl in IMPLS:
                info = agg.get((ds, learner, impl))
                if info is None:
                    cells.append("---")
                elif info["kind"] == "unsupported":
                    cells.append("N/A")
                elif info["kind"] == "ok":
                    cells.append(info.get("tex") or info["text"])
                else:
                    cells.append(info["text"])
            lines.append(
                f"{ds_cell} & {LEARNER_LABEL[learner]} & "
                + " & ".join(cells)
                + r" \\"
            )
        lines.append(r"\midrule")
    # replace last midrule with bottomrule
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table*}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return agg


def write_markdown_table(agg, metric_name):
    lines = [
        f"| Dataset | Learner | pgml | scikit-learn | mlpack | R/caret |",
        f"|---|---|---:|---:|---:|---:|",
    ]
    for ds in DATASETS:
        for learner in LEARNERS:
            cells = []
            for impl in IMPLS:
                info = agg.get((ds, learner, impl))
                if info is None:
                    cells.append("---")
                elif info["kind"] == "unsupported":
                    cells.append("N/A")
                else:
                    cells.append(info["text"])
            lines.append(
                f"| {ds} | {LEARNER_LABEL[learner]} | " + " | ".join(cells) + " |"
            )
    return "\n".join(lines)


def validate(runs):
    issues = []
    # Recompute metrics from stored y_true/y_pred (central check)
    for r in runs:
        if r["status"] != "ok":
            continue
        key = (r["dataset"], r["learner"], r["implementation"], r["split_seed"])
        y_true = np.asarray(r["y_true"], dtype=np.int64)
        y_pred = np.asarray(r["y_pred"], dtype=np.int64)
        row_ids = np.asarray(r["test_row_ids"], dtype=np.int64)
        split = load_split(r["dataset"], r["split_seed"])
        if len(y_pred) != len(split["y_test"]):
            issues.append(f"{key}: pred length mismatch")
            continue
        if not np.array_equal(row_ids, split["test_row_ids"]):
            issues.append(f"{key}: test_row_ids order mismatch")
        if not np.array_equal(y_true, split["y_test"]):
            issues.append(f"{key}: y_true mismatch vs split")
        recomputed = metrics_from_pred(y_true, y_pred, r.get("score"), r["positive_class"])
        for mname in ("accuracy", "balanced_accuracy", "macro_f1"):
            a, b = r["metrics"][mname], recomputed[mname]
            if abs(a - b) > 1e-9:
                issues.append(f"{key}: {mname} mismatch {a} vs {b}")
        tr = set(split["train_row_ids"].tolist())
        te = set(split["test_row_ids"].tolist())
        if tr & te:
            issues.append(f"{key}: train/test overlap")
        if r["train_ids_sha256"] != split["meta"]["train_ids_sha256"]:
            issues.append(f"{key}: train hash drift")
        if r["test_ids_sha256"] != split["meta"]["test_ids_sha256"]:
            issues.append(f"{key}: test hash drift")
        if r["feature_names"] != split["meta"]["feature_names"]:
            issues.append(f"{key}: feature name order mismatch")
        if r["label_encoding"] != split["meta"]["label_encoding"]:
            issues.append(f"{key}: label encoding mismatch")
        # prediction file exists
        pname = (
            f"{r['dataset']}__{r['learner']}__{r['implementation']}"
            f"__split{r['split_seed']}__model{r['model_seed']}.json"
        )
        if not (PRED_DIR / pname).exists():
            issues.append(f"{key}: missing prediction file {pname}")
    return issues


def check_forest_reproducibility(cur, model_seed=MODEL_SEED):
    """Same data+seed twice; new connection; after PG restart (caller may restart)."""
    split = load_split("wdbc", 0)
    Xtr = split["X_train"].astype(np.float32)
    Xte = split["X_test"].astype(np.float32)
    ytr = split["y_train"].astype(np.int64)

    def one(c):
        c.execute(
            "SELECT pgml_train_forest(%s::float4[][], %s::int[], 15, 8, 2, %s)",
            (Xtr.tolist(), ytr.tolist(), int(model_seed)),
        )
        model = c.fetchone()[0]
        c.execute(
            "SELECT pgml_predict_forest_batch(%s, %s::float4[][])",
            (psycopg2.Binary(model), Xte.tolist()),
        )
        pred = np.asarray(c.fetchone()[0], dtype=np.int64)
        return bytes(model), pred

    m1, p1 = one(cur)
    m2, p2 = one(cur)
    same = {
        "pred_equal": bool(np.array_equal(p1, p2)),
        "model_bytes_equal": m1 == m2,
        "note": "serialization is ordered tree dump; byte identity expected when training is deterministic",
    }

    conn2, cur2 = connect_pg()
    m3, p3 = one(cur2)
    new_conn = {
        "pred_equal_to_first": bool(np.array_equal(p1, p3)),
        "model_bytes_equal_to_first": m1 == m3,
    }
    conn2.close()

    # Restart PostgreSQL and recheck
    restart_info = {"attempted": True}
    try:
        subprocess.check_call(["docker", "restart", "pgml"])
        for _ in range(60):
            try:
                conn3, cur3 = connect_pg()
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("PostgreSQL did not become ready after restart")
        m4, p4 = one(cur3)
        restart_info.update({
            "pred_equal_to_first": bool(np.array_equal(p1, p4)),
            "model_bytes_equal_to_first": m1 == m4,
        })
        conn3.close()
    except Exception as e:
        restart_info["error"] = str(e)

    out = {
        "dataset": "wdbc",
        "split_seed": 0,
        "model_seed": model_seed,
        "same_connection": same,
        "new_connection": new_conn,
        "after_pg_restart": restart_info,
        "passed": bool(
            same["pred_equal"]
            and new_conn["pred_equal_to_first"]
            and restart_info.get("pred_equal_to_first") is True
        ),
    }
    return out


def collect_environment():
    import platform
    import sklearn

    env = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "threads_env": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "libraries": {},
        "r": {},
        "postgresql": {},
        "pgml_extension": {},
        "model_seed": MODEL_SEED,
        "split_seeds": SPLIT_SEEDS,
    }
    try:
        import psutil
        env["ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:
        env["ram_gb"] = None
    env["libraries"]["numpy"] = np.__version__
    env["libraries"]["sklearn"] = sklearn.__version__
    try:
        import mlpack
        env["libraries"]["mlpack"] = getattr(mlpack, "__version__", "4.x")
    except Exception as e:
        env["libraries"]["mlpack"] = f"unavailable: {e}"
    env["libraries"]["psycopg2"] = psycopg2.__version__
    if RSCRIPT is not None:
        try:
            out = subprocess.check_output(
                [
                    str(RSCRIPT), "-e",
                    "cat(R.version.string); cat('|'); "
                    "cat(paste(sapply(c('caret','rpart','ipred','gbm','adabag','e1071','kernlab'), "
                    "function(p) paste(p, packageVersion(p))), collapse='; '))",
                ],
                text=True, stderr=subprocess.STDOUT,
            )
            env["r"]["rscript"] = str(RSCRIPT)
            env["r"]["versions"] = out.strip()
        except Exception as e:
            env["r"]["error"] = str(e)
    else:
        env["r"]["error"] = "Rscript missing"
    try:
        conn, cur = connect_pg()
        cur.execute("SELECT version()")
        env["postgresql"]["native"] = cur.fetchone()[0]
        cur.execute("SELECT extversion FROM pg_extension WHERE extname='pgml'")
        row = cur.fetchone()
        env["pgml_extension"]["extversion"] = row[0] if row else None
        try:
            h = subprocess.check_output(
                ["docker", "exec", "pgml", "bash", "-lc",
                 "sha256sum $(pg_config --pkglibdir)/pgml.so"],
                text=True, stderr=subprocess.STDOUT,
            ).strip()
            env["pgml_extension"]["binary_sha256"] = h.split()[0] if h else None
            env["pgml_extension"]["binary_sha256_raw"] = h
        except Exception as e:
            env["pgml_extension"]["binary_sha256_error"] = str(e)
        # Confirm seeded overload exists
        cur.execute(
            "SELECT pronargs FROM pg_proc WHERE proname='pgml_train_forest' ORDER BY pronargs"
        )
        env["pgml_extension"]["pgml_train_forest_arities"] = [r[0] for r in cur.fetchall()]
        conn.close()
    except Exception as e:
        env["postgresql"]["error"] = str(e)
    # Source stamps (no git repository in this workspace)
    try:
        src = Path(_REPO.joinpath('src', 'c', 'pgml.c')).read_bytes()
        env["pgml_extension"]["pgml_c_sha256"] = hashlib.sha256(src).hexdigest()
    except Exception as e:
        env["pgml_extension"]["pgml_c_sha256_error"] = str(e)
    try:
        env["git_commit"] = subprocess.check_output(
            ["git", "-C", str(_REPO), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        env["git_commit"] = None
        env["git_commit_note"] = "workspace has no git repository; use source/binary sha256"
    return env


def write_protocol(forest_repro=None):
    catalog = json.loads((ROOT / "splits_catalog.json").read_text(encoding="utf-8"))
    # Ensure catalog version stamp for v2
    catalog["version"] = "uci_multilibrary_v2"
    catalog["inherited_split_hashes_from"] = "uci_multilibrary_v1"
    (ROOT / "splits_catalog.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    protocol = {
        "version": "uci_multilibrary_v2",
        "inherits_from": "uci_multilibrary_v1 / experiments/common/rerun.py::cls_suite",
        "changes_from_v1": [
            "split_seed separated from model_seed (model_seed=42 for all seeded APIs)",
            "pgml_train_forest uses explicit seed (local LCG); bagged trees re-run",
            "Primary tables aggregate mean±sample std over five splits (ddof=1)",
            "predictions/ per-run files; centralized metric recomputation",
            "LaTeX booktabs table* for accuracy, balanced accuracy, macro-F1",
        ],
        "split": {
            "test_size": 0.2,
            "stratify": True,
            "scaler": "StandardScaler fit on train only; same transformed features for all libraries",
            "split_seeds": SPLIT_SEEDS,
            "model_seed": MODEL_SEED,
        },
        "datasets": catalog["datasets"],
        "learners": PARAMS,
        "parameter_alignment": {
            "decision_tree": {
                "pgml": "pgml_train_tree(x,y, max_depth=8, min_samples=2); gini; all features; no cost-complexity prune",
                "sklearn": "DecisionTreeClassifier(max_depth=8, min_samples_split=2, min_samples_leaf=1); gini",
                "mlpack": "DecisionTree(maximum_depth=8, minimum_leaf_size=1); default minimum_gain_split",
                "r_caret": "rpart maxdepth=8, minsplit=2, minbucket=1, cp=1e-8, xval=0",
                "remaining_difference": (
                    "Splitting enumeration / tie-breaks differ; rpart uses surrogate splits machinery; "
                    "mlpack gain threshold defaults differ from sklearn impurity decrease. "
                    "No claim of numerical parameter equivalence."
                ),
            },
            "bagged_trees": {
                "pgml": "pgml_train_forest(..., seed): bootstrap n, all features, 15 trees, depth 8, local LCG",
                "sklearn": "BaggingClassifier(DecisionTree(...), n_estimators=15, max_features=1.0, max_samples=1.0, bootstrap=True) — NOT RandomForest",
                "mlpack": "UNSUPPORTED (RandomForest uses feature subspace); alternative saved separately",
                "r_caret": "treebag nbagg=15 with rpart maxdepth=8 (ipred)",
                "remaining_difference": (
                    "Bootstrap RNG and tree implementations differ across stacks; "
                    "v1/rerun sklearn RandomForest(max_features=sqrt) is not comparable to this bagging row"
                ),
            },
            "gradient_boosting": {
                "pgml": "exact pgml_train_gbm 30 rounds depth 3 lr 0.1; binary log-loss",
                "sklearn": "GradientBoostingClassifier(n_estimators=30, max_depth=3, learning_rate=0.1)",
                "mlpack": "UNSUPPORTED — no GBM classifier API in installed mlpack",
                "r_caret": "gbm n.trees=30 interaction.depth=3 shrinkage=0.1 bag.fraction=1 train.fraction=1",
                "remaining_difference": "Line search / shrinkage internals differ; no early stopping on test",
            },
            "adaboost": {
                "pgml": "weighted decision stumps, 30 rounds",
                "sklearn": "AdaBoostClassifier SAMME, DecisionTree(max_depth=1), 30 estimators",
                "mlpack": "AdaBoost.MH, weak_learner=decision_stump, iterations=30",
                "r_caret": "AdaBoost.M1 (adabag) mfinal=30 maxdepth=1 coeflearn=Breiman",
                "remaining_difference": (
                    "SAMME / AdaBoost.MH / AdaBoost.M1 are one family but differ; not claimed identical"
                ),
            },
            "linear_svm": {
                "pgml": "SGD hinge, lambda=0.001, iters=20",
                "sklearn": "LinearSVC C=1.0 dual=False max_iter=5000 (liblinear)",
                "mlpack": "linear_svm L2 lambda_=0.001 optimizer=lbfgs",
                "r_caret": "svmLinear (kernlab vanilladot) C=1.0, scaled=FALSE",
                "remaining_difference": (
                    "Solvers and regularization parameterization differ; family-level comparison only"
                ),
            },
        },
        "mlpack_coverage": {
            "decision_tree": "supported",
            "bagged_trees": "unsupported (RF ≠ bagging)",
            "gradient_boosting": "unsupported (no API)",
            "adaboost": "supported",
            "linear_svm": "supported",
        },
        "caret_backends": {
            "decision_tree": "caret method=rpart (package rpart)",
            "bagged_trees": "caret method=treebag (package ipred + rpart)",
            "gradient_boosting": "caret method=gbm (package gbm)",
            "adaboost": "caret method=AdaBoost.M1 (package adabag)",
            "linear_svm": "caret method=svmLinear (package kernlab)",
        },
        "timing_policy": (
            "Timings are auxiliary with heterogeneous boundaries "
            "(SQL API wall vs in-process fit vs Rscript subprocess). "
            "Do not publish a cross-library speedup table."
        ),
        "forest_reproducibility": forest_repro,
        "shark": "not included",
        "planned_cells": len(DATASETS) * len(LEARNERS) * len(IMPLS) * len(SPLIT_SEEDS),
    }
    (ROOT / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    return protocol


def write_report(runs, issues, smoke_ok, forest_repro, env, reused_note):
    agg_acc = aggregate_metric(runs, "accuracy")
    agg_bal = aggregate_metric(runs, "balanced_accuracy")
    agg_f1 = aggregate_metric(runs, "macro_f1")
    n_ok = sum(1 for r in runs if r["status"] == "ok")
    n_un = sum(1 for r in runs if r["status"] == "unsupported")
    n_fail = sum(1 for r in runs if r["status"] == "failed")
    planned = len(DATASETS) * len(LEARNERS) * len(IMPLS) * len(SPLIT_SEEDS)

    lines = [
        "# UCI multilibrary v2 report",
        "",
        "## Summary counts",
        f"- Planned combinations: **{planned}** "
        f"({len(DATASETS)} datasets × {len(LEARNERS)} learners × {len(IMPLS)} libraries × "
        f"{len(SPLIT_SEEDS)} splits)",
        f"- Successful (ok): {n_ok}",
        f"- Unsupported: {n_un}",
        f"- Failed: {n_fail}",
        f"- Smoke test: {'ok' if smoke_ok else 'see failures'}",
        f"- model_seed: {MODEL_SEED} (split_seeds: {SPLIT_SEEDS})",
        "",
        "## Reuse vs re-run",
        reused_note,
        "",
        "## pgml bagged trees reproducibility",
        f"- same connection pred equal: {forest_repro.get('same_connection', {}).get('pred_equal')}",
        f"- same connection model bytes equal: {forest_repro.get('same_connection', {}).get('model_bytes_equal')}",
        f"- new connection pred equal: {forest_repro.get('new_connection', {}).get('pred_equal_to_first')}",
        f"- after PostgreSQL restart pred equal: {forest_repro.get('after_pg_restart', {}).get('pred_equal_to_first')}",
        f"- overall passed: {forest_repro.get('passed')}",
        f"- pgml.so sha256: {env.get('pgml_extension', {}).get('binary_sha256')}",
        f"- pgml.c sha256: {env.get('pgml_extension', {}).get('pgml_c_sha256')}",
        f"- extension version: {env.get('pgml_extension', {}).get('extversion')}",
        f"- git commit: {env.get('git_commit') or env.get('git_commit_note')}",
        "",
        "## mlpack availability (installed version)",
        f"- recorded version: {env.get('libraries', {}).get('mlpack')}",
        "- decision_tree: supported",
        "- bagged_trees: **unsupported** — RandomForest uses feature-subspace sampling; "
        "not equivalent to bootstrap bagging with all features. RF full-subspace alternative "
        "stored in `runs.json` → `alternatives`, excluded from primary tables.",
        "- gradient_boosting: **unsupported** — no gradient-boosting classifier API in this mlpack build",
        "- adaboost: supported (AdaBoost.MH + decision stump)",
        "- linear_svm: supported (L2 + L-BFGS)",
        "",
        "## Comparison limitations (must read)",
        "- AdaBoost variants SAMME (sklearn), AdaBoost.MH (mlpack), and AdaBoost.M1 (caret/adabag) "
        "belong to one family but differ algorithmically.",
        "- Linear SVM solvers and regularization differ (pgml SGD hinge / sklearn liblinear / "
        "mlpack L-BFGS L2 / kernlab vanilladot); family-level comparison only.",
        "- caret is a wrapper: rpart, treebag(ipred+rpart), gbm, AdaBoost.M1(adabag), svmLinear(kernlab).",
        "- This suite compares implementation quality of algorithm *families*, not claimed "
        "hyperparameter-numerical equivalence across libraries.",
        "- Previous sklearn `RandomForestClassifier` (rerun.json / early drafts) is **not** "
        "equivalent to the bagging row here (`BaggingClassifier`, all features).",
        "- Timings are auxiliary with heterogeneous boundaries (SQL exchange vs pure fit); "
        "no cross-library speedup table.",
        "",
        "## Accuracy (mean ± sample std over 5 splits, ddof=1)",
        "",
        write_markdown_table(agg_acc, "accuracy"),
        "",
        "## Balanced accuracy (mean ± sample std)",
        "",
        write_markdown_table(agg_bal, "balanced_accuracy"),
        "",
        "## Macro-F1 (mean ± sample std)",
        "",
        write_markdown_table(agg_f1, "macro_f1"),
        "",
        "## Incomplete / unsupported / failed cells",
    ]
    # List incomplete groups and seed0 failures
    for key, info in sorted(agg_acc.items()):
        if info["kind"] != "ok":
            ds, learner, impl = key
            lines.append(
                f"- {ds} / {learner} / {impl}: {info['kind']} — {info.get('reason') or info['text']}"
            )
    for r in runs:
        if r["status"] == "failed":
            lines.append(
                f"- FAIL {r['dataset']} / {r['learner']} / {r['implementation']} "
                f"split={r['split_seed']}: {r.get('error')}"
            )

    lines += [
        "",
        "## Validation issues",
    ]
    if not issues:
        lines.append("- none")
    else:
        for x in issues:
            lines.append(f"- {x}")

    lines += [
        "",
        "## Differences from v1",
        "- Output directory: `results/uci_multilibrary_v2` (v1 preserved).",
        "- `model_seed=42` independent of `split_seed`; v1 used model random_state=0 / R seed=split.",
        "- pgml bagging uses explicit seed (local LCG); v1 used unseeded libc `rand()`.",
        "- Primary tables aggregate over five splits rather than reporting a single primary seed.",
        "- Per-run files under `predictions/`; LaTeX `table*` + booktabs for three metrics.",
        "",
        "## Remaining limitations",
        "- mlpack bagged_trees and gradient_boosting remain unsupported (documented).",
        "- Cross-library parameter settings are aligned where practical but not proven equivalent.",
        "- Shark not included.",
        "- Workspace has no git commit; extension identity pinned via source/binary SHA-256.",
        "",
        "## Reproduce",
        "```",
        "python experiments/classification/uci_multilibrary/prepare_splits.py",
        "docker exec pgml bash /pgml/experiments/classification/uci_multilibrary/_rebuild_pgml.sh",
        "docker restart pgml",
        "python experiments/classification/uci_multilibrary/run_experiment.py",
        "```",
        "",
        "See also `REPRODUCE.md` in this results directory.",
        "",
        f"Dependencies: PostgreSQL+pgml (docker `pgml`, port 5433); R at `{RSCRIPT}`",
        "with caret, rpart, ipred, gbm, adabag, e1071, kernlab.",
    ]
    (ROOT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_reproduce_md(env):
    text = f"""# Reproduce uci_multilibrary_v2

## Prerequisites
- Docker container `pgml` on localhost:5433 (db `ml_test`, user/password `postgres`)
- Python: numpy, scikit-learn, mlpack==4.8.0, psycopg2-binary
- R at `{RSCRIPT}` with caret, rpart, ipred, gbm, adabag, e1071, kernlab

## Commands
```bat
python experiments/classification/uci_multilibrary/prepare_splits.py
docker exec pgml bash /pgml/experiments/classification/uci_multilibrary/_rebuild_pgml.sh
docker restart pgml
python experiments/classification/uci_multilibrary/run_experiment.py
```

## Pins recorded in environment.json
- pgml extversion: {env.get('pgml_extension', {}).get('extversion')}
- pgml.so sha256: {env.get('pgml_extension', {}).get('binary_sha256')}
- pgml.c sha256: {env.get('pgml_extension', {}).get('pgml_c_sha256')}
- git commit: {env.get('git_commit') or '(none — no git repo)'}
- model_seed: {MODEL_SEED}
- split_seeds: {SPLIT_SEEDS}

## Notes
- v1 results under `results/uci_multilibrary_v1` are left unchanged.
- Do not merge v1 and v2 runs: model seeds and pgml forest RNG differ.
"""
    (ROOT / "REPRODUCE.md").write_text(text, encoding="utf-8")


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    ROOT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    if not (DATA / "wdbc" / "seed_0" / "arrays.npz").exists():
        print("ERROR: splits missing; run prepare_splits.py first", flush=True)
        sys.exit(1)

    reused_note = (
        "- Split arrays/hashes: **reused** from uci_multilibrary_v1 "
        "(identical 80/20 stratified protocol; copied into v2/data).\n"
        "- All learner/library cells: **re-run** under v2 "
        "(model_seed=42; seeded pgml forest; no silent merge with v1 predictions)."
    )

    conn, cur = connect_pg()
    print("== forest reproducibility check (includes PG restart) ==", flush=True)
    forest_repro = check_forest_reproducibility(cur, MODEL_SEED)
    # Reconnect after possible restart inside the check
    try:
        conn.close()
    except Exception:
        pass
    conn, cur = connect_pg()
    (ROOT / "forest_reproducibility.json").write_text(
        json.dumps(forest_repro, indent=2), encoding="utf-8"
    )
    print(json.dumps(forest_repro, indent=2), flush=True)
    if not forest_repro.get("passed"):
        print("WARNING: forest reproducibility check did not fully pass", flush=True)

    write_protocol(forest_repro)
    env = collect_environment()
    (ROOT / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    write_reproduce_md(env)

    smoke = smoke_test(cur, MODEL_SEED)
    (ROOT / "smoke_runs.json").write_text(json.dumps(smoke, indent=2), encoding="utf-8")
    smoke_fail = [r for r in smoke if r["status"] == "failed"]
    if smoke_fail:
        print(f"SMOKE FAILURES: {len(smoke_fail)}", flush=True)
        for r in smoke_fail:
            print(" ", r["learner"], r["implementation"], r.get("error"), flush=True)
        print("Aborting full matrix after smoke failures.", flush=True)
        conn.close()
        sys.exit(2)
    smoke_ok = True

    print("== full matrix (split_seeds 0..4, model_seed=42) ==", flush=True)
    runs = run_matrix(cur, SPLIT_SEEDS, MODEL_SEED)
    (ROOT / "runs.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    write_predictions_csv(runs, ROOT / "predictions.csv")
    write_summary(runs, ROOT / "summary.csv")

    cap_common = (
        "Mean $\\pm$ sample standard deviation (ddof=1) of hold-out metric over five "
        "shared stratified 80/20 splits (\\texttt{split\\_seed}$\\in\\{0,1,2,3,4\\}$), "
        f"with \\texttt{{model\\_seed}}={MODEL_SEED} where the library exposes a seed. "
        "Features standardized with StandardScaler fit on train only; all libraries receive "
        "identical rows in identical order. mlpack bagged trees and gradient boosting are "
        "unsupported (N/A). Cells are family-level comparisons; parameters are aligned where "
        "practical but not claimed numerically equivalent across stacks."
    )
    write_metric_tex(
        runs, "accuracy", ROOT / "accuracy.tex",
        "UCI multilibrary comparison --- accuracy. " + cap_common,
    )
    write_metric_tex(
        runs, "balanced_accuracy", ROOT / "balanced_accuracy.tex",
        "UCI multilibrary comparison --- balanced accuracy. " + cap_common,
    )
    write_metric_tex(
        runs, "macro_f1", ROOT / "macro_f1.tex",
        "UCI multilibrary comparison --- macro-F1. " + cap_common,
    )

    issues = validate(runs)
    (ROOT / "validation.json").write_text(
        json.dumps({"issues": issues, "n_issues": len(issues)}, indent=2), encoding="utf-8"
    )
    write_report(runs, issues, smoke_ok, forest_repro, env, reused_note)
    # refresh protocol with final forest_repro already stored
    write_protocol(forest_repro)
    conn.close()
    print(f"done. issues={len(issues)} out={ROOT}", flush=True)
    if issues:
        print("VALIDATION ISSUES:", *issues[:20], sep="\n  ", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
