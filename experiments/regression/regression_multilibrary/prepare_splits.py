#!/usr/bin/env python3
"""Shared 80/20 splits for Diabetes / Airfoil / Concrete (regression_multilibrary_v1).

Protocol:
  - test_size=0.2, split seeds 0..4 (no stratify; continuous target)
  - StandardScaler fit on train only; same transformed features for all libraries
  - does not overwrite classification multilibrary or results/rerun.json
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
_REPO = Path(__file__).resolve().parents[3]

import numpy as np
from sklearn.datasets import fetch_openml, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(_REPO.joinpath('results', 'regression_multilibrary_v1'))
DATA = ROOT / "data"
SPLIT_SEEDS = [0, 1, 2, 3, 4]
TEST_SIZE = 0.2


def sha256_ids(ids: np.ndarray) -> str:
    arr = np.asarray(ids, dtype=np.int64)
    return hashlib.sha256(np.sort(arr).tobytes()).hexdigest()


def load_datasets():
    out = {}

    diab = load_diabetes()
    out["diabetes"] = {
        "source": "sklearn.datasets.load_diabetes",
        "source_note": "sklearn diabetes regression (not OpenML CC18 diabetes classification)",
        "openml_id": None,
        "n_features": int(diab.data.shape[1]),
        "feature_names": [str(x) for x in diab.feature_names],
        "target_name": "disease_progression",
        "X": diab.data.astype(np.float64),
        "y": diab.target.astype(np.float64),
        "row_ids": np.arange(diab.data.shape[0], dtype=np.int64),
        "cleaning": "none beyond sklearn loader",
    }

    air = fetch_openml(data_id=44957, as_frame=False, parser="auto")
    Xa = np.asarray(air.data, dtype=np.float64)
    ya = np.asarray(air.target, dtype=np.float64).ravel()
    out["airfoil"] = {
        "source": "openml data_id=44957 (airfoil_self_noise)",
        "source_note": "UCI Airfoil Self-Noise via OpenML; target sound_pressure",
        "openml_id": 44957,
        "n_features": int(Xa.shape[1]),
        "feature_names": [str(x) for x in air.feature_names],
        "target_name": "sound_pressure",
        "X": Xa,
        "y": ya,
        "row_ids": np.arange(Xa.shape[0], dtype=np.int64),
        "cleaning": "none; OpenML numeric matrix",
    }

    con = fetch_openml(data_id=44959, as_frame=False, parser="auto")
    Xc = np.asarray(con.data, dtype=np.float64)
    yc = np.asarray(con.target, dtype=np.float64).ravel()
    out["concrete"] = {
        "source": "openml data_id=44959 (concrete_compressive_strength)",
        "source_note": "UCI Concrete Compressive Strength via OpenML; target strength",
        "openml_id": 44959,
        "n_features": int(Xc.shape[1]),
        "feature_names": [str(x) for x in con.feature_names],
        "target_name": "strength",
        "X": Xc,
        "y": yc,
        "row_ids": np.arange(Xc.shape[0], dtype=np.int64),
        "cleaning": "none; OpenML numeric matrix",
    }
    return out


def make_split(ds: dict, seed: int):
    X = ds["X"]
    y = ds["y"]
    row_ids = ds["row_ids"]
    idx = np.arange(len(y), dtype=np.int64)
    tr, te = train_test_split(idx, test_size=TEST_SIZE, random_state=seed)
    tr = np.asarray(tr, dtype=np.int64)
    te = np.asarray(te, dtype=np.int64)
    assert len(set(tr) & set(te)) == 0
    scaler = StandardScaler().fit(X[tr])
    Xtr = scaler.transform(X[tr]).astype(np.float64)
    Xte = scaler.transform(X[te]).astype(np.float64)
    return {
        "seed": seed,
        "test_size": TEST_SIZE,
        "stratify": False,
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "train_pos": tr,
        "test_pos": te,
        "train_row_ids": row_ids[tr],
        "test_row_ids": row_ids[te],
        "train_ids_sha256": sha256_ids(row_ids[tr]),
        "test_ids_sha256": sha256_ids(row_ids[te]),
        "train_pos_sha256": sha256_ids(tr),
        "test_pos_sha256": sha256_ids(te),
        "X_train": Xtr,
        "X_test": Xte,
        "y_train": y[tr].astype(np.float64),
        "y_test": y[te].astype(np.float64),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "feature_names": ds["feature_names"],
        "target_name": ds["target_name"],
    }


def save_split(dataset: str, split: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "arrays.npz",
        X_train=split["X_train"],
        X_test=split["X_test"],
        y_train=split["y_train"],
        y_test=split["y_test"],
        train_pos=split["train_pos"],
        test_pos=split["test_pos"],
        train_row_ids=split["train_row_ids"],
        test_row_ids=split["test_row_ids"],
        scaler_mean=np.asarray(split["scaler_mean"], dtype=np.float64),
        scaler_scale=np.asarray(split["scaler_scale"], dtype=np.float64),
    )
    meta = {
        k: v
        for k, v in split.items()
        if k
        not in (
            "X_train",
            "X_test",
            "y_train",
            "y_test",
            "train_pos",
            "test_pos",
            "train_row_ids",
            "test_row_ids",
        )
    }
    meta["train_row_ids"] = split["train_row_ids"].tolist()
    meta["test_row_ids"] = split["test_row_ids"].tolist()
    meta["train_pos"] = split["train_pos"].tolist()
    meta["test_pos"] = split["test_pos"].tolist()
    meta["dataset"] = dataset
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    datasets = load_datasets()
    catalog = {
        "version": "regression_multilibrary_v1",
        "test_size": TEST_SIZE,
        "split_seeds": SPLIT_SEEDS,
        "scaler": "StandardScaler fit on train only",
        "stratify": False,
        "datasets": {},
    }
    for name, ds in datasets.items():
        catalog["datasets"][name] = {
            "source": ds["source"],
            "source_note": ds["source_note"],
            "openml_id": ds["openml_id"],
            "n": int(ds["X"].shape[0]),
            "n_features": ds["n_features"],
            "feature_names": ds["feature_names"],
            "target_name": ds["target_name"],
            "cleaning": ds["cleaning"],
            "splits": {},
        }
        raw_dir = DATA / name / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(raw_dir / "xy.npz", X=ds["X"], y=ds["y"], row_ids=ds["row_ids"])
        for seed in SPLIT_SEEDS:
            split = make_split(ds, seed)
            meta = save_split(name, split, DATA / name / f"seed_{seed}")
            catalog["datasets"][name]["splits"][str(seed)] = {
                "n_train": meta["n_train"],
                "n_test": meta["n_test"],
                "train_ids_sha256": meta["train_ids_sha256"],
                "test_ids_sha256": meta["test_ids_sha256"],
                "train_pos_sha256": meta["train_pos_sha256"],
                "test_pos_sha256": meta["test_pos_sha256"],
            }
            print(
                f"{name} seed={seed} n_train={meta['n_train']} n_test={meta['n_test']} "
                f"train_sha={meta['train_ids_sha256'][:12]}...",
                flush=True,
            )
    (ROOT / "splits_catalog.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    print(f"wrote {ROOT / 'splits_catalog.json'}", flush=True)


if __name__ == "__main__":
    main()
