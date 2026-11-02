#!/usr/bin/env python3
"""Build versioned shared splits for WDBC / Ionosphere / Sonar (uci_multilibrary_v2).

Protocol mirrors experiments/common/rerun.py cls_suite where recoverable:
  - 80/20 stratified hold-out, random_state=split_seed
  - StandardScaler fit on train only
  - exact (non-histogram) pgml path for quality
Does NOT overwrite results/rerun.json or results/uci_multilibrary_v1.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
_REPO = Path(__file__).resolve().parents[3]

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(_REPO.joinpath('results', 'uci_multilibrary_v2'))
DATA = ROOT / "data"
SEEDS_PRIMARY = [0]
SEEDS_EXTRA = [0, 1, 2, 3, 4]
TEST_SIZE = 0.2


def sha256_ids(ids: np.ndarray) -> str:
    arr = np.asarray(ids, dtype=np.int64)
    return hashlib.sha256(np.sort(arr).tobytes()).hexdigest()


def load_uci_csv(url: str, label_last: bool = True):
    with urllib.request.urlopen(url, timeout=60) as resp:
        raw = resp.read().decode("utf-8", errors="replace").strip().splitlines()
    rows = [line.split(",") for line in raw if line and not line.startswith("@")]
    X, y = [], []
    labels = {}
    kept_row_ids = []
    for i, row in enumerate(rows):
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
        kept_row_ids.append(i)
    return (
        np.asarray(X, dtype=np.float64),
        np.asarray(y, dtype=np.int64),
        np.asarray(kept_row_ids, dtype=np.int64),
        {str(k): int(v) for k, v in labels.items()},
    )


def load_datasets():
    out = {}
    # WDBC via sklearn = Wisconsin Diagnostic Breast Cancer
    bc = load_breast_cancer()
    out["wdbc"] = {
        "source": "sklearn.datasets.load_breast_cancer",
        "source_note": "sklearn wrapper of UCI Breast Cancer Wisconsin (Diagnostic) / WDBC",
        "uci_id": "breast-cancer-wisconsin-diagnostic",
        "n_features": int(bc.data.shape[1]),
        "feature_names": [str(x) for x in bc.feature_names],
        "class_names": [str(x) for x in bc.target_names],
        "label_encoding": {str(n): int(i) for i, n in enumerate(bc.target_names)},
        "X": bc.data.astype(np.float64),
        "y": bc.target.astype(np.int64),
        "row_ids": np.arange(bc.data.shape[0], dtype=np.int64),
        "cleaning": "none beyond sklearn loader (no missing values in WDBC)",
    }

    ion_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
    Xi, yi, ion_ids, ion_lab = load_uci_csv(ion_url, label_last=True)
    out["ionosphere"] = {
        "source": ion_url,
        "source_note": "UCI Ionosphere; rows with ? skipped (none expected)",
        "uci_id": "ionosphere",
        "n_features": int(Xi.shape[1]),
        "feature_names": [f"f{i}" for i in range(Xi.shape[1])],
        "class_names": [k for k, _ in sorted(ion_lab.items(), key=lambda kv: kv[1])],
        "label_encoding": ion_lab,
        "X": Xi,
        "y": yi,
        "row_ids": ion_ids,
        "cleaning": "drop rows containing ? or empty feature cells; label last column mapped to 0..K-1 by first-seen order",
    }

    sonar_url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "undocumented/connectionist-bench/sonar/sonar.all-data"
    )
    Xs, ys, sonar_ids, sonar_lab = load_uci_csv(sonar_url, label_last=True)
    out["sonar"] = {
        "source": sonar_url,
        "source_note": "UCI Connectionist Bench (Sonar, Mines vs. Rocks)",
        "uci_id": "sonar",
        "n_features": int(Xs.shape[1]),
        "feature_names": [f"f{i}" for i in range(Xs.shape[1])],
        "class_names": [k for k, _ in sorted(sonar_lab.items(), key=lambda kv: kv[1])],
        "label_encoding": sonar_lab,
        "X": Xs,
        "y": ys,
        "row_ids": sonar_ids,
        "cleaning": "drop rows containing ? or empty feature cells; label last column mapped to 0..K-1 by first-seen order",
    }
    return out


def make_split(ds: dict, seed: int):
    X = ds["X"]
    y = ds["y"]
    row_ids = ds["row_ids"]
    idx = np.arange(len(y), dtype=np.int64)
    tr, te = train_test_split(idx, test_size=TEST_SIZE, random_state=seed, stratify=y)
    tr = np.asarray(tr, dtype=np.int64)
    te = np.asarray(te, dtype=np.int64)
    assert len(set(tr) & set(te)) == 0
    scaler = StandardScaler().fit(X[tr])
    Xtr = scaler.transform(X[tr]).astype(np.float64)
    Xte = scaler.transform(X[te]).astype(np.float64)
    return {
        "seed": seed,
        "test_size": TEST_SIZE,
        "stratify": True,
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "train_pos": tr,  # positions in X/y arrays
        "test_pos": te,
        "train_row_ids": row_ids[tr],
        "test_row_ids": row_ids[te],
        "train_ids_sha256": sha256_ids(row_ids[tr]),
        "test_ids_sha256": sha256_ids(row_ids[te]),
        "train_pos_sha256": sha256_ids(tr),
        "test_pos_sha256": sha256_ids(te),
        "X_train": Xtr,
        "X_test": Xte,
        "y_train": y[tr].astype(np.int64),
        "y_test": y[te].astype(np.int64),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "feature_names": ds["feature_names"],
        "label_encoding": ds["label_encoding"],
        "class_names": ds["class_names"],
        "positive_class": 1 if 1 in set(y.tolist()) else int(np.max(y)),
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
    meta = {k: v for k, v in split.items() if k not in (
        "X_train", "X_test", "y_train", "y_test", "train_pos", "test_pos",
        "train_row_ids", "test_row_ids",
    )}
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
        "version": "uci_multilibrary_v2",
        "test_size": TEST_SIZE,
        "split_seeds": SEEDS_EXTRA,
        "note": "split_seed controls train_test_split only; model_seed is separate in the runner",
        "datasets": {},
    }
    for name, ds in datasets.items():
        catalog["datasets"][name] = {
            "source": ds["source"],
            "source_note": ds["source_note"],
            "uci_id": ds["uci_id"],
            "n": int(ds["X"].shape[0]),
            "n_features": ds["n_features"],
            "feature_names": ds["feature_names"],
            "class_names": ds["class_names"],
            "label_encoding": ds["label_encoding"],
            "cleaning": ds["cleaning"],
            "splits": {},
        }
        # persist raw for audit
        raw_dir = DATA / name / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(raw_dir / "xy.npz", X=ds["X"], y=ds["y"], row_ids=ds["row_ids"])
        for seed in SEEDS_EXTRA:
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
