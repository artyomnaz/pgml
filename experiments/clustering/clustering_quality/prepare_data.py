#!/usr/bin/env python3
"""Prepare canonical float32 standardized matrices for clustering_quality_v1."""
from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
_REPO = Path(__file__).resolve().parents[3]

import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
    make_blobs,
    make_circles,
    make_moons,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(_REPO.joinpath('results', 'clustering_quality_v1'))
DATA = ROOT / "data"
CACHE = DATA / "cache"
CFG = json.loads(Path(__file__).with_name("config.json").read_text(encoding="utf-8"))


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    fetched = datetime.now(timezone.utc).isoformat()
    if not dest.exists():
        with urllib.request.urlopen(url, timeout=120) as resp:
            data = resp.read()
        dest.write_bytes(data)
    else:
        data = dest.read_bytes()
    return {
        "url": url,
        "path": str(dest),
        "fetched_at_utc": fetched if dest.exists() else None,
        "sha256": sha256_bytes(data),
        "nbytes": len(data),
    }


def save_dataset(name: str, X: np.ndarray, y: np.ndarray | None, meta: dict):
    out = DATA / name
    out.mkdir(parents=True, exist_ok=True)
    X = np.asarray(X, dtype=np.float64)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X).astype(np.float32)
    # re-expand float32 for libraries that want float64 without re-standardizing
    Xs64 = Xs.astype(np.float64)
    row_id = np.arange(Xs.shape[0], dtype=np.int64)
    np.savez_compressed(
        out / "canonical.npz",
        X_f32=Xs,
        X_f64_from_f32=Xs64,
        y=None if y is None else np.asarray(y),
        row_id=row_id,
        scaler_mean=scaler.mean_.astype(np.float64),
        scaler_scale=scaler.scale_.astype(np.float64),
    )
    # CSV for Julia (rows = observations)
    np.savetxt(out / "X_f32.csv", Xs, delimiter=",")
    if y is not None:
        np.savetxt(out / "y.csv", np.asarray(y), delimiter=",", fmt="%d")
    np.savetxt(out / "row_id.csv", row_id, delimiter=",", fmt="%d")
    meta = {
        **meta,
        "name": name,
        "n": int(Xs.shape[0]),
        "d": int(Xs.shape[1]),
        "n_classes": None if y is None else int(len(np.unique(y))),
        "K_expected": None if y is None else int(len(np.unique(y))),
        "dtype_canonical": "float32",
        "standardization": "StandardScaler fit on full matrix (unsupervised descriptive benchmark)",
        "sha256_X_f32": sha256_bytes(Xs.tobytes()),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_seeds():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt"
    info = download(url, CACHE / "seeds_dataset.txt")
    raw = Path(info["path"]).read_text(encoding="utf-8", errors="replace").strip().splitlines()
    rows = []
    for line in raw:
        parts = [p for p in line.replace("\t", " ").split(" ") if p]
        if len(parts) < 8:
            continue
        rows.append([float(x) for x in parts])
    arr = np.asarray(rows, dtype=np.float64)
    X, y = arr[:, :7], arr[:, 7].astype(int) - 1  # varieties 1..3 -> 0..2
    return X, y, {
        "source": info,
        "feature_names": [
            "area", "perimeter", "compactness", "length", "width", "asymmetry", "groove_length"
        ],
        "label_name": "variety",
        "rows_dropped": 0,
        "notes": "UCI seeds_dataset.txt; label remapped 1..3 -> 0..2",
    }


def load_penguins():
    url = (
        "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/"
        "master/inst/extdata/penguins.csv"
    )
    info = download(url, CACHE / "penguins.csv")
    df = pd.read_csv(info["path"])
    feats = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    n_before = len(df)
    sub = df[feats + ["species"]].copy()
    sub = sub.dropna()
    dropped = n_before - len(sub)
    # encode species
    classes = sorted(sub["species"].unique().tolist())
    mapping = {c: i for i, c in enumerate(classes)}
    y = sub["species"].map(mapping).to_numpy(dtype=np.int64)
    X = sub[feats].to_numpy(dtype=np.float64)
    return X, y, {
        "source": info,
        "feature_names": feats,
        "label_name": "species",
        "label_encoding": mapping,
        "rows_before": n_before,
        "rows_dropped": int(dropped),
        "notes": "palmerpenguins inst/extdata/penguins.csv; drop NA on features+species; no island/sex/id as features",
    }


def real_datasets():
    out = {}
    iris = load_iris()
    out["iris"] = save_dataset(
        "iris",
        iris.data,
        iris.target,
        {
            "source": {"library": "sklearn.datasets.load_iris"},
            "feature_names": list(iris.feature_names),
            "label_name": "species",
            "label_encoding": {str(n): int(i) for i, n in enumerate(iris.target_names)},
            "rows_dropped": 0,
        },
    )
    wine = load_wine()
    out["wine"] = save_dataset(
        "wine",
        wine.data,
        wine.target,
        {
            "source": {"library": "sklearn.datasets.load_wine"},
            "feature_names": list(wine.feature_names),
            "label_name": "cultivar",
            "label_encoding": {str(n): int(i) for i, n in enumerate(wine.target_names)},
            "rows_dropped": 0,
        },
    )
    bc = load_breast_cancer()
    out["wdbc"] = save_dataset(
        "wdbc",
        bc.data,
        bc.target,
        {
            "source": {"library": "sklearn.datasets.load_breast_cancer", "note": "WDBC"},
            "feature_names": list(bc.feature_names),
            "label_name": "diagnosis",
            "label_encoding": {str(n): int(i) for i, n in enumerate(bc.target_names)},
            "rows_dropped": 0,
        },
    )
    digits = load_digits()
    out["digits"] = save_dataset(
        "digits",
        digits.data,
        digits.target,
        {
            "source": {"library": "sklearn.datasets.load_digits"},
            "feature_names": [f"px{i}" for i in range(digits.data.shape[1])],
            "label_name": "digit",
            "n_classes_note": "10 digits",
            "rows_dropped": 0,
        },
    )
    Xs, ys, meta = load_seeds()
    out["seeds"] = save_dataset("seeds", Xs, ys, meta)
    Xp, yp, metap = load_penguins()
    out["penguins"] = save_dataset("penguins", Xp, yp, metap)

    # 2D real for DBSCAN
    out["iris_2d"] = save_dataset(
        "iris_2d",
        iris.data[:, 2:4],
        iris.target,
        {
            "source": {"library": "sklearn iris petal length/width"},
            "feature_names": list(iris.feature_names[2:4]),
            "label_name": "species",
            "rows_dropped": 0,
            "for": "dbscan",
        },
    )
    out["seeds_2d"] = save_dataset(
        "seeds_2d",
        Xs[:, 0:2],
        ys,
        {
            "source": meta["source"],
            "feature_names": ["area", "perimeter"],
            "label_name": "variety",
            "rows_dropped": 0,
            "for": "dbscan",
        },
    )
    out["penguins_2d"] = save_dataset(
        "penguins_2d",
        Xp[:, 0:2],
        yp,
        {
            "source": metap["source"],
            "feature_names": ["bill_length_mm", "bill_depth_mm"],
            "label_name": "species",
            "rows_dropped": metap["rows_dropped"],
            "for": "dbscan",
        },
    )
    return out


def synthetic_datasets():
    syn_cfg = CFG["synthetic"]
    centers = np.asarray(syn_cfg["blob_centers"], dtype=np.float64)
    A = np.asarray(syn_cfg["aniso_matrix"], dtype=np.float64)
    n = syn_cfg["n"]
    out = {}
    for seed in syn_cfg["seeds"]:
        rng = np.random.default_rng(seed)
        # blobs with fixed centers
        Xb, yb = make_blobs(
            n_samples=n,
            centers=centers,
            cluster_std=syn_cfg["blob_std"],
            random_state=seed,
        )
        name = f"blobs_seed{seed}"
        out[name] = save_dataset(
            name,
            Xb,
            yb,
            {
                "source": {"generator": "sklearn.make_blobs", "centers": centers.tolist(), "seed": seed},
                "synthetic_family": "blobs",
                "dataset_seed": seed,
                "for": "dbscan",
            },
        )
        Xa = Xb @ A.T
        name = f"aniso_seed{seed}"
        out[name] = save_dataset(
            name,
            Xa,
            yb,
            {
                "source": {
                    "generator": "blobs then @ aniso_matrix.T",
                    "matrix": A.tolist(),
                    "seed": seed,
                },
                "synthetic_family": "aniso",
                "dataset_seed": seed,
                "for": "dbscan",
            },
        )
        Xm, ym = make_moons(n_samples=n, noise=syn_cfg["moons_noise"], random_state=seed)
        name = f"moons_seed{seed}"
        out[name] = save_dataset(
            name,
            Xm,
            ym,
            {
                "source": {"generator": "sklearn.make_moons", "noise": syn_cfg["moons_noise"], "seed": seed},
                "synthetic_family": "moons",
                "dataset_seed": seed,
                "for": "dbscan",
            },
        )
        Xc, yc = make_circles(
            n_samples=n,
            factor=syn_cfg["circles_factor"],
            noise=syn_cfg["circles_noise"],
            random_state=seed,
        )
        name = f"circles_seed{seed}"
        out[name] = save_dataset(
            name,
            Xc,
            yc,
            {
                "source": {
                    "generator": "sklearn.make_circles",
                    "factor": syn_cfg["circles_factor"],
                    "noise": syn_cfg["circles_noise"],
                    "seed": seed,
                },
                "synthetic_family": "circles",
                "dataset_seed": seed,
                "for": "dbscan",
            },
        )
        _ = rng  # silence unused in some branches
    return out


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    real = real_datasets()
    syn = synthetic_datasets()
    rows = []
    for name, meta in {**real, **syn}.items():
        rows.append(
            {
                "dataset": name,
                "n": meta["n"],
                "d": meta["d"],
                "n_classes": meta.get("n_classes"),
                "K_expected": meta.get("K_expected"),
                "rows_dropped": meta.get("rows_dropped"),
                "sha256_X_f32": meta.get("sha256_X_f32"),
                "for": meta.get("for", "kmeans_agg"),
                "synthetic_family": meta.get("synthetic_family"),
                "dataset_seed": meta.get("dataset_seed"),
            }
        )
    import csv

    man = ROOT / "dataset_manifest.csv"
    with man.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (ROOT / "datasets_index.json").write_text(json.dumps({"real": real, "synthetic_keys": list(syn)}, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} datasets -> {DATA}", flush=True)


if __name__ == "__main__":
    main()
