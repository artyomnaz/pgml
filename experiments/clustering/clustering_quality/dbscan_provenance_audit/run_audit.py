#!/usr/bin/env python3
"""
Independent DBSCAN provenance audit.

Writes ONLY under results/clustering_quality_v1/dbscan_provenance_audit/
Does NOT read or write clustering_quality_v1/predictions or raw_results.jsonl.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adapters import (  # noqa: E402
    CFG,
    connect_pg,
    julia_run_batch,
    load_canonical,
    mlpack_dbscan,
    pgml_dbscan,
    sklearn_dbscan,
)

DATA = Path(CFG["paths"]["data"])
OUT = Path(CFG["paths"]["results"]) / "dbscan_provenance_audit"
CASES = [
    ("iris_2d", None),
    ("penguins_2d", None),
    ("aniso_seed8", 8),
]
EPS = 0.30
MS = 5
IMPLS = ("pgml", "sklearn", "mlpack", "julia")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_labels(labels: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(labels, dtype=np.int64))
    return sha256_bytes(arr.tobytes())


def copy_inputs(ds_name: str) -> dict:
    src = DATA / ds_name
    dst = OUT / "inputs" / ds_name
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("canonical.npz", "X_f32.csv", "y.csv", "row_id.csv", "meta.json"):
        p = src / name
        if p.exists():
            shutil.copy2(p, dst / name)
    # also write dedicated X32/X64 npy for explicit hashing
    ds = load_canonical(src)
    np.save(dst / "X32.npy", ds["X32"])
    np.save(dst / "X64_from_f32.npy", ds["X64"])
    np.save(dst / "row_id.npy", ds["row_id"])
    if ds["y"] is not None:
        np.save(dst / "y.npy", ds["y"])
    return {
        "dataset": ds_name,
        "source_dir": str(src.resolve()),
        "audit_input_dir": str(dst.resolve()),
        "canonical_npz_sha256": sha256_file(dst / "canonical.npz"),
        "X32_sha256": sha256_file(dst / "X32.npy"),
        "X64_from_f32_sha256": sha256_file(dst / "X64_from_f32.npy"),
        "row_id_sha256": sha256_file(dst / "row_id.npy"),
        "n": int(ds["X32"].shape[0]),
        "d": int(ds["X32"].shape[1]),
        "X32": ds["X32"],
        "X64": ds["X64"],
        "y": ds["y"],
        "row_id": ds["row_id"],
    }


def run_one(impl: str, cur, X32, X64, eps: float, ms: int):
    if impl == "pgml":
        labels, info = pgml_dbscan(cur, X32, eps, ms)
        call = (
            f"SELECT pgml_dbscan(%s::float4[][], {eps}::float4, {ms})  "
            f"# via adapters.pgml_dbscan; X rows={X32.shape[0]} float32"
        )
    elif impl == "sklearn":
        labels, info = sklearn_dbscan(X64, eps, ms)
        call = (
            f"sklearn.cluster.DBSCAN(eps={eps}, min_samples={ms}, metric='euclidean')"
            f".fit_predict(X64_from_f32)"
        )
    elif impl == "mlpack":
        labels, info = mlpack_dbscan(X64, eps, ms)
        call = (
            f"mlpack.dbscan(input_=Xf, epsilon={eps}, min_size={ms}, naive=True)  "
            f"# Xf=asfortranarray(X64_from_f32)"
        )
    elif impl == "julia":
        batch = julia_run_batch(
            [{"id": "db", "algo": "dbscan", "X32": X32, "eps": eps, "min_samples": ms}]
        )
        labels, info = batch["db"]
        if labels is None:
            raise RuntimeError(f"julia failed: {info}")
        call = (
            f"Clustering.dbscan(X';d×n, {eps}; min_neighbors={ms}, min_cluster_size=1)  "
            f"# via julia_batch.jl; X = Float64.(X32)' "
        )
    else:
        raise ValueError(impl)
    return np.asarray(labels, dtype=np.int64), info, call


def save_pred(case_dir: Path, impl: str, row_id, labels, meta: dict) -> Path:
    pred_dir = case_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    path = pred_dir / f"{impl}.npz"
    # also plain csv for inspection
    csv_path = pred_dir / f"{impl}_labels.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("row_id,label\n")
        for rid, lab in zip(row_id.tolist(), labels.tolist()):
            f.write(f"{rid},{lab}\n")
    np.savez_compressed(
        path,
        row_id=np.asarray(row_id, dtype=np.int64),
        labels=np.asarray(labels, dtype=np.int64),
        meta_json=np.array(json.dumps(meta)),
    )
    return path


def main():
    if OUT.exists():
        # wipe only this audit dir, never the main suite
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    # versions
    import mlpack
    import sklearn

    versions = {
        "sklearn": sklearn.__version__,
        "mlpack": getattr(mlpack, "__version__", "4.8.0"),
        "numpy": np.__version__,
        "adapter_file": str((ROOT / "adapters.py").resolve()),
        "julia_batch": str((ROOT / "julia_batch.jl").resolve()),
        "Clustering_jl": "0.15.8",
        "pgml_fn": "pgml_dbscan",
        "suite_results_untouched": str(Path(CFG["paths"]["results"]) / "predictions"),
    }
    (OUT / "environment.json").write_text(json.dumps(versions, indent=2) + "\n", encoding="utf-8")

    # provenance map from code (static)
    provenance = {
        "note": "Fresh audit run; does not read suite prediction cache.",
        "implementations": {
            "pgml": {
                "adapter": "experiments/clustering/clustering_quality/adapters.py::pgml_dbscan",
                "call": "SELECT pgml_dbscan(float4[][], eps, min_pts)",
                "fallback_to_sklearn": False,
                "input": "X32 float32 from canonical.npz",
            },
            "sklearn": {
                "adapter": "experiments/clustering/clustering_quality/adapters.py::sklearn_dbscan",
                "call": "sklearn.cluster.DBSCAN(...).fit_predict(X64)",
                "fallback_to_sklearn": "N/A (is sklearn)",
                "input": "X64 expanded from float32",
            },
            "mlpack": {
                "adapter": "experiments/clustering/clustering_quality/adapters.py::mlpack_dbscan",
                "call": "mlpack.dbscan(..., naive=True)",
                "fallback_to_sklearn": False,
                "input": "Fortran X64 from float32",
            },
            "julia": {
                "adapter": "adapters.julia_run_batch + julia_batch.jl",
                "call": "Clustering.dbscan(..., min_neighbors=ms, min_cluster_size=1)",
                "fallback_to_sklearn": False,
                "input": "X32 written to temp CSV, read as Float64 d×n",
            },
        },
        "run_key_pattern_suite": "dbscan__{impl}__{dataset}__run0__eps{eps}_ms{ms}",
        "audit_pred_pattern": "{case}/predictions/{impl}.npz  (separate files per impl)",
    }
    (OUT / "provenance_static.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    conn, cur = connect_pg()
    manifest = []

    for ds_name, dseed in CASES:
        print(f"== {ds_name} ==", flush=True)
        inp = copy_inputs(ds_name)
        case_dir = OUT / "cases" / ds_name
        case_dir.mkdir(parents=True, exist_ok=True)
        case_meta = {
            "dataset": ds_name,
            "dataset_seed": dseed,
            "eps": EPS,
            "min_samples": MS,
            "input_sha256": {
                "canonical_npz": inp["canonical_npz_sha256"],
                "X32": inp["X32_sha256"],
                "X64_from_f32": inp["X64_from_f32_sha256"],
                "row_id": inp["row_id_sha256"],
            },
            "n": inp["n"],
            "d": inp["d"],
            "runs": {},
        }
        for impl in IMPLS:
            labels, info, call = run_one(impl, cur, inp["X32"], inp["X64"], EPS, MS)
            assert labels.shape[0] == inp["n"], (impl, labels.shape, inp["n"])
            meta = {
                "implementation": impl,
                "call": call,
                "adapter_info": info,
                "eps": EPS,
                "min_samples": MS,
                "labels_sha256": sha256_labels(labels),
                "n_labels": int(labels.shape[0]),
                "unique_labels": sorted(set(int(x) for x in labels.tolist())),
            }
            pred_path = save_pred(case_dir, impl, inp["row_id"], labels, meta)
            meta["prediction_npz"] = str(pred_path.resolve())
            meta["prediction_csv"] = str((case_dir / "predictions" / f"{impl}_labels.csv").resolve())
            meta["prediction_npz_sha256"] = sha256_file(pred_path)
            meta["prediction_csv_sha256"] = sha256_file(case_dir / "predictions" / f"{impl}_labels.csv")
            case_meta["runs"][impl] = meta
            print(
                f"  {impl}: K={len([u for u in meta['unique_labels'] if u >= 0])} "
                f"noise={sum(1 for x in labels if x < 0)} sha={meta['labels_sha256'][:12]}",
                flush=True,
            )
        (case_dir / "run_manifest.json").write_text(
            json.dumps({k: v for k, v in case_meta.items() if k != "runs"} | {"runs": case_meta["runs"]}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        # strip arrays if any leaked
        manifest.append(
            {
                "dataset": ds_name,
                "manifest": str((case_dir / "run_manifest.json").resolve()),
                "predictions_dir": str((case_dir / "predictions").resolve()),
            }
        )

    # ---- isolation diagnostic: change eps only for sklearn on iris_2d ----
    print("== isolation diagnostic (sklearn eps=0.31 on iris_2d) ==", flush=True)
    iso_dir = OUT / "isolation_eps_sklearn_only"
    iso_dir.mkdir(parents=True)
    # snapshot hashes of other impl predictions before
    iris_pred = OUT / "cases" / "iris_2d" / "predictions"
    before = {p.name: sha256_file(p) for p in iris_pred.glob("*")}
    inp = load_canonical(DATA / "iris_2d")
    labels_sk, info, call = run_one("sklearn", cur, inp["X32"], inp["X64"], 0.31, MS)
    # write ONLY to isolation dir — must NOT touch cases/iris_2d/predictions
    iso_pred = iso_dir / "sklearn_eps0p31.npz"
    np.savez_compressed(iso_pred, row_id=inp["row_id"], labels=labels_sk)
    after = {p.name: sha256_file(p) for p in iris_pred.glob("*")}
    isolation = {
        "description": "Ran sklearn DBSCAN with eps=0.31; wrote only under isolation_eps_sklearn_only/",
        "changed_adapter": "sklearn",
        "eps_diagnostic": 0.31,
        "isolation_output": str(iso_pred.resolve()),
        "isolation_labels_sha256": sha256_labels(labels_sk),
        "baseline_iris_predictions_before": before,
        "baseline_iris_predictions_after": after,
        "baseline_unchanged": before == after,
        "note": "Suite predictions/ and other audit case files must remain byte-identical.",
    }
    (iso_dir / "isolation_report.json").write_text(json.dumps(isolation, indent=2) + "\n", encoding="utf-8")
    print(f"  baseline_unchanged={isolation['baseline_unchanged']}", flush=True)

    (OUT / "audit_index.json").write_text(
        json.dumps({"cases": manifest, "isolation": isolation, "versions": versions}, indent=2) + "\n",
        encoding="utf-8",
    )
    try:
        conn.close()
    except Exception:
        pass
    print(f"DONE -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
