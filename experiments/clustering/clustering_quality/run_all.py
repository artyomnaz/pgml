#!/usr/bin/env python3
"""Run clustering_quality_v1 with resume + batched Julia."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import traceback
from pathlib import Path

import numpy as np

from adapters import (
    CFG,
    JULIA,
    connect_pg,
    julia_run,
    julia_run_batch,
    load_canonical,
    mlpack_dbscan,
    mlpack_kmeans,
    pgml_agg,
    pgml_dbscan,
    pgml_kmeans,
    sklearn_agg,
    sklearn_dbscan,
    sklearn_kmeans,
)
from metrics import evaluate

ROOT = Path(CFG["paths"]["results"])
DATA = Path(CFG["paths"]["data"])
PRED = ROOT / "predictions"
RAW = ROOT / "raw_results.jsonl"
LOG = ROOT / "errors.log"
JULIA_WORK = ROOT / "julia_work"

REAL_FULL = ["iris", "wine", "wdbc", "digits", "seeds", "penguins"]
DBSCAN_REAL_2D = ["iris_2d", "seeds_2d", "penguins_2d"]
SYN_FAMILIES = ["blobs", "aniso", "moons", "circles"]


def log_err(msg: str):
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")


def done_keys():
    keys = set()
    if not RAW.exists():
        return keys
    with RAW.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                keys.add(json.loads(line)["run_key"])
    return keys


def append_raw(row: dict):
    with RAW.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def save_pred(run_key: str, row_id, labels):
    PRED.mkdir(parents=True, exist_ok=True)
    path = PRED / f"{run_key}.npz"
    np.savez_compressed(path, row_id=np.asarray(row_id), labels=np.asarray(labels))
    return str(path)


def versions():
    import sklearn
    import mlpack

    ver = {
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "mlpack": getattr(mlpack, "__version__", "4.8.0"),
        "os": platform.platform(),
    }
    try:
        out = subprocess.check_output(
            [str(JULIA), f"--project={Path(__file__).resolve().parent}", "-e",
             "using Clustering; println(pkgversion(Clustering))"],
            text=True, stderr=subprocess.STDOUT,
        ).strip()
        ver["Clustering_jl"] = out
    except Exception as e:
        ver["Clustering_jl"] = f"error: {e}"
    try:
        conn, cur = connect_pg()
        cur.execute("SELECT version()")
        ver["postgresql"] = cur.fetchone()[0]
        try:
            h = subprocess.check_output(
                ["docker", "exec", "pgml", "sh", "-c",
                 "sha256sum $(pg_config --pkglibdir)/pgml.so 2>/dev/null | head -1"],
                text=True,
            ).strip()
            ver["pgml_so_sha256"] = h.split()[0] if h else None
        except Exception as e:
            ver["pgml_so_sha256_error"] = str(e)
        conn.close()
    except Exception as e:
        ver["postgresql_error"] = str(e)
    ver["git_commit"] = None
    return ver


def finalize_ok(base, labels, row_id, y, X64, algo, info):
    if len(labels) != len(row_id):
        raise RuntimeError(f"label length {len(labels)} != n {len(row_id)}")
    mets = evaluate(y, labels, X64, algo)
    base.update(mets)
    base["adapter_info"] = info
    base["prediction_path"] = save_pred(base["run_key"], row_id, labels)
    base["labels_sha256"] = hashlib.sha256(np.asarray(labels, dtype=np.int64).tobytes()).hexdigest()
    return base


def make_base(algo, impl, dataset, params, run_id, dataset_seed, X32):
    run_key = (
        f"{algo}__{impl}__{dataset}__run{run_id}"
        + (f"__ds{dataset_seed}" if dataset_seed is not None else "")
        + (f"__eps{params.get('eps')}_ms{params.get('min_samples')}" if algo == "dbscan" else "")
    ).replace(".", "p")
    return {
        "run_key": run_key,
        "dataset": dataset,
        "dataset_seed": dataset_seed,
        "algorithm": algo,
        "implementation": impl,
        "run_id": run_id,
        "initialization_seed": run_id if algo == "kmeans" else None,
        "params": params,
        "n": int(X32.shape[0]),
        "d": int(X32.shape[1]),
        "K_expected": params.get("K"),
        "status": "ok",
        "error_message": None,
        "prediction_path": None,
    }


def run_python_impl(cur, algo, impl, dataset, X32, X64, y, row_id, params, run_id, dataset_seed=None):
    base = make_base(algo, impl, dataset, params, run_id, dataset_seed, X32)
    try:
        if algo == "kmeans":
            k, mi, seed = params["K"], params["max_iter"], run_id
            if impl == "pgml":
                labels, info = pgml_kmeans(cur, X32, k, mi, seed)
            elif impl == "sklearn":
                labels, info = sklearn_kmeans(X64, k, mi, seed)
            elif impl == "mlpack":
                labels, info = mlpack_kmeans(X64, k, mi, seed)
            else:
                raise ValueError(impl)
        elif algo == "agglomerative":
            k = params["K"]
            if impl == "pgml":
                labels, info = pgml_agg(cur, X32, k)
            elif impl == "sklearn":
                labels, info = sklearn_agg(X64, k)
            else:
                raise ValueError(impl)
        elif algo == "dbscan":
            eps, ms = params["eps"], params["min_samples"]
            if impl == "pgml":
                labels, info = pgml_dbscan(cur, X32, eps, ms)
            elif impl == "sklearn":
                labels, info = sklearn_dbscan(X64, eps, ms)
            elif impl == "mlpack":
                labels, info = mlpack_dbscan(X64, eps, ms)
            else:
                raise ValueError(impl)
        else:
            raise ValueError(algo)
        return finalize_ok(base, labels, row_id, y, X64, algo, info)
    except Exception as e:
        base["status"] = "failed"
        base["error_message"] = str(e)
        log_err(f"{base['run_key']}: {e}\n{traceback.format_exc()[-800:]}")
        return base


def smoke(cur):
    print("== smoke iris ==", flush=True)
    ds = load_canonical(DATA / "iris")
    k = int(ds["meta"]["K_expected"])
    for impl in ("pgml", "sklearn", "mlpack"):
        row = run_python_impl(
            cur, "kmeans", impl, "iris_smoke", ds["X32"], ds["X64"], ds["y"], ds["row_id"],
            {"K": k, "max_iter": 50}, 0,
        )
        print(f"  kmeans/{impl}: {row['status']} ARI={row.get('ARI')}", flush=True)
    batch = julia_run_batch([{"id": "km", "algo": "kmeans", "X32": ds["X32"], "k": k, "seed": 0}])
    labels, info = batch["km"]
    print(f"  kmeans/julia: {'ok' if labels is not None else info}", flush=True)
    for impl in ("pgml", "sklearn"):
        row = run_python_impl(
            cur, "agglomerative", impl, "iris_smoke", ds["X32"], ds["X64"], ds["y"], ds["row_id"],
            {"K": k}, 0,
        )
        print(f"  agg/{impl}: {row['status']} ARI={row.get('ARI')}", flush=True)
    batch = julia_run_batch([{"id": "ag", "algo": "agg", "X32": ds["X32"], "k": k}])
    print(f"  agg/julia: {'ok' if batch['ag'][0] is not None else batch['ag'][1]}", flush=True)
    ds2 = load_canonical(DATA / "iris_2d")
    for impl in ("pgml", "sklearn", "mlpack"):
        row = run_python_impl(
            cur, "dbscan", impl, "iris_2d_smoke", ds2["X32"], ds2["X64"], ds2["y"], ds2["row_id"],
            {"eps": 0.3, "min_samples": 5}, 0,
        )
        print(f"  dbscan/{impl}: {row['status']} ARI={row.get('ARI')}", flush=True)
    batch = julia_run_batch([{"id": "db", "algo": "dbscan", "X32": ds2["X32"], "eps": 0.3, "min_samples": 5}])
    print(f"  dbscan/julia: {'ok' if batch['db'][0] is not None else batch['db'][1]}", flush=True)


def dbscan_targets():
    for name in DBSCAN_REAL_2D:
        yield name, None
    for fam in SYN_FAMILIES:
        for seed in CFG["synthetic"]["seeds"]:
            yield f"{fam}_seed{seed}", seed


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    PRED.mkdir(parents=True, exist_ok=True)
    JULIA_WORK.mkdir(parents=True, exist_ok=True)
    (ROOT / "environment.json").write_text(json.dumps(versions(), indent=2), encoding="utf-8")
    src_rep = Path(__file__).with_name("compatibility_report.md")
    (ROOT / "compatibility_report.md").write_text(src_rep.read_text(encoding="utf-8"), encoding="utf-8")

    conn, cur = connect_pg()
    smoke(cur)
    finished = done_keys()
    print(f"resume: {len(finished)} runs already done", flush=True)

    # ---- real kmeans (non-julia) ----
    for ds_name in REAL_FULL:
        ds = load_canonical(DATA / ds_name)
        k = int(ds["meta"]["K_expected"])
        print(f"== real {ds_name} K={k} ==", flush=True)
        for run_id in range(CFG["kmeans"]["n_runs"]):
            for impl in ("pgml", "sklearn", "mlpack"):
                base = make_base("kmeans", impl, ds_name, {"K": k, "max_iter": CFG["kmeans"]["max_iter"]}, run_id, None, ds["X32"])
                if base["run_key"] in finished:
                    continue
                row = run_python_impl(
                    cur, "kmeans", impl, ds_name, ds["X32"], ds["X64"], ds["y"], ds["row_id"],
                    {"K": k, "max_iter": CFG["kmeans"]["max_iter"]}, run_id,
                )
                append_raw(row)
                finished.add(row["run_key"])
                print(f"  kmeans {impl} run{run_id}: {row['status']} ARI={row.get('ARI')}", flush=True)

        # julia kmeans batch for this dataset
        jobs = []
        meta_jobs = []
        for run_id in range(CFG["kmeans"]["n_runs"]):
            base = make_base("kmeans", "julia", ds_name, {"K": k, "max_iter": CFG["kmeans"]["max_iter"]}, run_id, None, ds["X32"])
            if base["run_key"] in finished:
                continue
            jid = f"km_{ds_name}_{run_id}"
            jobs.append({"id": jid, "algo": "kmeans", "X32": ds["X32"], "k": k, "seed": run_id})
            meta_jobs.append((base, run_id, jid))
        if jobs:
            print(f"  julia kmeans batch n={len(jobs)}", flush=True)
            batch = julia_run_batch(jobs, work_dir=JULIA_WORK / f"km_{ds_name}")
            for base, run_id, jid in meta_jobs:
                labels, info = batch[jid]
                if labels is None:
                    base["status"] = info.get("status", "failed")
                    base["error_message"] = info.get("error")
                else:
                    try:
                        base = finalize_ok(base, labels, ds["row_id"], ds["y"], ds["X64"], "kmeans", info)
                    except Exception as e:
                        base["status"] = "failed"
                        base["error_message"] = str(e)
                append_raw(base)
                finished.add(base["run_key"])
                print(f"  kmeans julia run{run_id}: {base['status']} ARI={base.get('ARI')}", flush=True)

        # agglomerative non-julia
        for impl in ("pgml", "sklearn"):
            base = make_base("agglomerative", impl, ds_name, {"K": k, "linkage": "complete"}, 0, None, ds["X32"])
            if base["run_key"] in finished:
                continue
            print(f"  agg {impl} ...", flush=True)
            row = run_python_impl(
                cur, "agglomerative", impl, ds_name, ds["X32"], ds["X64"], ds["y"], ds["row_id"],
                {"K": k, "linkage": "complete"}, 0,
            )
            append_raw(row)
            finished.add(row["run_key"])
            print(f"  agg {impl}: {row['status']} ARI={row.get('ARI')}", flush=True)

        base = make_base("agglomerative", "julia", ds_name, {"K": k, "linkage": "complete"}, 0, None, ds["X32"])
        if base["run_key"] not in finished:
            print("  julia agg ...", flush=True)
            batch = julia_run_batch(
                [{"id": "ag", "algo": "agg", "X32": ds["X32"], "k": k}],
                work_dir=JULIA_WORK / f"ag_{ds_name}",
            )
            labels, info = batch["ag"]
            if labels is None:
                base["status"] = info.get("status", "failed")
                base["error_message"] = info.get("error")
            else:
                try:
                    base = finalize_ok(base, labels, ds["row_id"], ds["y"], ds["X64"], "agglomerative", info)
                except Exception as e:
                    base["status"] = "failed"
                    base["error_message"] = str(e)
            append_raw(base)
            finished.add(base["run_key"])
            print(f"  agg julia: {base['status']} ARI={base.get('ARI')}", flush=True)

    # ---- DBSCAN: non-julia first, then one big julia batch per dataset ----
    db_cfg = CFG["dbscan"]
    for ds_name, dseed in dbscan_targets():
        ds = load_canonical(DATA / ds_name)
        print(f"== dbscan {ds_name} ==", flush=True)
        julia_jobs = []
        julia_meta = []
        for eps in db_cfg["eps_grid"]:
            for ms in db_cfg["min_samples_grid"]:
                params = {"eps": eps, "min_samples": ms}
                for impl in ("pgml", "sklearn", "mlpack"):
                    base = make_base("dbscan", impl, ds_name, params, 0, dseed, ds["X32"])
                    if base["run_key"] in finished:
                        continue
                    row = run_python_impl(
                        cur, "dbscan", impl, ds_name, ds["X32"], ds["X64"], ds["y"], ds["row_id"],
                        params, 0, dseed,
                    )
                    append_raw(row)
                    finished.add(row["run_key"])
                base = make_base("dbscan", "julia", ds_name, params, 0, dseed, ds["X32"])
                if base["run_key"] in finished:
                    continue
                jid = f"db_{ds_name}_e{eps}_m{ms}".replace(".", "p")
                julia_jobs.append({
                    "id": jid, "algo": "dbscan", "X32": ds["X32"], "eps": eps, "min_samples": ms,
                })
                julia_meta.append((base, jid))
        if julia_jobs:
            print(f"  julia dbscan batch n={len(julia_jobs)}", flush=True)
            batch = julia_run_batch(julia_jobs, work_dir=JULIA_WORK / f"db_{ds_name}")
            for base, jid in julia_meta:
                labels, info = batch[jid]
                if labels is None:
                    base["status"] = info.get("status", "failed")
                    base["error_message"] = info.get("error")
                else:
                    try:
                        base = finalize_ok(base, labels, ds["row_id"], ds["y"], ds["X64"], "dbscan", info)
                    except Exception as e:
                        base["status"] = "failed"
                        base["error_message"] = str(e)
                append_raw(base)
                finished.add(base["run_key"])
        print(f"  done {ds_name}", flush=True)

    conn.close()
    print("all scheduled runs complete; run aggregate.py next", flush=True)


if __name__ == "__main__":
    main()
