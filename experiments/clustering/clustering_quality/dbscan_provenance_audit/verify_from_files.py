#!/usr/bin/env python3
"""
Recompute DBSCAN metrics/agreement ONLY from audit prediction files.
Does not call clustering libraries. Does not touch suite results.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
_REPO = Path(__file__).resolve().parents[4]

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, silhouette_score

OUT = Path(_REPO.joinpath('results', 'clustering_quality_v1', 'dbscan_provenance_audit'))
CASES = ["iris_2d", "penguins_2d", "aniso_seed8"]
IMPLS = ["pgml", "sklearn", "mlpack", "julia"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pred(path: Path):
    z = np.load(path, allow_pickle=True)
    row_id = np.asarray(z["row_id"], dtype=np.int64)
    labels = np.asarray(z["labels"], dtype=np.int64)
    order = np.argsort(row_id)
    return row_id[order], labels[order]


def cluster_sizes(labels: np.ndarray) -> dict:
    c = Counter(int(x) for x in labels.tolist())
    return {str(k): v for k, v in sorted(c.items(), key=lambda kv: (kv[0] < 0, kv[0]))}


def disagree_indices(a: np.ndarray, b: np.ndarray) -> dict:
    """Indices where partitions differ after accounting for label permutation via ARI/noise.

    We report:
    - noise_only_a / noise_only_b
    - among non-noise points with identical noise masks, points whose co-clustering
      with the rest differs (contingency mismatch under majority matching).
    """
    noise_a = a < 0
    noise_b = b < 0
    only_a = np.where(noise_a & ~noise_b)[0].tolist()
    only_b = np.where(~noise_a & noise_b)[0].tolist()

    # majority-map cluster ids from b -> a on jointly non-noise points
    both = ~noise_a & ~noise_b
    mapping = {}
    if np.any(both):
        for lab_b in np.unique(b[both]):
            mask = both & (b == lab_b)
            # majority a-label
            labs, counts = np.unique(a[mask], return_counts=True)
            mapping[int(lab_b)] = int(labs[np.argmax(counts)])
    remapped = np.array(
        [(-1 if x < 0 else mapping.get(int(x), -999999)) for x in b.tolist()],
        dtype=np.int64,
    )
    label_mismatch = np.where((a != remapped) & (remapped != -999999))[0].tolist()
    unmatched_b = np.where(remapped == -999999)[0].tolist()
    return {
        "only_a_noise": only_a,
        "only_b_noise": only_b,
        "after_majority_map_label_mismatch": label_mismatch,
        "b_clusters_unmatched": unmatched_b,
        "majority_map_b_to_a": {str(k): v for k, v in mapping.items()},
    }


def metrics_vs_y(y, labels, X):
    labels = np.asarray(labels, dtype=np.int64)
    out = {
        "ARI": float(adjusted_rand_score(y, labels)) if y is not None else None,
        "AMI": float(adjusted_mutual_info_score(y, labels, average_method="arithmetic"))
        if y is not None
        else None,
        "K_found": int(len(np.unique(labels[labels >= 0]))),
        "noise_pct": float(100.0 * np.mean(labels < 0)),
        "silhouette": None,
        "silhouette_status": "ok",
    }
    non = labels >= 0
    K = out["K_found"]
    n_non = int(np.sum(non))
    if n_non < 2 or K < 2 or K > n_non - 1:
        out["silhouette_status"] = "undefined_metric"
    else:
        out["silhouette"] = float(silhouette_score(X[non], labels[non], metric="euclidean"))
    return out


def fmt10(x):
    if x is None:
        return None
    return f"{float(x):.10f}"


def main():
    report = {"cases": {}, "file_checks": []}
    lines = ["# DBSCAN provenance audit — metrics from files only", ""]

    for ds in CASES:
        case_dir = OUT / "cases" / ds
        inp_dir = OUT / "inputs" / ds
        X = np.load(inp_dir / "X64_from_f32.npy")
        y_path = inp_dir / "y.npy"
        y = np.load(y_path) if y_path.exists() else None
        man = json.loads((case_dir / "run_manifest.json").read_text(encoding="utf-8"))

        labs = {}
        case_out = {"per_impl": {}, "pairwise": {}}
        lines.append(f"## {ds}")
        lines.append("")
        for impl in IMPLS:
            pred = case_dir / "predictions" / f"{impl}.npz"
            csvp = case_dir / "predictions" / f"{impl}_labels.csv"
            row_id, labels = load_pred(pred)
            labs[impl] = labels
            # verify csv matches npz
            csv_labs = []
            with csvp.open(encoding="utf-8") as f:
                next(f)
                for line in f:
                    csv_labs.append(int(line.strip().split(",")[1]))
            csv_labs = np.asarray(csv_labs, dtype=np.int64)
            assert np.array_equal(labels, csv_labs), impl

            # verify manifest sha
            lab_sha = hashlib.sha256(np.ascontiguousarray(labels).tobytes()).hexdigest()
            file_sha = sha256_file(pred)
            claimed = man["runs"][impl]["labels_sha256"]
            claimed_file = man["runs"][impl]["prediction_npz_sha256"]
            mets = metrics_vs_y(y, labels, X)
            entry = {
                "prediction_npz": str(pred.resolve()),
                "prediction_csv": str(csvp.resolve()),
                "labels_sha256_recomputed": lab_sha,
                "labels_sha256_manifest": claimed,
                "labels_sha_match": lab_sha == claimed,
                "npz_sha256_recomputed": file_sha,
                "npz_sha256_manifest": claimed_file,
                "npz_sha_match": file_sha == claimed_file,
                "cluster_sizes": cluster_sizes(labels),
                "metrics": {k: (fmt10(v) if isinstance(v, float) else v) for k, v in mets.items()},
                "metrics_raw": mets,
                "call": man["runs"][impl]["call"],
                "n": int(labels.shape[0]),
            }
            case_out["per_impl"][impl] = entry
            report["file_checks"].append(
                {"dataset": ds, "impl": impl, "labels_sha_match": entry["labels_sha_match"], "npz_sha_match": entry["npz_sha_match"]}
            )
            lines.append(f"### {impl}")
            lines.append(f"- call: `{entry['call']}`")
            lines.append(f"- pred: `{pred}`")
            lines.append(f"- labels_sha256: `{lab_sha}` match_manifest={entry['labels_sha_match']}")
            lines.append(f"- sizes: `{entry['cluster_sizes']}`")
            lines.append(
                f"- ARI={entry['metrics']['ARI']} AMI={entry['metrics']['AMI']} "
                f"Sil={entry['metrics']['silhouette']} K={entry['metrics']['K_found']} "
                f"noise%={entry['metrics']['noise_pct']}"
            )
            lines.append("")

        # pairwise
        lines.append("### Pairwise (label-permutation aware via ARI; noise mask separate)")
        lines.append("")
        for a, b in combinations(IMPLS, 2):
            la, lb = labs[a], labs[b]
            ari = float(adjusted_rand_score(la, lb))
            noise_agree = float(np.mean((la < 0) == (lb < 0)))
            diffs = disagree_indices(la, lb)
            key = f"{a}__vs__{b}"
            case_out["pairwise"][key] = {
                "ari_partition": fmt10(ari),
                "ari_partition_raw": ari,
                "noise_mask_agreement": fmt10(noise_agree),
                "noise_mask_agreement_raw": noise_agree,
                "K": [int(len(np.unique(la[la >= 0]))), int(len(np.unique(lb[lb >= 0])))],
                "sizes_a": cluster_sizes(la),
                "sizes_b": cluster_sizes(lb),
                "disagreements": diffs,
                "n_noise_diffs": len(diffs["only_a_noise"]) + len(diffs["only_b_noise"]),
                "n_label_mismatch_after_map": len(diffs["after_majority_map_label_mismatch"]),
            }
            lines.append(
                f"- **{a} vs {b}**: ARI={fmt10(ari)} noise_agree={fmt10(noise_agree)} "
                f"noise_idx_only_{a}={diffs['only_a_noise']} only_{b}={diffs['only_b_noise']} "
                f"mismatch_after_map={diffs['after_majority_map_label_mismatch']}"
            )
        lines.append("")
        report["cases"][ds] = case_out
        (case_dir / "verify_from_files.json").write_text(
            json.dumps(case_out, indent=2) + "\n", encoding="utf-8"
        )

    # isolation check
    iso = json.loads((OUT / "isolation_eps_sklearn_only" / "isolation_report.json").read_text(encoding="utf-8"))
    report["isolation"] = iso
    lines.append("## Isolation diagnostic")
    lines.append(f"- baseline_unchanged: **{iso['baseline_unchanged']}**")
    lines.append(f"- diagnostic output only: `{iso['isolation_output']}`")
    lines.append("")

    (OUT / "verify_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (OUT / "verify_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((OUT / "verify_report.md").read_text(encoding="utf-8"))
    print("Wrote", OUT / "verify_report.md")


if __name__ == "__main__":
    main()
