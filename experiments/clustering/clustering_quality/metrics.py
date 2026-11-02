#!/usr/bin/env python3
"""Shared metrics for clustering_quality_v1 (single sklearn implementation)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
)


def inertia_from_labels(X: np.ndarray, labels: np.ndarray) -> float:
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    total = 0.0
    for c in np.unique(labels):
        mask = labels == c
        if not np.any(mask):
            continue
        center = X[mask].mean(axis=0)
        diff = X[mask] - center
        total += float(np.sum(diff * diff))
    return total


def evaluate(y_true, labels, X, algorithm: str):
    labels = np.asarray(labels, dtype=np.int64)
    n = len(labels)
    out = {
        "ARI": None,
        "AMI": None,
        "silhouette": None,
        "silhouette_status": "ok",
        "silhouette_reason": None,
        "K_found": None,
        "noise_pct": None,
        "inertia": None,
        "status": "ok",
    }
    if y_true is not None:
        y_true = np.asarray(y_true)
        out["ARI"] = float(adjusted_rand_score(y_true, labels))
        out["AMI"] = float(
            adjusted_mutual_info_score(y_true, labels, average_method="arithmetic")
        )

    if algorithm == "dbscan":
        noise = labels < 0
        out["noise_pct"] = float(100.0 * np.mean(noise))
        non = ~noise
        clusters = np.unique(labels[non]) if np.any(non) else np.array([])
        out["K_found"] = int(len(clusters))
        n_non = int(np.sum(non))
        K = out["K_found"]
        if n_non < 2 or K < 2 or K > n_non - 1:
            out["silhouette"] = None
            out["silhouette_status"] = "undefined_metric"
            out["silhouette_reason"] = (
                f"need 2 <= K <= n_non_noise-1; got K={K}, n_non={n_non}"
            )
        else:
            out["silhouette"] = float(
                silhouette_score(X[non], labels[non], metric="euclidean")
            )
    else:
        out["K_found"] = int(len(np.unique(labels)))
        out["noise_pct"] = 0.0
        K = out["K_found"]
        if K < 2 or K > n - 1:
            out["silhouette"] = None
            out["silhouette_status"] = "undefined_metric"
            out["silhouette_reason"] = f"need 2 <= K <= n-1; got K={K}, n={n}"
        else:
            out["silhouette"] = float(silhouette_score(X, labels, metric="euclidean"))
        if algorithm == "kmeans":
            out["inertia"] = inertia_from_labels(X, labels)
    return out


def agreement_dbscan(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a)
    b = np.asarray(b)
    noise_a = a < 0
    noise_b = b < 0
    return {
        "ari_partition": float(adjusted_rand_score(a, b)),
        "noise_mask_agreement": float(np.mean(noise_a == noise_b)),
        "K_a": int(len(np.unique(a[a >= 0]))),
        "K_b": int(len(np.unique(b[b >= 0]))),
        "noise_pct_a": float(100 * np.mean(noise_a)),
        "noise_pct_b": float(100 * np.mean(noise_b)),
    }
