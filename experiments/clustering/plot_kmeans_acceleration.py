"""Plots for k-means exact vs approximate acceleration."""
from __future__ import annotations

import json
from pathlib import Path
_REPO = Path(__file__).resolve().parents[2]

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(_REPO)
SRC = ROOT / "results" / "kmeans_acceleration.json"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "lloyd_reference": "#6b4c3b",
    "sklearn_lloyd": "#7a8a9a",
    "sklearn_elkan": "#1f4e79",
    "mlpack_naive": "#5c4d7a",
    "mlpack_hamerly": "#1b7f4a",
    "mlpack_elkan": "#2a9d5c",
    "sklearn_minibatch": "#b35c1e",
    "pgml_plusplus": "#c4a035",
    "pgml_random": "#9b2226",
    "pgml_parallel": "#0a7a6d",
}
MARKERS = {
    "lloyd_reference": "D",
    "sklearn_lloyd": "s",
    "sklearn_elkan": "o",
    "mlpack_naive": "^",
    "mlpack_hamerly": "*",
    "mlpack_elkan": "P",
    "sklearn_minibatch": "v",
    "pgml_plusplus": "X",
    "pgml_random": "x",
    "pgml_parallel": "h",
}
LABELS = {
    "lloyd_reference": "Lloyd reference",
    "sklearn_lloyd": "sklearn Lloyd",
    "sklearn_elkan": "sklearn Elkan",
    "mlpack_naive": "mlpack Lloyd",
    "mlpack_hamerly": "mlpack Hamerly",
    "mlpack_elkan": "mlpack Elkan",
    "sklearn_minibatch": "MiniBatch (approx.)",
    "pgml_plusplus": "pgml ++",
    "pgml_random": "pgml random",
    "pgml_parallel": "pgml parallel-4",
}


def load():
    return json.loads(SRC.read_text(encoding="utf-8"))


def valid(pts):
    return [p for p in pts if "error" not in p and "train_ms" in p]


def plot(data):
    pts = valid(data["points"])
    exact = [p for p in pts if p.get("family") == "exact_acceleration" and p.get("shared_init")]
    approx = [p for p in pts if p.get("family") == "approximate"]

    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.0), layout="constrained")

    # A: time to target inertia vs n (d=32, k=8)
    ax = axes[0]
    systems_a = [
        "lloyd_reference", "sklearn_lloyd", "sklearn_elkan",
        "mlpack_naive", "mlpack_hamerly", "mlpack_elkan", "sklearn_minibatch",
    ]
    for sys in systems_a:
        xs, ys = [], []
        for n in data["grid"]["n"]:
            hit = next(
                (p for p in pts if p["system"] == sys and p["n"] == n
                 and p["d"] == 32 and p["k"] == 8
                 and p.get("time_to_target_ms") is not None),
                None,
            )
            if hit:
                xs.append(n)
                ys.append(hit["time_to_target_ms"])
        if xs:
            ax.plot(xs, ys, MARKERS[sys] + "-", color=COLORS[sys], label=LABELS[sys],
                    markersize=7, linewidth=1.4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"rows $n$")
    ax.set_ylabel("time to target inertia (ms)")
    ax.set_title(r"A  Time to target ($d{=}32$, $k{=}8$)")
    ax.grid(True, which="both", ls=":", alpha=0.45)
    ax.legend(fontsize=6.5, loc="best", framealpha=0.92)

    # B: ms/iter
    ax = axes[1]
    for sys in systems_a:
        xs, ys = [], []
        for n in data["grid"]["n"]:
            hit = next(
                (p for p in pts if p["system"] == sys and p["n"] == n
                 and p["d"] == 32 and p["k"] == 8
                 and p.get("ms_per_iter") is not None),
                None,
            )
            if hit:
                xs.append(n)
                ys.append(hit["ms_per_iter"])
        if xs:
            ax.plot(xs, ys, MARKERS[sys] + "-", color=COLORS[sys], label=LABELS[sys],
                    markersize=7, linewidth=1.4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"rows $n$")
    ax.set_ylabel("ms / iteration")
    ax.set_title(r"B  Per-iteration cost ($d{=}32$, $k{=}8$)")
    ax.grid(True, which="both", ls=":", alpha=0.45)
    ax.legend(fontsize=6.5, loc="best", framealpha=0.92)

    # C: inertia vs time scatter (all shared-init exact + approx)
    ax = axes[2]
    seen = set()
    for p in exact + approx:
        sys = p["system"]
        ax.scatter(
            max(p["train_ms"], 0.05),
            p["inertia"],
            s=30 + 6 * np.log2(max(p["n"] / 1000, 1)),
            c=COLORS.get(sys, "#333"),
            marker=MARKERS.get(sys, "o"),
            edgecolors="#222",
            linewidths=0.5,
            alpha=0.85,
            label=LABELS.get(sys, sys) if sys not in seen else None,
        )
        seen.add(sys)
    ax.set_xscale("log")
    ax.set_xlabel("train time (ms)")
    ax.set_ylabel("final inertia")
    ax.set_title("C  Inertia vs time (shared init)")
    ax.grid(True, which="both", ls=":", alpha=0.45)
    ax.legend(fontsize=6.5, loc="best", framealpha=0.92)

    fig.suptitle(
        f"k-means exact acceleration vs MiniBatch  (mlpack {data['mlpack_version']}; "
        f"shared k-means++ centers)",
        fontsize=10,
    )
    png = OUT / "kmeans_acceleration.png"
    pdf = OUT / "kmeans_acceleration.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    plot(load())
