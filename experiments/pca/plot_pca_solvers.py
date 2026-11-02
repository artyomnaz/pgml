"""Plots for PCA Jacobi vs sklearn solvers."""
from __future__ import annotations

import json
from pathlib import Path
_REPO = Path(__file__).resolve().parents[2]

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(_REPO)
SRC = ROOT / "results" / "pca_solvers.json"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "pgml_jacobi": "#1b7f4a",
    "sklearn_full": "#1f4e79",
    "sklearn_covariance_eigh": "#5c4d7a",
    "sklearn_randomized": "#b35c1e",
}
MARKERS = {
    "pgml_jacobi": "*",
    "sklearn_full": "o",
    "sklearn_covariance_eigh": "s",
    "sklearn_randomized": "^",
}
LABELS = {
    "pgml_jacobi": "pgml Jacobi",
    "sklearn_full": "sklearn full",
    "sklearn_covariance_eigh": "sklearn cov. eigh",
    "sklearn_randomized": "sklearn randomized (approx.)",
}


def load():
    return json.loads(SRC.read_text(encoding="utf-8"))


def valid(pts):
    return [p for p in pts if "error" not in p and "train_ms" in p]


def plot(data):
    pts = valid(data["points"])
    systems = data["systems"]

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.0), layout="constrained")

    # A: reconstruction vs time
    ax = axes[0]
    seen = set()
    for p in pts:
        sys = p["system"]
        ax.scatter(
            max(p["train_ms"], 0.05),
            p["reconstruction_rel_frob"],
            s=28 + 8 * np.log2(max(p["d"], 2)),
            c=COLORS[sys],
            marker=MARKERS[sys],
            edgecolors="#222",
            linewidths=0.55,
            alpha=0.85,
            label=LABELS[sys] if sys not in seen else None,
            zorder=3 if sys == "pgml_jacobi" else 2,
        )
        seen.add(sys)
    ax.set_xscale("log")
    ax.set_xlabel("train time (ms)")
    ax.set_ylabel(r"rel. reconstruction $\|X-\hat X\|_F/\|X\|_F$")
    ax.set_title("A  Reconstruction vs time")
    ax.grid(True, which="both", ls=":", alpha=0.45)
    ax.legend(fontsize=7, loc="best", framealpha=0.92)

    # B: train time vs d at n=5000, k=8
    ax = axes[1]
    for sys in systems:
        xs, ys = [], []
        for d in data["grid"]["d"]:
            hit = next(
                (p for p in pts if p["system"] == sys and p["n"] == 5000
                 and p["d"] == d and p["k"] == 8),
                None,
            )
            if hit:
                xs.append(d)
                ys.append(hit["train_ms"])
        if xs:
            ax.plot(xs, ys, MARKERS[sys] + "-", color=COLORS[sys], label=LABELS[sys],
                    markersize=7, linewidth=1.5)
    ax.set_yscale("log")
    ax.set_xlabel(r"dimensionality $d$")
    ax.set_ylabel("train time (ms)")
    ax.set_title(r"B  Time vs $d$ ($n{=}5$k, $k{=}8$)")
    ax.grid(True, which="both", ls=":", alpha=0.45)
    ax.legend(fontsize=7, loc="best", framealpha=0.92)

    # C: subspace distance vs sklearn full
    ax = axes[2]
    for sys in systems:
        if sys == "sklearn_full":
            continue
        xs, ys = [], []
        # sort by d*k roughly
        sub = [p for p in pts if p["system"] == sys and p.get("subspace_dist_vs_sklearn_full") is not None]
        sub.sort(key=lambda p: (p["d"], p["k"], p["n"]))
        for p in sub:
            xs.append(p["d"])
            ys.append(max(p["subspace_dist_vs_sklearn_full"], 1e-16))
        ax.scatter(xs, ys, c=COLORS[sys], marker=MARKERS[sys], s=40,
                   edgecolors="#222", linewidths=0.5, label=LABELS[sys], alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlabel(r"dimensionality $d$")
    ax.set_ylabel(r"Grassmann dist. to sklearn full")
    ax.set_title("C  Subspace distance")
    ax.axhline(1e-6, color="#888", ls="--", lw=0.8, label=r"$10^{-6}$")
    ax.grid(True, which="both", ls=":", alpha=0.45)
    ax.legend(fontsize=7, loc="best", framealpha=0.92)

    fig.suptitle(
        "PCA solvers: pgml Jacobi vs sklearn full / covariance_eigh / randomized",
        fontsize=10,
    )
    png = OUT / "pca_solvers.png"
    pdf = OUT / "pca_solvers.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    plot(load())
