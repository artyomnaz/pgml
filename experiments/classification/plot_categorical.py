"""Plots for categorical feature comparison."""
from __future__ import annotations

import json
from pathlib import Path
_REPO = Path(__file__).resolve().parents[2]

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(_REPO)
SRC = ROOT / "results" / "categorical_comparison.json"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "pgml_cat": "#1b7f4a",
    "pgml_codes": "#6b4c3b",
    "onehot_sklearn_tree": "#7a8a9a",
    "onehot_sklearn_histgbm": "#1f4e79",
    "catboost_native": "#b35c1e",
    "lightgbm_native": "#5c4d7a",
}
MARKERS = {
    "pgml_cat": "*",
    "pgml_codes": "D",
    "onehot_sklearn_tree": "s",
    "onehot_sklearn_histgbm": "o",
    "catboost_native": "^",
    "lightgbm_native": "v",
}
LABELS = {
    "pgml_cat": "pgml cat splits",
    "pgml_codes": "pgml numeric codes",
    "onehot_sklearn_tree": "one-hot + tree",
    "onehot_sklearn_histgbm": "one-hot + HistGBM",
    "catboost_native": "CatBoost native",
    "lightgbm_native": "LightGBM native",
}


def load():
    return json.loads(SRC.read_text(encoding="utf-8"))


def valid(pts):
    return [p for p in pts if "error" not in p and "train_ms" in p]


def plot(data):
    pts = valid(data["points"])
    systems = data["systems"]
    cards = data["grid"]["cardinality"]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0), layout="constrained")

    # A: quality vs train pipeline time
    ax = axes[0]
    seen = set()
    for p in pts:
        sys = p["system"]
        ax.scatter(
            max(p["total_pipeline_ms"], 0.1),
            p["quality"]["primary"],
            s=35 + 10 * np.log2(max(p["cardinality"], 2)),
            c=COLORS[sys],
            marker=MARKERS[sys],
            edgecolors="#222",
            linewidths=0.6,
            alpha=0.85,
            label=LABELS[sys] if sys not in seen else None,
            zorder=3 if sys.startswith("pgml") else 2,
        )
        seen.add(sys)
    ax.set_xscale("log")
    ax.set_xlabel("encode + train (ms)")
    ax.set_ylabel("holdout accuracy")
    ax.set_title("A  Quality vs pipeline time")
    ax.grid(True, which="both", ls=":", alpha=0.45)
    ax.legend(fontsize=7, loc="best", framealpha=0.92)

    # B: heatmap-like lines — accuracy vs cardinality at rare=0.5
    ax = axes[1]
    rare_focus = 0.5
    for sys in systems:
        xs, ys = [], []
        for k in cards:
            hit = next(
                (p for p in pts if p["system"] == sys and p["cardinality"] == k
                 and abs(p["rare_frac"] - rare_focus) < 1e-9),
                None,
            )
            if hit:
                xs.append(k)
                ys.append(hit["quality"]["primary"])
        if xs:
            ax.plot(xs, ys, MARKERS[sys] + "-", color=COLORS[sys], label=LABELS[sys],
                    markersize=7, linewidth=1.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks(cards)
    ax.set_xticklabels([str(k) for k in cards])
    ax.set_xlabel("cardinality $K$")
    ax.set_ylabel("holdout accuracy")
    ax.set_title(r"B  Accuracy vs $K$ (rare frac $=0.5$)")
    ax.grid(True, ls=":", alpha=0.45)
    ax.legend(fontsize=7, loc="best", framealpha=0.92)

    # C: model size + encoded width at rare=0.5
    ax = axes[2]
    width_ax = ax.twinx()
    for sys in systems:
        xs, ms, ws = [], [], []
        for k in cards:
            hit = next(
                (p for p in pts if p["system"] == sys and p["cardinality"] == k
                 and abs(p["rare_frac"] - rare_focus) < 1e-9),
                None,
            )
            if hit:
                xs.append(k)
                ms.append(hit["model_bytes"] / 1024)
                ws.append(hit["encoded_width"])
        if xs:
            ax.plot(xs, ms, MARKERS[sys] + "-", color=COLORS[sys], label=LABELS[sys],
                    markersize=6, linewidth=1.4)
            if sys.startswith("onehot"):
                width_ax.plot(xs, ws, ":", color=COLORS[sys], linewidth=1.2, alpha=0.8)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(cards)
    ax.set_xticklabels([str(k) for k in cards])
    ax.set_xlabel("cardinality $K$")
    ax.set_ylabel("model size (KiB)")
    width_ax.set_ylabel("encoded width (one-hot systems, dotted)")
    ax.set_title(r"C  Model size vs $K$ (rare $=0.5$)")
    ax.grid(True, which="both", ls=":", alpha=0.45)
    ax.legend(fontsize=6.5, loc="upper left", framealpha=0.92)

    fig.suptitle(
        "Categorical features: pgml vs one-hot vs CatBoost / LightGBM  "
        f"(n={data['hyperparams']['n_rows']}, depth={data['hyperparams']['max_depth']})",
        fontsize=10,
    )
    png = OUT / "categorical_comparison.png"
    pdf = OUT / "categorical_comparison.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}")
    print(f"wrote {pdf}")

    # Secondary heatmap: accuracy gap pgml_cat - pgml_codes over K x rare
    fig2, axes2 = plt.subplots(1, 2, figsize=(9.5, 3.8), layout="constrained")
    rares = data["grid"]["rare_fraction"]

    def mat(sys):
        Z = np.full((len(rares), len(cards)), np.nan)
        for i, r in enumerate(rares):
            for j, k in enumerate(cards):
                hit = next(
                    (p for p in pts if p["system"] == sys and p["cardinality"] == k
                     and abs(p["rare_frac"] - r) < 1e-9),
                    None,
                )
                if hit:
                    Z[i, j] = hit["quality"]["primary"]
        return Z

    Zcat = mat("pgml_cat")
    Zcode = mat("pgml_codes")
    gap = Zcat - Zcode
    im0 = axes2[0].imshow(gap, aspect="auto", cmap="RdYlGn", vmin=-0.05, vmax=0.15)
    axes2[0].set_xticks(range(len(cards)))
    axes2[0].set_xticklabels([str(k) for k in cards])
    axes2[0].set_yticks(range(len(rares)))
    axes2[0].set_yticklabels([f"{r:.2f}" for r in rares])
    axes2[0].set_xlabel("cardinality $K$")
    axes2[0].set_ylabel("rare-category fraction")
    axes2[0].set_title("A  Acc(pgml cat) − Acc(pgml codes)")
    for i in range(gap.shape[0]):
        for j in range(gap.shape[1]):
            if np.isnan(gap[i, j]):
                continue
            axes2[0].text(j, i, f"{gap[i, j]:+.3f}", ha="center", va="center", fontsize=8)
    fig2.colorbar(im0, ax=axes2[0], fraction=0.046, pad=0.04)

    # pipeline time heatmap for onehot tree vs pgml_cat ratio
    Zoh = np.full_like(gap, np.nan)
    Zpg = np.full_like(gap, np.nan)
    for i, r in enumerate(rares):
        for j, k in enumerate(cards):
            a = next((p for p in pts if p["system"] == "onehot_sklearn_tree"
                      and p["cardinality"] == k and abs(p["rare_frac"] - r) < 1e-9), None)
            b = next((p for p in pts if p["system"] == "pgml_cat"
                      and p["cardinality"] == k and abs(p["rare_frac"] - r) < 1e-9), None)
            if a and b:
                Zoh[i, j] = a["total_pipeline_ms"]
                Zpg[i, j] = b["total_pipeline_ms"]
    ratio = Zoh / Zpg
    im1 = axes2[1].imshow(np.log2(ratio), aspect="auto", cmap="coolwarm",
                          vmin=-2, vmax=4)
    axes2[1].set_xticks(range(len(cards)))
    axes2[1].set_xticklabels([str(k) for k in cards])
    axes2[1].set_yticks(range(len(rares)))
    axes2[1].set_yticklabels([f"{r:.2f}" for r in rares])
    axes2[1].set_xlabel("cardinality $K$")
    axes2[1].set_ylabel("rare-category fraction")
    axes2[1].set_title(r"B  $\log_2$(one-hot tree time / pgml cat)")
    for i in range(ratio.shape[0]):
        for j in range(ratio.shape[1]):
            if np.isnan(ratio[i, j]):
                continue
            axes2[1].text(j, i, f"{ratio[i, j]:.1f}×", ha="center", va="center", fontsize=8)
    cbar = fig2.colorbar(im1, ax=axes2[1], fraction=0.046, pad=0.04)
    cbar.set_label(r"$\log_2$ time ratio")

    png2 = OUT / "categorical_heatmaps.png"
    pdf2 = OUT / "categorical_heatmaps.pdf"
    fig2.savefig(png2, dpi=180, bbox_inches="tight")
    fig2.savefig(pdf2, bbox_inches="tight")
    plt.close(fig2)
    print(f"wrote {png2}")
    print(f"wrote {pdf2}")


if __name__ == "__main__":
    plot(load())
