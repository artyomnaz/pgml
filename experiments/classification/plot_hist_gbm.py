"""Quality vs training time (+ memory/time heat facets) for hist GBM comparison."""
from __future__ import annotations

import json
from pathlib import Path
_REPO = Path(__file__).resolve().parents[2]

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(_REPO)
SRC = ROOT / "results" / "hist_gbm_comparison.json"
OUT_DIR = ROOT / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "pgml_exact": "#6b4c3b",
    "pgml_hist": "#1b7f4a",
    "sklearn_gbm": "#7a8a9a",
    "sklearn_hist_gbm": "#1f4e79",
    "xgboost_hist": "#b35c1e",
    "lightgbm": "#5c4d7a",
}
MARKERS = {
    "pgml_exact": "D",
    "pgml_hist": "*",
    "sklearn_gbm": "s",
    "sklearn_hist_gbm": "o",
    "xgboost_hist": "^",
    "lightgbm": "v",
}
LABELS = {
    "pgml_exact": "pgml exact",
    "pgml_hist": "pgml hist (32 bins)",
    "sklearn_gbm": "sklearn GBM",
    "sklearn_hist_gbm": "sklearn HistGBM",
    "xgboost_hist": "XGBoost hist",
    "lightgbm": "LightGBM",
}


def load():
    return json.loads(SRC.read_text(encoding="utf-8"))


def valid(points):
    return [
        p for p in points
        if "error" not in p and not p.get("skipped") and "train_ms" in p and "quality" in p
    ]


def plot(data):
    pts = valid(data["points"])
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), layout="constrained")

    for ax, task, ylab, title in (
        (axes[0], "binary", "holdout accuracy", "A  Binary classification"),
        (axes[1], "regression", "holdout $R^2$", "B  Regression"),
    ):
        sub = [p for p in pts if p["task"] == task]
        # draw parity-ish reference: none; just scatter
        seen = set()
        for p in sub:
            sys = p["system"]
            q = p["quality"]["primary"]
            t = max(p["train_ms"], 0.1)
            ms = 40 + 18 * np.log10(max(p["n"], 10) / 1000)
            # bins encoded as edge width for hist systems
            bins = p.get("bins")
            mew = 1.2
            if bins == 32:
                mew = 1.0
            elif bins == 64:
                mew = 1.6
            elif bins == 255:
                mew = 2.2
            ax.scatter(
                t, q,
                s=ms,
                c=COLORS[sys],
                marker=MARKERS[sys],
                alpha=0.85,
                edgecolors="#222",
                linewidths=mew if sys.endswith("hist") or sys == "lightgbm" or sys == "sklearn_hist_gbm" else 0.6,
                label=LABELS[sys] if sys not in seen else None,
                zorder=3 if sys.startswith("pgml") else 2,
            )
            seen.add(sys)
        ax.set_xscale("log")
        ax.set_xlabel("training time (ms)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(True, which="both", ls=":", alpha=0.45)
        ax.legend(fontsize=7.5, loc="best", framealpha=0.92)

    fig.suptitle(
        f"Histogram GBM comparison — equal tree budget "
        f"({data['hyperparams']['n_trees']} trees, depth "
        f"{data['hyperparams']['max_depth']}, lr "
        f"{data['hyperparams']['learning_rate']}). "
        f"Marker size ∼ log $n$; thicker edge ∼ more bins.",
        fontsize=10,
    )

    png = OUT_DIR / "hist_gbm_quality_time.png"
    pdf = OUT_DIR / "hist_gbm_quality_time.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}")
    print(f"wrote {pdf}")

    # Secondary: train time bars at a representative cell (binary, n=20k, d=64, bins=32/255)
    fig2, ax = plt.subplots(figsize=(8.2, 4.0), layout="constrained")
    cell = [p for p in pts if p["task"] == "binary" and p["n"] == 20000 and p["d"] == 64]
    # prefer bins=32 for hist, and exact rows
    pick = []
    for sys in ("pgml_exact", "pgml_hist", "sklearn_gbm", "sklearn_hist_gbm", "xgboost_hist", "lightgbm"):
        cands = [p for p in cell if p["system"] == sys]
        if not cands:
            continue
        if sys in ("pgml_hist",):
            cands = [p for p in cands if p.get("bins") == 32] or cands
        elif sys in ("sklearn_hist_gbm", "xgboost_hist", "lightgbm"):
            # show both 32 and 255 if present
            for b in (32, 255):
                hit = next((p for p in cands if p.get("bins") == b), None)
                if hit:
                    pick.append(hit)
            continue
        pick.append(cands[0])

    labels, times, mems, colors = [], [], [], []
    for p in pick:
        b = p.get("bins")
        lab = LABELS[p["system"]]
        if b is not None and p["system"] != "pgml_hist":
            lab = f"{lab}\n({b} bins)"
        labels.append(lab)
        times.append(p["train_ms"])
        if p.get("memory_metric") == "process_rss_MB" or (
            "rss_delta_mb" in p and not str(p.get("system", "")).startswith("pgml")
        ):
            mems.append(max(p.get("rss_delta_mb") or 0.0, 0.0))
        else:
            mems.append(None)  # pgml contexts — not plotted as RSS
        colors.append(COLORS[p["system"]])

    x = np.arange(len(labels))
    bars = ax.bar(x, times, color=colors, edgecolor="#222", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("training time (ms)")
    ax.set_title(r"Training time @ binary $n{=}2{\cdot}10^4$, $d{=}64$ (log scale)")
    ax.grid(True, axis="y", ls=":", alpha=0.45)
    for i, (t, m) in enumerate(zip(times, mems)):
        if m is None:
            ax.text(i, t * 1.15, f"{t:.0f} ms\n(contexts)", ha="center", va="bottom", fontsize=7)
        else:
            ax.text(i, t * 1.15, f"{t:.0f} ms\nRSS {m:.0f} MB", ha="center", va="bottom", fontsize=7)

    png2 = OUT_DIR / "hist_gbm_time_memory.png"
    pdf2 = OUT_DIR / "hist_gbm_time_memory.pdf"
    fig2.savefig(png2, dpi=180, bbox_inches="tight")
    fig2.savefig(pdf2, bbox_inches="tight")
    plt.close(fig2)
    print(f"wrote {png2}")
    print(f"wrote {pdf2}")


if __name__ == "__main__":
    plot(load())
