"""Plot horizontal stacked share-of-time bars for train stage breakdown."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "train_stage_breakdown" / "stages.json"
OUT_PNG = ROOT / "figures" / "train_stage_breakdown.png"
OUT_PDF = ROOT / "figures" / "train_stage_breakdown.pdf"

# Academic palette close to the reference figure
C_SQL = "#9ecae1"
C_CONV = "#bcbddc"
C_NATIVE = "#3182bd"
C_SER = "#fff7bc"

STAGES = [
    ("sql_materialization", "SQL materialization (SELECT, aggregation)", C_SQL),
    ("array_conversion", "Array conversion (SQL arrays → C buffers)", C_CONV),
    ("native_computation", "Native computation (training)", C_NATIVE),
    ("model_serialization", "Model serialization (to bytea)", C_SER),
]

ORDER = [
    "decision_tree",
    "gradient_boosting",
    "linear_svm",
    "kmeans",
    "dbscan",
    "linear_regression",
    "pca",
]


def main():
    data = json.loads(SRC.read_text(encoding="utf-8"))
    by = {p["algorithm"]: p for p in data["points"]}
    rows = [by[k] for k in ORDER if k in by]
    labels = [r["label"] for r in rows]
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=160)
    left = np.zeros(len(rows))
    legend_added = []

    for key, leg, color in STAGES:
        widths = np.array([r["share_pct"][key] for r in rows], dtype=float)
        bars = ax.barh(
            y,
            widths,
            left=left,
            height=0.72,
            color=color,
            edgecolor="#222222",
            linewidth=0.6,
            label=leg,
        )
        legend_added.append(bars)
        for i, (w, l0) in enumerate(zip(widths, left)):
            if w >= 4.5:
                ax.text(
                    l0 + w / 2.0,
                    i,
                    f"{w:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="#111111",
                )
            elif w >= 2.2:
                ax.text(
                    l0 + w / 2.0,
                    i,
                    f"{w:.1f}",
                    ha="center",
                    va="center",
                    fontsize=9.5,
                    color="#111111",
                )
        left = left + widths

    for i, r in enumerate(rows):
        ax.text(
            101.5,
            i,
            f"{r['total_s']:.2f} s",
            ha="left",
            va="center",
            fontsize=11,
            color="#222222",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel("Share of total time", fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.55, len(rows) - 0.45)
    ax.invert_yaxis()
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"], fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    ax.set_title("Execution time breakdown (training)", loc="left", fontsize=14, pad=10)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.48, -0.16),
        ncol=2,
        frameon=False,
        fontsize=11,
        handlelength=1.4,
        columnspacing=1.2,
    )

    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
