"""Plot regime map from results/regime_map.json."""
from __future__ import annotations

import json
from pathlib import Path
_REPO = Path(__file__).resolve().parents[2]

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

ROOT = Path(_REPO)
SRC = ROOT / "results" / "regime_map.json"
OUT_DIR = ROOT / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VERDICT_EDGE = {
    "faster": "#1b7f4a",
    "comparable": "#8a6d1d",
    "slower": "#6b6b6b",
    "oom": "#9b2226",
}


def load():
    return json.loads(SRC.read_text(encoding="utf-8"))


def speedup_matrix(points):
    """Task×n grid for scale_task axis (d=32 train)."""
    rows = [
        ("binary / tree", "binary_classification", "tree"),
        ("multiclass / tree", "multiclass", "tree"),
        ("regression / linreg", "regression", "linreg"),
        ("clustering / kmeans", "clustering", "kmeans"),
        ("PCA", "pca", "pca"),
    ]
    ns = [1000, 10000, 50000, 100000]
    Z = np.full((len(rows), len(ns)), np.nan)
    V = [[None] * len(ns) for _ in rows]
    for p in points:
        c = p["config"]
        if c.get("axis") != "scale_task":
            continue
        for i, (_, task, model) in enumerate(rows):
            if c["task"] == task and c["model"] == model:
                j = ns.index(c["n"]) if c["n"] in ns else None
                if j is None:
                    continue
                Z[i, j] = p.get("speedup_vs_sklearn") or np.nan
                V[i][j] = p["verdict"]
    return rows, ns, Z, V


def plot(data):
    points = data["points"]
    rows, ns, Z, V = speedup_matrix(points)

    fig = plt.figure(figsize=(10.2, 5.6), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    right = gs[0, 1].subgridspec(2, 1, height_ratios=[1.0, 1.05], hspace=0.35)
    ax1 = fig.add_subplot(right[0, 0])
    ax2 = fig.add_subplot(right[1, 0])

    # --- A: scale × task heatmap (log2 speedup) ---
    # clip for color; NaN stays blank
    Zplot = np.array(Z, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        L = np.log2(Zplot)
    # diverging around parity (speedup=1 → log2=0)
    norm = mcolors.TwoSlopeNorm(vmin=-6, vcenter=0.0, vmax=2)
    cmap = plt.get_cmap("RdYlGn")
    im = ax0.imshow(L, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax0.set_xticks(range(len(ns)))
    ax0.set_xticklabels([f"$10^{{{int(np.log10(n))}}}$" if n in (1000, 10000, 100000)
                         else r"$5\!\cdot\!10^{4}$" for n in ns])
    ax0.set_yticks(range(len(rows)))
    ax0.set_yticklabels([r[0] for r in rows])
    ax0.set_xlabel("rows $n$ (features $d{=}32$)")
    ax0.set_title("A  Speedup vs in-process sklearn\n(train+predict, scale × task)")

    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            verd = V[i][j]
            if verd is None or np.isnan(Z[i, j]):
                ax0.text(j, i, "—", ha="center", va="center", color="#444", fontsize=9)
                continue
            s = Z[i, j]
            label = f"{s:.2f}×"
            if verd == "faster":
                label = f"{s:.2f}× ↑"
            elif verd == "comparable":
                label = f"{s:.2f}× ≈"
            color = "#111" if abs(L[i, j]) < 2.5 else "#f7f7f7"
            ax0.text(j, i, label, ha="center", va="center", fontsize=8.2, color=color)
            # thin verdict border
            rect = plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96,
                                 fill=False, linewidth=1.6 if verd == "faster" else 0.8,
                                 edgecolor=VERDICT_EDGE[verd])
            ax0.add_patch(rect)

    cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\log_2$(pgml speedup)")
    cbar.set_ticks([-6, -4, -2, 0, 1, 2])
    cbar.set_ticklabels(["1/64×", "1/16×", "1/4×", "1×", "2×", "4×"])

    # --- B: batch inference ---
    batch_pts = [p for p in points if p["config"].get("axis") == "batch_inference"]
    batch_pts.sort(key=lambda p: p["config"]["batch"])
    bs = [p["config"]["batch"] for p in batch_pts]
    sp = [p["speedup_vs_sklearn"] for p in batch_pts]
    ax1.plot(bs, sp, "o-", color="#1f4e79", markersize=6, linewidth=1.6)
    ax1.axhline(1.0, color="#888", ls="--", lw=0.9, label="parity")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("batch size $b$")
    ax1.set_ylabel("speedup")
    ax1.set_title("B  Batch predict only\n(tree, $n{=}5\\!\\cdot\\!10^4$)")
    ax1.set_ylim(1e-3, 2)
    ax1.grid(True, which="both", ls=":", alpha=0.45)

    # --- C: model complexity + OOM marker ---
    mc = [p for p in points if p["config"].get("axis") == "model_complexity"]
    order = ["tree", "forest", "gbm", "exact_gbm"]
    labels, vals, verd = [], [], []
    for m in order:
        hit = next((p for p in mc if p["config"]["model"] == m), None)
        if not hit:
            continue
        labels.append(m.replace("_", " "))
        vals.append(hit["speedup_vs_sklearn"] or 0.0)
        verd.append(hit["verdict"])
    # append memory probe as special bar
    oom = next(p for p in points if p["config"].get("axis") == "memory_probe")
    labels.append(r"mem $2{\cdot}10^5{\times}128$")
    vals.append(np.nan)
    verd.append(oom.get("verdict", "materialization_failure"))

    y = np.arange(len(labels))
    colors = []
    for v in verd:
        if v == "faster":
            colors.append("#2a9d5c")
        elif v == "comparable":
            colors.append("#c4a035")
        elif v in ("oom", "materialization_failure"):
            colors.append("#9b2226")
        else:
            colors.append("#7a8a9a")
    plot_vals = [v if (isinstance(v, (int, float)) and v == v) else 1e-3 for v in vals]
    bars = ax2.barh(y, plot_vals, color=colors, height=0.62, edgecolor="#222", linewidth=0.5)
    ax2.axvline(1.0, color="#888", ls="--", lw=0.9)
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels)
    ax2.set_xscale("log")
    ax2.set_xlim(1e-3, 4)
    ax2.set_xlabel("speedup vs sklearn")
    ax2.set_title("C  Model complexity (+ materialization probe)")
    for i, (v, vd) in enumerate(zip(vals, verd)):
        if vd in ("oom", "materialization_failure"):
            ax2.text(0.002, i, "mat. fail", va="center", ha="left", color="white",
                     fontsize=7.5, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.15", fc="#9b2226", ec="none"))
        else:
            ax2.text(max(v, 1e-3) * 1.12, i, f"{v:.2f}×", va="center", fontsize=8)

    # counts strip
    from collections import Counter
    cnt = Counter(p["verdict"] for p in points)
    mat = cnt.get("materialization_failure", 0) + cnt.get("oom", 0)
    fig.suptitle(
        f"Regime map: pgml vs in-process scikit-learn  "
        f"(n={len(points)}; faster {cnt['faster']} · comparable {cnt['comparable']} · "
        f"slower {cnt['slower']} · materialization_failure {mat})",
        fontsize=11, y=1.02,
    )

    png = OUT_DIR / "regime_map.png"
    pdf = OUT_DIR / "regime_map.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    plot(load())
