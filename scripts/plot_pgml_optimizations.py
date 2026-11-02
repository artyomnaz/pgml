#!/usr/bin/env python3
"""Impact of native pgml optimizations — figure from results/pgml_optimizations_v2.

Falls back to documenting missing cells; never invents timings.
Legacy single-run sources are not mixed into the plotted medians.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "results" / "pgml_optimizations_v2"
OUT = ROOT / "figures"
PAPER = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
PAPER.mkdir(parents=True, exist_ok=True)

C_OPT = "#0072B2"
C_BASE = "#7A7A7A"
C_IDEAL = "#BBBBBB"
C_TIMEOUT = "#000000"


def setup_style():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.6,
            "grid.linewidth": 0.4,
            "grid.alpha": 0.35,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "mathtext.fontset": "cm",
        }
    )


def load_runs():
    path = V2 / "runs.jsonl"
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def median_iqr(vals):
    a = np.asarray(vals, dtype=float)
    if a.size == 0:
        return None, None, None
    return float(np.median(a)), float(np.percentile(a, 25)), float(np.percentile(a, 75))


def panel_a_series(rows):
    times = defaultdict(list)
    acc = defaultdict(list)
    timeouts = defaultdict(int)
    for r in rows:
        if r.get("panel") != "a_histogram" or r.get("phase") != "measured":
            continue
        key = (int(r["n_train"]), int(r["d"]), r["variant"])
        if r.get("status") == "ok" and "train_ms" in r:
            times[key].append(float(r["train_ms"]))
            acc[key].append(float(r["quality"]["accuracy"]))
        elif r.get("status") == "timeout":
            timeouts[key] += 1
    return times, acc, timeouts


def panel_b_series(rows):
    times = defaultdict(list)
    agrees = defaultdict(list)
    for r in rows:
        if r.get("panel") == "b_dbscan" and r.get("phase") == "measured" and r.get("status") == "ok":
            times[(int(r["n"]), r["variant"])].append(float(r["train_ms"]))
        if r.get("panel") == "b_dbscan_agreement" and r.get("status") == "ok":
            agrees[int(r["n"])].append(
                (float(r["ari_between_implementations"]), float(r["noise_mask_agreement"]))
            )
    return times, agrees


def panel_c_series(rows):
    times = defaultdict(list)
    for r in rows:
        if r.get("panel") == "c_kmeans" and r.get("phase") == "measured" and r.get("status") == "ok":
            times[(int(r["n"]), int(r["threads"]))].append(float(r["train_ms"]))
    return times


def errorbar_line(ax, xs, meds, q1s, q3s, color, marker, ls, label):
    yerr = np.vstack([np.asarray(meds) - np.asarray(q1s), np.asarray(q3s) - np.asarray(meds)])
    ax.errorbar(
        xs,
        meds,
        yerr=yerr,
        color=color,
        marker=marker,
        linestyle=ls,
        linewidth=1.4,
        markersize=5.5,
        capsize=2.5,
        elinewidth=0.8,
        label=label,
    )


def plot_a_on_axes(axes, rows):
    times, acc, timeouts = panel_a_series(rows)
    markers = {16: "o", 64: "s"}
    dacc_lines = []
    for ax, d in zip(axes, (16, 64)):
        for variant, color, marker, ls, label in (
            ("exact", C_BASE, markers[d], "-", "exact"),
            ("histogram", C_OPT, markers[d], "-", "histogram"),
        ):
            ns, meds, q1s, q3s = [], [], [], []
            for n in sorted({k[0] for k in times if k[1] == d and k[2] == variant}):
                m, q1, q3 = median_iqr(times[(n, d, variant)])
                if m is None:
                    continue
                ns.append(n)
                meds.append(m / 1000.0)
                q1s.append(q1 / 1000.0)
                q3s.append(q3 / 1000.0)
            if ns:
                errorbar_line(ax, ns, meds, q1s, q3s, color, marker, ls, label)
            # timeout markers: n where exact timed out and no ok median
            if variant == "exact":
                for n in sorted({k[0] for k in timeouts if k[1] == d and k[2] == "exact"}):
                    if not times.get((n, d, "exact")):
                        # place timeout marker at timeout ceiling for visibility note only
                        ax.plot(
                            n,
                            15 * 60,
                            marker="x",
                            color=C_TIMEOUT,
                            markersize=8,
                            linestyle="none",
                            label="exact timeout" if n == sorted({k[0] for k in timeouts if k[1]==d and k[2]=='exact'})[0] else None,
                        )
                        ax.annotate(
                            "timeout",
                            xy=(n, 15 * 60),
                            xytext=(0, -10),
                            textcoords="offset points",
                            ha="center",
                            fontsize=5.5,
                            color=C_TIMEOUT,
                        )

        # speedup annotations where both medians exist
        for n in sorted({k[0] for k in times if k[1] == d}):
            te = times.get((n, d, "exact"))
            th = times.get((n, d, "histogram"))
            ae = acc.get((n, d, "exact"))
            ah = acc.get((n, d, "histogram"))
            if te and th:
                spd = float(np.median(te) / np.median(th))
                y = float(np.median(th)) / 1000.0
                ax.annotate(
                    f"{spd:.0f}×",
                    xy=(n, y),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=6,
                    color=C_OPT,
                )
            elif (not te) and th and timeouts.get((n, d, "exact"), 0) > 0:
                spd_lb = (15 * 60 * 1000) / float(np.median(th))
                y = float(np.median(th)) / 1000.0
                ax.annotate(
                    f">|{spd_lb:.0f}×",
                    xy=(n, y),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=6,
                    color=C_OPT,
                )
            if ae and ah:
                dacc_lines.append(f"d={d} n={n}: {100*(np.median(ah)-np.median(ae)):+.2f}")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$n_{\mathrm{train}}$")
        ax.set_title(f"d={d}", fontsize=8)
        ax.grid(True, which="both", ls=":", alpha=0.35)
        ax.legend(loc="best", frameon=True, fancybox=False, edgecolor="#CCCCCC", fontsize=6)

    axes[0].set_ylabel("train time (s)")
    axes[0].text(-0.18, 1.08, "(a)", transform=axes[0].transAxes, fontsize=10, fontweight="bold", ha="right")
    # compact quality note under left panel
    if dacc_lines:
        axes[1].text(
            0.98,
            0.02,
            "$\\Delta$acc pp (hist$-$exact):\n" + "\n".join(dacc_lines[:6]),
            transform=axes[1].transAxes,
            fontsize=5.5,
            ha="right",
            va="bottom",
            bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="#CCCCCC", lw=0.5),
        )


def plot_b(ax, rows):
    times, agrees = panel_b_series(rows)
    for variant, color, marker, label in (
        ("brute", C_BASE, "s", "brute-force"),
        ("grid", C_OPT, "o", "spatial grid"),
    ):
        ns, meds, q1s, q3s = [], [], [], []
        for n in sorted({k[0] for k in times if k[1] == variant}):
            m, q1, q3 = median_iqr(times[(n, variant)])
            ns.append(n)
            meds.append(m)
            q1s.append(q1)
            q3s.append(q3)
        if ns:
            errorbar_line(ax, ns, meds, q1s, q3s, color, marker, "-", label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"rows $n$")
    ax.set_ylabel("DBSCAN time (ms)")
    ax.grid(True, which="both", ls=":", alpha=0.35)
    ax.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="#CCCCCC")
    ax.text(-0.14, 1.08, "(b)", transform=ax.transAxes, fontsize=10, fontweight="bold", ha="right")

    # annotate largest n speedup
    ns_all = sorted({k[0] for k in times})
    if ns_all:
        n = max(ns_all)
        tb = times.get((n, "brute"))
        tg = times.get((n, "grid"))
        if tb and tg:
            spd = float(np.median(tb) / np.median(tg))
            ax.annotate(
                f"{spd:.2f}× at n={n}",
                xy=(n, float(np.median(tg))),
                xytext=(-40, 14),
                textcoords="offset points",
                fontsize=6.5,
                color=C_OPT,
                arrowprops=dict(arrowstyle="->", color=C_OPT, lw=0.7),
            )
    # agreement note
    if agrees:
        aris = [np.median([a for a, _ in agrees[n]]) for n in agrees]
        noises = [np.median([b for _, b in agrees[n]]) for n in agrees]
        ax.text(
            0.98,
            0.02,
            f"impl. ARI med={np.min(aris):.3f}–{np.max(aris):.3f}\n"
            f"noise agree med≥{np.min(noises):.3f}",
            transform=ax.transAxes,
            fontsize=5.5,
            ha="right",
            va="bottom",
            color="#555555",
        )


def plot_c(ax, rows):
    times = panel_c_series(rows)
    ns = sorted({k[0] for k in times})
    markers = {100000: "o", 500000: "s"}
    for n in ns:
        ps = sorted({k[1] for k in times if k[0] == n})
        t1 = times.get((n, 1))
        if not t1:
            continue
        t1m = float(np.median(t1))
        sp_m, sp_q1, sp_q3, p_plot = [], [], [], []
        for p in ps:
            ts = times[(n, p)]
            # speedup per-repeat vs median(t1)? Use median(t1)/median(tp) with IQR from bootstrap of ratios of medians:
            # Prefer ratios of paired repeats when equal count
            if len(ts) == len(t1) and len(ts) > 0:
                # align by sorting times is wrong; use independent median IQR on speedup samples if we had pairs.
                # Without paired indices, report median speedup = med(t1)/med(tp) and
                # approximate IQR via med(t1)/q3(tp) .. med(t1)/q1(tp) inverted carefully.
                m, q1, q3 = median_iqr(ts)
                spd = t1m / m
                # conservative band from quartile times
                spd_hi = t1m / q1 if q1 > 0 else spd
                spd_lo = t1m / q3 if q3 > 0 else spd
                sp_m.append(spd)
                sp_q1.append(min(spd_lo, spd_hi))
                sp_q3.append(max(spd_lo, spd_hi))
            else:
                m, q1, q3 = median_iqr(ts)
                spd = t1m / m
                sp_m.append(spd)
                sp_q1.append(t1m / q3)
                sp_q3.append(t1m / q1)
            p_plot.append(p)
        errorbar_line(
            ax,
            p_plot,
            sp_m,
            sp_q1,
            sp_q3,
            C_OPT,
            markers.get(n, "o"),
            "-" if n == ns[0] else "--",
            f"n={n//1000}k",
        )
        for p, s in zip(p_plot, sp_m):
            ax.annotate(f"{s:.2f}×", xy=(p, s), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=6, color=C_OPT)

    pmax = 8
    ps_line = np.linspace(1, pmax, 50)
    ax.plot(ps_line, ps_line, color=C_IDEAL, ls=":", lw=1.0, label="ideal")
    ax.set_xticks([1, 2, 4, 8])
    ax.set_xlim(0.8, 8.5)
    ax.set_ylim(0.8, 8.5)
    ax.set_xlabel("threads $p$")
    ax.set_ylabel(r"speedup $T_1/T_p$")
    ax.grid(True, ls=":", alpha=0.35)
    ax.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="#CCCCCC", fontsize=6)
    ax.text(-0.14, 1.08, "(c)", transform=ax.transAxes, fontsize=10, fontweight="bold", ha="right")


def export_csv(rows, path: Path):
    fieldnames = [
        "panel",
        "variant",
        "n_train",
        "n",
        "d",
        "threads",
        "phase",
        "repeat",
        "status",
        "train_ms",
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "speedup_note",
        "source",
    ]
    out = []
    for r in rows:
        if r.get("phase") not in ("measured", "warmup") and r.get("panel") not in (
            "b_dbscan_agreement",
            "c_kmeans_agreement",
            "a_pilot",
        ):
            continue
        q = r.get("quality") or {}
        out.append(
            {
                "panel": r.get("panel"),
                "variant": r.get("variant"),
                "n_train": r.get("n_train", ""),
                "n": r.get("n", ""),
                "d": r.get("d", ""),
                "threads": r.get("threads", ""),
                "phase": r.get("phase", ""),
                "repeat": r.get("repeat", ""),
                "status": r.get("status", ""),
                "train_ms": r.get("train_ms", ""),
                "accuracy": q.get("accuracy", ""),
                "balanced_accuracy": q.get("balanced_accuracy", ""),
                "f1_macro": q.get("f1_macro", ""),
                "speedup_note": r.get("speedup_note", r.get("note", "")),
                "source": "results/pgml_optimizations_v2/runs.jsonl",
            }
        )
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out)


def write_caption(path: Path, rows):
    times_a, acc_a, to_a = panel_a_series(rows)
    bits = []
    for d in (16, 64):
        for n in sorted({k[0] for k in times_a if k[1] == d}):
            ae, ah = acc_a.get((n, d, "exact")), acc_a.get((n, d, "histogram"))
            if ae and ah:
                bits.append(f"$d={d}$, $n_{{\\mathrm{{train}}}}={n}$: ${100*(np.median(ah)-np.median(ae)):+.2f}$ pp")
    dacc = "; ".join(bits) if bits else "see report"
    text = r"""% Written by scripts/plot_pgml_optimizations.py from results/pgml_optimizations_v2
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{paper/figures/pgml_optimizations_v2.pdf}
  \caption{Impact of three native optimizations in \textsc{pgml}
  (remeasured series: one warmup and five timed repeats; median with IQR).
  \textbf{(a)}~Binary gradient boosting ($50$ trees, depth~$3$, learning rate~$0.1$):
  SQL-side training time for sort-based exact search versus fixed-bin histogram
  search ($B{=}32$) versus $n_{\mathrm{train}}$ at $d{=}16$ and $d{=}64$.
  Packing is excluded from the timer. Annotated factors are
  $\mathrm{median}(T_{\mathrm{exact}})/\mathrm{median}(T_{\mathrm{hist}})$;
  a ``$>|$'' mark denotes a lower bound when exact fits hit the 15-minute
  statement timeout. Hold-out accuracy change (hist$-$exact): """ + dacc + r""".
  \textbf{(b)}~DBSCAN wall time for brute-force versus spatial-grid search on
  blob data ($\varepsilon{=}0.5$, $\mathrm{minPts}{=}5$). The factor at the
  largest completed $n$ is annotated; inter-implementation ARI and noise-mask
  agreement are summarized in-panel (not ground-truth quality).
  \textbf{(c)}~k-means assignment speedup $T_1/T_p$ versus thread count for
  pre-materialized inputs ($d{=}20$, $k{=}8$, 20 fixed iterations; same seed for
  k-means++ initialization), with an ideal linear reference.}
  \label{fig:pgml-optimizations-v2}
\end{figure*}
"""
    path.write_text(text, encoding="utf-8")


def write_audit(path: Path, rows):
    n_ok = sum(1 for r in rows if r.get("phase") == "measured" and r.get("status") == "ok")
    n_to = sum(1 for r in rows if r.get("status") == "timeout")
    text = f"""# pgml_optimizations_v2 figure audit

Source directory: `results/pgml_optimizations_v2/`
Runner: `experiments/systems/optimizations/run_experiment.py`

## Protocol
- 1 warmup + 5 measured repeats; alternating variant order
- SQL-side kernel timing only (pack outside timer)
- Fit timeout 15 minutes (panel a)
- Medians and IQR from completed measured repeats only

## Record counts
- total JSONL records: {len(rows)}
- measured ok: {n_ok}
- timeouts: {n_to}

## Panel notes
- (a) uses explicit `n_train` (test = n_train//3). Legacy `hist_gbm_comparison.json`
  used total-n before 75/25 split and is not mixed into plotted medians.
- (a) exact path is `pgml_train_gbm` → `train_gbm` (no silent hist switch).
- (b) agreement metrics are between implementations, not ground truth.
- (c) uses corrected parallel API; old saturated `mt_speedup.png` numbers excluded.
- (c) centers not injectable; shared seed for k-means++ init.

## Reproduce
```
python experiments/systems/optimizations/run_experiment.py
python scripts/plot_pgml_optimizations.py
```
"""
    path.write_text(text, encoding="utf-8")


def main():
    setup_style()
    rows = load_runs()
    if not rows:
        raise SystemExit(
            "No results/pgml_optimizations_v2/runs.jsonl yet. "
            "Run experiments/systems/optimizations/run_experiment.py first."
        )

    # Layout: (a) two subpanels share left third? Better: top-level 1x3 where (a) is itself 1x2.
    fig = plt.figure(figsize=(7.2, 2.9), layout="constrained")
    sub = fig.subfigures(1, 3, width_ratios=[1.35, 1.0, 0.95], wspace=0.03)
    axes_a = sub[0].subplots(1, 2, sharey=True)
    ax_b = sub[1].subplots(1, 1)
    ax_c = sub[2].subplots(1, 1)

    plot_a_on_axes(axes_a, rows)
    plot_b(ax_b, rows)
    plot_c(ax_c, rows)

    export_csv(rows, OUT / "pgml_optimizations_v2_data.csv")
    write_caption(OUT / "pgml_optimizations_v2_caption.tex", rows)
    write_audit(OUT / "pgml_optimizations_v2_audit.md", rows)

    for name in ("pgml_optimizations_v2.pdf", "pgml_optimizations_v2.svg", "pgml_optimizations_v2.png"):
        dest = OUT / name
        if name.endswith(".png"):
            fig.savefig(dest, dpi=300, bbox_inches="tight")
        else:
            fig.savefig(dest, bbox_inches="tight")
        (PAPER / name).write_bytes(dest.read_bytes())

    # also refresh canonical names used by prior caption if desired
    for name in ("pgml_optimizations.pdf", "pgml_optimizations.svg", "pgml_optimizations.png"):
        src = OUT / name.replace("pgml_optimizations", "pgml_optimizations_v2")
        if src.exists():
            (OUT / name).write_bytes(src.read_bytes())
            (PAPER / name).write_bytes(src.read_bytes())

    plt.close(fig)
    print(f"wrote figures from {len(rows)} run records")


if __name__ == "__main__":
    main()
