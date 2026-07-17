"""Overlay figure: real benign GRPO runs (no hack possible) placed on the same
warmup-normalized peak-CUSUM axis as the strength-swept synthetic onsets (the crossover).
The point: benign runs reach S values that interleave with / exceed real onsets, so no
threshold on S separates benign convergence from a hack. Pure offline replay."""
from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_batch import replay  # noqa: E402
from flightrecorder.config import load_thresholds  # noqa: E402

# Synthetic onset strength -> peak CUSUM S (the crossover sweep, from the paper).
SYNTH = [(0.02, 27.6), (0.024, 37.8), (0.04, 78.5), (0.10, 231.8)]
TASK_LABEL = {"complex": "lus2", "gentle": "lus2", "maxrun": "maxrun", "secondmax": "secondmax"}
TASK_ROW = {"lus2": 3, "maxrun": 2, "secondmax": 1}      # benign rows; synthetic = row 0
TASK_COLOR = {"lus2": "#1f77b4", "maxrun": "#2ca02c", "secondmax": "#d62728"}


def task_of(name):
    for k in ("complex", "gentle", "maxrun", "secondmax"):
        if name.startswith(k):
            return TASK_LABEL[k]
    return "?"


def collect():
    thr = load_thresholds(str(ROOT / "configs" / "thresholds.yaml"))
    # the 18-run batch (complex_seed0 is the batch equivalent of the original gentle_seed0)
    paths = sorted(glob.glob(str(ROOT / "runs" / "batch" / "*.jsonl")))
    runs = []
    for p in paths:
        name = Path(p).stem
        onset, S, n = replay(p, thr)
        if n < 50:
            continue
        runs.append((task_of(name), name, onset, S))
    return runs


def main():
    runs = collect()
    import statistics
    fires = [(t, nm, o, s) for t, nm, o, s in runs if o is not None]
    s_all = sorted(s for _, _, _, s in runs)
    print(f"{len(runs)} runs | {len(fires)} fire (false positives) | "
          f"peak_S range [{s_all[0]:.1f}, {s_all[-1]:.1f}] median {statistics.median(s_all):.1f}")
    for t, nm, o, s in sorted(runs):
        print(f"  {nm:22s} task={t:9s} onset={str(o):>4s} peak_S={s:7.2f}")

    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    s_vals = sorted(s for _, _, _, s in runs)
    fires = [s for _, _, o, s in runs if o is not None]
    # candidate threshold = peak CUSUM of an s=0.04 synthetic onset (S=78.5). To catch that
    # hack you must set h below it; every benign run to its RIGHT is then a false positive.
    thr_S = 78.5
    n_fp = sum(1 for s in s_vals if s >= thr_S)
    # "indistinguishable zone": from the weakest synthetic onset to the loudest benign run
    lo, hi = min(s for _, s in SYNTH), max(s_vals)
    ax.axvspan(lo, hi, color="#fde0e0", zorder=0)
    ax.axvline(thr_S, color="0.35", lw=1.0, ls="--", zorder=1)
    ax.text(thr_S, 3.74, f"  threshold to catch an s=0.04 hack\n  -> {n_fp}/{len(s_vals)} benign runs false-positive",
            fontsize=7.0, color="0.2", va="top")

    # synthetic onsets (row 0)
    for s_strength, S in SYNTH:
        ax.scatter(S, 0, marker="D", s=42, color="black", zorder=3)
        ax.annotate(f"s={s_strength:g}", (S, 0), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7, color="black")

    # benign runs (rows 1-3), jittered slightly; filled=FIRES, open=quiet
    import numpy as np
    rng = np.random.default_rng(0)
    for t, nm, onset, S in runs:
        row = TASK_ROW[t]; y = row + (rng.random() - 0.5) * 0.28
        filled = onset is not None
        ax.scatter(S, y, s=46, color=TASK_COLOR[t], zorder=3,
                   facecolors=TASK_COLOR[t] if filled else "none",
                   edgecolors=TASK_COLOR[t], linewidths=1.4)

    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["synthetic\nonset", "secondmax", "maxrun", "lus2"], fontsize=8)
    ax.set_ylim(-0.7, 4.2)
    ax.set_xscale("log")
    ax.set_xlim(6, 320)
    from matplotlib.ticker import FixedLocator, FixedFormatter
    ax.xaxis.set_major_locator(FixedLocator([10, 30, 100, 300]))
    ax.xaxis.set_major_formatter(FixedFormatter(["10", "30", "100", "300"]))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_xlabel("peak warmup-normalized CUSUM  $S$  (detector statistic)", fontsize=9)
    ax.set_title("Benign runs (no hack possible) overlap real onsets on the detector's own axis",
                 fontsize=9.5)
    # legend below the axes so it never covers data points
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker="o", color="0.3", lw=0, label="benign, detector fires (false positive)",
                  markerfacecolor="0.3", markersize=7),
           Line2D([0], [0], marker="o", color="0.3", lw=0, label="benign, detector quiet",
                  markerfacecolor="none", markersize=7),
           Line2D([0], [0], marker="D", color="black", lw=0, label="synthetic onset (strength s)",
                  markersize=6)]
    ax.legend(handles=leg, fontsize=6.8, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=3, framealpha=0.9, columnspacing=1.0, handletextpad=0.3)
    ax.grid(axis="x", color="0.9", lw=0.5)
    fig.tight_layout()
    out_pdf = ROOT / "docs" / "fig_benign_overlay.pdf"
    out_png = ROOT / "docs" / "fig_benign_overlay.png"
    fig.savefig(out_pdf); fig.savefig(out_png, dpi=150)
    print("wrote", out_pdf.name, "and", out_png.name)


if __name__ == "__main__":
    main()
