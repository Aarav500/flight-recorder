"""Schematic of Proposition 1: a hack/benign twin pair with IDENTICAL rollout geometry R
(top) and divergent held-out oracle O (bottom). A detector sees only the top panel and
must return one verdict for both columns; the only signal that separates them is the
bottom panel, which an oracle-blind detector cannot observe. Illustrative (not a trained
run); the 18-run batch (Figure 1) supplies the real-data evidence that such twins occur."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
t = np.arange(0, 120)
onset = 60.0


def sig(x, x0, k):
    return 1.0 / (1.0 + np.exp(-(x - x0) / k))

# shared geometry R (identical for both twins) -----------------------------------------
conv = sig(t, onset, 7.0)                      # convergence progress 0->1
kl_accel = np.gradient(np.gradient(conv)) * 60 + 0.02 * np.sin(t / 3)   # curvature spikes at onset
entropy = 1.9 - 1.45 * conv                    # entropy collapse
adv_drift = 1.2 * conv * (1 - 0.3 * conv)      # advantage drift rises

# divergent oracle O (the only difference) ---------------------------------------------
oracle_benign = 0.18 + 0.72 * conv             # true quality rises
oracle_hack = 0.18 - 0.13 * conv               # true quality falls while proxy reward rises

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.6, 4.4), sharex=True,
                               gridspec_kw={"height_ratios": [1.15, 1.0]})

ax1.plot(t, kl_accel, color="#1f77b4", lw=1.8, label="KL acceleration")
ax1.plot(t, entropy, color="#2ca02c", lw=1.8, label="entropy")
ax1.plot(t, adv_drift, color="#d62728", lw=1.8, label="advantage drift")
ax1.axvline(onset, color="0.6", lw=0.8, ls="--")
ax1.set_ylabel("rollout geometry $R$", fontsize=9)
ax1.set_title("Identical geometry, divergent oracle: one twin pair (schematic)", fontsize=10)
ax1.legend(fontsize=7.2, loc="center right", ncol=1, framealpha=0.9)
ax1.text(2, ax1.get_ylim()[1] * 0.93, "the detector sees ONLY this panel",
         fontsize=8, style="italic", color="0.25")

ax2.plot(t, oracle_benign, color="#1f77b4", lw=2.0, label="benign twin: oracle rises (true quality up)")
ax2.plot(t, oracle_hack, color="#d62728", lw=2.0, ls=(0, (5, 2)),
         label="hack twin: oracle falls (proxy up, quality down)")
ax2.axvline(onset, color="0.6", lw=0.8, ls="--")
ax2.fill_between(t, oracle_hack, oracle_benign, where=(t >= onset), color="0.9", zorder=0)
ax2.set_ylabel("held-out oracle $O$", fontsize=9)
ax2.set_xlabel("training step", fontsize=9)
ax2.legend(fontsize=7.2, loc="center left", framealpha=0.9)
ax2.text(62, 0.04, "the only signal that separates the twins\n(unavailable to an oracle-blind detector)",
         fontsize=7.6, style="italic", color="0.25")
ax2.set_ylim(-0.02, 1.0)

fig.tight_layout()
fig.savefig(ROOT / "docs" / "fig_twin.pdf")
fig.savefig(ROOT / "docs" / "fig_twin.png", dpi=150)
print("wrote docs/fig_twin.pdf and .png")
