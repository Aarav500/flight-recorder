"""Offline cross-detector check on the persisted benign run (runs/gentle_seed0.jsonl).

Question: is the benign false positive an artifact of the ContractionTube's specific
construction (robust MAD tube + Page CUSUM, two-of-N), or a property of the geometry
itself? We replay the SAME persisted, warmup-normalized kl_accel through two detectors
from structurally different families and report the onset step each declares.

  - EWMA control chart (Roberts, 1959): exponentially-weighted control-chart family.
  - Sliding-window Mann-Whitney U (nonparametric rank-based two-sample test): a
    distributional-shift family, no changepoint/control-chart machinery.

Both are warmup-normalized to the run's own first W steps, exactly like the tube/CUSUM,
so the comparison is scale-fair. No GPU, no oracle: pure offline replay.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

W = 30          # warmup window (matches the pre-registered detector)
P = 5           # persistence: sustained for P steps (matches two-of-N persistence)


def load_kl_accel(run_path: Path) -> tuple[list[int], np.ndarray]:
    steps, vals = [], []
    for line in run_path.read_text().splitlines():
        e = json.loads(line)
        if e["kind"] == "frame":
            steps.append(e["payload"]["step"])
            vals.append(e["payload"]["kl_accel"])
    return steps, np.asarray(vals, float)


def ewma_control_chart(steps, x, lam=0.3, L=3.0):
    """Warmup-standardized kl_accel through an EWMA chart. Report onset under several
    persistence rules and the run-length structure, so the reporting is transparent."""
    mu, sd = x[:W].mean(), x[:W].std() + 1e-9
    limit = L * math.sqrt(lam / (2 - lam))      # steady-state EWMA limit for unit-var input
    g, peak = 0.0, 0.0
    first_cross = None
    above = []           # post-warmup booleans
    onset = {1: None, 3: None, 5: None}
    dwell = 0
    for i, (s, xi) in enumerate(zip(steps, x)):
        z = (xi - mu) / sd
        g = lam * z + (1 - lam) * g
        peak = max(peak, g)
        if i < W:
            continue
        hit = g > limit
        above.append(hit)
        if hit and first_cross is None:
            first_cross = s
        dwell = dwell + 1 if hit else 0
        for pp in onset:
            if dwell >= pp and onset[pp] is None:
                onset[pp] = s
    # longest sustained run above the limit
    longest, cur = 0, 0
    for h in above:
        cur = cur + 1 if h else 0
        longest = max(longest, cur)
    frac = sum(above) / len(above) if above else 0.0
    return {"limit": limit, "peak": peak, "first_cross": first_cross,
            "onset": onset, "longest_run": longest, "frac_above": frac}


def mannwhitney_window(steps, x, w=10):
    """Sliding recent window vs the warmup window; report min p and onset at two
    significance levels (sustained P steps)."""
    warm = x[:W]
    min_p = 1.0
    dwell = {0.05: 0, 0.01: 0}
    onset = {0.05: None, 0.01: None}
    for i, s in enumerate(steps):
        if i < W:
            continue
        window = x[max(0, i - w + 1): i + 1]
        try:
            _, p = mannwhitneyu(window, warm, alternative="greater")
        except ValueError:
            p = 1.0
        min_p = min(min_p, p)
        for a in dwell:
            dwell[a] = dwell[a] + 1 if p < a else 0
            if dwell[a] >= P and onset[a] is None:
                onset[a] = s
    return {"min_p": min_p, "onset": onset}


def main():
    run = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/gentle_seed0.jsonl")
    steps, kl = load_kl_accel(run)
    print(f"run={run.name}  frames={len(steps)}  steps[{steps[0]}..{steps[-1]}]  W={W} P={P}")
    print(f"  ContractionTube (MAD tube + Page CUSUM, two-of-N): onset step 94  [reference]")

    e = ewma_control_chart(steps, kl)
    print(f"  EWMA control chart (Roberts 1959):")
    print(f"     limit(3-sigma)={e['limit']:.3f}  peak EWMA={e['peak']:.3f}"
          f"  ({e['peak']/e['limit']*3:.1f}-sigma_g)  first 3-sigma crossing=step {e['first_cross']}")
    print(f"     frac of post-warmup steps out-of-control={e['frac_above']:.2f}"
          f"  longest sustained run={e['longest_run']} steps")
    print(f"     onset @ persistence P=1/3/5: {e['onset'][1]} / {e['onset'][3]} / {e['onset'][5]}")

    m = mannwhitney_window(steps, kl)
    print(f"  Mann-Whitney sliding two-sample (nonparametric): min p={m['min_p']:.2e}")
    print(f"     onset @ alpha=0.05 / 0.01 (sustained P=5): {m['onset'][0.05]} / {m['onset'][0.01]}")


if __name__ == "__main__":
    main()
