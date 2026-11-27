"""THROWAWAY SPIKE. Do more samples per step, or gradients averaged over steps, lift phi above its noise? Fixed before
the runs. Both runs: train_scale.py, symmetric agents, csv_narrow, team=mean, lr 3e-5, 30 steps, decay horizon 400.

  m8_ema  (seed 902)  8 samples per step; reads the running averages (emaB_*). EMA needs warm-up, so steps 10-29,
                      no extra smoothing; G2 uses the SD of emaB_phi over steps 15-29.
  m64     (seed 903)  64 samples per step; reads per-step gradients (B_*) over steps 5-29; G2 uses the SD of the
                      trailing 10-step mean of B_phi over steps 14-29.
Primary measure is the lora_B form, decided before the runs (see train_scale.py). The all-weights form is printed
alongside and decides nothing.

  G1 shared direction:  C >= 0.5 * W and C >= 0.05
  G2 room above noise:  the SD above <= (W / 2) / 3, a third of the swing phi would make if the agents split
PASS for a run: G1 and G2. Either run passing makes that configuration the one to take to a training test, which
needs Aarav's go-ahead and its own fixed rule. Both failing: phi is noise-bound at every scale this budget reaches
(more samples per step and averaging over steps both tried), and MARL stays closed.

    python gate_scale.py ../runs/scalegate/out
"""
import json
import sys
from pathlib import Path

import numpy as np

CONFIGS = {
    "m8_ema_s902": dict(prefix="emaB_", lo=10, hi=30, smooth=1, sd_from=15, alt="ema_"),
    "m64_s903": dict(prefix="B_", lo=5, hi=30, smooth=10, sd_from=14, alt=""),
}


def gate(rows, prefix, lo, hi, smooth, sd_from):
    w = np.array([r[prefix + "within"] for r in rows[lo:hi]])
    c = np.array([r[prefix + "cross"] for r in rows[lo:hi]])
    p = np.array([r[prefix + "phi"] for r in rows])
    W, C = float(w.mean()), float(c.mean())
    trail = np.array([p[max(0, i - smooth + 1):i + 1].mean() for i in range(len(p))])
    sd = float(trail[sd_from:hi].std(ddof=1))
    swing = W / 2
    g1 = C >= 0.5 * W and C >= 0.05
    g2 = swing > 0 and sd <= swing / 3
    return dict(W=W, C=C, phi=float(p[lo:hi].mean()), sd=sd, swing=swing, G1=bool(g1), G2=bool(g2),
                verdict="PASS" if g1 and g2 else "FAIL")


def main():
    root = Path(sys.argv[1])
    passed = []
    for name, c in CONFIGS.items():
        f = root / name / "phi.jsonl"
        rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        if len(rows) < c["hi"]:
            raise SystemExit(f"{name}: only {len(rows)} steps")
        r = gate(rows, c["prefix"], c["lo"], c["hi"], c["smooth"], c["sd_from"])
        alt = gate(rows, c["alt"], c["lo"], c["hi"], c["smooth"], c["sd_from"])
        print(f"{name:12s} {r['verdict']}  B weights: W {r['W']:+.3f} C {r['C']:+.3f} phi {r['phi']:.3f} "
              f"SD {r['sd']:.4f} swing {r['swing']:.4f} G1 {r['G1']} G2 {r['G2']}")
        print(f"{'':12s} (all weights, decides nothing: W {alt['W']:+.3f} C {alt['C']:+.3f} phi {alt['phi']:.3f} "
              f"SD {alt['sd']:.4f} {alt['verdict']})")
        if r["verdict"] == "PASS":
            passed.append(name)
    print(f"\nGATE: {'PASS via ' + ', '.join(passed) if passed else 'FAIL, both configurations'}")


if __name__ == "__main__":
    main()
