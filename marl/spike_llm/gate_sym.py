"""THROWAWAY SPIKE. Can phi see anything at all with symmetric agents? Rule fixed before the gate run.

Run: train_sym.py, csv_narrow, seed 901 (never used for a result), team=mean, lr 3e-5, 30 steps, decay horizon 400.
This only tests the instrument. It can't show that the agents diverge, or that phi leads anything.

From phi.jsonl over steps 5-29:
  W  mean within-agent half-batch cosine (how much an agent agrees with itself)
  C  mean cross-agent half-batch cosine (how much the two agents agree)
  G1 shared direction:  C >= 0.5 * W and C >= 0.05. In the D3 runs C was 0.00-0.01 against W of 0.04-0.28.
  G2 room above noise:  if the agents split completely, phi moves from about (1 - W) / 2 up to 0.5, a swing of
                        W / 2. The SD of phi's trailing 10-step mean over steps 14-29 must be at most a third of it.
PASS on both: the instrument could see a full divergence. Next would be a training test with its own fixed rule,
which needs Aarav's go-ahead. FAIL on either: phi can't work at this scale. MARL stays closed.

    python gate_sym.py ../runs/symgate/out/sym_mean_s901
"""
import json
import sys
from pathlib import Path

import numpy as np

LO, HI = 5, 30
SMOOTH = 10


def gate(phi_rows):
    w = np.array([r["within"] for r in phi_rows[LO:HI]])
    c = np.array([r["cross"] for r in phi_rows[LO:HI]])
    p = np.array([r["phi"] for r in phi_rows])
    W, C = float(w.mean()), float(c.mean())
    trail = np.array([p[max(0, i - SMOOTH + 1):i + 1].mean() for i in range(len(p))])
    sd = float(trail[LO + SMOOTH - 1:HI].std(ddof=1))
    swing = W / 2
    g1 = C >= 0.5 * W and C >= 0.05
    g2 = swing > 0 and sd <= swing / 3
    return dict(W=W, C=C, phi_mean=float(p[LO:HI].mean()), sd_smoothed_phi=sd, swing=swing, G1=bool(g1), G2=bool(g2),
                verdict="PASS" if g1 and g2 else "FAIL")


def main():
    d = Path(sys.argv[1])
    rows = [json.loads(x) for x in (d / "phi.jsonl").read_text().splitlines() if x.strip()]
    if len(rows) < HI:
        raise SystemExit(f"only {len(rows)} steps, needs {HI}")
    r = gate(rows)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}))
    print(f"GATE: {r['verdict']}  (G1 shared direction {'yes' if r['G1'] else 'NO'}: C {r['C']:.3f} against W {r['W']:.3f};"
          f" G2 room above noise {'yes' if r['G2'] else 'NO'}: smoothed SD {r['sd_smoothed_phi']:.4f} against swing {r['swing']:.4f})")


if __name__ == "__main__":
    main()
