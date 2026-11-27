"""THROWAWAY SPIKE, exploratory, written after seeing that the patched arm collapses too.

analyze_phi.py's alarm table compares two arms that both collapse, and its collapse time fires on benign
runs, so it can't say whether phi leads. This asks the narrower question the data can answer: lining up
every run whose true reward really falls on the moment it falls, does a gradient statistic move earlier
than the same statistic moves in benign runs over the same updates?

  t_q     first eval after the peak where the trailing 20-eval mean of true reward has crossed halfway from
          peak to the run's final level. Only runs falling by at least MIN_FALL count.
  change  statistic in a 25-update trailing window ending at t_q + offset, minus the same run's window ending
          at t_q - 300, minus the same difference in benign (control) runs at the same absolute updates.
          Positive means the statistic rose relative to benign runs. Sign test across runs; exploit and
          patched share seeds, so the p values are optimistic.

    python analyze_aligned.py ../runs/jax/results/phi
"""
import glob
import json
import os
import sys
from math import comb

import numpy as np

MIN_FALL = 0.10
OFFSETS = (-200, -150, -100, -50, -25, 0, 25, 50, 100)
BASE = -300
WIN = 25
STATS = ("phi_gae", "phi_mc", "gap_gae", "gap_mc", "gnorm_gae")


def load(root, params="shared"):
    runs = {}
    for f in glob.glob(os.path.join(root, params, "N*_k0.0_s*.json")):
        h = json.load(open(f))
        h["rt_m"] = np.array(h["rt"]).mean(1)
        for a in ("gae", "mc"):
            h["gap_" + a] = np.array(h["within_" + a]) - np.array(h["cross_" + a])
        runs[(h["n"], h["cond"], h["seed"])] = h
    return runs


def t_q(h):
    u, y = np.array(h["update"]), h["rt_m"]
    s = np.convolve(y, np.ones(20) / 20, mode="valid")          # s[i] covers evals i .. i + 19
    pk = int(np.argmax(s))
    late = y[-40:].mean()
    fall = s[pk] - late
    if fall < MIN_FALL:
        return None, fall
    half = late + 0.5 * fall
    for i in range(pk, len(s)):
        if s[i] <= half:
            return int(u[i + 19]), fall
    return None, fall


def window(x, end):
    return float(np.mean(x[max(0, end - WIN + 1):end + 1]))


def sign_p(pos, neg):
    n = pos + neg
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(pos, neg) + 1)) / 2 ** n)


def main():
    runs = load(sys.argv[1])
    for n in sorted({k[0] for k in runs}):
        hack = []
        falls = {}
        for (m, c, s), h in sorted(runs.items()):
            if m != n or c == "control":
                continue
            t, fall = t_q(h)
            falls.setdefault(c, []).append(round(float(fall), 3))
            if t is not None and t + BASE >= WIN:
                hack.append((c, s, h, t))
        ctrl = [h for (m, c, s), h in sorted(runs.items()) if m == n and c == "control"]
        print(f"\n=== N={n}: fall from peak to final, exploit {falls.get('exploit')}")
        print(f"             patched {falls.get('patched')}")
        print(f"control falls {[round(float(t_q(h)[1]), 3) for h in ctrl]}")
        if not hack:
            print(f"no run falls by {MIN_FALL} with {-BASE} updates of history before it")
            continue
        ts = [t for *_, t in hack]
        print(f"{len(hack)} runs fall (exploit {sum(c == 'exploit' for c, *_ in hack)}, "
              f"patched {sum(c == 'patched' for c, *_ in hack)}); t_q median {int(np.median(ts))}, "
              f"range {min(ts)}-{max(ts)}")
        print(f"{'stat':10s} " + " ".join(f"{o:>11d}" for o in OFFSETS))
        for st in STATS:
            cells = []
            for o in OFFSETS:
                d = []
                for c, s, h, t in hack:
                    a, b = t + o, t + BASE
                    if a >= h["num_updates"]:
                        continue
                    run = window(np.array(h[st]), a) - window(np.array(h[st]), b)
                    bg = np.mean([window(np.array(x[st]), a) - window(np.array(x[st]), b) for x in ctrl])
                    d.append(run - bg)
                pos, neg = sum(x > 0 for x in d), sum(x < 0 for x in d)
                cells.append(f"{np.mean(d):+.3f}({pos:2d}/{len(d):2d})" + ("*" if sign_p(pos, neg) < 0.01 else " "))
            print(f"{st:10s} " + " ".join(f"{c:>11s}" for c in cells))
    print("\neach cell: mean change relative to benign (runs where it rose / runs). * = sign test p < 0.01, optimistic")


if __name__ == "__main__":
    main()
