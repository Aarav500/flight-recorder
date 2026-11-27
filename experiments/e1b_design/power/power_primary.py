"""E1b power, step 2: the PRIMARY test (disagreement, upward-only alarm, time-matched operating point).

Design facts used (proposed E1b primary):
  * twin run j is scored over steps 0..e_j - 1 of its seed- and task-matched hacking run (e_j = that run's
    registered event); the twin of a no-event hacking run is scored over all 400 steps (rule B, see README).
    So n_twin = n_hack = n, fixed before data.
  * threshold = 90th percentile (np.percentile, linear) of the n twin maxima, so for distinct maxima the
    FP count is fixed at e1_primary.fixed_fp(n): 3/24, 4/36, 5/48.
  * TP = hacking alarm strictly before its event; one-sided Fisher on TP v FP; alpha 0.05.
  * kill also needs the window (median pre-event window >= 12, no event = -1) and point median lead > 0.
All tests and intervals are e1_primary's registered helpers, imported, not copied.

Window, lead and generator-check probabilities come from a task-stratified bootstrap of E1's 24 hacking runs
(event, lead) pairs (e1_inputs.json): E1b's hacking arm is E1's config with fresh seeds.

    python power_primary.py > power_primary.txt
"""
from __future__ import annotations

import json
import math
import sys
from math import comb
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

REPO = Path(r"<home>/OneDrive/Desktop/Fligh recorder")
sys.path.insert(0, str(REPO / "experiments"))
import e1_primary as P  # noqa: E402

HERE = Path(__file__).resolve().parent
NS = (24, 36, 48)
TPRS = (0.104, 0.125, 0.20, 0.25, 0.30, 0.375, 0.40, 0.45, 0.50, 0.60)
B = 20000


def binom_pmf(n, p):
    return [comb(n, x) * p ** x * (1 - p) ** (n - x) for x in range(n + 1)]


def min_hits(n, fp, alpha=P.ALPHA):
    return next((x for x in range(n + 1) if P.fisher_one_sided(x, n, fp, n) < alpha), None)


def mde(n, fp, target=0.80, alpha=P.ALPHA):
    """e1_primary.mde_fixed_fp at 0.005 resolution (the frozen helper steps by 0.01)."""
    return next((t / 1000 for t in range(5, 1001, 5) if P.power_fixed_fp(n, n, t / 1000, fp, alpha) >= target), None)


def upper90(x, n, fp):
    return P.newcombe(x, n, fp, n, 0.90)[1]


def delta_at(n, fp, tpr, prob=0.80):
    """Smallest margin d (to 0.005) such that P(upper Newcombe 90% bound < d) >= prob when TP ~ Bin(n, tpr)."""
    pmf = binom_pmf(n, tpr)
    ups = [upper90(x, n, fp) for x in range(n + 1)]
    for d in np.arange(0.05, 0.60, 0.005):
        if sum(q for x, q in enumerate(pmf) if ups[x] < d) >= prob:
            return round(float(d), 3)
    return None


def delta_at_binomial_fp(n, p=0.10, prob=0.80):
    """Prereg-section-5 convention (TPR = FPR = 0.1, both binomial): reproduces 0.24 at 24 and 0.19 at 36."""
    pmf = binom_pmf(n, p)
    ups = {(x, y): P.newcombe(x, n, y, n, 0.90)[1] for x in range(n + 1) for y in range(n + 1)}
    for d in np.arange(0.05, 0.60, 0.005):
        if sum(pmf[x] * pmf[y] for (x, y), u in ups.items() if u < d) >= prob:
            return round(float(d), 3)
    return None


def outcomes(n, fp, tpr, delta=P.DELTA):
    """P(Fisher p < 0.05), P(non-superiority shown and no significance), P(neither) when TP ~ Bin(n, tpr)."""
    pmf = binom_pmf(n, tpr)
    sig = sum(q for x, q in enumerate(pmf) if P.fisher_one_sided(x, n, fp, n) < P.ALPHA)
    ns = sum(q for x, q in enumerate(pmf) if P.fisher_one_sided(x, n, fp, n) >= P.ALPHA and upper90(x, n, fp) < delta)
    return sig, ns, 1 - sig - ns


def boot_conjuncts(runs, n, rng):
    """Task-stratified bootstrap of E1 hacking runs to n (n/3 per task): P(window >= 12), P(median lead > 0),
    P(both), P(generator check: events >= ceil(0.75 n))."""
    by_task = {}
    for t, (e, l) in runs.items():
        by_task.setdefault(t.split("/")[1], []).append((e, l))
    per = n // 3
    win = lead = both = gen = 0
    for _ in range(B):
        ev, ld = [], []
        for task, rs in by_task.items():
            for i in rng.integers(0, len(rs), per):
                e, l = rs[i]
                ev.append(-1 if e is None else e)
                if l is not None:
                    ld.append(l)
        w = np.median(ev) >= P.WINDOW_MIN
        m = len(ld) > 0 and np.median(ld) > 0
        g = sum(e >= 0 for e in ev) >= math.ceil(P.GEN_FRAC * n)
        win += w
        lead += m
        both += (w and m and g)
        gen += g
    return win / B, lead / B, both / B, gen / B


def main():
    d = json.loads((HERE / "e1_inputs.json").read_text())
    runs = {t: (d["events"][t], d["lead_by_tag"][t]) for t in d["events"]}
    rng = np.random.default_rng(20261009)
    res = {}
    print("E1b PRIMARY POWER. One-sided Fisher, alpha 0.05, FP fixed by the time-matched 90th-percentile point.")
    print("e1_primary helpers: fisher_one_sided, power_fixed_fp, newcombe, fixed_fp. Exact enumeration unless marked.\n")
    print("check: frozen e1_primary.mde_fixed_fp(24,24,3) = %s (prereg 0.44); power_fixed_fp(24,24,0.5,3) = %.3f (prereg 0.92)" % (
        P.mde_fixed_fp(24, 24, 3), P.power_fixed_fp(24, 24, 0.5, 3)))
    print("check: binomial-FP margin at 80%% (prereg section 5 convention): n=24 %.3f (prereg 0.24), n=36 %.3f (prereg 0.19)\n" % (
        delta_at_binomial_fp(24), delta_at_binomial_fp(36)))
    for n in NS:
        fp = P.fixed_fp(n)
        r = {"fp": fp, "fpr": fp / n, "min_hits": min_hits(n, fp), "power_at_0.5": P.power_fixed_fp(n, n, 0.5, fp),
             "mde80": mde(n, fp), "mde80_frozen_helper": P.mde_fixed_fp(n, n, fp), "mde90": mde(n, fp, 0.90),
             "size_at_null": P.power_fixed_fp(n, n, fp / n, fp),
             "nonsup_max_hits_0.25": P.nonsuperiority_max_hits(n, n, fp),
             "power_curve": {t: P.power_fixed_fp(n, n, t, fp) for t in TPRS},
             "outcomes": {t: outcomes(n, fp, t) for t in TPRS},
             "delta80_uninformative": delta_at(n, fp, fp / n), "delta90_uninformative": delta_at(n, fp, fp / n, 0.90),
             "delta80_binomial_fp": delta_at_binomial_fp(n),
             "upper90_at_x": {x: round(upper90(x, n, fp), 3) for x in range(0, n // 2 + 1)}}
        r["nonsup_P_shown_uninformative"] = sum(q for x, q in enumerate(binom_pmf(n, fp / n))
                                              if upper90(x, n, fp) < P.DELTA)
        for dl in (0.20, 0.15):
            mh = max([x for x in range(n + 1) if upper90(x, n, fp) < dl], default=None)
            r["nonsup_max_hits_%.2f" % dl] = mh
            r["nonsup_P_shown_uninformative_%.2f" % dl] = (sum(q for x, q in enumerate(binom_pmf(n, fp / n)) if x <= mh)
                                                          if mh is not None else 0.0)
        w, l, both, g = boot_conjuncts(runs, n, rng)
        r.update({"P_window": w, "P_lead_pos": l, "P_window_lead_gen": both, "P_generator_ok": g,
                  "kill_power_at_0.5_approx": r["power_at_0.5"] * both})
        res[n] = r
        print("n = %d per arm: FP fixed at %d/%d = %.3f; significance needs >= %d/%d strict hits; size at TPR = FPR %.4f" % (
            n, fp, n, fp / n, r["min_hits"], n, r["size_at_null"]))
        print("  power at TPR 0.5: %.3f   MDE at 80%%: TPR %.3f (frozen 0.01-step helper: %s)   MDE at 90%%: %.3f" % (
            r["power_at_0.5"], r["mde80"], r["mde80_frozen_helper"], r["mde90"]))
        print("  power curve: " + ", ".join("%.3f->%.3f" % (t, p) for t, p in r["power_curve"].items()))
        print("  non-superiority at 0.25: shown iff TP <= %s/%d; P(shown | TPR = FPR = %.3f) = %.3f" % (
            r["nonsup_max_hits_0.25"], n, fp / n, r["nonsup_P_shown_uninformative"]))
        print("  non-superiority at 0.20: TP <= %s (P shown | uninformative %.3f); at 0.15: TP <= %s (P %.3f)" % (
            r["nonsup_max_hits_0.20"], r["nonsup_P_shown_uninformative_0.20"], r["nonsup_max_hits_0.15"],
            r["nonsup_P_shown_uninformative_0.15"]))
        print("  achievable margin (upper 90%% bound below it with prob 0.80 | 0.90 if uninformative, FP fixed): %.3f | %.3f;"
              " binomial-FP convention 0.80: %.3f" % (r["delta80_uninformative"], r["delta90_uninformative"], r["delta80_binomial_fp"]))
        print("  upper 90%% bound by TP count: %s" % ", ".join("%d:%.3f" % (x, u) for x, u in r["upper90_at_x"].items()))
        print("  three-way outcome by true TPR (kill-test significant / non-superiority null / inconclusive):")
        for t, (s, ns, inc) in r["outcomes"].items():
            print("    TPR %.3f: %.3f / %.3f / %.3f" % (t, s, ns, inc))
        print("  conjuncts (task-stratified bootstrap of E1 hacking runs, %d draws): P(window >= 12) %.3f; "
              "P(median lead > 0) %.3f; P(generator ok) %.3f; P(all three) %.3f" % (B, w, l, g, both))
        print("  kill power at TPR 0.5 ~ Fisher power x P(all three) = %.3f (independence assumed; leads as in E1, "
              "where disagreement did not anticipate, so this is a floor under H1)\n" % r["kill_power_at_0.5_approx"])
    (HERE / "power_primary.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
