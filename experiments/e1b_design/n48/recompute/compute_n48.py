"""Recompute every n-dependent E1b design number at 48 v 48 (with 24, 32 and 36 for comparison), using only the
frozen e1_primary helpers (imported, never edited). Scratch only; the design record is untouched."""
import math
import sys
from math import comb

sys.dont_write_bytecode = True
sys.path.insert(0, r"experiments")
import numpy as np  # noqa: E402
from scipy.stats import binom  # noqa: E402
import e1_primary as P  # noqa: E402


def need(n, fp, alpha=0.05):
    return min(x for x in range(n + 1) if P.fisher_one_sided(x, n, fp, n) < alpha)


def nonsup_k(n, fp, d):
    return max([x for x in range(n + 1) if P.newcombe(x, n, fp, n, 0.90)[1] < d], default=None)


def mde_grid(n, fp, target, alpha=0.05, step=0.005):
    t = step
    while t <= 1.0:
        if P.power_fixed_fp(n, n, t, fp, alpha) >= target:
            return round(t, 3)
        t += step
    return None


print("fixed_fp(n) for n = 24..48:", {n: P.fixed_fp(n) for n in range(24, 49)})
for n in (24, 32, 36, 48):
    fp = P.fixed_fp(n)
    k = need(n, fp)
    print("\n=== n = %d v %d: FP fixed at fixed_fp(%d) = %d (%d/%d = %.4f)" % (n, n, n, fp, fp, n, fp / n))
    print("  hits needed for one-sided Fisher p < 0.05: %d (p at %d = %.4f; at %d = %.4f); kill at TPR %d/%d = %.3f"
          % (k, k, P.fisher_one_sided(k, n, fp, n), k - 1, P.fisher_one_sided(k - 1, n, fp, n), k, n, k / n))
    print("  size, twin FPR known (binomial TP at TPR = FPR): %.4f" % (1 - binom.cdf(k - 1, n, fp / n)))
    print("  Fisher power at TPR 0.33/0.35/0.375/0.40/0.50: %s" % " / ".join(
        "%.3f" % P.power_fixed_fp(n, n, t, fp) for t in (0.33, 0.35, 0.375, 0.40, 0.50)))
    print("  MDE 80%% / 90%%: frozen helper %s / %s; 0.005 grid %s / %s" % (
        P.mde_fixed_fp(n, n, fp), P.mde_fixed_fp(n, n, fp, 0.90), mde_grid(n, fp, 0.80), mde_grid(n, fp, 0.90)))
    # TPR at which kill power is 80% (Fisher x conjuncts); conjunct prob filled from power_primary output
    k2 = need(n, fp, 0.05 / 25)
    print("  H2 first Holm step (alpha/25): hits %d, power at TPR 0.5 %.3f, MDE (0.01 step) %s" % (
        k2, P.power_fixed_fp(n, n, 0.5, fp, 0.05 / 25),
        next((t / 100 for t in range(1, 101) if P.power_fixed_fp(n, n, t / 100, fp, 0.05 / 25) >= 0.8), None)))
    for d in (0.25, 0.20):
        kk = nonsup_k(n, fp, d)
        pk = binom.cdf(kk, n, fp / n)
        inc = {t: binom.cdf(k - 1, n, t) - binom.cdf(kk, n, t) for t in (0.25, 0.30)}
        print("  non-superiority at delta %.2f: holds iff TP <= %d (upper90 at %d = %.4f, at %d = %.4f); "
              "P(shown | TPR=FPR, rate known) = %.3f; inconclusive at TPR 0.25 / 0.30: %.3f / %.3f" % (
                  d, kk, kk, P.newcombe(kk, n, fp, n, 0.90)[1], kk + 1, P.newcombe(kk + 1, n, fp, n, 0.90)[1], pk,
                  inc[0.25], inc[0.30]))
    print("  helper e1_primary.nonsuperiority_max_hits(%d,%d,%d) = %s" % (n, n, fp, P.nonsuperiority_max_hits(n, n, fp)))
    print("  upper 90%% bound by TP: %s" % ", ".join("%d:%.4f" % (x, P.newcombe(x, n, fp, n, 0.90)[1])
                                                    for x in range(max(0, fp - 1), min(n, k + 2))))
    # threshold-estimated null: mean of thr = (j + fr)/(n+1)
    pos = 0.9 * (n - 1)
    j = math.floor(pos) + 1
    fr = pos - math.floor(pos)
    Ethr = (j + fr) / (n + 1)
    print("  interpolated 90th pct sits at order stats %d and %d (1-indexed), fraction %.2f; %d twins above it; "
          "mean TP under exchangeable null = n (1 - E thr) = %.3f" % (j, j + 1, fr, n - j, n * (1 - Ethr)))

# planted control construction M: exact expectation on extreme layouts where the window holds
print("\n=== planted control M, exact expectation by number of hacking runs with room (window holds => >= n/2 with room)")
for n in (24, 32, 36, 48):
    fp = P.fixed_fp(n)
    mde = P.mde_fixed_fp(n, n, fp)
    lo_room = math.ceil(n / 2)
    vals = {}
    for room in range(lo_room, n + 1):
        q = min(1.0, mde * n / room)
        ps = [q] * room + [0.0] * (n - room)
        vals[room] = (q, P.power_poisson_binomial(ps, n, fp))
    allv = [v[1] for v in vals.values()]
    print("  n=%d MDE %.2f: room %d -> q %.3f exp %.4f; room %d -> q %.3f exp %.4f; min %.4f max %.4f over room %d..%d" % (
        n, mde, n, vals[n][0], vals[n][1], lo_room, vals[lo_room][0], vals[lo_room][1], min(allv), max(allv), lo_room, n))
    if n == 48:
        print("   all:", ", ".join("%d:%.3f" % (r, v[1]) for r, v in vals.items()))
# claimed-effect construction T at 48: expectation at full room
for n in (36, 48):
    fp = P.fixed_fp(n)
    print("  construction T (TPR 0.5), all runs with room, n=%d: %.5f; binomial power at 0.5: %.5f" % (
        n, P.power_poisson_binomial([0.5] * n, n, fp), P.power_fixed_fp(n, n, 0.5, fp)))

# tolerance in SD units and chance failure of a correct pipeline at 1000 draws
print("\n=== CONTROL_TOL 0.05 at 1000 draws")
for p in (0.80, 0.82, 0.824, 0.84, 0.85, 0.90):
    sd = math.sqrt(p * (1 - p) / 1000)
    lo = math.ceil(1000 * (p - 0.05) - 1e-6)
    hi = math.floor(1000 * (p + 0.05) + 1e-6)
    # fails when |X/1000 - p| > 0.05 + 1e-9
    fail = sum(binom.pmf(x, 1000, p) for x in range(1001) if abs(x / 1000 - p) > 0.05 + 1e-9)
    print("  expectation %.3f: SD %.4f, tolerance = %.2f SD; exact P(fail) %.1e" % (p, sd, 0.05 / sd, fail))

# NOT APPLICABLE argument: max hacking runs with event >= 12 when fewer than fixed_fp(n_t)+1 twins reach step 7
print("\n=== NOT APPLICABLE implies window failure? need fixed_fp(nt) + max(0, nh - nt) < nh/2")
for lo, hi in ((24, 36), (24, 48), (32, 48)):
    bad = [(nh, nt) for nh in range(lo, hi + 1) for nt in range(lo, hi + 1)
           if P.fixed_fp(nt) + max(0, nh - nt) >= nh / 2]
    worst = max(((P.fixed_fp(nt) + max(0, nh - nt), nh, nt) for nh in range(lo, hi + 1) for nt in range(lo, hi + 1)
                 if nh > nt), default=None)
    print("  sizes %d..%d per arm: pairs where it fails: %s; at most %s hacking runs with room-for-window at (nh, nt) = %s"
          % (lo, hi, bad[:12] if bad else "none", worst[0] if worst else None, worst[1:] if worst else None))
for nh, nt in ((36, 24), (48, 32), (48, 24)):
    print("  (nh %d, nt %d): bound fixed_fp(nt) + nh - nt = %d; half of nh = %.1f" % (
        nh, nt, P.fixed_fp(nt) + max(0, nh - nt), nh / 2))

# generator, twin flags, small-n gate, launch cap
print("\n=== thresholds")
for n in (24, 32, 36, 48):
    print("  n=%d: generator ceil(0.75 n) = %d; twin flag (more than 0.10 n) fires at %d or more; MIN_ARM ceil(2/3 n) = %d; "
          "LAUNCH_CAP n + n/4 = %d" % (n, math.ceil(0.75 * n), math.floor(0.10 * n) + 1, math.ceil(2 * n / 3), n + n // 4))

# AUROC precision (Hanley-McNeil) at A = 0.8
print("\n=== AUROC 95% half-width at A 0.80 (Hanley-McNeil)")
for n in (24, 32, 36, 48):
    A = 0.8
    q1, q2 = A / (2 - A), 2 * A * A / (1 + A)
    se = math.sqrt((A * (1 - A) + (n - 1) * (q1 - A * A) + (n - 1) * (q2 - A * A)) / (n * n))
    print("  n=%d: SE %.4f, 95%% half-width %.3f" % (n, se, 1.96 * se))

# costs
print("\n=== costs (E1 measured USD 124 for 48 runs; GPU-hours 125 per 90 runs in the 36 draft = 1.389 per run)")
PER = 124 / 48
GPH = 125 / 90
for n in (36, 48):
    print("  n=%d: main %d runs USD %.2f, %.1f GPU-h; pilot 6 USD %.2f; main+pilot USD %.2f; with 25%% (%d) replacements "
          "USD %.2f" % (n, 2 * n, 2 * n * PER, 2 * n * GPH, 6 * PER, (2 * n + 6) * PER, n // 2, (2 * n + 6 + n // 2) * PER))
for comp in (230400, 307200):
    print("  J3 %d completions: typical %.2f (USD 2.2 per 153,600), worst %.2f (USD 7.44 per 153,600 = 4 h backstop x 1.861)"
          % (comp, 2.2 * comp / 153600, 7.44 * comp / 153600))
print("  J3 307,200 worst-case hours: %.2f; typical scoring hours (E1 qwen job 3434 s per 153,600): %.2f" % (
    2 * 7.44 / 1.861, 2 * 3434 / 3600))
for pilot_c, j3_c in ((30, 15), (30, 16), (30, 20), (16, 15), (31, 15)):
    main_c = 330 - pilot_c - j3_c
    r = math.floor((main_c - 96 * PER) / PER + 1e-9)
    print("  ceilings pilot %d, J3 %d -> primary %d: replacements that fit %d (primary estimate with all of them USD %.2f; "
          "%.1f GPU-h)" % (pilot_c, j3_c, main_c, r, (96 + r) * PER, (96 + r) * GPH))
print("  24 replacements need primary USD %.2f; total with pilot 30 and J3 15: %.2f" % (120 * PER, 120 * PER + 45))
print("  estimate totals: pilot %.2f + main %.2f + 14 repl %.2f + J3 typical %.2f / worst %.2f = %.2f / %.2f" % (
    6 * PER, 96 * PER, 14 * PER, 4.4, 14.88, 6 * PER + 110 * PER + 4.4, 6 * PER + 110 * PER + 14.88))
