"""E1 power: one-sided Fisher exact test of early-warning TPR (hacking runs) > benign FPR.

The two arms are DIFFERENT runs, so the samples are independent and Fisher's exact test is the
right test. (The kit 02 template's warning against Fisher applies to PAIRED detectors -- two
detectors scored on the same runs -- which is not this comparison.)

Exact enumeration, no simulation: for every outcome (x hacking alarms of n_h, y benign false alarms
of n_b), compute the one-sided Fisher p-value for "hacking rate > benign rate" and sum the
probability of every outcome with p < alpha.

Kit 02 lets the assistant calculate power; it does NOT let the assistant choose the effect size or
the kill rule. The TPR/FPR pairs below are the ones the planning doc names, not choices made here.
"""
from functools import lru_cache
from math import comb


def fisher_one_sided(x, n1, y, n2):
    """P(X >= x) under the hypergeometric null, X = hacking alarms, margins fixed."""
    k = x + y
    N = n1 + n2
    denom = comb(N, k)
    p = 0.0
    for xi in range(x, min(k, n1) + 1):
        yi = k - xi
        if 0 <= yi <= n2:
            p += comb(n1, xi) * comb(n2, yi) / denom
    return p


def power(n_h, n_b, tpr, fpr, alpha=0.05):
    tot = 0.0
    for x in range(n_h + 1):
        px = comb(n_h, x) * tpr ** x * (1 - tpr) ** (n_h - x)
        for y in range(n_b + 1):
            py = comb(n_b, y) * fpr ** y * (1 - fpr) ** (n_b - y)
            if fisher_one_sided(x, n_h, y, n_b) < alpha:
                tot += px * py
    return tot


def mde(n_h, n_b, fpr, target=0.80, alpha=0.05):
    """Smallest TPR reaching the target power, to 0.01."""
    for t in [i / 100 for i in range(int(fpr * 100) + 1, 101)]:
        if power(n_h, n_b, t, fpr, alpha) >= target:
            return t
    return None


if __name__ == "__main__":
    print("one-sided Fisher exact, alpha=0.05, exact enumeration\n")
    print("%-24s %8s" % ("design", "power"))
    for (nh, nb, t, f, label) in [
        (9, 9, 0.5, 0.1, "current paper 9 vs 9"),
        (24, 24, 0.5, 0.1, "E1 planned 24 vs 24"),
        (24, 24, 0.4, 0.1, "E1, if TPR only 0.4"),
        (24, 24, 0.3, 0.1, "E1, if TPR only 0.3"),
    ]:
        print("%-24s %8.3f   (TPR %.1f vs FPR %.1f)" % (label, power(nh, nb, t, f), t, f))
    print()
    for nh in (9, 24, 36):
        print("MDE at 80%% power, n=%d vs %d, FPR 0.1:  TPR >= %s" % (nh, nh, mde(nh, nh, 0.1)))
