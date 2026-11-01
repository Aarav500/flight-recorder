"""E1b power, step 4: the AUDIT test.

Test (proposed for E1b section 2/6): per run, the signed change of each signal (mean of the last 25 steps minus the
first 25, e1_supplementary.chg). Disagreement's direction is fixed in advance (hacking higher). For each
comparator the AUROC is taken in its better direction, max(A, 1 - A), which favours the comparators.
  Delta = AUROC(disagreement) - max over comparators of AUROC(comparator).
Paired, arm-stratified bootstrap: resample hacking runs and twin runs with replacement (B = 2000), the SAME
resample for every signal, recompute Delta on each resample with the max re-taken inside the resample.
Reject "disagreement is no better than the best comparator" when the 5th percentile of the bootstrap Delta
is above 0 (one-sided 95% percentile bound).

Generative model (assumptions, stated in the output): each run's vector of signal changes is multivariate
normal with unit variances. Twin arm mean 0; hacking arm mean d_j = sqrt(2) * Phi^-1(A_j), which gives
population AUROC A_j. Comparators: 1 "best" comparator at A_best and K_NUIS nuisance comparators at 0.60.
corr(disagreement, best) = rho; every other pair 0.3. Planned comparator set (planning doc :143 plus the
ungated judge): 0.5B judge alone, the training reward (gated in the twin), 1.5B judge alone, completion length,
fence/parse rate. The ceiling row (A_best = 0.99) stands for the fence/parse rate, which separates E1's
hacking runs from E1's benign runs at AUROC 1.00 (-0.465 v +0.420, e1_inputs.json) and which the twin's gate
holds near 1.

Bootstrap AUROC by weights: with C[i, l] = 1[h_i > t_l], a resample with multiplicity vectors w_h, w_t has
AUROC = w_h C w_t / n^2, so each resample costs one matrix product.

    python power_audit.py [sims] > power_audit.txt
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, norm

HERE = Path(__file__).resolve().parent
NS = (48,)
SIMS = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
BOOT = 2000
PAIRS = [(0.95, 0.75), (0.95, 0.85), (0.90, 0.75), (0.90, 0.80), (0.85, 0.65), (0.85, 0.75), (0.80, 0.60),
         (0.75, 0.60), (0.95, 0.99)]
RHOS = (0.0, 0.5, 0.9)
K_NUIS, A_NUIS, R_OTHER = 4, 0.60, 0.3      # 5 comparators in all: the best one plus 4 nuisance


def d_of(a):
    return np.sqrt(2) * norm.ppf(a)


def cov(rho, m):
    S = np.full((m, m), R_OTHER)
    np.fill_diagonal(S, 1.0)
    S[0, 1] = S[1, 0] = rho
    return S


def weights(n, rng):
    idx = rng.integers(0, n, (BOOT, n)) + n * np.arange(BOOT)[:, None]
    return np.bincount(idx.ravel(), minlength=BOOT * n).reshape(BOOT, n).astype(float)


def boot_auroc(h, t, Wh, Wt):
    C = (h[:, None] > t[None, :]).astype(float)
    return ((Wh @ C) * Wt).sum(1) / (len(h) * len(t))


def one_sim(n, a_dis, a_best, rho, rng):
    m = 2 + K_NUIS
    mu = np.array([d_of(a_dis), d_of(a_best)] + [d_of(A_NUIS)] * K_NUIS)
    S = cov(rho, m)
    H = rng.multivariate_normal(mu, S, n)
    T = rng.multivariate_normal(np.zeros(m), S, n)
    Wh, Wt = weights(n, rng), weights(n, rng)
    a0 = boot_auroc(H[:, 0], T[:, 0], Wh, Wt)
    comps = []
    for j in range(1, m):
        a = boot_auroc(H[:, j], T[:, j], Wh, Wt)
        comps.append(np.maximum(a, 1 - a))
    delta = a0 - np.max(np.vstack(comps), axis=0)
    return bool(np.percentile(delta, 5) > 0)


def cell(args):
    n, a_dis, a_best, rho, seed = args
    rng = np.random.default_rng(seed)
    return float(np.mean([one_sim(n, a_dis, a_best, rho, rng) for _ in range(SIMS)]))


def mw_power(n, a, rng, sims=4000):
    d = d_of(a)
    hit = 0
    for _ in range(sims):
        h, t = rng.normal(d, 1, n), rng.normal(0, 1, n)
        hit += mannwhitneyu(h, t, alternative="greater").pvalue < 0.05
    return hit / sims


def main():
    res = {"assumptions": {"sims": SIMS, "boot": BOOT, "k_nuisance": K_NUIS, "a_nuisance": A_NUIS, "r_other": R_OTHER}}
    print("E1b AUDIT TEST POWER: paired bootstrap of AUROC(disagreement) - max AUROC(5 comparators), one-sided 95%% "
          "percentile bound > 0. %d simulated experiments per cell, B = %d. Monte Carlo SE <= %.3f." % (
              SIMS, BOOT, 0.5 / np.sqrt(SIMS)))
    print("Model: MVN signal changes, unit variance; best comparator at A_best, %d nuisance comparators at %.2f; "
          "corr(disagreement, best) = rho, all other pairs %.1f; comparator AUROCs direction-free.\n" % (
              K_NUIS, A_NUIS, R_OTHER))
    jobs = [(n, 0.80, 0.80, 0.5, 1000 + n) for n in NS]
    jobs += [(n, a, b, r, 7 + 1000 * i + n) for i, (a, b) in enumerate(PAIRS) for r in RHOS for n in NS]
    with Pool(18) as pool:
        out = dict(zip([j[:4] for j in jobs], pool.map(cell, jobs)))
    for n in NS:
        res.setdefault("null_size", {})[n] = out[(n, 0.80, 0.80, 0.5)]
        print("size check, A_dis = A_best = 0.80, rho 0.5, n = %d: rejection %.3f (nominal 0.05)" % (
            n, out[(n, 0.80, 0.80, 0.5)]))
    print("\nA_dis  A_best  rho  " + "  ".join("n=%d " % n for n in NS))
    res["power"] = {}
    for a_dis, a_best in PAIRS:
        for rho in RHOS:
            row = [out[(n, a_dis, a_best, rho)] for n in NS]
            for n, p in zip(NS, row):
                res["power"]["%.2f|%.2f|%.1f|%d" % (a_dis, a_best, rho, n)] = p
            print("%.2f   %.2f   %.1f  %s" % (a_dis, a_best, rho, "  ".join("%.3f" % p for p in row)))
    rng = np.random.default_rng(20261009)
    print("\nOne-sample alternative (disagreement AUROC > 0.5 alone, one-sided Mann-Whitney, alpha 0.05):")
    res["mw"] = {}
    for a in (0.60, 0.65, 0.70, 0.75, 0.80):
        row = [mw_power(n, a, rng) for n in NS]
        res["mw"][a] = row
        print("  AUROC %.2f: %s" % (a, "  ".join("n=%d %.3f" % (n, p) for n, p in zip(NS, row))))
    d = json.loads((HERE / "e1_inputs.json").read_text())
    hk = [r for r in d["runs"] if r["arm"] == "hacking"]
    print("\nAnchors: AUROC implied by a hacking-minus-twin mean difference, both arms with E1's hacking-arm SD:")
    for key, diffs in (("disagree", (0.04, 0.08, 0.12)), ("judge_small", (0.01, 0.02, 0.04)),
                       ("judge_big", (0.02, 0.04, 0.08)), ("mean_len", (10, 20, 40))):
        sd = float(np.std([r["chg"][key] for r in hk], ddof=1))
        print("  %-11s SD %.3f: %s" % (key, sd, ", ".join("diff %g -> %.2f" % (x, norm.cdf(x / (sd * np.sqrt(2))))
                                                          for x in diffs)))
    (HERE / "power_audit.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
