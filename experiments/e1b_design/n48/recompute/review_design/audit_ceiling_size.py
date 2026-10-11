"""Size of the RESCUE rule (5th pct of Delta_max > 0) when disagreement and the best tier-1 comparator have EQUAL
true AUROC, across the range up to the ceiling. Two comparators (C1 best, C2 at 0.75), MVN changes, n = 36 per arm.
Bootstrap within arm (the registered one is within arm x task; same idea). B = 2000."""
import numpy as np
from multiprocessing import Pool
from scipy.stats import norm
N, BOOT, SIMS = 36, 2000, 1000
def d_of(a): return np.sqrt(2) * norm.ppf(a)
def weights(n, rng):
    idx = rng.integers(0, n, (BOOT, n)) + n * np.arange(BOOT)[:, None]
    return np.bincount(idx.ravel(), minlength=BOOT * n).reshape(BOOT, n).astype(float)
def boot_auroc(h, t, Wh, Wt):
    C = (h[:, None] > t[None, :]).astype(float) + 0.5 * (h[:, None] == t[None, :])
    return ((Wh @ C) * Wt).sum(1) / (len(h) * len(t))
def cell(args):
    a, rho, seed = args
    rng = np.random.default_rng(seed)
    mu = np.array([d_of(a), d_of(a), d_of(0.75)])
    S = np.array([[1, rho, 0.3], [rho, 1, 0.3], [0.3, 0.3, 1.0]])
    rej = ceil = 0
    for _ in range(SIMS):
        H = rng.multivariate_normal(mu, S, N); T = rng.multivariate_normal(np.zeros(3), S, N)
        Wh, Wt = weights(N, rng), weights(N, rng)
        s = boot_auroc(H[:, 0], T[:, 0], Wh, Wt)
        c = [boot_auroc(H[:, j], T[:, j], Wh, Wt) for j in (1, 2)]
        c = np.max(np.vstack([np.maximum(x, 1 - x) for x in c]), 0)
        rej += np.percentile(s - c, 5) > 0
        ceil += np.all(s == 1.0)
    return a, rho, rej / SIMS, ceil / SIMS
if __name__ == "__main__":
    jobs = [(a, r, 100 + i) for i, (a, r) in enumerate([(a, r) for a in (0.80, 0.90, 0.95, 0.97, 0.98, 0.99) for r in (0.5, 0.9)])]
    with Pool(12) as p:
        for a, r, rej, ceil in p.map(cell, jobs):
            print("equal true AUROC %.2f, rho %.1f: RESCUE rate %.3f (nominal 0.05); share of experiments with AUROC(S)=1 in every draw %.2f" % (a, r, rej, ceil))
