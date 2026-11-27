"""Uninformative detector with the threshold estimated from the twins (iid windowed maxima)."""
import sys
import numpy as np
from scipy.stats import binom
sys.path.insert(0, r"experiments")
import e1_primary as P
rng = np.random.default_rng(7)
R = 1_000_000
for n in (24, 36, 48):
    fp = P.fixed_fp(n)
    need = min(x for x in range(n + 1) if P.fisher_one_sided(x, n, fp, n) < 0.05)
    # threshold = np.percentile of n iid U(0,1) at 90 (linear): order stats via beta draws is exact enough by simulation
    tw = np.sort(rng.random((R, n)), axis=1)
    pos = 0.9 * (n - 1); lo = int(np.floor(pos)); fr = pos - lo
    thr = tw[:, lo] + fr * (tw[:, lo + 1] - tw[:, lo])
    tp = rng.binomial(n, 1 - thr)            # hacking maxima iid U(0,1): exceedances given thr
    k20 = max(x for x in range(n + 1) if P.newcombe(x, n, fp, n, 0.90)[1] < 0.20)
    k19 = max(x for x in range(n + 1) if P.newcombe(x, n, fp, n, 0.90)[1] < 0.19)
    print("n=%d: size P(TP>=%d)=%.4f (stated binomial %.4f); mean TP %.2f; P(nonsup at 0.20: TP<=%d)=%.3f (binomial %.3f); at 0.19 TP<=%d: %.3f" % (
        n, need, (tp >= need).mean(), 1 - binom.cdf(need - 1, n, fp / n), tp.mean(), k20, (tp <= k20).mean(), binom.cdf(k20, n, fp / n), k19, (tp <= k19).mean()))
    # smallest margin reached with prob >= 0.80 under this model
    for k in range(fp, n):
        if (tp <= k).mean() >= 0.80:
            print("    smallest k with P(TP<=k)>=0.80: %d; Newcombe upper at k: %.3f" % (k, P.newcombe(k, n, fp, n, 0.90)[1]))
            break
