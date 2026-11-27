import sys
from math import comb
from pathlib import Path
import numpy as np
from scipy.stats import binom
sys.path.insert(0, r"experiments")
sys.path.insert(0, r"<home>/OneDrive/Desktop/Fligh recorder")
import e1_primary as P
for n in (24, 36, 48):
    fp = P.fixed_fp(n)
    need = min(x for x in range(n + 1) if P.fisher_one_sided(x, n, fp, n) < 0.05)
    size_binom = 1 - binom.cdf(need - 1, n, fp / n)
    print("n=%d fp=%d need=%d p(need)=%.4f p(need-1)=%.4f size_binom(TPR=FPR)=%.4f pow0.5=%.3f mde80=%s pow0.375=%.3f pow0.33=%.3f pow0.35=%.3f pow0.40=%.3f" % (
        n, fp, need, P.fisher_one_sided(need, n, fp, n), P.fisher_one_sided(need - 1, n, fp, n), size_binom,
        P.power_fixed_fp(n, n, 0.5, fp), P.mde_fixed_fp(n, n, fp), P.power_fixed_fp(n, n, 0.375, fp),
        P.power_fixed_fp(n, n, 0.33, fp), P.power_fixed_fp(n, n, 0.35, fp), P.power_fixed_fp(n, n, 0.40, fp)))
    print("   mde90=%s" % P.mde_fixed_fp(n, n, fp, 0.90))
    ns = [x for x in range(n + 1) if P.newcombe(x, n, fp, n, 0.90)[1] < 0.20]
    k = max(ns)
    print("   nonsup0.20 max TP=%d  P(TP<=k | TPR=FPR)=%.3f  upper bounds:" % (k, binom.cdf(k, n, fp / n)),
          {x: round(P.newcombe(x, n, fp, n, 0.90)[1], 3) for x in range(fp, fp + 5)})
    for t in (0.25, 0.30):
        print("   inconclusive at TPR %.2f: P(%d<=TP<=%d)=%.2f" % (t, k + 1, need - 1, binom.cdf(need - 1, n, t) - binom.cdf(k, n, t)))
    # H2 at alpha/25
    need2 = min(x for x in range(n + 1) if P.fisher_one_sided(x, n, fp, n) < 0.05 / 25)
    print("   H2 first Holm step need=%d pow0.5=%.3f mde=%s" % (need2, P.power_fixed_fp(n, n, 0.5, fp, 0.05 / 25),
          next((t / 100 for t in range(1, 101) if P.power_fixed_fp(n, n, t / 100, fp, 0.05 / 25) >= 0.8), None)))
# E1 anchors
print("E1 anchors p(8/24 v 3/24)=%.4f p(9/24)=%.4f" % (P.fisher_one_sided(8, 24, 3, 24), P.fisher_one_sided(9, 24, 3, 24)))
print("newcombe90 8v3", P.newcombe(8, 24, 3, 24, 0.9), "9v3", P.newcombe(9, 24, 3, 24, 0.9))
print("nonsup at 0.25 at 36:", max(x for x in range(37) if P.newcombe(x, 36, 4, 36, 0.9)[1] < 0.25), binom.cdf(7, 36, 4 / 36))
print("planted Bin(36,0.36) power:", P.power_fixed_fp(36, 36, 0.36, 4))
# true size under the exchangeable null: 36 hacking and 36 twin maxima iid continuous
rng = np.random.default_rng(1)
for n in (24, 36, 48):
    fp = P.fixed_fp(n)
    need = min(x for x in range(n + 1) if P.fisher_one_sided(x, n, fp, n) < 0.05)
    R = 400000
    tw = rng.random((R, n)); hk = rng.random((R, n))
    thr = np.percentile(tw, 90, axis=1)
    tp = (hk > thr[:, None]).sum(1)
    fpc = (tw > thr[:, None]).sum(1)
    print("n=%d exchangeable null: FP always %s; mean TP %.3f (binomial model %.3f); var TP %.3f (binomial %.3f); P(TP>=%d)=%.4f (binomial %.4f); P(reject Fisher at realized FP)=%.4f" % (
        n, np.unique(fpc), tp.mean(), fp, tp.var(), n * (fp / n) * (1 - fp / n), need, (tp >= need).mean(),
        1 - binom.cdf(need - 1, n, fp / n), np.mean([P.fisher_one_sided(int(t), n, fp, n) < 0.05 for t in tp[:20000]])))
