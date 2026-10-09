"""Exact: n iid twin maxima, threshold = numpy linear 90th pct = U(j) + fr (U(j+1) - U(j)); TP | thr ~ Bin(n, 1 - thr)."""
import sys
from math import factorial, floor
import numpy as np
from scipy import integrate
from scipy.stats import binom
sys.path.insert(0, r"experiments")
import e1_primary as P
def prob(n, event):
    pos = 0.9 * (n - 1); j = floor(pos) + 1; fr = pos - floor(pos)     # 1-indexed order stats j, j+1
    c = factorial(n) / (factorial(j - 1) * factorial(n - j - 1))
    f = lambda v, u: c * u ** (j - 1) * (1 - v) ** (n - j - 1) * event(1 - (u + fr * (v - u)))
    val, err = integrate.dblquad(f, 0, 1, lambda u: u, lambda u: 1, epsabs=1e-10, epsrel=1e-10)
    return val
for n in (24, 36, 48):
    fp = P.fixed_fp(n)
    need = min(x for x in range(n + 1) if P.fisher_one_sided(x, n, fp, n) < 0.05)
    k20 = max(x for x in range(n + 1) if P.newcombe(x, n, fp, n, 0.90)[1] < 0.20)
    tot = prob(n, lambda q: 1.0)
    size = prob(n, lambda q: 1 - binom.cdf(need - 1, n, q))
    ns = prob(n, lambda q: binom.cdf(k20, n, q))
    ns25 = prob(n, lambda q: binom.cdf(max(x for x in range(n + 1) if P.newcombe(x, n, fp, n, 0.90)[1] < 0.25), n, q))
    print("n=%d total=%.6f size=%.4f P(nonsup0.20, TP<=%d)=%.4f P(nonsup0.25)=%.4f" % (n, tot, size, k20, ns, ns25))
