import sys, math
sys.dont_write_bytecode = True
sys.path.insert(0, r"experiments")
import e1_primary as P
from math import comb

def binom_pmf(n, p):
    return [comb(n, x) * p**x * (1-p)**(n-x) for x in range(n+1)]

for n in (24, 36):
    fp = P.fixed_fp(n)
    pw = P.power_fixed_fp(n, n, 0.5, fp)
    mde = P.mde_fixed_fp(n, n, fp)
    kmin = next(x for x in range(n+1) if P.fisher_one_sided(x, n, fp, n) < 0.05)
    # non-superiority: smallest delta (2 dp) with P(upper90 < delta) >= 0.8 under TPR = fp/n
    p0 = fp / n
    pm = binom_pmf(n, p0)
    best = None
    for d100 in range(5, 60):
        d = d100 / 100
        prob = sum(q for x, q in enumerate(pm) if P.newcombe(x, n, fp, n, 0.90)[1] < d)
        if prob >= 0.80:
            best = (d, prob); break
    # max hits for non-sup at that delta and at 0.25
    def maxhits(delta):
        ok = [x for x in range(n+1) if P.newcombe(x, n, fp, n, 0.90)[1] < delta]
        return max(ok) if ok else None
    a25 = 0.05 / 25
    pw_h = P.power_fixed_fp(n, n, 0.5, fp, a25)
    mde_h = next((t/100 for t in range(1, 101) if P.power_fixed_fp(n, n, t/100, fp, a25) >= 0.8), None)
    kmin_h = next((x for x in range(n+1) if P.fisher_one_sided(x, n, fp, n) < a25), None)
    print(f"n={n}: fixed_fp={fp} (FPR {fp/n:.3f}); power@TPR0.5={pw:.3f}; MDE80={mde}; kill needs >= {kmin} TP")
    print(f"   margin: smallest delta with P(upper90<delta)>=0.8 under TPR=FPR={p0:.3f}: {best}; maxhits(delta)={maxhits(best[0])}; maxhits(0.25)={maxhits(0.25)}; maxhits(0.20)={maxhits(0.20)}")
    print(f"   Holm first step alpha/25: power@0.5={pw_h:.3f}; MDE80={mde_h}; needs >= {kmin_h} TP")
    for d in (0.20, 0.25):
        prob = sum(q for x, q in enumerate(pm) if P.newcombe(x, n, fp, n, 0.90)[1] < d)
        print(f"   P(non-sup shown | uninformative) at delta {d}: {prob:.3f}")
# planted-control expectation at E1's realized events, rescaled
ev = [None, 5, 5, 6, 7, 11, 12, 17, 17, 17, 22, 29, 35, 35, 43, 46, 51, 60, 71, 78, 110, 119, 161, 247]
print("planted expectation at E1 events, 24v24:", round(P.power_poisson_binomial(P.planted_ps(ev), 24, P.fixed_fp(24)), 3))
ev36 = ev + ev[::2]
print("planted expectation at E1-like events, 36v36:", round(P.power_poisson_binomial(P.planted_ps(ev36), 36, P.fixed_fp(36)), 3))
print("planted expectation all events >=18, 36v36:", round(P.power_fixed_fp(36, 36, 0.5, P.fixed_fp(36)), 3))
