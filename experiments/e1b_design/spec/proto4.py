import sys, math
sys.dont_write_bytecode = True
sys.path.insert(0, r"experiments")
import numpy as np
import e1_primary as P

def dev_up(x):
    x = np.asarray(x, float); xf = x[np.isfinite(x)]
    d = x - np.median(xf[:5]); return np.where(np.isfinite(d), np.maximum(d, 0.0), 0.0)
def start_of(x, k=5):
    fin = np.flatnonzero(np.isfinite(x)); return int(fin[k - 1]) + 1
def first_move_up_c(x, k=3.0, fl=0.04):
    fin = np.flatnonzero(np.isfinite(x)); xf = x[fin]
    b = xf[:3].mean(); n = max(xf[:6].std(), 1e-6); thr = max(k*n, fl)
    for i in range(int(fin[5]) + 1, len(x)):
        if x[i] - b > thr: return i
    return None

def planted(events, T=400, draws=1000, seed=20261009):
    rng = np.random.default_rng(seed); n = len(events); rej = 0
    W = [T if e is None else e for e in events]
    for _ in range(draws):
        m, tp, leads = [], 0, []
        for w in W:                                     # twins
            x = rng.uniform(0, 1, T)
            if w >= 7:
                x[int(rng.integers(6, w))] = rng.uniform(1.5, 2.5)
            d = dev_up(x); s = start_of(x)
            m.append(float(np.max(d[s:w])) if w > s else 0.0)
        thr = float(np.percentile(m, 90)); fp = sum(v > thr for v in m)
        for e in events:                                # hacking
            x = rng.uniform(0, 1, T)
            if e is not None and rng.random() < 0.5:
                at = e - int(rng.integers(1, 13))
                if at >= 6: x[at] = rng.uniform(4, 5)
            d = dev_up(x); s = start_of(x)
            a = next((t for t in range(s, T) if d[t] > thr), None)
            tp += int(a is not None and e is not None and a < e)
            fm = first_move_up_c(x)
            if e is not None and fm is not None: leads.append(e - fm)
        p = P.fisher_one_sided(tp, n, fp, n)
        med = np.median(leads) if leads else None
        rej += int(p < 0.05 and med is not None and med > 0)
    k_sp = sum(w >= 7 for w in W)
    exp = P.power_poisson_binomial(P.planted_ps(events), n, P.fixed_fp(n)) if k_sp >= P.fixed_fp(n) + 1 else None
    return rej / draws, exp
E1 = [None, 5, 5, 6, 7, 11, 12, 17, 17, 17, 22, 29, 35, 35, 43, 46, 51, 60, 71, 78, 110, 119, 161, 247]
for lay in (E1, E1 + E1[::2], [11, 21, 32] * 12, [12] * 18 + [380] * 18):
    print(len(lay), planted(lay, draws=600))
for t in (0.30, 0.33, 0.35, 0.40):
    print("power 36v36 FP4 at TPR", t, round(P.power_fixed_fp(36, 36, t, 4), 3), "| 24v24 FP3", round(P.power_fixed_fp(24, 24, t, 3), 3))
