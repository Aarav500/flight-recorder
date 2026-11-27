"""THROWAWAY SPIKE analysis. Everything here is exploratory: seeds 0-9 are spent once this is read, and a
confirmatory run needs fresh seeds and a registered rule.

For each team size:
  mechanism   does the exploit arm cluster and lose true reward, and does the patched arm not?
  collapse    t_oracle: first eval where the trailing OLS slope of true reward is negative at p < 0.01
              (tex section "How the two times are estimated"; no peak-minus-final)
  alarm       t_alarm: first update where a causal 10-update mean of the statistic is at least c reference
              SDs from the patched arm at that update, for 2 updates running. c is the smallest value at
              which at most 10% of patched runs ever alarm. A patched run's own rate is leave-one-out.
              A self-standardised variant uses the run's first 40 updates in place of the reference.
  lead        t_oracle - t_alarm, positive when the alarm comes first. Exact two-sided sign test across
              seeds (paired, so no Fisher: the AAAI review's correction).

    python analyze_phi.py results/phi [--check-against <study L results dir>]
"""
import glob
import json
import os
import sys
from math import comb

import numpy as np
from scipy import stats as st

WINDOW_EVALS = 20       # 100 updates
P_SLOPE = 0.01
SMOOTH = 10
HOLD = 2
START = 5
REF_HALF = 10           # reference mean and SD pooled over +-10 updates (reference runs are complete)
SELF_N = 40
MAX_FPR = 0.10


def load(root, params="shared"):
    runs = {}
    for f in glob.glob(os.path.join(root, params, "N*_k0.0_s*.json")):
        h = json.load(open(f))
        h["rt_m"] = np.array(h["rt"]).mean(1)
        h["rp_m"] = np.array(h["rp"]).mean(1)
        h["cluster_a"] = np.array(h["cluster"])
        runs[(h["n"], h["cond"], h["seed"])] = h
    return runs


def series(h):
    out = {k: np.array(h[k]) for k in ("phi_gae", "phi_mc", "gnorm_gae", "gnorm_mc")}
    out["gap_gae"] = np.array(h["within_gae"]) - np.array(h["cross_gae"])
    out["gap_mc"] = np.array(h["within_mc"]) - np.array(h["cross_mc"])
    return out


def t_oracle(h):
    u, y = np.array(h["update"]), h["rt_m"]
    x = np.arange(WINDOW_EVALS)
    for i in range(WINDOW_EVALS, len(y) + 1):
        r = st.linregress(x, y[i - WINDOW_EVALS:i])
        if r.slope < 0 and r.pvalue / 2 < P_SLOPE:
            return int(u[i - 1])
    return None


def causal(x):
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        lo = max(0, i - SMOOTH + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def ref_stats(curves):
    a = np.stack(curves)
    T = a.shape[1]
    mu, sd = np.empty(T), np.empty(T)
    for t in range(T):
        w = a[:, max(0, t - REF_HALF):min(T, t + REF_HALF + 1)]
        mu[t], sd[t] = w.mean(), max(w.std(ddof=1), 1e-9)
    return mu, sd


def alarm(z, c):
    hot = np.abs(z) >= c
    for t in range(START, len(z) - HOLD + 1):
        if hot[t:t + HOLD].all():
            return t
    return None


def peak_z(z):
    """Largest c at which this series still alarms: the run's alarm threshold."""
    m = np.minimum(np.abs(z[START:-1]), np.abs(z[START + 1:]))
    return float(m.max())


def threshold(null_zs):
    """Smallest c with at most MAX_FPR of null runs ever alarming."""
    peaks = sorted(peak_z(z) for z in null_zs)
    allowed = int(np.floor(MAX_FPR * len(peaks)))
    return peaks[len(peaks) - 1 - allowed] + 1e-6


def sign_p(pos, neg):
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def mechanism(runs, n):
    print(f"\n=== N={n} mechanism (mean over seeds; first 50 updates vs last 200)")
    for cond in ("exploit", "patched", "control"):
        rows = [h for (m, c, s), h in sorted(runs.items()) if m == n and c == cond]
        if not rows:
            continue
        early = [(h["rt_m"][:10].mean(), h["cluster_a"][:10].mean(), h["rp_m"][:10].mean()) for h in rows]
        late = [(h["rt_m"][-40:].mean(), h["cluster_a"][-40:].mean(), h["rp_m"][-40:].mean()) for h in rows]
        tos = [t_oracle(h) for h in rows]
        e, l = np.mean(early, 0), np.mean(late, 0)
        fired = [t for t in tos if t is not None]
        print(f"{cond:8s} n={len(rows):2d} rt {e[0]:+.3f} -> {l[0]:+.3f}  cluster {e[1]:.3f} -> {l[1]:.3f}  "
              f"rp {e[2]:.2f} -> {l[2]:.2f}  collapse fired {len(fired)}/{len(rows)}"
              + (f" median update {int(np.median(fired))}" if fired else ""))


def detect(runs, n, ref_cond="patched"):
    ex = [h for (m, c, s), h in sorted(runs.items()) if m == n and c == "exploit"]
    nu = [h for (m, c, s), h in sorted(runs.items()) if m == n and c == ref_cond]
    if not ex or len(nu) < 3:
        print(f"N={n}: need exploit and at least 3 {ref_cond} runs")
        return
    print(f"\n=== N={n} alarms, exploit against {ref_cond} ({len(ex)} v {len(nu)})")
    print(f"{'statistic':10s} {'mode':5s} {'c':>7s} {'FPR(LOO)':>9s} {'fired':>6s} {'before':>7s} {'after':>6s} "
          f"{'p':>7s} {'med lead':>9s} {'med alarm':>10s}")
    for stat in ("phi_gae", "phi_mc", "gap_gae", "gap_mc", "gnorm_gae", "gnorm_mc"):
        ex_c = [causal(series(h)[stat]) for h in ex]
        nu_c = [causal(series(h)[stat]) for h in nu]
        for mode in ("ref", "self"):
            if mode == "ref":
                mu, sd = ref_stats(nu_c)
                ex_z = [(x - mu) / sd for x in ex_c]
                # leave-one-out, so a null run is not scored against itself
                nu_z = []
                for i, x in enumerate(nu_c):
                    m_, s_ = ref_stats(nu_c[:i] + nu_c[i + 1:])
                    nu_z.append((x - m_) / s_)
            else:
                def own(x):
                    return (x - x[:SELF_N].mean()) / max(x[:SELF_N].std(ddof=1), 1e-9)
                ex_z = [own(x) for x in ex_c]
                nu_z = [own(x) for x in nu_c]
            c = threshold(nu_z)
            fpr = np.mean([alarm(z, c) is not None for z in nu_z])
            leads, alarms, fired = [], [], 0
            for h, z in zip(ex, ex_z):
                a = alarm(z, c)
                if a is None:
                    continue
                fired += 1
                alarms.append(a)
                to = t_oracle(h)
                if to is not None:
                    leads.append(to - a)
            pos, neg = sum(l > 0 for l in leads), sum(l < 0 for l in leads)
            ml = f"{int(np.median(leads)):+d}" if leads else "-"
            ma = f"{int(np.median(alarms))}" if alarms else "-"
            print(f"{stat:10s} {mode:5s} {c:7.2f} {fpr:9.2f} {fired:3d}/{len(ex):<2d} {pos:7d} {neg:6d} "
                  f"{sign_p(pos, neg):7.3f} {ml:>9s} {ma:>10s}")


def check_against(runs, old_root):
    same, diff, missing = 0, 0, 0
    for (n, cond, seed), h in sorted(runs.items()):
        if cond == "patched":
            continue
        f = os.path.join(old_root, "shared", f"N{n}_{cond}_k0.0_s{seed}.json")
        if not os.path.exists(f):
            missing += 1
            continue
        o = json.load(open(f))
        if o["rt"] == h["rt"] and o["cluster"] == h["cluster"]:
            same += 1
        else:
            diff += 1
    print(f"\nreplay of study L: identical {same}, different {diff}, no old run {missing}"
          " (differences expected if the CPU differs from the old instance's)")


def main():
    root = sys.argv[1]
    runs = load(root)
    if "--check-against" in sys.argv:
        check_against(runs, sys.argv[sys.argv.index("--check-against") + 1])
    for n in sorted({k[0] for k in runs}):
        mechanism(runs, n)
    for n in sorted({k[0] for k in runs}):
        detect(runs, n)
    ci = load(root, "indep_ci")
    if ci:
        print("\n=== separate networks, same init (the proposal's A1 and A2)")
        for n in sorted({k[0] for k in ci}):
            mechanism(ci, n)


if __name__ == "__main__":
    main()
