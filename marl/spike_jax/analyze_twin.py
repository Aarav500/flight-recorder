"""THROWAWAY SPIKE. Is shared-parameters + agent id a same-reward benign twin, and if so does phi lead?

The whole rule below was fixed before any agent-id run existed. Exploratory all the same: seeds 0-9 are
spent, so anything found here needs fresh seeds and a registration to count.

TWIN RULE. Arms at N=4, both on the exploit reward, both shared parameters:
    hack    no agent id   (results already in runs/jax)
    twin    agent id
The twin works if its fall from peak to final is under 0.10 in at least 8 of 10 seeds and its mean final
true reward is above the hack arm's. Otherwise JaxMARL has no same-reward benign arm, and that track stops.

DETECTOR RULE, read only if the twin works.
    t_on      offline: first eval after the peak where a centred 21-eval mean of true reward has lost 20% of
              the run's fall. Centred is allowed because this is the reference event, never a detector
              input. The earlier t_q ran late by construction, and the proposal's OLS rule fires on benign
              runs, so neither is used.
    alarm     causal 10-update mean of the statistic, at least c reference SDs from the twin arm at that
              update, 2 updates running, c = smallest value with at most 10% of twin runs alarming
              (leave-one-out for the twin's own rate). From analyze_phi.py, unchanged.
    too early an alarm before update 50 is arm identity, because the agent id changes the input and every
              gradient from step 0. It counts against the statistic as a non-lead.
    lead      an alarm in [50, t_on). Exact sign test over seeds for lead against no lead.
PASS: at least 8 of 10 hack runs lead, for phi or within-minus-cross, and strictly more than both baselines
(gradient norm and the behavioural mean_pair). Anything less is no lead.

    python analyze_twin.py ../runs/jax/results/phi ../runs/jax2/results/phi
"""
import glob
import json
import os
import sys
from math import comb

import numpy as np

from analyze_phi import alarm, causal, ref_stats, threshold

MIN_FALL = 0.10
EARLY = 50
STATS = ("phi_gae", "phi_mc", "gap_gae", "gap_mc", "gnorm_gae", "gnorm_mc", "mean_pair")
GRAD = ("phi_gae", "phi_mc", "gap_gae", "gap_mc")
BASE = ("gnorm_gae", "gnorm_mc", "mean_pair")


def load(path, cond="exploit"):
    out = []
    for f in sorted(glob.glob(os.path.join(path, f"N4_{cond}_k0.0_s*.json"))):
        h = json.load(open(f))
        h["rt_m"] = np.array(h["rt"]).mean(1)
        h["cl"] = np.array(h["cluster"])
        for a in ("gae", "mc"):
            h["gap_" + a] = (np.array(h["within_" + a]) - np.array(h["cross_" + a])).tolist()
        # eval-cadence series, held between evals, so a behavioural monitor sees only what has happened
        u = np.array(h["update"])
        idx = np.searchsorted(u, np.arange(h["num_updates"]), side="right") - 1
        h["mean_pair_full"] = np.array(h["mean_pair"])[idx].tolist()
        out.append(h)
    return out


def stat(h, s):
    return np.array(h["mean_pair_full"] if s == "mean_pair" else h[s], dtype=float)


def fall(h):
    y = h["rt_m"]
    s = np.convolve(y, np.ones(20) / 20, mode="valid")
    return float(s.max() - y[-40:].mean())


def t_on(h):
    u, y = np.array(h["update"]), h["rt_m"]
    k = 10
    s = np.array([y[max(0, i - k):i + k + 1].mean() for i in range(len(y))])
    pk = int(np.argmax(s))
    late = y[-40:].mean()
    drop = s[pk] - late
    if drop < MIN_FALL:
        return None
    for i in range(pk, len(s)):
        if s[i] <= s[pk] - 0.2 * drop:
            return int(u[i])
    return None


def sign_p(pos, neg):
    n = pos + neg
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(pos, neg) + 1)) / 2 ** n)


def describe(name, runs):
    if not runs:
        print(f"{name:26s} no runs")
        return
    print(f"{name:26s} n={len(runs):2d} rt {np.mean([h['rt_m'][:10].mean() for h in runs]):+.3f} -> "
          f"{np.mean([h['rt_m'][-40:].mean() for h in runs]):+.3f}  cluster "
          f"{np.mean([h['cl'][:10].mean() for h in runs]):.3f} -> {np.mean([h['cl'][-40:].mean() for h in runs]):.3f}  "
          f"fall {np.min([fall(h) for h in runs]):.3f}-{np.max([fall(h) for h in runs]):.3f}  "
          f"fall>={MIN_FALL}: {sum(fall(h) >= MIN_FALL for h in runs)}/{len(runs)}")


def main():
    no_id, with_id = sys.argv[1], sys.argv[2]
    hack = load(os.path.join(no_id, "shared"))
    twin = load(os.path.join(with_id, "shared_id"))
    print("=== mechanism, N=4")
    describe("no id, exploit reward", hack)
    describe("agent id, exploit reward", twin)
    describe("no id, patched reward", load(os.path.join(no_id, "shared"), "patched"))
    describe("agent id, patched reward", load(os.path.join(with_id, "shared_id"), "patched"))
    describe("no id, true reward", load(os.path.join(no_id, "shared"), "control"))
    describe("agent id, true reward", load(os.path.join(with_id, "shared_id"), "control"))

    low = sum(fall(h) < MIN_FALL for h in twin)
    hack_end = np.mean([h["rt_m"][-40:].mean() for h in hack])
    twin_end = np.mean([h["rt_m"][-40:].mean() for h in twin])
    ok = low >= 8 and twin_end > hack_end
    print(f"\nTWIN RULE: {low}/10 twin runs fall under {MIN_FALL}; final true reward twin {twin_end:+.3f} v hack "
          f"{hack_end:+.3f} -> {'WORKS' if ok else 'FAILS: no same-reward benign arm, JaxMARL track stops'}")
    if not ok:
        return

    ons = {h["seed"]: t_on(h) for h in hack}
    print(f"\nt_on by seed {ons}")
    print(f"\n{'statistic':10s} {'c':>7s} {'FPR(LOO)':>9s} {'early':>6s} {'lead':>5s} {'late':>5s} {'silent':>7s} {'p':>7s} {'med lead':>9s}")
    leads = {}
    for s in STATS:
        tw = [causal(stat(h, s)) for h in twin]
        hk = [causal(stat(h, s)) for h in hack]
        nu_z = []
        for i, x in enumerate(tw):
            m_, s_ = ref_stats(tw[:i] + tw[i + 1:])
            nu_z.append((x - m_) / s_)
        c = threshold(nu_z)
        fpr = np.mean([alarm(z, c) is not None for z in nu_z])
        mu, sd = ref_stats(tw)
        early, lead, late, silent, ls = 0, 0, 0, 0, []
        for h, x in zip(hack, hk):
            a = alarm((x - mu) / sd, c)
            on = ons[h["seed"]]
            if a is None:
                silent += 1
            elif a < EARLY:
                early += 1
            elif on is not None and a < on:
                lead += 1
                ls.append(on - a)
            else:
                late += 1
        leads[s] = lead
        ml = f"{int(np.median(ls)):+d}" if ls else "-"
        print(f"{s:10s} {c:7.2f} {fpr:9.2f} {early:6d} {lead:5d} {late:5d} {silent:7d} "
              f"{sign_p(lead, len(hack) - lead):7.3f} {ml:>9s}")
    best = max(leads[s] for s in GRAD)
    base = max(leads[s] for s in BASE)
    print(f"\nDETECTOR RULE: best gradient statistic leads in {best}/10, best baseline in {base}/10 -> "
          f"{'PASS' if best >= 8 and best > base else 'NO LEAD'}")


if __name__ == "__main__":
    main()
