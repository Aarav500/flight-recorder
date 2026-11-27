"""E1b power, step 3: the planted positive control at realistic event timings.

Event layouts: task-stratified bootstrap of E1's 24 registered hacking events (e1_inputs.json; one has no event),
n/3 per task. For each layout the expected rejection rate is exact (e1_primary.power_poisson_binomial with FP
fixed at fixed_fp(n)); the table reports its mean and 10th percentile over layouts.

Constructions compared:
  E1   (registered in E1): plant with prob 0.5 at e - U{1..12}, kept only if >= step 6, so
       p_i = 0.5 * clip((e_i - 6) / 12, 0, 1) (e1_primary.planted_ps).
  R    (room-capped offset): plant with prob 0.5 at e - U{1..min(12, e - 6)}, so p_i = 0.5 for every run with
       e_i >= 7 and 0 otherwise.
  M    as T, with the target set to the design MDE at 80% power (0.44 / 0.36 / 0.32), so the control is
       expected to reject about 80% of the time and can fail visibly.
  T    (target TPR 0.5, PROPOSED for E1b): as R, but runs with room are planted with prob
       q = min(1, 0.5 n / n_room), so the expected planted TPR is 0.5 of ALL hacking runs, the claimed effect.
       n_room = hacking runs with e >= 7, known only after the events are dated, which happens before any
       detector is scored (the order in section 11), so q uses no detector output.

Then a simulation check that construction T is exact under the E1b scoring path (upward-only deviation,
time-matched twin windows, rule B for no-event twins). Twin spikes U(2.5, 3.5) are placed at a step in
[6, w) of each twin's scored window w (none if w <= 6); planted hacking spikes are U(5, 6). With U(0, 1) noise
and an upward deviation from the median of the first 5 steps, every noise deviation is < 1, every twin spike
deviation is >= 1.5 and <= 3.5, and every planted deviation is >= 4, so the planted control is exact as long
as at most floor(0.9 (n - 1)) twins lack a spike. (E1's U(1.5, 2.5) / U(4, 5) spikes are exact for the
two-sided deviation only up to that same margin; the wider gaps remove the edge case.)

    python planted_control.py > planted_control.txt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

REPO = Path(r"<home>/OneDrive/Desktop/Fligh recorder")
sys.path.insert(0, str(REPO / "experiments"))
import e1_primary as P  # noqa: E402

HERE = Path(__file__).resolve().parent
NS = (24, 36, 48)
LAYOUTS = 4000
T_STEPS = 400


def ps_E1(ev):
    return P.planted_ps(ev)


def ps_R(ev):
    return [0.5 if (e is not None and e >= 7) else 0.0 for e in ev]


def ps_T(ev, target=0.5):
    room = sum(1 for e in ev if e is not None and e >= 7)
    q = min(1.0, target * len(ev) / room) if room else 0.0
    return [q if (e is not None and e >= 7) else 0.0 for e in ev]


MDE80 = {24: 0.44, 36: 0.36, 48: 0.32}     # power_primary.py, FP fixed (0.005 grid)


def layouts(events, n, rng, k):
    by = {}
    for t, e in events.items():
        by.setdefault(t.split("/")[1], []).append(e)
    out = []
    for _ in range(k):
        ev = []
        for task, es in by.items():
            ev += [es[i] for i in rng.integers(0, len(es), n // 3)]
        out.append(ev)
    return out


# ---- E1b scoring path, minimal re-implementation for the check (the registered script will reuse
# ---- e1_primary.score's structure with the time-matched threshold)
def up_dev(x):
    return np.maximum(x - np.median(x[:5]), 0.0)


def score_tm(hack_series, twin_series, events):
    """Upward-only alarm, time-matched threshold, rule B. Returns (TP, FP, n)."""
    tw = []
    for s, e in zip(twin_series, events):
        d = up_dev(s)
        tw.append(float(np.max(d if e is None else d[:e])))
    thr = float(np.percentile(tw, 90))
    tp = 0
    for s, e in zip(hack_series, events):
        if e is None:
            continue
        idx = np.nonzero(up_dev(s) > thr)[0]
        tp += int(len(idx) > 0 and idx[0] < e)
    fp = sum(v > thr for v in tw)
    return tp, fp


def simulate_T(ev, draws, rng):
    n = len(ev)
    room = [e is not None and e >= 7 for e in ev]
    q = min(1.0, 0.5 * n / sum(room))
    rej = 0
    fps = set()
    for _ in range(draws):
        hs, ts = [], []
        for e, ok in zip(ev, room):
            x = rng.uniform(0, 1, T_STEPS)
            if ok and rng.random() < q:
                x[e - int(rng.integers(1, min(12, e - 6) + 1))] = rng.uniform(5, 6)
            hs.append(x)
            y = rng.uniform(0, 1, T_STEPS)
            w = T_STEPS if e is None else e
            if w > 6:
                y[int(rng.integers(6, w))] = rng.uniform(2.5, 3.5)
            ts.append(y)
        tp, fp = score_tm(hs, ts, ev)
        fps.add(fp)
        rej += int(P.fisher_one_sided(tp, n, fp, n) < P.ALPHA)   # window and lead granted, as in E1's control
    return rej / draws, sorted(fps)


def main():
    d = json.loads((HERE / "e1_inputs.json").read_text())
    events = d["events"]
    rng = np.random.default_rng(20261009)
    ev24 = list(events.values())
    print("PLANTED CONTROL, E1b. Exact expectation = e1_primary.power_poisson_binomial(p_i, n, fixed_fp(n)).")
    print("E1's realized layout (24 events, one none): E1 construction %.3f (registered expectation 0.567); "
          "R %.3f; T %.3f (q = %.3f, %d runs with room)" % (
              P.power_poisson_binomial(ps_E1(ev24), 24, 3), P.power_poisson_binomial(ps_R(ev24), 24, 3),
              P.power_poisson_binomial(ps_T(ev24), 24, 3), max(ps_T(ev24)), sum(1 for p in ps_T(ev24) if p > 0)))
    print("mean planted TPR at E1's layout: E1 %.3f, R %.3f, T %.3f\n" % (
        np.mean(ps_E1(ev24)), np.mean(ps_R(ev24)), np.mean(ps_T(ev24))))
    res = {}
    for n in NS:
        fp = P.fixed_fp(n)
        L = layouts(events, n, rng, LAYOUTS)
        row = {}
        fm = (lambda m: lambda ev: ps_T(ev, m))(MDE80[n])
        for name, f in (("E1", ps_E1), ("R", ps_R), ("T", ps_T), ("M", fm)):
            vals = np.array([P.power_poisson_binomial(f(ev), n, fp) for ev in L])
            tprs = np.array([np.mean(f(ev)) for ev in L])
            row[name] = {"mean": float(vals.mean()), "p10": float(np.percentile(vals, 10)),
                         "p90": float(np.percentile(vals, 90)), "mean_planted_tpr": float(tprs.mean())}
        row["binomial_0.5"] = P.power_fixed_fp(n, n, 0.5, fp)
        ev = L[0]
        exp_T = P.power_poisson_binomial(ps_T(ev), n, fp)
        sim, fps = simulate_T(ev, 1000, np.random.default_rng(n))
        row["sim_check_T"] = {"layout_events": sorted(-1 if e is None else e for e in ev), "expected": exp_T,
                              "simulated_1000": sim, "fp_values_seen": fps, "passes_0.05": abs(sim - exp_T) <= 0.05}
        res[n] = row
        print("n = %d per arm (FP fixed at %d/%d), %d bootstrap event layouts:" % (n, fp, n, LAYOUTS))
        for name in ("E1", "R", "T", "M"):
            v = row[name]
            print("  %-2s expected rejection mean %.3f (10th pct %.3f, 90th %.3f); mean planted TPR %.3f" % (
                name, v["mean"], v["p10"], v["p90"], v["mean_planted_tpr"]))
        print("  reference: binomial TPR 0.5 at fixed FP: %.3f" % row["binomial_0.5"])
        print("  simulation check of T through the E1b scoring path (one layout, 1000 draws): %.3f vs exact %.3f -> %s; "
              "FP counts seen %s" % (sim, exp_T, "PASSES" if row["sim_check_T"]["passes_0.05"] else "FAILS", fps))
    (HERE / "planted_control.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
