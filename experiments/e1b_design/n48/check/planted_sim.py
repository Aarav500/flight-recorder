"""Own vectorized simulation of construction M at n = 48 (upward dev from median of first 5, time-matched twin
windows, 90th pct threshold, alarm strictly before event), vs the exact binomial expectation."""
import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
N, T, K, MDE = 48, 400, 13, 0.32


def fisher_p(tp, fp):
    return stats.hypergeom(2 * N, tp + fp, N).sf(tp - 1)


def run(ev, draws, rng):
    ev_arr = np.array([-1 if e is None else e for e in ev])
    room = ev_arr >= 7
    q = min(1.0, MDE * N / room.sum())
    W = np.where(ev_arr < 0, T, ev_arr)
    rej = 0
    fps = set()
    for _ in range(draws):
        h = rng.uniform(0, 1, (N, T))
        t = rng.uniform(0, 1, (N, T))
        plant = room & (rng.random(N) < q)
        for i in np.nonzero(plant)[0]:
            e = ev_arr[i]
            h[i, e - rng.integers(1, min(12, e - 6) + 1)] = rng.uniform(5, 6)
        for j in range(N):
            if W[j] > 6:
                t[j, rng.integers(6, W[j])] = rng.uniform(2.5, 3.5)
        hd = np.maximum(h - np.median(h[:, :5], axis=1, keepdims=True), 0)
        td = np.maximum(t - np.median(t[:, :5], axis=1, keepdims=True), 0)
        m = np.array([td[j, :W[j]].max() if W[j] > 0 else 0.0 for j in range(N)])
        thr = np.percentile(m, 90)
        fp = int((m > thr).sum())
        fps.add(fp)
        tp = 0
        for i in range(N):
            if ev_arr[i] < 0:
                continue
            idx = np.nonzero(hd[i] > thr)[0]
            tp += int(len(idx) > 0 and idx[0] < ev_arr[i])
        rej += int(fisher_p(tp, fp) < 0.05)
    return rej / draws, sorted(fps), float(stats.binom.sf(K - 1, int(room.sum()), q))


ev = json.loads((HERE / "e1_inputs.json").read_text())["events"]
by = {}
for tg, e in ev.items():
    by.setdefault(tg.split("/")[1], []).append(e)
rng = np.random.default_rng(99)
lay = []
for task, es in by.items():
    lay += [es[i] for i in rng.integers(0, len(es), N // 3)]
for name, L in (("E1-like", lay), ("48 room", [40] * 48), ("24 room", [6] * 24 + [30] * 24)):
    sim, fps, exp = run(L, 1000, np.random.default_rng(5))
    print("%-8s sim %.3f exact %.4f |d| %.3f FP seen %s" % (name, sim, exp, abs(sim - exp), fps))
