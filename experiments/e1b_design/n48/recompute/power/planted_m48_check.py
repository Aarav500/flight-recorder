"""Construction M at n = 48: exact expectation vs a 1000-draw simulation through planted_control.py's E1b scoring path,
on an E1-like bootstrap layout and two extreme layouts where the window holds (48 and 24 runs with room).
Also the tolerance in SD units and the exact chance failure of a correct pipeline at the extreme expectations."""
import math
import sys

sys.dont_write_bytecode = True
import json  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import binom  # noqa: E402
from planted_control import P, score_tm, layouts, HERE, T_STEPS  # noqa: E402

N = 48
FP = P.fixed_fp(N)
MDE = P.mde_fixed_fp(N, N, FP)


def ps_M(ev):
    room = sum(1 for e in ev if e is not None and e >= 7)
    q = min(1.0, MDE * len(ev) / room) if room else 0.0
    return [q if (e is not None and e >= 7) else 0.0 for e in ev]


def simulate_M(ev, draws, rng):
    n = len(ev)
    ps = ps_M(ev)
    rej, fps = 0, set()
    for _ in range(draws):
        hs, ts = [], []
        for e, p in zip(ev, ps):
            x = rng.uniform(0, 1, T_STEPS)
            if p > 0 and rng.random() < p:
                x[e - int(rng.integers(1, min(12, e - 6) + 1))] = rng.uniform(5, 6)
            hs.append(x)
            y = rng.uniform(0, 1, T_STEPS)
            w = T_STEPS if e is None else e
            if w > 6:
                y[int(rng.integers(6, w))] = rng.uniform(2.5, 3.5)
            ts.append(y)
        tp, fp = score_tm(hs, ts, ev)
        fps.add(fp)
        rej += int(P.fisher_one_sided(tp, n, fp, n) < P.ALPHA)
    return rej / draws, sorted(fps)


d = json.loads((HERE / "e1_inputs.json").read_text())
e1like = layouts(d["events"], N, np.random.default_rng(11), 1)[0]
all_room = [40] * N                                  # every run has room (late events)
half_room = [6] * 24 + [30] * 24                     # 24 with room; median event (6 + 30) / 2 = 18 -> window holds
for name, ev in (("E1-like bootstrap layout", e1like), ("48 with room", all_room),
                 ("24 with room, window holds", half_room)):
    exp = P.power_poisson_binomial(ps_M(ev), N, FP)
    sim, fps = simulate_M(ev, 1000, np.random.default_rng(N))
    med = float(np.median([-1 if e is None else e for e in ev]))
    print("%-28s n_room %2d, q %.3f, median event %.1f: exact %.4f, simulated %.3f, |diff| %.3f -> %s; FP seen %s" % (
        name, sum(1 for e in ev if e is not None and e >= 7), max(ps_M(ev)), med, exp, sim, abs(sim - exp),
        "PASSES" if abs(sim - exp) <= 0.05 + 1e-9 else "FAILS", fps))
for p in (0.8107, 0.82, 0.8868):
    sd = math.sqrt(p * (1 - p) / 1000)
    fail = sum(binom.pmf(x, 1000, p) for x in range(1001) if abs(x / 1000 - p) > 0.05 + 1e-9)
    print("expectation %.4f: SD %.4f; tolerance %.2f SD; exact P(correct pipeline fails) %.1e" % (p, sd, 0.05 / sd, fail))
