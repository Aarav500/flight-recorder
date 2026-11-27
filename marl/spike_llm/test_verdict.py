"""verdict_train.decide on synthetic runs, one per branch. Run: python test_verdict.py"""
import numpy as np

import verdict_train as V

T = np.arange(400)


def run(px0, px1, or0, or1, seed):
    rng = np.random.default_rng(seed)
    n = lambda: rng.normal(0, 0.05, 400)
    return (np.clip(np.linspace(px0, px1, 400) + n(), 0, 1), np.clip(np.linspace(or0, or1, 400) + n(), 0, 1))


def grid(narrow, ctrl):
    out = {}
    for i, s in enumerate(V.SEEDS):
        out[("csv_narrow", s)] = run(*narrow, 10 + i)
        out[("csv_exchangeable", s)] = run(*ctrl, 20 + i)
    return out


def main():
    hack = (0.3, 0.9, 0.2, 0.02)
    flat_up = (0.3, 0.8, 0.2, 0.6)
    gap = (0.3, 0.9, 0.1, 0.1)
    still = (0.3, 0.3, 0.1, 0.1)
    assert V.decide(grid(hack, flat_up))[0] == "HACK"
    assert V.decide(grid(hack, hack))[0] != "HACK"      # a control that hacks too is no result
    assert V.decide(grid(gap, flat_up))[0] == "GAP"
    assert V.decide(grid(flat_up, flat_up))[0] == "NULL"
    assert V.decide(grid(still, still))[0] == "UNDER-TRAINED"
    trap = lambda args: {("csv_narrow", s): run(*args, 10 + i) for i, s in enumerate(V.SEEDS)}
    assert V.decide_trap_only(trap(hack))[0] == "RUN CONTROLS"
    assert V.decide_trap_only(trap(gap))[0] == "RUN CONTROLS"
    assert V.decide_trap_only(trap(flat_up))[0] == "NULL (trap only)"
    assert V.decide_trap_only(trap(still))[0] == "UNDER-TRAINED"
    # a fast fall that finishes before step 150 and rests on the floor
    rng = np.random.default_rng(1)
    fast = np.concatenate([np.linspace(0.2, 0.0, 100), np.zeros(300)]) + np.abs(rng.normal(0, 0.01, 400))
    assert V.falls(fast)
    # a rise then a fall, which only the short window sees
    hump = np.concatenate([np.linspace(0.1, 0.5, 150), np.linspace(0.5, 0.3, 250)]) + rng.normal(0, 0.03, 400)
    assert V.falls(hump) and any(s < 0 for s, _ in V.slopes(hump))
    print("ok")


if __name__ == "__main__":
    main()
