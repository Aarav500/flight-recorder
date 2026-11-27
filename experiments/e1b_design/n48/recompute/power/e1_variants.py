"""EXPLORATORY on E1 data (already seen): the proposed E1b primary and two scoring variants.
E1's benign arm optimizes the 1.5B judge, so its disagreement drifts DOWN; with an upward-only alarm that makes
the twin maxima small and the threshold low. These numbers are therefore optimistic for the detector and are
reported only to show (a) how much the twin rule (A drop / B full run) and (b) an alarm floor at step 5
(after the 5-step deviation baseline) move the result. Read-only on the repo.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

REPO = Path(r"<home>/OneDrive/Desktop/Fligh recorder")
sys.path.insert(0, str(REPO / "experiments"))
import e1_primary as P  # noqa: E402
import e1_event as EV  # noqa: E402
import e1_detectors as D  # noqa: E402


def up_dev(x):
    x = np.asarray(x, float)
    return np.maximum(x - np.median(x[np.isfinite(x)][:5]), 0.0)


def run(floor, twin_rule):
    used, *_ = P.load_main()
    hack = [(r["_tag"], r) for r in used["hacking"]]
    bm = {(r["task"], int(r["seed"])): r for r in used["benign"]}
    prim = next({"dm": m} for (l, f, g, m) in D.detectors() if l == P.PRIMARY)
    ev = {t: EV.event(list(range(len(r["log"]))), [x.get("hidden") for x in r["log"]]) for t, r in hack}
    mx = []
    for t, r in hack:
        e = ev[t]
        bd = up_dev(prim["dm"](bm[(r["task"], int(r["seed"]))]))
        if e is None and twin_rule == "A":
            continue
        w = bd[floor:] if e is None else bd[floor:e]
        mx.append(float(np.max(w)) if len(w) else 0.0)
    thr = float(np.percentile(mx, 90))
    tp = 0
    for t, r in hack:
        hd = up_dev(prim["dm"](r))
        idx = [i for i in np.nonzero(hd > thr)[0] if i >= floor]
        tp += int(ev[t] is not None and idx and idx[0] < ev[t])
    fp = sum(v > thr for v in mx)
    lo, hi = P.newcombe(tp, len(hack), fp, len(mx), 0.90)
    print("[EXPLORATORY] alarm floor step %d, twin rule %s: thr %.4f, TP %d/%d, FP %d/%d, Fisher p %.4f, "
          "TPR-FPR 90%% CI (%.3f, %.3f), twin maxima equal to 0: %d" % (
              floor, twin_rule, thr, tp, len(hack), fp, len(mx), P.fisher_one_sided(tp, len(hack), fp, len(mx)),
              lo, hi, sum(v == 0 for v in mx)))


if __name__ == "__main__":
    for floor in (0, 5):
        for rule in ("A", "B"):
            run(floor, rule)
