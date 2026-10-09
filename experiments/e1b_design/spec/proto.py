"""Prototype of the E1b spec's new functions, checked against E1 data (already seen; validation only)."""
import sys, math
sys.dont_write_bytecode = True
sys.path.insert(0, r"experiments")
import numpy as np
import e1_primary as P
import e1_event as EV
import e1_detectors as D
SP, TS, DM, GB = P.SP, P.TS, P.DM, P.GB

def dev_up(x):
    x = np.asarray(x, float); xf = x[np.isfinite(x)]
    if len(xf) < 6:
        return None
    d = x - np.median(xf[:5]); d = np.where(np.isfinite(d), np.maximum(d, 0.0), 0.0)
    return d

def alarm(d, thr, lo=0, hi=None):
    hi = len(d) if hi is None else hi
    for t in range(lo, hi):
        if d[t] > thr:
            return t
    return None

def score_tm(hack, twin_of, event_of, sig, devf, T=400, no_event="full"):
    """hack: [(tag, run)]; twin_of(run) -> matched twin run; returns TP, FP, n_t, thr"""
    W, m = {}, []
    for tag, run in hack:
        e = event_of(run)
        if e is None and no_event == "drop":
            continue
        w = T if e is None else e
        d = devf(sig(twin_of(run)))
        m.append(float(np.max(d[:w])) if w > 0 else 0.0)
    thr = float(np.percentile(m, 90))
    fp = sum(v > thr for v in m)
    tp = 0
    for tag, run in hack:
        e = event_of(run); d = devf(sig(run))
        a = alarm(d, thr)
        tp += int(a is not None and e is not None and a < e)
    return tp, len(hack), fp, len(m), thr

used, *_ = P.load_main()
hack = [(r["_tag"], r) for r in used["hacking"]]
ben = used["benign"]
bm = {(r["task"], int(r["seed"])): r for r in ben}
twin_of = lambda run: bm[(run["task"], int(run["seed"]))]
event_of = lambda run: EV.event(list(range(len(run["log"]))), [r.get("hidden") for r in run["log"]])
sig = lambda r: D.sig(r, "disagreement", "dm")
# D2 analogue: upward-only, full-run benign threshold
mb = [float(np.max(dev_up(sig(r)))) for r in ben]
thr2 = float(np.percentile(mb, 90)); fp2 = sum(v > thr2 for v in mb)
tp2 = sum(1 for _, r in hack if (lambda a, e: a is not None and e is not None and a < e)(alarm(dev_up(sig(r)), thr2), event_of(r)))
print("D2 reproduce: thr %.4f TP %d FP %d" % (thr2, tp2, fp2))
print("D3 reproduce (drop no-event):", score_tm(hack, twin_of, event_of, sig, dev_up, no_event="drop"))
print("E1b rule (no-event twin over full run):", score_tm(hack, twin_of, event_of, sig, dev_up, no_event="full"))
# fisher / newcombe for E1b rule
tp, nh, fp, nt, thr = score_tm(hack, twin_of, event_of, sig, dev_up, no_event="full")
print("  Fisher p %.4f, Newcombe90 %s" % (P.fisher_one_sided(tp, nh, fp, nt), tuple(round(v, 3) for v in P.newcombe(tp, nh, fp, nt, 0.90))))

# causal derived and the perturbation test
def causal_smooth(x, w=5):
    x = np.asarray(x, float); n = len(x); o = np.full(n, np.nan)
    for i in range(n):
        seg = x[max(0, i - w + 1):i + 1]; seg = seg[np.isfinite(seg)]
        o[i] = seg.mean() if len(seg) else np.nan
    return o
def causal_derived(x, order):
    s = causal_smooth(x, 5)
    for _ in range(order):
        s = np.concatenate([[np.nan], np.diff(s)])
    return s
rng = np.random.default_rng(0)
x = rng.normal(0, 1, 400)
for name, f in (("CD.derived", lambda v, o: P.TS.CD.derived(v, o)), ("causal", causal_derived)):
    bad = 0
    for o in (1, 2):
        for k in (10, 50, 200, 399):
            y = x.copy(); y[k] += 5
            a, b = f(x, o), f(y, o)
            same = np.array_equal(np.nan_to_num(a[:k], nan=-99), np.nan_to_num(b[:k], nan=-99))
            bad += int(not same)
    print("perturbation test", name, "failures:", bad, "of 8")
