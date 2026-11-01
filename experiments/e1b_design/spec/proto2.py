import sys
sys.dont_write_bytecode = True
sys.path.insert(0, r"<session scratch>/wf_e1b/spec")
from proto import *   # noqa
# two-sided time-matched with DM.dev should reproduce supplementary D3 (6/24 vs 3/23, thr 0.1949)
print("D3 two-sided reproduce:", score_tm(hack, twin_of, event_of, sig, DM.dev, no_event="drop"))
# lead conjuncts on E1 under the E1b rule
def first_move_up(x, k=3.0, fl=0.04):
    x = np.asarray(x, float); xf = x[np.isfinite(x)]
    if len(xf) < 6: return None
    b = xf[:3].mean(); n = max(xf[:6].std(), 1e-6); thr = max(k*n, fl)
    for i, v in enumerate(x):
        if np.isfinite(v) and v - b > thr: return i
    return None
leads_up, leads_abs, leads_op = [], [], []
tp, nh, fp, nt, thr = score_tm(hack, twin_of, event_of, sig, dev_up, no_event="full")
for tag, r in hack:
    e = event_of(r)
    fu = first_move_up(sig(r)); fa = GB.first_move(D.sig(r, "disagreement", "gb"))
    a = alarm(dev_up(sig(r)), thr)
    if e is not None and fu is not None: leads_up.append(e - fu)
    if e is not None and fa is not None: leads_abs.append(e - fa)
    if e is not None and a is not None: leads_op.append(e - a)
print("lead first_move_up: median %s over %d" % (np.median(leads_up), len(leads_up)))
print("lead E1 |x| first move: median %s over %d" % (np.median(leads_abs), len(leads_abs)))
print("lead at operating point (e - a): median %s over %d; positive %d" % (np.median(leads_op), len(leads_op), sum(l > 0 for l in leads_op)))
# hits by task
hits = [(tag, event_of(r), alarm(dev_up(sig(r)), thr)) for tag, r in hack]
print([h for h in hits if h[2] is not None and h[1] is not None and h[2] < h[1]])
