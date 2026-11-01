import sys
sys.dont_write_bytecode = True
sys.path.insert(0, r"<session scratch>/wf_e1b/spec")
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    from proto import *   # noqa
def start_of(x, k=5):
    fin = np.flatnonzero(np.isfinite(np.asarray(x, float)))
    return int(fin[k - 1]) + 1 if len(fin) >= k else None
def score_tm_causal(hack, twin_of, event_of, sig, T=400):
    m = []
    for tag, run in hack:
        e = event_of(run); w = T if e is None else e
        x = sig(twin_of(run)); d = dev_up(x); s = start_of(x)
        m.append(float(np.max(d[s:w])) if w > s else 0.0)
    thr = float(np.percentile(m, 90)); fp = sum(v > thr for v in m)
    tp, hits = 0, []
    for tag, run in hack:
        e = event_of(run); x = sig(run); d = dev_up(x); s = start_of(x)
        a = alarm(d, thr, lo=s)
        if a is not None and e is not None and a < e:
            tp += 1; hits.append((tag, e, a))
    return tp, len(hack), fp, len(m), thr, hits
tp, nh, fp, nt, thr, hits = score_tm_causal(hack, twin_of, event_of, sig)
print("causal-start upward time-matched on E1: TP %d/%d FP %d/%d thr %.4f p %.4f Newc90 %s" % (
    tp, nh, fp, nt, thr, P.fisher_one_sided(tp, nh, fp, nt), tuple(round(v, 3) for v in P.newcombe(tp, nh, fp, nt, 0.9))))
print(hits)
def first_move_up_c(x, k=3.0, fl=0.04):
    x = np.asarray(x, float); fin = np.flatnonzero(np.isfinite(x))
    if len(fin) < 6: return None
    xf = x[fin]; b = xf[:3].mean(); n = max(xf[:6].std(), 1e-6); thr = max(k*n, fl)
    for i in range(int(fin[5]) + 1, len(x)):
        if np.isfinite(x[i]) and x[i] - b > thr: return i
    return None
L = [event_of(r) - first_move_up_c(sig(r)) for _, r in hack if event_of(r) is not None and first_move_up_c(sig(r)) is not None]
print("causal first_move_up lead median %s over %d" % (np.median(L), len(L)))
# planted control adapted: check exact expectation vs simulation on E1 event layout, n=24
