import sys, io, contextlib
sys.dont_write_bytecode = True
sys.path.insert(0, r"<session scratch>/wf_e1b/spec")
with contextlib.redirect_stdout(io.StringIO()):
    from proto import *  # noqa
m = [float(np.max(DM.dev(sig(twin_of(r))))) for _, r in hack]
thr = float(np.percentile(m, 90)); fp = sum(v > thr for v in m)
tp = sum(1 for _, r in hack if (lambda a, e: a is not None and e is not None and a < e)(alarm(DM.dev(sig(r)), thr), event_of(r)))
print("full-run two-sided start0: thr %.4f TP %d FP %d" % (thr, tp, fp))
