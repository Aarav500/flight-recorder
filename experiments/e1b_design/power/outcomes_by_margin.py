"""Three-way outcome (kill-test significant / non-superiority null / inconclusive) by margin, FP fixed."""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, r"experiments")
import e1_primary as P  # noqa: E402
from power_primary import outcomes  # noqa: E402

for n, deltas in ((24, (0.25, 0.20)), (36, (0.25, 0.20)), (48, (0.25, 0.20, 0.15))):
    fp = P.fixed_fp(n)
    for dl in deltas:
        mh = max([x for x in range(n + 1) if P.newcombe(x, n, fp, n, 0.90)[1] < dl], default=None)
        row = ["TPR %.3f: %.2f/%.2f/%.2f" % ((t,) + outcomes(n, fp, t, dl)) for t in (fp / n, 0.20, 0.25, 0.30, 0.375, 0.50)]
        print("n=%d delta %.2f (non-sup iff TP <= %s, kill iff TP >= %s): %s" % (
            n, dl, mh, next(x for x in range(n + 1) if P.fisher_one_sided(x, n, fp, n) < 0.05), "; ".join(row)))
