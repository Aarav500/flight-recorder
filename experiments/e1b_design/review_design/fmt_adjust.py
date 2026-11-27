"""EXPLORATORY, E1 data (seen). How much of the low upward threshold comes from the twin-side format drift?
Remove b * (fail share_t - fail share base) from the stand-in twins' disagreement and recompute the E1b rule."""
import gzip, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
REPO = Path(r"<home>/OneDrive/Desktop/Fligh recorder")
sys.path.insert(0, str(REPO / "experiments"))
import e1_primary as P, e1_event as EV, e1_detectors as D
feat = defaultdict(list)
with gzip.open(Path(__file__).resolve().parents[1] / "twin/features.jsonl.gz", "rt") as f:
    for l in f:
        r = json.loads(l)
        feat[(r["arm"], r["task"], r["seed"], r["step"])].append(float(r["parses"] and r["defines"]))
used, *_ = P.load_main()
hack = [(r["_tag"], r) for r in used["hacking"]]
bm = {(r["task"], int(r["seed"])): r for r in used["benign"]}
prim = next(m for (l, f, g, m) in D.detectors() if l == P.PRIMARY)
ev = {t: EV.event(list(range(len(r["log"]))), [x.get("hidden") for x in r["log"]]) for t, r in hack}
def fail_share(arm, r):
    return np.array([1 - np.mean(feat[(arm, r["task"], r["seed"], s)]) for s in range(len(r["log"]))])
def up_dev(x):
    x = np.asarray(x, float)
    return np.maximum(x - np.median(x[np.isfinite(x)][:5]), 0.0)
def rule(b_twin, start=5):
    mx = []
    for t, r in hack:
        tw = bm[(r["task"], int(r["seed"]))]
        x = np.asarray(prim(tw), float)
        fs = fail_share("benign", tw)
        x = x - b_twin * (fs - np.median(fs[:5]))
        d = up_dev(x); e = ev[t]
        w = d[start:] if e is None else d[start:e]
        mx.append(float(np.max(w)) if len(w) else 0.0)
    thr = float(np.percentile(mx, 90))
    tp = 0
    for t, r in hack:
        hd = up_dev(prim(r)); idx = [i for i in np.nonzero(hd > thr)[0] if i >= start]
        tp += int(ev[t] is not None and bool(idx) and idx[0] < ev[t])
    fp = sum(v > thr for v in mx)
    print("[EXPLORATORY] twin format adjustment b=%.2f: thr %.4f TP %d/24 FP %d/24 p %.4f" % (b_twin, thr, tp, fp, P.fisher_one_sided(tp, 24, fp, 24)))
for b in (0.0, 0.10, 0.177, 0.30):
    rule(b)
# drift of stand-in twins' disagreement explained by format, steps 0-4 to 45-55
dd, ff = [], []
for r in used["benign"]:
    x = np.asarray(prim(r), float); fs = fail_share("benign", r)
    dd.append(np.nanmean(x[45:55]) - np.nanmean(x[:5])); ff.append(np.mean(fs[45:55]) - np.mean(fs[:5]))
print("[EXPLORATORY] E1 benign, steps 0-4 to 45-54: disagreement change mean %.3f; fail-share change mean %.3f; format part at b=0.177: %.3f" % (np.mean(dd), np.mean(ff), 0.177 * np.mean(ff)))
