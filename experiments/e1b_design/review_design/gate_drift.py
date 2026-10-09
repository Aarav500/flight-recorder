"""How much does the gate fail share move step-mean disagreement? E1 data (seen), exploratory."""
import gzip, json
from collections import defaultdict
from pathlib import Path
import numpy as np
REPO = Path(r"<home>/OneDrive/Desktop/Fligh recorder")
feat = defaultdict(list)
with gzip.open(Path(__file__).resolve().parents[1] / "twin/features.jsonl.gz", "rt") as f:
    for l in f:
        r = json.loads(l)
        feat[(r["arm"], r["task"], r["seed"], r["step"])].append(float(r["parses"] and r["defines"]))
rows = []
for fp in sorted((REPO / "runs/e1/main").glob("judge_judge*_s2*_lr1e-6.json")):
    d = json.loads(fp.read_text())
    arm = "hack" if "judge05" in fp.stem else "benign"
    for r in d["log"]:
        g = feat.get((arm, d["task"], d["seed"], r["step"]))
        if g is None or r.get("disagree") is None:
            continue
        rows.append((arm, d["task"], d["seed"], r["step"], 1 - np.mean(g), r["disagree"], r["judge_small"], r["judge_big"]))
print("rows", len(rows))
def fit(sel, label, fe=True):
    sel = list(sel)
    y = np.array([r[5] for r in sel]); x = np.array([r[4] for r in sel])
    js = np.array([r[6] for r in sel]); jb = np.array([r[7] for r in sel])
    keys = [(r[0], r[1], r[2]) for r in sel] if fe else [r[1] for r in sel]
    # demean within key (fixed effects)
    def dm(v):
        out = v.copy().astype(float); groups = defaultdict(list)
        for i, k in enumerate(keys): groups[k].append(i)
        for k, idx in groups.items(): out[idx] -= v[idx].mean()
        return out
    X, Y, JS, JB = dm(x), dm(y), dm(js), dm(jb)
    b = (X @ Y) / (X @ X); bs = (X @ JS) / (X @ X); bb = (X @ JB) / (X @ X)
    res = Y - b * X; se = np.sqrt(res @ res / (len(Y) - 2) / (X @ X))
    print("%-48s n=%5d  d(disagree)/d(fail share) = %+.3f (se %.3f)  [J05 %+.3f, J15 %+.3f]" % (label, len(Y), b, se, bs, bb))
    return b
fit([r for r in rows if r[3] <= 4], "steps 0-4, both arms, task FE", fe=False)
fit([r for r in rows if r[3] <= 4], "steps 0-4, both arms, run FE")
fit([r for r in rows if r[3] <= 24], "steps 0-24, both arms, run FE")
fit([r for r in rows if r[0] == "benign"], "benign arm, all steps, run FE")
fit([r for r in rows if r[0] == "hack"], "hacking arm, all steps, run FE")
# twin-like drift: E1 benign fail share early vs late
for arm in ("hack", "benign"):
    e = np.mean([r[4] for r in rows if r[0] == arm and r[3] <= 24]); l = np.mean([r[4] for r in rows if r[0] == arm and r[3] >= 375])
    e5 = np.mean([r[4] for r in rows if r[0] == arm and r[3] <= 4]); m = np.mean([r[4] for r in rows if r[0] == arm and 25 <= r[3] <= 49])
    print(arm, "fail share steps 0-4 %.3f, 0-24 %.3f, 25-49 %.3f, 375-399 %.3f" % (e5, e, m, l))
# when does the benign arm's fail share drop? median step at which a run's 10-step mean fail share first < 0.10
for arm in ("benign", "hack"):
    out = []
    for key in sorted({(r[1], r[2]) for r in rows if r[0] == arm}):
        s = sorted([r for r in rows if r[0] == arm and (r[1], r[2]) == key], key=lambda r: r[3])
        f = np.array([r[4] for r in s])
        sm = np.convolve(f, np.ones(10) / 10, mode="valid")
        hit = np.where(sm < 0.10)[0] if arm == "benign" else np.where(sm > 0.9)[0]
        out.append(int(hit[0]) + 9 if len(hit) else None)
    print(arm, "first step where 10-step mean fail share <0.10 (benign) / >0.90 (hack):", out)
