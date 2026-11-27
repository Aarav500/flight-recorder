"""THROWAWAY SPIKE, written AFTER seeing the verdict, so nothing here counts toward it. Never reads phi.jsonl."""
import json, re, sys
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy import stats as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step0_llm import extract_code
ROOT = Path(__file__).resolve().parents[1] / "runs"
norm = lambda s: re.sub(r"\s+", " ", extract_code(s)).strip()

print("POST HOC oracle slope per 100 steps over 25-150, and the windows around it")
for seed in (300, 301, 302):
    d = ROOT / f"tr_csv_narrow_s{seed}" / "out" / f"csv_narrow_s{seed}"
    S = [json.loads(x) for x in (d / "steps.jsonl").read_text().splitlines()]
    C = [json.loads(x) for x in (d / "completions.jsonl").read_text().splitlines()]
    o = np.array([r["oracle_mean"] for r in S])
    r = st.linregress(np.arange(25, 150), o[25:150])
    win = lambda a, b: o[a:b].mean()
    se = lambda a, b: np.std(np.concatenate([x["oracle"] for x in S[a:b]])) / np.sqrt(8 * (b - a))
    print(f"seed {seed}: slope {r.slope * 100:+.3f} (p {r.pvalue:.2g}); oracle 25-49 {win(25, 50):.3f} (se {se(25, 50):.3f}), "
          f"100-149 {win(100, 150):.3f}, 300-399 {win(300, 400):.3f} (se {se(300, 400):.3f})")
    k1, k2 = np.array([r["kl1"] for r in S]), np.array([r["kl2"] for r in S])
    print(f"   KL over steps 25-399: agent 1 {k1[25:].mean():.4f}, agent 2 {k2[25:].mean():.4f}; largest 10-step KL1 at step "
          f"{int(np.argmax(np.convolve(k1, np.ones(10) / 10, 'valid')))} = {np.convolve(k1, np.ones(10) / 10, 'valid').max():.3f}")
    late = C[300:400]
    u1 = np.mean([len({norm(a) for a in c["a1"]}) for c in late])
    u2 = np.mean([len({norm(a) for a in c["a2"]}) for c in late])
    # within a late group, does the oracle track which helper was written, or which main?
    same_m, diff_h, tot = 0, 0, 0
    for c, s in zip(late, S[300:400]):
        mains = [norm(a) for a in c["a2"]]
        top = max(set(mains), key=mains.count)
        idx = [i for i, m in enumerate(mains) if m == top]
        vals = [s["oracle"][i] for i in idx]
        if len(idx) >= 2:
            tot += 1
            diff_h += len(set(vals)) > 1
    print(f"   late groups (of 8): {u1:.1f} distinct helpers, {u2:.1f} distinct mains. Among groups with a repeated main, "
          f"oracle still varies with the helper in {diff_h}/{tot}")

price = {"g6e.xlarge": 1.861, "g6e.2xlarge": 2.242, "c7i.16xlarge": 2.856, "g6e.4xlarge": 3.004}
end = {"jax": "2026-10-30T20:53:27", "llm": "2026-10-30T21:33:31", "jax2": "2026-10-30T23:25:43", "llm2": "2026-10-30T23:40:10",
       "trcal": "2026-10-31T01:01:50", "tr_csv_narrow_s300": "2026-10-31T03:28:33", "tr_csv_narrow_s301": "2026-10-31T03:40:05",
       "tr_csv_narrow_s302": "2026-10-31T03:47:56"}
tot = 0
for k, e in end.items():
    r = json.load(open(ROOT / f"{k}_launch.json"))
    h = (datetime.fromisoformat(e) - datetime.fromisoformat(r["launched"]).replace(tzinfo=None)).total_seconds() / 3600
    tot += h * price[r["instance_type"]]
    print(f"{k:20s} {r['instance_type']:12s} {h:5.2f} h ${h * price[r['instance_type']]:.2f}")
print(f"total ${tot:.2f}")
