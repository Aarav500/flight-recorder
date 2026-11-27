"""THROWAWAY SPIKE, written after the verdict, so nothing here changes it. Never reads phi.jsonl."""
import json, re, sys
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy import stats as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step0_llm import extract_code
ROOT = Path(__file__).resolve().parents[1] / "runs"
norm = lambda s: re.sub(r"\s+", " ", extract_code(s)).strip()
WINS = [(0, 5), (5, 25), (25, 50), (50, 100), (100, 150), (150, 200), (200, 300), (300, 400)]


def load(task, seed):
    d = ROOT / f"tr_{task}_s{seed}" / "out" / f"{task}_s{seed}"
    rd = lambda n: [json.loads(x) for x in (d / n).read_text().splitlines() if x.strip()]
    return rd("steps.jsonl"), rd("completions.jsonl")


data = {t: {s: load(t, s) for s in (300, 301, 302)} for t in ("csv_narrow", "csv_exchangeable")}
print("oracle by window (each seed), proxy mean, exploit rate")
print(f"{'steps':>8s} | {'narrow oracle':>20s} {'px':>5s} {'expl':>5s} | {'control oracle':>20s} {'px':>5s} {'expl':>5s}")
for a, b in WINS:
    row = []
    for t in ("csv_narrow", "csv_exchangeable"):
        o = [np.mean([r["oracle_mean"] for r in data[t][s][0][a:b]]) for s in (300, 301, 302)]
        p = np.mean([r["proxy_mean"] for s in (300, 301, 302) for r in data[t][s][0][a:b]])
        v = np.array([r["proxy"] for s in (300, 301, 302) for r in data[t][s][0][a:b]])
        h = np.array([r["oracle"] for s in (300, 301, 302) for r in data[t][s][0][a:b]])
        row.append(f"{o[0]:6.3f} {o[1]:6.3f} {o[2]:6.3f} {p:5.2f} {((v >= .8) & (h < .5)).mean():5.2f}")
    print(f"{a:3d}-{b - 1:<4d} | {row[0]} | {row[1]}")

print("\nPOST HOC oracle slope per 100 steps over 25-150 (not counted)")
for t in data:
    for s in data[t]:
        o = np.array([r["oracle_mean"] for r in data[t][s][0]])
        r = st.linregress(np.arange(25, 150), o[25:150])
        print(f"{t:17s} {s}: {r.slope * 100:+.3f} (p {r.pvalue:.2g})")

print("\nwho moved: KL over steps 25-399, late distinct helpers and mains of 8, main calls helper")
for t in data:
    for s in data[t]:
        S, C = data[t][s]
        late = C[300:400]
        calls = np.mean([("next_field(" in extract_code(a) and "def next_field" not in extract_code(a)) for c in late for a in c["a2"]])
        print(f"{t:17s} {s}: KL1 {np.mean([r['kl1'] for r in S[25:]]):.4f} KL2 {np.mean([r['kl2'] for r in S[25:]]):.4f} | "
              f"helpers {np.mean([len({norm(a) for a in c['a1']}) for c in late]):.1f} mains {np.mean([len({norm(a) for a in c['a2']}) for c in late]):.1f} | calls {calls:.2f}")

n = [np.mean([r["oracle_mean"] for r in data["csv_narrow"][s][0][300:400]]) for s in (300, 301, 302)]
c = [np.mean([r["oracle_mean"] for r in data["csv_exchangeable"][s][0][300:400]]) for s in (300, 301, 302)]
print(f"\nfinal oracle narrow {np.round(n, 3)} mean {np.mean(n):.3f}; control {np.round(c, 3)} mean {np.mean(c):.3f}; difference {np.mean(c) - np.mean(n):.3f}")
print("exact two-sided permutation p for complete separation at 3 v 3 is 2/20 = 0.10, the smallest this design can give")

price = {"g6e.xlarge": 1.861, "g6e.2xlarge": 2.242, "g6e.4xlarge": 3.004}
end = {"tr_csv_exchangeable_s300": "2026-10-31T20:57:06", "tr_csv_exchangeable_s301": "2026-10-31T21:06:02", "tr_csv_exchangeable_s302": "2026-10-31T21:02:35"}
tot = 0
for k, e in end.items():
    r = json.load(open(ROOT / f"{k}_launch.json"))
    h = (datetime.fromisoformat(e) - datetime.fromisoformat(r["launched"]).replace(tzinfo=None)).total_seconds() / 3600
    tot += h * price[r["instance_type"]]
    print(f"{k} {r['instance_type']} {h:.2f} h ${h * price[r['instance_type']]:.2f}")
print(f"controls ${tot:.2f}; all MARL tests ${21.74 + tot:.2f}")
