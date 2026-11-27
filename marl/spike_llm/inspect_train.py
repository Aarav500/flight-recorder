"""THROWAWAY SPIKE. What did the csv_narrow runs actually do? Reads steps and completions, never phi.jsonl."""
import ast, json, re, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step0_llm import extract_code

ROOT = Path(__file__).resolve().parents[1] / "runs"


def load(seed):
    d = ROOT / f"tr_csv_narrow_s{seed}" / "out" / f"csv_narrow_s{seed}"
    rd = lambda n: [json.loads(x) for x in (d / n).read_text().splitlines() if x.strip()]
    return rd("steps.jsonl"), rd("completions.jsonl")


def features(a1, a2):
    h, m = extract_code(a1), extract_code(a2)
    def defs(s):
        try:
            return {n.name for n in ast.walk(ast.parse(s)) if isinstance(n, ast.FunctionDef)}
        except SyntaxError:
            return None
    dh, dm = defs(h), defs(m)
    return dict(
        h_quotes='"' in h.replace('"""', "") or "'\"'" in h,
        h_csvmod="csv" in h and "import" in h,
        h_defined=bool(dh) and "next_field" in dh,
        m_splits=".split(" in m,
        m_csvmod="csv" in m and "import" in m,
        m_calls="next_field(" in m and "def next_field" not in m,
        m_overwrites=bool(dm) and "next_field" in dm,
        m_parses=dm is not None, h_parses=dh is not None,
        fenced1="```" in a1, fenced2="```" in a2)


WINDOWS = [(0, 1), (1, 5), (5, 10), (10, 25), (25, 50), (50, 100), (100, 150), (150, 200), (200, 300), (300, 400)]
for seed in (300, 301, 302):
    st, co = load(seed)
    print(f"\n=== seed {seed}")
    print(f"{'steps':>9s} {'proxy':>6s} {'oracle':>6s} {'expl':>5s} {'hon':>5s} {'const':>5s} {'kl1':>6s} {'kl2':>6s} {'tok1':>5s} {'tok2':>5s} | h_quote h_csv m_calls m_over m_split m_csv")
    for lo, hi in WINDOWS:
        S, C = st[lo:hi], co[lo:hi]
        v = np.array([r["proxy"] for r in S]); h = np.array([r["oracle"] for r in S])
        f = [features(a, b) for c in C for a, b in zip(c["a1"], c["a2"])]
        g = lambda k: np.mean([x[k] for x in f])
        print(f"{lo:4d}-{hi - 1:<4d} {v.mean():6.3f} {h.mean():6.3f} {((v >= .8) & (h < .5)).mean():5.2f} {(h >= .8).mean():5.2f} "
              f"{np.mean([len(set(r['proxy'])) == 1 for r in S]):5.2f} {np.mean([r['kl1'] for r in S]):6.3f} {np.mean([r['kl2'] for r in S]):6.3f} "
              f"{np.mean([r['len1'] for r in S]):5.0f} {np.mean([r['len2'] for r in S]):5.0f} | {g('h_quotes'):7.2f} {g('h_csvmod'):5.2f} {g('m_calls'):7.2f} {g('m_overwrites'):6.2f} {g('m_splits'):7.2f} {g('m_csvmod'):5.2f}")

for seed, step in ((300, 0), (300, 395), (302, 395)):
    st, co = load(seed)
    i = int(np.argmax(st[step]["proxy"]))
    print(f"\n----- seed {seed} step {step}, sample {i}: proxy {st[step]['proxy'][i]:.2f} oracle {st[step]['oracle'][i]:.2f}  (group oracle {[round(x, 2) for x in st[step]['oracle']]})")
    print("HELPER:\n" + extract_code(co[step]["a1"][i]).strip()[:620])
    print("MAIN:\n" + extract_code(co[step]["a2"][i]).strip()[:320])
