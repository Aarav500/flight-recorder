"""THROWAWAY SPIKE. The one exploratory read of the sealed phi.jsonl files, written before any of them was opened.
The verdict was GAP, so phi.jsonl stays sealed unless Aarav chooses to open it. Run this only on his say-so.

Question: does a gradient statistic separate csv_narrow from csv_exchangeable before their true quality does?

  Oracle by window, already seen: overlapping through steps 25-49, fully separated from steps 50-99 on. So the
  windows that could show a lead are fixed as steps 5-24 and 25-49.
  Statistics: gap = within - cross (primary, it cancels half-batch noise), phi (the proposal's statistic), and
  gnorm1, kl1 as baselines that need no second agent.
  Window 0-4 is the identity check. The two arms' visible tests differ from step 0, so a statistic already
  separated there is reading the reward's identity, and a later separation doesn't count as a lead for it.

  EARLY SEPARATION  a gradient statistic has all 3 narrow values on one side of all 3 control values in window
                    5-24 or 25-49, isn't separated in 0-4, and neither baseline separates in that same window.
  NO LEAD           anything else.

Complete separation at 3 v 3 has chance 0.10 per look, and this makes 4 looks, so about a 1 in 3 chance of a false
EARLY SEPARATION. It could only ever be a hypothesis for fresh seeds, and NO LEAD closes the question on this setup.

    python analyze_train_phi.py ../runs
"""
import json
import sys
from pathlib import Path

import numpy as np

TASKS = ("csv_narrow", "csv_exchangeable")
SEEDS = (300, 301, 302)
WINDOWS = ((0, 5), (5, 25), (25, 50), (50, 100), (100, 150), (150, 400))
LEAD_WINDOWS = ((5, 25), (25, 50))
GRAD = ("gap", "phi")
BASE = ("gnorm1", "kl1")


def run_stats(root, task, seed):
    name = f"{task}_s{seed}"
    d = sorted(root.glob(f"**/{name}/phi.jsonl"))
    if len(d) != 1:
        raise SystemExit(f"{name}: expected one phi.jsonl, found {len(d)}")
    phi = [json.loads(x) for x in d[0].read_text().splitlines() if x.strip()]
    st = [json.loads(x) for x in (d[0].parent / "steps.jsonl").read_text().splitlines() if x.strip()]
    return dict(gap=np.array([r["within"] - r["cross"] for r in phi]), phi=np.array([r["phi"] for r in phi]),
                gnorm1=np.array([r["gnorm1"] for r in phi]), kl1=np.array([r["kl1"] for r in st]))


def separated(a, b):
    return max(a) < min(b) or max(b) < min(a)


def decide(table):
    """table[stat][window] = (narrow values, control values). -> (verdict, the reasons)."""
    hits = []
    for s in GRAD:
        if separated(*table[s][(0, 5)]):
            continue
        for w in LEAD_WINDOWS:
            if separated(*table[s][w]) and not any(separated(*table[b][w]) for b in BASE):
                hits.append((s, w))
    return ("EARLY SEPARATION (hypothesis only)", hits) if hits else ("NO LEAD", hits)


def main():
    root = Path(sys.argv[1])
    data = {t: [run_stats(root, t, s) for s in SEEDS] for t in TASKS}
    table = {}
    for stat in GRAD + BASE:
        table[stat] = {}
        for lo, hi in WINDOWS:
            table[stat][(lo, hi)] = tuple([float(r[stat][lo:hi].mean()) for r in data[t]] for t in TASKS)
    print(f"{'stat':7s} {'steps':>8s}  {'narrow':>24s}  {'control':>24s}  separated")
    for stat in GRAD + BASE:
        for w in table[stat]:
            n, c = table[stat][w]
            print(f"{stat:7s} {w[0]:3d}-{w[1] - 1:<4d}  {' '.join(f'{x:+.4f}' for x in n)}  {' '.join(f'{x:+.4f}' for x in c)}  "
                  f"{'yes' if separated(n, c) else ''}")
    verdict, hits = decide(table)
    print(f"\nPHI READ: {verdict} {hits if hits else ''}")


if __name__ == "__main__":
    main()
