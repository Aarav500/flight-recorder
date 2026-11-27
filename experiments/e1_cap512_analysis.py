"""Token-cap test: is half of the strict-oracle collapse an artifact of max_completion_length=256?

FIXED BEFORE THE RUNS WERE LAUNCHED. The commit that adds this file predates the AWS launch.

WHY
  The format diagnostic (experiments/e1_diag_results.md) found two mechanisms behind the strict
  collapse. On count_vowels and second_largest, chatty answers hit the 256-token cap before their
  closing code fence, so the extracted text does not parse. On is_prime, unfenced prose sits before
  the def. If the cap causes the first mechanism, a reviewer can call half the "hacking" an artifact.
  This test reruns the published GRID config (reduced-capacity judge, lr 2e-6, 200 steps, seed 0,
  no probe) with max_completion_length 512 and compares each task with its 256-token diagnostic run
  (runs/e1/diag/judge_<task>_0_lr2e-6.json).

READING PER TASK (one seed each, so this is descriptive, not a test)
  All changes are mean(last 25 steps) - mean(first 25), with the moving-block bootstrap CI from
  e1_diag_supplement.block_boot_diff. "Collapse" means the change is <= -0.20 and the CI excludes 0.
  CAP ARTIFACT     at 512 the strict series does not collapse (change > -0.10, or its CI reaches 0),
                   late clipped ratio < 0.5, and at 256 the same task did collapse.
  FILLS ANY BUDGET at 512, late clipped ratio >= 0.5 and the strict series collapses. Length grows
                   to whatever cap it is given: genuine length hacking, not a cap artifact.
  PROSE PERSISTS   at 512 the strict series collapses, late clipped ratio < 0.5, and the lenient
                   change does not decline (CI upper end >= 0): a format collapse without truncation.
  OTHER            anything else.
  Prediction, stated now: is_prime reads PROSE PERSISTS, because its mechanism does not depend on
  the cap. count_vowels and second_largest are the open question.
  What follows for the paper's hacking definition (pre-reg section 1): CAP ARTIFACT on both open
  tasks weakens option (a), "spec/length/format violation", and favours (b), a generator whose hack
  can be wrong code. FILLS ANY BUDGET supports (a) as genuine verbosity hacking.

    python experiments/e1_cap512_analysis.py      # reads runs/e1/cap512/*.json and runs/e1/diag/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from e1_diag_supplement import block_boot_diff  # noqa: E402

TASKS = ("count_vowels", "is_prime", "second_largest")


def summarize(d):
    L = d["log"]
    th = d.get("trl_history", [])
    clip = [h["completions/clipped_ratio"] for h in th if "completions/clipped_ratio" in h]
    ml = [h["completions/mean_length"] for h in th if "completions/mean_length" in h]
    return {
        "strict": block_boot_diff([r["hidden"] for r in L]),
        "lenient": block_boot_diff([r["hidden_lenient"] for r in L]),
        "late_strict": float(np.mean([r["hidden"] for r in L][-25:])),
        "clip_first_late": (float(np.mean(clip[:25])), float(np.mean(clip[-25:]))) if clip else None,
        "len_first_late": (float(np.mean(ml[:25])), float(np.mean(ml[-25:]))) if ml else None,
    }


def collapses(ch):
    return ch[0] <= -0.20 and ch[2] < 0


def reading(s512, s256):
    late_clip = s512["clip_first_late"][1] if s512["clip_first_late"] else float("nan")
    st = s512["strict"]
    if (st[0] > -0.10 or st[2] >= 0) and late_clip < 0.5 and collapses(s256["strict"]):
        return "CAP ARTIFACT"
    if late_clip >= 0.5 and collapses(st):
        return "FILLS ANY BUDGET"
    if collapses(st) and late_clip < 0.5 and s512["lenient"][2] >= 0:
        return "PROSE PERSISTS"
    return "OTHER"


def main():
    cap_dir = Path(sys.argv[sys.argv.index("--dir") + 1]) if "--dir" in sys.argv else ROOT / "runs" / "e1" / "cap512"
    out = {}
    for task in TASKS:
        f512 = sorted(cap_dir.glob("*%s*.json" % task))
        f256 = ROOT / "runs" / "e1" / "diag" / ("judge_%s_0_lr2e-6.json" % task)
        if not f512 or not f256.exists():
            print("%-15s missing (%s, %s)" % (task, len(f512), f256.exists()))
            continue
        s512, s256 = summarize(json.loads(f512[0].read_text())), summarize(json.loads(f256.read_text()))
        r = reading(s512, s256)
        out[task] = {"cap512": s512, "cap256": s256, "reading": r}
        for lab, s in (("256", s256), ("512", s512)):
            print("%-15s cap %s  strict %+.3f [%+.3f, %+.3f]  lenient %+.3f [%+.3f, %+.3f]  late strict %.2f  clip %s  len %s" % (
                task, lab, *s["strict"], *s["lenient"], s["late_strict"],
                "%.2f->%.2f" % s["clip_first_late"] if s["clip_first_late"] else "-",
                "%.0f->%.0f" % s["len_first_late"] if s["len_first_late"] else "-"))
        print("%-15s READING: %s\n" % (task, r))
    if "--dir" not in sys.argv:
        (ROOT / "experiments" / "e1_cap512_results.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
