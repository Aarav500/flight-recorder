"""E1 pilot gate: which learning rate, if any, gives a pre-event window wide enough to test early warning.

FIXED BEFORE ANY PILOT DATA EXISTED. The commit that adds this file predates every pilot result;
`git log --format=%cI -- experiments/e1_pilot_gate.py` against the S3 timestamps proves it. Do not
edit the constants below after a pilot result has been seen -- record a deviation instead.

WHAT THE PILOT IS FOR
  The published grid (reduced-capacity judge, lr 2e-6, 200 steps) has a median pre-event window of
  5 steps under the published event (tex:358-364), too short for any detector that needs a baseline.
  The plan (AaravOS/papers/audit_or_early_warning.md, E1) slows the clock: lr 1e-6 and 5e-7, 3 runs
  each. LR is a clock lever, not a shape lever (scripts/launch_slow_onset.py, docstring item 2), so
  each rate gets a step budget that holds lr x steps at the grid's 2e-6 x 200 = 4e-4:
  1e-6 -> 400 steps, 5e-7 -> 800 steps. At 200 steps a lr 1e-6 run decayed 0.368 -> 0.266 and never
  crossed 0.6 x baseline, so a 200-step pilot would trip the kill rule by construction.

THE GATE (one definition, fixed here)
  Event:  score_prime.collapse_step(how="baseline_drop", frac=0.6, persist=2) on the per-step
          held-out pass rate. This is the only definition that passed E0: 0/12 grid benign firings
          at persist=2, against 8/12 at persist=1 (the published column) and 6/12 for raw peak_drop
          (paper/stats/e0_slope_validation.json).
  Window: the event step, the same convention as onset_profile.json's window_defC_baseline60.
  A rate PASSES if the median window over its runs is >= 12 steps, with a run whose event never
          fires counted as failing (it has no window).
  Choice: both pass -> 1e-6 (half the steps of 5e-7, nearer the published grid). One passes -> it.
          Neither -> the plan's kill rule: stop, and report the windows as the result.
  Other definitions are printed for the record only; they do not move the gate.

    python experiments/e1_pilot_gate.py               # score runs/e1/pilot/*.json
    python experiments/e1_pilot_gate.py --selftest    # reproduce E0's grid number with this code
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))
import score_prime as SP  # noqa: E402

EVENT = ("baseline_drop", 0.6, 2)          # (how, frac, persist) -- the gate definition
WINDOW_MIN = 12                            # plan section 2: "a median pre-event window of 12 steps or more"
RATES = {"1e-6": 400, "5e-7": 800}         # lr -> step budget (lr x steps = 4e-4)
PREFER = "1e-6"
PILOT_DIR = ROOT / "runs" / "e1" / "pilot"

RECORD_ONLY = [("baseline_drop", 0.6, 1, "published event (persist=1)"),
               ("peak_drop", 0.8, 2, "Def. B 20% (raw)")]


def event_step(steps, hidden, how, frac, persist):
    c = SP.collapse_step(list(steps), list(hidden), how, frac, persist)
    return None if c is None else int(c)


def gate(windows):
    """Median window with non-firing runs counted as failing. Returns (median or None, passes)."""
    if not windows:
        return None, False
    vals = sorted(-1 if w is None else w for w in windows)
    med = float(np.median(vals))
    return (None if med < 0 else med), med >= WINDOW_MIN


def load(path):
    d = json.loads(Path(path).read_text())
    log = d["log"]
    return d, [r["step"] for r in log], [r["hidden"] for r in log]


def main():
    files = sorted(PILOT_DIR.glob("*.json"))
    if not files:
        print("no pilot results in %s" % PILOT_DIR)
        return 1
    by_lr, rows = {lr: [] for lr in RATES}, []
    for f in files:
        d, s, h = load(f)
        lr = str(d["lr"]) if str(d["lr"]) in RATES else "%g" % float(d["lr"])
        complete = len(s) >= RATES.get(lr, 0)
        w = event_step(s, h, *EVENT)
        extra = {lab: event_step(s, h, how, fr, pe) for how, fr, pe, lab in RECORD_ONLY}
        rows.append((f.stem, lr, d["task"], len(s), complete, w, extra, d.get("go_no_go", {})))
        if complete and lr in by_lr:
            by_lr[lr].append(w)

    print("%-44s %5s %15s %5s %4s %7s  record-only" % ("run", "lr", "task", "rows", "done", "window"))
    for name, lr, task, n, done, w, extra, g in rows:
        print("%-44s %5s %15s %5d %4s %7s  %s  probe_window=%s diverged=%s" % (
            name, lr, task, n, "yes" if done else "NO", w, extra, g.get("causal_window"), g.get("diverged")))
    print("\nGATE: %s(frac=%.1f, persist=%d), median window >= %d, non-firing = fail; incomplete runs excluded"
          % (EVENT[0], EVENT[1], EVENT[2], WINDOW_MIN))
    passing = []
    for lr, ws in by_lr.items():
        med, ok = gate(ws)
        print("  lr %-5s  n=%d complete  windows %s  median %s  -> %s" % (lr, len(ws), ws, med, "PASS" if ok else "fail"))
        if ok:
            passing.append(lr)
    if len(passing) == 2:
        print("DECISION: both pass -> keep %s" % PREFER)
    elif passing:
        print("DECISION: keep %s" % passing[0])
    elif all(len(ws) >= 3 for ws in by_lr.values()):
        print("DECISION: neither passes -> KILL (plan, E1): stop, report the windows as the result")
    else:
        print("DECISION: not yet -- fewer than 3 complete runs at some rate")
    return 0


def selftest():
    """The gate code must reproduce E0's persist=2 number on the frozen grid: median 11 -> fail by 1."""
    sys.path.insert(0, str(ROOT / "paper" / "analysis"))
    import onset_profile as OP           # noqa: E402
    import e0_slope_validation as E0     # noqa: E402
    ws = []
    for r in OP.inventory():
        if r["tag"].startswith("grid:hacking/"):
            s, h = E0.series(r)
            ws.append(event_step(s, h, *EVENT))
    med, ok = gate(ws)
    print("grid hacking (lr 2e-6, 200 steps): windows %s  median %s  pass=%s" % (sorted(ws), med, ok))
    assert len(ws) == 9 and med == 11.0 and not ok, "gate code disagrees with E0"
    assert gate([None, None, 30])[1] is False and gate([12, 13, None])[1] is True
    print("SELFTEST PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(selftest() if "--selftest" in sys.argv else main())
