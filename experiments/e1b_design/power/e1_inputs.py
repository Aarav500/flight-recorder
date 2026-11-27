"""E1b power, step 1: pull the per-run E1 inputs the power analysis needs. Read-only on the repo.

Writes e1_inputs.json next to this file:
  * hacking events (registered sustained_drop_v2 on the strict series), per task
  * the primary detector's leads (event - 3-sigma first move of |disagreement|), as e1_primary.score computes them
  * the proposed E1b primary on E1 data (upward-only alarm, time-matched operating point), with the two ways
    of scoring the twin of a no-event hacking run: (A) drop it (e1_supplementary D3) and (B) score it over
    all 400 steps
  * per-run signed changes (mean of last 25 steps minus first 25) for the audit-test effect sizes:
    disagreement, 0.5B judge, 1.5B judge, completion length, truncation rate, and the gate-pass rate
    (closed fence, parses, not clipped) computed from the stored completions
E1's benign arm optimizes the 1.5B judge, so it is NOT the E1b twin. It is used only where stated.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

REPO = Path(r"<home>/OneDrive/Desktop/Fligh recorder")
sys.path.insert(0, str(REPO / "experiments"))
import e1_primary as P  # noqa: E402
import e1_event as EV  # noqa: E402
import e1_detectors as D  # noqa: E402

OUT = Path(__file__).resolve().parent / "e1_inputs.json"
_FENCE = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)   # reward_testhack._FENCE


def chg(x):
    x = np.asarray(x, float)
    return float(np.nanmean(x[-25:]) - np.nanmean(x[:25]))


def gate_ok(text):
    m = _FENCE.search(text)
    if not m:
        return 0.0
    try:
        ast.parse(m.group(1))
    except SyntaxError:
        return 0.0
    return 1.0


def up_dev(x):
    x = np.asarray(x, float)
    return np.maximum(x - np.median(x[np.isfinite(x)][:5]), 0.0)


def first_alarm(dev, thr, upto=None):
    d = dev if upto is None else dev[:upto]
    idx = np.nonzero(d > thr)[0]
    return int(idx[0]) if len(idx) else None


def main():
    P.verify_pins()
    used, excluded, unreg, _, surplus = P.load_main()
    hack = [(r["_tag"], r) for r in used["hacking"]]
    ben = used["benign"]
    event_of = lambda run: EV.event(list(range(len(run["log"]))), [r.get("hidden") for r in run["log"]])
    specs = {l: {"label": l, "family": f, "gb": g, "dm": m} for (l, f, g, m) in D.detectors()}
    prim = specs[P.PRIMARY]
    sc = P.score(prim, hack, ben, event_of)
    events = sc["events"]

    # proposed E1b primary on E1 data: upward-only, time-matched. bm = seed- and task-matched benign run
    bm = {(r["task"], int(r["seed"])): r for r in ben}
    tm = {}
    for mode in ("D3_check_twosided_drop", "A_drop", "B_full"):
        devf = P.DM.dev if mode.startswith("D3") else up_dev
        hdev = {t: devf(prim["dm"](r)) for t, r in hack}
        maxima = []
        for t, r in hack:
            e = events[t]
            bd = devf(prim["dm"](bm[(r["task"], int(r["seed"]))]))
            if e is None:
                if mode.endswith("drop"):
                    continue
                maxima.append(float(np.max(bd)))
            else:
                maxima.append(float(np.max(bd[:e])))
        thr = float(np.percentile(maxima, 90))
        tp = sum(1 for t, _ in hack if events[t] is not None and first_alarm(hdev[t], thr) is not None
                 and first_alarm(hdev[t], thr) < events[t])
        fp = sum(v > thr for v in maxima)
        lo, hi = P.newcombe(tp, len(hack), fp, len(maxima), 0.90)
        hits = sorted((events[t], first_alarm(hdev[t], thr)) for t, _ in hack if events[t] is not None
                      and first_alarm(hdev[t], thr) is not None and first_alarm(hdev[t], thr) < events[t])
        tm[mode] = {"hits_event_alarm": hits, "thr": thr, "tp": tp, "nh": len(hack), "fp": fp, "nb": len(maxima),
                    "p": P.fisher_one_sided(tp, len(hack), fp, len(maxima)), "newcombe90": [lo, hi],
                    "twin_max_sorted": sorted(maxima)}

    runs = []
    for arm, rs in (("hacking", [r for _, r in hack]), ("benign", ben)):
        for r in rs:
            log = r["log"]
            rows = D.rows(r)
            gate = []
            for c, row in zip(r["completions"], rows):
                g = [gate_ok(tx) for tx in c["texts"]]
                gate.append(float(np.mean(g)) if g else np.nan)
            runs.append({
                "tag": r["_tag"], "arm": arm, "task": r["task"], "seed": int(r["seed"]),
                "event": events.get(r["_tag"]) if arm == "hacking" else None,
                "chg": {"disagree": chg([x["disagree"] for x in log]),
                        "judge_small": chg([x["judge_small"] for x in log]),
                        "judge_big": chg([x["judge_big"] for x in log]),
                        "mean_len": chg([x.get("mean_len", np.nan) for x in rows]),
                        "clipped": chg([x.get("clipped", np.nan) for x in rows]),
                        "fence_parse": chg(gate),
                        "hidden": chg([x["hidden"] for x in log]),
                        "hidden_lenient": chg([x["hidden_lenient"] for x in log])},
                "fence_parse_first25": float(np.nanmean(gate[:25])),
                "fence_parse_last25": float(np.nanmean(gate[-25:])),
            })
    lead_by_tag = {}
    for t, r in hack:
        fm = P.GB.first_move(prim["gb"](r))
        lead_by_tag[t] = (float(events[t]) - float(fm)) if (events[t] is not None and fm is not None) else None
    assert sorted(v for v in lead_by_tag.values() if v is not None) == sorted(sc["leads"])
    out = {"events": {t: events[t] for t, _ in hack}, "lead_by_tag": lead_by_tag,
           "leads": sc["leads"], "median_lead": sc["median_lead"],
           "registered": {"tp": sc["n_tp_strict"], "fp": sc["n_fp"]},
           "time_matched_upward": tm, "runs": runs}
    OUT.write_text(json.dumps(out, indent=1))
    ev = [e for e in events.values()]
    print("events (None = no event):", sorted(-1 if e is None else e for e in ev))
    print("leads:", sorted(sc["leads"]), "median", sc["median_lead"])
    for k, v in tm.items():
        print("time-matched upward-only, twin of no-event run %s: thr %.4f TP %d/%d FP %d/%d p %.4f Newcombe90 %s" % (
            k, v["thr"], v["tp"], v["nh"], v["fp"], v["nb"], v["p"], tuple(round(x, 3) for x in v["newcombe90"])))
        print("   hits (event, alarm):", v["hits_event_alarm"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
