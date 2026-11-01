"""X2 (EXPLORATORY, e1_preregistration.md section 10): all 26 E1 detectors under the PUBLISHED events.

Prereg row X2: "All detectors under the published event (persist=1) and smoothed peak_drop | Continuity with
tab:sweep". Written 2026-10-08, AFTER the registered result (experiments/e1_primary_report.txt: CLEAN NULL,
non-superiority shown). It changes no pinned file and no registered rule, and it can never change the verdict.

Definitions, all pre-freeze (d1aa118, 2026-10-07 22:25 -0400), imported and not copied:
  published event (persist=1)  threshold_sweep.ANCHORS "published_baseline_60" = score_prime.collapse_step(
                               how="baseline_drop", frac=0.6, persist=1) = detection_metrics.event_series:
                               strict < 0.6 x mean(first 5 steps). score_prime.PUBLISHED_HOW/FRAC/PERSIST.
  smoothed peak_drop           paper/analysis/onset_profile.py: smooth(h, SMOOTH_W=15), a centred moving average,
                               then score_prime.collapse_step(how="peak_drop", frac=0.8, persist=1) (canonical
                               Definition B = the 20% drop, row["window_defB_smoothed"]). tex:703-707. ACAUSAL.
  the rest of tab:sweep        threshold_sweep.ANCHORS "published_first_move" (Definition A) and the four raw
                               peak_drop thresholds threshold_sweep.DROPS (5/10/20/40%), tex:639-688, 722-737.
Scoring is threshold_sweep.score_at (via e1_primary.score, asserted equal cell by cell) at the registered
operating point (threshold_sweep.fpr10_threshold on the 24 benign runs).

    python experiments/e1_exploratory_parts/x2_published_events.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True       # read-only: leave no new __pycache__ next to the pinned files
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))
import numpy as np  # noqa: E402

import e1_primary as P  # noqa: E402
import e1_event as EV  # noqa: E402
import e1_detectors as D  # noqa: E402
import onset_profile as OP  # noqa: E402  (paper/analysis is on sys.path via e1_primary)

TS, SP, GB = P.TS, P.SP, P.GB
X = "[EXPLORATORY] X2"
C = "CHOICE (made after the data):"
SCORE_KEYS = ("median_lead", "n_lead_pos", "n_lead_runs", "tpr_at_fpr10", "tpr_strict", "realized_benign_fpr",
              "n_alarming", "n_event_defined", "fpr10_threshold")


def cp(x, n):
    return "[%.3f, %.3f]" % P.clopper_pearson(x, n)


def cnt(x, n):
    """A count with its Clopper-Pearson 95% interval."""
    return "%d/%d %s" % (x, n, cp(x, n))


def memo(f):
    """Cache a detector series per run object: score_at and score call each series once per event."""
    cache = {}
    return lambda r: cache[id(r)] if id(r) in cache else cache.setdefault(id(r), f(r))


def provenance():
    pins = json.loads(P.PINS.read_text())
    pins.pop("_env", None)
    bad = [rel for rel, h in pins.items() if P._sha(P.ROOT / rel) != h]
    if bad:
        raise SystemExit("PROVENANCE FAILURE: %s differ from the frozen versions" % bad)
    op = Path(OP.__file__)
    return len(pins), P._sha(op), datetime.fromtimestamp(op.stat().st_mtime).strftime("%Y-%m-%d %H:%M")


def smoothed_peak_drop(run):
    """onset_profile.smooth_profile's drop20_step at w = SMOOTH_W, on the strict series (index == step)."""
    h = OP.smooth(GB.series(run["rows"], "hidden"), OP.SMOOTH_W)
    return SP.collapse_step(None, h, how="peak_drop", frac=TS.peak_frac(0.20), persist=1)


def first_move_up(run, e):
    """Direction of Definition A's event at index e. grid_benchmark.first_move fires on |v - b| > thr with
    b = mean of the first 3 finite strict points (grid_benchmark.first_move_pred), so it is TWO-SIDED: it also
    dates a quality RISE. Upward = h[e] > b. b is restated from first_move_pred (it is not exposed), so the
    event index and the predicate are asserted against the imported rule."""
    h = GB.series(run["rows"], "hidden")
    i = int(e)                         # collapse_step(steps=None) returns the index as a float
    assert i == e and GB.first_move(h) == i and GB.first_move_pred(h)(h[i]), (run.get("_tag"), e)
    return bool(h[i] > h[np.isfinite(h)][:3].mean())


def events_table():
    """(key, description, causal, how/frac for threshold_sweep.score_at or None, event_of)."""
    ev = [("published_baseline_60", "baseline_drop(0.6, persist=1): the published TPR event (TS.ANCHORS)",
           True, ("baseline_drop", 0.60)),
          ("smoothed_peak_drop20_w%d" % OP.SMOOTH_W, "peak_drop 20%% on a centred w=%d moving average "
           "(onset_profile.py; tex:703-707)" % OP.SMOOTH_W, False, None),
          ("published_first_move", "3-sigma first movement of strict quality, Definition A (TS.ANCHORS)",
           True, ("first_move", None))]
    ev += [("peak_drop_%d" % int(d * 100), "raw peak_drop %d%% (tab:sweep column)" % int(d * 100), True,
            ("peak_drop", TS.peak_frac(d))) for d in TS.DROPS]
    out = []
    for key, desc, causal, hf in ev:
        f = smoothed_peak_drop if hf is None else (lambda how, frac: lambda r: TS.event_index(r, how, frac))(*hf)
        out.append((key, desc, causal, hf, f))
    return out


def run():
    lines, data = [], {"events": {}}
    say = lines.append
    npin, op_sha, op_mtime = provenance()
    used = P.load_main()[0]
    hack = [(r["_tag"], r) for r in used["hacking"]]
    ben = used["benign"]
    for _, r in hack:
        r["rows"] = r["log"]           # in memory only: threshold_sweep.event_index reads run["rows"]
    for r in ben:
        r["rows"] = r["log"]
    nh, nb = len(hack), len(ben)
    specs = [{"label": l, "family": fam, "gb": memo(g), "dm": memo(m)} for (l, fam, g, m) in D.detectors()]
    say("%s All %d E1 detectors under the published events (prereg section 10, X2: 'continuity with tab:sweep'). "
        "EXPLORATORY: none of it can change the registered verdict (CLEAN NULL, non-superiority shown)." % (X, len(specs)))
    say("%s provenance: %d pinned files match experiments/e1_pins.json; paper/analysis/onset_profile.py (unpinned, "
        "gitignored; the smoothed event) sha256 %s..., last modified %s, before the freeze d1aa118 (2026-10-07 22:25)" % (
            X, npin, op_sha[:16], op_mtime))
    say("%s runs (e1_primary.load_main): %d hacking vs %d benign. Operating point: the registered "
        "threshold_sweep.fpr10_threshold on the %d benign runs, so each detector's FP is the same under every event." % (
            X, nh, nb, nb))
    say("%s scoring: threshold_sweep.score_at (lead = event - 3-sigma first move of the gb series; alarm = first "
        "dev(dm) > threshold). e1_primary.score is asserted equal to it on every anchor and raw peak_drop cell below." % X)
    say("%s NOTE: the smoothed event is ACAUSAL (a centred w=%d moving average uses up to %d future steps). It may "
        "label the reference event offline but can never be used by a detector; no detector here sees the smoothed "
        "series. (The 26 detectors are the registered ones, unchanged: the 8 level and 2 CUSUM detectors use only "
        "past steps after their first-5/6-step baseline, while the 16 velocity/acceleration forms inherit "
        "curvature_detectors.derived's centred +/-2-step pre-smoothing, i.e. 2 steps of look-ahead, as in tab:sweep.)" % (
            X, OP.SMOOTH_W, OP.SMOOTH_W // 2))
    say("%s The published TPR event (baseline_drop 0.6, persist=1) fired on 8/12 benign grid runs in E0 (prereg :57), "
        "and under it the pilot median windows would have been 5 and 8 (prereg :312); the registered primary event "
        "is a different one (sustained_drop_v2)." % X)
    say("%s Not the source definition: 'smoothed peak_drop' fixes neither the width nor the drop. Used w = "
        "onset_profile.SMOOTH_W = 15 (its headline) and the 20%% drop (Definition B, the tex:703-707 smoothed "
        "window 16 and leads +2.5 / +2.0). tex reported only the six reference leads under it; scoring TP/FP for "
        "all 26 detectors against it is new." % C)
    say("%s 'Every anchor tab:sweep reports' read as all six of its definitions (tex:677-688): both TS.ANCHORS "
        "plus the four raw peak_drop thresholds, so 7 events in all with the smoothed one." % C)
    say("%s Events are dated on the STRICT per-step oracle (log 'hidden', option (a), as registered); tab:sweep's "
        "grid had one held-out series. The lenient oracle is not used here." % C)
    say("%s TP uses the registered strict rule (alarm < event); tab:sweep's published <= rule is printed beside it. "
        "p is E1's one-sided Fisher on strict TP vs FP (tab:sweep's two-tailed threshold_sweep.fisher_exact is "
        "printed as a per-event minimum, for continuity)." % C)
    say("%s Benign false events = the same event function on each benign run's strict series (the E0 convention, "
        "prereg :55-59). Median window is given two ways: over all 24 with no event = -1 (registered "
        "window_precondition) and over the runs with an event (onset_profile / tex:690-697: e - first step)." % C)
    say("%s Holm over the 26 detectors within each event is printed as a descriptive count only." % C)

    headline = []
    for key, desc, causal, hf, ev_of in events_table() + [("registered_sustained_drop_v2", "registered primary event, "
                                                           "restated for reference (not re-decided)", True, "reg",
                                                           lambda r: EV.event(list(range(len(r["log"]))),
                                                                              [x.get("hidden") for x in r["log"]]))]:
        ev_h = {tag: ev_of(r) for tag, r in hack}
        ev_b = {r["_tag"]: ev_of(r) for r in ben}
        win = P.window_precondition(ev_h)
        got = [e for e in ev_h.values() if e is not None]
        n_be = sum(e is not None for e in ev_b.values())
        med_def = float(np.median(got)) if got else None
        say("%s == event %s: %s%s" % (X, key, desc, "" if causal else " [ACAUSAL: reference label only]"))
        say("%s   hacking events %s; benign false events %s; median window %s over %d (no event = -1; >= 12: %s); "
            "median over the %d with an event %s; at step 0: %d" % (
                X, cnt(win["n_event"], nh), cnt(n_be, nb), win["median_window"], nh, "yes" if win["holds"] else "no",
                len(got), med_def, sum(1 for e in got if e <= 0)))
        fm_dir = None
        if key == "published_first_move":     # two-sided rule: how many of its "collapses" are quality RISES
            hu = sum(first_move_up(r, ev_h[t]) for t, r in hack if ev_h[t] is not None)
            bu = sum(first_move_up(r, ev_b[r["_tag"]]) for r in ben if ev_b[r["_tag"]] is not None)
            fm_dir = {"hack_up": hu, "hack_n": len(got), "benign_up": bu, "benign_n": n_be}
            say("%s   direction: Definition A is two-sided (|v - b| > thr, b = mean of the first 3 strict points), so "
                "on E1 it also fires on UPWARD moves of the strict oracle: upward in %s of its hacking events and %s "
                "of its benign false events (an upward move is quality rising, not a collapse)" % (
                    X, cnt(hu, len(got)), cnt(bu, n_be)))
        say("%s   hacking event steps: %s" % (X, sorted(-1 if e is None else int(e) for e in ev_h.values())))
        reg = hf == "reg"
        if not reg:
            say("%s   %-34s %-29s %-20s %-20s %-20s %s" % (X, "detector", "median lead (lead>0/n [CP 95%])",
                                                           "TP<= x/24 [CP 95%]", "TP< x/24 [CP 95%]",
                                                           "FP y/24 [CP 95%]", "p one-sided"))
        rows, ps = {}, []
        for s in specs:
            sc = P.score(s, hack, ben, ev_of)
            if isinstance(hf, tuple):          # the tab:sweep code path itself, cell by cell
                ref = TS.score_at(s, hack, ben, *hf)
                for k in SCORE_KEYS:
                    assert sc[k] == ref[k], (key, s["label"], k, sc[k], ref[k])
            x, xl, y = sc["n_tp_strict"], sc["n_tp_le"], sc["n_fp"]
            p = P.fisher_one_sided(x, nh, y, nb)
            ps.append(p)
            rows[s["label"]] = {"tp_strict": x, "tp_le": xl, "fp": y, "p_one_sided": p,
                                "p_two_tailed_strict": TS.fisher_exact(x, nh - x, y, nb - y),
                                "p_two_tailed_le": TS.fisher_exact(xl, nh - xl, y, nb - y),
                                "median_lead": sc["median_lead"], "n_lead_pos": sc["n_lead_pos"],
                                "n_lead_runs": sc["n_lead_runs"], "threshold": sc["fpr10_threshold"],
                                "benign_activity": sc["benign_activity"], "hit_runs_strict": sc["hit_runs_strict"]}
            lead = "--" if sc["median_lead"] is None else "%+.1f (%s)" % (
                sc["median_lead"], cnt(sc["n_lead_pos"], sc["n_lead_runs"]))
            if not reg:
                say("%s   %-34s %-29s %-20s %-20s %-20s %.4f" % (X, s["label"], lead, cnt(xl, nh), cnt(x, nh),
                                                               cnt(y, nb), p))
        adj = P.holm(ps)
        for (l, r), a in zip(rows.items(), adj):
            r["holm26"] = a
        pos = [l for l, r in rows.items() if (r["median_lead"] or 0) > 0]
        best = min(rows, key=lambda l: rows[l]["p_one_sided"])
        n_tie = sum(1 for r in rows.values() if abs(r["p_one_sided"] - rows[best]["p_one_sided"]) < 1e-12)
        nz = [r for r in rows.values() if r["tp_le"] > 0]
        p2 = min([min(r["p_two_tailed_le"], r["p_two_tailed_strict"]) for r in nz], default=None)
        # the cell(s) giving that minimum; "wrong direction" = its TP rate below the benign FP rate
        p2_cells = [{"detector": l, "cell": c, "tp": r[tk], "fp": r["fp"], "wrong_direction": r[tk] / nh < r["fp"] / nb}
                    for l, r in rows.items() if r["tp_le"] > 0
                    for c, pk, tk in (("TP<=", "p_two_tailed_le", "tp_le"), ("TP<", "p_two_tailed_strict", "tp_strict"))
                    if p2 is not None and abs(r[pk] - p2) < 1e-12]
        p2_wrong = bool(p2_cells) and all(c["wrong_direction"] for c in p2_cells)
        say("%s   summary: positive median lead %d/26 %s; min one-sided p %.4f (%s, TP< %d vs FP %d%s); nominal p < 0.05: "
            "%d/26; Holm over 26 < 0.05: %d/26; min two-tailed p over the %d non-zero TPR cells (tab:sweep's "
            "statistic): %s%s" % (X, len(pos), pos, rows[best]["p_one_sided"], best, rows[best]["tp_strict"],
                                  rows[best]["fp"], "" if n_tie == 1 else "; %d detectors tie at this p, first listed" % n_tie,
                                  sum(p < 0.05 for p in ps), sum(a < 0.05 for a in adj), len(nz), p2,
                                  " (wrong-direction cell: hacking alarms less than benign; the one-sided p is the "
                                  "meaningful one)" if p2_wrong else ""))
        if reg:                                # the scoring path reproduces the saved registered result
            R = json.loads((P.ROOT / "experiments" / "e1_primary_results.json").read_text())
            want = {P.PRIMARY: (R["primary"]["fisher_p_one_sided"], R["primary"]["median_lead"])}
            want.update({l: (v["p"], v["median_lead"]) for l, v in R["secondary"].items()})
            same = all(abs(rows[l]["p_one_sided"] - w[0]) < 1e-12 and rows[l]["median_lead"] == w[1]
                       for l, w in want.items())
            data["reproduces_registered_results"] = same
            say("%s   per-detector rows not repeated (see experiments/e1_primary_report.txt); this path reproduces "
                "experiments/e1_primary_results.json (p and median lead, all %d detectors): %s" % (
                    X, len(want), "yes" if same else "NO"))
        dg = rows[P.PRIMARY]
        data["events"][key] = {"description": desc, "causal": causal, "n_event": win["n_event"], "n_hack": nh,
                               "n_benign_event": n_be, "n_benign": nb, "median_window_registered": win["median_window"],
                               "median_window_defined": med_def,
                               "event_steps": {t: (None if e is None else int(e)) for t, e in ev_h.items()},
                               "benign_event_steps": {t: (None if e is None else int(e)) for t, e in ev_b.items()},
                               "detectors": rows, "n_positive_lead": len(pos), "positive_lead": pos,
                               "min_p_one_sided": rows[best]["p_one_sided"], "min_p_detector": best,
                               "n_tied_at_min_p": n_tie,
                               "n_nominal_p05": sum(p < 0.05 for p in ps), "n_holm26_p05": sum(a < 0.05 for a in adj),
                               "min_p_two_tailed_nonzero": p2, "min_p_two_tailed_cells": p2_cells,
                               "min_p_two_tailed_wrong_direction": p2_wrong}
        if fm_dir is not None:
            data["events"][key]["first_move_direction"] = fm_dir
        headline.append("%s   %-30s events %-20s benign %-20s window %5s | disagreement TP< %d/24 FP %d/24 p %.4f "
                        "lead %s | min p %.4f; lead > 0 in %d/26" % (
                            X, key, cnt(win["n_event"], nh), cnt(n_be, nb), win["median_window"], dg["tp_strict"],
                            dg["fp"], dg["p_one_sided"], "--" if dg["median_lead"] is None else "%+.1f" % dg["median_lead"],
                            rows[best]["p_one_sided"], len(pos)))
    say("%s HEADLINE (window = registered convention; the last row is the registered event, restated):" % X)
    lines.extend(headline)
    say("%s No event changes the registered verdict; X2 is reporting for continuity with tab:sweep only." % X)
    return {"id": "X2", "title": "All 26 detectors under the published events (baseline_drop 0.6 persist=1, smoothed "
                                 "peak_drop, and the other tab:sweep definitions)",
            "lines": lines, "data": data, "figures": []}


if __name__ == "__main__":
    for line in run()["lines"]:
        print(line)
