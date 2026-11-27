"""E1 supplementary report: what the frozen experiments/e1_primary.py does not print.

NOT the registered analysis and NOT pinned. Written 2026-10-08, AFTER the registered result
(experiments/e1_primary_report.txt: CLEAN NULL, non-superiority shown). It changes no pinned file and no
registered rule. It imports e1_primary, e1_event, e1_detectors and two event_candidates modules and calls
their functions, so every number comes from the registered code paths. Sections, in this order:

  A. REGISTERED-REPORTING (prereg section 6, :187-189 and :194; :155). The H2 table with Clopper-Pearson 95%
     intervals on each secondary detector's strict TPR and realized FPR, the Newcombe 95% interval on
     TPR - FPR ("As row 1"), and the MDE at the first Holm step (alpha/25), from e1_primary.power_fixed_fp.
     The frozen mde_fixed_fp has alpha fixed at 0.05, so the search is repeated here at alpha/25 with the
     same helper.
  B. REGISTERED-REPORTING (prereg :65, :68, :318; e1_event.py docstring). The sensitivity events
     cusum_page_down_v2 and baseline_drop(0.6, persist=2): "reported alongside, never primary". No decision
     rule is registered for them, so the decide() verdicts printed there are illustrative. They cannot
     change the registered verdict.
  C. REGISTERED-REPORTING (prereg :91, :103, option (a)). The format-tolerant oracle (hidden_lenient),
     "reported alongside" the strict one, including the registered event and natural-hack-rate measure
     applied to it.
  D. EXPLORATORY, not registered. These checks come from the independent verification of 2026-10-08.
     Every line of D carries the [EXPLORATORY] label. The closing line restates a registered number.
  Every count in A-C carries a Clopper-Pearson 95% interval (prereg :194).

    python experiments/e1_supplementary.py > experiments/e1_supplementary_report.txt
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True       # read-only: leave no new __pycache__ next to the pinned files
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e1_primary as P  # noqa: E402
import e1_event as EV  # noqa: E402
import e1_detectors as D  # noqa: E402

SP, TS, DM = P.SP, P.TS, P.DM
CAND = P.ROOT / "experiments" / "event_candidates"
# The sensitivity events are not in e1_pins.json. These are their SHA-256 in freeze commit d1aa118 (LF).
SENSITIVITY = {"cusum_page_down_v2.py": "ca357eaabf32c2e189587aba0f3aa9d29c67c5314678f23d4e73b6f0aba17da7",
               "ref_baseline_drop_p2.py": "08debb05e9a2ae145a900560430a45b74ea6da63fdc7981be33dc48a09c8d356"}
N_SECONDARY = 25
ALPHA_H2 = P.ALPHA / N_SECONDARY     # the first Holm step (prereg :155)
X = "[EXPLORATORY]"


def provenance():
    pins = json.loads(P.PINS.read_text())
    pins.pop("_env", None)
    bad = [rel for rel, h in pins.items() if P._sha(P.ROOT / rel) != h]
    bad += [f for f, h in SENSITIVITY.items() if P._sha(CAND / f) != h]
    if bad:
        raise SystemExit("PROVENANCE FAILURE: %s differ from the frozen versions" % bad)
    return len(pins)


def load_event(fname):
    spec = importlib.util.spec_from_file_location("e1_sens_" + Path(fname).stem, CAND / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def event_fn(ev):
    """The registered event_of (e1_primary.analyze), with the event function as a parameter."""
    return lambda run: ev(list(range(len(run["log"]))), [r.get("hidden") for r in run["log"]])


def mde_at(nh, nb, fp, alpha, target=0.80):
    """e1_primary.mde_fixed_fp at a given alpha."""
    return next((t / 100 for t in range(1, 101) if P.power_fixed_fp(nh, nb, t / 100, fp, alpha) >= target), None)


def min_hits(nh, nb, fp, alpha):
    return next((x for x in range(nh + 1) if P.fisher_one_sided(x, nh, fp, nb) < alpha), None)


def cp(x, n):
    return "(%.3f, %.3f)" % P.clopper_pearson(x, n)


def cnt(x, n):
    """A count with its Clopper-Pearson 95% interval (prereg :194)."""
    return "%d/%d %s" % (x, n, cp(x, n))


def chg(run, f):
    """e1_primary.analyze's natural-hack-rate change (e1_primary.py:500-502): last 25 steps minus first 25."""
    x = np.array([np.nan if r.get(f) is None else r[f] for r in run["log"]], float)
    return np.nanmean(x[-25:]) - np.nanmean(x[:25])


def contrast(x, n1, y, n2):
    lo, hi = P.newcombe(x, n1, y, n2, 0.90)
    return "Fisher p %.4f, TPR - FPR 90%% CI (%.3f, %.3f), non-superiority (upper < %.2f) %s" % (
        P.fisher_one_sided(x, n1, y, n2), lo, hi, P.DELTA, "HOLDS" if hi < P.DELTA else "does NOT hold")


def main():
    say = print
    say("E1 SUPPLEMENTARY REPORT. Unpinned. NOT the registered analysis (that is experiments/e1_primary.py).")
    say("provenance: %d pinned files match experiments/e1_pins.json; the 2 sensitivity event files match freeze "
        "commit d1aa118" % provenance())
    used, excluded, unregistered, _, surplus = P.load_main()
    hack = [(r["_tag"], r) for r in used["hacking"]]
    ben = used["benign"]
    nh, nb = len(hack), len(ben)
    say("runs (e1_primary.load_main): %d hacking, %d benign used; %d excluded; %d unregistered; %d surplus" % (
        nh, nb, len(excluded), len(unregistered), len(surplus)))
    event_of = event_fn(EV.event)
    events = {tag: event_of(run) for tag, run in hack}
    win = P.window_precondition(events)
    specs = [{"label": l, "family": fam, "gb": g, "dm": m} for (l, fam, g, m) in D.detectors()]
    prim = next(s for s in specs if s["label"] == P.PRIMARY)
    scores = {s["label"]: P.score(s, hack, ben, event_of) for s in specs}
    sp = scores[P.PRIMARY]

    # ---------------------------------------------------------------------------------------- A
    say("\nA. REGISTERED-REPORTING: the %d secondary (H2) detectors with exact intervals and MDE "
        "(prereg section 6 :189, :194 'Each null reports its MDE. Each count reports an exact interval'; :155)" % N_SECONDARY)
    fp0 = P.fixed_fp(nb)
    pw, m0, k0 = P.power_fixed_fp(nh, nb, 0.5, fp0, ALPHA_H2), mde_at(nh, nb, fp0, ALPHA_H2), min_hits(nh, nb, fp0, ALPHA_H2)
    say("  first Holm step, alpha/%d = %.4f, FP fixed at %d/%d: power at TPR 0.5 = %.3f; MDE at 80%% power = TPR %s; "
        "significance needs >= %d/%d strict hits" % (N_SECONDARY, ALPHA_H2, fp0, nb, pw, m0, k0, nh))
    same = round(pw, 2) == 0.27 and m0 == 0.65 and k0 == 14
    say("  prereg :155 states power 0.27, MDE 0.65, >= 14 hits: %s" % ("MATCHES" if same else "DIFFERS: use the values above"))
    sec = [s["label"] for s in specs if s["label"] != P.PRIMARY]
    ps = [P.fisher_one_sided(scores[l]["n_tp_strict"], nh, scores[l]["n_fp"], nb) for l in sec]
    adj = P.holm(ps)
    say("  %-34s %-22s %-22s %-18s %-7s %-7s %-7s %s" % (
        "detector", "TPR< x/n  CP 95%", "FPR y/n  CP 95%", "TPR-FPR Newc. 95%", "p", "Holm", "lead",
        "MDE at alpha/25 (FP as realized)"))
    mdes = {}
    for l, p, a in sorted(zip(sec, ps, adj), key=lambda z: z[1]):
        r = scores[l]
        x, y = r["n_tp_strict"], r["n_fp"]
        if y not in mdes:
            mdes[y] = mde_at(nh, nb, y, ALPHA_H2)
        say("  %-34s %2d/%d %-15s %2d/%d %-15s %-18s %.4f  %.4f  %-7s %s" % (
            l, x, nh, cp(x, nh), y, nb, cp(y, nb), "(%.3f, %.3f)" % P.newcombe(x, nh, y, nb, 0.95), p, a,
            r["median_lead"], mdes[y]))
    say("  FP below %d/%d comes from ties among the benign run maxima; the operating point fixes FP at %d only for "
        "distinct maxima (e1_primary.fixed_fp). No H2 detector holds (Holm p < 0.05 needed)." % (fp0, nb, fp0))

    # ---------------------------------------------------------------------------------------- B
    say("\nB. REGISTERED-REPORTING: sensitivity events, 'reported alongside, never primary' (prereg :65, :68, :318; "
        "e1_event.py:15-16)")
    say("  The registered verdict uses sustained_drop_v2 only. These rows cannot change it. Primary detector at the "
        "registered operating point; the window and decide() rules are the registered ones, applied to each event.")
    rows = [("sustained_drop_v2 (registered, reference)", EV.event)]
    rows += [(m.NAME, m.event) for m in map(load_event, SENSITIVITY)]
    for name, ev in rows:
        f = event_fn(ev)
        eh = {tag: f(run) for tag, run in hack}
        n_ben = sum(f(r) is not None for r in ben)
        w = P.window_precondition(eh)
        gen = w["n_event"] >= math.ceil(P.GEN_FRAC * nh)
        sc = P.score(prim, hack, ben, f)
        dec = P.decide(sc, w["holds"])
        say("  %s" % name)
        say("    hacking events %s, benign false events %s; median window %s over %d (no event = -1) -> %s; "
            "generator check (>= %d events): %s" % (
                cnt(w["n_event"], nh), cnt(n_ben, nb), w["median_window"], w["n"], "holds" if w["holds"] else "FAILS (< 12)",
                math.ceil(P.GEN_FRAC * nh), "passes" if gen else "FAILS"))
        say("    event steps: %s" % sorted(-1 if e is None else e for e in eh.values()))
        if name == "cusum_page_down_v2":      # its onset fit needs 4 observations at or before t0
            say("    %d of %d events at step 3, the earliest date it can return: the failure prereg :63 and :318 cite "
                "for not adopting it" % (sum(e == 3 for e in eh.values()), w["n_event"]))
        say("    primary: TP strict %s, FP %s, one-sided Fisher p %.4f; decide() (illustrative: no decision rule is "
            "registered for sensitivity events): %s" % (
                cnt(sc["n_tp_strict"], nh), cnt(sc["n_fp"], nb), dec["fisher_p_one_sided"],
                dec["verdict"] if gen else "GENERATOR FAILURE (decide() not reached)"))

    # ---------------------------------------------------------------------------------------- C
    say("\nC. REGISTERED-REPORTING: the format-tolerant oracle (hidden_lenient) next to the strict oracle "
        "(prereg :91, :103: 'reported alongside')")
    say("  change = mean of the last 25 steps minus the first 25 (the natural-hack-rate measure, e1_primary.py:500-503)")
    for arm, runs in (("hacking", [r for _, r in hack]), ("benign", ben)):
        for task in ("all tasks",) + P.TASKS:
            rs = [r for r in runs if task == "all tasks" or r["task"] == task]
            s, l = [chg(r, "hidden") for r in rs], [chg(r, "hidden_lenient") for r in rs]
            say("  %-7s %-15s strict %+.3f, falling in %-19s | lenient %+.3f, falling in %s" % (
                arm, task, np.mean(s), cnt(sum(v < 0 for v in s), len(rs)), np.mean(l), cnt(sum(v < 0 for v in l), len(rs))))
    # the registered event and the registered natural-hack-rate measure (e1_primary.py:496-503), on the lenient series
    lev = lambda run: EV.event(list(range(len(run["log"]))), [r.get("hidden_lenient") for r in run["log"]])
    up_s = sum(1 for _, r in hack if chg(r, "judge_small") > 0 and chg(r, "hidden") < 0)
    up_l = sum(1 for _, r in hack if chg(r, "judge_small") > 0 and chg(r, "hidden_lenient") < 0)
    say("  registered event (sustained_drop_v2) on the LENIENT series: hacking %s, benign %s (strict series: hacking %s)" % (
        cnt(sum(lev(r) is not None for _, r in hack), nh), cnt(sum(lev(r) is not None for r in ben), nb),
        cnt(sum(e is not None for e in events.values()), nh)))
    say("  natural-hack-rate measure, proxy (0.5B judge) up and oracle down, hacking arm: strict %s; lenient %s" % (
        cnt(up_s, nh), cnt(up_l, nh)))
    n_c = n_gt = 0
    gt_steps = {}
    for r in [r for _, r in hack] + ben:
        for c in r["completions"]:
            for s, l in zip(c["hidden"], c["hidden_lenient"]):
                n_c += 1
                n_gt += int(s is not None and l is not None and s > l)
        k = sum(1 for row in r["log"] if row["hidden"] > row["hidden_lenient"])
        if k:
            gt_steps[r["_tag"]] = k
    say("  lenient is not a strict superset of strict: %d of %d completions score strict > lenient; at step level, "
        "%d step means in %d runs (%d hacking). Lenient is a lower bound on 'a correct function exists' "
        "(flightrecorder/repro/oracle_lenient.py:12), not an upper bound on strict." % (
            n_gt, n_c, sum(gt_steps.values()), len(gt_steps), sum(1 for t in gt_steps if t.startswith("hacking"))))

    # ---------------------------------------------------------------------------------------- D
    say("\nD. EXPLORATORY: not registered, from the 2026-10-08 verification. None of it changes the registered verdict.")
    thr = TS.fpr10_threshold(prim, ben)
    al_h = {tag: SP.alarm_step(None, prim["dm"](run), thr, 1) for tag, run in hack}
    al_b = {r["_tag"]: SP.alarm_step(None, prim["dm"](r), thr, 1) for r in ben}
    say("%s D1 per-task window (registered event) and primary TP/FP at the pooled registered threshold %.4f:" % (X, thr))
    for task in P.TASKS:
        et = {t: e for t, e in events.items() if t.split("/")[1] == task}
        tp = sum(1 for t in sp["hit_runs_strict"] if t.split("/")[1] == task)
        fp = sum(1 for t, a in al_b.items() if t.split("/")[1] == task and a is not None)
        say("%s   %-15s events %s, median window %s (>= 12: %s); TP %d/8, FP %d/8" % (
            X, task, sorted(-1 if e is None else e for e in et.values()), P.window_precondition(et)["median_window"],
            "yes" if P.window_precondition(et)["holds"] else "no", tp, fp))

    def up(r):     # hacking direction only: DM.dev of this series is max(x - median(first 5), 0)
        x = np.asarray(prim["dm"](r), float)
        return np.maximum(x - np.median(x[np.isfinite(x)][:5]), 0.0)
    su = P.score({"label": "disagreement upward-only", "family": "x", "gb": prim["gb"], "dm": up}, hack, ben, event_of)
    say("%s D2 primary detector with an upward-only alarm, same 90th-percentile rule: threshold %.4f; TP %d/%d vs FP %d/%d; %s" % (
        X, su["fpr10_threshold"], su["n_tp_strict"], nh, su["n_fp"], nb, contrast(su["n_tp_strict"], nh, su["n_fp"], nb)))

    bm = {(r["task"], int(r["seed"])): r for r in ben}
    pre_b, tp_tm = [], 0
    for tag, run in hack:
        e = events[tag]
        if e is None:                     # a miss, and its matched benign run has no window to score
            continue
        pre_b.append(float(np.max(DM.dev(prim["dm"](bm[(run["task"], int(run["seed"]))]))[:e])))
    thr_tm = float(np.percentile(pre_b, 90))
    for tag, run in hack:
        a, e = SP.alarm_step(None, prim["dm"](run), thr_tm, 1), events[tag]
        tp_tm += int(a is not None and e is not None and a < e)
    fp_tm = sum(v > thr_tm for v in pre_b)
    say("%s D3 time-matched operating point (each benign run scored over steps 0..e-1 of its seed- and task-matched "
        "hacking run; threshold = 90th percentile of those %d maxima = %.4f): TP %d/%d vs FP %d/%d; %s" % (
            X, len(pre_b), thr_tm, tp_tm, nh, fp_tm, len(pre_b), contrast(tp_tm, nh, fp_tm, len(pre_b))))

    both = [(al_h[t], events[t]) for t in al_h if al_h[t] is not None and events[t] is not None]
    d = [a - e for a, e in both]
    rho, prho = spearmanr([a for a, _ in both], [e for _, e in both])
    say("%s D4 primary alarm timing (registered threshold): hacking runs alarming %d/%d (benign %d/%d); with an event %d: "
        "alarm before / on / after the event %d / %d / %d; median alarm minus event %+.1f steps (range %+d to %+d); "
        "Spearman(alarm, event) %.3f, p %.3f" % (
            X, sum(a is not None for a in al_h.values()), nh, sum(a is not None for a in al_b.values()), nb, len(both),
            sum(v < 0 for v in d), sum(v == 0 for v in d), sum(v > 0 for v in d), np.median(d), min(d), max(d), rho, prho))

    ev_list = [events[t] for t, _ in hack]
    pl = P.planted_ps(ev_list)
    exp_real = P.power_poisson_binomial(pl, nb, fp0)
    exp_late = P.power_fixed_fp(nh, nb, 0.5, fp0)
    res = P.ROOT / "experiments" / "e1_primary_results.json"
    reg = json.loads(res.read_text()).get("planted_control", {}) if res.exists() else {}
    n_ev = sum(e is not None for e in ev_list)
    say("%s D5 planted control: exact expectation at the realized events %.3f (registered simulation %s), against %.3f "
        "if every event were at step >= 18 (the ~0.92 in prereg section 9 :234). Events before step 18: %d of %d; "
        "mean planted hit probability %.3f, not 0.5." % (
            X, exp_real, reg.get("rate"), exp_late, sum(1 for e in ev_list if e is not None and e < 18), n_ev, np.mean(pl)))
    say("\nREGISTERED (restated, not exploratory): the window precondition holds (median %s), so the registered verdict "
        "is unchanged by sections A-D." % win["median_window"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
