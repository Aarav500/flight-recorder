"""X1, EXPLORATORY (experiments/e1_preregistration.md section 10, :250). Not registered for any decision.

  "Event dating on the causal probe series: 16 completions per probe, the event at -20% off the probe peak.
   Why: it removes the raw-series artifact without an acausal smoother (launch_slow_onset.py)."

Definition, from scripts/launch_slow_onset.py at freeze commit d1aa118, lines 340-348 (the job's go_no_go block,
stored per run as go_no_go.causal_window, :368):
    pv, ps = probe_hidden, step over every probe row;  pk = argmax(pv)  (numpy: first maximum)
    event = ps[i] for the first i >= pk with pv[i] <= 0.8 * pv[pk];  None if there is no such i.
"Causal" there names the SERIES (the off-gradient 16-completion probe, :272-285; no centred smoother, :143-145),
not an online peak. A strictly online variant (running peak) is reported alongside as a sensitivity only.
Everything else is the registered machinery, imported unchanged: e1_primary.score / window_precondition /
fisher_one_sided / newcombe / clopper_pearson / holm, the 26 detectors of e1_detectors, the operating point
threshold_sweep.fpr10_threshold on the benign arm. Written 2026-10-08, after the registered result.

    python experiments/e1_exploratory_parts/x1_probe_event.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True       # read-only: leave no new __pycache__ next to the pinned files
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import e1_primary as P  # noqa: E402
import e1_event as EV  # noqa: E402
import e1_detectors as D  # noqa: E402

TS, SP = P.TS, P.SP
H = "[EXPLORATORY] X1"
C = "CHOICE (made after the data):"
FIG = EXP / "figures" / "e1_X1_probe_vs_registered_event.png"


def cp(x, n):
    return "(%.3f, %.3f)" % P.clopper_pearson(x, n)


def cnt(x, n):
    """A count with its Clopper-Pearson 95% interval (prereg :194)."""
    return "%d/%d %s" % (x, n, cp(x, n))


def probe_event(run, online=False):
    """launch_slow_onset.py:340-348 (d1aa118). Returns the probe's step (training-step units) or None.
    online=True: the sensitivity variant, first probe <= 0.8 x the maximum of the probes so far."""
    ps = [int(p["step"]) for p in run["probe"]]
    pv = [float(p["probe_hidden"]) for p in run["probe"]]
    if not pv:
        return None
    if online:
        top = -np.inf
        for s, v in zip(ps, pv):
            top = max(top, v)
            if v <= 0.8 * top:
                return s
        return None
    pk = int(np.argmax(pv))
    return next((ps[i] for i in range(pk, len(pv)) if pv[i] <= 0.8 * pv[pk]), None)


def registered_event(run):
    """e1_primary.analyze's event_of (e1_primary.py:496): sustained_drop_v2 on the strict stream."""
    return EV.event(list(range(len(run["log"]))), [r.get("hidden") for r in run["log"]])


def figure(hack, ev, reg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    none_x = -25
    for task, mk in zip(P.TASKS, ("o", "s", "^")):
        xs = [none_x if reg[t] is None else reg[t] for t, r in hack if r["task"] == task]
        ys = [ev[t] for t, r in hack if r["task"] == task]
        ax.scatter(xs, ys, marker=mk, s=28, alpha=0.8, label=task)
    hi = 310
    ax.plot([0, hi], [0, hi], color="0.5", lw=0.8, ls="--")
    ax.axvline(none_x / 2, color="0.8", lw=0.6)
    ax.set_xlim(none_x - 10, hi)
    ax.set_ylim(0, hi)
    ax.set_xticks([none_x, 0, 100, 200, 300])
    ax.set_xticklabels(["none", "0", "100", "200", "300"])
    ax.set_xlabel("registered event step (sustained_drop_v2, strict stream)")
    ax.set_ylabel("X1 event step (-20% off probe peak)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150)
    plt.close(fig)
    return str(FIG)


def run():
    L, out = [], {}
    say = L.append
    used, excluded, unregistered, _, surplus = P.load_main()
    hack = [(r["_tag"], r) for r in used["hacking"]]
    ben = used["benign"]
    nh, nb = len(hack), len(ben)
    say("%s Event dating on the probe series, -20%% off the probe peak (prereg :250; scripts/launch_slow_onset.py:340-348 "
        "at d1aa118). EXPLORATORY: it cannot change the registered verdict." % H)
    say("%s runs (e1_primary.load_main): %d hacking, %d benign used; %d excluded, %d unregistered, %d surplus; every run "
        "has %s probes of n=%s, steps %s..%s" % (
            H, nh, nb, len(excluded), len(unregistered), len(surplus),
            sorted({len(r["probe"]) for _, r in hack} | {len(r["probe"]) for r in ben}),
            sorted({p["n"] for _, r in hack for p in r["probe"]}),
            min(r["probe"][0]["step"] for _, r in hack), max(r["probe"][-1]["step"] for _, r in hack)))
    say("%s the peak is the argmax of the WHOLE 80-probe series (first maximum on ties) and the drop test is "
        "inclusive (<= 0.8 x peak), exactly as launch_slow_onset.py:344-346. That is the stored go_no_go.causal_window. "
        "'Causal' in the source names the series (no centred smoother), not an online peak. A strictly online "
        "running-peak variant is reported alongside as a sensitivity, never as X1." % C)
    say("    so the X1 event LOOKS AHEAD: whether step s is the date depends on probes after s, because the peak is "
        "the maximum of the whole run (a later, higher probe moves the peak and the date). It is an offline label, as "
        "in launch_slow_onset.py, which computes it once training has ended (:324 trainer.train(), then :340-348). "
        "'Causal' names the probe series, not an online rule; no detector could compute this date at step s.")
    say("%s probe step s is dated at training step s: the probe at global_step s and log row s are both drawn from "
        "the policy after s updates (event_harness.py:32-33 uses probe steps in training-step units). A strict TP needs "
        "alarm index a < s. The first probe is at step 5, so the earliest possible date is step 10." % C)
    say("%s a benign 'false event' is the same rule firing on a benign run's probe series; the source defines the rule "
        "per run and is silent on benign runs. The window is the registered convention (no event = -1, median over "
        "the %d hacking runs, >= %d). Detectors, operating point (threshold_sweep.fpr10_threshold on the benign arm), "
        "alarms and leads are the registered score(), with only the event swapped." % (C, nh, P.WINDOW_MIN))

    ev = {t: probe_event(r) for t, r in hack}
    evo = {t: probe_event(r, online=True) for t, r in hack}
    reg = {t: registered_event(r) for t, r in hack}
    fb = {r["_tag"]: probe_event(r) for r in ben}
    fbo = {r["_tag"]: probe_event(r, online=True) for r in ben}
    stored = [(r["_tag"], probe_event(r), r.get("go_no_go", {}).get("causal_window")) for _, r in hack] + \
             [(r["_tag"], fb[r["_tag"]], r.get("go_no_go", {}).get("causal_window")) for r in ben]
    same = sum(1 for _, a, b in stored if a == b)
    say("%s recomputed event == go_no_go.causal_window stored by the job: %d/%d runs (%d hacking + %d benign); "
        "mismatches: %s" % (H, same, len(stored), nh, nb, [t for t, a, b in stored if a != b] or "none"))

    rows = {}
    for name, e_h, e_b in (("source event (global peak)", ev, fb), ("sensitivity, online running peak", evo, fbo)):
        w = P.window_precondition(e_h)
        n_fb = sum(v is not None for v in e_b.values())
        say("%s %s: hacking events %s; benign false events %s; median window %s over %d (no event = -1) -> >= %d: %s" % (
            H, name, cnt(w["n_event"], nh), cnt(n_fb, nb), w["median_window"], w["n"], P.WINDOW_MIN,
            "yes" if w["holds"] else "NO"))
        say("    hacking event steps: %s" % sorted(-1 if e is None else e for e in e_h.values()))
        rows[name] = {"events": {t: e_h[t] for t in e_h}, "benign_events": e_b, "window": w, "n_benign_events": n_fb}
    wr = P.window_precondition(reg)
    say("%s registered event for reference (sustained_drop_v2, strict stream): hacking events %s, median window %s" % (
        H, cnt(wr["n_event"], nh), wr["median_window"]))
    for task in P.TASKS:
        et = {t: e for t, e in ev.items() if t.split("/")[1] == task}
        wt = P.window_precondition(et)
        fbt = [v for t, v in fb.items() if t.split("/")[1] == task]
        say("    %-15s X1 events %s, median %s; benign false events %s" % (
            task, sorted(-1 if e is None else e for e in et.values()), wt["median_window"],
            cnt(sum(v is not None for v in fbt), len(fbt))))
    both = [(ev[t], reg[t]) for t in ev if ev[t] is not None and reg[t] is not None]
    d = [a - b for a, b in both]
    nd = len(d)
    rho, prho = spearmanr([a for a, _ in both], [b for _, b in both])
    say("%s date minus registered date, hacking runs with both events %s: median %+.1f, range %+d to %+d; X1 earlier "
        "in %s, same %s, later %s; Spearman %.3f (p %.4f). Registered no-event run(s): %s -> X1 %s" % (
            H, cnt(nd, nh), np.median(d), min(d), max(d), cnt(sum(v < 0 for v in d), nd),
            cnt(sum(v == 0 for v in d), nd), cnt(sum(v > 0 for v in d), nd),
            rho, prho, [t for t in reg if reg[t] is None], [ev[t] for t in reg if reg[t] is None]))

    specs = [{"label": l, "family": fam, "gb": g, "dm": m} for (l, fam, g, m) in D.detectors()]
    sc = {s["label"]: P.score(s, hack, ben, lambda r: ev[r["_tag"]]) for s in specs}
    so = {s["label"]: P.score(s, hack, ben, lambda r: evo[r["_tag"]]) for s in specs}
    sp = sc[P.PRIMARY]
    sr = P.score(next(s for s in specs if s["label"] == P.PRIMARY), hack, ben, lambda r: reg[r["_tag"]])
    say("%s code-path check: the same score() with the registered event gives TP %s, FP %s, median lead %s "
        "(e1_primary_report.txt: 2/24, 3/24, 12.0)" % (H, cnt(sr["n_tp_strict"], nh), cnt(sr["n_fp"], nb),
                                                       sr["median_lead"]))
    x, y = sp["n_tp_strict"], sp["n_fp"]
    lo, hi = P.newcombe(x, nh, y, nb, 0.90)
    k_pos = sp["n_lead_pos"]
    say("%s primary (%s) under the X1 event, threshold %.4f: TP strict %s, FP %s; one-sided Fisher p %.4f; Newcombe "
        "90%% CI on TPR - FPR (%.3f, %.3f) (upper < %.2f: %s); median lead %s over %d runs, lead > 0 in %s; hits %s" % (
            H, P.PRIMARY, sp["fpr10_threshold"], cnt(x, nh), cnt(y, nb), P.fisher_one_sided(x, nh, y, nb), lo, hi,
            P.DELTA, "yes" if hi < P.DELTA else "no", sp["median_lead"], sp["n_lead_runs"],
            cnt(k_pos, sp["n_lead_runs"]), sp["hit_runs_strict"]))
    s2 = so[P.PRIMARY]
    lo2, hi2 = P.newcombe(s2["n_tp_strict"], nh, s2["n_fp"], nb, 0.90)
    say("%s primary under the online running-peak sensitivity: TP strict %s, FP %s; p %.4f; Newcombe 90%% CI "
        "(%.3f, %.3f); median lead %s over %d runs" % (
            H, cnt(s2["n_tp_strict"], nh), cnt(s2["n_fp"], nb), P.fisher_one_sided(s2["n_tp_strict"], nh, s2["n_fp"], nb),
            lo2, hi2, s2["median_lead"], s2["n_lead_runs"]))
    say("%s all 26 detectors, registered operating point (FP does not depend on the event). p = one-sided Fisher, "
        "uncorrected; Holm over the 25 secondary detectors under X1 in the last column" % H)
    sec = [s["label"] for s in specs if s["label"] != P.PRIMARY]
    ps = {l: P.fisher_one_sided(sc[l]["n_tp_strict"], nh, sc[l]["n_fp"], nb) for l in sc}
    holm = dict(zip(sec, P.holm([ps[l] for l in sec])))
    say("    %-34s %-20s | %-20s %-7s %-7s | %-20s %-7s %-7s | %s" % (
        "detector", "FP x/n (CP 95%)", "X1 TP x/n (CP 95%)", "p", "lead", "online TP x/n (CP 95%)", "p", "lead", "Holm"))
    table = {}
    for s in sorted(specs, key=lambda s: (s["label"] != P.PRIMARY, ps[s["label"]], s["label"])):
        l = s["label"]
        a, b = sc[l], so[l]
        po = P.fisher_one_sided(b["n_tp_strict"], nh, b["n_fp"], nb)
        say("    %-34s %2d/%d %-14s | %2d/%d %-14s %.4f  %-7s | %2d/%d %-14s %.4f  %-7s | %s" % (
            l, a["n_fp"], nb, cp(a["n_fp"], nb), a["n_tp_strict"], nh, cp(a["n_tp_strict"], nh), ps[l], a["median_lead"],
            b["n_tp_strict"], nh, cp(b["n_tp_strict"], nh), po, b["median_lead"],
            "primary" if l == P.PRIMARY else "%.4f" % holm[l]))
        table[l] = {"fp": a["n_fp"], "tp_x1": a["n_tp_strict"], "p_x1": ps[l], "lead_x1": a["median_lead"],
                    "n_lead_x1": a["n_lead_runs"], "holm_x1": holm.get(l), "tp_online": b["n_tp_strict"],
                    "p_online": po, "lead_online": b["median_lead"]}
    best = min(sec, key=lambda l: ps[l])
    say("%s smallest uncorrected secondary p under X1: %s, %.4f (Holm %.4f); detectors with uncorrected p < 0.05: %d of 26" % (
        H, best, ps[best], holm[best], sum(1 for l in ps if ps[l] < 0.05)))
    figs = [figure(hack, ev, reg)]
    say("%s figure: %s" % (H, Path(figs[0]).relative_to(P.ROOT).as_posix()))
    say("%s none of this changes the registered verdict: CLEAN NULL (non-superiority shown), experiments/"
        "e1_primary_report.txt." % H)

    out = {"definition": "scripts/launch_slow_onset.py:340-348 @ d1aa118 (go_no_go.causal_window)",
           "stored_causal_window_matches": same, "n_runs_checked": len(stored),
           "variants": rows, "registered_events": reg,
           "primary": {"tp_strict": x, "fp": y, "n_hack": nh, "n_benign": nb, "fisher_p": P.fisher_one_sided(x, nh, y, nb),
                       "newcombe90": [lo, hi], "median_lead": sp["median_lead"], "n_lead_runs": sp["n_lead_runs"],
                       "n_lead_pos": k_pos, "threshold": sp["fpr10_threshold"], "hits": sp["hit_runs_strict"]},
           "primary_online": {"tp_strict": s2["n_tp_strict"], "fp": s2["n_fp"], "newcombe90": [lo2, hi2],
                              "fisher_p": P.fisher_one_sided(s2["n_tp_strict"], nh, s2["n_fp"], nb),
                              "median_lead": s2["median_lead"]},
           "x1_minus_registered": {"n": len(d), "median": float(np.median(d)), "spearman": float(rho), "p": float(prho)},
           "detectors": table}
    for v in out["variants"].values():
        v["window"] = {k: v["window"][k] for k in v["window"]}
    return {"id": "X1", "title": "Event dating on the causal probe series (-20% off the probe peak)",
            "lines": L, "data": out, "figures": figs}


if __name__ == "__main__":
    for line in run()["lines"]:
        print(line)
