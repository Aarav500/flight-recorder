"""X4, EXPLORATORY (experiments/e1_preregistration.md section 10, :253). Not registered for any decision.

  "Lead time against window: Spearman, replicating the 0.83 and 0.87 at tex:710.
   Why: tests the 'no warning was available' account."

Definition, from paper/analysis/onset_profile.py (written 2026-08-17, before the freeze; paper/ is gitignored, so
d1aa118 cannot hold it; its SHA-256 is checked below), the code behind tex:710:
    window    = event step - first logged step                                  (:468-474, start = first logged step)
    alarm     = grid_benchmark.first_move (3-sigma first move) of each of the six reference detectors
                grid_benchmark.DETS, on the |disagreement| ("gb") variant      (alarms, :345-362)
    best lead = max over those detectors of (event - alarm); None without an event or an alarm (best_lead, :365-375)
    statistic = spearman (ordinal ranks, :379-387), two-sided permutation p, 20000 perms, seed 0 (perm_p_corr,
                :390-403), plus the OLS fit "best lead = a x window + b" and the lead <= window self-check
                (leadcorr, :665-687); one detector at a time in per_detector (:689-703); used at :706-720.
The paper's numbers are that code's output in paper/stats/onset_profile.json: 0.83 = grid_hacking_defB_smoothed
(n 9, perm p 0.0078), 0.87 = all_hacking_defB (raw, n 60), 0.96 = all_hacking_defB_smoothed (n 45). Both 0.83 and
0.87 were computed on Definition B (peak_drop 20%; smoothed and raw). The headline here uses E1's registered event
instead (e1_event, sustained_drop_v2 on the strict series), a choice made after the data and printed as one.
E1 logs start at step 0, so window = event step, the same quantity as e1_primary.window_precondition. The
registered lead in e1_primary.score is the same event - first_move.
Like-for-like check: the same statistic (same six first moves, OP.best_lead, OP.spearman / perm_p_corr) under
the Definition B events behind tex:710 (raw peak_drop 20% = threshold_sweep.event_index as X2 uses it; smoothed
peak_drop 20% = x2_published_events.smoothed_peak_drop, onset_profile SMOOTH_W = 15, acausal) and the published
persist=1 event (threshold_sweep.ANCHORS published_baseline_60). Benign context: the |disagreement| 3-sigma first
move on the 24 benign runs. Written 2026-10-08, after the registered result.

    python experiments/e1_exploratory_parts/x4_lead_vs_window.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True       # read-only: leave no new __pycache__ next to the pinned files
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))
import numpy as np  # noqa: E402
from scipy.stats import norm, spearmanr  # noqa: E402

import e1_primary as P  # noqa: E402
import e1_event as EV  # noqa: E402
import e1_detectors as D  # noqa: E402

GB, TS, SP = P.GB, P.TS, P.SP
SOURCE = P.ROOT / "paper" / "analysis" / "onset_profile.py"
SOURCE_JSON = P.ROOT / "paper" / "stats" / "onset_profile.json"
SOURCE_SHA = {SOURCE: "57e3a9b97ae4c226217462b24a431a58672662332625216acad88bbb8949e666",
              SOURCE_JSON: "13274b11dcb3619eda4082b51a0e8bd92f26e5f7951f7e1e85c5b779f8412b07"}
_bad = [p.name for p, h in SOURCE_SHA.items() if P._sha(p) != h]
if _bad:
    raise SystemExit("PROVENANCE FAILURE: %s differ from the versions this analysis was written against" % _bad)
_saved = list(sys.path)
sys.path.insert(0, str(SOURCE.parent))
import onset_profile as OP  # noqa: E402  the definition itself: best_lead, spearman, perm_p_corr
sys.path[:] = _saved
from e1_exploratory_parts import x2_published_events as X2  # noqa: E402  X2's smoothed_peak_drop, not copied

PUB = {k: (how, frac) for k, how, frac in TS.ANCHORS}["published_baseline_60"]     # ("baseline_drop", 0.6)
assert 0.20 in TS.DROPS and PUB == ("baseline_drop", 0.60)

H = "[EXPLORATORY] X4"
C = "CHOICE (made after the data):"
FIG = EXP / "figures" / "e1_X4_lead_vs_window.png"
TEX = {"grid_hacking_defB_smoothed": "0.83 (grid, smoothed event)", "all_hacking_defB": "0.87 (all 60, raw event)",
       "all_hacking_defB_smoothed": "0.96 (45, smoothed event)"}


def cp(x, n):
    return "(%.3f, %.3f)" % P.clopper_pearson(x, n)


def cnt(x, n):
    """A count with its Clopper-Pearson 95% interval (prereg :194)."""
    return "%d/%d %s" % (x, n, cp(x, n))


def rho_ci(r, n, conf=0.95):
    """Fisher-z interval for Spearman's rho with the Bonett-Wright variance (1 + r^2/2) / (n - 3)."""
    if r is None or n < 4 or abs(r) >= 1:
        return None
    z, se = math.atanh(r), math.sqrt((1 + r * r / 2) / (n - 3))
    q = float(norm.ppf(1 - (1 - conf) / 2))
    return math.tanh(z - q * se), math.tanh(z + q * se)


def leadcorr(w, l):
    """onset_profile.leadcorr (:665-687) on paired lists (window, lead); runs missing either are dropped."""
    pairs = [(a, b) for a, b in zip(w, l) if a is not None and b is not None]
    if len(pairs) < 3:
        return {"n": len(pairs), "rho": None, "perm_p": None}
    x, y = [a for a, _ in pairs], [b for _, b in pairs]
    slope, icpt = np.linalg.lstsq(np.array([[a, 1.0] for a in x]), np.array(y, float), rcond=None)[0]
    fr = [b / a for a, b in pairs if a > 0]
    sc = spearmanr(x, y)
    return {"n": len(pairs), "rho": OP.spearman(x, y), "perm_p": OP.perm_p_corr(x, y),
            "rho_scipy_avg_ranks": round(float(sc.correlation), 3) if np.isfinite(sc.correlation) else None,
            "p_scipy": float(sc.pvalue) if np.isfinite(sc.pvalue) else None,
            "slope": round(float(slope), 3), "intercept": round(float(icpt), 3),
            "median_window": float(np.median(x)), "median_lead": float(np.median(y)),
            "median_frac_of_ceiling": round(float(np.median(fr)), 3) if fr else None,
            "n_lead_pos": sum(1 for b in y if b > 0), "violations": sum(1 for a, b in pairs if b > a + 1e-9)}


def pp(p):
    """The source's permutation p is (hits + 1) / 20001, so its floor is 5e-05."""
    return "< 0.0001 (floor 1/20001)" if p is not None and p < 1e-4 else ("%.4f" % p if p is not None else "n/a")


def fmt(tag, s, ci=False):
    if s["rho"] is None:
        return "%s %s: n=%d, too few pairs for the source's spearman (needs >= 4)" % (H, tag, s["n"])
    out = "%s %s: n=%d, Spearman rho %.3f, permutation p %s (scipy average-rank rho %.3f, p %.2g)" % (
        H, tag, s["n"], s["rho"], pp(s["perm_p"]), s["rho_scipy_avg_ranks"], s["p_scipy"])
    if ci:
        lo, hi = rho_ci(s["rho"], s["n"])
        out += ", 95%% CI (%.3f, %.3f)" % (lo, hi)
    return out


def first_moves(run, specs):
    out = {}
    for sp in specs:
        i = GB.first_move(sp["gb"](run))
        if i is not None:
            out[sp["label"]] = float(i)        # E1 logs are indexed 0..T-1 (registered exclusion (c)): index = step
    return out


def op_alarms(run, specs, thr):
    """The registered operating point: e1_primary.score's alarm, SP.alarm_step at TS.fpr10_threshold (benign arm)."""
    out = {}
    for sp in specs:
        a = SP.alarm_step(None, sp["dm"](run), thr[sp["label"]], 1)
        if a is not None:
            out[sp["label"]] = float(a)
    return out


def defb_events():
    """(key, label, tex:710 source key or None, event_of) for the like-for-like check. threshold_sweep.event_index
    and X2.smoothed_peak_drop read run["rows"]; a shallow copy with rows = log leaves the loaded runs unchanged."""
    rows_of = lambda r: dict(r, rows=r["log"])  # noqa: E731
    return [("defB_raw_peak_drop20", "raw peak_drop 20% (Definition B; tex:710's 0.87 event)", "all_hacking_defB",
             lambda r: TS.event_index(rows_of(r), "peak_drop", TS.peak_frac(0.20))),
            ("defB_smoothed_peak_drop20_w%d" % OP.SMOOTH_W, "smoothed peak_drop 20%%, centred w=%d (Definition B; "
             "tex:710's 0.83 event; ACAUSAL, a reference label only)" % OP.SMOOTH_W, "grid_hacking_defB_smoothed",
             lambda r: X2.smoothed_peak_drop(rows_of(r))),
            ("published_baseline_60", "published baseline_drop 0.6, persist=1 (threshold_sweep.ANCHORS)", None,
             lambda r: TS.event_index(rows_of(r), *PUB))]


def windows(ws):
    """Median window two ways: over the runs with an event, and the registered window_precondition convention
    (a run without an event counts as -1)."""
    got = [w for w in ws if w is not None]
    wp = P.window_precondition({i: (None if w is None else int(w)) for i, w in enumerate(ws)})
    return {"median_with_event": (float(np.median(got)) if got else None), "n_with_event": len(got),
            "median_registered": wp["median_window"], "n": len(ws)}


def wtxt(w):
    return "median window %s over the %d runs with an event; %s by the registered convention (no event = -1, over %d)" % (
        w["median_with_event"], w["n_with_event"], w["median_registered"], w["n"])


def figure(rows, fit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"count_vowels": "#2a78d6", "is_prime": "#eb6834", "second_largest": "#1baf7a"}
    mk = {"count_vowels": "o", "is_prime": "s", "second_largest": "^"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharex=True, sharey=True, facecolor="#fcfcfb")
    for ax, key, lab in ((axes[0], "best_lead", "best lead, six reference detectors"),
                         (axes[1], "primary_lead", "lead, disagreement (primary)")):
        ax.set_facecolor("#fcfcfb")
        top = max(r["window"] for r in rows if r["window"] is not None) * 1.05
        ax.plot([0, top], [0, top], color="#8a8985", lw=1.2, ls="--", label="ceiling: lead = window")
        ax.axhline(0, color="#c3c2b7", lw=1)
        for t in col:
            pts = [(r["window"], r[key]) for r in rows if r["task"] == t and r["window"] is not None
                   and r[key] is not None]
            if pts:
                ax.scatter(*zip(*pts), s=42, color=col[t], marker=mk[t], edgecolor="#fcfcfb", linewidth=1.5,
                           label=t, zorder=3)
        if key == "best_lead":
            xs = np.array([0, top])
            ax.plot(xs, fit["slope"] * xs + fit["intercept"], color="#52514e", lw=1,
                    label="OLS %.2f x window %+.1f" % (fit["slope"], fit["intercept"]))
        ax.set_xlabel("window = registered event step", color="#52514e")
        ax.set_ylabel(lab + " (steps)", color="#52514e")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors="#52514e")
        ax.grid(alpha=0.25, lw=0.6)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.text(0.01, 0.01, "EXPLORATORY (E1 X4). Lead = event - 3-sigma first move; negative = alarm after event.",
             fontsize=7.5, color="#52514e")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(FIG)


def run():
    lines = []
    say = lines.append
    used, *_ = P.load_main()
    hack, ben = used["hacking"], used["benign"]
    nh = len(hack)
    say("%s Lead time against window (Spearman): does the tex:710 correlation recur in E1. EXPLORATORY: it cannot "
        "change the registered verdict (CLEAN NULL, non-superiority shown)." % H)
    src = json.loads(SOURCE_JSON.read_text())["window_vs_lead"]
    say("%s source check: paper/stats/onset_profile.json (output of the definition code) gives %s." % (H, "; ".join(
        "%s -> rho %s, n %d, perm p %s, OLS %s" % (TEX[k], src[k]["spearman_window_vs_best_lead"], src[k]["n_pairs"],
                                                  src[k]["perm_p"], src[k]["ols_best_lead_on_window"]) for k in TEX)))
    say("%s definition: paper/analysis/onset_profile.py (pre-freeze, 2026-08-17, sha256 %s...) best_lead :365-375, "
        "spearman :379-387, perm_p_corr :390-403, leadcorr :665-687, per_detector :689-703; imported, not copied."
        % (H, SOURCE_SHA[SOURCE][:12]))
    say(C + " the headline event is E1's registered sustained_drop_v2 on the strict series (window = event step; logs "
        "start at 0), not the event behind tex:710: its 0.83 and 0.87 were computed on Definition B (peak_drop 20%%, "
        "onset_profile.py), 0.83 on the smoothed series (w=%d, grid hacking, n 9) and 0.87 on the raw series (all 60). "
        "The like-for-like check below reruns the statistic under Definition B." % OP.SMOOTH_W)
    say(C + " the detector set for 'best lead' is the source's six reference detectors (grid_benchmark.DETS: reward "
        "variance, gradient norm, disagreement, held-out judge drop, KL, entropy), each on its e1_detectors 'gb' series. "
        "The 8 E1 base signals (adding completion length and truncation rate) and all 26 E1 detectors are sensitivities.")
    say(C + " lead uses the source's (and e1_primary.score's) 3-sigma first move, not the operating point, because "
        "that is how tex:710 defines it. Leads at the registered operating point (TS.fpr10_threshold on the 24 benign "
        "runs, SP.alarm_step) are a sensitivity.")
    say(C + " the source's spearman uses ordinal ranks (argsort of argsort, numpy's default sort, which is not "
        "stable), so tied windows and tied leads are ordered by numpy's sort, not by run order; rho can shift by up to a "
        "few hundredths with the tie order. scipy's average-rank rho is printed beside it as a cross-check.")
    say(C + " 'the tex:710 correlation recurs' = headline rho > 0 with permutation p < 0.05. That is a statement "
        "about the correlation only, not about warning being available (see the mechanical caveat). The 95% CI on rho "
        "is Fisher-z with the Bonett-Wright variance; it is used only to say whether 0.83 and 0.87 are inside it.")
    say(C + " like-for-like events, all on the strict series and scored with the same six first moves and statistic: "
        "raw peak_drop 20%% = threshold_sweep.event_index(how='peak_drop', frac=peak_frac(0.20)), as X2 uses it; "
        "smoothed peak_drop 20%% = x2_published_events.smoothed_peak_drop (onset_profile.smooth, centred w = "
        "SMOOTH_W = %d, then peak_drop 20%%; imported); published persist=1 = threshold_sweep.ANCHORS "
        "published_baseline_60 (baseline_drop 0.6). tex:710 also used the grid and all-60 run sets; E1 has only its "
        "24 hacking runs, so the run set is not like-for-like." % OP.SMOOTH_W)

    ev_of = lambda r: EV.event(list(range(len(r["log"]))), [x.get("hidden") for x in r["log"]])  # noqa: E731
    specs = [{"label": l, "family": f, "gb": g, "dm": m} for (l, f, g, m) in D.detectors()]
    six = [s for s in specs if s["label"] in GB.DETS]
    eight = [s for s in specs if s["label"] in D.BASE]
    assert [s["label"] for s in six] == [n for n in D.BASE if n in GB.DETS] and len(six) == 6 and len(eight) == 8
    thr = {s["label"]: TS.fpr10_threshold(s, ben) for s in six}
    dbe = defb_events()

    rows = []
    for r in hack:
        e = ev_of(r)
        fm6, fm8, fm26 = first_moves(r, six), first_moves(r, eight), first_moves(r, specs)
        b6, b8, b26 = (OP.best_lead(e, fm, 0) for fm in (fm6, fm8, fm26))
        db = {}
        for key, _, _, f in dbe:
            eb = f(r)
            db[key] = {"window": (None if eb is None else float(eb)), "best_lead": OP.best_lead(eb, fm6, 0)["best_lead"]}
        op = OP.best_lead(e, op_alarms(r, six, thr), 0)
        pm = fm6.get(P.PRIMARY)
        pop = op_alarms(r, [s for s in six if s["label"] == P.PRIMARY], thr).get(P.PRIMARY)
        rows.append({"tag": r["_tag"], "task": r["task"], "window": (None if e is None else float(e)),
                     "best_lead": b6["best_lead"], "best_detector": b6["best_detector"],
                     "best_alarm": (None if b6["best_lead"] is None else float(e) - b6["best_lead"]),
                     "best_lead_8": b8["best_lead"], "best_lead_26": b26["best_lead"],
                     "best_lead_op": op["best_lead"],
                     "primary_lead": (None if e is None or pm is None else float(e) - pm),
                     "primary_lead_op": (None if e is None or pop is None else float(e) - pop),
                     "primary_first_move": pm, "defB": db})

    n_ev = sum(r["window"] is not None for r in rows)
    win = P.window_precondition({r["tag"]: (None if r["window"] is None else int(r["window"])) for r in rows})
    reg = P.score(next(s for s in specs if s["label"] == P.PRIMARY), [(r["_tag"], r) for r in hack], ben, ev_of)
    wv = [r["window"] for r in rows if r["window"] is not None]
    say("%s hacking runs with a registered event (the windows): %s; median window %.1f over those, range %d-%d "
        "(registered window_precondition, a no-event run counting as failing: median %s over %d)." % (
            H, cnt(n_ev, nh), np.median(wv), min(wv), max(wv), win["median_window"], win["n"]))
    col = lambda k: [r[k] for r in rows]  # noqa: E731
    res = {"best6": leadcorr(col("window"), col("best_lead"))}
    s = res["best6"]
    lo, hi = rho_ci(s["rho"], s["n"])
    say("%s best lead defined (event and at least one first move): %s." % (H, cnt(s["n"], nh)))
    say(fmt("HEADLINE window vs best lead (six reference detectors)", s, ci=True))
    say("  OLS best lead = %.3f x window %+.2f (tex:710: ~0.85 x window - 6); median window %.1f over the %d runs "
        "with an event and a lead, median best lead %.1f, median lead/window %.3f; best lead > 0 in %s; lead > window "
        "in %d runs (self-check, must be 0)." % (
            s["slope"], s["intercept"], s["median_window"], s["n"], s["median_lead"], s["median_frac_of_ceiling"],
            cnt(s["n_lead_pos"], s["n"]), s["violations"]))
    ba = [r["best_alarm"] for r in rows if r["best_alarm"] is not None]
    ra = leadcorr(col("window"), col("best_alarm"))
    say("  lead = window - alarm: the best detector's first-move step has median %.1f, IQR (%.1f, %.1f), range %d-%d; "
        "Spearman(window, best alarm step) %s, perm p %s." % (
            np.median(ba), *np.percentile(ba, [25, 75]), min(ba), max(ba), ra["rho"], pp(ra["perm_p"])))
    bd = {}
    for r in rows:
        if r["best_detector"]:
            bd[r["best_detector"]] = bd.get(r["best_detector"], 0) + 1
    say("  best detector per run: %s." % ", ".join("%s %s" % (k, cnt(v, s["n"])) for k, v in
                                                  sorted(bd.items(), key=lambda kv: -kv[1])))

    res["primary"] = leadcorr(col("window"), col("primary_lead"))
    say(fmt("PRIMARY ALONE window vs disagreement lead (per_detector)", res["primary"], ci=True))
    say("  OLS lead = %.3f x window %+.2f; median lead %.1f (e1_primary.score median lead %s, the same quantity); "
        "lead > 0 in %s." % (res["primary"]["slope"], res["primary"]["intercept"], res["primary"]["median_lead"],
                             reg["median_lead"], cnt(res["primary"]["n_lead_pos"], res["primary"]["n"])))

    pspec = next(sp for sp in six if sp["label"] == P.PRIMARY)
    fmh = [r["primary_first_move"] for r in rows]
    fmb = [GB.first_move(pspec["gb"](r)) for r in ben]
    mh, mb = [float(v) for v in fmh if v is not None], [float(v) for v in fmb if v is not None]
    ev_med = float(np.median(wv))
    by_ev = sum(1 for v in mb if v <= ev_med)
    res["primary_first_move"] = {
        "hacking": {"moved": len(mh), "n": nh, "median_step": float(np.median(mh)),
                    "iqr": [float(v) for v in np.percentile(mh, [25, 75])], "steps": sorted(mh)},
        "benign": {"moved": len(mb), "n": len(ben), "median_step": float(np.median(mb)),
                   "iqr": [float(v) for v in np.percentile(mb, [25, 75])], "steps": sorted(mb),
                   "moved_by_median_hacking_event_step": by_ev}}
    say("%s benign context, |disagreement| 3-sigma first-move step (grid_benchmark.first_move on its gb series, the "
        "alarm behind the registered lead): hacking median %.1f, IQR (%.1f, %.1f), moved in %s; benign median %.1f, "
        "IQR (%.1f, %.1f), moved in %s; benign runs moving by step %.0f (the median hacking event step): %s." % (
            H, np.median(mh), *np.percentile(mh, [25, 75]), cnt(len(mh), nh), np.median(mb),
            *np.percentile(mb, [25, 75]), cnt(len(mb), len(ben)), ev_med, cnt(by_ev, len(ben))))
    say("%s the positive registered first-move lead (median %+.1f, e1_primary.score) comes from an early first move "
        "(hacking median step %.1f vs median event step %.1f) that also fires in benign runs (%s, median step %.1f); "
        "the lead is event minus a move that is not specific to hacking." % (
            H, reg["median_lead"], np.median(mh), ev_med, cnt(len(mb), len(ben)), np.median(mb)))

    say("%s per task (n <= 8 each; the source's spearman needs >= 4 pairs). Windows are given in both conventions:" % H)
    res["per_task"] = {}
    for t in P.TASKS:
        sub = [r for r in rows if r["task"] == t]
        b = leadcorr([r["window"] for r in sub], [r["best_lead"] for r in sub])
        pr = leadcorr([r["window"] for r in sub], [r["primary_lead"] for r in sub])
        w = windows([r["window"] for r in sub])
        res["per_task"][t] = {"best6": b, "primary": pr, "window": w}
        say("  %s: best lead n=%d rho %s perm p %s, best lead > 0 in %s; primary n=%d rho %s perm p %s, lead > 0 in %s; "
            "%s." % (t, b["n"], b["rho"], pp(b["perm_p"]), cnt(b.get("n_lead_pos", 0), b["n"]) if b["n"] else "0/0",
                     pr["n"], pr["rho"], pp(pr["perm_p"]),
                     cnt(pr.get("n_lead_pos", 0), pr["n"]) if pr["n"] else "0/0", wtxt(w)))

    say("%s sensitivities (same statistic):" % H)
    for key, lab in (("best_lead_8", "best of the 8 E1 base signals (+ length, truncation)"),
                     ("best_lead_26", "best of all 26 E1 detectors"),
                     ("best_lead_op", "best of six, alarm at the registered operating point"),
                     ("primary_lead_op", "disagreement alone, alarm at the registered operating point")):
        res[key] = leadcorr(col("window"), col(key))
        q = res[key]
        say("  " + fmt(lab, q)[len(H) + 1:] + ("; OLS %.3f x window %+.2f; median lead %.1f; lead > 0 in %s" % (
            q["slope"], q["intercept"], q["median_lead"], cnt(q["n_lead_pos"], q["n"])) if q["rho"] is not None else ""))

    say("%s LIKE-FOR-LIKE under the events behind tex:710 (same six first moves, OP.best_lead, OP.spearman / "
        "perm_p_corr). The best alarm is the earliest of the six first moves whatever the event (best_lead maximises "
        "event - alarm), so it is the same step under every event below:" % H)
    res["defB"] = {}
    for key, lab, sk, _ in dbe:
        dw = [r["defB"][key]["window"] for r in rows]
        q = leadcorr(dw, [r["defB"][key]["best_lead"] for r in rows])
        w = windows(dw)
        ci = rho_ci(q["rho"], q["n"])
        ins = None if sk is None or ci is None else ci[0] <= src[sk]["spearman_window_vs_best_lead"] <= ci[1]
        res["defB"][key] = {"label": lab, "stats": q, "window": w, "rho_ci95": ci, "tex_key": sk, "tex_inside_ci": ins}
        say("  %s: hacking events %s; %s." % (lab, cnt(w["n_with_event"], nh), wtxt(w)))
        if q["rho"] is None:
            say("    n=%d, too few pairs for the source's spearman (needs >= 4)" % q["n"])
            continue
        say("    " + fmt("window vs best lead", q, ci=True)[len(H) + 1:] + "; OLS %.3f x window %+.2f; median best "
            "lead %.1f; best lead > 0 in %s%s" % (
                q["slope"], q["intercept"], q["median_lead"], cnt(q["n_lead_pos"], q["n"]),
                "" if ins is None else "; tex:710 %s inside this CI: %s" % (
                    src[sk]["spearman_window_vs_best_lead"], "yes" if ins else "no")))
    dr = [(res["defB"][k]["label"].split(" (")[0], res["defB"][k]["stats"]["rho"]) for k, *_ in dbe]
    say("%s like-for-like reading: rho %s, vs %.3f under the registered event. The correlation recurs under every "
        "event, including the persist=1 event with its median window of %s steps (over the runs with an event): it is "
        "what best lead = window - (an event-independent early alarm) gives, whatever the event." % (
            H, "; ".join("%s under %s" % (v, k) for k, v in dr), s["rho"],
            res["defB"]["published_baseline_60"]["window"]["median_with_event"]))

    rec = s["rho"] is not None and s["rho"] > 0 and s["perm_p"] < 0.05
    inside = {k: lo <= src[k]["spearman_window_vs_best_lead"] <= hi for k in ("grid_hacking_defB_smoothed", "all_hacking_defB")}
    say("%s RECURRENCE: %s. E1 rho %.3f (n %d, perm p %s, 95%% CI %.3f-%.3f) vs tex:710 0.83 (inside CI: %s) and "
        "0.87 (inside CI: %s); OLS slope %.2f, intercept %.1f vs ~0.85 and -6. Mechanical caveat: the best detector "
        "first-moves at a near-constant early step (median %.0f, IQR %.1f-%.1f; Spearman with the window %s), so best "
        "lead ~ window - constant and the correlation is largely the window itself, not evidence that warning was "
        "available." % (
            H, "the tex:710 correlation recurs (best lead tracks the window)" if rec else
            "the tex:710 correlation does NOT recur", s["rho"], s["n"], pp(s["perm_p"]), lo, hi,
            "yes" if inside["grid_hacking_defB_smoothed"] else "no", "yes" if inside["all_hacking_defB"] else "no",
            s["slope"], s["intercept"], np.median(ba), *np.percentile(ba, [25, 75]), ra["rho"]))
    q = res["best_lead_op"]
    pt = res["per_task"]
    longw = [t for t in P.TASKS if (pt[t]["window"]["median_registered"] or -1) >= P.WINDOW_MIN]
    shortw = [t for t in P.TASKS if t not in longw]
    wr = lambda ts, tail="": " and ".join("%s (%s%s)" % (t, pt[t]["window"]["median_registered"], tail)  # noqa: E731
                                          for t in ts)
    say("%s reading: the tex:710 account is that lead was capped by a short window. Pooled, E1's windows are long "
        "(median %.0f over the %d runs with an event; %s by the registered convention over %d), and the best "
        "first-move alarm is early and near-constant (above), so the rank correlation is largely the window itself. At "
        "the registered operating point the best of six leads in only %s, and the registered test is null (strict TP "
        "%s vs FP %s). Per task (median window, registered convention, no event = -1; minimum %d): %s%s. "
        "Exploratory only; it does not change the registered verdict." % (
            H, np.median(wv), len(wv), win["median_window"], win["n"], cnt(q["n_lead_pos"], q["n"]),
            cnt(reg["n_tp_strict"], reg["n_hack"]), cnt(reg["n_fp"], reg["n_benign"]), P.WINDOW_MIN,
            ("on %s a short window does not by itself explain the E1 null" % wr(longw)) if longw else "",
            ("%son %s a short window may explain it" % ("; " if longw else "", wr(shortw, ", below the minimum")))
            if shortw else ""))

    figs = [figure(rows, s)]
    return {"id": "X4", "title": "Lead time against window (Spearman): does the tex:710 correlation recur",
            "lines": lines,
            "data": {"runs": rows, "stats": res, "recurs": rec,
                     "source_values": {k: src[k]["spearman_window_vs_best_lead"] for k in TEX},
                     "headline_rho_ci95": [lo, hi], "source_inside_ci": inside,
                     "window_pooled": {"median_with_event": float(np.median(wv)), "n_with_event": len(wv),
                                       "median_registered": win["median_window"], "n": win["n"]}},
            "figures": figs}


if __name__ == "__main__":
    for ln in run()["lines"]:
        print(ln)
