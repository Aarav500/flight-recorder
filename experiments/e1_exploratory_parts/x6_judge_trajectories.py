"""X6, EXPLORATORY (experiments/e1_preregistration.md section 10, :255). Not registered for any decision.

  "Delta-judge disagreement trajectories, with both judges logged in every run.
   Why: the 'disagreement leads hacking' reversal (tex:279-286)."

Definition of the reversal, from paper/neurips_paper.tex (mtime 2026-10-06, before the freeze at d1aa118,
2026-10-07T22:25; paper/ is gitignored, so the tex is not in the commit):
  tex:279-282  "Evaluator disagreement first-moves after the hidden collapse in every seed, or never."
  tex:291-304  tab:lowlr, "First-movement step (departure > 3 sigma from baseline)": hidden 23 / ~6 / 22 against
               disagreement 42 / never / 88 (n=3, lr 2e-6). Produced by paper/analysis/figures_phase2.py:27-29 and
               :79-86 (-> paper/stats/lowlr_firstmove.json) on the per-step `hidden` and `disagree` rows.
  tex:261-263  fig:dis caption: disagreement widens because "the held-out 1.5B judge's score falls".
The first-move rule is grid_benchmark.first_move (paper/analysis/grid_benchmark.py:39-62, SHA-pinned in
e1_pins.json): baseline = mean of the first 3 finite points, noise = std of the first 6, threshold max(3 sigma, 0.04).
It is the rule the registered lead uses (e1_primary.score, :279) and, on fully finite series, the same step as
figures_phase2.first_move (asserted below). Written 2026-10-08, after the registered result (CLEAN NULL).

    python experiments/e1_exploratory_parts/x6_judge_trajectories.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True       # read-only: leave no new __pycache__ next to the pinned files
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))
import numpy as np  # noqa: E402

import e1_primary as P  # noqa: E402
import e1_event as EV  # noqa: E402
import e1_detectors as D  # noqa: E402

GB = P.GB
H = "[EXPLORATORY] X6"
C = "CHOICE (made after the data):"
FIG = EXP / "figures" / "e1_X6_judge_trajectories.png"
LO, HI, MIN_N = -40, 120, 5
SER = (("judge_small", "judge 0.5B", "#2a78d6", "-"), ("judge_big", "judge 1.5B", "#eb6834", "-"),
       ("disagree", "disagreement (0.5B - 1.5B)", "#1baf7a", "-"), ("hidden", "strict oracle", "#4a3aa7", "-"),
       ("hidden_lenient", "lenient oracle", "#4a3aa7", "--"))
WIN_EV = ((-20, 0), (0, 20), (20, 60), (60, 121))         # event-aligned windows, offsets [a, b)
WIN_CAL = ((0, 25), (25, 100), (100, 200), (200, 400))     # calendar windows, steps [a, b)


def cp(x, n):
    return "(%.3f, %.3f)" % P.clopper_pearson(x, n)


def cnt(x, n):
    """A count with its Clopper-Pearson 95% interval (prereg :194)."""
    return "%d/%d %s" % (x, n, cp(x, n)) if n else "0/0"


def ser(run, f):
    return np.array([np.nan if r.get(f) is None else r[f] for r in run["log"]], float)


def chg(run, f):
    """e1_primary.analyze's natural-hack-rate change (e1_primary.py:500-502): last 25 steps minus first 25."""
    x = ser(run, f)
    return float(np.nanmean(x[-25:]) - np.nanmean(x[:25]))


def fm_down(x):
    """Sensitivity only: grid_benchmark.first_move_pred's threshold, counting downward departures only."""
    x = np.asarray(x, float)
    pred = GB.first_move_pred(x)
    if pred is None:
        return None
    b = x[np.isfinite(x)][:3].mean()
    return next((i for i, v in enumerate(x) if np.isfinite(v) and pred(v) and v < b), None)


def order(fd, fo):
    if fo is None:              # ordering is defined only where the oracle first-moves (CHOICE above)
        return "oracle undated"
    if fd is None:
        return "disagreement never"
    return "before" if fd < fo else ("with" if fd == fo else "after")


def tally(pairs):
    """pairs: [(fm_disagreement, fm_oracle)] -> counts over runs whose oracle first-moves (dated), plus undated."""
    o = [order(d, h) for d, h in pairs]
    dated = sum(1 for h in (h for _, h in pairs) if h is not None)
    k = {c: sum(1 for z in o if z == c) for c in ("before", "with", "after", "disagreement never", "oracle undated")}
    k["never_oracle_dated"] = sum(1 for (d, h) in pairs if d is None and h is not None)
    k["dated"], k["n"] = dated, len(pairs)
    return k


def tally_line(k):
    m = k["dated"]
    return ("disagreement first-moves BEFORE the oracle %s, WITH it %s, AFTER it %s, never (oracle dated) %s; "
            "oracle never first-moves %s" % (cnt(k["before"], m), cnt(k["with"], m), cnt(k["after"], m),
                                              cnt(k["never_oracle_dated"], m), cnt(k["n"] - m, k["n"])))


def aligned(runs, events, f):
    """Rows = runs, columns = offsets LO..HI from the run's event; NaN outside the run."""
    M = np.full((len(runs), HI - LO + 1), np.nan)
    for i, (run, e) in enumerate(zip(runs, events)):
        x = ser(run, f)
        for j, off in enumerate(range(LO, HI + 1)):
            if 0 <= e + off < len(x):
                M[i, j] = x[e + off]
    return M


def band(M):
    n = np.isfinite(M).sum(0)
    m = np.array([np.nanmean(M[:, j]) if n[j] else np.nan for j in range(M.shape[1])])
    s = np.array([np.nanstd(M[:, j], ddof=1) / np.sqrt(n[j]) if n[j] > 1 else np.nan for j in range(M.shape[1])])
    m[n < MIN_N], s[n < MIN_N] = np.nan, np.nan
    return m, 1.96 * s, n


def wmean(x, a, b):
    return float(np.nanmean(x[a:b]))


def msem(v):
    v = np.asarray(v, float)
    return "%+.3f (SEM %.3f)" % (v.mean(), v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else "n/a"


def figure(hack_runs, hack_ev, ben):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcdefaults()
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6})
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True, gridspec_kw={"width_ratios": [1.0, 1.25, 1.25]})
    panels = [(axs[0], "Hacking arm (0.5B judge), aligned on registered event", hack_runs, hack_ev, True),
              (axs[1], "Hacking arm (0.5B judge), calendar steps", hack_runs, None, False),
              (axs[2], "Benign arm (1.5B judge), calendar steps", ben, None, False)]
    n_at = {}
    for ax, title, runs, ev, is_al in panels:
        for f, lab, col, ls in SER:
            if is_al:
                M = aligned(runs, ev, f)
                x = np.arange(LO, HI + 1)
            else:
                M = np.vstack([ser(r, f) for r in runs])
                x = np.arange(M.shape[1])
            m, h, n = band(M)
            ax.fill_between(x, m - h, m + h, color=col, alpha=0.13 if ls == "-" else 0.06, lw=0)
            ax.plot(x, m, color=col, lw=1.6 if ls == "-" else 1.0, ls=ls, alpha=1.0 if ls == "-" else 0.7, label=lab)
            if f.startswith("hidden") and runs is not ben:   # selective direct labels: the two oracles, hacking arm
                ax.annotate("strict" if f == "hidden" else "lenient", (x[np.isfinite(m)][-1], m[np.isfinite(m)][-1]),
                            xytext=(3, -10 if f == "hidden" else 4), textcoords="offset points", fontsize=7,
                            color="#52514e")
            if is_al:
                n_at = {o: int(n[o - LO]) for o in (LO, -20, -10, 0, HI)}
        ax.axhline(0, color="#52514e", lw=0.7)
        ax.set_title(title, fontsize=9, color="#0b0b0b", loc="left")
        ax.tick_params(colors="#52514e")
        if is_al:
            ax.axvline(0, color="#52514e", lw=0.9, ls=":")
            ax.set_xlabel("steps from the registered event (sustained_drop_v2, strict oracle)", color="#52514e")
            ax.text(0.02, 0.03, "runs per offset: %d at %d, %d at -20, %d at 0, %d at +%d" % (
                n_at[LO], LO, n_at[-20], n_at[0], n_at[HI], HI), transform=ax.transAxes, fontsize=7, color="#52514e")
        else:
            ax.set_xlabel("GRPO step", color="#52514e")
    axs[0].set_ylabel("score / pass rate (mean, 95% band = 1.96 SEM)", color="#52514e")
    axs[0].set_ylim(-0.35, 1.05)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.text(0.995, 0.005, "EXPLORATORY (X6), not registered", ha="right", fontsize=7, color="#52514e")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150)
    plt.close(fig)
    return n_at


def run():
    L = []
    say = L.append
    used, *_ = P.load_main()
    hack, ben = used["hacking"], used["benign"]
    nh, nb = len(hack), len(ben)
    event_of = lambda r: EV.event(list(range(len(r["log"]))), [x.get("hidden") for x in r["log"]])
    ev = {r["_tag"]: event_of(r) for r in hack}
    he = [r for r in hack if ev[r["_tag"]] is not None]
    ne = len(he)

    # integrity: disagree is exactly judge_small - judge_big, every series fully finite, log indexed 0..T-1
    gap = max(float(np.nanmax(np.abs(ser(r, "judge_small") - ser(r, "judge_big") - ser(r, "disagree"))))
              for r in hack + ben)
    fin = all(np.isfinite(ser(r, f)).all() for r in hack + ben for f, *_ in SER)
    idx = all([x["step"] for x in r["log"]] == list(range(len(r["log"]))) for r in hack + ben)
    # the reward each arm trains on: proxy is judge_small in the hacking arm and judge_big in the benign arm
    pgap = {arm: {j: max(float(np.nanmax(np.abs(ser(r, "proxy") - ser(r, j)))) for r in runs)
                  for j in ("judge_small", "judge_big")} for arm, runs in (("hacking", hack), ("benign", ben))}

    say("%s Delta-judge disagreement trajectories and the 'disagreement leads hacking' reversal (tex:279-286). "
        "EXPLORATORY: cannot change the registered verdict (CLEAN NULL)." % H)
    say("%s runs (e1_primary.load_main): %d hacking, %d benign; hacking runs with a registered event %s; "
        "both judges logged in every run; disagree == judge_small - judge_big to %.1e; all five series finite: %s; "
        "log indexed 0..T-1 (index == step): %s" % (H, nh, nb, cnt(ne, nh), gap, fin, idx))

    say("%s SOURCE: tex:281-282 'disagreement first-moves after the hidden collapse in every seed, or never'; "
        "tab:lowlr (tex:291-304; figures_phase2.py:79-86) dates hidden and disagreement by the 3-sigma first move "
        "(hidden 23/~6/22, disagreement 42/never/88, n=3)." % H)
    say("%s first-move rule = grid_benchmark.first_move (k=3, floor 0.04, baseline = first 3 points, noise = "
        "first 6), the rule of the registered lead (e1_primary.py:279) and of tab:lowlr; applied to the per-step log "
        "series (tab:lowlr also used the per-step `hidden` rows)." % C)
    say("%s disagreement series for the first move = |disagree| (the registered lead's 'gb' series, and what "
        "fig:resolution plots); tab:lowlr's code used the signed series, reported below as a sensitivity." % C)
    say("%s oracle = the STRICT per-step series (option (a), the registered event's series). The first move is "
        "two-sided as published; its direction is reported, and a downward-only move and the lenient oracle are "
        "sensitivities." % C)
    say("%s the ordering is scored on hacking runs with a registered event (n=%d); it is defined only where the strict "
        "oracle first-moves ('dated'). 'Disagreement never' counts as consistent with the reversal (the paper's "
        "'or never')." % (C, ne))
    say("%s 'E1 reproduces the reversal' = no dated run has disagreement first-moving strictly before the oracle "
        "(tex:281 'in every seed'); a weaker majority form (after + never > before + with) is printed too." % C)

    # ------------------------------------------------------------------------------- cross-checks
    fp_ok = None
    try:
        import figures_phase2 as F2                     # the code that produced tab:lowlr (paper/analysis on path)
        fp_ok = all(F2.first_move(x) == GB.first_move(x) for r in hack + ben
                    for x in (ser(r, "hidden"), ser(r, "disagree"), np.abs(ser(r, "disagree"))))
    except Exception as exc:                            # noqa: BLE001  cross-check only
        fp_ok = "not run (%s)" % type(exc).__name__
    prim = next({"label": l, "family": fam, "gb": g, "dm": m} for (l, fam, g, m) in D.detectors() if l == P.PRIMARY)
    sc = P.score(prim, [(r["_tag"], r) for r in hack], ben, event_of)
    fmd = {r["_tag"]: GB.first_move(np.abs(ser(r, "disagree"))) for r in hack + ben}
    mine = sorted(float(ev[r["_tag"]] - fmd[r["_tag"]]) for r in he if fmd[r["_tag"]] is not None)
    lead_ok = mine == sorted(sc["leads"])
    say("%s cross-check: figures_phase2.first_move (tab:lowlr code) == grid_benchmark.first_move on every E1 strict, "
        "signed and |disagree| series: %s. My |disagree| first moves reproduce the registered lead list "
        "(e1_primary.score): %s." % (H, fp_ok, lead_ok))
    say("    (restated from the registered report, not exploratory) event - first move of |disagree|, median %s over "
        "%d runs, positive in %s." % (sc["median_lead"], len(sc["leads"]), cnt(sc["n_lead_pos"], len(sc["leads"]))))

    # ------------------------------------------------------------------------------- (b) first-move ordering
    say("%s (b) FIRST-MOVE ORDERING, hacking runs with an event (n=%d)" % (H, ne))
    rows = []
    for r in he:
        h, d = ser(r, "hidden"), ser(r, "disagree")
        fo, fd = GB.first_move(h), fmd[r["_tag"]]
        thr_h, thr_d = (max(3 * max(x[:6].std(), 1e-6), 0.04) for x in (h, np.abs(d)))
        dirn = None if fo is None else ("up" if h[fo] > h[:3].mean() else "down")
        rows.append({"tag": r["_tag"], "task": r["task"], "event": ev[r["_tag"]], "fm_strict": fo, "fm_strict_dir": dirn,
                     "fm_abs_disagree": fd, "fm_signed_disagree": GB.first_move(d), "fm_strict_down": fm_down(h),
                     "fm_lenient": GB.first_move(ser(r, "hidden_lenient")),
                     "fm_judge_small": GB.first_move(ser(r, "judge_small")),
                     "fm_judge_big": GB.first_move(ser(r, "judge_big")),
                     "thr_strict": round(float(thr_h), 6), "thr_abs_disagree": round(float(thr_d), 6),
                     "order": order(fd, fo)})
    k = tally([(z["fm_abs_disagree"], z["fm_strict"]) for z in rows])
    say("%s   primary form (|disagree| vs strict oracle): %s" % (H, tally_line(k)))
    fm_arm = {}
    for arm, runs in (("hacking", hack), ("benign", ben)):
        mv = [fmd[r["_tag"]] for r in runs if fmd[r["_tag"]] is not None]
        q1, q3 = np.percentile(mv, [25, 75])
        fm_arm[arm] = {"n_moving": len(mv), "n": len(runs), "median_step": float(np.median(mv)),
                       "iqr": [float(q1), float(q3)], "min": int(min(mv)), "max": int(max(mv)), "steps": [int(s) for s in mv]}
    fa, fb = fm_arm["hacking"], fm_arm["benign"]
    fmt = lambda f: "moves in %s, median step %.1f (IQR %.1f-%.1f, range %d-%d)" % (
        cnt(f["n_moving"], f["n"]), f["median_step"], f["iqr"][0], f["iqr"][1], f["min"], f["max"])
    say("%s   context, |disagree| 3-sigma first move by calendar step (same rule, every run, not event-aligned): "
        "hacking %s; benign %s. A |disagree| first move happens in most benign runs too, at a comparable step." % (
            H, fmt(fa), fmt(fb)))
    both = [z for z in rows if z["fm_strict"] is not None and z["fm_abs_disagree"] is not None]
    dlt = [z["fm_strict"] - z["fm_abs_disagree"] for z in both]
    kp, mp, pp = P.sign_test_greater(dlt)
    say("%s   where both move (n=%d): oracle first move minus disagreement first move, median %+.1f steps (range %+d to "
        "%+d); exact sign test that disagreement moves first, %d/%d, one-sided p %.3f (descriptive)" % (
            H, len(both), float(np.median(dlt)), min(dlt), max(dlt), kp, mp, pp))
    ups = sum(1 for z in rows if z["fm_strict_dir"] == "up")
    say("%s   the strict oracle's first move is UPWARD (not a collapse) in %s dated runs; the 3-sigma threshold on the "
        "8-completion strict series is median %.3f (range %.3f-%.3f) vs %.3f for |disagree|, which is why the oracle "
        "goes undated in %d of %d runs" % (
            H, cnt(ups, k["dated"]), np.median([z["thr_strict"] for z in rows]), min(z["thr_strict"] for z in rows),
            max(z["thr_strict"] for z in rows), np.median([z["thr_abs_disagree"] for z in rows]), k["n"] - k["dated"], k["n"]))
    rep = k["before"] == 0
    maj = k["after"] + k["never_oracle_dated"] > k["before"] + k["with"]
    say("%s   Reading on the reversal: E1 %s it in the paper's form ('in every seed': %d dated run(s) have "
        "disagreement strictly first) and %s it in the majority form (after + never %d vs before + with %d)." % (
            H, "REPRODUCES" if rep else "does NOT reproduce", k["before"], "reproduces" if maj else "does NOT reproduce",
            k["after"] + k["never_oracle_dated"], k["before"] + k["with"]))
    per_task = {}
    for t in P.TASKS:
        kt = tally([(z["fm_abs_disagree"], z["fm_strict"]) for z in rows if z["task"] == t])
        per_task[t] = kt
        m = kt["dated"]
        say("%s   %-15s of dated: before %s, with %s, after %s, never %s; oracle undated %s" % (
            H, t, cnt(kt["before"], m), cnt(kt["with"], m), cnt(kt["after"], m), cnt(kt["never_oracle_dated"], m),
            cnt(kt["n"] - m, kt["n"])))
    sens = {}
    for name, fd_key, fo_key in (("signed disagree (tab:lowlr's series)", "fm_signed_disagree", "fm_strict"),
                                 ("strict oracle, downward first move only", "fm_abs_disagree", "fm_strict_down"),
                                 ("lenient oracle", "fm_abs_disagree", "fm_lenient")):
        ks = tally([(z[fd_key], z[fo_key]) for z in rows])
        sens[name] = ks
        say("%s   sensitivity, %s: %s" % (H, name, tally_line(ks)))
    bk = tally([(fmd[r["_tag"]], GB.first_move(ser(r, "hidden"))) for r in ben])
    sens["benign arm, calendar (no event)"] = bk
    say("%s   context, benign arm (no event; n=%d): %s" % (H, nb, tally_line(bk)))
    jj = [(z["fm_judge_small"], z["fm_judge_big"]) for z in rows]
    js_first = sum(1 for a, b in jj if a is not None and (b is None or a < b))
    jb_first = sum(1 for a, b in jj if b is not None and (a is None or b < a))
    say("%s   which judge first-moves first (hacking, n=%d): judge 0.5B %s, judge 1.5B %s, same step %d, neither %d" % (
        H, ne, cnt(js_first, ne), cnt(jb_first, ne), sum(1 for a, b in jj if a is not None and a == b),
        sum(1 for a, b in jj if a is None and b is None)))
    say("%s   per run: tag, event, strict first move (direction), |disagree| first move, order" % H)
    for z in rows:
        say("    %-28s event %3d  strict %-10s |disagree| %-5s %s" % (
            z["tag"], z["event"], "never" if z["fm_strict"] is None else "%d %s" % (z["fm_strict"], z["fm_strict_dir"]),
            "never" if z["fm_abs_disagree"] is None else z["fm_abs_disagree"], z["order"]))

    # ------------------------------------------------------------------------------- which judge drives disagreement
    say("%s WHICH JUDGE DRIVES DISAGREEMENT: change = mean of the last 25 steps minus the first 25 "
        "(the natural-hack-rate measure, e1_primary.py:500-502)" % H)
    say("%s 'driver' = the judge with the larger absolute change; disagree change = judge_small change - "
        "judge_big change exactly." % C)
    drive = {}
    for arm, runs in (("hacking", hack), ("benign", ben)):
        c = {f: [chg(r, f) for r in runs] for f in ("judge_small", "judge_big", "disagree", "hidden")}
        n = len(runs)
        small_dom = sum(1 for a, b in zip(c["judge_small"], c["judge_big"]) if abs(a) > abs(b))
        drive[arm] = {f: [round(v, 4) for v in vs] for f, vs in c.items()}
        drive[arm]["small_dominates"] = small_dom
        say("%s   %-7s judge 0.5B %s, rising in %s | judge 1.5B %s, FALLING in %s | disagreement %s, rising in %s | "
            "|change 0.5B| > |change 1.5B| in %s" % (
                H, arm, msem(c["judge_small"]), cnt(sum(v > 0 for v in c["judge_small"]), n), msem(c["judge_big"]),
                cnt(sum(v < 0 for v in c["judge_big"]), n), msem(c["disagree"]), cnt(sum(v > 0 for v in c["disagree"]), n),
                cnt(small_dom, n)))
    say("%s   fig:dis (tex:261-263) attributes the widening to the 1.5B judge falling. In the E1 hacking arm the "
        "widening is carried by the 0.5B judge rising (%d/%d runs dominated by it); the 1.5B judge falls in %d/%d." % (
            H, drive["hacking"]["small_dominates"], nh, sum(v < 0 for v in drive["hacking"]["judge_big"]), nh))
    say("    This is partly by construction: in the hacking arm the 0.5B judge IS the reward (logged proxy == "
        "judge_small, max |difference| %.1e over every step of every run), so much of disagreement's rise is the "
        "reward curve going up under optimisation, not the two evaluators diverging. The benign arm is the mirror "
        "case: its reward is the 1.5B judge (proxy == judge_big to %.1e), and there disagreement falls in %s." % (
            pgap["hacking"]["judge_small"], pgap["benign"]["judge_big"],
            cnt(sum(v < 0 for v in drive["benign"]["disagree"]), nb)))

    # ------------------------------------------------------------------------------- (a) trajectories
    say("%s (a) TRAJECTORIES. Figure: %s" % (H, FIG.relative_to(P.ROOT).as_posix()))
    say("%s band = mean +/- 1.96 SEM across runs; event-aligned offsets %d..+%d drawn where >= %d runs contribute "
        "(an unbalanced panel: early offsets come only from late-event runs). The window numbers below use a balanced "
        "panel, the runs with event >= 20, so the same runs enter every window." % (C, LO, HI, MIN_N))
    hr = [r for r in he if ev[r["_tag"]] >= 20]
    traj = {"aligned_balanced": {}, "calendar": {"hacking": {}, "benign": {}}}
    say("%s   hacking, event-aligned, balanced panel (n=%d runs, event >= 20): window means [%s]" % (
        H, len(hr), ", ".join("%+d..%+d" % (a, b - 1) for a, b in WIN_EV)))
    for f, lab, *_ in SER:
        vals = [[wmean(ser(r, f), ev[r["_tag"]] + a, ev[r["_tag"]] + b) for r in hr] for a, b in WIN_EV]
        traj["aligned_balanced"][f] = [round(float(np.mean(v)), 4) for v in vals]
        d01 = [b - a for a, b in zip(vals[0], vals[1])]
        say("%s     %-27s %s | change [-20,0)->[0,20) %+.3f, up in %s" % (
            H, lab, "  ".join("%.3f" % np.mean(v) for v in vals), float(np.mean(d01)), cnt(sum(x > 0 for x in d01), len(d01))))
    for arm, runs in (("hacking", hack), ("benign", ben)):
        say("%s   %s, calendar steps (n=%d): window means [%s]" % (
            H, arm, len(runs), ", ".join("%d-%d" % (a, b - 1) for a, b in WIN_CAL)))
        for f, lab, *_ in SER:
            v = [float(np.mean([wmean(ser(r, f), a, b) for r in runs])) for a, b in WIN_CAL]
            traj["calendar"][arm][f] = [round(x, 4) for x in v]
            say("%s     %-27s %s" % (H, lab, "  ".join("%.3f" % x for x in v)))
    n_at = figure(he, [ev[r["_tag"]] for r in he], ben)
    say("%s SUMMARY: disagreement first-moves before the strict oracle in %d, with it in %d, after it in %d of %d dated "
        "hacking runs (oracle undated in %d); the reversal %s, and the opposite order is not shown either (sign test "
        "p %.2f). Disagreement widens through the 0.5B judge rising, not the 1.5B judge falling; this is partly by "
        "construction, since in the hacking arm the 0.5B judge is the reward (proxy == judge_small), so much of the "
        "rise is the reward curve rather than evaluator divergence. None of this changes the registered verdict." % (
            H, k["before"], k["with"], k["after"], k["dated"], k["n"] - k["dated"],
            "is reproduced" if rep else "is NOT reproduced", pp))
    data = {"n_hack": nh, "n_benign": nb, "n_event": ne, "ordering_primary": k, "ordering_sensitivity": sens,
            "both_move_n": len(both), "oracle_minus_disagreement_fm": dlt, "sign_test": [kp, mp, pp],
            "strict_first_move_up": ups, "reproduces_every_seed_form": rep, "reproduces_majority_form": maj,
            "judge_first_mover": {"judge_small": js_first, "judge_big": jb_first},
            "ordering_per_task": per_task, "abs_disagree_first_move_by_arm": fm_arm, "proxy_minus_judge_max_abs": pgap,
            "per_run": rows, "change_last25_minus_first25": drive, "trajectories": traj, "runs_per_offset": n_at,
            "registered_lead_restated": {"median": sc["median_lead"], "n": len(sc["leads"]), "n_pos": sc["n_lead_pos"]},
            "crosscheck": {"figures_phase2_equal": fp_ok, "registered_leads_equal": lead_ok, "disagree_gap": gap}}
    return {"id": "X6", "title": "Delta-judge disagreement trajectories and the first-move reversal", "lines": L,
            "data": data, "figures": [str(FIG)]}


if __name__ == "__main__":
    for line in run()["lines"]:
        print(line)
