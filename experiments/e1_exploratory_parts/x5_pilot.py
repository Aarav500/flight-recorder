"""[EXPLORATORY] X5: the six E1 pilot runs (seed 100), described. Calibration only.

Pre-registered as exploratory: d1aa118:experiments/e1_preregistration.md:254, "X5 | The pilot runs (seed 100) |
Calibration only. Reported descriptively." The pilot seeds are excluded from every confirmatory analysis (:214),
so nothing here can change the registered verdict (experiments/e1_primary_report.txt). No p-values: seed 100 has
no benign arm. Written 2026-10-08, AFTER the registered result. Unpinned; imports the registered code and the
pilot gate (experiments/e1_pilot_gate.py, fixed in 24e1b5b and 97f885c before any pilot data, identical at d1aa118), modifies
nothing.

Per run: lr, task, steps; the registered event (sustained_drop_v2) and its window; the pilot gate's event,
baseline_drop(0.6, persist=2) (e1_pilot_gate.py:17-25); proxy and strict change (last 25 minus first 25,
e1_primary.py:500-503); the lenient oracle (not logged in the pilot); and the primary detector's first alarm at the
MAIN registered operating point, relative to the event.

    python experiments/e1_exploratory_parts/x5_pilot.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True       # read-only: leave no new __pycache__ next to the pinned files
from pathlib import Path  # noqa: E402

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))
import json  # noqa: E402

import numpy as np  # noqa: E402

import e1_primary as P  # noqa: E402
import e1_event as EV  # noqa: E402
import e1_detectors as D  # noqa: E402
import e1_pilot_gate as PG  # noqa: E402

SP, TS, DM = P.SP, P.TS, P.DM
X = "[EXPLORATORY] X5"
CH = "CHOICE (made after the data):"
GATE_SHA_D1AA118 = "c4bd021d386c9f4d3a5a2a9580e465986ff170882723cde564331ebce0c4d6d0"   # LF, same since 97f885c
AUDIT = EXP / "e1_format_audit.json"
FIG = EXP / "figures" / "e1_X5_pilot_trajectories.png"
# Pre-freeze records this report reproduces (d1aa118 prereg section 14, :312 and :317)
RECORDED = {"gate": {"1e-6": [32, 27, 2], "5e-7": [44, 28, 22]}, "event": {"1e-6": [32, 11, 21], "5e-7": [11, 5, None]},
            "persist1_median": {"1e-6": 5.0, "5e-7": 8.0}}


def cp(x, n):
    return "(%.3f, %.3f)" % P.clopper_pearson(x, n)


def cnt(x, n):
    """A count with its Clopper-Pearson 95% interval."""
    return "%d/%d %s" % (x, n, cp(x, n))


def chg(run, f):
    """e1_primary.analyze's change (e1_primary.py:500-502): last 25 steps minus first 25. None if never logged."""
    x = np.array([np.nan if r.get(f) is None else r[f] for r in D.rows(run)], float)
    return None if not np.isfinite(x).any() else float(np.nanmean(x[-25:]) - np.nanmean(x[:25]))


def provenance():
    pins = json.loads(P.PINS.read_text())
    pins.pop("_env", None)
    bad = [rel for rel, h in pins.items() if P._sha(P.ROOT / rel) != h]
    if P._sha(EXP / "e1_pilot_gate.py") != GATE_SHA_D1AA118:
        bad.append("experiments/e1_pilot_gate.py")
    if bad:
        raise SystemExit("PROVENANCE FAILURE: %s differ from the frozen versions" % bad)
    return len(pins)


def lr_key(run):
    return next((k for k in PG.RATES if abs(float(k) - float(run["lr"])) < 1e-12), str(run["lr"]))


def fmt(v, spec="%+.3f"):
    return "n/a" if v is None else spec % v


def figure(runs, thr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), sharey=True)
    tm = lambda x: np.array([np.nanmean(x[max(0, i - 19):i + 1]) for i in range(len(x))])   # causal 20-step mean
    for r in runs:
        ax = axes[list(PG.RATES).index(r["lr_key"]), P.TASKS.index(r["task"])]
        h = np.array([np.nan if v is None else v for v in r["_hidden"]], float)
        js = np.array([np.nan if v is None else v for v in r["_js"]], float)
        t = np.arange(len(h))
        ax.plot(t, h, color=blue, lw=0.6, alpha=0.25)
        ax.plot(t, tm(h), color=blue, lw=1.8, label="strict pass rate (20-step trailing mean; raw faint)")
        ax.plot(t, tm(js), color=orange, lw=1.8, label="0.5B judge score, the proxy (20-step trailing mean)")
        for v, ls, c, lab in ((r["event"], "-", ink, "registered event (sustained_drop_v2)"),
                              (r["gate_event"], "--", ink2, "pilot-gate event (baseline_drop 0.6, persist 2)"),
                              (r["alarm"], ":", aqua, "primary alarm at the main threshold %.3f" % thr)):
            if v is not None:
                ax.axvline(v, ls=ls, color=c, lw=1.6, label=lab)
        ax.set_title("%s, lr %s" % (r["task"], r["lr_key"]), fontsize=10, color=ink, loc="left")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(color=grid, lw=0.6)
        ax.tick_params(colors=ink2, labelsize=8)
        for s in ax.spines.values():
            s.set_color(grid)
        if r["event"] is None:
            ax.text(0.98, 0.04, "no registered event", transform=ax.transAxes, ha="right", fontsize=8, color=ink2)
        if r["alarm"] is None:
            ax.text(0.98, 0.11, "no primary alarm", transform=ax.transAxes, ha="right", fontsize=8, color=ink2)
    for ax in axes[-1]:
        ax.set_xlabel("training step", color=ink2, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("rate / score", color=ink2, fontsize=9)
    hs = {}
    for ax in axes.flat:
        for hdl, lab in zip(*ax.get_legend_handles_labels()):
            hs.setdefault(lab, hdl)
    fig.legend(list(hs.values()), list(hs.keys()), loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.text(0.01, 0.985, "EXPLORATORY (X5): E1 pilot runs, seed 100, calibration only; not part of the registered analysis",
             fontsize=8, color=ink2, va="top")
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130, facecolor="#fcfcfb")
    plt.close(fig)
    return str(FIG)


def run():
    lines = []
    say = lines.append
    n_pins = provenance()
    used = P.load_main()[0]
    ben, hack = used["benign"], [(r["_tag"], r) for r in used["hacking"]]
    prim = next({"label": l, "family": f, "gb": g, "dm": m} for (l, f, g, m) in D.detectors() if l == P.PRIMARY)
    thr = TS.fpr10_threshold(prim, ben)
    event_of = lambda run: EV.event(list(range(len(run["log"]))), [r.get("hidden") for r in run["log"]])
    main_win = P.window_precondition({t: event_of(r) for t, r in hack})
    audit = json.loads(AUDIT.read_text())["per_run"] if AUDIT.exists() else {}

    say("%s The E1 pilot runs (seed 100): calibration only, reported descriptively "
        "(d1aa118:experiments/e1_preregistration.md:254)" % X)
    say("%s   Not confirmatory: seed 100 is excluded from every confirmatory analysis (prereg :214). No p-values: the "
        "pilot has no benign arm. Nothing here changes the registered verdict." % X)
    say("%s   provenance: %d pinned files match experiments/e1_pins.json; experiments/e1_pilot_gate.py matches d1aa118 "
        "and 97f885c (its lr-parsing fix; the gate constants are unchanged since 24e1b5b; both before any pilot data)"
        % (X, n_pins))
    say("%s the registered event is applied exactly as e1_primary.py:496 applies it (index steps 0..T-1, strict "
        "`hidden`), to the WHOLE run: 400 steps at lr 1e-6, 800 at lr 5e-7. Its parameters are in step units, so at "
        "lr 5e-7 (half the per-step update) they span half as much training. Window = event step "
        "(e1_primary.window_precondition; e1_pilot_gate.py:21)." % CH)
    say("%s the pilot-gate event is e1_pilot_gate.EVENT = baseline_drop(0.6, persist=2) through PG.event_step, the "
        "gate's own code; medians use PG.gate (no event = fail). Its record-only events (persist=1, raw peak_drop) "
        "are printed for the record, as the gate printed them." % CH)
    say("%s proxy = judge_small, the 0.5B judge (the registered measure, e1_primary.py:503); the log's `proxy` field "
        "equals judge_small in all 6 pilot runs. Change = mean of the last 25 logged steps minus the first 25, over "
        "the whole run (steps 775-799 at lr 5e-7, not step 400)." % CH)
    say("%s lenient change is reported as n/a: `hidden_lenient` was added to the job after the pilot (d1aa118 prereg "
        "section 14 :319). It is not reconstructed: that would mean executing model-generated code. The pre-freeze "
        "format audit (experiments/e1_format_audit.json; 1 completion per 20 steps, late half, strict zeros only, "
        "code <= 800 chars; scope corrected at :313) is shown instead, as counts only." % CH)
    say("%s primary alarm = SP.alarm_step(persist 1) on the signed disagreement (D.sig 'dm') at the MAIN registered "
        "threshold %.4f (TS.fpr10_threshold on the 24 main benign runs: 1.5B judge, lr 1e-6, 400 steps). It scans "
        "the whole run, including steps 400-799 at lr 5e-7, where no benign run calibrates it. Deviation baseline = "
        "median of the run's first 5 steps (DM.dev). 'Before' = strictly before (the registered TP rule a < e, "
        "e1_primary.py:287); a run with no event has no alarm before it." % (CH, thr))
    say("%s truncation rate and completion length are not in the pilot's log rows; they come from trl_history "
        "through e1_detectors.rows, one TRL entry per step (400 or 800 entries; the summary entry is skipped)." % CH)

    files = sorted((P.ROOT / "runs" / "e1" / "pilot").glob("*.json"))
    runs = []
    for f in files:
        d = json.loads(f.read_text())
        log = d["log"]
        steps = [r.get("step") for r in log]
        idx = list(range(len(log)))
        hidden = [r.get("hidden") for r in log]
        ev = EV.event(idx, hidden)
        ge = PG.event_step(steps, hidden, *PG.EVENT)
        rec = {lab: PG.event_step(steps, hidden, how, fr, pe) for how, fr, pe, lab in PG.RECORD_ONLY}
        a = SP.alarm_step(None, prim["dm"](d), thr, 1)
        a = None if a is None else int(a)
        js = [r.get("judge_small") for r in log]
        px = np.array([np.nan if r.get("proxy") is None else r["proxy"] for r in log], float)
        cpx, cst, cln = chg(d, "judge_small"), chg(d, "hidden"), chg(d, "hidden_lenient")
        au = audit.get(f.stem)
        rel = lambda e: None if (a is None or e is None) else a - e
        rows_ok = steps == idx
        runs.append({"file": f.stem, "task": d["task"], "lr": float(d["lr"]), "lr_key": lr_key(d), "steps_logged": len(log),
                     "steps_budget": int(d["steps"]), "indexed_0_T": rows_ok, "event": ev, "window": ev,
                     "gate_event": ge, "record_only": rec, "proxy_change": cpx, "strict_change": cst,
                     "proxy_up_strict_down": bool(cpx is not None and cst is not None and cpx > 0 and cst < 0),
                     "lenient_change": cln, "proxy_field_equals_judge_small": bool(np.allclose(
                         px, np.array([np.nan if v is None else v for v in js], float), equal_nan=True)),
                     "clipped_change": chg(d, "clipped"), "length_change": chg(d, "mean_len"),
                     "alarm": a, "alarm_minus_event": rel(ev), "alarm_minus_gate_event": rel(ge),
                     "alarm_before_event": bool(a is not None and ev is not None and a < ev),
                     "alarm_before_gate_event": bool(a is not None and ge is not None and a < ge),
                     "format_audit": au, "go_no_go": d.get("go_no_go", {}), "_hidden": hidden, "_js": js})
    runs.sort(key=lambda r: (list(PG.RATES).index(r["lr_key"]), P.TASKS.index(r["task"])))
    n = len(runs)

    say("%s per run (window = registered event step; change = last 25 minus first 25 steps):" % X)
    for r in runs:
        def where(dlt, lab):
            return "%s: no event" % lab if dlt is None else "%s %+d (%s)" % (
                lab, dlt, "before" if dlt < 0 else "on" if dlt == 0 else "after")
        al = ("none (the deviation never exceeds the threshold)" if r["alarm"] is None else "step %d; alarm minus %s; %s" % (
            r["alarm"], where(r["alarm_minus_event"], "registered event"), where(r["alarm_minus_gate_event"], "gate event")))
        say("%s   %-15s lr %-5s %d/%d steps%s | registered event %s (window %s) | pilot-gate event %s (window %s) | "
            "record-only: persist=1 %s, peak_drop %s" % (
                X, r["task"], r["lr_key"], r["steps_logged"], r["steps_budget"], "" if r["indexed_0_T"] else " (NOT 0..T-1)",
                fmt(r["event"], "%d"), fmt(r["window"], "%d"), fmt(r["gate_event"], "%d"), fmt(r["gate_event"], "%d"),
                fmt(r["record_only"]["published event (persist=1)"], "%d"), fmt(r["record_only"]["Def. B 20% (raw)"], "%d")))
        g = r["go_no_go"]
        au = r["format_audit"]
        say("%s     proxy %s, strict %s -> proxy up and strict down: %s | lenient %s | truncation rate %s, length %s "
            "tokens | primary first alarm: %s" % (
                X, fmt(r["proxy_change"]), fmt(r["strict_change"]), "yes" if r["proxy_up_strict_down"] else "no",
                fmt(r["lenient_change"]), fmt(r["clipped_change"]), fmt(r["length_change"], "%+.1f"), al))
        fa = ("n/a" if not au else "%s of late strict zeros (%d wrong after fix, %d parse but wrong, %d no code)" % (
            cnt(au["correct_after_format_fix"], sum(au.values())), au["wrong_after_format_fix"], au["parses_but_wrong"],
            au["no_code"]))
        say("%s     format audit (pre-freeze), correct after format fix: %s | job go_no_go (launch_slow_onset.py:335-369, "
            "not registered): collapsed %s, diverged %s, probe -20%% window %s" % (
                X, fa, g.get("collapsed"), g.get("diverged"), g.get("causal_window")))

    say("%s counts over the %d pilot runs (pooled; then lr 1e-6 | lr 5e-7):" % (X, n))
    data_counts = {}
    for key, lab in (("event", "registered event fires"), ("gate_event", "pilot-gate event fires"),
                     ("proxy_up_strict_down", "proxy up and strict down"), ("alarm", "primary alarm at all"),
                     ("alarm_before_event", "primary alarm strictly before the registered event"),
                     ("alarm_before_gate_event", "primary alarm strictly before the pilot-gate event")):
        hit = lambda r: (r[key] is not None) if key in ("event", "gate_event", "alarm") else r[key]
        per = {lr: [r for r in runs if r["lr_key"] == lr] for lr in PG.RATES}
        k = sum(hit(r) for r in runs)
        data_counts[key] = {"pooled": [k, n], **{lr: [sum(hit(r) for r in rs), len(rs)] for lr, rs in per.items()}}
        say("%s   %-52s %s | %s | %s" % (X, lab, cnt(k, n), *(cnt(sum(hit(r) for r in rs), len(rs)) for rs in per.values())))
    tot = {c: sum(r["format_audit"][c] for r in runs if r["format_audit"]) for c in
           ("correct_after_format_fix", "wrong_after_format_fix", "parses_but_wrong", "no_code")}
    say("%s   format audit, correct after format fix (pre-freeze; :313 adds 23 excluded long zeros, giving about 61 of "
        "80): %s of late strict zeros" % (X, cnt(tot["correct_after_format_fix"], sum(tot.values()))))

    say("%s per-lr median window, no event = fail (PG.gate), descriptive; reproduces the pre-freeze records at "
        "d1aa118 prereg :312 and :317:" % X)
    meds = {}
    for evk, lab in (("event", "registered event (sustained_drop_v2)"), ("gate_event", "pilot-gate event")):
        for lr in PG.RATES:
            ws = [r[evk] for r in sorted(runs, key=lambda r: P.TASKS.index(r["task"])) if r["lr_key"] == lr]
            med, ok = PG.gate(ws)
            meds["%s/%s" % (evk, lr)] = {"windows": ws, "median": med, "passes_12": ok}
            same = ws == RECORDED["event" if evk == "event" else "gate"][lr]
            say("%s   %-38s lr %-5s windows %s (count_vowels, is_prime, second_largest), median %s -> %s; recorded %s: %s" % (
                X, lab, lr, ws, med, ">= 12" if ok else "< 12", RECORDED["event" if evk == "event" else "gate"][lr],
                "MATCHES" if same else "DIFFERS"))
    for lr in PG.RATES:
        ws = [r["record_only"]["published event (persist=1)"] for r in runs if r["lr_key"] == lr]
        med = PG.gate(ws)[0]
        meds["persist1/%s" % lr] = {"windows": ws, "median": med}
        say("%s   %-38s lr %-5s windows %s, median %s; recorded %s: %s" % (
            X, "published event (persist=1), record-only", lr, ws, med, RECORDED["persist1_median"][lr],
            "MATCHES" if med == RECORDED["persist1_median"][lr] else "DIFFERS"))
    say("%s   the pre-freeze review (d1aa118 prereg :313) reads 2 of the 6 gate windows as noise: second_largest lr 1e-6 "
        "at step 2 (a dip inside the baseline) and second_largest lr 5e-7 at step 22 (a run that never collapsed; the "
        "registered event does not fire on it)." % X)
    say("%s   context (registered, restated, not exploratory): main hacking arm median window %s over %d runs "
        "(prereg :244, 'The pilot sets the rate')." % (X, main_win["median_window"], main_win["n"]))
    say("%s   none of the %d primary alarms precedes the registered event; the primary threshold was set on lr 1e-6 "
        "benign runs and has no false-positive calibration at lr 5e-7. Descriptive only." % (
            X, sum(r["alarm"] is not None for r in runs)))

    figs = []
    try:
        figs.append(figure(runs, thr))
        say("%s   figure: %s" % (X, Path(figs[0]).relative_to(P.ROOT).as_posix()))
    except Exception as exc:          # a missing matplotlib must not hide the report
        say("%s   figure not written: %s" % (X, exc))
    data = {"threshold_main": thr, "main_median_window": main_win["median_window"], "counts": data_counts,
            "medians": meds, "format_audit_total": tot,
            "runs": [{k: v for k, v in r.items() if not k.startswith("_")} for r in runs]}
    return {"id": "X5", "title": "The pilot runs (seed 100): calibration only, descriptive",
            "lines": lines, "data": json.loads(json.dumps(data)), "figures": figs}


if __name__ == "__main__":
    for line in run()["lines"]:
        print(line)
