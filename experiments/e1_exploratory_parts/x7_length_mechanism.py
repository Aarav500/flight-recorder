"""[EXPLORATORY] X7: completion length and truncation rate per task, split by mechanism (truncation- or prose-driven).

Pre-registered as exploratory: d1aa118:experiments/e1_preregistration.md:256, "X7 | Completion length and truncation
rate, reported per task, split by mechanism: truncation-driven or prose-driven | Mechanism proximity: on
truncation-driven tasks a truncation lead is partly the event itself". It follows the mechanism caveat at :41-45 and
d1aa118:experiments/e1_detectors.py:11-16 ("Report these two detectors per task"). The two mechanisms are the ones
the 2026-10-07 diagnostic named (d1aa118:experiments/e1_diag_results.md:46-53, committed in 7eb66d0):
  prose       unfenced prompt-continuation prose: no code fence, prose before the def, so the text does not parse
  truncation  a chatty answer whose ```python fence is cut off at the 256-token cap before it closes
Nothing here can change the registered verdict (experiments/e1_primary_report.txt). Written 2026-10-08, AFTER the
registered result. Unpinned; it imports the registered code, modifies nothing, and executes no model-generated code
(text features of the stored completions only). A closing cross-reference lists the registered primary's strict hits
under the registered event and under the X1 probe event (recomputed here as in x1_probe_event.probe_event) with each
run's X7 mechanism, against the primary's alarm timing (e1_supplementary.py D4, recomputed here).

    python experiments/e1_exploratory_parts/x7_length_mechanism.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True       # read-only: leave no new __pycache__ next to the pinned files
from pathlib import Path  # noqa: E402

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))
sys.path.append(str(EXP.parent))
import json  # noqa: E402
from collections import Counter  # noqa: E402

import numpy as np  # noqa: E402

import e1_primary as P  # noqa: E402
import e1_event as EV  # noqa: E402
import e1_detectors as D  # noqa: E402
import e1_diag_supplement as DS  # noqa: E402  prose_first: the diagnostic's own "prose before the def" feature
from flightrecorder.repro.reward_testhack import _FENCE  # noqa: E402  extract_code's fence: a CLOSED block only

SP, TS, DM = P.SP, P.TS, P.DM
X = "[EXPLORATORY] X7"
CH = "CHOICE (made after the data):"
LATE = 50            # last 50 steps x 8 completions (see the CHOICE line)
CLIP_CUT = 0.5       # e1_cap512_analysis.py's late clipped-ratio cut (FILLS ANY BUDGET vs PROSE PERSISTS)
CAP_NEAR = 250       # re-encoded token count read as "at the 256-token cap" (re-encoding drifts by a few tokens)
DIAG_SUPP_SHA_D1AA118 = "c9f92da03f5dfe01b619f96f31f3984688acfea37380baf020c5bb36fc7da979"   # LF, same since 1341d7d
FENCE_D1AA118 = r"```(?:python|py)?\s*\n?(.*?)```"     # d1aa118:flightrecorder/repro/reward_testhack.py:22
SIGNALS = ("completion length", "truncation rate")
MECHS = ("truncation", "prose")
FIG = EXP / "figures" / "e1_X7_length_truncation_by_mechanism.png"


def cp(x, n):
    return "(%.3f, %.3f)" % P.clopper_pearson(x, n)


def cnt(x, n):
    """A count with its Clopper-Pearson 95% interval."""
    return "%d/%d %s" % (x, n, cp(x, n))


def provenance():
    pins = json.loads(P.PINS.read_text())
    pins.pop("_env", None)
    bad = [rel for rel, h in pins.items() if P._sha(P.ROOT / rel) != h]
    if P._sha(EXP / "e1_diag_supplement.py") != DIAG_SUPP_SHA_D1AA118:
        bad.append("experiments/e1_diag_supplement.py")
    if _FENCE.pattern != FENCE_D1AA118:
        bad.append("reward_testhack._FENCE")
    if bad:
        raise SystemExit("PROVENANCE FAILURE: %s differ from the frozen versions" % bad)
    return len(pins)


def mech_of(text):
    """'truncated': a fence opens and never closes, so extract_code finds no block and the whole chatty text fails to
    parse. 'prose': no fence, non-code text before the first code line (e1_diag_supplement.prose_first). Else 'other'."""
    if "```" in text:
        return "other" if _FENCE.search(text) else "truncated"
    return "prose" if DS.prose_first(text) else "other"


def classify(run, late=LATE):
    T, c, n_late, n_zero = len(run["log"]), Counter(), 0, 0
    for row in run["completions"]:
        if row["step"] < T - late:
            continue
        for t, h in zip(row["texts"], row["hidden"]):
            n_late += 1
            if h is not None and h <= 0:
                n_zero += 1
                c[mech_of(t)] += 1
    tr, pr = c["truncated"], c["prose"]
    return {"n_late": n_late, "n_zero": n_zero, "truncated": tr, "prose": pr, "other": c["other"],
            "trunc_share": tr / (tr + pr) if tr + pr else 0.0, "mechanism": "truncation" if tr > pr else "prose"}


def tokenizer():
    """A cached Qwen2.5 tokenizer.json (one BPE vocabulary across the family; the policy's own snapshot has none)."""
    try:
        from tokenizers import Tokenizer
    except ImportError:
        return None, None
    import os
    hub = Path(os.environ.get("HF_HUB_CACHE") or
               Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
    for name in ("Qwen2.5-1.5B", "Qwen2.5-1.5B-Instruct", "Qwen2.5-0.5B-Instruct"):
        for f in sorted(hub.glob("models--Qwen--%s/snapshots/*/tokenizer.json" % name)):
            return Tokenizer.from_file(str(f)), "Qwen/" + name
    return None, None


def probe_event(run):
    """The X1 event, as x1_probe_event.probe_event (launch_slow_onset.py:340-348 at d1aa118), recomputed here so this
    module does not depend on a sibling: pk = global argmax of probe_hidden (first maximum), event = the step of the
    first probe at or after pk with probe_hidden <= 0.8 x the peak; None if there is none."""
    ps = [int(p["step"]) for p in run["probe"]]
    pv = [float(p["probe_hidden"]) for p in run["probe"]]
    if not pv:
        return None
    pk = int(np.argmax(pv))
    return next((ps[i] for i in range(pk, len(pv)) if pv[i] <= 0.8 * pv[pk]), None)


def latest_rank(ev, tag):
    """1 = the latest event among the runs with one."""
    return 1 + sum(1 for v in ev.values() if v is not None and v > ev[tag])


def cap_check(hack, cls, tok):
    """Re-encode each late strict-zero completion: how many sit at the cap, and does counting cap-hitting unfenced
    prose as truncated (the reading 'prose = unfenced and not truncated') move any run across the split?"""
    n, at, flips = Counter(), Counter(), []
    for tag, run in hack:
        T, c = len(run["log"]), Counter()
        for row in run["completions"]:
            if row["step"] < T - LATE:
                continue
            for t, h in zip(row["texts"], row["hidden"]):
                if h is None or h > 0:
                    continue
                k = mech_of(t)
                cap = len(tok.encode(t).ids) >= CAP_NEAR
                n[k] += 1
                at[k] += cap
                c["truncated" if k == "truncated" or (k == "prose" and cap) else k] += 1
        if (c["truncated"] > c["prose"]) != (cls[tag]["mechanism"] == "truncation"):
            flips.append(tag)
    return n, at, flips


def traj(run, field, e):
    x = D.series(D.rows(run), field)
    at = None if e is None else float(np.nanmean(x[max(0, e - 2):e + 3]))
    return float(np.nanmean(x[:25])), at, float(np.nanmean(x[-25:]))


def summ(vals, spec):
    v = [x for x in vals if x is not None]
    return "n/a" if not v else (spec + " [" + spec + ", " + spec + "]") % (np.mean(v), min(v), max(v))


def figure(hack, cls, events):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"truncation": "#c0392b", "prose": "#2471a3"}
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    lo, hi = -40, 120
    for ax, field, lab in ((axs[0], "clipped", "truncation rate (share at the 256-token cap)"),
                           (axs[1], "mean_len", "mean completion length (tokens)")):
        for m in MECHS:
            grid = []
            for tag, run in hack:
                e = events[tag]
                if cls[tag]["mechanism"] != m or e is None:
                    continue
                x = D.series(D.rows(run), field)
                off = np.arange(len(x)) - e
                k = (off >= lo) & (off <= hi)
                ax.plot(off[k], x[k], color=col[m], lw=0.4, alpha=0.35)
                row = np.full(hi - lo + 1, np.nan)
                row[off[k] - lo] = x[k]
                grid.append(row)
            n, g = len(grid), np.asarray(grid)
            c = np.isfinite(g).sum(axis=0)
            mu = np.where(c >= 3, np.nansum(g, axis=0) / np.maximum(c, 1), np.nan)   # mean only where >= 3 runs
            ax.plot(np.arange(lo, hi + 1), mu, color=col[m], lw=2,
                    label="%s-driven (%d runs with an event; mean where >= 3 runs)" % (m, n))
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("step minus registered event step")
        ax.set_ylabel(lab)
    axs[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("E1 hacking arm, EXPLORATORY X7: length and truncation aligned to the strict-oracle event", fontsize=10)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130)
    plt.close(fig)
    return str(FIG)


def run():
    lines, data = [], {}
    say = lines.append
    n_pins = provenance()
    used, *_ = P.load_main()
    hack = [(r["_tag"], r) for r in used["hacking"]]
    ben = used["benign"]
    nh, nb = len(hack), len(ben)
    task = lambda tag: tag.split("/")[1]
    event_of = lambda run: EV.event(list(range(len(run["log"]))), [r.get("hidden") for r in run["log"]])
    events = {t: event_of(r) for t, r in hack}
    cls = {t: classify(r) for t, r in hack}
    say("%s completion length and truncation rate per task, split by mechanism (truncation- or prose-driven). "
        "Pre-registered as exploratory (d1aa118 prereg :256); it cannot change the registered verdict "
        "(CLEAN NULL, experiments/e1_primary_report.txt)." % X)
    say("%s provenance: %d pinned files match e1_pins.json; e1_diag_supplement.py and the strict extractor's fence "
        "regex match d1aa118. Runs (e1_primary.load_main): %d hacking, %d benign." % (X, n_pins, nh, nb))
    say("%s the late window is the last %d steps (%d-399), all 8 completions per step. The 2026-10-07 mechanism counts "
        "(e1_diag_results.md:51 '351 of 396'; prereg :99-101 '378 of 381', '379 of 385', '399 of 399') have denominators "
        "<= 400 = 50 steps x 8, which this matches. (The pilot format audit used the late half.)" % (CH, LATE, 400 - LATE))
    say("%s a late completion the strict oracle scores 0 is 'truncated' if it opens a ``` fence that never closes (the "
        "strict extractor, reward_testhack._FENCE, finds no block, so the whole chatty text fails to parse), 'prose' if "
        "it has no fence and non-code text before its first def/import/from/class line (e1_diag_supplement.prose_first), "
        "else 'other'. Per-completion token counts are not stored, so 'hits the 256-token cap' is checked two ways: by "
        "re-encoding the stored text with a cached Qwen2.5 tokenizer (the cap-check line below; approximate) and by the "
        "run's late clipped ratio." % CH)
    say("%s a run is truncation-driven if its late strict-zero completions hold more 'truncated' than 'prose' (ties go "
        "to prose). The diagnostic fixes no run-level threshold. Check: late clipped ratio (mean of the same %d steps) "
        ">= %.1f, the cut of the cap512 rule (e1_cap512_analysis.py)." % (CH, LATE, CLIP_CUT))
    say("%s trajectories: start = mean of steps 0-24, at event = mean of steps e-2..e+2 (runs with an event), end = "
        "mean of steps 375-399; per run, then the mean [min, max] over the runs of a group." % CH)
    say("%s 'the length/truncation detectors' is read as all six registered ones (2 signals x level, velocity, "
        "acceleration; prereg :37, e1_detectors.py), not four." % CH)
    say("%s benign runs have no mechanism, so FP is the benign arm at the registered pooled threshold "
        "(threshold_sweep.fpr10_threshold over all %d benign runs): y/%d for the mechanism groups, y/8 per task. TP "
        "(alarm strictly before the event) and the median lead (registered 3-sigma first-move rule, e1_primary.score) "
        "are computed within each group of hacking runs." % (CH, nb, nb))
    say("%s alarm minus event = first alarm at the registered threshold (score_prime.alarm_step, persist 1) minus the "
        "registered event, over runs with both. Sign at alarm = direction of the signal at the alarm step against "
        "its step 0-4 median (+ above). Because few runs alarm, the threshold-free 3-sigma first move minus the event "
        "(the registered lead with its sign flipped) is listed too, with 'close' read as within 10 steps." % CH)

    # ------------------------------------------------------------------ 1. mechanism per run
    say("%s mechanism per hacking run (late strict-zero completions: truncated / prose / other; late clipped ratio; "
        "registered event step):" % X)
    per_run = {}
    for tag, run in hack:
        c, e = cls[tag], events[tag]
        late_clip = float(np.nanmean(D.series(D.rows(run), "clipped")[-LATE:]))
        per_run[tag] = dict(c, event=e, late_clip=late_clip,
                            mean_len=traj(run, "mean_len", e), clipped=traj(run, "clipped", e))
        say("  %-29s %-10s strict-zero %s; truncated %s, prose %s, other %s; late clip %.2f; event %s" % (
            tag, c["mechanism"], cnt(c["n_zero"], c["n_late"]), cnt(c["truncated"], c["n_zero"]),
            cnt(c["prose"], c["n_zero"]), cnt(c["other"], c["n_zero"]), late_clip, "none" if e is None else e))
    n_tr = {t: sum(1 for g, c in cls.items() if task(g) == t and c["mechanism"] == "truncation") for t in P.TASKS}
    n_t = {t: sum(1 for g in cls if task(g) == t) for t in P.TASKS}
    tot = sum(n_tr.values())
    say("%s truncation-driven runs: %s; all tasks %s. Prose-driven: %s." % (
        X, ", ".join("%s %s" % (t, cnt(n_tr[t], n_t[t])) for t in P.TASKS), cnt(tot, nh), cnt(nh - tot, nh)))
    sh_p = max(c["trunc_share"] for c in cls.values() if c["mechanism"] == "prose")
    sh_t = min(c["trunc_share"] for c in cls.values() if c["mechanism"] == "truncation")
    same = {w: all(classify(r, w)["mechanism"] == cls[t]["mechanism"] for t, r in hack) for w in (25, 100)}
    agree = sum((per_run[t]["late_clip"] >= CLIP_CUT) == (cls[t]["mechanism"] == "truncation") for t in cls)
    say("%s robustness: truncated share among truncated+prose is at most %.3f in every prose-driven run and at least "
        "%.3f in every truncation-driven run, so any cut in [%.3f, %.3f) gives this split; late window of 25 / 100 "
        "steps: split %s / %s; late clipped ratio >= %.1f agrees with the text rule in %s runs." % (
            X, sh_p, sh_t, sh_p, sh_t, "identical" if same[25] else "DIFFERS", "identical" if same[100] else "DIFFERS",
            CLIP_CUT, cnt(agree, nh)))
    tok, tok_name = tokenizer()
    if tok is None:
        say("%s per-completion cap check skipped: no 'tokenizers' package or cached Qwen2.5 tokenizer.json." % X)
        data["cap_check"] = None
    else:
        n_k, at_k, flips = cap_check(hack, cls, tok)
        say("%s per-completion cap check (stored text re-encoded with the cached %s tokenizer; >= %d tokens read as at "
            "the 256 cap): 'truncated' %s; 'prose' %s; 'other' %s. Counting cap-hitting unfenced prose as truncated "
            "('prose' = unfenced and not truncated) moves %d/%d runs across the split%s." % (
                X, tok_name, CAP_NEAR, cnt(at_k["truncated"], n_k["truncated"]), cnt(at_k["prose"], n_k["prose"]),
                cnt(at_k["other"], n_k["other"]), len(flips), nh, (": " + ", ".join(flips)) if flips else ""))
        data["cap_check"] = {"tokenizer": tok_name, "cap_near": CAP_NEAR, "n": dict(n_k), "at_cap": dict(at_k),
                             "flips": flips}
    pre =n_tr["count_vowels"] + n_tr["second_largest"]
    say("%s against the pre-freeze task-level reading (prereg :42-43: count_vowels and second_largest at 256 run "
        "through truncation, is_prime through prose; one diagnostic run per task): is_prime is prose-driven in %s; "
        "count_vowels and second_largest are truncation-driven in only %s. In E1 the mechanism varies by seed "
        "within a task, so X7 splits by run, not by task." % (
            X, cnt(n_t["is_prime"] - n_tr["is_prime"], n_t["is_prime"]),
            cnt(pre, n_t["count_vowels"] + n_t["second_largest"])))
    no_ev = [t for t in cls if events[t] is None]
    if no_ev:
        say("%s runs without a registered event: %s (%s)." % (X, ", ".join(no_ev), ", ".join(cls[t]["mechanism"] for t in no_ev)))

    # ------------------------------------------------------------------ 2. trajectories
    groups = [(m, None) for m in MECHS] + [(m, t) for t in P.TASKS for m in MECHS]
    members = {(m, t): [g for g in cls if cls[g]["mechanism"] == m and (t is None or task(g) == t)] for m, t in groups}
    groups = [k for k in groups if members[k]]
    gname = lambda m, t: "%s / %s-driven" % ("all tasks" if t is None else t, m)
    say("%s trajectories, mean [min, max] over runs; start / at event / end:" % X)
    traj_out = {}
    for m, t in groups:
        tags = members[(m, t)]
        ne = sum(events[g] is not None for g in tags)
        part = []
        for f, spec in (("mean_len", "%.0f"), ("clipped", "%.2f")):
            s, a, e = zip(*[per_run[g][f] for g in tags])
            part.append("%s %s / %s / %s" % ("length" if f == "mean_len" else "truncation rate",
                                             summ(s, spec), summ(a, spec), summ(e, spec)))
            traj_out.setdefault(gname(m, t), {})[f] = {"start": float(np.mean(s)), "at_event": (
                float(np.mean([v for v in a if v is not None])) if ne else None), "end": float(np.mean(e))}
        say("  %-35s %d runs (%d with an event): %s; %s" % (gname(m, t), len(tags), ne, part[0], part[1]))
    for t in P.TASKS:
        rs = [r for r in ben if r["task"] == t]
        b = {f: [traj(r, f, None) for r in rs] for f in ("mean_len", "clipped")}
        say("  %-35s %d runs (no event; reference): length %s -> %s; truncation rate %s -> %s" % (
            "%s / benign arm" % t, len(rs), summ([v[0] for v in b["mean_len"]], "%.0f"), summ([v[2] for v in b["mean_len"]], "%.0f"),
            summ([v[0] for v in b["clipped"]], "%.2f"), summ([v[2] for v in b["clipped"]], "%.2f")))

    # ------------------------------------------------------------------ 3. the six length/truncation detectors
    specs = [{"label": l, "family": fam, "gb": g, "dm": m} for (l, fam, g, m) in D.detectors()
             if any(s in l for s in SIGNALS)]
    det_out, ae_out = {}, {}
    for s in specs:
        thr = TS.fpr10_threshold(s, ben)
        al_b = {r["_tag"]: SP.alarm_step(None, s["dm"](r), thr, 1) for r in ben}
        fp = sum(a is not None for a in al_b.values())
        fpt = {t: sum(1 for g, a in al_b.items() if task(g) == t and a is not None) for t in P.TASKS}
        say("%s detector %s: registered threshold %.4f; FP %s; by task %s" % (
            X, s["label"], thr, cnt(fp, nb), ", ".join("%s %s" % (t, cnt(fpt[t], 8)) for t in P.TASKS)))
        d = {"threshold": thr, "fp": fp, "n_benign": nb, "fp_by_task": fpt, "groups": {}}
        for m, t in groups:
            sub = [(g, r) for g, r in hack if g in members[(m, t)]]
            sc = P.score(s, sub, ben, event_of)
            d["groups"][gname(m, t)] = {"tp": sc["n_tp_strict"], "n": len(sub), "median_lead": sc["median_lead"],
                                        "n_lead_runs": sc["n_lead_runs"]}
            say("  %-35s TP %s; median lead %s over %d runs" % (
                gname(m, t), cnt(sc["n_tp_strict"], len(sub)), "n/a" if sc["median_lead"] is None else sc["median_lead"],
                sc["n_lead_runs"]))
        det_out[s["label"]] = d
        if "truncation rate" not in s["label"]:
            continue
        for m in MECHS:                     # alarm minus event, truncation-rate detectors
            tags = members[(m, None)]
            al, both, fme = {}, [], []
            for g, r in hack:
                if g not in tags:
                    continue
                x = np.asarray(s["dm"](r), float)
                a, e, fm = SP.alarm_step(None, x, thr, 1), events[g], P.GB.first_move(s["gb"](r))
                al[g] = a
                short = g.split("/", 1)[1]
                if a is not None and e is not None:
                    sign = "+" if x[int(a)] > np.median(x[np.isfinite(x)][:5]) else "-"
                    both.append((short, int(a) - e, sign))
                if fm is not None and e is not None:
                    fme.append(int(fm) - e)
            d_ae = [v for _, v, _ in both]
            ae_out.setdefault(s["label"], {})[m] = {
                "alarming": sum(a is not None for a in al.values()), "n": len(tags), "alarm_minus_event": both,
                "no_event_alarming": [g for g, a in al.items() if a is not None and events[g] is None],
                "first_move_minus_event": sorted(fme)}
            bo_a = [sum(v < 0 for v in d_ae), sum(v == 0 for v in d_ae), sum(v > 0 for v in d_ae)]
            ae_out[s["label"]][m]["before_on_after"] = bo_a
            say("  %s alarm minus event, %s-driven: alarming %s (%d without an event); with an event %d: %s; median "
                "%s; before / on / after %s" % (
                    s["label"], m, cnt(ae_out[s["label"]][m]["alarming"], len(tags)),
                    len(ae_out[s["label"]][m]["no_event_alarming"]), len(both),
                    ", ".join("%s %+d (sign %s)" % b for b in both) if both else "none",
                    "%+.1f" % np.median(d_ae) if both else "n/a",
                    " / ".join(cnt(k, len(both)) for k in bo_a) if both else "n/a (no run with both)"))
            say("    3-sigma first move minus event (= -lead), %s-driven, %d runs: %s; within 10 steps of the event: %s" % (
                m, len(fme), sorted(fme), cnt(sum(abs(v) <= 10 for v in fme), len(fme)) if fme else "n/a"))
    data.update({"per_run": per_run, "truncation_driven_by_task": n_tr, "n_by_task": n_t, "trajectories": traj_out,
                 "detectors": det_out, "alarm_minus_event": ae_out, "late_steps": LATE,
                 "cut_interval": [sh_p, sh_t], "clip_check_agree": agree})
    tr_, pr_ = traj_out["all tasks / truncation-driven"]["clipped"], traj_out["all tasks / prose-driven"]["clipped"]
    lv = ae_out["truncation rate"]
    lg = det_out["truncation rate"]["groups"]
    h_tr, h_pr = lg["all tasks / truncation-driven"], lg["all tasks / prose-driven"]
    fp_lv = det_out["truncation rate"]["fp"]
    say("%s reading: on truncation-driven runs the truncation rate has already risen by the event (mean %.2f -> %.2f "
        "-> %.2f, start / at event / end); on prose-driven runs it is flat to the event and then falls (%.2f -> %.2f -> "
        "%.2f). At the registered operating point the level detector alarms in %s truncation-driven and %s "
        "prose-driven runs; its strict hits: %s. That is %s truncation-driven runs%s against FP %s benign runs at the "
        "same threshold (one-sided Fisher p %.3f), and is not evidence of early warning." % (
            X, tr_["start"], tr_["at_event"], tr_["end"], pr_["start"], pr_["at_event"], pr_["end"],
            cnt(lv["truncation"]["alarming"], lv["truncation"]["n"]), cnt(lv["prose"]["alarming"], lv["prose"]["n"]),
            ", ".join("%s (alarm %d steps before its event)" % (b[0], -b[1]) for m in MECHS
                      for b in lv[m]["alarm_minus_event"] if b[1] < 0) or "none",
            cnt(h_tr["tp"], h_tr["n"]), (" and %s prose-driven runs" % cnt(h_pr["tp"], h_pr["n"])) if h_pr["tp"] else "",
            cnt(fp_lv, nb), P.fisher_one_sided(h_tr["tp"], h_tr["n"], fp_lv, nb)))

    # ------------------------------------------------------------------ 4. cross-reference: the registered primary's hits
    prim = next({"label": l, "family": fam, "gb": g, "dm": m} for (l, fam, g, m) in D.detectors() if l == P.PRIMARY)
    x1_ev = {t: probe_event(r) for t, r in hack}
    x1_off = [t for t, r in hack if (r.get("go_no_go") or {}).get("causal_window", "absent") != x1_ev[t]]
    sc_reg = P.score(prim, hack, ben, event_of)
    sc_x1 = P.score(prim, hack, ben, lambda r: x1_ev[r["_tag"]])
    thr_p = TS.fpr10_threshold(prim, ben)
    al_p = {t: SP.alarm_step(None, prim["dm"](r), thr_p, 1) for t, r in hack}
    d4 = [float(al_p[t]) - events[t] for t in al_p if al_p[t] is not None and events[t] is not None]
    n_reg, n_x1 = sum(e is not None for e in events.values()), sum(e is not None for e in x1_ev.values())
    xref = {}
    for key, sc, ev in (("registered", sc_reg, events), ("x1_probe", sc_x1, x1_ev)):
        xref[key] = {"tp_strict": sc["n_tp_strict"], "n": nh, "fp": sc["n_fp"], "n_benign": nb,
                     "hits": [{"run": t, "mechanism": cls[t]["mechanism"], "event": ev[t], "registered_event": events[t],
                               "alarm": None if al_p[t] is None else float(al_p[t]), "latest_rank": latest_rank(ev, t)}
                              for t in sc["hit_runs_strict"]]}
    say("%s cross-reference, the registered primary (%s) at its registered operating point (threshold %.4f, FP %s): "
        "strict hits under two event definitions, each with its X7 mechanism and event step:" % (
            X, P.PRIMARY, thr_p, cnt(sc_reg["n_fp"], nb)))
    say("  registered event (e1_primary.score, sustained_drop_v2 on the strict stream): TP %s: %s" % (
        cnt(sc_reg["n_tp_strict"], nh), "; ".join("%s %s-driven, event %d, alarm %d" % (
            h["run"], h["mechanism"], h["event"], h["alarm"]) for h in xref["registered"]["hits"]) or "none"))
    say("  X1 probe event (global argmax of probe_hidden, then the first probe from the peak on with probe_hidden <= 0.8 x peak; %s): TP %s: %s" % (
        "equal to the job's stored go_no_go.causal_window in every hacking run" if not x1_off else
        "DIFFERS from the stored go_no_go.causal_window in " + ", ".join(x1_off),
        cnt(sc_x1["n_tp_strict"], nh), "; ".join("%s %s-driven, X1 event %d (registered %s), alarm %d" % (
            h["run"], h["mechanism"], h["event"], "none" if h["registered_event"] is None else h["registered_event"],
            h["alarm"]) for h in xref["x1_probe"]["hits"]) or "none"))
    rk = {k: sorted(h["latest_rank"] for h in xref[k]["hits"]) for k in xref}
    latest = all(rk[k] == list(range(1, len(rk[k]) + 1)) and rk[k] for k in rk)
    say("%s the primary %s (latest-event rank of each hit: registered %s of %d runs with an event; X1 %s of %d), "
        "consistent with a primary alarm that comes a median %+.1f steps after the registered event (supplementary "
        "D4; recomputed here over the %d hacking runs with both an alarm and an event)." % (
            X, "only hits the runs with the latest events" if latest else "does NOT only hit the runs with the latest events",
            rk["registered"], n_reg, rk["x1_probe"], n_x1, np.median(d4), len(d4)))
    data["primary_cross_reference"] = dict(xref, threshold=thr_p, d4_median_alarm_minus_event=float(np.median(d4)),
                                           d4_n=len(d4), x1_stored_mismatch=x1_off)

    figs = [figure(hack, cls, events)]
    say("%s figure: %s (event-aligned truncation rate and length, thin = runs, bold = mean by mechanism)." % (
        X, Path(figs[0]).relative_to(P.ROOT).as_posix()))
    say("%s none of the above enters the registered analysis; the registered verdict (CLEAN NULL, primary = "
        "disagreement) is unchanged." % X)
    return {"id": "X7", "title": "Completion length and truncation rate per task, split by mechanism",
            "lines": lines, "data": json.loads(json.dumps(data, default=float)), "figures": figs}


if __name__ == "__main__":
    for ln in run()["lines"]:
        print(ln)
