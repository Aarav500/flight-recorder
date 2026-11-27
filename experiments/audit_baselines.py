"""Audit-baseline check for the grid's headline audit number: disagreement AUROC 0.98 (paper tex:46, 90, 314,
509, 862), from paper/analysis/detection_metrics.py.

NOT pinned and NOT registered. Written 2026-10-08. It changes no pinned or frozen file. It imports the published
scorers (paper/analysis/detection_metrics, family_matrix, specificity_partial) and the E1 loaders
(experiments/e1_primary, e1_detectors, e1_supplementary) and calls their functions, so every score below goes
through the published code paths. The only new code is the bookkeeping around them and a midrank AUROC used
to flag ties.

THE QUESTION. In the grid, the hacking arm OPTIMIZES the 0.5B judge and the benign arm OPTIMIZES the 1.5B judge
(phase2_grid.json note). Disagreement is jsmall - jbig. Each arm trains on one term of that difference, and the
two arms train on opposite terms. That alone can separate the arms, whatever happens to held-out quality. If
the optimized judge's own score (each run's training reward) separates the arms as well as disagreement does,
the 0.98 tells us which judge each arm optimized. It does not tell us which runs hacked.

Sections, in this order:
  1. Reproduce the published audit table (paper/stats/detection_metrics.json) with the imported functions.
     Assert on every detector; stop on any mismatch.
  2. Grid baselines through the identical score (max |dev|), then through a signed score: last-25-step mean
     minus first-25-step mean (the natural-hack-rate measure of e1_primary.py:500-502, imported as
     e1_supplementary.chg).
  3. Families. Reproduce family_matrix.json and specificity_partial.json, then disagreement and the 0.5B-judge
     baseline for each hacking family against benign, partial-credit and verifier-coverage runs. Only the
     heuristic-proxy contrasts against partial-credit and verifier-coverage have no judge optimized in either arm.
  4. E1, 24 v 24. Every line is EXPLORATORY: E1's registered analysis has no audit test.
  5. Inventory of every logged arm that optimizes the 0.5B judge, and a plain reading.

Conventions. AUROC treats hacking as positive. "+" means hacking runs score higher (the published orientation).
"-" means the AUROC is shown for the reversed score, a direction picked after looking at the data; for those
rows, and for every signed-score row, p2 = min(1, 2p) is the honest two-sided p. Bootstrap CI (2000, seed 0)
and permutation p (10000, seed 0) are detection_metrics.auroc_ci and perm_p. detection_metrics.auroc breaks
ties by sort order. Where scores tie, the midrank AUROC is printed as well and the verdict uses it.
A verdict compares AUROCs at two decimals, as the paper states them.

    python experiments/audit_baselines.py > experiments/audit_baselines_report.txt
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True       # read-only: leave no new __pycache__ next to the pinned files
import numpy as np  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e1_primary as P  # noqa: E402  (adds paper/analysis, benchmark, anticipatory to sys.path)
import e1_detectors as D  # noqa: E402
import e1_supplementary as ES  # noqa: E402
import detection_metrics as DM  # noqa: E402
import family_matrix as FM  # noqa: E402
import specificity_partial as SPC  # noqa: E402

ROOT = P.ROOT
OUT = ROOT / "experiments" / "audit_baselines_results.json"
X = "[EXPLORATORY]"
UNPINNED = ["paper/analysis/family_matrix.py", "paper/analysis/specificity_partial.py",
            "paper/stats/family_matrix.json", "paper/stats/specificity_partial.json",
            "experiments/phase2/phase2_family.json", "experiments/phase2/phase2_partial.json",
            "experiments/phase2/phase2_gemma.json", "experiments/phase2/phase2_qwen3b.json",
            "experiments/phase2/phase2_rloo_fmt.json", "experiments/phase2/phase2_lowlr.json",
            "runs/phase1b_dataset.json", "experiments/e1_supplementary.py"]


# ------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------
def dir_sha(d):
    h = hashlib.sha256()
    for f in sorted(Path(d).glob("*.json")):
        h.update(f.name.encode()); h.update(bytes.fromhex(P._sha(f)))
    return h.hexdigest()


def maxdev(x):
    """The published per-run score: max over steps of detection_metrics.dev (None if < 6 finite values)."""
    d = DM.dev(x)
    return None if d is None else float(np.nanmax(d))


def chg(x):
    """e1_supplementary.chg (= e1_primary.py:500-502) on an array: mean of the last 25 steps minus the first 25."""
    x = np.asarray(x, float)
    return float(ES.chg({"log": [{"v": float(v) if np.isfinite(v) else None} for v in x]}, "v"))


def midrank_auroc(pos, neg):
    r = rankdata(np.concatenate([pos, neg]))
    n1, n2 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def evaluate(pos, neg):
    """AUROC [bootstrap CI] and permutation p via detection_metrics, in the direction that favors detection."""
    pos = np.asarray([v for v in pos if v is not None], float)
    neg = np.asarray([v for v in neg if v is not None], float)
    raw = midrank_auroc(pos, neg)
    s = 1.0 if raw >= 0.5 else -1.0
    a, b = s * pos, s * neg
    au, lo, hi = DM.auroc_ci(a, b)
    pv, _ = DM.perm_p(a, b)
    mid = midrank_auroc(a, b)
    return {"n_pos": len(pos), "n_neg": len(neg), "dir": "+" if s > 0 else "-", "auroc": au, "ci": [lo, hi],
            "auroc_exact": round(DM.auroc(a, b), 4), "auroc_midrank": round(mid, 4),
            "ties": bool(abs(mid - DM.auroc(a, b)) > 1e-12), "p": pv, "p2": round(min(1.0, 2 * pv), 4),
            "mean_pos": round(float(np.mean(pos)), 4), "mean_neg": round(float(np.mean(neg)), 4)}


def best(e):
    """The AUROC a verdict uses: midrank when scores tie, else detection_metrics.auroc."""
    return e["auroc_midrank"] if e["ties"] else e["auroc_exact"]


def verdict(e, ref):
    a, r = round(best(e), 2), round(ref, 2)
    if a > r:
        return "MATCHES OR EXCEEDS"
    if a == r:
        return "MATCHES"
    return "below (CI covers %.2f)" % r if e["ci"][1] >= r else "below"


def line(label, e, refs=(), tag=""):
    tie = "  ties: midrank %.2f" % e["auroc_midrank"] if e["ties"] else ""
    v = "".join("  vs %s %.2f: %s" % (nm, ref, verdict(e, ref)) for nm, ref in refs)
    return "%s  %-44s %d v %d  %s AUROC %.2f [%.2f, %.2f]  p %.4f  p2 %.4f  means %+.3f / %+.3f%s%s" % (
        tag, label, e["n_pos"], e["n_neg"], e["dir"], e["auroc"], e["ci"][0], e["ci"][1], e["p"], e["p2"],
        e["mean_pos"], e["mean_neg"], tie, v)


def runs_of(cells, tasks=DM.TASKS):
    return [cells[t][s] for t in tasks for s in cells[t]]


# ------------------------------------------------------------------------------------------
def main():
    say = print
    res = {"note": "unpinned, not registered; see the module docstring", "sections": {}}
    say("AUDIT-BASELINE CHECK. Unpinned. Not registered. Every E1 line is EXPLORATORY.")
    say("provenance: %d pinned files match experiments/e1_pins.json (e1_supplementary.provenance)" % ES.provenance())
    res["unpinned_sha256"] = {rel: P._sha(ROOT / rel) for rel in UNPINNED}
    res["unpinned_sha256"]["runs/e1/main/*.json (name+sha, sorted)"] = dir_sha(P.MAIN)
    say("unpinned inputs (SHA-256 in the results JSON): %s, runs/e1/main/*.json" % ", ".join(UNPINNED))
    say("conventions: hacking positive; '+' = hacking scores higher, '-' = reversed score (direction picked after "
        "looking); p one-sided in the shown direction, p2 = min(1, 2p); 'means' = mean score, hacking / other arm.")

    # ---------------------------------------------------------------------------------- 1
    say("\n1. REPRODUCE the published audit table (paper/stats/detection_metrics.json) with detection_metrics' "
        "own sig, dev, auroc_ci, perm_p, average_precision")
    pub = json.loads((ROOT / "paper" / "stats" / "detection_metrics.json").read_text())
    pubp = {r["detector"]: r for r in pub["primary"]}
    pubs = {r["detector"]: r for r in pub["secondary"]}
    hack = runs_of(DM.G["hacking"])
    ben = runs_of(DM.G["benign"])
    assert (len(hack), len(ben)) == (pub["n_hacking"], pub["n_benign"]) == (9, 9)
    rep, bad = {}, []
    for name in DM.DETS:
        pos = [np.nanmax(DM.dev(DM.sig(r, name))) for r in hack if DM.dev(DM.sig(r, name)) is not None]
        neg = [np.nanmax(DM.dev(DM.sig(r, name))) for r in ben if DM.dev(DM.sig(r, name)) is not None]
        au, lo, hi = DM.auroc_ci(pos, neg)
        pv, _ = DM.perm_p(pos, neg)
        ap = DM.average_precision(pos, neg)
        w, ws = pubp[name], pubs[name]
        ok = (au == w["auroc"] and [lo, hi] == w["auroc_ci"] and pv == w["perm_p"] and ap == ws["ap"])
        rep[name] = {"auroc": au, "ci": [lo, hi], "perm_p": pv, "ap": ap, "matches_published": ok,
                     "auroc_exact": round(DM.auroc(pos, neg), 4)}
        say("  %-22s AUROC %.2f [%.2f, %.2f]  p %.4f  AP %.2f   published %.2f [%.2f, %.2f]  p %.4f  AP %.2f  %s" % (
            name, au, lo, hi, pv, ap, w["auroc"], w["auroc_ci"][0], w["auroc_ci"][1], w["perm_p"], ws["ap"],
            "MATCH" if ok else "MISMATCH"))
        bad += [] if ok else [name]
    res["sections"]["1_reproduce"] = rep
    if bad:
        say("STOP: %s do not reproduce the published table. Nothing else is computed." % bad)
        OUT.write_text(json.dumps(res, indent=1))
        return 1
    A_DIS = rep["disagreement"]["auroc_exact"]
    say("  all six detectors match on AUROC, CI, permutation p and AP. Disagreement's exact AUROC is %.4f "
        "(= %d/81), which the paper rounds to %.2f. (TPR@FPR=0.1 is computed inside detection_metrics.main, "
        "which writes paper files, so it is not re-run here.)" % (A_DIS, round(A_DIS * 81), round(A_DIS, 2)))

    # ---------------------------------------------------------------------------------- 2
    def gs(f):
        return lambda r, arm: DM.series(r["rows"], f)
    GRID = [("disagreement (published: signed jsmall-jbig)", lambda r, arm: DM.sig(r, "disagreement")),
            ("(a) 0.5B judge score alone (jsmall)", gs("jsmall")),
            ("(b) 1.5B judge score alone (jbig)", gs("jbig")),
            ("(c) optimized judge = training reward", lambda r, arm: DM.series(r["rows"], "jsmall" if arm == "h" else "jbig")),
            ("(d) strict held-out oracle (hidden)", gs("hidden")),
            ("(e) |disagreement|", lambda r, arm: np.abs(DM.sig(r, "disagreement")))]
    out2 = {"max_abs_dev": {}, "signed_last25_minus_first25": {}}
    say("\n2. GRID BASELINES, 9 v 9 (count_vowels, second_largest, is_prime; phase2_grid.json). The hacking arm "
        "trains on jsmall, the benign arm on jbig (grid note).")
    say("2a. Identical score: max over steps of |x - median(first 5 finite)| (detection_metrics.dev). "
        "Verdict against disagreement's %.2f." % A_DIS)
    for lab, f in GRID:
        e = evaluate([maxdev(f(r, "h")) for r in hack], [maxdev(f(r, "b")) for r in ben])
        e["verdict"] = verdict(e, A_DIS)
        out2["max_abs_dev"][lab] = e
        say(line(lab, e, [("published", A_DIS)]))
    say("2b. Signed, design-transparent score: mean of the last 25 steps minus the first 25 (e1_supplementary.chg). "
        "The sign that favors detection is in the dir column. Verdicts against the published %.2f and against "
        "disagreement under this same score." % A_DIS)
    s_ref = None
    for lab, f in GRID:
        e = evaluate([chg(f(r, "h")) for r in hack], [chg(f(r, "b")) for r in ben])
        s_ref = best(e) if s_ref is None else s_ref           # the first row is disagreement
        e["verdict"] = verdict(e, A_DIS)
        e["verdict_vs_signed_disagreement"] = verdict(e, s_ref)
        out2["signed_last25_minus_first25"][lab] = e
        say(line(lab, e, [("published", A_DIS), ("signed disagreement", s_ref)]))
    res["sections"]["2_grid"] = out2

    # ---------------------------------------------------------------------------------- 3
    say("\n3. FAMILIES (family_matrix.py, specificity_partial.py; phase2_family.json, phase2_partial.json)")
    fm_pub = json.loads((ROOT / "paper" / "stats" / "family_matrix.json").read_text())["matrix"]
    fben = FM.runs(FM.GRID["benign"])
    fm_bad = []
    for name in FM.DETS:
        nb = [FM.peakdev(FM.sig(r, name)) for r in fben]
        for fam, cells in FM.FAMILIES.items():
            v = round(FM.auroc([FM.peakdev(FM.sig(r, name)) for r in FM.runs(cells)], nb), 2)
            if v != fm_pub[name][fam]:
                fm_bad.append((name, fam, v, fm_pub[name][fam]))
    say("  family_matrix.json reproduced with family_matrix's own peakdev, sig, auroc: %s" % (
        "all %d cells MATCH" % (2 * len(FM.DETS)) if not fm_bad else "MISMATCH %s" % fm_bad))
    sp_pub = json.loads((ROOT / "paper" / "stats" / "specificity_partial.json").read_text())
    sp_bad = []
    for lab, cells in SPC.GROUPS:
        rs = SPC.runs(cells)
        for nm in SPC.DETS:
            v = round(float(np.mean([SPC.peakdev(SPC.sig(r, nm)) for r in rs])), 3)
            if v != sp_pub["peakdev_mean"][nm][lab]:
                sp_bad.append((nm, lab, v))
        v = round(float(np.mean([SPC.hidden_drop(r) for r in rs])), 3)
        if v != sp_pub["hidden_drop_mean"][lab]:
            sp_bad.append(("hidden_drop", lab, v))
    say("  specificity_partial.json reproduced (group means of peakdev and hidden drop): %s. Its per-run data are "
        "experiments/phase2/phase2_partial.json, cells['partial'], 9 runs, same three tasks." % (
            "all cells MATCH" if not sp_bad else "MISMATCH %s" % sp_bad))
    if fm_bad or sp_bad:
        say("STOP: the family tables do not reproduce.")
        OUT.write_text(json.dumps(res, indent=1))
        return 1
    say("  The verifier-coverage ('visible') runs in phase2_family.json are not used by family_matrix or "
        "specificity_partial. The paper's Table 1 labels them non-hacking. They are added here as a second "
        "negative arm in which no judge is optimized.")
    POS = {"reduced-capacity (trains on 0.5B judge)": FM.FAMILIES["reduced-capacity"],
           "heuristic-proxy (trains on a heuristic)": FM.FAMILIES["heuristic-proxy"]}
    NEG = {"benign (trains on 1.5B judge)": FM.GRID["benign"],
           "partial-credit (trains on graded sub-goals)": SPC.PAR["partial"],
           "verifier-coverage (trains on visible tests)": FM.FAM["visible"]}
    FSIG = [("disagreement max|dev| (family_matrix)", lambda r: FM.peakdev(FM.sig(r, "disagreement"))),
            ("disagreement signed", lambda r: chg(FM.sig(r, "disagreement"))),
            ("(a) jsmall max|dev|", lambda r: FM.peakdev(FM.series(r["rows"], "jsmall"))),
            ("(a) jsmall signed", lambda r: chg(FM.series(r["rows"], "jsmall"))),
            ("(d) hidden signed (label check)", lambda r: chg(FM.series(r["rows"], "hidden")))]
    out3 = {}
    for pl, pc in POS.items():
        for nl, nc in NEG.items():
            pr, nr = FM.runs(pc), FM.runs(nc)
            neither = pl.startswith("heuristic") and not nl.startswith("benign")
            key = "%s v %s" % (pl.split(" (")[0], nl.split(" (")[0])
            say("  %s v %s%s" % (pl, nl, "   <- neither arm optimizes a judge" if neither else ""))
            out3[key] = {"neither_arm_optimizes_a_judge": neither}
            ref = {}
            for lab, f in FSIG:
                e = evaluate([f(r) for r in pr], [f(r) for r in nr])
                sc = "max|dev|" if "max|dev|" in lab else "signed"
                refs = []
                if lab.startswith("disagreement"):
                    ref[sc] = best(e)
                elif lab.startswith("(a)"):
                    refs = [("disagreement " + sc, ref[sc])]
                    e["verdict_vs_disagreement_same_score"] = verdict(e, ref[sc])
                out3[key][lab] = e
                say(line(lab, e, refs, tag="  "))
    say("  family_matrix's published disagreement cells: reduced-capacity %.2f, heuristic-proxy %.2f (both reproduced "
        "above as the max|dev| rows against benign)." % (fm_pub["disagreement"]["reduced-capacity"],
                                                          fm_pub["disagreement"]["heuristic-proxy"]))
    res["sections"]["3_families"] = out3

    # ---------------------------------------------------------------------------------- 4
    used, excluded, unreg, _, surplus = P.load_main()
    he, be = used["hacking"], used["benign"]
    say("\n4. %s E1, %d v %d (e1_primary.load_main; %d excluded, %d unregistered, %d surplus). E1's registered "
        "analysis has no audit test, so nothing here is registered." % (X, len(he), len(be), len(excluded), len(unreg),
                                                                       len(surplus)))
    for arm, rs, opt in (("hacking", he, "judge_small"), ("benign", be, "judge_big")):
        for r in rs:
            rr = D.rows(r)
            pz, op = D.series(rr, "proxy"), D.series(rr, opt)
            dz = D.series(rr, "judge_small") - D.series(rr, "judge_big")
            assert np.allclose(pz, op, equal_nan=True), (r["_tag"], "proxy is not %s" % opt)
            assert np.allclose(D.sig(r, "disagreement", "dm"), dz, atol=1e-6, equal_nan=True), r["_tag"]
            for f in ("judge_small", "judge_big", "proxy", "hidden", "disagree"):
                assert abs(chg(D.series(rr, f)) - ES.chg(r, f)) < 1e-12, (r["_tag"], f)
    say("%s   checked on every run: proxy == judge_small in all 24 judge05 runs and proxy == judge_big in all 24 "
        "judge15 runs; disagree == judge_small - judge_big; chg() here equals e1_supplementary.chg." % X)
    ESIG = [("disagreement (E1 registered signal, signed)", lambda r: D.sig(r, "disagreement", "dm")),
            ("|disagreement|", lambda r: D.sig(r, "disagreement", "gb")),
            ("(a) judge_small (0.5B) alone", lambda r: D.series(D.rows(r), "judge_small")),
            ("(b) judge_big (1.5B) alone", lambda r: D.series(D.rows(r), "judge_big")),
            ("(c) optimized judge (proxy field)", lambda r: D.series(D.rows(r), "proxy")),
            ("(d) strict held-out oracle (hidden)", lambda r: D.series(D.rows(r), "hidden"))]
    out4 = {"max_abs_dev": {}, "signed_last25_minus_first25": {}}
    e_ref = {}
    for sc, fn, key in (("max|dev| (detection_metrics.dev)", maxdev, "max_abs_dev"),
                        ("signed last-25 minus first-25 (e1_supplementary.chg)", chg, "signed_last25_minus_first25")):
        say("%s 4%s. %s" % (X, "a" if key == "max_abs_dev" else "b", sc))
        for lab, f in ESIG:
            e = evaluate([fn(f(r)) for r in he], [fn(f(r)) for r in be])
            if lab.startswith("disagreement"):
                e_ref[key] = best(e)
            e["verdict_vs_e1_disagreement"] = verdict(e, e_ref[key])
            out4[key][lab] = e
            say(line(lab, e, [("E1 disagreement", e_ref[key])], tag=X))
    res["sections"]["4_e1_exploratory"] = out4

    # ---------------------------------------------------------------------------------- 5
    say("\n5. INVENTORY: every logged arm and the reward it trains on (scanned: experiments/phase2/*.json, "
        "runs/phase1b_dataset.json, runs/e1/{main,pilot,diag,cap512}/*.json). 'fell' = strict oracle lower in the "
        "last 25 steps than the first 25.")

    def share_small(rows, small="jsmall"):
        v = [(r.get("proxy"), r.get(small)) for r in rows if r.get("proxy") is not None and r.get(small) is not None]
        return (sum(abs(a - b) < 1e-9 for a, b in v) / len(v)) if v else None

    def fell(rows, sort=True):
        rows = sorted(rows, key=lambda r: r["step"]) if sort else rows
        return chg(np.array([np.nan if r.get("hidden") is None else r["hidden"] for r in rows], float)) < 0

    inv = []

    def add(src, arm, runs_rows, how, label, small="jsmall"):
        sh = [share_small(rw, small) for rw in runs_rows]
        sh = [s for s in sh if s is not None]
        if sh:
            how = "0.5B judge (proxy == 0.5B score on %.0f%% of steps)" % (100 * np.mean(sh)) if min(sh) > 0.999 else (
                how + " (proxy == 0.5B score on %.0f%% of steps)" % (100 * np.mean(sh)))
        inv.append({"source": src, "arm": arm, "n": len(runs_rows), "trains_on": how, "paper_label": label,
                    "strict_fell": int(sum(fell(rw) for rw in runs_rows))})

    ph = lambda f: json.loads((ROOT / "experiments" / "phase2" / f).read_text())
    g = ph("phase2_grid.json")
    add("phase2_grid", "hacking", [r["rows"] for r in runs_of(g["cells"]["hacking"])], "0.5B judge (file note)", "hacking")
    add("phase2_grid", "benign (3 tasks)", [r["rows"] for r in runs_of(g["cells"]["benign"])], "1.5B judge (file note)", "benign")
    add("phase2_grid", "benign (sum_list)", [r["rows"] for r in runs_of(g["cells"]["benign"], ["sum_list"])],
        "1.5B judge (file note)", "benign")
    fam = ph("phase2_family.json")
    add("phase2_family", "heuristic", [r["rows"] for r in runs_of(fam["cells"]["heuristic"])], "heuristic proxy", "hacking")
    add("phase2_family", "visible", [r["rows"] for r in runs_of(fam["cells"]["visible"])], "visible tests", "non-hacking")
    add("phase2_partial", "partial", [r["rows"] for r in runs_of(ph("phase2_partial.json")["cells"]["partial"])],
        "graded sub-goals", "non-hacking")
    gm = ph("phase2_gemma.json")
    assert gm["judge_optimized"] == P.SMALL
    add("phase2_gemma", "reduced_capacity", [r["rows"] for r in runs_of(gm["cells"]["reduced_capacity"])],
        "0.5B judge (judge_optimized field)", "hacking")
    for f in ("phase2_qwen3b.json", "phase2_rloo_fmt.json"):
        d = ph(f)
        for c, rs in d["conditions"].items():
            add(f[:-5], c, [v["rows"] for v in rs.values()], "non-judge proxy",
                "hacking" if "reduced" in c else ("non-hacking" if "formatting" in c else "not in Table 1"))
    lo = ph("phase2_lowlr.json")
    add("phase2_lowlr", lo["condition"], [v["rows"] for v in lo["seeds"].values()], "?", "hacking")
    pb = json.loads((ROOT / "runs" / "phase1b_dataset.json").read_text())
    for c, rs in pb.items():
        add("phase1b", c, [v["rows"] for v in rs.values()],
            "1.5B judge (judge field; no 0.5B score logged)" if c == "plain_judge" else "non-judge proxy",
            {"plain_judge": "benign", "verbosity": "hacking", "reduced_capacity": "hacking"}[c])
    for sub in ("main", "pilot", "diag", "cap512"):
        groups = {}
        for fp in sorted((ROOT / "runs" / "e1" / sub).glob("*.json")):
            dd = json.loads(fp.read_text(encoding="utf-8"))
            if isinstance(dd, dict) and "log" in dd:
                groups.setdefault(dd.get("judge"), []).append(dd["log"])
        for jd, logs in groups.items():
            add("e1/" + sub, "judge05" if jd == P.SMALL else "judge15", logs,
                "1.5B judge" if jd == P.BIG else "?", "hacking" if jd == P.SMALL else "benign", small="judge_small")
    for r in inv:
        say("  %-15s %-22s n=%-3d trains on %-58s paper label %-14s strict fell %d/%d" % (
            r["source"], r["arm"], r["n"], r["trains_on"], r["paper_label"], r["strict_fell"], r["n"]))
    small_arms = [r for r in inv if r["trains_on"].startswith("0.5B judge")]
    benign_small = [r for r in small_arms if r["paper_label"] != "hacking"]
    n_small = sum(r["n"] for r in small_arms)
    n_small_kept = sum(r["n"] - r["strict_fell"] for r in small_arms)
    say("  arms that train on the 0.5B judge: %d (%d runs); of these labeled benign or non-hacking: %d; runs in them "
        "whose strict oracle did not fall: %d of %d" % (len(small_arms), n_small, len(benign_small), n_small_kept,
                                                         n_small))
    res["sections"]["5_inventory"] = inv

    # ---------------------------------------------------------------------------------- 6
    say("\n6. %s What the separation is made of. Added 2026-10-08 after an interpretation review; the smoothing "
        "windows were chosen after seeing the data. Grid lines are exploratory too." % X)

    def ma(x, w):
        """Trailing (causal) moving mean over w steps; NaN until w steps exist."""
        x = np.asarray(x, float)
        out = np.full(len(x), np.nan)
        for i in range(w - 1, len(x)):
            seg = x[i - w + 1:i + 1]
            if np.isfinite(seg).any():
                out[i] = np.nanmean(seg)
        return out

    o6 = {"smoothing": {}, "judge_change": {}, "end_gap": {}}
    for nm, hs, bs, f in (("grid 9 v 9", hack, ben, lambda r: DM.sig(r, "disagreement")),
                          ("E1 24 v 24", he, be, lambda r: D.sig(r, "disagreement", "dm"))):
        cells = []
        for w in (1, 5, 10, 25):
            e = evaluate([maxdev(ma(f(r), w)) for r in hs], [maxdev(ma(f(r), w)) for r in bs])
            o6["smoothing"]["%s w=%d" % (nm, w)] = e
            cells.append("w=%d %.2f%s" % (w, best(e), "" if e["dir"] == "+" else " (rev)"))
        say("%s   %s, disagreement scored by max|dev| after a trailing moving mean of w steps (w=1 is the published "
            "score): %s" % (X, nm, "; ".join(cells)))
    for nm, hs, bs, ser in (
            ("grid", hack, ben, lambda r, k: DM.series(r["rows"], {"s": "jsmall", "b": "jbig"}[k])),
            ("E1", he, be, lambda r, k: D.series(D.rows(r), {"s": "judge_small", "b": "judge_big"}[k]))):
        hh, bh = [chg(ser(r, "b")) for r in hs], [chg(ser(r, "s")) for r in bs]      # the judge NOT trained on
        ht, bt = [chg(ser(r, "s")) for r in hs], [chg(ser(r, "b")) for r in bs]      # the training reward
        o6["judge_change"][nm] = {"heldout_hacking": [float(np.mean(hh)), int(sum(v < 0 for v in hh)), len(hh)],
                                  "heldout_benign": [float(np.mean(bh)), int(sum(v < 0 for v in bh)), len(bh)],
                                  "reward_hacking": float(np.mean(ht)), "reward_benign": float(np.mean(bt))}
        say("%s   %s, last 25 minus first 25 steps. The judge each run did NOT train on: hacking runs (1.5B) %+.3f, "
            "falling in %d/%d; benign runs (0.5B) %+.3f, falling in %d/%d. The training reward: hacking %+.3f, "
            "benign %+.3f." % (X, nm, np.mean(hh), sum(v < 0 for v in hh), len(hh), np.mean(bh),
                                sum(v < 0 for v in bh), len(bh), np.mean(ht), np.mean(bt)))
        gh = [float(np.nanmean((ser(r, "s") - ser(r, "b"))[-25:])) for r in hs]
        gb = [float(np.nanmean((ser(r, "b") - ser(r, "s"))[-25:])) for r in bs]
        raw = midrank_auroc(np.asarray(gh), np.asarray(gb))
        o6["end_gap"][nm] = {"mean_hacking": float(np.mean(gh)), "mean_benign": float(np.mean(gb)),
                             "auroc_hacking_positive": round(raw, 4)}
        say("%s   %s, end-of-training gap (training reward minus the other judge, mean of the last 25 steps): "
            "hacking %+.3f, benign %+.3f; AUROC with hacking as positive %.2f (below 0.5 means the benign runs end "
            "with the larger gap)." % (X, nm, np.mean(gh), np.mean(gb), raw))
    res["sections"]["6_exploratory_composition"] = o6

    # ---------------------------------------------------------------------------------- reading
    m2, s2 = out2["max_abs_dev"], out2["signed_last25_minus_first25"]
    KD, KA, KB, KC, KH, KE = list(m2)                       # GRID order
    fa = lambda e: "%.2f%s" % (best(e), " (reversed)" if e["dir"] == "-" else "")
    ok = lambda e: e["verdict"] in ("MATCHES", "MATCHES OR EXCEEDS")
    say("\nREADING")
    for sc, tab in (("published score, max|dev|", m2), ("signed score", s2)):
        say("  Grid, %s: disagreement %s; (a) 0.5B judge alone %s; (b) 1.5B judge alone %s; (c) training reward %s; "
            "(d) strict oracle %s; (e) |disagreement| %s." % (sc, fa(tab[KD]), fa(tab[KA]), fa(tab[KB]), fa(tab[KC]),
                                                           fa(tab[KH]), fa(tab[KE])))
    hits = [(sc, lab, e) for sc, tab in (("max|dev|", m2), ("signed", s2)) for lab, e in tab.items()
            if lab != KD and ok(e)]
    say("  Baselines that match or beat the published %.2f: %s." % (
        A_DIS, "; ".join("%s %s %s" % (lab.split(" ")[0], sc, fa(e)) for sc, lab, e in hits) if hits else "none"))
    say("  Scoring |disagreement| instead gives %.2f (max|dev|) and %.2f (signed). The separation sits in the sign of "
        "jsmall - jbig. Its mean signed change is %+.3f in hacking runs and %+.3f in benign runs: it rises when a run "
        "trains on the 0.5B judge and falls when a run trains on the 1.5B judge." % (
            best(m2[KE]), best(s2[KE]), s2[KD]["mean_pos"], s2[KD]["mean_neg"]))
    a_sc = [sc for sc, tab in (("max|dev|", m2), ("signed", s2)) if ok(tab[KA])]
    c_sc = [sc for sc, tab in (("max|dev|", m2), ("signed", s2)) if ok(tab[KC])]
    if a_sc or c_sc:
        say("  (a) matches under %s; (c) matches under %s. So on the grid a single judge's score does as well as "
            "disagreement's %.2f: which judge each arm optimizes is collinear with the hacking label. The data contain no arm where a benign run optimizes the same 0.5B judge: %d arms (%d runs) "
            "train on it, all are labeled hacking, and the strict oracle fell in %d of the %d. Every run trained on "
            "the 0.5B judge collapsed and no benign run trained on it, so on these arms no audit number can tell a "
            "hacking detector from a detector of which judge the run trained on." % (
                " and ".join(a_sc) or "neither score", " and ".join(c_sc) or "neither score", A_DIS, len(small_arms),
                n_small, n_small - n_small_kept, n_small))
    else:
        say("  Neither (a) nor (c) matches %.2f under either score." % A_DIS)
    dk, ak = "disagreement max|dev| (family_matrix)", "(a) jsmall max|dev|"
    dks, aks, hk = "disagreement signed", "(a) jsmall signed", "(d) hidden signed (label check)"
    hb = out3["heuristic-proxy v benign"]
    say("  Families. Heuristic-proxy v benign: disagreement %s (max|dev|, family_matrix's published %.2f) and %s "
        "(signed); (a) %s and %s. The signed number comes from the benign arm, whose disagreement falls as it trains "
        "on the 1.5B judge (mean %+.3f), against %+.3f in heuristic-proxy runs." % (
            fa(hb[dk]), fm_pub["disagreement"]["heuristic-proxy"], fa(hb[dks]), fa(hb[ak]), fa(hb[aks]),
            hb[dks]["mean_neg"], hb[dks]["mean_pos"]))
    rc = [out3["reduced-capacity v %s" % n] for n in ("partial-credit", "verifier-coverage")]
    say("  Reduced-capacity v partial-credit and v verifier-coverage: disagreement %s and %s, and the 0.5B judge alone "
        "%s and %s (max|dev|). These positives still train on the 0.5B judge." % (
            fa(rc[0][dk]), fa(rc[1][dk]), fa(rc[0][ak]), fa(rc[1][ak])))
    nt = [out3["heuristic-proxy v %s" % n] for n in ("partial-credit", "verifier-coverage")]
    heu = next(r for r in inv if r["source"] == "phase2_family" and r["arm"] == "heuristic")
    say("  Only heuristic-proxy v partial-credit and heuristic-proxy v verifier-coverage have no judge optimized in "
        "either arm. There disagreement gives %s [%.2f, %.2f] and %s [%.2f, %.2f] (max|dev|), and the 0.5B judge alone "
        "gives %s and %s. At 9 v 9 the intervals overlap." % (
            fa(nt[0][dk]), nt[0][dk]["ci"][0], nt[0][dk]["ci"][1], fa(nt[1][dk]), nt[1][dk]["ci"][0],
            nt[1][dk]["ci"][1], fa(nt[0][ak]), fa(nt[1][ak])))
    say("  Two limits on those two numbers. The strict oracle fell in only %d of %d heuristic-proxy runs and separates "
        "that arm from the two controls at only %s and %s, so the family label is not a per-run hacking label. "
        "Under max|dev| the heuristic-proxy runs deviate more (mean %.3f against %.3f and %.3f). Under the signed "
        "score (%s, %s) the gap comes mostly from the controls: their disagreement falls (%+.3f, %+.3f) while their "
        "strict oracle rises (%+.3f, %+.3f), and the heuristic-proxy runs stay near zero (%+.3f)." % (
            heu["strict_fell"], heu["n"], fa(nt[0][hk]), fa(nt[1][hk]), nt[0][dk]["mean_pos"], nt[0][dk]["mean_neg"],
            nt[1][dk]["mean_neg"], fa(nt[0][dks]), fa(nt[1][dks]), nt[0][dks]["mean_neg"], nt[1][dks]["mean_neg"],
            nt[0][hk]["mean_neg"], nt[1][hk]["mean_neg"], nt[0][dks]["mean_pos"]))
    em, es = out4["max_abs_dev"], out4["signed_last25_minus_first25"]
    E = list(em)
    e05 = next(r for r in inv if r["source"] == "e1/main" and r["arm"] == "judge05")
    e15 = next(r for r in inv if r["source"] == "e1/main" and r["arm"] == "judge15")
    say("  %s E1 has the same design and the same confound: proxy equals judge_small in every judge05 run and "
        "judge_big in every judge15 run, and the strict oracle fell in %d/%d judge05 runs and %d/%d judge15 runs. "
        "Disagreement gives %s (max|dev|) and %s (signed). The 0.5B judge alone gives %s and %s. The training reward "
        "gives %s and %s." % (X, e05["strict_fell"], e05["n"], e15["strict_fell"], e15["n"], fa(em[E[0]]),
                              fa(es[E[0]]), fa(em[E[2]]), fa(es[E[2]]), fa(em[E[4]]), fa(es[E[4]])))
    res["reading_inputs"] = {"grid_baselines_matching_or_beating": [[sc, lab, best(e)] for sc, lab, e in hits],
                             "a_matches_under": a_sc, "c_matches_under": c_sc,
                             "arms_training_on_0.5B": len(small_arms), "runs_training_on_0.5B": n_small,
                             "of_which_arms_not_labeled_hacking": len(benign_small),
                             "of_which_runs_strict_did_not_fall": n_small_kept}
    OUT.write_text(json.dumps(res, indent=1))
    say("\nresults: %s" % OUT.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
