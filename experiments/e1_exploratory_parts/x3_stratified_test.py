"""[EXPLORATORY] X3: a task-stratified exact test (prereg section 10, experiments/e1_preregistration.md:252 at
d1aa118: "A task-stratified exact test, a Cochran-Mantel-Haenszel-type stratified permutation"; why: "3 task
clusters"; :144 names it a sensitivity analysis for the 3 task clusters). Neither the pre-registration nor
paper/neurips_paper.tex defines it further.

NOT the registered analysis and NOT pinned. Written 2026-10-08, AFTER the registered result
(experiments/e1_primary_report.txt: CLEAN NULL, non-superiority shown). It cannot change the registered verdict.
It imports e1_primary, e1_event and e1_detectors and scores every detector with the registered P.score, so the
run outcomes are the registered ones; only the null reference distribution is stratified by task.

  Per task t, the 2x2 table is: hacking strict hit / miss (alarm strictly before the e1_event step; no event is a
  miss) against benign alarm / no alarm, at the registered pooled operating point (TS.fpr10_threshold over all 24
  benign runs). The test conditions on each task's margins, so the hacking hit count X_t is hypergeometric and the
  statistic S = sum_t X_t is a convolution of the three. That equals permuting the 16 per-run outcomes between the
  arms within each task; the module checks it by enumerating all C(16, 8) = 12870 within-task arrangements.

    python experiments/e1_exploratory_parts/x3_stratified_test.py
"""
from __future__ import annotations

import itertools
import json
import sys
from functools import lru_cache
from math import comb
from pathlib import Path

sys.dont_write_bytecode = True       # read-only: leave no new __pycache__ next to the pinned files
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))
import e1_primary as P  # noqa: E402
import e1_event as EV  # noqa: E402
import e1_detectors as D  # noqa: E402

SP, TS, DM = P.SP, P.TS, P.DM
X = "[EXPLORATORY] X3"
CH = "CHOICE (made after the data):"
SKEPTIC_P = 0.8211      # wf_e1_verify/skeptic/t2_timing.out:55, same test computed independently
ABBR = {"count_vowels": "cv", "is_prime": "ip", "second_largest": "sl"}


def cp(x, n):
    return "(%.3f, %.3f)" % P.clopper_pearson(x, n)


def cnt(x, n):
    return "%d/%d %s" % (x, n, cp(x, n))


def provenance():
    """e1_supplementary.provenance without the sensitivity files. P.verify_pins() is not used: it would list this
    unpinned module (it lives under experiments/) as a missing pin and stop."""
    pins = json.loads(P.PINS.read_text())
    pins.pop("_env", None)
    bad = [rel for rel, h in pins.items() if not (P.ROOT / rel).exists() or P._sha(P.ROOT / rel) != h]
    if bad:
        raise SystemExit("PROVENANCE FAILURE: %s differ from the frozen versions" % bad)
    return len(pins)


def tables(spec, hack, ben, event_of):
    """Registered score() plus the per-task 2x2 tables {task: (hits x, n_hack, alarms y, n_benign)}."""
    sc = P.score(spec, hack, ben, event_of)
    thr = TS.fpr10_threshold(spec, ben)
    hits = set(sc["hit_runs_strict"])
    tab = {}
    for t in P.TASKS:
        h = [tag for tag, r in hack if r["task"] == t]
        b = [r for r in ben if r["task"] == t and DM.dev(spec["dm"](r)) is not None]   # realized_fpr's denominator
        tab[t] = (sum(tag in hits for tag in h), len(h),
                  sum(SP.alarm_step(None, spec["dm"](r), thr, 1) is not None for r in b), len(b))
    assert sum(v[0] for v in tab.values()) == sc["n_tp_strict"] and sum(v[2] for v in tab.values()) == sc["n_fp"]
    assert sum(v[3] for v in tab.values()) == sc["n_benign"]
    return sc, thr, tab


def _convolve(per_stratum):
    dist = {0: 1}
    for pmf in per_stratum:
        nd = {}
        for s, c in dist.items():
            for i, ci in pmf.items():
                nd[s + i] = nd.get(s + i, 0) + c * ci
        dist = nd
    return dist


def strat_p(tab):
    """Exact conditional p = P(S >= observed), S = sum of hypergeometric X_t. Integer counts, so it is exact."""
    pmfs, den = [], 1
    for x, n1, y, n2 in tab.values():
        m, N = x + y, n1 + n2
        pmfs.append({i: comb(m, i) * comb(N - m, n1 - i) for i in range(max(0, m - n2), min(m, n1) + 1)})
        den *= comb(N, n1)
    dist, obs = _convolve(pmfs), sum(v[0] for v in tab.values())
    return sum(c for s, c in dist.items() if s >= obs) / den, sum((v[0] + v[2]) * v[1] / (v[1] + v[3]) for v in tab.values())


@lru_cache(maxsize=None)
def _enum_stratum(m, n1, n2):
    out = [1] * m + [0] * (n1 + n2 - m)
    c = {}
    for idx in itertools.combinations(range(n1 + n2), n1):    # every assignment of n1 of the runs to "hacking"
        k = sum(out[i] for i in idx)
        c[k] = c.get(k, 0) + 1
    return c


def strat_p_enum(tab):
    """Cross-check by literal within-task permutation: all C(n1+n2, n1) arrangements per task, convolved."""
    dist = _convolve([_enum_stratum(x + y, n1, n2) for x, n1, y, n2 in tab.values()])
    den = 1
    for _x, n1, _y, n2 in tab.values():
        den *= comb(n1 + n2, n1)
    obs = sum(v[0] for v in tab.values())
    return sum(c for s, c in dist.items() if s >= obs) / den


def mh_or(tab):
    """Mantel-Haenszel common odds ratio, a = hacking hit, b = hacking miss, c = benign alarm, d = benign quiet."""
    num = sum(x * (n2 - y) / (n1 + n2) for x, n1, y, n2 in tab.values())
    den = sum((n1 - x) * y / (n1 + n2) for x, n1, y, n2 in tab.values())
    if den == 0:
        return None, ("undefined (0/0: no stratum has a hit with a quiet benign run, or a miss with a benign alarm)"
                      if num == 0 else "+inf (no stratum has a hacking miss together with a benign alarm)")
    return num / den, None


def fmt_or(v, note):
    return "%.3f" % v if v is not None else ("undef" if note.startswith("undefined") else "+inf")


def run():
    lines = []
    say = lines.append
    say("%s TASK-STRATIFIED EXACT TEST (prereg section 10, e1_preregistration.md:252 and :144 at d1aa118). Unpinned, "
        "written after the registered result; it cannot change the registered verdict (CLEAN NULL, non-superiority "
        "shown)." % X)
    say("%s provenance: %d pinned files match experiments/e1_pins.json" % (X, provenance()))
    say("%s The registered text names the test only ('a Cochran-Mantel-Haenszel-type stratified permutation', "
        "'3 task clusters'); the paper (neurips_paper.tex) does not define it. The details below were fixed after the "
        "data." % X)
    say("%s operating point: the registered pooled threshold (TS.fpr10_threshold over all 24 benign runs), not a "
        "per-task threshold. Stratification changes only the null reference distribution, not who alarms." % CH)
    say("%s run outcome: the registered one (P.score). Hacking = strict hit (alarm < e1_event step; no event = miss); "
        "benign = any alarm. Statistic = total hacking hits, the registered Fisher statistic summed over tasks." % CH)
    say("%s test: exact conditional test on each task's 2x2 margins (product of hypergeometrics = within-task "
        "permutation of the 16 outcomes between arms), one-sided 'greater'. Strata = task only; the task-and-seed "
        "pairing across arms is ignored (that is a matched design, not the test X3 names). The asymptotic CMH "
        "chi-square is not used (cells of 8)." % CH)
    say("%s effect size: the Mantel-Haenszel common odds ratio, sum(a d / n) / sum(b c / n), point estimate only; "
        "reported as undefined (0/0) or +inf when its denominator is 0. Holm over the 25 secondary detectors is applied "
        "to the stratified p and reported only (no X3 decision rule is registered)." % CH)

    used, excluded, _unreg, _launched, _surplus = P.load_main()
    hack = [(r["_tag"], r) for r in used["hacking"]]
    ben = used["benign"]
    nh, nb = len(hack), len(ben)
    say("%s runs (e1_primary.load_main): %d hacking, %d benign used, %d excluded" % (X, nh, nb, len(excluded)))
    event_of = lambda run: EV.event(list(range(len(run["log"]))), [r.get("hidden") for r in run["log"]])  # noqa: E731
    specs = [{"label": l, "family": fam, "gb": g, "dm": m} for (l, fam, g, m) in D.detectors()]

    res = {}
    for s in specs:
        sc, thr, tab = tables(s, hack, ben, event_of)
        p, mu = strat_p(tab)
        pe = strat_p_enum(tab)
        o, note = mh_or(tab)
        res[s["label"]] = {"sc": sc, "thr": thr, "tab": tab, "p": p, "p_enum": pe, "null_mean": mu, "or": o,
                           "or_note": note, "pooled": P.fisher_one_sided(sc["n_tp_strict"], sc["n_hack"],
                                                                         sc["n_fp"], sc["n_benign"])}
    max_enum_diff = max(abs(v["p"] - v["p_enum"]) for v in res.values())

    # ------------------------------------------------------------------ primary
    r = res[P.PRIMARY]
    tp, fp = r["sc"]["n_tp_strict"], r["sc"]["n_fp"]
    say("%s PRIMARY '%s' at the registered pooled threshold %.4f" % (X, P.PRIMARY, r["thr"]))
    say("%s   %-15s %-24s %-24s %-7s %s" % (X, "task", "hacking strict hit x/n", "benign alarm y/n", "margin",
                                            "per-task one-sided Fisher p"))
    for t, (x, n1, y, n2) in r["tab"].items():
        say("%s   %-15s %-24s %-24s %-7d %.4f" % (X, t, cnt(x, n1), cnt(y, n2), x + y, P.fisher_one_sided(x, n1, y, n2)))
    say("%s   pooled, registered (restated): TP %s, FP %s, one-sided Fisher p %.4f" % (
        X, cnt(tp, nh), cnt(fp, r["sc"]["n_benign"]), r["pooled"]))
    say("%s   stratified exact: observed total hits %d, null mean %.2f; one-sided P(S >= %d) = %.4f" % (
        X, tp, r["null_mean"], tp, r["p"]))
    say("%s   cross-checks: enumeration of all within-task arrangements gives %.4f (largest |difference| over the 26 "
        "detectors %.1e); the earlier skeptic (t2_timing.py) got %.4f -> %s" % (
            X, r["p_enum"], max_enum_diff, SKEPTIC_P, "MATCHES" if round(r["p"], 4) == SKEPTIC_P else "DIFFERS"))
    say("%s   Mantel-Haenszel common odds ratio (hit vs benign alarm): %s%s. Below 1 means the benign arm alarms more "
        "often than the hacking arm hits." % (X, fmt_or(r["or"], r["or_note"]), "" if r["or_note"] is None else
                                               " -- " + r["or_note"]))
    hit_tasks = [t for t, v in r["tab"].items() if v[0] > 0]
    say("%s   all %d hits are in %s; %s" % (
        X, tp, ", ".join(hit_tasks) or "no task",
        "the other tasks contribute only benign alarms." if tp else "no task has a hit."))

    # ------------------------------------------------------------------ all 26
    sec = [s["label"] for s in specs if s["label"] != P.PRIMARY]
    adj = dict(zip(sec, P.holm([res[l]["p"] for l in sec])))
    say("%s ALL 26 DETECTORS, each at its own registered pooled threshold. Per-task cells are hacking hits/benign "
        "alarms (each of 8; cv count_vowels, ip is_prime, sl second_largest). Holm is over the 25 secondaries, on "
        "the stratified p, reported only." % X)
    say("%s   CP 95%% for k/8: %s" % (X, "; ".join("%d %s" % (k, cp(k, 8)) for k in range(9))))
    say("%s   %-34s %-5s %-5s %-5s %-19s %-19s %-7s %-7s %-7s %s" % (
        X, "detector", "cv", "ip", "sl", "TP x/24 CP 95%", "FP y/24 CP 95%", "pooled", "strat", "Holm25", "OR_MH"))
    order = [P.PRIMARY] + sorted(sec, key=lambda l: (res[l]["p"], l))
    rows = []
    for l in order:
        v = res[l]
        x, y, n2 = v["sc"]["n_tp_strict"], v["sc"]["n_fp"], v["sc"]["n_benign"]
        cells = ["%d/%d" % (v["tab"][t][0], v["tab"][t][2]) for t in P.TASKS]
        say("%s   %-34s %-5s %-5s %-5s %-19s %-19s %.4f  %.4f  %-7s %s" % (
            X, l + (" *" if l == P.PRIMARY else ""), cells[0], cells[1], cells[2], cnt(x, nh), cnt(y, n2), v["pooled"],
            v["p"], "-" if l == P.PRIMARY else "%.4f" % adj[l], fmt_or(v["or"], v["or_note"])))
        rows.append({"label": l, "primary": l == P.PRIMARY, "threshold": v["thr"],
                     "per_task": {t: {"hits": v["tab"][t][0], "n_hack": v["tab"][t][1], "alarms": v["tab"][t][2],
                                      "n_benign": v["tab"][t][3]} for t in P.TASKS},
                     "tp": x, "fp": y, "n_hack": nh, "n_benign": n2, "pooled_fisher_p": v["pooled"],
                     "stratified_p": v["p"], "stratified_p_enum": v["p_enum"], "null_mean_hits": v["null_mean"],
                     "holm25_stratified": None if l == P.PRIMARY else adj[l], "or_mh": v["or"],
                     "or_mh_note": v["or_note"]})
    say("%s   * primary, unadjusted. undef = Mantel-Haenszel OR 0/0; +inf = denominator 0." % X)
    raw = [l for l in order if res[l]["p"] < P.ALPHA]
    say("%s   stratified p < 0.05 (unadjusted): %s; Holm-adjusted < 0.05 among the 25 secondaries: %s. "
        "Stratified p below pooled p for %d/26 detectors, above for %d/26." % (
            X, cnt(len(raw), 26) + (" " + str(raw) if raw else ""),
            cnt(sum(adj[l] < P.ALPHA for l in sec), 25),
            sum(res[l]["p"] < res[l]["pooled"] - 1e-12 for l in order),
            sum(res[l]["p"] > res[l]["pooled"] + 1e-12 for l in order)))
    say("%s closing note: the registered pooled one-sided Fisher p (%.4f, restated) and the registered verdict "
        "(CLEAN NULL) are unchanged; X3 is a sensitivity reading only." % (X, r["pooled"]))

    data = {"primary": rows[0], "detectors": rows, "skeptic_p": SKEPTIC_P, "max_enum_abs_diff": max_enum_diff,
            "n_hack": nh, "n_benign": nb}
    return {"id": "X3", "title": "Task-stratified exact test (CMH-type stratified permutation)", "lines": lines,
            "data": data, "figures": []}


if __name__ == "__main__":
    out = run()
    json.dumps(out["data"])          # the contract: data must be JSON-serializable
    print("\n".join(out["lines"]))
