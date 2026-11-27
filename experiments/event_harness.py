"""Fixed harness for choosing E1's replacement collapse event. Committed BEFORE any candidate is scored.

WHY
  The current event, baseline_drop(0.6, persist=2) on 8-completion step means, fires on noise: 41-99%
  on flat series at mid pass rates (review wf_f6879608-9fa). It needs a replacement that:
    (1) almost never fires on a series that is not falling, at any level, for runs up to 800 steps;
    (2) fires on runs that genuinely decline;
    (3) dates the decline close to where the low-noise probe series dates it.

A CANDIDATE is a Python file defining NAME (str), PARAMS (dict) and event(steps, series) -> int | None.
  It is applied to a per-step held-out pass-rate series (8 completions per step, so values are k/8).
  The returned step is the event's date; None means no event.

CORPUS (everything that exists; nothing is selected by outcome)
  frozen  paper/analysis/onset_profile.inventory(): 84 runs. Strict hidden series; arms and tasks vary.
  pilot   runs/e1/pilot/*.json: 6 runs. Strict series plus a 16-completion probe every 5 or 10 steps.
  diag    runs/e1/diag/*.json: 6 runs. Strict AND lenient series; probes on the 3 E1-config runs.
  cap512  runs/e1/cap512/*.json: whatever exists at evaluation time.
  GROUND TRUTH comes from the series itself, never from the arm label. In the frozen corpus the
  "hacking" regime includes arms that never collapse (visible tests rise in 9/9).
    declined      mean(last 25) - mean(first 25) <= -0.20, and its block-bootstrap 95% CI excludes 0
    not declined  the CI upper end >= 0 (flat or rising)
    ambiguous     anything else: excluded from both rates

METRICS
  null_synth   fire rate on synthetic flat series: k/8 with k ~ Binomial(8, p) per step,
               p in {0.2, ..., 0.9}, T in {200, 400, 800}, 300 series per cell. The max over cells.
  null_perm    mean fire rate on block-permuted (block 10) NOT-DECLINED real series, 50 permutations
               each. Level and noise are kept, any trend is removed.
  benign_fire  fraction of not-declined real series (any arm, strict or lenient) on which it fires
  hit          fraction of declined real series on which it fires
  dating       median |stream event - probe event| over runs with a probe, the candidate applied
               to both (probe steps in training-step units)
  window       median event step over the declined frozen grid hacking runs (the published comparison)

SELECTION RULE (fixed now)
  Hard constraints: null_synth <= 0.05, null_perm <= 0.05, benign_fire <= 0.05, hit >= 0.85.
  Among the candidates that pass: the smallest dating error wins; then the fewest PARAMS.
  If none passes, report the trade-off table and choose nothing. The decision returns to Aarav.
  Aarav adopts or rejects the winner in the pre-registration (section 1).

AMENDMENT 1 (2026-10-07, after the design panel and BEFORE any candidate was scored on declined runs)
  The design panel's skeptics showed that stream-vs-probe agreement cannot catch a date that is late
  on BOTH series. All three candidates date rise-then-collapse runs and slow declines 45-135 steps
  late, the bias that flatters detectors. So dating is now also scored on a SYNTHETIC BATTERY with
  known onsets. Each shape is drawn at n=8 per step, 200 series per shape:
    step       T=400  0.6 -> 0.2 at 200                                    onset 200
    ramp100    T=400  0.6 linear to 0.1 over steps 200-300                 onset 200
    ramp300    T=800  0.6 linear to 0.1 over steps 200-500                 onset 200
    rise_fall  T=400  0.3 -> 0.75 over 0-100, flat, then 0.75 -> 0.1 over 250-330   onset 250
    two_stage  T=400  0.6 -> 0.4 at 150, -> 0.1 at 250                     onset 150
    early      T=200  0.6 -> 0.2 at 40                                     onset 40
    late       T=200  0.6 -> 0.2 at 150                                    onset 150
  New hard constraints: on every shape, fire rate >= 0.90 AND median |date - onset| <= 25 steps.
  New ranking among passing candidates: the smallest mean over shapes of median |date - onset|; then
  stream-vs-probe dating; then the fewest PARAMS. The battery is synthetic and outcome-blind, so it
  also runs in --null-only mode, and designers may use it.

    python experiments/event_harness.py --null-only CAND.py    # designer feedback: nulls and benign only
    python experiments/event_harness.py CAND.py [CAND2.py ...] # full table + rule (selection stage only)
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "paper" / "analysis"), str(ROOT / "benchmark"), str(ROOT / "experiments")]

NULL_P = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
NULL_T = [200, 400, 800]
NULL_REPS = 300
PERM_REPS = 50
SEED = 20261007
RULE = {"null_synth_max": 0.05, "null_perm_max": 0.05, "benign_fire_max": 0.05, "hit_min": 0.85}


def load_candidate(path):
    spec = importlib.util.spec_from_file_location("cand_" + Path(path).stem, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _label(series):
    from e1_diag_supplement import block_boot_diff
    d, lo, hi = block_boot_diff(series)
    if d <= -0.20 and hi < 0:
        return "declined"
    if hi >= 0:
        return "not_declined"
    return "ambiguous"


def corpus():
    """List of dicts: tag, source, kind (strict|lenient), steps, series, label, probe (steps, series) or None."""
    import onset_profile as OP
    import e0_slope_validation as E0
    out = []
    for r in OP.inventory():
        s, h = E0.series(r)
        s, h = list(map(int, s)), [float(x) for x in h]
        out.append({"tag": "frozen:" + r["tag"], "source": "frozen", "grid_hacking": r["tag"].startswith("grid:hacking"),
                    "kind": "strict", "steps": s, "series": h, "probe": None})
    for sub in ("pilot", "diag", "cap512"):
        for f in sorted((ROOT / "runs" / "e1" / sub).glob("*.json")):
            d = json.loads(f.read_text())
            if "log" not in d:
                continue
            s = [int(x["step"]) for x in d["log"]]
            pr = d.get("probe") or []
            probe = ([int(p["step"]) for p in pr], [float(p["probe_hidden"]) for p in pr]) if pr else None
            out.append({"tag": "%s:%s" % (sub, f.stem), "source": sub, "grid_hacking": False, "kind": "strict",
                        "steps": s, "series": [float(x["hidden"]) for x in d["log"]], "probe": probe})
            if "hidden_lenient" in d["log"][0]:
                out.append({"tag": "%s:%s:lenient" % (sub, f.stem), "source": sub, "grid_hacking": False,
                            "kind": "lenient", "steps": s, "series": [float(x["hidden_lenient"]) for x in d["log"]],
                            "probe": None})
    for c in out:
        c["label"] = _label(c["series"])
    return out


def _fires(cand, steps, series):
    try:
        return cand.event(list(steps), list(series)) is not None
    except Exception as e:                       # a crash counts as a fire: it must not pass silently
        print("  CRASH in %s: %r" % (getattr(cand, "NAME", "?"), e))
        return True


def null_synth(cand):
    rng = np.random.default_rng(SEED)
    cells = {}
    for T in NULL_T:
        steps = list(range(T))
        for p in NULL_P:
            fires = sum(_fires(cand, steps, (rng.binomial(8, p, T) / 8.0).tolist()) for _ in range(NULL_REPS))
            cells["p%.1f_T%d" % (p, T)] = fires / NULL_REPS
    return max(cells.values()), cells


def null_perm(cand, runs):
    rng = np.random.default_rng(SEED)
    rates = []
    for c in runs:
        x = np.asarray(c["series"], float)
        blocks = [x[i:i + 10] for i in range(0, len(x), 10)]
        f = sum(_fires(cand, c["steps"], np.concatenate([blocks[j] for j in rng.permutation(len(blocks))]).tolist())
                for _ in range(PERM_REPS))
        rates.append(f / PERM_REPS)
    return float(np.mean(rates)) if rates else float("nan")


BATTERY_REPS = 200
BATTERY_TOL = 25
BATTERY_FIRE_MIN = 0.90


def _shape(name, T):
    t = np.arange(T, dtype=float)
    if name == "step":
        return np.where(t < 200, 0.6, 0.2), 200
    if name == "ramp100":
        return np.clip(0.6 - 0.5 * (t - 200) / 100.0, 0.1, 0.6), 200
    if name == "ramp300":
        return np.clip(0.6 - 0.5 * (t - 200) / 300.0, 0.1, 0.6), 200
    if name == "rise_fall":
        up = 0.3 + 0.45 * np.clip(t / 100.0, 0, 1)
        return np.where(t < 250, up, np.clip(0.75 - 0.65 * (t - 250) / 80.0, 0.1, 0.75)), 250
    if name == "two_stage":
        return np.where(t < 150, 0.6, np.where(t < 250, 0.4, 0.1)), 150
    if name == "early":
        return np.where(t < 40, 0.6, 0.2), 40
    if name == "late":
        return np.where(t < 150, 0.6, 0.2), 150
    raise ValueError(name)


BATTERY = [("step", 400), ("ramp100", 400), ("ramp300", 800), ("rise_fall", 400),
           ("two_stage", 400), ("early", 200), ("late", 200)]


def battery(cand):
    rng = np.random.default_rng(SEED + 1)
    out = {}
    for name, T in BATTERY:
        mean, onset = _shape(name, T)
        steps = list(range(T))
        fires, errs = 0, []
        for _ in range(BATTERY_REPS):
            x = (rng.binomial(8, mean) / 8.0).tolist()
            try:
                e = cand.event(steps, x)
            except Exception:
                e = None
            if e is not None:
                fires += 1
                errs.append(e - onset)
        errs = np.asarray(errs, float)
        out[name] = {"fire": fires / BATTERY_REPS,
                     "median_bias": float(np.median(errs)) if len(errs) else None,
                     "median_abs": float(np.median(np.abs(errs))) if len(errs) else None}
    return out


def battery_ok(b):
    return all(v["fire"] >= BATTERY_FIRE_MIN and v["median_abs"] is not None and v["median_abs"] <= BATTERY_TOL
               for v in b.values())


def battery_score(b):
    vals = [v["median_abs"] for v in b.values() if v["median_abs"] is not None]
    return float(np.mean(vals)) if len(vals) == len(b) else 1e9


def evaluate(cand, runs, null_only=False):
    nd = [c for c in runs if c["label"] == "not_declined"]
    res = {"name": cand.NAME, "n_params": len(getattr(cand, "PARAMS", {})), "params": getattr(cand, "PARAMS", {})}
    res["null_synth"], res["null_synth_cells"] = null_synth(cand)
    res["null_perm"] = null_perm(cand, nd)
    res["benign_fire"] = float(np.mean([_fires(cand, c["steps"], c["series"]) for c in nd]))
    res["n_not_declined"] = len(nd)
    res["battery"] = battery(cand)
    res["battery_ok"] = battery_ok(res["battery"])
    res["battery_score"] = battery_score(res["battery"])
    if null_only:
        return res
    dec = [c for c in runs if c["label"] == "declined"]
    res["n_declined"] = len(dec)
    res["hit"] = float(np.mean([_fires(cand, c["steps"], c["series"]) for c in dec]))
    errs = []
    for c in runs:
        if c["probe"] and c["label"] == "declined":
            a = cand.event(c["steps"], c["series"])
            b = cand.event(*c["probe"])
            if a is not None and b is not None:
                errs.append(abs(a - b))
    res["dating_median_abs"] = float(np.median(errs)) if errs else None
    res["dating_n"] = len(errs)
    win = [cand.event(c["steps"], c["series"]) for c in dec if c["grid_hacking"]]
    win = [w for w in win if w is not None]
    res["window_grid_median"] = float(np.median(win)) if win else None
    res["passes"] = (res["null_synth"] <= RULE["null_synth_max"] and res["null_perm"] <= RULE["null_perm_max"]
                     and res["benign_fire"] <= RULE["benign_fire_max"] and res["hit"] >= RULE["hit_min"]
                     and res["battery_ok"])
    return res


def main(argv):
    null_only = "--null-only" in argv
    paths = [a for a in argv if not a.startswith("--")]
    runs = corpus()
    labels = {}
    for c in runs:
        labels[c["label"]] = labels.get(c["label"], 0) + 1
    print("corpus: %d series %s" % (len(runs), labels))
    results = []
    for p in paths:
        cand = load_candidate(p)
        r = evaluate(cand, runs, null_only)
        results.append(r)
        worst = max(r["null_synth_cells"], key=r["null_synth_cells"].get)
        print("\n%s  params=%s" % (r["name"], r["params"]))
        print("  null_synth max %.3f (worst cell %s)  null_perm %.3f  benign_fire %.3f (n=%d)" % (
            r["null_synth"], worst, r["null_perm"], r["benign_fire"], r["n_not_declined"]))
        print("  battery (fire / median bias / median |err|), ok=%s, score %.1f:" % (r["battery_ok"], r["battery_score"]))
        for k, v in r["battery"].items():
            print("    %-10s %.2f  %s  %s" % (k, v["fire"], v["median_bias"], v["median_abs"]))
        if not null_only:
            print("  hit %.3f (n=%d)  dating median |stream-probe| %s (n=%d)  grid window median %s  PASSES %s" % (
                r["hit"], r["n_declined"], r["dating_median_abs"], r["dating_n"], r["window_grid_median"], r["passes"]))
    if not null_only and results:
        ok = [r for r in results if r["passes"]]
        if ok:
            best = sorted(ok, key=lambda r: (r["battery_score"],
                                             r["dating_median_abs"] if r["dating_median_abs"] is not None else 1e9,
                                             r["n_params"]))[0]
            print("\nSELECTED by the fixed rule: %s" % best["name"])
        else:
            print("\nNO CANDIDATE PASSES. Report the trade-offs; the choice returns to Aarav.")
        out = ROOT / "experiments" / "event_harness_results.json"
        out.write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
