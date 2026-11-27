"""E1 detector set: the 20 published detectors (tex:723) plus completion length and truncation rate.

WHY THE TWO NEW SIGNALS
  The format diagnostic and the 512-token test (experiments/e1_diag_results.md) showed that the
  strict-oracle collapse runs through presentation. In count_vowels and second_largest at a 256-token
  cap, and second_largest at 512, it runs through answer LENGTH: answers grow until they hit the token
  cap, and the code fence is cut off. None of the 20 published detectors sees length, and the
  published grid never logged it. A negative early-warning result that omits the variable the
  collapse runs through invites the obvious reviewer objection.

MECHANISM PROXIMITY (state this in H1 and the paper)
  Under option (a), a truncated answer FAILS the strict oracle by construction. So on tasks whose
  collapse runs through truncation, truncation rate is partly the event itself, not a precursor. A
  lead by truncation rate there is close to tautological. On prose tasks (is_prime, and count_vowels
  at 512) it is not. Report these two detectors per task, and do not let a truncation-driven lead
  stand as "early warning" without saying so.

STRUCTURE (identical to paper/analysis/threshold_sweep.detectors(), with 2 more base signals)
  8 base signals x {level, velocity, acceleration} + CUSUM(disagree) + CUSUM(kl) = 26 detectors.
  Base signals and their E1 log fields:
    reward variance      adv_std                    (published: adv)
    gradient norm        grad_norm
    disagreement         disagree   ("gb": |.| for lead, "dm": signed for TPR, as in the sweep)
    held-out judge drop  max(judge_big[:5]) - judge_big
    KL                   kl
    entropy              entropy
    completion length    mean_len  = TRL completions/mean_length   (tokens)          NEW
    truncation rate      clipped   = TRL completions/clipped_ratio (share at the cap) NEW
  Velocity and acceleration use curvature_detectors.derived; CUSUM uses threshold_sweep.cusum_series.
  Both are imported, not copied, so E1 scores them exactly as the paper does. --selftest asserts that
  the 6 published base signals equal grid_benchmark.sig / detection_metrics.sig on a frozen grid run.

Runs logged before 2026-10-07 18:3x keep length and clipping only in trl_history. rows() fills them
in, one TRL entry per step in order. Later runs log them directly (launch_slow_onset.py
GeomCallback).

    python experiments/e1_detectors.py --selftest
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "paper" / "analysis"), str(ROOT / "benchmark"), str(ROOT / "anticipatory")]
import curvature_detectors as CD  # noqa: E402
from threshold_sweep import cusum_series  # noqa: E402

BASE = ["reward variance", "gradient norm", "disagreement", "held-out judge drop", "KL", "entropy",
        "completion length", "truncation rate"]
NEW = {"completion length", "truncation rate"}
PUBLISHED_COUNT = 20


def rows(run):
    """Per-step rows with mean_len and clipped filled in from trl_history when the row lacks them."""
    log = [dict(r) for r in sorted(run["log"], key=lambda r: r["step"])]
    if log and "mean_len" not in log[0]:
        th = [h for h in run.get("trl_history", []) if "completions/mean_length" in h]
        for r, h in zip(log, th):
            r["mean_len"] = float(h["completions/mean_length"])
            r["clipped"] = float(h.get("completions/clipped_ratio", np.nan))
    return log


def series(rs, f):
    return np.array([(r.get(f) if r.get(f) is not None else np.nan) for r in rs], float)


def sig(run, name, variant="dm"):
    rs = rows(run)
    if name == "disagreement":
        s = series(rs, "disagree")
        return np.abs(s) if variant == "gb" else s
    if name == "reward variance":
        return series(rs, "adv_std")
    if name == "held-out judge drop":
        jb = series(rs, "judge_big")
        return (np.nanmax(jb[:5]) - jb) if np.isfinite(jb).any() else jb
    if name == "entropy":
        return series(rs, "entropy")
    if name == "KL":
        return series(rs, "kl")
    if name == "gradient norm":
        return series(rs, "grad_norm")
    if name == "completion length":
        return series(rs, "mean_len")
    if name == "truncation rate":
        return series(rs, "clipped")
    raise KeyError(name)


def detectors():
    """[(label, family, f_gb, f_dm)]: f_gb gives the series used for lead, f_dm the series for TPR."""
    out = []
    for n in BASE:
        out.append((n, "reference", (lambda n: lambda r: sig(r, n, "gb"))(n), (lambda n: lambda r: sig(r, n, "dm"))(n)))
    for field, lab in (("disagree", "CUSUM(disagree)"), ("kl", "CUSUM(kl)")):
        f = (lambda field: lambda r: cusum_series(series(rows(r), field)))(field)
        out.append((lab, "cusum", f, f))
    for order, form in ((1, "velocity"), (2, "acceleration")):
        for n in BASE:
            out.append(("%s(%s)" % (form, n), "anticipatory:higher-order",
                        (lambda n, o: lambda r: CD.derived(sig(r, n, "gb"), o))(n, order),
                        (lambda n, o: lambda r: CD.derived(sig(r, n, "dm"), o))(n, order)))
    return out


def selftest():
    import grid_benchmark as GB
    import detection_metrics as DM
    dets = detectors()
    assert len(dets) == 26, len(dets)
    published = [d for d in dets if not any(n in d[0] for n in NEW)]
    assert len(published) == PUBLISHED_COUNT, len(published)
    # 1) the 6 published signals are IDENTICAL to the paper's, on a frozen grid run renamed to E1 fields
    g = json.loads((ROOT / "experiments" / "phase2" / "phase2_grid.json").read_text())
    fr = g["cells"]["hacking"]["is_prime"]["0"]
    n = len(fr["rows"])
    e1 = {"log": [{"step": r["step"], "disagree": r.get("disagree"), "adv_std": r.get("adv"), "judge_big": r.get("jbig"),
                   "entropy": fr["entropy"][i] if i < len(fr["entropy"]) else None,
                   "kl": fr["kl"][i] if i < len(fr["kl"]) else None,
                   "grad_norm": fr.get("grad_norm", [None] * n)[i] if i < len(fr.get("grad_norm", [])) else None,
                   "mean_len": None, "clipped": None} for i, r in enumerate(sorted(fr["rows"], key=lambda r: r["step"]))]}
    for name in GB.DETS:
        for variant, ref in (("gb", GB.sig(fr, name)), ("dm", DM.sig(fr, name))):
            mine = sig(e1, name, variant)
            assert np.allclose(np.nan_to_num(mine, nan=-9), np.nan_to_num(np.asarray(ref, float)[:len(mine)], nan=-9)), (name, variant)
    print("published signals identical to grid_benchmark / detection_metrics on frozen grid run: OK")
    # 2) every detector yields a full-length series with finite values on every E1 run (no outcome is computed)
    files = sorted((ROOT / "runs" / "e1").glob("*/*.json"))
    for f in files:
        run = json.loads(f.read_text())
        if "log" not in run:
            continue
        T = len(run["log"])
        for lab, _, fgb, fdm in dets:
            for fn in (fgb, fdm):
                x = np.asarray(fn(run), float)
                assert len(x) == T, (f.name, lab, len(x), T)
                assert np.isfinite(x).sum() >= T // 2, (f.name, lab, int(np.isfinite(x).sum()))
    print("26 detectors produce full-length, mostly finite series on %d E1 runs: OK" % len(files))
    print("SELFTEST PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(selftest() if "--selftest" in sys.argv else 0)
