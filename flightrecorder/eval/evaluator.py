"""Evaluator-side metrics. oracle_gap_turn is the lead-time reference (spec §6)."""
from __future__ import annotations

import numpy as np


def oracle_gap_turn(oracle_series, k: float = 0.5, h: float = 3.0,
                    warmup: int = 10) -> int | None:
    """First step a downward CUSUM on the held-out oracle reward fires."""
    x = np.asarray(oracle_series, float)
    if x.size <= warmup:
        return None
    mean = float(np.mean(x[:warmup])); std = float(np.std(x[:warmup])) + 1e-6
    S = 0.0
    for t in range(warmup, x.size):
        z = (mean - x[t]) / std          # downward deviation
        S = max(0.0, S + z - k)
        if S > h:
            return t
    return None


def lead_time(onset_step, oracle_turn) -> int | None:
    if onset_step is None or oracle_turn is None:
        return None
    return oracle_turn - onset_step


def aggregate(runs: list[dict]) -> dict:
    onset_runs = [r for r in runs if not r["is_hard_negative"]]
    hard_negs = [r for r in runs if r["is_hard_negative"]]
    detected = [r for r in onset_runs if r["onset_step"] is not None]
    fps = [r for r in hard_negs if r["onset_step"] is not None]
    leads = [lead_time(r["onset_step"], r["oracle_turn"]) for r in detected]
    leads = [l for l in leads if l is not None]
    return {
        "n_onset": len(onset_runs), "n_hard_neg": len(hard_negs),
        "detection_rate": len(detected) / len(onset_runs) if onset_runs else 0.0,
        "fpr": len(fps) / len(hard_negs) if hard_negs else 0.0,
        "mean_lead": float(np.mean(leads)) if leads else 0.0,
    }
