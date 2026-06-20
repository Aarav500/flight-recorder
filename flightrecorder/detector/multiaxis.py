"""Multi-axis onset detector for the detector-characterization study.

Extends the contraction-tube idea beyond convergence-SHARPNESS to the two candidate axes the
extractors already compute but the tube ignores:
  - DEGENERACY: output collapses to short / low-diversity / high-confidence (gen_len_mean down,
    ngram_diversity down, logprob_concentration up).
  - ADV_SHAPE: the GRPO advantage distribution becomes heavy-tailed / bimodal (adv_kurtosis up,
    |adv_skew| up) — the signature of a group splitting into reward modes.

Each axis fits a robust (median/MAD) reference on a warm-up window, then fires when its signals
deviate in the 'suspicious' direction by > z robust-SDs, sustained for `persistence` steps. The
axes are reported SEPARATELY so the study can ask which (if any) survives a HARD benign control.
This is oracle-blind by construction: it only reads RolloutFrame fields.
"""
from __future__ import annotations

import numpy as np
from ..types import RolloutFrame

AXES = ("sharpness", "degeneracy", "adv_shape")


class MultiAxisDetector:
    def __init__(self, warmup: int = 30, persistence: int = 5, z: float = 3.5):
        self.warmup = warmup
        self.persistence = persistence
        self.z = z
        self._buf: list[dict] = []
        self._ref: dict[str, tuple[float, float]] | None = None
        self._run = {a: 0 for a in AXES}
        self.onset = {a: None for a in AXES}
        self.step = -1

    @staticmethod
    def _feats(f: RolloutFrame) -> dict:
        return {
            "kl_accel": f.kl_accel,
            "entropy_trend": f.entropy_trend,
            "gen_len": f.gen_len_mean,
            "concentration": f.logprob_concentration,
            "adv_kurtosis": f.adv_kurtosis,
            "adv_skew": abs(f.adv_skew),
        }

    def update(self, f: RolloutFrame) -> dict:
        self.step += 1
        v = self._feats(f)
        if self.step < self.warmup:
            self._buf.append(v)
            return self.onset
        if self._ref is None:
            self._ref = {}
            for k in self._buf[0]:
                arr = np.array([b[k] for b in self._buf], float)
                med = float(np.median(arr))
                mad = float(np.median(np.abs(arr - med))) * 1.4826 + 1e-9
                self._ref[k] = (med, mad)

        def z(k: str) -> float:
            med, mad = self._ref[k]
            return (v[k] - med) / mad

        fired = {
            # sharp convergence: KL accelerating up OR entropy collapsing down
            "sharpness": (z("kl_accel") > self.z) or (z("entropy_trend") < -self.z),
            # degeneracy: output got SHORT and CONFIDENT
            "degeneracy": (z("gen_len") < -self.z) and (z("concentration") > self.z),
            # advantage shape: heavy-tailed / skewed advantages
            "adv_shape": (z("adv_kurtosis") > self.z) or (z("adv_skew") > self.z),
        }
        for a in AXES:
            if fired[a]:
                self._run[a] += 1
                if self._run[a] >= self.persistence and self.onset[a] is None:
                    self.onset[a] = self.step - self.persistence + 1
            else:
                self._run[a] = 0
        return self.onset
