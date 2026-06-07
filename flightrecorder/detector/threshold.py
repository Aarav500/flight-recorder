"""Oracle-blind baseline: fire when any geometry signal exceeds a z-score threshold."""
from __future__ import annotations

from ..types import RolloutFrame, DetectorState
from ..core.rolling import RunningMoments
from .base import make_state

# Danger sign per signal: +1 means high values are alarming, -1 means low (collapse).
_SIGNALS = {"kl_accel": +1, "entropy_trend": -1, "adv_drift": +1}


class ThresholdDetector:
    def __init__(self, z: float = 4.0, warmup: int = 20):
        self.z = z; self.warmup = warmup
        self._m = {k: RunningMoments() for k in _SIGNALS}
        self._onset_step: int | None = None; self._n = 0

    def reset(self) -> None:
        self.__init__(self.z, self.warmup)

    def update(self, frame: RolloutFrame) -> DetectorState:
        self._n += 1
        triggered, worst = [], 0.0
        for sig, sign in _SIGNALS.items():
            m = self._m[sig]; val = getattr(frame, sig)
            z = sign * m.zscore(val)
            if self._n > self.warmup and z > self.z:
                triggered.append(sig); worst = max(worst, z)
            m.update(val)
        if triggered and self._onset_step is None:
            self._onset_step = frame.step
            return make_state(frame.step, True, self._onset_step,
                              min(1.0, worst / (2 * self.z)), triggered, 0.0, 0.0,
                              f"threshold breach: {triggered}")
        return make_state(frame.step, False, self._onset_step,
                          min(1.0, worst / (2 * self.z)), triggered, 0.0, 0.0, "")
