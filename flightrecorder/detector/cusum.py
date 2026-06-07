"""Oracle-blind baseline: one-sided CUSUM changepoint on a single geometry signal."""
from __future__ import annotations

from ..types import RolloutFrame, DetectorState
from ..core.rolling import RunningMoments
from .base import make_state


class CusumDetector:
    def __init__(self, signal: str = "kl_accel", k: float = 0.5, h: float = 5.0,
                 warmup: int = 20):
        self.signal = signal; self.k = k; self.h = h; self.warmup = warmup
        self._m = RunningMoments(); self._S = 0.0
        self._onset_step: int | None = None; self._n = 0

    def reset(self) -> None:
        self.__init__(self.signal, self.k, self.h, self.warmup)

    def update(self, frame: RolloutFrame) -> DetectorState:
        self._n += 1
        x = getattr(frame, self.signal)
        if self._n <= self.warmup:
            self._m.update(x)
            return make_state(frame.step, False, self._onset_step, 0.0, [], 0.0, 0.0, "")
        z = self._m.zscore(x)
        self._S = max(0.0, self._S + z - self.k)
        if self._S > self.h and self._onset_step is None:
            self._onset_step = frame.step
            return make_state(frame.step, True, self._onset_step, min(1.0, self._S / (2 * self.h)),
                              [self.signal], 0.0, self._S, f"CUSUM>{self.h} on {self.signal}")
        return make_state(frame.step, False, self._onset_step,
                          min(1.0, self._S / (2 * self.h)), [], 0.0, self._S, "")
