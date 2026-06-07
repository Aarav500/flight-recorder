"""Stateful helpers: EWMA smoothing with slope/accel, and online moments."""
from __future__ import annotations

import math


class Smoother:
    """EWMA value + first/second finite differences of the smoothed series."""

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.value: float = 0.0
        self.slope: float = 0.0
        self.accel: float = 0.0
        self._prev_value: float | None = None
        self._prev_slope: float | None = None
        self._init = False

    def update(self, x: float) -> "Smoother":
        if not self._init:
            self.value = x
            self._init = True
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        if self._prev_value is not None:
            new_slope = self.value - self._prev_value
            if self._prev_slope is not None:
                self.accel = new_slope - self._prev_slope
            self._prev_slope = new_slope
            self.slope = new_slope
        self._prev_value = self.value
        return self


class RunningMoments:
    """Welford online mean/variance with a z-score helper."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self._m2 = 0.0

    def update(self, x: float) -> "RunningMoments":
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self._m2 += delta * (x - self.mean)
        return self

    @property
    def var(self) -> float:
        return self._m2 / self.n if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.var)

    def zscore(self, x: float) -> float:
        s = self.std
        return 0.0 if s < 1e-12 else (x - self.mean) / s
