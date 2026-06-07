"""GRPO advantage moments + Wasserstein drift vs a rolling baseline."""
from __future__ import annotations

from collections import deque
import numpy as np
from scipy.stats import skew, kurtosis, wasserstein_distance


class AdvantageExtractor:
    def __init__(self, window: int = 20):
        self.window = window
        self._hist: deque[np.ndarray] = deque(maxlen=window + 1)

    def update(self, advantages) -> dict:
        if advantages is None:
            return {"adv_var": float("nan"), "adv_skew": float("nan"),
                    "adv_kurtosis": float("nan"), "adv_drift": float("nan")}
        a = np.asarray(advantages, float)
        drift = 0.0
        if len(self._hist) >= self.window:
            baseline = self._hist[0]
            drift = float(wasserstein_distance(a, baseline))
        self._hist.append(a)
        return {"adv_var": float(np.var(a)),
                "adv_skew": float(skew(a)) if a.size > 2 else 0.0,
                "adv_kurtosis": float(kurtosis(a)) if a.size > 2 else 0.0,
                "adv_drift": drift}
