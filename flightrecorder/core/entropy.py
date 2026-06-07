"""Policy entropy + collapse trend. Falls back to mean surprisal if no entropy."""
from __future__ import annotations

import numpy as np
from .rolling import Smoother


class EntropyExtractor:
    def __init__(self, alpha: float = 0.3):
        self._s = Smoother(alpha=alpha)

    def update(self, entropy, logprobs) -> dict:
        if entropy is not None:
            h = float(np.mean(np.asarray(entropy, float)))
        elif logprobs is not None:
            h = float(np.mean(-np.asarray(logprobs, float)))  # surprisal proxy
        else:
            return {"entropy_mean": float("nan"), "entropy_trend": 0.0}
        return self.update_scalar(h)

    def update_scalar(self, entropy_mean: float) -> dict:
        """Feed a trainer-logged entropy scalar directly."""
        h = float(entropy_mean)
        self._s.update(h)
        return {"entropy_mean": h, "entropy_trend": self._s.slope}
