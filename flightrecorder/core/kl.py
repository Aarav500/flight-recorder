"""KL(policy || frozen-SFT-ref). k1 = mean(lp - rlp); k3 = exp(-d)-1+d, d=lp-rlp."""
from __future__ import annotations

import numpy as np
from .rolling import Smoother


class KLExtractor:
    def __init__(self, estimator: str = "k1", alpha: float = 0.3):
        assert estimator in ("k1", "k3")
        self.estimator = estimator
        self._s = Smoother(alpha=alpha)

    def update(self, logprobs, ref_logprobs) -> dict:
        if logprobs is None or ref_logprobs is None:
            return {"kl_mean": float("nan"), "kl_slope": 0.0, "kl_accel": 0.0}
        d = np.asarray(logprobs, float) - np.asarray(ref_logprobs, float)
        per_tok = d if self.estimator == "k1" else (np.exp(-d) - 1.0 + d)
        kl = float(np.mean(per_tok))
        self._s.update(kl)
        return {"kl_mean": kl, "kl_slope": self._s.slope, "kl_accel": self._s.accel}
