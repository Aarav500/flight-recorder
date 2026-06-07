"""Evaluator-only divergence signal. Output lands in OracleFrame, never RolloutFrame."""
from __future__ import annotations

from .rolling import Smoother


class OracleExtractor:
    def __init__(self, alpha: float = 0.3):
        self._train = Smoother(alpha=alpha)
        self._oracle = Smoother(alpha=alpha)
        self.scissors = 0.0

    def update(self, train_reward: float, oracle_reward: float) -> dict:
        self._train.update(train_reward)
        self._oracle.update(oracle_reward)
        self.scissors += self._train.slope - self._oracle.slope
        return {"oracle_reward": float(oracle_reward),
                "oracle_gap": float(train_reward - oracle_reward),
                "scissors": float(self.scissors)}
