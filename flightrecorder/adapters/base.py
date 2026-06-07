"""Adapter contract + shared helpers."""
from __future__ import annotations

from typing import Protocol
import numpy as np
from ..types import RolloutBatch


class Adapter(Protocol):
    """Single extension point. A concrete adapter (TRL/verl/OpenRLHF) maps trainer-native
    per-step data to a RolloutBatch and hands it to the recorder."""
    def to_batch(self, **trainer_data) -> RolloutBatch: ...


def group_normalized_advantages(rewards, group_size: int | None):
    """GRPO advantages: within each group of `group_size` completions, (r - mean)/std.

    Returns None if group_size is falsy or rewards don't divide evenly into groups."""
    r = np.asarray(rewards, float)
    if not group_size or r.size % group_size != 0:
        return None
    g = r.reshape(-1, group_size)
    mean = g.mean(axis=1, keepdims=True)
    std = g.std(axis=1, keepdims=True) + 1e-8
    return ((g - mean) / std).reshape(-1)
