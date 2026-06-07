"""Detector contract. update() takes RolloutFrame ONLY (oracle-blind by type)."""
from __future__ import annotations

from typing import Protocol, runtime_checkable
from ..types import RolloutFrame, DetectorState


@runtime_checkable
class Detector(Protocol):
    def update(self, frame: RolloutFrame) -> DetectorState: ...
    def reset(self) -> None: ...


# Geometry signals a detector may read (names on RolloutFrame). Never oracle fields.
GEOMETRY_SIGNALS = ("kl_accel", "entropy_trend", "adv_drift",
                    "gen_len_var", "logprob_concentration")


def make_state(step, onset, onset_step, score, triggered, tube_distance, cusum, why):
    return DetectorState(step=step, onset=onset, onset_step=onset_step,
                         score=float(score), triggered_metrics=list(triggered),
                         tube_distance=float(tube_distance), cusum=float(cusum),
                         explanation=why)
