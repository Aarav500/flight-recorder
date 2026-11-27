"""Pre-registered threshold config (GATE 1).

The detector thresholds live in configs/thresholds.yaml, frozen before any real-run data
exists. The eval and the real run read them through here and must never re-tune to a
run artifact. `check_threshold_provenance` enforces that with a loud warning.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import yaml


@dataclass
class Thresholds:
    tube_tau: float = 1.0
    cusum_h: float = 5.0
    persistence: int = 5
    rho: float = 3.0
    warmup: int = 30
    cusum_k: float = 0.5
    cusum_signal: str = "kl_accel"
    oracle_turn_k: float = 0.5
    oracle_turn_h: float = 3.0
    oracle_turn_warmup: int = 20
    source: str = "defaults"


def load_thresholds(path: str | None) -> Thresholds:
    """Load thresholds from a YAML file. Missing path -> built-in defaults (source noted)."""
    if not path or not os.path.exists(path):
        return Thresholds()
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    fields = set(Thresholds.__dataclass_fields__) - {"source"}
    keep = {k: data[k] for k in fields if k in data}
    return Thresholds(**keep, source=path)


def detector_from_thresholds(t: Thresholds):
    """Construct the default ContractionTubeDetector from pre-registered thresholds."""
    from .detector.contraction_tube import ContractionTubeDetector

    return ContractionTubeDetector(
        warmup=t.warmup, rho=t.rho, tau=t.tube_tau, k=t.cusum_k,
        h=t.cusum_h, cusum_signal=t.cusum_signal, persistence=t.persistence,
    )


def check_threshold_provenance(thresholds_path: str | None, artifact_path: str | None) -> list[str]:
    """GATE 1 enforcement. Returns a list of warnings (empty == clean)."""
    warns: list[str] = []
    if not thresholds_path or not os.path.exists(thresholds_path):
        warns.append(
            "THRESHOLDS CONFIG MISSING — detector thresholds are not pre-registered. "
            "Any reported lead time is not credible (GATE 1)."
        )
        return warns
    if artifact_path and os.path.exists(artifact_path):
        if os.path.getmtime(thresholds_path) > os.path.getmtime(artifact_path):
            warns.append(
                "THRESHOLDS CONFIG MODIFIED AFTER THE RUN ARTIFACT — thresholds may have been "
                "tuned to this run. Re-freeze the config and re-run before trusting results (GATE 1)."
            )
    return warns
