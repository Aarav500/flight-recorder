"""Core data model. Split frames make oracle leakage a type error (spec §4)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Any
import numpy as np


@dataclass
class RolloutBatch:
    """Per-step rollout data pushed by a trainer adapter."""
    step: int
    train_rewards: np.ndarray                      # gameable reward, per sample
    meta: dict
    oracle_rewards: np.ndarray | None = None       # held-out true quality (eval only)
    advantages: np.ndarray | None = None           # GRPO group-normalized
    logprobs: np.ndarray | None = None             # policy logprobs (token-level)
    ref_logprobs: np.ndarray | None = None         # frozen-SFT-ckpt logprobs
    entropy: np.ndarray | None = None              # token entropy if provided
    completions: list[str] | None = None           # raw text for genstats/labeling
    logged: dict | None = None                     # trainer-logged scalars (kl, entropy)


@dataclass
class RolloutFrame:
    """The ONLY thing Detector.update() receives. No oracle fields exist here."""
    step: int
    kl_mean: float
    kl_slope: float
    kl_accel: float
    entropy_mean: float
    entropy_trend: float
    adv_var: float
    adv_skew: float
    adv_kurtosis: float
    adv_drift: float
    train_reward: float
    train_reward_slope: float
    gen_len_mean: float
    gen_len_var: float
    ngram_diversity: float
    logprob_concentration: float
    raw: dict = field(default_factory=dict)


@dataclass
class OracleFrame:
    """Evaluator-only ground truth. Never passed to a detector."""
    step: int
    oracle_reward: float
    oracle_gap: float
    scissors: float
    oracle_turn_step: int | None = None
    onset_behavioral: int | None = None
    onset_oracle: int | None = None
    synthetic_tstar: int | None = None


@dataclass
class DetectorState:
    step: int
    onset: bool
    onset_step: int | None
    score: float
    triggered_metrics: list[str]
    tube_distance: float
    cusum: float
    explanation: str


@dataclass
class Event:
    kind: Literal["frame", "oracle", "detector", "onset", "run_start", "run_end"]
    step: int
    payload: dict[str, Any]
    ts: float
