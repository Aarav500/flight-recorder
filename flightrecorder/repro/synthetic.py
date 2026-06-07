"""HARNESS VALIDATION ONLY (spec §8). Authored RolloutFrame/OracleFrame trajectories
with a ground-truth onset (or t*=inf hard negatives). NOT evidence the signal exists in
real runs -- the onset is authored, so detecting it is tautological. Use it to validate
plumbing and tune false-positive rate against hard negatives."""
from __future__ import annotations

import numpy as np
from ..types import RolloutFrame, OracleFrame
from ..core.oracle import OracleExtractor


def _logistic(t, t0, sharp):
    return 1.0 / (1.0 + np.exp(-sharp * (t - t0)))


def generate_run(seed: int, n_steps: int = 200, tstar: int | None = 100,
                 hard_negative: bool = False, sharpness: float = 0.3,
                 noise: float = 0.05) -> dict:
    """Returns {rollout_frames, oracle_frames, tstar, is_hard_negative}.

    Onset run: after t*, kl accelerates, entropy collapses, advantages drift, and the
    held-out oracle stalls/declines while training reward keeps rising.
    Hard negative: t*=inf, but a SHORT transient spike (~3 steps) makes geometry look
    alarming without any sustained break or oracle decline.
    """
    rng = np.random.default_rng(seed)
    oracle_ext = OracleExtractor(alpha=0.4)
    rollout_frames, oracle_frames = [], []
    spike_at = int(rng.integers(40, n_steps - 40)) if hard_negative else -1

    for t in range(n_steps):
        on = 0.0 if (hard_negative or tstar is None) else float(_logistic(t, tstar, sharpness))
        # short transient (3 steps) for hard negatives: spikes then gone
        spike = 1.2 * (0.6 ** (t - spike_at)) if (hard_negative and spike_at <= t < spike_at + 3) else 0.0

        progress = float(_logistic(t, n_steps * 0.4, 0.06))     # capability 0->1
        train_reward = 0.15 + 0.75 * progress + 0.05 * on + rng.normal(0, noise)
        if hard_negative:
            oracle_reward = 0.15 + 0.75 * progress + rng.normal(0, noise)   # tracks train
        else:
            oracle_reward = 0.15 + 0.75 * progress - on * 0.9 + rng.normal(0, noise)

        kl_accel = on * 2.5 + spike + rng.normal(0, noise)
        entropy_trend = -on * 1.2 + rng.normal(0, noise)
        adv_drift = on * 3.0 + spike + abs(rng.normal(0, noise))

        rf = RolloutFrame(
            step=t, kl_mean=float(on * 5 + spike), kl_slope=float(on),
            kl_accel=float(kl_accel), entropy_mean=float(3.0 - on * 2),
            entropy_trend=float(entropy_trend), adv_var=float(1 + on),
            adv_skew=float(on), adv_kurtosis=float(on * 2), adv_drift=float(adv_drift),
            train_reward=float(train_reward), train_reward_slope=0.0,
            gen_len_mean=float(50 - on * 20), gen_len_var=float(5 + on * 10),
            ngram_diversity=float(max(0.0, 0.8 - on * 0.5)),
            logprob_concentration=float(min(1.0, 0.3 + on * 0.6)))
        od = oracle_ext.update(float(train_reward), float(oracle_reward))
        of = OracleFrame(step=t, synthetic_tstar=(None if hard_negative else tstar), **od)
        rollout_frames.append(rf); oracle_frames.append(of)

    return {"rollout_frames": rollout_frames, "oracle_frames": oracle_frames,
            "tstar": (None if hard_negative else tstar),
            "is_hard_negative": hard_negative}
