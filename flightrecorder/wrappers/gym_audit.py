"""RewardAuditWrapper: an opt-in, episode-level reward-audit wrapper for Gymnasium.

## Scope and boundary (read this before using or extending this module)

This wrapper records ONLY what the environment or caller explicitly exposes:

- episode-level reward statistics (mean, std, min, max) computed from the rewards
  ``env.step()`` actually returned -- nothing inferred from inside the environment;
- a caller-supplied reward-function version/digest, carried in ``ProducerRef.version``
  so records from different reward-function revisions are never silently conflated;
- readiness/derivation metadata for the one rolling signal this wrapper computes
  (reward drift vs. a reference window), so a cold-start "not enough episodes yet"
  state is never confused with a genuine zero-drift reading.

It does **not**, and structurally cannot, attribute *why* a reward changed, decompose
a reward into components, or claim any insight into the environment's internal reward
computation. A wrapper only ever sees the scalar `env.step()` returns -- claiming more
than that would mean silently assuming things about a closed environment's internals
that may not hold. If you need component-level attribution, that has to be built by
subclassing or instrumenting the environment itself, not by wrapping it from outside.

This boundary was worked out with a Gymnasium maintainer
(https://github.com/Farama-Foundation/Gymnasium/issues/1619) specifically to keep this
from overclaiming; see that thread for the full reasoning. This module is intentionally
NOT proposed as a built-in Gymnasium wrapper -- it lives here as a concrete, runnable
example first, per that discussion's conclusion.
"""

from __future__ import annotations

import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any

import gymnasium as gym
import numpy as np
from scipy.stats import wasserstein_distance

from flightrecorder.wrappers.schema import EpisodeRecord, ProducerRef, SignalDerivation, SignalValue

_WRAPPER_VERSION = "0.1.0"


class RewardAuditWrapper(gym.Wrapper):
    """Records one :class:`EpisodeRecord` per completed episode.

    Parameters:
        env: the Gymnasium environment to wrap.
        reward_fn_version: a caller-supplied string identifying the reward function's
            revision (e.g. a git SHA, a semantic version, or a hash of its config) --
            REQUIRED, not optional, because the whole point of ``ProducerRef.version``
            is to make cross-run audits possible; a wrapper that silently defaulted
            this to something meaningless would defeat that purpose.
        run_id: an identifier grouping records from one run (e.g. a training run ID).
            Defaults to a fresh UUID if not supplied.
        window_size: number of most-recent episodes' reward means used as the rolling
            comparison window for the drift signal. The first ``window_size`` episodes
            establish this window; before that, the drift signal reports
            ``state="unavailable"`` rather than a filler value.
        on_record: optional callback invoked with each :class:`EpisodeRecord` as it is
            emitted (e.g. to append it to a JSONL sink). If not supplied, records are
            only kept in ``self.records`` (in-memory), which is fine for the small
            runnable example this module ships with but not for a long training run.
    """

    def __init__(
        self,
        env: gym.Env,
        reward_fn_version: str,
        run_id: str | None = None,
        window_size: int = 20,
        on_record: Any | None = None,
    ) -> None:
        super().__init__(env)
        if not reward_fn_version:
            raise ValueError(
                "reward_fn_version is required -- this wrapper's entire purpose is "
                "cross-run audit via ProducerRef.version, which is meaningless if "
                "every run reports the same placeholder version."
            )
        self._reward_fn_version = reward_fn_version
        self._run_id = run_id or str(uuid.uuid4())
        self._window_size = window_size
        self._on_record = on_record

        self._episode_index = -1
        self._episode_rewards: list[float] = []
        self._episode_seed: int | None = None
        self._reference_window: deque[float] = deque(maxlen=window_size)
        self.records: list[EpisodeRecord] = []

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._episode_index += 1
        self._episode_rewards = []
        self._episode_seed = seed
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._episode_rewards.append(float(reward))
        if terminated or truncated:
            self._emit_episode_record()
        return obs, reward, terminated, truncated, info

    def _emit_episode_record(self) -> None:
        rewards = np.asarray(self._episode_rewards, dtype=np.float64)
        reward_mean = float(rewards.mean()) if rewards.size else 0.0

        drift_signal = self._compute_drift_signal(reward_mean)
        self._reference_window.append(reward_mean)

        signals = {
            "reward_mean": SignalValue(
                value=reward_mean,
                state="observed",
                derivation=SignalDerivation(method="instant"),
            ),
            "reward_std": SignalValue(
                value=float(rewards.std()) if rewards.size else 0.0,
                state="observed",
                derivation=SignalDerivation(method="instant"),
            ),
            "reward_drift": drift_signal,
        }

        record = EpisodeRecord(
            schema_version="1",
            record_id=f"{self._run_id}-ep{self._episode_index:06d}",
            record_kind="episode",
            emitted_at=datetime.now(timezone.utc),
            run_id=self._run_id,
            env_id=getattr(self.env.spec, "id", self.env.unwrapped.__class__.__name__),
            episode_uuid=str(uuid.uuid4()),
            episode_index=self._episode_index,
            seed=self._episode_seed,
            producer=ProducerRef(
                name="RewardAuditWrapper",
                version=_WRAPPER_VERSION,
                config={"reward_fn_version": self._reward_fn_version, "window_size": self._window_size},
            ),
            audit_status="complete",
            signals=signals,
        )
        self.records.append(record)
        if self._on_record is not None:
            self._on_record(record)

    def _compute_drift_signal(self, reward_mean: float) -> SignalValue:
        """Wasserstein distance between the current reference window and the window
        that would result from including this episode -- reports
        ``state="unavailable"`` (not a filler ``0.0``) until the reference window has
        filled, per this module's stated boundary on readiness metadata."""
        derivation = SignalDerivation(
            method="rolling",
            window_size=self._window_size,
            parameters={"metric": "wasserstein"},
        )
        if len(self._reference_window) < self._window_size:
            filled = len(self._reference_window)
            return SignalValue(
                value=None,
                state="unavailable",
                reason=f"reference window {filled}/{self._window_size}",
                derivation=derivation,
            )
        before = np.array(self._reference_window)
        after = np.array(list(self._reference_window)[1:] + [reward_mean])
        distance = float(wasserstein_distance(before, after))
        return SignalValue(value=distance, state="observed", derivation=derivation)
