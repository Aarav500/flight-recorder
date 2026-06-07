"""TRL GRPO integration.

Two pieces that work together:
  - wrap_reward_fns(coord, train_fn, oracle_fn): wraps TRL reward functions so per-sample
    train rewards (returned to the trainer) and oracle rewards (captured for the evaluator
    ONLY, never returned) plus completions are stashed each time TRL scores a group.
  - FlightRecorderCallback(recorder, coord, group_size): a transformers TrainerCallback
    that on each step assembles a RolloutBatch (group-normalized advantages + logged
    KL/entropy scalars) and calls recorder.record(batch).

transformers is imported lazily/optionally so this module imports & unit-tests with no
torch/transformers installed.
"""
from __future__ import annotations

import numpy as np
from ..types import RolloutBatch
from .base import group_normalized_advantages

try:  # real base when available; stub otherwise so the module imports anywhere
    from transformers import TrainerCallback as _TrainerCallback
except Exception:  # pragma: no cover - exercised only without transformers
    class _TrainerCallback:  # noqa: D401
        pass

_KL_KEYS = ("kl", "objective/kl", "train/kl")
_ENTROPY_KEYS = ("entropy", "train/entropy", "objective/entropy")


class TRLCoordinator:
    """Buffer shared between the reward wrappers and the callback (one step's worth)."""

    def __init__(self):
        self._completions = None
        self._train = None
        self._oracle = None

    def stash(self, completions, train_rewards, oracle_rewards) -> None:
        self._completions = list(completions) if completions is not None else None
        self._train = np.asarray(train_rewards, float)
        self._oracle = (np.asarray(oracle_rewards, float)
                        if oracle_rewards is not None else None)

    def drain(self):
        c, t, o = self._completions, self._train, self._oracle
        self._completions = self._train = self._oracle = None
        return c, t, o


def wrap_reward_fns(coordinator: TRLCoordinator, train_fn, oracle_fn=None):
    """Return a TRL-compatible reward fn that also stashes rewards/completions.

    TRL calls reward_fn(prompts=..., completions=..., **kw) -> list[float]. We return the
    TRAIN rewards (the optimization signal) and separately capture ORACLE rewards, which
    are never returned to the trainer."""
    def wrapped(prompts=None, completions=None, **kw):
        train = [float(x) for x in train_fn(prompts=prompts, completions=completions, **kw)]
        oracle = None
        if oracle_fn is not None:
            oracle = [float(x) for x in oracle_fn(prompts=prompts, completions=completions, **kw)]
        coordinator.stash(completions, train, oracle)
        return train
    return wrapped


def _extract_logged(state) -> dict:
    logs: dict = {}
    history = getattr(state, "log_history", None) if state is not None else None
    if history:
        last = history[-1]
        for out_key, candidates in (("kl", _KL_KEYS), ("entropy", _ENTROPY_KEYS)):
            for c in candidates:
                if c in last and last[c] is not None:
                    logs[out_key] = float(last[c])
                    break
    return logs


class FlightRecorderCallback(_TrainerCallback):
    """Drop into TRL's GRPOTrainer(callbacks=[...]). Records one RolloutBatch per step."""

    def __init__(self, recorder, coordinator: TRLCoordinator, group_size: int | None = None):
        self.recorder = recorder
        self.coord = coordinator
        self.group_size = group_size

    def on_step_end(self, args=None, state=None, control=None, **kwargs):
        completions, train, oracle = self.coord.drain()
        if train is None:
            return control
        step = int(getattr(state, "global_step", 0)) if state is not None else 0
        batch = RolloutBatch(
            step=step,
            train_rewards=train,
            oracle_rewards=oracle,
            advantages=group_normalized_advantages(train, self.group_size),
            completions=completions,
            logged=_extract_logged(state),
            meta={"group_size": self.group_size})
        self.recorder.record(batch)
        return control

    def on_train_end(self, args=None, state=None, control=None, **kwargs):
        self.recorder.close()
        return control
