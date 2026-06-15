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


def wrap_reward_fns(coordinator: TRLCoordinator, train_fn, oracle_fn=None, seed_hack=None):
    """Return a TRL-compatible reward fn that also stashes rewards/completions.

    TRL calls reward_fn(prompts=..., completions=..., **kw) -> list[float]. We return the
    TRAIN rewards (the optimization signal) and separately capture ORACLE rewards, which
    are never returned to the trainer.

    seed_hack (optional, SHAKEOUT ONLY): inject a known test-overwrite completion so the
    capture -> label -> isolation -> report path is exercised on a known hack. Shape:
    {"from_call": int, "hack_text": str}. From that reward-call index onward, >=50%% of each
    step's completions are replaced by hack_text, scored as the (gameable) train reward, and
    that seeded reward is BOTH stashed for the recorder AND returned to the trainer. Returning
    it to the trainer gives the group a non-zero reward variance, so the policy actually moves
    and TRL logs a genuine non-zero, varying KL -- otherwise a tiny model that never earns
    reward produces zero gradient and a flat KL, which the capture path cannot fix. The ORACLE
    reward is never returned to the trainer. This verifies plumbing on real captured geometry;
    it does NOT claim the model emergently hacked (that is a full-run question)."""
    counter = {"calls": 0}

    def _score(fn, prompts, comps, kw):
        return [float(x) for x in fn(prompts=prompts, completions=comps, **kw)]

    def wrapped(prompts=None, completions=None, **kw):
        comps = list(completions or [])
        rec_comps = comps
        if seed_hack is not None and counter["calls"] >= seed_hack["from_call"]:
            rec_comps = list(comps)
            n = len(rec_comps)
            k = max(1, (n + 1) // 2)            # >= half, so behavioral_onset (frac .5) fires
            for j in range(n - k, n):
                rec_comps[j] = seed_hack["hack_text"]
        counter["calls"] += 1
        rec_train = _score(train_fn, prompts, rec_comps, kw)
        rec_oracle = _score(oracle_fn, prompts, rec_comps, kw) if oracle_fn is not None else None
        coordinator.stash(rec_comps, rec_train, rec_oracle)
        return rec_train                        # trainer optimizes this -> policy moves -> KL>0
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

    def __init__(self, recorder, coordinator: TRLCoordinator, group_size: int | None = None,
                 collector: list | None = None):
        self.recorder = recorder
        self.coord = coordinator
        self.group_size = group_size
        self.collector = collector  # optional list of StepRecord for the integrity report
        # HF Trainer appends a step's metrics to log_history AFTER on_step_end, so the
        # callback reads the *previous* step's scalars and step 1 has none. Carry the last
        # seen scalars forward (init 0.0) so kl/entropy are always finite (no NaN frame).
        self._last_logged = {"kl": 0.0, "entropy": 0.0}

    def on_step_end(self, args=None, state=None, control=None, **kwargs):
        completions, train, oracle = self.coord.drain()
        if train is None:
            return control
        step = int(getattr(state, "global_step", 0)) if state is not None else 0
        self._last_logged = {**self._last_logged, **_extract_logged(state)}
        batch = RolloutBatch(
            step=step,
            train_rewards=train,
            oracle_rewards=oracle,
            advantages=group_normalized_advantages(train, self.group_size),
            completions=completions,
            logged=dict(self._last_logged),
            meta={"group_size": self.group_size})
        rf, of = self.recorder.record(batch)
        if self.collector is not None:
            from ..repro.integrity import StepRecord
            self.collector.append(StepRecord(
                step=step, rollout=rf, oracle=of, completions=completions or [],
                train_rewards=train, oracle_rewards=oracle))
        return control

    def on_train_end(self, args=None, state=None, control=None, **kwargs):
        self.recorder.close()
        return control
