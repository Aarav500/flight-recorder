"""Tests for RewardAuditWrapper."""

from __future__ import annotations

import pytest

pytest.importorskip("gymnasium")

import gymnasium as gym

from flightrecorder.wrappers import RewardAuditWrapper


def _make_env():
    return gym.make("CartPole-v1")


def test_requires_reward_fn_version():
    with pytest.raises(ValueError):
        RewardAuditWrapper(_make_env(), reward_fn_version="")


def test_emits_one_record_per_episode():
    env = RewardAuditWrapper(_make_env(), reward_fn_version="v1", window_size=5)
    for i in range(3):
        env.reset(seed=i)
        done = False
        while not done:
            _obs, _r, term, trunc, _info = env.step(env.action_space.sample())
            done = term or trunc
    env.close()
    assert len(env.records) == 3
    assert all(r.record_kind == "episode" for r in env.records)


def test_drift_signal_unavailable_until_window_fills():
    env = RewardAuditWrapper(_make_env(), reward_fn_version="v1", window_size=5)
    for i in range(7):
        env.reset(seed=i)
        done = False
        while not done:
            _obs, _r, term, trunc, _info = env.step(env.action_space.sample())
            done = term or trunc
    env.close()
    for r in env.records[:5]:
        assert r.signals["reward_drift"].state == "unavailable"
        assert r.signals["reward_drift"].value is None
    for r in env.records[5:]:
        assert r.signals["reward_drift"].state == "observed"
        assert r.signals["reward_drift"].value is not None


def test_producer_ref_carries_reward_fn_version():
    env = RewardAuditWrapper(_make_env(), reward_fn_version="my-reward-v3")
    env.reset(seed=0)
    done = False
    while not done:
        _obs, _r, term, trunc, _info = env.step(env.action_space.sample())
        done = term or trunc
    env.close()
    assert env.records[0].producer.config["reward_fn_version"] == "my-reward-v3"


def test_no_oracle_or_ground_truth_field_exists():
    """Oracle isolation: this schema must never grow a ground-truth field."""
    from flightrecorder.wrappers.schema import EpisodeRecord

    field_names = {f.name for f in EpisodeRecord.__dataclass_fields__.values()}
    for forbidden in ("oracle", "true_reward", "ground_truth", "target"):
        assert not any(forbidden in name for name in field_names), (
            f"EpisodeRecord must not carry an oracle/ground-truth field (found pattern "
            f"'{forbidden}' in {field_names})"
        )
