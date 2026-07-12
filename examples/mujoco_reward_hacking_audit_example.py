"""Real-use-case example: RewardAuditWrapper against a genuine MuJoCo locomotion
reward hack, not a synthetic reward-scaling stand-in.

Run: python examples/mujoco_reward_hacking_audit_example.py

Requires: pip install -e ".[gym]" gymnasium[mujoco]

Per goktugozkanmd's suggestion on
https://github.com/Farama-Foundation/Gymnasium/issues/1619 -- "try it against a real
use case before turning this into an external-tools docs PR" -- this wraps a real
``HalfCheetah-v5`` Gymnasium/MuJoCo environment and drives it with two genuinely
different action policies:

1. A real alternating-leg bounding gait (back legs vs. front legs in antiphase) that
   runs upright for the whole episode -- ordinary, non-hacking locomotion.
2. The documented "flip-and-slide" MuJoCo exploit: a stronger wind-up gait, a brief
   synchronized-extensor flip impulse, then a low constant torque that lets the
   cheetah scoot along on its back using residual momentum and ground contact instead
   of running -- a real, widely-documented degenerate reward hack on this task family,
   not a contrived one built for this demo.

The environment's native reward function never changes between the two phases -- this
example deliberately keeps ``reward_fn_version`` constant throughout, because the point
being illustrated is exactly what a reward-audit wrapper is for: it cannot see *why*
behavior changed, only that the reward statistics it observes shift when the *policy*
starts exploiting the environment instead of solving it. That is a stronger, more
honest demonstration than a synthetic reward-function-version swap.

This script has no dependency on any other project's source code -- the gait/flip
parameters are reimplemented here directly from the publicly-documented HalfCheetah
qpos layout, not imported from anywhere.
"""

from __future__ import annotations

import numpy as np

import gymnasium as gym

from flightrecorder.wrappers import RewardAuditWrapper

# HalfCheetah-v5 qpos layout: [rootx, rootz, rooty, bthigh, bshin, bfoot, fthigh, fshin, ffoot]
_FLIP_PITCH_THRESHOLD = 1.0  # rad; beyond this the torso counts as "flipped"

_GAIT_OFFSETS = np.array([0.0, 0.0, 0.0, np.pi, np.pi, np.pi])  # back legs vs. front legs
_GAIT_AMPLITUDE = 0.25
_GAIT_PERIOD = 15.0

_WINDUP_AMPLITUDE = 0.5
_WINDUP_PERIOD = 20.0
_WINDUP_STEPS = 30
_FLIP_TICKS = 15
_FLIP_IMPULSE = np.full(6, 1.0)
_SLIDE_AMPLITUDE = 0.2

_HORIZON = 150
_LEGIT_EPISODES = 25
_HACKING_EPISODES = 25


def _bounding_gait_action(t: int) -> np.ndarray:
    phase = 2 * np.pi * t / _GAIT_PERIOD
    return _GAIT_AMPLITUDE * np.sin(phase + _GAIT_OFFSETS)


def _flip_and_slide_action(t: int) -> np.ndarray:
    if t < _WINDUP_STEPS:
        phase = 2 * np.pi * t / _WINDUP_PERIOD
        return _WINDUP_AMPLITUDE * np.sin(phase + _GAIT_OFFSETS)
    if t < _WINDUP_STEPS + _FLIP_TICKS:
        return _FLIP_IMPULSE
    return np.full(6, _SLIDE_AMPLITUDE)


def _run_episodes(env: RewardAuditWrapper, action_fn, n_episodes: int, seed_offset: int) -> None:
    for ep in range(n_episodes):
        env.reset(seed=seed_offset + ep)
        t = 0
        terminated = truncated = False
        while not (terminated or truncated):
            action = np.clip(action_fn(t), env.action_space.low, env.action_space.high)
            _obs, _reward, terminated, truncated, _info = env.step(action)
            t += 1


def main() -> None:
    # max_episode_steps=_HORIZON so Gymnasium's own TimeLimit wrapper truncates each
    # episode -- without it HalfCheetah-v5's default 1000-step limit means the wrapper
    # never sees an episode boundary within this demo's short runs.
    base_env = gym.make("HalfCheetah-v5", max_episode_steps=_HORIZON)
    audited_env = RewardAuditWrapper(
        base_env,
        reward_fn_version="halfcheetah-native-v1",  # unchanged throughout -- see module docstring
        run_id="mujoco-camping-real-use-case",
        window_size=10,
    )

    print("Phase 1: genuine bounding gait (upright locomotion, no exploit)")
    _run_episodes(audited_env, _bounding_gait_action, _LEGIT_EPISODES, seed_offset=0)

    print("Phase 2: flip-and-slide exploit (same reward function, hacking behavior)")
    _run_episodes(audited_env, _flip_and_slide_action, _HACKING_EPISODES, seed_offset=1000)

    audited_env.close()

    print(f"\nRecorded {len(audited_env.records)} episodes.\n")
    legit_means = [r.signals["reward_mean"].value for r in audited_env.records[:_LEGIT_EPISODES]]
    hacking_means = [r.signals["reward_mean"].value for r in audited_env.records[_LEGIT_EPISODES:]]
    print(f"Phase 1 (legit) reward_mean:   avg={np.mean(legit_means):.2f}  last={legit_means[-1]:.2f}")
    print(f"Phase 2 (hacking) reward_mean: avg={np.mean(hacking_means):.2f}  last={hacking_means[-1]:.2f}")

    print("\nFirst record after the phase switch (episode index {}):".format(_LEGIT_EPISODES))
    print(audited_env.records[_LEGIT_EPISODES])
    print("\nA record well into the hacking phase, drift signal should be a real, non-warmup value:")
    print(audited_env.records[-1])


if __name__ == "__main__":
    main()
