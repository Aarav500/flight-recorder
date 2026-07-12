"""Concrete, runnable example of RewardAuditWrapper against a real Gymnasium env.

Run: python examples/gym_reward_audit_example.py

This is deliberately small and un-fancy: it exists to prove the wrapper produces real,
inspectable records against a genuine environment, not synthetic data -- per the
"concrete runnable use case" step agreed on
https://github.com/Farama-Foundation/Gymnasium/issues/1619 before this project
proposes anything to Gymnasium's external-tools list.

Two runs are shown: a "stable" reward version and a "drifted" one (CartPole's reward
scaled down partway through, simulating a reward-function revision) so the rolling
drift signal has something real to detect once its reference window fills.
"""

from __future__ import annotations

import gymnasium as gym

from flightrecorder.wrappers import RewardAuditWrapper


class _ScaledRewardEnv(gym.Wrapper):
    """Wraps CartPole so its reward can be deliberately changed partway through a
    run -- used here only to give the drift signal something real to detect."""

    def __init__(self, env: gym.Env, scale_after_episode: int, scale: float):
        super().__init__(env)
        self._scale_after = scale_after_episode
        self._scale = scale
        self._episode_index = -1

    def reset(self, **kwargs):
        self._episode_index += 1
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self._episode_index >= self._scale_after:
            reward = reward * self._scale
        return obs, reward, terminated, truncated, info


def main() -> None:
    base_env = gym.make("CartPole-v1")
    scaled_env = _ScaledRewardEnv(base_env, scale_after_episode=30, scale=0.5)
    audited_env = RewardAuditWrapper(
        scaled_env,
        reward_fn_version="cartpole-native-v1",
        run_id="example-run-001",
        window_size=20,
    )

    rng_seed = 0
    for _ in range(50):
        audited_env.reset(seed=rng_seed)
        rng_seed += 1
        done = False
        while not done:
            action = audited_env.action_space.sample()
            _obs, _reward, terminated, truncated, _info = audited_env.step(action)
            done = terminated or truncated
    audited_env.close()

    print(f"Recorded {len(audited_env.records)} episodes.\n")
    print("First record (early, drift signal still warming up):")
    print(audited_env.records[0])
    print()
    print("A record right at the reward-scaling boundary (episode 30):")
    print(audited_env.records[30])
    print()
    print("A record well past it, drift signal should show a real, non-warmup value:")
    print(audited_env.records[40])


if __name__ == "__main__":
    main()
