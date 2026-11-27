"""The patched proxy must equal the exploit proxy on duplicate-free states and pay less on duplicates."""
import jax
import jax.numpy as jnp

from train_ippo_phi import ExploitEnv, PatchedEnv, CLAIM_BONUS


class S:
    def __init__(self, p_pos):
        self.p_pos = jnp.array(p_pos, dtype=jnp.float32)


def rewards(env, pos):
    r = env.proxy_rewards(S(pos))
    return jnp.stack([r[a] for a in env.agents])


def main():
    ex, pa = ExploitEnv(num_agents=2), PatchedEnv(num_agents=2)
    # agents then landmarks
    no_claim = [[0.9, 0.9], [-0.9, -0.9], [0.0, 0.0], [0.5, 0.5]]
    one_each = [[0.0, 0.0], [0.5, 0.5], [0.05, 0.0], [0.5, 0.45]]
    both_on_one = [[0.0, 0.0], [0.02, 0.0], [0.03, 0.0], [0.9, 0.9]]
    assert jnp.allclose(rewards(ex, no_claim), rewards(pa, no_claim))
    assert jnp.allclose(rewards(ex, one_each), rewards(pa, one_each))
    re, rp = rewards(ex, both_on_one), rewards(pa, both_on_one)
    assert bool((re - rp).max() > CLAIM_BONUS * 0.99), (re, rp)
    assert float(re.sum() - rp.sum()) > CLAIM_BONUS * 0.99, (re, rp)

    # random duplicate-free states at N=4: the two rewards must match exactly
    env4e, env4p = ExploitEnv(num_agents=4), PatchedEnv(num_agents=4)
    checked = 0
    for s in range(400):
        pos = jax.random.uniform(jax.random.PRNGKey(s), (8, 2), minval=-1.0, maxval=1.0)
        d = jnp.linalg.norm(pos[:4, None] - pos[4:][None], axis=-1)
        if bool(((d < 0.2).sum(0) >= 2).any()):
            continue
        checked += 1
        assert jnp.allclose(rewards(env4e, pos), rewards(env4p, pos))
    assert checked > 100, checked
    print(f"ok: {checked} random duplicate-free states match")


if __name__ == "__main__":
    main()
