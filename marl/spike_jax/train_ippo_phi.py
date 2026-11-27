"""
train_ippo_phi.py -- THROWAWAY SPIKE. Does a cross-agent gradient statistic move before true reward falls?

Copied from Desktop/AAAI/idea2_marl_benchmark/code_v2/train_ippo.py (study L). Training is unchanged:
the readout below uses its own random stream, so a run with the same seed and machine reproduces the
old run's rt and cluster series. That is checked in analyze_phi.py.

What is added:

1. Per-agent gradients under SHARED parameters. docs/marl_proposal_rigorous.tex A1 says sharing makes
   every agent's gradient the same vector. That holds for the team loss, not for each agent's own
   samples, and those are what this computes: same parameter space, so no coordinate-frame problem.
   It matters because only shared-parameter teams exploit the proxy (paper_v2 sec_learned_design).

2. Two readouts per update, both policy-loss only, taken before that update's SGD steps:
     gae  advantage from GAE, which is what the optimiser uses (carries critic drift)
     mc   reward-to-go minus the agent's batch mean, no critic (tex section whygrpo, reason 1)

3. A same-noise control. A batch is split in half per agent. `within` is the cosine between one agent's
   two halves; `cross` is the cosine between different agents' halves. Both see half-batch noise, so
   within - cross separates "agents disagree" from "gradients turned to noise as training converged",
   which phi alone can't do.

4. A matched control, cond=patched. The old `control` arm trains on a different reward (true reward),
   so any statistic can separate it from `exploit` from update 0 without hacking being involved. That
   is the same confound as the 0.5B judge in experiments/audit_baselines.py. `patched` keeps the
   exploit arm's proxy and removes only the flaw: each landmark pays only its nearest agent. The two
   rewards are identical on every state with no duplicate claim.

Run one job:
    python train_ippo_phi.py --n 4 --cond exploit --seed 0 --params shared
Output: results/phi/<params>/N<n>_<cond>_k<kappa>_s<seed>.json (skipped if it exists).
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import distrax
from flax.linen.initializers import constant, orthogonal

from coupling_wrapper import ProxyKappaSpreadMPE, PROXY_COVER_RADIUS, kappa_mix
from jaxmarl.wrappers.baselines import MPELogWrapper as LogWrapper

CLAIM_BONUS = 15.0
READOUT_SEED_OFFSET = 777001

CONFIG = dict(
    LR=2.5e-4, NUM_ENVS=32, NUM_STEPS=128, TOTAL_TIMESTEPS=6_000_000,
    UPDATE_EPOCHS=4, NUM_MINIBATCHES=4, GAMMA=0.99, GAE_LAMBDA=0.95, CLIP_EPS=0.2,
    ENT_COEF=0.01, VF_COEF=0.5, MAX_GRAD_NORM=0.5, ACTIVATION="tanh", ANNEAL_LR=True,
)
EVAL_EVERY = 5
NUM_EVAL_ENVS = 64
EVAL_STEPS = 25


class ExploitEnv(ProxyKappaSpreadMPE):
    """Trains on the kappa-mixed proxy with the strengthened claim bonus."""

    def proxy_rewards(self, state):
        agent_pos = state.p_pos[: self.num_agents]
        landmark_pos = state.p_pos[self.num_agents:]
        dists = jnp.linalg.norm(agent_pos[:, None, :] - landmark_pos[None, :, :], axis=-1)
        near = jnp.min(dists, axis=1)
        claimed = (near < PROXY_COVER_RADIUS).astype(jnp.float32)
        r = -near + CLAIM_BONUS * claimed
        return {a: r[i] for i, a in enumerate(self.agents)}


class PatchedEnv(ProxyKappaSpreadMPE):
    """ExploitEnv's proxy with the flaw removed: a landmark pays only its nearest agent. Equal to
    ExploitEnv's proxy on any state where no landmark is claimed twice."""

    def proxy_rewards(self, state):
        agent_pos = state.p_pos[: self.num_agents]
        landmark_pos = state.p_pos[self.num_agents:]
        dists = jnp.linalg.norm(agent_pos[:, None, :] - landmark_pos[None, :, :], axis=-1)  # [N, L]
        near = jnp.min(dists, axis=1)
        closest = jnp.argmin(dists, axis=0)                                                 # [L]
        covered = (jnp.min(dists, axis=0) < PROXY_COVER_RADIUS).astype(jnp.float32)         # [L]
        paid = jnp.zeros(self.num_agents).at[closest].add(covered)
        r = -near + CLAIM_BONUS * jnp.minimum(paid, 1.0)
        return {a: r[i] for i, a in enumerate(self.agents)}


class ControlEnv(ExploitEnv):
    """Same proxy is still logged, but the learner is trained on the kappa-mixed true reward."""

    def rewards(self, state):
        return kappa_mix(self.true_rewards(state), self.kappa)

    def step_env(self, key, state, actions):
        obs, state, _, dones, info = super().step_env(key, state, actions)
        return obs, state, info["reward_true_mixed"], dones, info


ENVS = dict(exploit=ExploitEnv, patched=PatchedEnv, control=ControlEnv)


class ActorCritic(nn.Module):
    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        act = nn.relu if self.activation == "relu" else nn.tanh
        a = act(nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x))
        a = act(nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(a))
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(a)
        c = act(nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x))
        c = act(nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(c))
        v = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(c)
        return logits, jnp.squeeze(v, axis=-1)


def stack_obs(obs, agents):
    return jnp.stack([obs[a] for a in agents])  # [N, E, obs_dim]


def direction_stats(ga, gb, n):
    """ga, gb: [N, D] half-batch gradients. phi uses the full-batch direction, (ga + gb) / 2."""
    unit = lambda g: g / (jnp.linalg.norm(g, axis=1, keepdims=True) + 1e-12)
    full = 0.5 * (ga + gb)
    ua, ub = unit(ga), unit(gb)
    cross_m = ua @ ub.T
    return dict(
        phi=1.0 - jnp.sum(unit(full).mean(0) ** 2),
        within=jnp.trace(cross_m) / n,
        cross=(cross_m.sum() - jnp.trace(cross_m)) / (n * (n - 1)),
        gnorm=jnp.linalg.norm(full, axis=1).mean(),
    )


def build(n, cond, kappa, seed, params_mode, cfg, common_init=False, agent_id=False):
    base = ENVS[cond](num_agents=n, kappa=kappa)
    env = LogWrapper(base)
    agents = env.agents
    E, T = cfg["NUM_ENVS"], cfg["NUM_STEPS"]
    num_updates = cfg["TOTAL_TIMESTEPS"] // T // E
    net = ActorCritic(env.action_space(agents[0]).n, activation=cfg["ACTIVATION"])
    indep = params_mode == "indep"

    def sched(count):
        frac = 1.0 - (count // (cfg["NUM_MINIBATCHES"] * cfg["UPDATE_EPOCHS"])) / num_updates
        return cfg["LR"] * frac

    tx = optax.chain(optax.clip_by_global_norm(cfg["MAX_GRAD_NORM"]),
                     optax.adam(learning_rate=sched if cfg["ANNEAL_LR"] else cfg["LR"], eps=1e-5))

    def obs_of(o):
        x = stack_obs(o, agents)                               # [N, E, obs]
        if not agent_id:
            return x
        ids = jnp.broadcast_to(jnp.eye(n)[:, None, :], x.shape[:2] + (n,))
        return jnp.concatenate([x, ids], axis=-1)

    rng = jax.random.PRNGKey(seed)
    rng, net_rng = jax.random.split(rng)
    x0 = jnp.zeros((env.observation_space(agents[0]).shape[0] + (n if agent_id else 0),))
    if indep:
        inits = jnp.stack([net_rng] * n) if common_init else jax.random.split(net_rng, n)
        params = jax.vmap(lambda r: net.init(r, x0))(inits)
        opt_state = jax.vmap(tx.init)(params)
    else:
        params = net.init(net_rng, x0)
        opt_state = tx.init(params)

    def apply(p, x):  # x: [N, B, obs] -> logits [N, B, A], v [N, B]
        if indep:
            return jax.vmap(net.apply)(p, x)
        lg, v = net.apply(p, x.reshape((-1, x.shape[-1])))
        return lg.reshape(x.shape[:2] + lg.shape[-1:]), v.reshape(x.shape[:2])

    rng, reset_rng = jax.random.split(rng)
    obs, env_state = jax.vmap(env.reset)(jax.random.split(reset_rng, E))

    def env_step(carry, _):
        params, env_state, last_obs, rng = carry
        ob = obs_of(last_obs)                                  # [N, E, obs]
        logits, value = apply(params, ob)
        rng, k1, k2 = jax.random.split(rng, 3)
        pi = distrax.Categorical(logits=logits)
        action = pi.sample(seed=k1)                            # [N, E]
        logp = pi.log_prob(action)
        act = {a: action[i] for i, a in enumerate(agents)}
        obs2, env_state, reward, done, _ = jax.vmap(env.step)(jax.random.split(k2, E), env_state, act)
        tr = dict(obs=ob, action=action, logp=logp, value=value,
                  reward=jnp.stack([reward[a] for a in agents]),
                  done=jnp.stack([done[a] for a in agents]).astype(jnp.float32))
        return (params, env_state, obs2, rng), tr

    def gae(tr, last_val):
        def f(carry, t):
            g, nv = carry
            delta = t["reward"] + cfg["GAMMA"] * nv * (1 - t["done"]) - t["value"]
            g = delta + cfg["GAMMA"] * cfg["GAE_LAMBDA"] * (1 - t["done"]) * g
            return (g, t["value"]), g
        _, adv = jax.lax.scan(f, (jnp.zeros_like(last_val), last_val), tr, reverse=True, unroll=8)
        return adv, adv + tr["value"]

    def reward_to_go(tr):
        """Critic-free: discounted return within the rollout, no bootstrap. [T, N, E]"""
        def f(g, t):
            g = t["reward"] + cfg["GAMMA"] * (1 - t["done"]) * g
            return g, g
        _, ret = jax.lax.scan(f, jnp.zeros_like(tr["reward"][0]),
                              dict(reward=tr["reward"], done=tr["done"]), reverse=True, unroll=8)
        return ret

    def loss_fn(p, mb):
        # mb arrays: [N, B, ...] for indep (p is one agent's params when vmapped, mb [B, ...])
        logits, value = (net.apply(p, mb["obs"]) if indep else apply(p, mb["obs"]))
        pi = distrax.Categorical(logits=logits)
        logp = pi.log_prob(mb["action"])
        vclip = mb["value"] + (value - mb["value"]).clip(-cfg["CLIP_EPS"], cfg["CLIP_EPS"])
        vloss = 0.5 * jnp.maximum(jnp.square(value - mb["targets"]), jnp.square(vclip - mb["targets"])).mean()
        ratio = jnp.exp(logp - mb["logp"])
        a = mb["adv"]
        a = (a - a.mean()) / (a.std() + 1e-8)
        aloss = -jnp.minimum(ratio * a, jnp.clip(ratio, 1 - cfg["CLIP_EPS"], 1 + cfg["CLIP_EPS"]) * a).mean()
        return aloss + cfg["VF_COEF"] * vloss - cfg["ENT_COEF"] * pi.entropy().mean()

    def sgd_step(p, o, mb):
        g = jax.grad(loss_fn)(p, mb)
        u, o = tx.update(g, o, p)
        return optax.apply_updates(p, u), o

    def pg_loss(p, ob, ac, a):
        logits, _ = net.apply(p, ob)
        return -(distrax.Categorical(logits=logits).log_prob(ac) * a).mean()

    def half_grads(p, ob, ac, adv, perm):
        """One agent's gradients on two disjoint halves of its own samples."""
        a = (adv - adv.mean()) / (adv.std() + 1e-8)
        h = perm.shape[0] // 2
        ia, ib = perm[:h], perm[h:2 * h]
        return (jax.grad(pg_loss)(p, ob[ia], ac[ia], a[ia]),
                jax.grad(pg_loss)(p, ob[ib], ac[ib], a[ib]))

    def flat(g):
        return jnp.concatenate([l.reshape(n, -1) for l in jax.tree.leaves(g)], axis=1)

    read_key = jax.random.PRNGKey(seed + READOUT_SEED_OFFSET)

    def readout(params, data, ret, u):
        perms = jax.vmap(lambda k: jax.random.permutation(k, T * E))(
            jax.random.split(jax.random.fold_in(read_key, u), n))
        grads = jax.vmap(half_grads, in_axes=(0 if indep else None, 0, 0, 0, 0))
        out = {}
        for name, adv in (("gae", data["adv"]), ("mc", ret)):
            ga, gb = grads(params, data["obs"], data["action"], adv, perms)
            for k, v in direction_stats(flat(ga), flat(gb), n).items():
                out[f"{k}_{name}"] = v
        return out

    def update(carry, u):
        params, opt_state, env_state, last_obs, rng = carry
        (params, env_state, last_obs, rng), tr = jax.lax.scan(
            env_step, (params, env_state, last_obs, rng), None, T)        # tr leaves [T, N, E, ...]
        _, last_val = apply(params, obs_of(last_obs))
        adv, targets = gae(tr, last_val)
        data = dict(obs=tr["obs"], action=tr["action"], logp=tr["logp"], value=tr["value"],
                    adv=adv, targets=targets)
        # -> [N, T*E, ...]
        data = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1).reshape((n, T * E) + x.shape[3:]), data)
        ret = jnp.swapaxes(reward_to_go(tr), 0, 1).reshape((n, T * E))
        read = readout(params, data, ret, u)

        def epoch(c, _):
            params, opt_state, rng = c
            rng, pk = jax.random.split(rng)
            perm = jax.random.permutation(pk, T * E)
            mbs = jax.tree.map(lambda x: jnp.take(x, perm, axis=1).reshape(
                (n, cfg["NUM_MINIBATCHES"], -1) + x.shape[2:]).swapaxes(0, 1), data)  # [M, N, B, ...]

            def mb_step(c2, mb):
                p, o = c2
                if indep:
                    p, o = jax.vmap(sgd_step)(p, o, mb)
                else:
                    p, o = sgd_step(p, o, mb)
                return (p, o), None

            (params, opt_state), _ = jax.lax.scan(mb_step, (params, opt_state), mbs)
            return (params, opt_state, rng), None

        (params, opt_state, rng), _ = jax.lax.scan(epoch, (params, opt_state, rng), None, cfg["UPDATE_EPOCHS"])
        return (params, opt_state, env_state, last_obs, rng), (tr["reward"].mean(), read)

    update_fn = jax.jit(update)

    def eval_fn(params, rng):
        rng, rk = jax.random.split(rng)
        obs, st = jax.vmap(env.reset)(jax.random.split(rk, NUM_EVAL_ENVS))

        def step(c, _):
            st, obs, rng = c
            logits, _ = apply(params, obs_of(obs))
            action = jnp.argmax(logits, axis=-1)
            act = {a: action[i] for i, a in enumerate(agents)}
            rng, k = jax.random.split(rng)
            obs, st, _, _, info = jax.vmap(env.step)(jax.random.split(k, NUM_EVAL_ENVS), st, act)
            pos = st.env_state.p_pos
            ag, lm = pos[:, :n], pos[:, n:]
            dl = jnp.linalg.norm(ag[:, :, None] - lm[:, None], axis=-1)            # [E, N, L]
            da = jnp.linalg.norm(ag[:, :, None] - ag[:, None], axis=-1)            # [E, N, N]
            d = dl.min(-1)
            nn_ = jnp.where(jnp.eye(n, dtype=bool)[None], jnp.inf, da).min(-1)
            claim = (dl < PROXY_COVER_RADIUS).astype(jnp.float32)                  # [E, N, L]
            cnt = claim.sum(1)                                                     # [E, L]
            dup = ((claim * (cnt[:, None, :] >= 2)).max(-1))                       # [E, N]
            iu = jnp.triu_indices(n, 1)
            out = dict(
                d=d, nn=nn_, dup=dup,
                rt=jnp.stack([info["reward_true"][a] for a in agents], -1),
                rp=jnp.stack([info["reward_proxy"][a] for a in agents], -1),
                cluster=(cnt >= 2).any(-1).astype(jnp.float32),
                team_std_d=d.std(-1),
                mean_pair=da[:, iu[0], iu[1]].mean(-1),
                uncovered=(cnt == 0).mean(-1),
            )
            return (st, obs, rng), out

        _, seq = jax.lax.scan(step, (st, obs, rng), None, EVAL_STEPS)
        per_agent = {k: seq[k].mean((0, 1)) for k in ("d", "nn", "dup", "rt", "rp")}
        team = {k: seq[k].mean() for k in ("cluster", "team_std_d", "mean_pair", "uncovered")}
        return per_agent, team

    return dict(env=env, update_fn=update_fn, eval_fn=jax.jit(eval_fn), params=params,
                opt_state=opt_state, env_state=env_state, obs=obs, rng=rng,
                num_updates=num_updates)


READ_KEYS = [f"{k}_{a}" for a in ("gae", "mc") for k in ("phi", "within", "cross", "gnorm")]


def run(n, cond, kappa, seed, params_mode, cfg, out_path, common_init=False, agent_id=False, log_every=250):
    t0 = time.time()
    c = build(n, cond, kappa, seed, params_mode, cfg, common_init, agent_id)
    carry = (c["params"], c["opt_state"], c["env_state"], c["obs"], c["rng"])
    hist = {k: [] for k in ("update", "train_reward", "d", "nn", "dup", "rt", "rp",
                            "cluster", "team_std_d", "mean_pair", "uncovered")}
    reads = {k: [] for k in READ_KEYS}
    erng = jax.random.PRNGKey(seed + 12345)
    for u in range(c["num_updates"]):
        carry, (rew, rd) = c["update_fn"](carry, u)
        for k in READ_KEYS:
            reads[k].append(round(float(rd[k]), 5))
        if u % EVAL_EVERY == 0 or u == c["num_updates"] - 1:
            erng, k = jax.random.split(erng)
            pa, tm = c["eval_fn"](carry[0], k)
            hist["update"].append(u)
            hist["train_reward"].append(float(rew))
            for key in ("d", "nn", "dup", "rt", "rp"):
                hist[key].append(np.asarray(pa[key]).round(5).tolist())
            for key in ("cluster", "team_std_d", "mean_pair", "uncovered"):
                hist[key].append(round(float(tm[key]), 5))
            if u % log_every == 0:
                print(f"[N={n} {cond} k={kappa} s={seed} {params_mode}] upd {u}/{c['num_updates']} "
                      f"cluster={float(tm['cluster']):.3f} uncovered={float(tm['uncovered']):.3f} "
                      f"rt={float(np.mean(pa['rt'])):.3f} rp={float(np.mean(pa['rp'])):.2f} "
                      f"phi_gae={reads['phi_gae'][-1]:.3f} ({time.time() - t0:.0f}s)", flush=True)
    hist.update(reads)
    hist.update(n=n, cond=cond, kappa=kappa, seed=seed, params=params_mode, common_init=common_init, agent_id=agent_id,
                num_updates=c["num_updates"], config=cfg, eval_every=EVAL_EVERY,
                num_eval_envs=NUM_EVAL_ENVS, eval_steps=EVAL_STEPS, claim_bonus=CLAIM_BONUS,
                readout_seed_offset=READOUT_SEED_OFFSET,
                wall_time_s=time.time() - t0, jax_version=jax.__version__)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(hist, f)
    os.replace(tmp, out_path)
    print(f"saved {out_path} ({hist['wall_time_s']:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--cond", choices=tuple(ENVS), required=True)
    ap.add_argument("--kappa", type=float, default=0.0)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--params", choices=("indep", "shared"), default="shared")
    ap.add_argument("--common-init", action="store_true", help="indep only: same initial weights per agent")
    ap.add_argument("--agent-id", action="store_true", help="append a one-hot agent id to each observation")
    ap.add_argument("--steps", type=int, default=CONFIG["TOTAL_TIMESTEPS"])
    ap.add_argument("--out", default=os.path.join(HERE, "results", "phi"))
    a = ap.parse_args()
    cfg = dict(CONFIG, TOTAL_TIMESTEPS=a.steps)
    d = os.path.join(a.out, a.params + ("_ci" if a.common_init else "") + ("_id" if a.agent_id else ""))
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"N{a.n}_{a.cond}_k{a.kappa}_s{a.seed}.json")
    if os.path.exists(path):
        print(f"exists, skipping: {path}")
        return
    try:  # several runners may share one job file; the first to lock a job runs it
        fd = os.open(path + ".lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        print(f"locked by another runner, skipping: {path}")
        return
    try:
        run(a.n, a.cond, a.kappa, a.seed, a.params, cfg, path, a.common_init, a.agent_id)
    finally:
        os.remove(path + ".lock")


if __name__ == "__main__":
    main()
