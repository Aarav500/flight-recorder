# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.11.0","transformers==5.19.0","trl==1.14.2","peft>=0.17","datasets>=3.0","accelerate>=1.0","numpy>=1.26"]
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# [tool.uv.sources]
# torch = [{ index = "pytorch-cu128" }]
# ///
"""THROWAWAY SPIKE. Symmetric agents with two fixes for gradient noise, the reason the symmetric gate failed.

At 8 samples per step an agent's two half-batches agreed at a cosine of 0.067 and the agents at 0.0002, so phi
sat at noise. Two ways to cut that noise, both logged here without changing training:
  --m 64      eight times the samples per step, which is the scale that matters for this failure. A bigger
              model doesn't help, because the noise comes from sampling, not from capacity.
  EMA         every step also keeps a running average of each half-batch gradient (beta 0.9, about 19 steps of
              memory), like the optimiser's own first moment, and compares those.
Each statistic comes in two forms. "all" uses every adapter weight. "B" uses only the lora_B weights. Their
gradient is G A^T, and A is common to both agents at the start and moves slowly, so B gradients share
coordinates across agents. lora_A gradients pass through each agent's own B, which drift apart. B is the gate's
primary measure, fixed before the run.

Microbatches of --chunk sequences keep the full-vocabulary log-softmax in memory at m=64. The loss is the same.
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import step0_llm as S  # noqa: E402
from train_sym import RULES, score_pairs  # noqa: E402
from train_two_agent import (BETA, LORA_R, SCORE_TIMEOUT, Team, advantages, direction_stats,  # noqa: E402
                             selfcheck, trainer_defaults)


class TeamMB(Team):
    def b_mask(self, agent):
        torch = self.torch
        if not hasattr(self, "_bmask"):
            self._bmask = {}
        if agent not in self._bmask:
            self._bmask[agent] = torch.cat([torch.full((p.numel(),), "lora_B" in n, dtype=torch.bool, device=self.device)
                                            for p, n in zip(self.params[agent], self.names[agent])])
        return self._bmask[agent]

    def update(self, agent, rows, adv, step, perm, chunk=4):
        """Team.update with microbatches. Same loss, same two half-batch gradients, one optimiser step."""
        torch = self.torch
        params, opt = self.params[agent], self.opt[agent]
        self.model.set_adapter(agent)
        self.model.train()
        opt.zero_grad(set_to_none=True)
        adv = torch.tensor(adv, dtype=torch.float32, device=self.device)
        tokens = sum(len(c) for _, c, _ in rows)
        h = len(rows) // 2
        cumulative, kl_sum = [], 0.0
        for half in (perm[:h], perm[h:]):
            for k in range(0, len(half), chunk):
                part = half[k:k + chunk]
                ids, attn, mask = self._pad([rows[i] for i in part])
                with torch.no_grad(), self.model.disable_adapter():
                    ref = self._logps(ids, attn)
                self.model.set_adapter(agent)
                lp = self._logps(ids, attn)
                kl = torch.exp(ref - lp) - (ref - lp) - 1
                pt = -adv[list(part), None] * lp + BETA * kl
                if self.cfg["loss_type"] == "dapo":
                    loss = (pt * mask).sum() / tokens
                elif self.cfg["loss_type"] == "grpo":
                    loss = ((pt * mask).sum(1) / mask.sum(1).clamp(min=1)).sum() / len(rows)
                else:
                    raise ValueError(self.cfg["loss_type"])
                loss.backward()
                kl_sum += float((kl.detach() * mask).sum())
            cumulative.append(torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                                         for p in params]).float().clone())
        ga, gb = cumulative[0], cumulative[1] - cumulative[0]
        torch.nn.utils.clip_grad_norm_(params, self.cfg["max_grad_norm"])
        for g in opt.param_groups:
            g["lr"] = self.lr0 * max(0.0, 1.0 - step / self.sched_steps)
        opt.step()
        return ga, gb, kl_sum / tokens


def prefixed(prefix, d):
    return {f"{prefix}{k}": v for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--team", choices=tuple(RULES), required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--sched-steps", type=int, default=400)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--ema-beta", type=float, default=0.9)
    ap.add_argument("--workers", type=int, default=max(2, cpu_count()))
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    task = S.HARDSET_BY_NAME[a.task].task
    cfg = trainer_defaults()
    rng = np.random.default_rng(a.seed)
    (out / "meta.json").write_text(json.dumps(dict(
        model=a.model, task=a.task, seed=a.seed, lr=a.lr, steps=a.steps, sched_steps=a.sched_steps, m=a.m,
        chunk=a.chunk, ema_beta=a.ema_beta, beta=BETA, lora_r=LORA_R, design="SYM-SCALE", team=a.team,
        score_timeout=SCORE_TIMEOUT, trainer=cfg), indent=1))

    with Pool(a.workers) as pool:  # before the model loads, so scoring workers inherit no GPU state
        team = TeamMB(a.model, a.seed, a.lr, a.sched_steps, cfg)
        print("META", json.dumps(dict(task=a.task, seed=a.seed, m=a.m, lr=a.lr, team=a.team, trainer=cfg)), flush=True)
        f_steps = open(out / "steps.jsonl", "a", buffering=1)
        f_comp = open(out / "completions.jsonl", "a", buffering=1)
        f_phi = open(out / "phi.jsonl", "a", buffering=1)
        checked, ema = False, None
        for step in range(a.steps):
            t0 = time.time()
            r1 = team.sample("a1", [task.prompt] * a.m)
            r2 = team.sample("a2", [task.prompt] * a.m)
            t1 = time.time()
            s = score_pairs(pool, task, [x[2] for x in r1], [y[2] for y in r2], a.team)
            t2 = time.time()
            adv = advantages(s["team"])
            perm = rng.permutation(a.m).tolist()
            varied = len(set(s["team"])) > 1
            before = ({n: p.detach().clone() for n, p in team.model.named_parameters()}
                      if a.selfcheck and varied and not checked else None)
            g1a, g1b, kl1 = team.update("a1", r1, adv, step, perm, a.chunk)
            g2a, g2b, kl2 = team.update("a2", r2, adv, step, perm, a.chunk)
            b1, b2 = team.b_mask("a1"), team.b_mask("a2")
            grads = (g1a, g1b, g2a, g2b)
            if ema is None:
                ema = [g.clone() for g in grads]
            else:
                for e, g in zip(ema, grads):
                    e.mul_(a.ema_beta).add_(g, alpha=1 - a.ema_beta)
            stats = direction_stats(*grads)
            stats.update(prefixed("B_", direction_stats(g1a[b1], g1b[b1], g2a[b2], g2b[b2])))
            stats.update(prefixed("ema_", direction_stats(*ema)))
            stats.update(prefixed("emaB_", direction_stats(ema[0][b1], ema[1][b1], ema[2][b2], ema[3][b2])))
            t3 = time.time()
            if before is not None:
                selfcheck(team, stats, kl1, kl2, before, step)
                checked = True
            rec = dict(step=step, proxy=s["team"], proxy_mean=float(np.mean(s["team"])),
                       oracle_mean=float(np.mean(s["h1"] + s["h2"])), v1=s["v1"], h1=s["h1"], v2=s["v2"], h2=s["h2"],
                       kl1=kl1, kl2=kl2, len1=float(np.mean([len(x[1]) for x in r1])),
                       len2=float(np.mean([len(x[1]) for x in r2])), t_gen=t1 - t0, t_score=t2 - t1, t_train=t3 - t2)
            f_steps.write(json.dumps(rec) + "\n")
            f_comp.write(json.dumps(dict(step=step, a1=[x[2] for x in r1], a2=[x[2] for x in r2])) + "\n")
            f_phi.write(json.dumps(dict(step=step, **stats)) + "\n")
            print(f"STEP {step} team {rec['proxy_mean']:.3f} kl {kl1:.5f} {kl2:.5f} "
                  f"gen {t1 - t0:.1f}s score {t2 - t1:.1f}s train {t3 - t2:.1f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
