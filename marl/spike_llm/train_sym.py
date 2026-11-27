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
"""THROWAWAY SPIKE. Symmetric agents: two LoRA adapters each write the WHOLE function for the same prompt.

In train_two_agent.py (design D3) the agents wrote different text, a helper and a loop, and their gradients came
out unrelated (cosine 0.00-0.01 every window), which pins phi at 0.5. Here both write the same kind of text, so
their gradients can line up or drift apart. The team is paid once per pair of samples:
    mean   (v1 + v2) / 2   loose coupling: either agent's gain pays the team
    min    min(v1, v2)     tight coupling: the team is paid only when both pass
The advantage is shared and each agent updates against its own log-probabilities, as in train_two_agent.py.
Everything else is imported from train_two_agent.py unchanged, so the D3 runs stay reproducible.

    uv run train_sym.py --task csv_narrow --seed 901 --lr 3e-5 --steps 30 --team mean --out out/
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
from train_two_agent import (BETA, LORA_R, M, SCORE_TIMEOUT, Team, _score_one, advantages,  # noqa: E402
                             direction_stats, selfcheck, trainer_defaults)

RULES = {"mean": lambda a, b: (a + b) / 2, "min": min}


def score_pairs(pool, task, texts1, texts2, rule):
    """-> dict of per-agent proxy and oracle, and the team proxy per pair."""
    codes1 = [S.extract_code(t).strip() + "\n" for t in texts1]
    codes2 = [S.extract_code(t).strip() + "\n" for t in texts2]
    uniq = sorted(set(codes1 + codes2))
    v = pool.map(_score_one, [(c, task.visible_tests, "test_visible.py") for c in uniq])
    h = pool.map(_score_one, [(c, task.hidden_tests, "test_hidden.py") for c in uniq])
    sc = {c: (x, y) for c, x, y in zip(uniq, v, h)}
    v1, h1 = [sc[c][0] for c in codes1], [sc[c][1] for c in codes1]
    v2, h2 = [sc[c][0] for c in codes2], [sc[c][1] for c in codes2]
    team = [RULES[rule](a, b) for a, b in zip(v1, v2)]
    return dict(v1=v1, h1=h1, v2=v2, h2=h2, team=team)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--team", choices=tuple(RULES), required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--sched-steps", type=int, default=400)
    ap.add_argument("--m", type=int, default=M)
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
        beta=BETA, lora_r=LORA_R, design="SYM", team=a.team, score_timeout=SCORE_TIMEOUT, trainer=cfg), indent=1))

    with Pool(a.workers) as pool:  # before the model loads, so scoring workers inherit no GPU state
        team = Team(a.model, a.seed, a.lr, a.sched_steps, cfg)
        print("META", json.dumps(dict(task=a.task, seed=a.seed, lr=a.lr, team=a.team, trainer=cfg)), flush=True)
        f_steps = open(out / "steps.jsonl", "a", buffering=1)
        f_comp = open(out / "completions.jsonl", "a", buffering=1)
        f_phi = open(out / "phi.jsonl", "a", buffering=1)
        checked = False
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
            g1a, g1b, kl1 = team.update("a1", r1, adv, step, perm)
            g2a, g2b, kl2 = team.update("a2", r2, adv, step, perm)
            stats = direction_stats(g1a, g1b, g2a, g2b)
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
