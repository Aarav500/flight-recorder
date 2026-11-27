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
"""THROWAWAY SPIKE. One agent on the matched pair: does csv_narrow cost true quality without a second agent?

The two-agent runs (train_two_agent.py, design D3) ended with true quality 0.43 on csv_narrow against 0.83 on
csv_exchangeable, at the same proxy. This trains a single LoRA adapter on the plain task prompt, with everything
else as in those runs: same model, rank, learning rate, group size, beta, loss and schedule. Team from
train_two_agent.py builds two adapters; only a1 is used here, and a2 is never sampled or trained.

    uv run train_single.py --task csv_narrow --seed 300 --lr 3e-5 --out out/
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
from train_two_agent import BETA, LORA_R, M, SCORE_TIMEOUT, Team, _score_one, advantages, trainer_defaults  # noqa: E402


def score_single(pool, task, texts):
    codes = [S.extract_code(t).strip() + "\n" for t in texts]
    uniq = sorted(set(codes))
    v = pool.map(_score_one, [(c, task.visible_tests, "test_visible.py") for c in uniq])
    h = pool.map(_score_one, [(c, task.hidden_tests, "test_hidden.py") for c in uniq])
    sc = {c: (x, y) for c, x, y in zip(uniq, v, h)}
    return [sc[c][0] for c in codes], [sc[c][1] for c in codes]


def grad_stats(ga, gb):
    import torch
    cos = float(torch.dot(ga, gb) / (ga.norm() * gb.norm() + 1e-12))
    return dict(within=cos, gnorm=float((ga + gb).norm()))


def selfcheck_single(team, stats, kl, before, step):
    """train_two_agent.selfcheck, for one agent: a1 moves, a2 and the base stay put."""
    torch = team.torch
    if step == 0:
        assert abs(kl) < 1e-3, ("zero-initialised adapter must equal the base", kl)
    assert stats["gnorm"] > 0, "no gradient reached the adapter"
    sd = dict(team.model.named_parameters())
    moved = {a: sum(float((sd[n] - before[n]).abs().sum()) for n in team.names[a]) for a in ("a1", "a2")}
    assert moved["a1"] > 0 and moved["a2"] == 0, moved
    assert all(sd[n].dtype == torch.float32 for n in team.names["a1"]), "adapter weights left float32"
    base = [n for n in sd if ".a1." not in n and ".a2." not in n]
    assert all(bool((sd[n] == before[n]).all()) for n in base[:40]), "the base moved"
    print("SELFCHECK ok", json.dumps(dict(kl=kl, moved=moved, stats=stats)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
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
        beta=BETA, lora_r=LORA_R, design="SINGLE", score_timeout=SCORE_TIMEOUT, trainer=cfg), indent=1))

    with Pool(a.workers) as pool:  # before the model loads, so scoring workers inherit no GPU state
        team = Team(a.model, a.seed, a.lr, a.sched_steps, cfg)
        print("META", json.dumps(dict(task=a.task, seed=a.seed, lr=a.lr, trainer=cfg)), flush=True)
        f_steps = open(out / "steps.jsonl", "a", buffering=1)
        f_comp = open(out / "completions.jsonl", "a", buffering=1)
        f_grad = open(out / "grad.jsonl", "a", buffering=1)
        checked = False
        for step in range(a.steps):
            t0 = time.time()
            r = team.sample("a1", [task.prompt] * a.m)
            t1 = time.time()
            proxy, oracle = score_single(pool, task, [x[2] for x in r])
            t2 = time.time()
            adv = advantages(proxy)
            perm = rng.permutation(a.m).tolist()
            before = ({n: p.detach().clone() for n, p in team.model.named_parameters()}
                      if a.selfcheck and len(set(proxy)) > 1 and not checked else None)
            ga, gb, kl = team.update("a1", r, adv, step, perm)
            stats = grad_stats(ga, gb)
            t3 = time.time()
            if before is not None:
                selfcheck_single(team, stats, kl, before, step)
                checked = True
            rec = dict(step=step, proxy=proxy, oracle=oracle, proxy_mean=float(np.mean(proxy)),
                       oracle_mean=float(np.mean(oracle)), kl=kl, len=float(np.mean([len(x[1]) for x in r])),
                       t_gen=t1 - t0, t_score=t2 - t1, t_train=t3 - t2)
            f_steps.write(json.dumps(rec) + "\n")
            f_comp.write(json.dumps(dict(step=step, a1=[x[2] for x in r])) + "\n")
            f_grad.write(json.dumps(dict(step=step, **stats)) + "\n")
            print(f"STEP {step} kl {kl:.5f} gen {t1 - t0:.1f}s score {t2 - t1:.1f}s train {t3 - t2:.1f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
