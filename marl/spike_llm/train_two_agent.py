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
"""THROWAWAY SPIKE. Two LoRA adapters on one frozen base, trained on one shared GRPO reward.

Design D3 from step0_llm.py: agent 1 writes the helper, agent 2 reads it and writes the main function, and
the assembled program is scored once. The advantage is common and each agent is updated against its own
log-probabilities (tex "Shared-Advantage MAGRPO"). Matches the single-agent 7B runs in group size 8, 256
tokens, beta 0.02, AdamW and linear decay. LoRA is the one difference, and it changes the optimisation
geometry, so these runs can't be compared with those run for run. The learning rate is set by KL calibration.

Three files come out:
  steps.jsonl        proxy, oracle, KL, timing. verdict_train.py reads this.
  completions.jsonl  every completion from both agents.
  phi.jsonl          SEALED. Gradient statistics, only read if the verdict says true quality fell.

    uv run train_two_agent.py --task csv_narrow --seed 300 --lr 1e-4 --out out/
    python train_two_agent.py --model Qwen/Qwen2.5-0.5B-Instruct --m 4 --steps 3 --selfcheck ...
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

M = 8
BETA = 0.02
LORA_R = 32
SCORE_TIMEOUT = 3.0
ADV_EPS = 1e-4


def trainer_defaults():
    """TRL's own defaults, read at run time. The fallback values are what the second screen recorded from
    TRL 1.14.2 (runs/llm2/result.json); it only applies to the local smoke test, where TRL isn't installed."""
    try:
        import dataclasses
        import trl
        d = {f.name: f.default for f in dataclasses.fields(trl.GRPOConfig)}
        keys = ("temperature", "top_p", "top_k", "repetition_penalty", "loss_type", "max_grad_norm",
                "weight_decay", "adam_beta1", "adam_beta2", "adam_epsilon")
        out = {k: d[k] for k in keys}
        out["source"] = "trl " + trl.__version__
        return out
    except ImportError:
        return dict(temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0, loss_type="dapo",
                    max_grad_norm=1.0, weight_decay=0.0, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,
                    source="fallback for local smoke test")


def _score_one(args):
    return S.score_with_status(args[0], args[1], args[2], SCORE_TIMEOUT)[0]


class Team:
    def __init__(self, model_id, seed, lr, sched_steps, cfg, base=None):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        self.torch, self.lr0, self.sched_steps, self.cfg = torch, lr, sched_steps, cfg
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        gpu = torch.cuda.is_available()
        if base is None:
            base = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch.bfloat16 if gpu else torch.float32, device_map="auto" if gpu else None)
        base.config.use_cache = False
        torch.manual_seed(seed)
        lora = dict(r=LORA_R, lora_alpha=2 * LORA_R, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                    target_modules="all-linear")
        # PEFT keeps adapter weights in float32 over a bf16 base. They must stay there: at bf16 precision
        # updates this small round to nothing on any weight of ordinary size.
        self.model = get_peft_model(base, LoraConfig(**lora), adapter_name="a1")
        self.model.add_adapter("a2", LoraConfig(**lora))
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        sd = dict(self.model.named_parameters())
        with torch.no_grad():   # common initialisation, the proposal's A2
            for n, p in sd.items():
                if ".a2." in n:
                    p.copy_(sd[n.replace(".a2.", ".a1.")])
        self.params = {a: [p for n, p in sd.items() if f".{a}." in n] for a in ("a1", "a2")}
        self.names = {a: [n for n in sd if f".{a}." in n] for a in ("a1", "a2")}
        self.device = self.params["a1"][0].device
        self.opt = {a: torch.optim.AdamW(self.params[a], lr=lr, weight_decay=cfg["weight_decay"],
                                         betas=(cfg["adam_beta1"], cfg["adam_beta2"]), eps=cfg["adam_epsilon"])
                    for a in ("a1", "a2")}
        self.gen = GenerationConfig(do_sample=True, max_new_tokens=S.MAX_NEW_TOKENS, temperature=cfg["temperature"],
                                    top_p=cfg["top_p"], top_k=cfg["top_k"] or 0,
                                    repetition_penalty=cfg["repetition_penalty"],
                                    pad_token_id=self.tok.pad_token_id, eos_token_id=self.tok.eos_token_id)

    def sample(self, agent, prompts):
        """-> list of (prompt_ids, completion_ids, text). completion_ids run through the first EOS."""
        torch = self.torch
        self.model.set_adapter(agent)
        self.model.eval()
        texts = [self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                              add_generation_prompt=True) for p in prompts]
        enc = self.tok(texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**enc, generation_config=self.gen, use_cache=True)
        plen = enc["input_ids"].shape[1]
        rows = []
        for i, row in enumerate(out):
            prompt_ids = enc["input_ids"][i][enc["attention_mask"][i].bool()].tolist()
            comp = row[plen:].tolist()
            if self.tok.eos_token_id in comp:
                comp = comp[:comp.index(self.tok.eos_token_id) + 1]
            rows.append((prompt_ids, comp, self.tok.decode(comp, skip_special_tokens=True)))
        return rows

    def _pad(self, rows):
        torch = self.torch
        T = max(len(p) + len(c) for p, c, _ in rows)
        ids = torch.full((len(rows), T), self.tok.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(rows), T), dtype=torch.long)
        mask = torch.zeros((len(rows), T), dtype=torch.bool)
        for i, (p, c, _) in enumerate(rows):
            n = len(p) + len(c)
            ids[i, :n] = torch.tensor(p + c)
            attn[i, :n] = 1
            mask[i, len(p):n] = True
        return ids.to(self.device), attn.to(self.device), mask[:, 1:].to(self.device)

    def _logps(self, ids, attn):
        logits = self.model(input_ids=ids, attention_mask=attn, use_cache=False).logits[:, :-1].float()
        if self.cfg["temperature"] != 1.0:
            logits = logits / self.cfg["temperature"]
        return self.torch.log_softmax(logits, -1).gather(2, ids[:, 1:, None]).squeeze(-1)

    def update(self, agent, rows, adv, step, perm):
        """Gradient on two disjoint halves of the group, then one optimiser step on the whole group.
        Returns half-batch gradients (for the sealed log) and the mean per-token KL against the base."""
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
            ids, attn, mask = self._pad([rows[i] for i in half])
            with torch.no_grad(), self.model.disable_adapter():
                ref = self._logps(ids, attn)
            self.model.set_adapter(agent)
            lp = self._logps(ids, attn)
            kl = torch.exp(ref - lp) - (ref - lp) - 1
            pt = -adv[list(half), None] * lp + BETA * kl
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


def direction_stats(g1a, g1b, g2a, g2b):
    """cross and within both see half-batch noise, so within - cross separates disagreement from noise."""
    import torch

    def cos(x, y):
        return float(torch.dot(x, y) / (x.norm() * y.norm() + 1e-12))
    g1, g2 = g1a + g1b, g2a + g2b
    c12 = cos(g1, g2)
    return dict(phi=(1 - c12) / 2, cos12=c12, within=(cos(g1a, g1b) + cos(g2a, g2b)) / 2,
                cross=(cos(g1a, g2b) + cos(g2a, g1b)) / 2, gnorm1=float(g1.norm()), gnorm2=float(g2.norm()))


def advantages(v):
    v = np.array(v, dtype=float)
    return ((v - v.mean()) / (v.std(ddof=1) + ADV_EPS)).tolist()


def selfcheck(team, stats, kl1, kl2, before, step):
    """Facts the loop depends on, checked on the first step whose group has any reward variation. With a
    constant group the advantage is zero, and at zero-initialised adapters so is the KL gradient, so no
    gradient at all is the correct answer there."""
    torch = team.torch
    if step == 0:
        assert abs(kl1) < 1e-3 and abs(kl2) < 1e-3, ("zero-initialised adapters must equal the base", kl1, kl2)
    assert stats["gnorm1"] > 0 and stats["gnorm2"] > 0, "no gradient reached an adapter"
    sd = dict(team.model.named_parameters())
    moved = {a: sum(float((sd[n] - before[n]).abs().sum()) for n in team.names[a]) for a in ("a1", "a2")}
    assert moved["a1"] > 0 and moved["a2"] > 0, moved
    for a in ("a1", "a2"):
        assert all(sd[n].dtype == torch.float32 for n in team.names[a]), "adapter weights left float32"
    base = [n for n in sd if ".a1." not in n and ".a2." not in n]
    assert all(bool((sd[n] == before[n]).all()) for n in base[:40]), "the base moved"
    print("SELFCHECK ok", json.dumps(dict(kl=[kl1, kl2], moved=moved, stats=stats)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--sched-steps", type=int, default=400, help="linear decay horizon, kept at 400 in calibration")
    ap.add_argument("--m", type=int, default=M)
    ap.add_argument("--workers", type=int, default=max(2, cpu_count()))
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    task = S.HARDSET_BY_NAME[a.task].task
    cfg = trainer_defaults()
    p1 = S.prompts(a.task, "D2")[0]
    rng = np.random.default_rng(a.seed)
    (out / "meta.json").write_text(json.dumps(dict(
        model=a.model, task=a.task, seed=a.seed, lr=a.lr, steps=a.steps, sched_steps=a.sched_steps, m=a.m,
        beta=BETA, lora_r=LORA_R, design="D3", score_timeout=SCORE_TIMEOUT, trainer=cfg), indent=1))

    with Pool(a.workers) as pool:  # before the model loads, so scoring workers inherit no GPU state
        team = Team(a.model, a.seed, a.lr, a.sched_steps, cfg)
        print("META", json.dumps(dict(task=a.task, seed=a.seed, lr=a.lr, trainer=cfg)), flush=True)
        f_steps = open(out / "steps.jsonl", "a", buffering=1)
        f_comp = open(out / "completions.jsonl", "a", buffering=1)
        f_phi = open(out / "phi.jsonl", "a", buffering=1)
        checked = False
        for step in range(a.steps):
            t0 = time.time()
            r1 = team.sample("a1", [p1] * a.m)
            p2 = [S.prompts(a.task, "D3", S.extract_code(x[2]).strip())[1] for x in r1]
            r2 = team.sample("a2", p2)
            t1 = time.time()
            codes = [S.assemble(x[2], y[2]) for x, y in zip(r1, r2)]
            uniq = sorted(set(codes))
            v = pool.map(_score_one, [(c, task.visible_tests, "test_visible.py") for c in uniq])
            h = pool.map(_score_one, [(c, task.hidden_tests, "test_hidden.py") for c in uniq])
            sc = {c: (x, y) for c, x, y in zip(uniq, v, h)}
            proxy = [sc[c][0] for c in codes]
            oracle = [sc[c][1] for c in codes]
            t2 = time.time()
            adv = advantages(proxy)
            perm = rng.permutation(a.m).tolist()
            varied = len(set(proxy)) > 1
            before = ({n: p.detach().clone() for n, p in team.model.named_parameters()}
                      if a.selfcheck and varied and not checked else None)
            g1a, g1b, kl1 = team.update("a1", r1, adv, step, perm)
            g2a, g2b, kl2 = team.update("a2", r2, adv, step, perm)
            stats = direction_stats(g1a, g1b, g2a, g2b)
            t3 = time.time()
            if before is not None:
                selfcheck(team, stats, kl1, kl2, before, step)
                checked = True
            rec = dict(step=step, proxy=proxy, oracle=oracle, proxy_mean=float(np.mean(proxy)),
                       oracle_mean=float(np.mean(oracle)), kl1=kl1, kl2=kl2,
                       len1=float(np.mean([len(x[1]) for x in r1])), len2=float(np.mean([len(x[1]) for x in r2])),
                       t_gen=t1 - t0, t_score=t2 - t1, t_train=t3 - t2)
            f_steps.write(json.dumps(rec) + "\n")
            f_comp.write(json.dumps(dict(step=step, a1=[x[2] for x in r1], a2=[x[2] for x in r2])) + "\n")
            f_phi.write(json.dumps(dict(step=step, **stats)) + "\n")
            print(f"STEP {step} proxy {rec['proxy_mean']:.3f} oracle {rec['oracle_mean']:.3f} kl {kl1:.5f} {kl2:.5f} "
                  f"gen {t1 - t0:.1f}s score {t2 - t1:.1f}s train {t3 - t2:.1f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
