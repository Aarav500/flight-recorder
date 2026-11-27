"""The two-adapter loop on a tiny random Qwen2, on CPU, in seconds. Run with:
    HF_HUB_OFFLINE=1 uv run --no-project --python ../../.venv/Scripts/python.exe --with peft python test_team_tiny.py
"""
import numpy as np
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

import train_two_agent as T

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def logits(team, rows):
    ids, attn, _ = team._pad(rows)
    with torch.no_grad():
        return team.model(input_ids=ids, attention_mask=attn, use_cache=False).logits.clone()


def main():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    cfg_m = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=512)
    base = Qwen2ForCausalLM(cfg_m).float()
    cfg = T.trainer_defaults()
    team = T.Team(MODEL, 1, 1e-3, 400, cfg, base=base)
    team.gen.max_new_tokens = 8
    sd = lambda: dict(team.model.named_parameters())

    # common initialisation, adapters in float32
    p = sd()
    pairs = [(n, n.replace(".a1.", ".a2.")) for n in team.names["a1"]]
    assert all(torch.equal(p[a], p[b]) for a, b in pairs), "adapters must start identical"
    assert all(p[n].dtype == torch.float32 for n in team.names["a1"] + team.names["a2"])
    assert any("lora_B" in n and float(p[n].abs().sum()) == 0 for n in team.names["a1"]), "lora_B starts at zero"

    # sampling respects the requested adapter and returns aligned rows
    rows = team.sample("a1", ["write code"] * 4)
    assert len(rows) == 4 and all(len(c) >= 1 for _, c, _ in rows)
    rows2 = team.sample("a2", ["write more code", "x", "write more code", "x"])
    assert len(rows2) == 4
    tokens = sum(len(c) for _, c, _ in rows)

    # a zero advantage at zero-initialised adapters gives no gradient: last run's "failure" on a constant group
    g1a, g1b, kl = team.update("a1", rows, [0.0] * 4, 0, [0, 1, 2, 3])
    assert float(g1a.norm()) == 0 and float(g1b.norm()) == 0 and abs(kl) < 1e-5, (g1a.norm(), kl)
    assert all(torch.equal(sd()[n], p[n]) for n in team.names["a1"]), "a zero gradient must not move anything"

    # a real advantage reaches lora_B, and half gradients sum to the single-pass gradient
    adv = T.advantages([1.0, 0.0, 0.5, 0.0])
    before = {n: q.detach().clone() for n, q in team.model.named_parameters()}
    ga, gb, kl1 = team.update("a1", rows, adv, 0, [2, 0, 3, 1])
    assert float((ga + gb).norm()) > 0, "no gradient reached the adapter with a varied group"
    assert abs(kl1) < 1e-5, kl1
    after = sd()
    assert sum(float((after[n] - before[n]).abs().sum()) for n in team.names["a1"]) > 0, "a1 didn't move"
    assert all(torch.equal(after[n], before[n]) for n in team.names["a2"]), "updating a1 moved a2"
    base_names = [n for n in after if ".a1." not in n and ".a2." not in n]
    assert all(torch.equal(after[n], before[n]) for n in base_names), "the frozen base moved"

    # one pass over the whole group gives the same gradient as the two halves (dapo normalises by all tokens)
    team.model.set_adapter("a2")
    team.model.train()
    team.opt["a2"].zero_grad(set_to_none=True)
    ids, attn, mask = team._pad(rows)
    with torch.no_grad(), team.model.disable_adapter():
        ref = team._logps(ids, attn)
    team.model.set_adapter("a2")
    lp = team._logps(ids, attn)
    kl_t = torch.exp(ref - lp) - (ref - lp) - 1
    a_t = torch.tensor(adv, dtype=torch.float32)
    ((-a_t[:, None] * lp + T.BETA * kl_t) * mask).sum().div(tokens).backward()
    full = torch.cat([(q.grad if q.grad is not None else torch.zeros_like(q)).flatten() for q in team.params["a2"]]).clone()
    h1, h2, _ = team.update("a2", rows, adv, 0, [2, 0, 3, 1])
    assert float((h1 + h2 - full).norm()) < 1e-4 * max(float(full.norm()), 1e-9), "halves don't sum to the whole"

    # the adapter changes the policy, disabling it restores the base, and a2 is independent of a1
    team.model.set_adapter("a1")
    l_a1 = logits(team, rows)
    with team.model.disable_adapter():
        l_base = logits(team, rows)
    team.model.set_adapter("a1")
    assert float((l_a1 - l_base).abs().max()) > 0, "training a1 changed nothing"
    team.update("a1", rows, adv, 1, [0, 1, 2, 3])
    assert float(next(iter([team.opt["a1"].param_groups[0]["lr"]]))) < 1e-3, "the LR schedule didn't decay"

    # the sealed statistics
    s = T.direction_stats(ga, gb, h1, h2)
    # same init + same samples at zero-initialised adapters: identical gradients, so phi is 0 up to rounding
    assert -1e-5 <= s["phi"] <= 1.0 and s["gnorm1"] > 0 and s["gnorm2"] > 0, s
    s0 = T.direction_stats(ga, gb, ga, gb)
    assert abs(s0["phi"]) < 1e-5 and abs(s0["cos12"] - 1) < 1e-5, s0
    a = T.advantages([1.0, 0.0, 0.5, 0.0])
    assert abs(sum(a)) < 1e-9 and np.isclose(np.std(a, ddof=1), 1.0, atol=1e-3), a
    print("ok", {k: round(v, 4) for k, v in s.items()}, "loss", cfg["loss_type"])


if __name__ == "__main__":
    main()
