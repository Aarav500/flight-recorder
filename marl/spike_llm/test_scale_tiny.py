"""Microbatched update matches the original, and the lora_B mask is right. Tiny random Qwen2 on CPU. Run with:
    HF_HUB_OFFLINE=1 uv run --no-project --python ../../.venv/Scripts/python.exe --with peft python test_scale_tiny.py
"""
import copy

import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

import gate_scale as G
import train_scale as TS
import train_two_agent as T

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def main():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    cfg_m = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=512)
    base0 = Qwen2ForCausalLM(cfg_m).float()
    cfg = T.trainer_defaults()
    ref = T.Team(MODEL, 3, 1e-3, 400, cfg, base=copy.deepcopy(base0))
    mb = TS.TeamMB(MODEL, 3, 1e-3, 400, cfg, base=copy.deepcopy(base0))
    for t in (ref, mb):
        t.gen.max_new_tokens = 8
    rows = ref.sample("a1", ["write code"] * 8)
    adv = T.advantages([1.0, 0.0, 0.5, 0.0, 1.0, 0.25, 0.0, 0.75])
    perm = [3, 1, 4, 0, 6, 2, 7, 5]
    ga, gb, kl = ref.update("a1", rows, adv, 0, perm)
    ha, hb, kl2 = mb.update("a1", rows, adv, 0, perm, chunk=1)
    tol = lambda x, y: float((x - y).norm()) <= 1e-4 * max(float(x.norm()), 1e-9)
    assert tol(ga, ha) and tol(gb, hb) and abs(kl - kl2) < 1e-6, "microbatching changed the gradient"
    sd_ref, sd_mb = dict(ref.model.named_parameters()), dict(mb.model.named_parameters())
    assert all(torch.allclose(sd_ref[n], sd_mb[n], atol=1e-6) for n in ref.names["a1"]), "optimiser steps differ"

    mask = mb.b_mask("a1")
    n_b = sum(p.numel() for p, n in zip(mb.params["a1"], mb.names["a1"]) if "lora_B" in n)
    assert int(mask.sum()) == n_b and mask.numel() == sum(p.numel() for p in mb.params["a1"])
    # at zero-initialised B, the A gradients are exactly zero, so all of the gradient sits in B
    rows2 = mb.sample("a2", ["write code"] * 8)
    a2a, a2b, _ = mb.update("a2", rows2, adv, 0, perm, chunk=3)
    m2 = mb.b_mask("a2")
    assert float(a2a[~m2].abs().sum()) == 0 and float(a2a[m2].norm()) > 0, "A gradient should be 0 at B = 0"

    s = TS.prefixed("B_", T.direction_stats(ha[mask], hb[mask], a2a[m2], a2b[m2]))
    assert set(s) == {"B_phi", "B_cos12", "B_within", "B_cross", "B_gnorm1", "B_gnorm2"}

    rows_g = [dict(emaB_within=0.4, emaB_cross=0.35, emaB_phi=0.31 + 0.001 * (i % 2)) for i in range(30)]
    assert G.gate(rows_g, "emaB_", 10, 30, 1, 15)["verdict"] == "PASS"
    rows_f = [dict(B_within=0.07, B_cross=0.0, B_phi=0.5) for i in range(30)]
    assert G.gate(rows_f, "B_", 5, 30, 10, 14)["verdict"] == "FAIL"
    print("ok")


if __name__ == "__main__":
    main()
