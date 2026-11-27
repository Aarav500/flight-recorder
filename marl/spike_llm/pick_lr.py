"""Picks the LoRA learning rate from the trcal runs, by how fast the policy moves and nothing else.

Rule, fixed before trcal ran. The earlier full-model 7B run (runs/prime/prime_hacking_max_last_wins_seed0.json)
sits at a mean per-token KL against the base of 0.074 over steps 25-30. Pick the rate whose mean KL over the same
steps, averaged across the two agents, is closest to that in log distance. If none is within a factor of 3, none
is picked and the caller decides. Proxy and oracle are not read. It's one yardstick from another task, so this
sets the scale and no more.

    python pick_lr.py ../runs/trcal/out [price_per_hour]
"""
import json
import math
import sys
from pathlib import Path

TARGET_KL = 0.0739
STEPS = (25, 31)
RUNS = 6
TRAIN_STEPS = 400


def main():
    root = Path(sys.argv[1])
    price = float(sys.argv[2]) if len(sys.argv) > 2 else 1.861
    best, secs = None, []
    print(f"target KL {TARGET_KL}")
    for d in sorted(root.glob("lr*")):
        rows = [json.loads(x) for x in (d / "steps.jsonl").read_text().splitlines() if x.strip()]
        window = rows[STEPS[0]:STEPS[1]]
        if not window:
            print(f"{d.name}: only {len(rows)} steps, skipped")
            continue
        kl = sum((r["kl1"] + r["kl2"]) / 2 for r in window) / len(window)
        tail = rows[3:]
        s = [sum(sum(r[k] for k in ("t_gen", "t_score", "t_train")) for r in tail) / len(tail)]
        secs += s
        dist = abs(math.log(kl / TARGET_KL)) if kl > 0 else float("inf")
        print(f"{d.name}: KL {kl:.4f} (factor {math.exp(dist):.2f} from target), {s[0]:.1f} s/step "
              f"(gen {sum(r['t_gen'] for r in tail) / len(tail):.1f}, score {sum(r['t_score'] for r in tail) / len(tail):.1f}, "
              f"train {sum(r['t_train'] for r in tail) / len(tail):.1f})")
        if best is None or dist < best[0]:
            best = (dist, d.name[2:], kl)
    if best is None:
        raise SystemExit("no usable runs")
    print(f"PICK {best[1]}" if math.exp(best[0]) <= 3 else f"NO PICK: closest is {best[1]}, outside a factor of 3")
    h = TRAIN_STEPS * sum(secs) / len(secs) / 3600 + 0.25
    print(f"each run about {h:.1f} h, {RUNS} runs about ${RUNS * h * price:.0f} at ${price}/h")


if __name__ == "__main__":
    main()
