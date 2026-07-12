# Flight Recorder

Reward-hacking **onset detection** for RL post-training (GRPO). A diagnostics layer that
hooks into existing trainers and streams the quantities theory says predict trouble
*before* the reward curve looks wrong — then flags the **moment** a run starts hacking.

The detector reads **rollout geometry only** — KL acceleration vs. the frozen reference,
entropy collapse, advantage-distribution drift — and is **oracle-blind by construction**:
the held-out oracle is split into a separate type the detector cannot receive, so oracle
leakage is a *type error*, not a discipline problem. The novel, falsifiable claim is that
rollout geometry fires *before* the oracle gap becomes visible.

## Install

```bash
uv venv && uv pip install -e ".[dev]"
```

## Quickstart (GPU-free harness validation)

```bash
flr synth --seed 1 --steps 200 --tstar 100 --out run.jsonl   # write a run artifact
flr eval --seeds 10 --steps 200 --tstar 100                  # compare detectors
```

`eval` reports, per detector, detection rate, **false-positive rate on hard negatives**,
and **mean lead time vs the oracle-gap turn**. On the synthetic suite the oracle-blind
ContractionTube detector leads the oracle-gap turn while holding FPR at 0, where the naive
threshold/CUSUM baselines false-positive on every transient alarm:

| detector | detection | FPR | mean lead |
|---|---|---|---|
| ContractionTube | 1.0 | 0.0 | +11.5 |
| CUSUM baseline | 1.0 | 1.0 | +19.1 |
| Threshold baseline | 1.0 | 1.0 | +15.5 |

The synthetic harness validates plumbing and tunes FPR; it is **not** evidence the signal
exists in real runs — the onset is authored, so detecting it is tautological. The thesis
is gated on a real TRL+Qwen GRPO run (see the spec, §14).

## How it fits together

```
trainer (TRL/verl/OpenRLHF)
   └─ RolloutBatch ─► Recorder ─► RolloutFrame ─► Detector ─► onset event ─► sinks (JSONL/WS)
                               └► OracleFrame  ─► Evaluator (lead time, ground-truth onset)
```

## Gym reward-audit wrapper (opt-in, separate from the detector above)

`flightrecorder.wrappers.RewardAuditWrapper` is a small, separate tool: an opt-in
Gymnasium environment wrapper that records episode-level reward statistics for
cross-run audit. It grew out of a discussion on
[Farama-Foundation/Gymnasium#1619](https://github.com/Farama-Foundation/Gymnasium/issues/1619)
about whether a general-purpose "reward-hacking audit wrapper" belonged in Gymnasium
core — the conclusion was **no**, and this module exists here instead, scoped narrowly:

**What it records**: episode-level reward mean/std (computed only from the scalar
`env.step()` rewards actually returned), a rolling reward-drift signal with an
explicit `state` field so a cold-start "not enough episodes yet" period is never
reported as a filler zero, and a caller-supplied reward-function version/digest
carried in `ProducerRef.config["reward_fn_version"]` (so records from different
reward-function revisions are never conflated). `ProducerRef.version` itself is the
wrapper's own version, not the reward function's — the two are kept separate on
purpose.

**What it deliberately does NOT do**: it does not, and structurally cannot, infer *why*
a reward changed or decompose a reward into components — a wrapper only ever sees the
scalar an environment returns. It also carries no ground-truth/oracle field of any
kind; if you want to compare an audited run against a held-out oracle signal, that
comparison has to happen in code you own, using data you supply out-of-band.

Install: `pip install -e ".[gym]"`. Runnable example:
`python examples/gym_reward_audit_example.py`. Tests: `pytest tests/test_gym_audit.py`.

This is a concrete, runnable case first — not yet proposed to Gymnasium's external-tools
list. See the issue thread above for why that ordering matters.

## Status

Core complete, harness-validated, real run launch-ready; **thesis: pending real run.**

License: Apache-2.0
