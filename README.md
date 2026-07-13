# Flight Recorder

[![tests](https://github.com/Aarav500/flight-recorder/actions/workflows/tests.yml/badge.svg)](https://github.com/Aarav500/flight-recorder/actions/workflows/tests.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

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

## Testing

```bash
pytest tests/ -q
```

68 tests covering the recorder/detector pipeline, oracle-blindness (a dedicated test
asserts no oracle/ground-truth field can leak into the types the detector receives),
the synthetic harness, the TRL adapter, and the CLI. CI runs the suite on Python
3.10-3.12 on every push and pull request.

## Status

Core complete, harness-validated, real run launch-ready; **thesis: pending real run.**

License: Apache-2.0
