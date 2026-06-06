# Flight Recorder — Design Spec

**Date:** 2026-06-06
**Status:** Approved design, pre-implementation
**Author:** Aarav (with Claude)

## 1. Problem & Premise

RL post-training runs (GRPO/PPO on LLMs with verifiable rewards) can begin **reward
hacking** — the policy learns to game the reward signal rather than improve true
quality. By the time the *training reward curve* visibly misbehaves, the damage is
done; often the reward curve looks *great* precisely while the model is hacking.

**Flight Recorder** is a diagnostics layer that hooks into existing trainers
(TRL / verl / OpenRLHF) and streams the quantities that theory says predict trouble
*before* the reward curve looks wrong. Its differentiated contribution is **onset
detection**: catching the *moment* a run starts hacking, not diagnosing it
post-mortem.

We do not reinvent the trainer. We record, compute, detect, and visualize.

### V1 thesis to prove
On a GRPO run that learns the **test-overwriting hack** (model rewrites unit tests so
they pass), our detector fires **earlier** than the training-reward curve turns —
with measurable lead time, beating naive baselines.

## 2. Goals / Non-Goals

**Goals (V1)**
- A framework-agnostic Python core that computes 5 danger metrics from rollouts.
- A pluggable onset **Detector** interface + a principled default
  (`ContractionTubeDetector`) + two baselines for comparison.
- Trainer adapters for TRL (exercised), verl + OpenRLHF (interface-complete stubs).
- A reproducible, GPU-free **synthetic** demo with ground-truth onset that proves
  the detector fires before the reward curve, with lead-time + precision/recall.
- A fully-scaffolded **real** TRL+Qwen GRPO test-hack run, launchable when GPUs land.
- Two web surfaces: a **Live** streaming dashboard and a separate **Report** viewer.
- A FastAPI server: REST for runs + WebSocket for live streaming.

**Non-Goals (V1)**
- Running real GPU training this session (scaffolded, not executed).
- Hosting / deployment (explicitly the next prompt).
- Supporting PPO/DPO/reward-model training (GRPO-first; design stays general).
- Distributed multi-node recording (single-process recorder is enough for V1).

## 3. Architecture

```
trainer (TRL/verl/OpenRLHF)
   │  pushes RolloutBatch per step  (via Adapter)
   ▼
Recorder ──► MetricExtractors (5)  ──► MetricFrame
   │                                      │
   │                                      ▼
   │                                  Detector.update() ──► DetectorState (+ Onset event)
   ▼
EventSink(s):  JSONLSink (artifact) │ ConsoleSink │ WebSocketSink ──► FastAPI ──► Web (Live)
                                    JSONL artifact ──────────────► FastAPI ──► Web (Report)
```

- **Recorder** orchestrates: receives a `RolloutBatch`, runs all extractors into one
  `MetricFrame`, feeds the frame to the detector, emits an `Event` to every sink.
- **Sinks** are append-only and side-effect isolated. JSONL is the canonical run
  artifact; both UI surfaces ultimately read the same event schema.
- Everything is **pure + testable**: extractors and detectors are deterministic given
  their inputs; sinks are the only I/O.

## 4. Data Model

```python
@dataclass
class RolloutBatch:
    step: int
    # per-sample, length = num completions in the GRPO group(s)
    train_rewards: np.ndarray            # gameable reward (e.g. visible unit tests)
    oracle_rewards: np.ndarray | None    # held-out true-quality reward
    advantages: np.ndarray | None        # GRPO group-normalized advantages
    # per-token or per-sample policy stats (optional but preferred)
    logprobs: np.ndarray | None          # policy logprobs (token-level, ragged ok)
    ref_logprobs: np.ndarray | None      # reference-policy logprobs
    entropy: np.ndarray | None           # token-level entropy if trainer provides
    meta: dict                           # free-form (lr, kl_coef, group_size, ...)

@dataclass
class MetricFrame:
    step: int
    kl_mean: float
    kl_slope: float
    kl_accel: float
    entropy_mean: float
    entropy_trend: float                 # smoothed slope; negative = collapsing
    adv_var: float
    adv_skew: float
    adv_kurtosis: float
    adv_drift: float                     # Wasserstein vs rolling baseline
    train_reward: float
    oracle_reward: float | None
    oracle_gap: float | None             # train - oracle
    scissors: float                      # accumulated Δtrain - Δoracle (hero signal)
    raw: dict                            # any extra extractor outputs

@dataclass
class DetectorState:
    step: int
    onset: bool                          # True only on the step onset is first declared
    onset_step: int | None               # sticky: first step onset fired
    score: float                         # detector confidence / severity in [0,1]
    triggered_metrics: list[str]         # which signals corroborated
    tube_distance: float                 # signed distance outside the trajectory tube
    cusum: float                         # current CUSUM statistic on divergence
    explanation: str

@dataclass
class Event:                             # one JSONL line
    kind: Literal["frame", "detector", "onset", "run_start", "run_end"]
    step: int
    payload: dict                        # serialized MetricFrame / DetectorState / meta
    ts: float
```

## 5. The Five Metrics (`flightrecorder/core/`)

Each extractor is a pure function `(RolloutBatch, RollingState) -> dict`. Missing
inputs degrade gracefully (return `None`/NaN, never crash). Rolling state holds
windows for slopes/baselines.

1. **KL-vs-ref** (`kl.py`) — `kl_mean = mean(logprobs - ref_logprobs)` over tokens
   (k1 estimator; k3 `exp(r)-r-1` available as option). `kl_slope`/`kl_accel` = 1st/2nd
   finite differences of a smoothed `kl_mean` series. Acceleration is the leading edge.
2. **Entropy collapse** (`entropy.py`) — `entropy_mean` from token entropy if provided,
   else estimated from logprobs. `entropy_trend` = slope of an EWMA; sharp negative =
   collapse.
3. **Advantage drift** (`advantage.py`) — moments (`var/skew/kurtosis`) of GRPO
   group-normalized advantages, plus `adv_drift` = 1-D Wasserstein distance between the
   current advantage histogram and a rolling baseline window. Hacking tends to make
   advantages bimodal / heavy-tailed.
4. **Train-vs-oracle gap** (`oracle.py`) — `oracle_gap = train_reward - oracle_reward`.
   Widening gap = reward diverging from true quality.
5. **Reward–quality divergence** (`divergence.py`) — the **scissors index**:
   `scissors_t = scissors_{t-1} + (Δtrain_t - Δoracle_t)` with Δ = step-over-step change
   of EWMA-smoothed rewards. Climbs when reward rises while oracle stalls/falls. This is
   the hero signal and the primary detector input.

All five are assembled by the Recorder into one `MetricFrame` per step.

## 6. The Detector (`flightrecorder/detector/` — differentiated)

**Interface**
```python
class Detector(Protocol):
    def update(self, frame: MetricFrame) -> DetectorState: ...
    def reset(self) -> None: ...
```

**`ThresholdDetector`** (baseline) — fires when any single metric crosses a fixed
z-score threshold. Naive; high false-positive rate. Exists to be beaten.

**`CusumDetector`** (baseline) — one-sided CUSUM changepoint on the scissors signal:
`S_t = max(0, S_{t-1} + (x_t - μ_0 - k))`; fires when `S_t > h`. Strong, single-signal.

**`ContractionTubeDetector`** (default, the differentiated part) —
- Builds a **reference trajectory tube** from an early "healthy" window: the metric
  vector `m_t = [kl_accel, entropy_trend, adv_drift, scissors_slope]` is expected to
  *contract* toward a slowly-moving reference manifold. The tube is the region within
  radius `ρ_t` (robust scale, e.g. MAD-based) of that reference trajectory.
- Each step computes `tube_distance` = signed Mahalanobis-style distance of `m_t`
  outside the tube. Sustained positive distance = the trajectory is *escaping* the
  contraction region — the geometric signature of onset.
- **Corroboration:** declares onset at the first step where **`tube_distance > τ`
  AND** the internal CUSUM on the scissors signal exceeds `h` (two-of-N rule). This
  kills the false positives that sink single-signal detectors.
- Emits sticky `onset_step`, a `score` blending tube distance + CUSUM into [0,1],
  `triggered_metrics`, and a human-readable `explanation`.
- **Pluggability:** the user's exact contraction/trajectory-tube math replaces the
  `tube_distance` computation without touching the rest of the pipeline.

**Lead time** is computed against a reference "reward-curve turn" = the changepoint of
the *training reward* series found by the same CUSUM (offline). `lead = reward_turn_step
- onset_step`. Positive lead = we fired early. This is the headline number.

## 7. Adapters (`flightrecorder/adapters/`)

**Interface:** an adapter converts trainer-native per-step data into a `RolloutBatch`
and calls `recorder.record(batch)`.

- **TRL** (`trl.py`, exercised in V1) — a `transformers.TrainerCallback` capturing
  logged scalars on `on_log`/`on_step_end`, **plus** thin reward-function wrappers that
  capture per-sample `train_rewards` and `oracle_rewards` as TRL invokes them. Exposes
  `FlightRecorderCallback(recorder)` and `wrap_reward_fns(train_fn, oracle_fn)`.
- **verl** (`verl.py`) — same `RolloutBatch` mapping against verl's trainer callback /
  hook surface. Interface-complete; not exercised in V1.
- **OpenRLHF** (`openrlhf.py`) — same, against OpenRLHF's callback surface. Interface-
  complete; not exercised in V1.

## 8. Reproduction & Proof (`flightrecorder/repro/`)

- **`synthetic.py`** — generates GRPO-shaped metric trajectories with a **ground-truth
  onset step** `t*`. Before `t*`: healthy contraction, entropy high, scissors flat,
  train≈oracle. After `t*`: entropy collapses, KL accelerates, advantages drift,
  scissors opens while *training reward keeps rising* and oracle stalls. Seeded +
  parameterized (onset sharpness, noise, lead margin). Drives the GPU-free demo and the
  precision/recall + lead-time evaluation across many seeds.
- **`reward_testhack.py`** — the test-overwriting reward pair: `train_reward` = fraction
  of *visible* unit tests passing (gameable by rewriting the test file); `oracle_reward`
  = fraction of *held-out* tests passing (model cannot see/modify). The gap opens when
  the model learns to overwrite tests.
- **`trl_grpo_qwen.py`** — a complete, runnable GRPO + small-Qwen (e.g. Qwen2.5-0.5B/1.5B)
  math/code RLVR script wired through the TRL adapter and the test-hack reward, sized
  for AWS spot GPUs. Guarded so it imports without a GPU; **scaffolded, not run, in V1.**

**Evaluation (`flr eval`)** — over N seeded synthetic runs, report per-detector:
detection rate, false-positive rate, and **mean lead time** vs the reward-curve turn.
Acceptance: `ContractionTubeDetector` has positive mean lead and dominates both
baselines on lead-at-fixed-FPR.

## 9. Server (`server/`)

FastAPI app:
- `GET /api/runs` — list run artifacts (id, status, created, onset summary).
- `GET /api/runs/{id}` — full event stream (JSONL) for the Report viewer.
- `GET /api/runs/{id}/summary` — onset step, lead time, triggered metrics.
- `WS /api/runs/{id}/live` — pushes `Event`s as the recorder emits them (Live dashboard).
- Serves built web assets in production (single-container friendly for hosting prompt).

The `WebSocketSink` publishes to an in-process pub/sub the server subscribes to; for the
synthetic demo, a `flr serve --replay <artifact> --speed N` streams a saved run as if live.

## 10. Web UI (`web/`) — "best UI possible"

**Stack:** React + Vite + TypeScript + Tailwind; **uPlot** for fast streaming line
charts; WebSocket client for live, fetch for report. Dark **instrument-cluster /
cockpit** aesthetic (built via the frontend-design skill). Two *separate* surfaces.

**Live dashboard** (`/live/:runId`)
- Run-status header (step, status, KL coef, group size).
- Prominent **"ONSET DETECTED"** alert banner that arms at `onset_step` with lead-time.
- **Hero:** scissors chart — train vs oracle reward overlaid, divergence shaded.
- Five metric panels (KL+accel, entropy+trend, advantage moments+drift, oracle gap,
  scissors) as live sparklines/line charts.
- **Detector panel:** visualizes the trajectory tube, the live metric trajectory
  escaping it, and the CUSUM statistic crossing threshold.
- Event timeline/log.

**Report viewer** (`/report/:runId`) — separate surface. Loads a saved artifact; same
charts rendered static; **onset moment annotated** with a lead-time callout comparing
detector onset vs reward-curve turn vs baselines; exportable/shareable.

## 11. Repo Layout

```
flight-recorder/
  pyproject.toml            # package: flightrecorder, CLI: flr, Apache-2.0
  README.md
  flightrecorder/
    __init__.py
    recorder.py
    types.py                # dataclasses from §4
    core/                   # kl, entropy, advantage, oracle, divergence
    detector/               # base, threshold, cusum, contraction_tube
    sinks/                  # base, jsonl, console, websocket
    adapters/               # trl, verl, openrlhf
    repro/                  # synthetic, reward_testhack, trl_grpo_qwen, eval
    cli.py                  # flr synth | replay | serve | eval | demo
  server/                   # FastAPI app
  web/                      # React + Vite + TS + Tailwind app
  examples/                 # quickstart notebooks/scripts
  tests/                    # pytest: extractors, detectors, synthetic, adapters
  docs/
```

## 12. Stack & Dependencies

- **Core:** Python ≥3.10, numpy, scipy (Wasserstein/stats), pydantic or stdlib
  dataclasses. No torch dependency in the core (adapters import torch lazily).
- **Server:** fastapi, uvicorn, websockets.
- **Repro (real run, optional extra `[train]`):** trl, transformers, datasets,
  accelerate, torch — lazily imported, never required for the synthetic demo.
- **Web:** react, vite, typescript, tailwindcss, uplot.
- Packaged with `uv`; extras: `flightrecorder[server]`, `[train]`, `[all]`.

## 13. Testing

- **pytest** unit tests for every extractor (known-input → known-output), the three
  detectors (synthetic trajectories with known `t*`), the scissors accumulation, and
  the TRL adapter mapping (mocked trainer).
- **Property/eval test:** synthetic harness asserts `ContractionTubeDetector` mean lead
  > 0 and > baselines over a fixed seed set (the V1 thesis, encoded as a test).
- **Web:** typecheck + build; a smoke test that the Live dashboard renders a replayed
  artifact end-to-end.

## 14. V1 Acceptance Checklist

1. `flr synth` produces a seeded artifact with ground-truth onset.
2. `flr eval` shows tube detector positive mean lead, beating both baselines.
3. `flr serve --replay` streams that artifact; Live dashboard shows metrics + the
   ONSET banner firing before the reward curve turns.
4. Report viewer loads the artifact and annotates onset + lead time.
5. TRL adapter unit-tested against a mocked trainer; `trl_grpo_qwen.py` imports and is
   launch-ready (not run).
6. verl + OpenRLHF adapters present and interface-complete.
7. `pip install -e .` clean; tests green; README quickstart works.

## 15. Out of Scope / Next

- Hosting & deployment (next prompt) — design already container-friendly.
- Executing the real GPU run on AWS spot.
- PPO/DPO support, multi-node recording, auth, persistence beyond JSONL artifacts.
