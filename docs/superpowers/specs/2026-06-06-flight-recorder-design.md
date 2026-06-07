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
they pass), our **oracle-blind** detector — reading rollout geometry only — fires
**earlier** than the held-out oracle gap becomes visible (the **oracle-gap turn**),
with measurable lead time, beating an oracle-watching practitioner baseline. The
training reward looks fine throughout, which is exactly why it cannot be the reference.

## 2. Goals / Non-Goals

**Goals (V1)**
- A framework-agnostic Python core that computes danger metrics from rollouts: five
  detector-visible **rollout-geometry** metrics + one **oracle** (ground-truth) signal,
  physically separated so the oracle can never reach a detector.
- A pluggable **oracle-blind** onset **Detector** interface + a principled default
  (`ContractionTubeDetector`) + two oracle-blind baselines for comparison.
- Trainer adapters for TRL (exercised), verl + OpenRLHF (interface-complete stubs).
- A reproducible, GPU-free **synthetic harness-validation** demo: confirms the pipeline
  computes metrics and the detector fires on authored trajectories, and tunes the
  false-positive rate against hard negatives. It is **not** evidence the signal exists
  in real runs.
- A fully-scaffolded **real** TRL+Qwen GRPO test-hack run, launchable when GPUs land —
  the run that actually tests the thesis.
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
Recorder ─► RolloutExtractors (5) ─► RolloutFrame ─► Detector.update() ─► DetectorState
   │        OracleExtractor       ─► OracleFrame  ─► Evaluator (lead time, onset labels)
   │                                                 evaluator-only; NEVER a detector input
   ▼
EventSink(s):  JSONLSink (artifact) │ ConsoleSink │ WebSocketSink ──► FastAPI ──► Web (Live)
                                    JSONL artifact ──────────────► FastAPI ──► Web (Report)
```

- **Recorder** orchestrates: receives a `RolloutBatch`, runs the rollout extractors into
  a `RolloutFrame` and the oracle extractor into an `OracleFrame`, feeds **only the
  `RolloutFrame`** to the detector, hands the `OracleFrame` to the evaluator, and emits
  an `Event` to every sink.
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
    ref_logprobs: np.ndarray | None      # reference-policy logprobs (frozen SFT ckpt)
    entropy: np.ndarray | None           # token-level entropy if trainer provides
    completions: list[str] | None        # raw completions (for genstats + onset label)
    meta: dict                           # free-form (lr, kl_coef, group_size, ...)

@dataclass
class RolloutFrame:          # the ONLY thing Detector.update() receives
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
    train_reward: float                  # gameable reward — deployment-available, legal
    train_reward_slope: float
    gen_len_mean: float                  # generation-stats extractor (corroboration only)
    gen_len_var: float
    ngram_diversity: float
    logprob_concentration: float
    raw: dict

@dataclass
class OracleFrame:           # evaluator ONLY — never passed to a detector
    step: int
    oracle_reward: float
    oracle_gap: float                    # train - oracle
    scissors: float                      # ground-truth divergence signal
    oracle_turn_step: int | None
    onset_behavioral: int | None         # from static analysis of completions
    onset_oracle: int | None             # from passes_visible ∧ ¬passes_held_out
    synthetic_tstar: int | None          # synthetic only

@dataclass
class DetectorState:
    step: int
    onset: bool                          # True only on the step onset is first declared
    onset_step: int | None               # sticky: first step onset fired
    score: float                         # detector confidence / severity in [0,1]
    triggered_metrics: list[str]         # which signals corroborated
    tube_distance: float                 # signed distance outside the trajectory tube
    cusum: float                         # current CUSUM statistic on a GEOMETRY signal
    explanation: str

@dataclass
class Event:                             # one JSONL line
    kind: Literal["frame", "oracle", "detector", "onset", "run_start", "run_end"]
    step: int
    payload: dict                        # serialized RolloutFrame / OracleFrame / DetectorState / meta
    ts: float
```

## 5. The Metrics (`flightrecorder/core/`)

Each extractor is a pure function `(RolloutBatch, RollingState) -> dict`. Missing
inputs degrade gracefully (return `None`/NaN, never crash). Rolling state holds windows
for slopes/baselines. **Extractors are split by who may see them:** five
**rollout-geometry** extractors feed the `RolloutFrame` (detector-visible,
deployment-available); the **oracle** extractor feeds the `OracleFrame` (evaluator-only,
never an input to any detector).

**Rollout-geometry extractors (detector-visible):**

1. **KL-vs-ref** (`kl.py`) — `kl_mean = mean(logprobs - ref_logprobs)` over tokens
   (k1 estimator; k3 `exp(r)-r-1` available as option). KL is computed against the
   **frozen SFT checkpoint for the whole run** (not a moving reference) — that is what
   makes `kl_accel` interpretable and deployment-available. `kl_slope`/`kl_accel` =
   1st/2nd finite differences of a smoothed `kl_mean` series; acceleration is the
   leading edge.
2. **Entropy collapse** (`entropy.py`) — `entropy_mean` from token entropy if provided,
   else estimated from logprobs. `entropy_trend` = slope of an EWMA; sharp negative =
   collapse.
3. **Advantage drift** (`advantage.py`) — moments (`var/skew/kurtosis`) of GRPO
   group-normalized advantages (advantages come from the **gameable train reward**,
   which is deployment-available — so they are legal detector input), plus `adv_drift`
   = 1-D Wasserstein distance between the current advantage histogram and a rolling
   baseline. Hacking tends to make advantages bimodal / heavy-tailed.
4. **Train reward** (passthrough) — `train_reward` and `train_reward_slope` from the
   batch (gameable, deployment-available; legal but **weak** — it looks fine while
   hacking, per §1).
5. **Generation stats** (`genstats.py`) — `gen_len_mean`, `gen_len_var`,
   `ngram_diversity`, `logprob_concentration` from completions alone (no oracle).
   **Corroboration-only: may never fire onset by itself**, because a benign capability
   gain (the model converging on a tight canonical solution) moves these the same
   direction as a hack.

**Oracle extractor (evaluator-only — never reaches a detector):**

6. **Reward–quality divergence** (`oracle.py`) — computes `oracle_gap = train_reward −
   oracle_reward` and the **scissors index** `scissors_t = scissors_{t-1} + (Δtrain_t −
   Δoracle_t)` (Δ = step-over-step change of EWMA-smoothed rewards). **Scissors is a
   ground-truth divergence signal, computed evaluator-side, and is never an input to any
   detector.** It lands in `OracleFrame`, not `RolloutFrame`.

The rollout extractors assemble a `RolloutFrame` per step (the only thing a detector
sees); the oracle extractor assembles an `OracleFrame` consumed only by the evaluator.

## 6. The Detector (`flightrecorder/detector/` — differentiated)

**Interface**
```python
class Detector(Protocol):
    def update(self, frame: RolloutFrame) -> DetectorState: ...   # oracle-blind by type
    def reset(self) -> None: ...
```

Both baselines are **oracle-blind** — they run on rollout-geometry signals, never on
scissors. (A baseline reading scissors would be the same circular detector, not a fair
comparison.)

**`ThresholdDetector`** (baseline) — fires when any single geometry signal crosses a
fixed z-score threshold. Naive; high false-positive rate. Exists to be beaten.

**`CusumDetector`** (baseline) — one-sided CUSUM changepoint on a single geometry signal
(default `kl_accel`): `S_t = max(0, S_{t-1} + (x_t - μ_0 - k))`; fires when `S_t > h`.
Strong, single-signal.

**`ContractionTubeDetector`** (default, the differentiated part) —
- Builds a **reference trajectory tube** from the robust-verifier reference run (see
  below): the **oracle-blind** metric vector `m_t = [kl_accel, entropy_trend, adv_drift]`
  is expected to *contract* toward a slowly-moving reference manifold. The tube is the
  region within radius `ρ_t` (robust scale, e.g. MAD-based) of that reference trajectory.
- Each step computes `tube_distance` = signed Mahalanobis-style distance of `m_t`
  outside the tube. Sustained positive distance = the trajectory is *escaping* the
  contraction region — the geometric signature of onset.
- **Corroboration:** declares onset at the first step where **`tube_distance > τ` AND**
  an internal CUSUM on a **geometry** signal (default `kl_accel`) exceeds `h` (two-of-N
  rule, **never on scissors**). This kills the false positives that sink single-signal
  detectors.
- Emits sticky `onset_step`, a `score` blending tube distance + CUSUM into [0,1],
  `triggered_metrics`, and a human-readable `explanation`.
- **Pluggability:** the user's exact contraction/trajectory-tube math replaces the
  `tube_distance` computation without touching the rest of the pipeline.

**Tube reference source.** The tube is fit on the **robust-verifier run** of a
controlled pair — same model, same conditions, two rewards: a gameable test-pass reward
vs. a robust/isomorphic verifier the policy provably can't hack. Fit on the run that
doesn't hack; judge the gameable run against it. *Deployment fallback (stated
limitation):* with no paired clean run, the tube falls back to an early-window or
known-healthy-run prior — which fails if a run hacks from step 0.

**Lead time** is computed against the **oracle-gap turn**: `lead = oracle_turn_step −
onset_step`, where `oracle_turn` is the changepoint of the *held-out oracle* series,
found offline evaluator-side. Positive lead = the oracle-blind detector fired before
true quality visibly degraded — the headline number. The honest baseline is the first
step a practitioner **watching the oracle curve** would call it; beating that on
oracle-blind signals is the result. (The training-reward turn is **not** a valid
reference: per §1 the reward looks fine while hacking, so it may never turn.)

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

- **`synthetic.py`** — **harness validation only.** Generates GRPO-shaped trajectories
  with a **ground-truth onset step** `t*` (before `t*`: healthy contraction, high
  entropy; after `t*`: entropy collapses, KL accelerates, advantages drift, oracle
  stalls while *training reward keeps rising*). Confirms the pipeline computes metrics
  and the detector fires on authored trajectories, and **tunes the false-positive
  rate**. It **does not constitute evidence the signal exists in real runs** — the onset
  is authored, so detecting it is tautological. Also emits **hard negatives**: healthy
  runs with `t* = ∞` that *look* alarming — a transient KL spike that recovers, entropy
  wobble, a benign reward jump from a real capability gain. **FPR = fraction of hard
  negatives on which the detector fires** (without them, precision/recall is measured
  against a flat line and means nothing). Seeded + parameterized (onset sharpness, noise,
  lead margin).
- **`reward_testhack.py`** — the test-overwriting reward pair: `train_reward` = fraction
  of *visible* unit tests passing (gameable by rewriting the test file); `oracle_reward`
  = fraction of *held-out* tests passing. **Sandbox isolation (requirement):** each
  completion is scored in an **ephemeral sandbox**; the held-out oracle tests live in a
  **separate clean sandbox applied after the model's code is frozen**, so
  test-overwriting can never contaminate the oracle. Without this, the hack corrupts the
  ground truth and "true quality" stops being true.
- **`onset_label.py`** — **behavioral onset labeling for the real run** (independent of
  geometry *and* oracle). Static-analyzes sampled completions (AST: writes to test
  files, `@patch`, `monkeypatch`, `os.remove`, mutated `assert`, `pytest.skip`) →
  `onset_behavioral`; paired with `onset_oracle` (first step of `passes_visible ∧
  ¬passes_held_out`). Two independent anchors, **neither derived from the detector's
  inputs**, so "geometry led the behavior by N steps" is falsifiable.
- **`trl_grpo_qwen.py`** — a complete, runnable GRPO + small-Qwen (e.g. Qwen2.5-0.5B/1.5B)
  math/code RLVR script wired through the TRL adapter and the test-hack reward, sized for
  AWS spot GPUs. Guarded so it imports without a GPU; **scaffolded, not run, in V1.**

**Evaluation (`flr eval`)** — over N seeded runs (onset runs **and** hard negatives),
report per-detector: detection rate, **false-positive rate on hard negatives**, and
**mean lead time** vs the **oracle-gap turn**. Acceptance on synthetic is harness-level
only: `ContractionTubeDetector` has positive mean lead and dominates both oracle-blind
baselines at fixed FPR. The thesis itself is gated on the real run (§14).

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
- **Hero:** scissors chart — train vs oracle reward overlaid, divergence shaded
  (evaluator-side ground truth, drawn for the human; the detector never saw it).
- Five geometry panels (KL+accel, entropy+trend, advantage moments+drift, generation
  stats, train reward) as live sparklines/line charts.
- **Detector panel:** visualizes the trajectory tube, the live metric trajectory
  escaping it, and the CUSUM statistic crossing threshold.
- Event timeline/log.

**Report viewer** (`/report/:runId`) — separate surface. Loads a saved artifact; same
charts rendered static; **onset moment annotated** with a lead-time callout comparing
detector onset vs the **oracle-gap turn** vs baselines vs behavioral onset;
exportable/shareable.

## 11. Repo Layout

```
flight-recorder/
  pyproject.toml            # package: flightrecorder, CLI: flr, Apache-2.0
  README.md
  flightrecorder/
    __init__.py
    recorder.py
    types.py                # dataclasses from §4 (RolloutBatch/Frame, OracleFrame, ...)
    core/                   # kl, entropy, advantage, genstats (rollout); oracle (evaluator)
    detector/               # base, threshold, cusum, contraction_tube
    sinks/                  # base, jsonl, console, websocket
    adapters/               # trl, verl, openrlhf
    repro/                  # synthetic, reward_testhack, onset_label, trl_grpo_qwen, eval
    cli.py                  # flr synth | replay | serve | eval | demo
  server/                   # FastAPI app
  web/                      # React + Vite + TS + Tailwind app
  examples/                 # quickstart notebooks/scripts
  tests/                    # pytest: extractors, detectors, synthetic, adapters
  docs/
```

## 12. Stack & Dependencies

- **Core:** Python ≥3.10, numpy, scipy (Wasserstein/stats), stdlib dataclasses.
  No torch dependency in the core (adapters import torch lazily).
- **Server:** fastapi, uvicorn, websockets.
- **Repro (real run, optional extra `[train]`):** trl, transformers, datasets,
  accelerate, torch — lazily imported, never required for the synthetic demo.
- **Web:** react, vite, typescript, tailwindcss, uplot.
- Packaged with `uv`; extras: `flightrecorder[server]`, `[train]`, `[all]`.

## 13. Testing

- **pytest** unit tests for every extractor (known-input → known-output), the three
  detectors (synthetic trajectories with known `t*`), the scissors accumulation
  (evaluator-side), the genstats extractor, the onset labeler (AST cases), and the TRL
  adapter mapping (mocked trainer). A **boundary test** asserts no oracle-derived field
  is reachable from `Detector.update()` (type-level oracle-blindness).
- **Property/eval test:** the synthetic harness asserts the **oracle-blind**
  `ContractionTubeDetector` leads the authored oracle-gap turn (mean lead > 0), beats the
  oracle-blind baselines at fixed FPR, and stays below the FPR ceiling on hard negatives,
  over a fixed seed set. This is a **harness-level** regression test, **not the thesis** —
  the thesis is gated on the real run (§14).
- **Web:** typecheck + build; a smoke test that the Live dashboard renders a replayed
  artifact end-to-end.

## 14. Acceptance

**Tier 1 — engineering (this session):**
1. `flr synth` produces seeded onset artifacts **and hard negatives** (`t* = ∞`).
2. `flr eval` shows the **oracle-blind** tube detector with positive mean lead vs the
   oracle-gap turn, beating both **oracle-blind** baselines at fixed FPR, with FPR below
   ceiling on hard negatives.
3. `flr serve --replay` streams an artifact; Live dashboard shows the geometry metrics +
   the ONSET banner firing before the oracle-gap turn.
4. Report viewer loads the artifact and annotates onset + lead time.
5. TRL adapter unit-tested against a mocked trainer; `trl_grpo_qwen.py` imports and is
   launch-ready (not run). verl + OpenRLHF adapters present and interface-complete.
6. Sandbox isolation enforced for the reward pair; behavioral onset labeler
   (`onset_label.py`) produces `onset_behavioral` + `onset_oracle` independent of
   geometry. `pip install -e .` clean; tests green; README quickstart works.

**Threshold provenance.** Detector thresholds are **frozen before the real run is seen**
(tuned only on synthetic), or real-run results are reported at **pre-registered**
thresholds — so the synthetic's built-in lead cannot leak into the real-run claim.

**Tier 2 — thesis (the actual proof; gates "done"):**
7. **One real TRL+Qwen GRPO run that learns the test-overwrite hack**, showing the
   oracle-blind detector fires with **positive lead vs the oracle-gap turn** and within
   range of the behavioral onset, at pre-registered thresholds.

Until #7 executes, status reads: **"core complete, harness-validated, real run
launch-ready; thesis: pending real run."**

## 15. Out of Scope / Next

- Hosting & deployment (next prompt) — design already container-friendly.
- Executing the real GPU run on AWS spot.
- PPO/DPO support, multi-node recording, auth, persistence beyond JSONL artifacts.
