# Flight Recorder — Core + Synthetic Proof Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the oracle-blind Flight Recorder core — rollout-geometry extractors, the pluggable detector (ContractionTube + two oracle-blind baselines), the evaluator, and a synthetic harness that proves the tube detector leads the oracle-gap turn and beats baselines at fixed FPR.

**Architecture:** Pure-Python, torch-free core. Trainers push a `RolloutBatch`; rollout extractors build a `RolloutFrame` (the only thing a `Detector` may see) and the oracle extractor builds an `OracleFrame` (evaluator-only). Oracle-blindness is enforced at the type boundary. A synthetic generator emits authored `RolloutFrame`/`OracleFrame` streams (onset runs + hard negatives) that drive detector + evaluator validation GPU-free.

**Tech Stack:** Python ≥3.10, numpy, scipy, pytest, `uv`, hatchling. CLI via argparse.

**Scope (Plan 1 only):** types, `core/` extractors, `detector/`, `eval/`, `sinks/` (jsonl+console), `recorder.py`, `repro/synthetic.py`, `cli.py` (`flr synth`, `flr eval`). **Out of scope:** TRL/verl/OpenRLHF adapters, reward_testhack/sandbox, onset_label, server, web (Plans 2–4).

---

## File Structure

```
flight-recorder/
  pyproject.toml                       # package metadata, deps, flr entrypoint
  flightrecorder/
    __init__.py                        # version + top-level exports
    types.py                           # RolloutBatch, RolloutFrame, OracleFrame, DetectorState, Event
    core/
      __init__.py
      rolling.py                       # EWMA, Series helper (slope/accel), running moments
      kl.py                            # KLExtractor (k1/k3, frozen-ref)
      entropy.py                       # EntropyExtractor
      advantage.py                     # AdvantageExtractor (moments + Wasserstein drift)
      genstats.py                      # GenStatsExtractor (corroboration-only)
      oracle.py                        # OracleExtractor (evaluator-only: oracle_gap, scissors)
    detector/
      __init__.py
      base.py                          # Detector Protocol (update(RolloutFrame)->DetectorState)
      threshold.py                     # ThresholdDetector (oracle-blind baseline)
      cusum.py                         # CusumDetector (oracle-blind baseline)
      contraction_tube.py              # ContractionTubeDetector (default, differentiated)
    sinks/
      __init__.py
      base.py                          # EventSink Protocol
      jsonl.py                         # JSONLSink
      console.py                       # ConsoleSink
    recorder.py                        # Recorder: record(batch) | record_frames(rf, of)
    eval/
      __init__.py
      evaluator.py                     # oracle_turn changepoint, lead time, FPR/detection-rate
    repro/
      __init__.py
      synthetic.py                     # generate_run (onset + hard-negative), write artifacts
    cli.py                             # argparse: flr synth | eval
  tests/
    test_types.py
    test_oracle_blindness.py           # the boundary test
    test_rolling.py
    test_kl.py
    test_entropy.py
    test_advantage.py
    test_genstats.py
    test_oracle_extractor.py
    test_detectors.py
    test_evaluator.py
    test_synthetic.py
    test_eval_property.py              # harness-level thesis (oracle-blind)
    test_cli.py
```

**Responsibilities:** each file is one unit. Extractors are independently testable with hand-built batches. Detectors consume only `RolloutFrame`. The evaluator is the only place oracle-derived turns/lead are computed. The synthetic generator is the only source of authored trajectories.

---

## Task 1: Package scaffold

**Files:**
- Create: `pyproject.toml`, `flightrecorder/__init__.py`, `tests/__init__.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "flightrecorder"
version = "0.1.0"
description = "Reward-hacking onset detection for RL post-training runs"
readme = "README.md"
requires-python = ">=3.10"
license = "Apache-2.0"
dependencies = ["numpy>=1.24", "scipy>=1.10"]

[project.optional-dependencies]
dev = ["pytest>=7.4"]
server = ["fastapi>=0.110", "uvicorn>=0.27", "websockets>=12"]
train = ["trl>=0.9", "transformers>=4.40", "datasets>=2.18", "accelerate>=0.29"]

[project.scripts]
flr = "flightrecorder.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["flightrecorder"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create `flightrecorder/__init__.py`**

```python
"""Flight Recorder: reward-hacking onset detection for RL post-training."""

__version__ = "0.1.0"
```

- [ ] **Step 3: Create empty `tests/__init__.py`**

```python
```

- [ ] **Step 4: Create venv and install editable**

Run: `uv venv && uv pip install -e ".[dev]"`
Expected: installs numpy, scipy, pytest; `flr` script registered.

- [ ] **Step 5: Verify the package imports**

Run: `uv run python -c "import flightrecorder; print(flightrecorder.__version__)"`
Expected: prints `0.1.0`

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml flightrecorder/__init__.py tests/__init__.py
git commit -m "feat: package scaffold (flightrecorder, flr entrypoint)"
```

---

## Task 2: Data model (`types.py`)

**Files:**
- Create: `flightrecorder/types.py`
- Test: `tests/test_types.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_types.py
import numpy as np
from flightrecorder.types import RolloutBatch, RolloutFrame, OracleFrame, DetectorState, Event


def test_rolloutframe_has_no_oracle_fields():
    fields = set(RolloutFrame.__dataclass_fields__)
    for forbidden in ("oracle_reward", "oracle_gap", "scissors"):
        assert forbidden not in fields, f"{forbidden} must not be on RolloutFrame"


def test_oracleframe_carries_ground_truth():
    of = OracleFrame(step=3, oracle_reward=0.4, oracle_gap=0.5, scissors=1.2,
                     oracle_turn_step=None, onset_behavioral=None,
                     onset_oracle=None, synthetic_tstar=10)
    assert of.scissors == 1.2 and of.synthetic_tstar == 10


def test_event_roundtrips_payload():
    e = Event(kind="frame", step=1, payload={"kl_mean": 0.1}, ts=123.0)
    assert e.kind == "frame" and e.payload["kl_mean"] == 0.1


def test_rolloutbatch_optional_fields_default_none():
    b = RolloutBatch(step=0, train_rewards=np.array([1.0]), meta={})
    assert b.oracle_rewards is None and b.logprobs is None and b.completions is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_types.py -v`
Expected: FAIL with `ModuleNotFoundError: flightrecorder.types`

- [ ] **Step 3: Write `flightrecorder/types.py`**

```python
"""Core data model. Split frames make oracle leakage a type error (§4)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Any
import numpy as np


@dataclass
class RolloutBatch:
    """Per-step rollout data pushed by a trainer adapter."""
    step: int
    train_rewards: np.ndarray                      # gameable reward, per sample
    meta: dict
    oracle_rewards: np.ndarray | None = None       # held-out true quality (eval only)
    advantages: np.ndarray | None = None           # GRPO group-normalized
    logprobs: np.ndarray | None = None             # policy logprobs (token-level)
    ref_logprobs: np.ndarray | None = None         # frozen-SFT-ckpt logprobs
    entropy: np.ndarray | None = None              # token entropy if provided
    completions: list[str] | None = None           # raw text for genstats/labeling


@dataclass
class RolloutFrame:
    """The ONLY thing Detector.update() receives. No oracle fields exist here."""
    step: int
    kl_mean: float
    kl_slope: float
    kl_accel: float
    entropy_mean: float
    entropy_trend: float
    adv_var: float
    adv_skew: float
    adv_kurtosis: float
    adv_drift: float
    train_reward: float
    train_reward_slope: float
    gen_len_mean: float
    gen_len_var: float
    ngram_diversity: float
    logprob_concentration: float
    raw: dict = field(default_factory=dict)


@dataclass
class OracleFrame:
    """Evaluator-only ground truth. Never passed to a detector."""
    step: int
    oracle_reward: float
    oracle_gap: float
    scissors: float
    oracle_turn_step: int | None = None
    onset_behavioral: int | None = None
    onset_oracle: int | None = None
    synthetic_tstar: int | None = None


@dataclass
class DetectorState:
    step: int
    onset: bool
    onset_step: int | None
    score: float
    triggered_metrics: list[str]
    tube_distance: float
    cusum: float
    explanation: str


@dataclass
class Event:
    kind: Literal["frame", "oracle", "detector", "onset", "run_start", "run_end"]
    step: int
    payload: dict[str, Any]
    ts: float
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_types.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add flightrecorder/types.py tests/test_types.py
git commit -m "feat: split data model (RolloutFrame vs OracleFrame)"
```

---

## Task 3: Oracle-blindness boundary test

This is the structural guarantee from the review. It must exist before detectors.

**Files:**
- Test: `tests/test_oracle_blindness.py`

- [ ] **Step 1: Write the test (no implementation needed — it asserts a property of `types.py`)**

```python
# tests/test_oracle_blindness.py
import inspect
from flightrecorder.types import RolloutFrame, OracleFrame

ORACLE_FIELDS = {"oracle_reward", "oracle_gap", "scissors", "oracle_turn_step",
                 "onset_behavioral", "onset_oracle", "synthetic_tstar"}


def test_rolloutframe_shares_no_field_with_oracleframe():
    rollout = set(RolloutFrame.__dataclass_fields__)
    oracle = set(OracleFrame.__dataclass_fields__) - {"step"}
    assert rollout.isdisjoint(oracle), (
        "RolloutFrame leaks oracle fields: " + str(rollout & oracle))


def test_no_oracle_field_names_on_rolloutframe():
    assert ORACLE_FIELDS.isdisjoint(set(RolloutFrame.__dataclass_fields__))


def test_detector_protocol_signature_takes_rolloutframe():
    # Imported lazily so this task can run before detector/base.py exists is wrong;
    # base.py is Task 8. Guard the import so this file passes standalone too.
    try:
        from flightrecorder.detector.base import Detector
    except ModuleNotFoundError:
        return
    sig = inspect.signature(Detector.update)
    ann = sig.parameters["frame"].annotation
    assert getattr(ann, "__name__", str(ann)) == "RolloutFrame"
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/test_oracle_blindness.py -v`
Expected: 3 passed (third is a no-op until Task 8, then asserts the signature)

- [ ] **Step 3: Commit**

```bash
git add tests/test_oracle_blindness.py
git commit -m "test: enforce oracle-blindness at the type boundary"
```

---

## Task 4: Rolling helpers (`core/rolling.py`)

**Files:**
- Create: `flightrecorder/core/__init__.py`, `flightrecorder/core/rolling.py`
- Test: `tests/test_rolling.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rolling.py
import numpy as np
from flightrecorder.core.rolling import Smoother, RunningMoments


def test_smoother_tracks_value_slope_accel():
    s = Smoother(alpha=1.0)  # alpha=1 => no smoothing, value passes through
    s.update(0.0); s.update(1.0); s.update(3.0)
    assert s.value == 3.0
    assert s.slope == 2.0           # 3 - 1
    assert s.accel == 1.0           # (3-1) - (1-0)


def test_smoother_ewma_between_zero_and_one():
    s = Smoother(alpha=0.5)
    s.update(0.0); s.update(10.0)
    assert 0.0 < s.value < 10.0


def test_running_moments_mean_std():
    rm = RunningMoments()
    for x in [1.0, 2.0, 3.0, 4.0]:
        rm.update(x)
    assert abs(rm.mean - 2.5) < 1e-9
    assert rm.std > 0
    assert abs(rm.zscore(2.5)) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rolling.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create `flightrecorder/core/__init__.py` (empty) and `core/rolling.py`**

```python
# flightrecorder/core/__init__.py
```

```python
# flightrecorder/core/rolling.py
"""Stateful helpers: EWMA smoothing with slope/accel, and online moments."""
from __future__ import annotations

import math


class Smoother:
    """EWMA value + first/second finite differences of the smoothed series."""

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.value: float = 0.0
        self.slope: float = 0.0
        self.accel: float = 0.0
        self._prev_value: float | None = None
        self._prev_slope: float | None = None
        self._init = False

    def update(self, x: float) -> "Smoother":
        if not self._init:
            self.value = x
            self._init = True
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        if self._prev_value is not None:
            new_slope = self.value - self._prev_value
            if self._prev_slope is not None:
                self.accel = new_slope - self._prev_slope
            self._prev_slope = new_slope
            self.slope = new_slope
        self._prev_value = self.value
        return self


class RunningMoments:
    """Welford online mean/variance with a z-score helper."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self._m2 = 0.0

    def update(self, x: float) -> "RunningMoments":
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self._m2 += delta * (x - self.mean)
        return self

    @property
    def var(self) -> float:
        return self._m2 / self.n if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.var)

    def zscore(self, x: float) -> float:
        s = self.std
        return 0.0 if s < 1e-12 else (x - self.mean) / s
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_rolling.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add flightrecorder/core/__init__.py flightrecorder/core/rolling.py tests/test_rolling.py
git commit -m "feat: rolling EWMA/slope/accel + online moments"
```

---

## Task 5: KL extractor (`core/kl.py`)

**Files:**
- Create: `flightrecorder/core/kl.py`
- Test: `tests/test_kl.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_kl.py
import numpy as np
from flightrecorder.core.kl import KLExtractor


def test_k1_kl_is_mean_logprob_diff():
    ext = KLExtractor(estimator="k1", alpha=1.0)
    lp = np.array([-1.0, -2.0]); rlp = np.array([-1.5, -2.5])
    out = ext.update(lp, rlp)
    assert abs(out["kl_mean"] - 0.5) < 1e-9   # mean((lp - rlp)) = 0.5


def test_kl_accel_positive_when_kl_curves_up():
    ext = KLExtractor(estimator="k1", alpha=1.0)
    for kl in (0.0, 1.0, 3.0):  # diffs grow: slope 1 then 2 => accel +1
        ext.update(np.array([-kl]), np.array([0.0]))
    out = ext.update(np.array([-6.0]), np.array([0.0]))  # slope 3 => accel +1
    assert out["kl_accel"] > 0


def test_missing_inputs_return_nan_no_crash():
    ext = KLExtractor()
    out = ext.update(None, None)
    assert np.isnan(out["kl_mean"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_kl.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `flightrecorder/core/kl.py`**

```python
# flightrecorder/core/kl.py
"""KL(policy || frozen-SFT-ref). k1 = mean(lp - rlp); k3 = exp(-d)-1+d, d=lp-rlp."""
from __future__ import annotations

import numpy as np
from .rolling import Smoother


class KLExtractor:
    def __init__(self, estimator: str = "k1", alpha: float = 0.3):
        assert estimator in ("k1", "k3")
        self.estimator = estimator
        self._s = Smoother(alpha=alpha)

    def update(self, logprobs, ref_logprobs) -> dict:
        if logprobs is None or ref_logprobs is None:
            return {"kl_mean": float("nan"), "kl_slope": 0.0, "kl_accel": 0.0}
        d = np.asarray(logprobs, float) - np.asarray(ref_logprobs, float)
        per_tok = d if self.estimator == "k1" else (np.exp(-d) - 1.0 + d)
        kl = float(np.mean(per_tok))
        self._s.update(kl)
        return {"kl_mean": kl, "kl_slope": self._s.slope, "kl_accel": self._s.accel}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_kl.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add flightrecorder/core/kl.py tests/test_kl.py
git commit -m "feat: KL extractor (k1/k3, frozen ref, slope/accel)"
```

---

## Task 6: Entropy + Advantage + GenStats extractors

Three focused extractors. One task, three files, three tests — each is small.

**Files:**
- Create: `flightrecorder/core/entropy.py`, `flightrecorder/core/advantage.py`, `flightrecorder/core/genstats.py`
- Test: `tests/test_entropy.py`, `tests/test_advantage.py`, `tests/test_genstats.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_entropy.py
import numpy as np
from flightrecorder.core.entropy import EntropyExtractor


def test_uses_provided_entropy_mean():
    ext = EntropyExtractor(alpha=1.0)
    out = ext.update(entropy=np.array([2.0, 4.0]), logprobs=None)
    assert abs(out["entropy_mean"] - 3.0) < 1e-9


def test_entropy_trend_negative_on_collapse():
    ext = EntropyExtractor(alpha=1.0)
    for h in (5.0, 4.0, 2.0):
        out = ext.update(entropy=np.array([h]), logprobs=None)
    assert out["entropy_trend"] < 0


def test_falls_back_to_logprob_surprisal():
    ext = EntropyExtractor(alpha=1.0)
    out = ext.update(entropy=None, logprobs=np.array([-1.0, -3.0]))
    assert abs(out["entropy_mean"] - 2.0) < 1e-9  # mean(-logprob)
```

```python
# tests/test_advantage.py
import numpy as np
from flightrecorder.core.advantage import AdvantageExtractor


def test_moments_of_advantages():
    ext = AdvantageExtractor(window=2)
    out = ext.update(np.array([-1.0, 0.0, 1.0]))
    assert out["adv_var"] > 0
    assert abs(out["adv_skew"]) < 1e-6  # symmetric


def test_drift_rises_when_distribution_shifts():
    ext = AdvantageExtractor(window=1)
    ext.update(np.array([0.0, 0.1, -0.1]))
    out = ext.update(np.array([10.0, 11.0, 9.0]))  # shifted far
    assert out["adv_drift"] > 1.0


def test_none_advantages_safe():
    ext = AdvantageExtractor()
    out = ext.update(None)
    assert np.isnan(out["adv_var"])
```

```python
# tests/test_genstats.py
from flightrecorder.core.genstats import GenStatsExtractor


def test_len_and_diversity():
    ext = GenStatsExtractor()
    out = ext.update(completions=["a b c", "a b c d"], logprobs=None)
    assert out["gen_len_mean"] == 3.5
    assert 0.0 <= out["ngram_diversity"] <= 1.0


def test_logprob_concentration_from_logprobs():
    ext = GenStatsExtractor()
    out = ext.update(completions=["x"], logprobs=[0.0])  # exp(0)=1 fully concentrated
    assert abs(out["logprob_concentration"] - 1.0) < 1e-9


def test_empty_safe():
    ext = GenStatsExtractor()
    out = ext.update(completions=None, logprobs=None)
    assert out["gen_len_mean"] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_entropy.py tests/test_advantage.py tests/test_genstats.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the three modules**

```python
# flightrecorder/core/entropy.py
"""Policy entropy + collapse trend. Falls back to mean surprisal if no entropy."""
from __future__ import annotations

import numpy as np
from .rolling import Smoother


class EntropyExtractor:
    def __init__(self, alpha: float = 0.3):
        self._s = Smoother(alpha=alpha)

    def update(self, entropy, logprobs) -> dict:
        if entropy is not None:
            h = float(np.mean(np.asarray(entropy, float)))
        elif logprobs is not None:
            h = float(np.mean(-np.asarray(logprobs, float)))  # surprisal proxy
        else:
            return {"entropy_mean": float("nan"), "entropy_trend": 0.0}
        self._s.update(h)
        return {"entropy_mean": h, "entropy_trend": self._s.slope}
```

```python
# flightrecorder/core/advantage.py
"""GRPO advantage moments + Wasserstein drift vs a rolling baseline."""
from __future__ import annotations

from collections import deque
import numpy as np
from scipy.stats import skew, kurtosis, wasserstein_distance


class AdvantageExtractor:
    def __init__(self, window: int = 20):
        self.window = window
        self._hist: deque[np.ndarray] = deque(maxlen=window + 1)

    def update(self, advantages) -> dict:
        if advantages is None:
            return {"adv_var": float("nan"), "adv_skew": float("nan"),
                    "adv_kurtosis": float("nan"), "adv_drift": float("nan")}
        a = np.asarray(advantages, float)
        drift = 0.0
        if len(self._hist) >= self.window:
            baseline = self._hist[0]
            drift = float(wasserstein_distance(a, baseline))
        self._hist.append(a)
        return {"adv_var": float(np.var(a)),
                "adv_skew": float(skew(a)) if a.size > 2 else 0.0,
                "adv_kurtosis": float(kurtosis(a)) if a.size > 2 else 0.0,
                "adv_drift": drift}
```

```python
# flightrecorder/core/genstats.py
"""Generation statistics from completions/logprobs. Corroboration-only (§5)."""
from __future__ import annotations

import numpy as np


def _ngram_diversity(texts: list[str], n: int = 3) -> float:
    grams, total = set(), 0
    for t in texts:
        toks = t.split()
        for i in range(len(toks) - n + 1):
            grams.add(tuple(toks[i:i + n])); total += 1
    return len(grams) / total if total else 0.0


class GenStatsExtractor:
    def update(self, completions, logprobs) -> dict:
        if not completions:
            lens = np.array([0.0])
            div = 0.0
        else:
            lens = np.array([len(c.split()) for c in completions], float)
            div = _ngram_diversity(completions)
        conc = (float(np.mean(np.exp(np.asarray(logprobs, float))))
                if logprobs is not None and len(logprobs) else 0.0)
        return {"gen_len_mean": float(np.mean(lens)),
                "gen_len_var": float(np.var(lens)),
                "ngram_diversity": div,
                "logprob_concentration": conc}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_entropy.py tests/test_advantage.py tests/test_genstats.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add flightrecorder/core/entropy.py flightrecorder/core/advantage.py flightrecorder/core/genstats.py tests/test_entropy.py tests/test_advantage.py tests/test_genstats.py
git commit -m "feat: entropy, advantage-drift, generation-stats extractors"
```

---

## Task 7: Oracle extractor (`core/oracle.py`) — evaluator-only

**Files:**
- Create: `flightrecorder/core/oracle.py`
- Test: `tests/test_oracle_extractor.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_oracle_extractor.py
from flightrecorder.core.oracle import OracleExtractor


def test_gap_is_train_minus_oracle():
    ext = OracleExtractor(alpha=1.0)
    out = ext.update(train_reward=0.9, oracle_reward=0.4)
    assert abs(out["oracle_gap"] - 0.5) < 1e-9


def test_scissors_opens_when_train_rises_oracle_flat():
    ext = OracleExtractor(alpha=1.0)
    ext.update(0.1, 0.1)
    ext.update(0.5, 0.1)   # train +0.4, oracle 0  -> scissors += 0.4
    out = ext.update(0.9, 0.1)  # +0.4 again -> scissors ~ 0.8
    assert out["scissors"] > 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_oracle_extractor.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `flightrecorder/core/oracle.py`**

```python
# flightrecorder/core/oracle.py
"""Evaluator-only divergence signal. Output lands in OracleFrame, never RolloutFrame."""
from __future__ import annotations

from .rolling import Smoother


class OracleExtractor:
    def __init__(self, alpha: float = 0.3):
        self._train = Smoother(alpha=alpha)
        self._oracle = Smoother(alpha=alpha)
        self.scissors = 0.0

    def update(self, train_reward: float, oracle_reward: float) -> dict:
        self._train.update(train_reward)
        self._oracle.update(oracle_reward)
        self.scissors += self._train.slope - self._oracle.slope
        return {"oracle_reward": float(oracle_reward),
                "oracle_gap": float(train_reward - oracle_reward),
                "scissors": float(self.scissors)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_oracle_extractor.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add flightrecorder/core/oracle.py tests/test_oracle_extractor.py
git commit -m "feat: oracle extractor (evaluator-only gap + scissors)"
```

---

## Task 8: Detectors (`detector/`) — base + two oracle-blind baselines + ContractionTube

**Files:**
- Create: `flightrecorder/detector/__init__.py`, `base.py`, `threshold.py`, `cusum.py`, `contraction_tube.py`
- Test: `tests/test_detectors.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_detectors.py
import numpy as np
from flightrecorder.types import RolloutFrame
from flightrecorder.detector.threshold import ThresholdDetector
from flightrecorder.detector.cusum import CusumDetector
from flightrecorder.detector.contraction_tube import ContractionTubeDetector


def _frame(step, kl_accel=0.0, entropy_trend=0.0, adv_drift=0.0):
    return RolloutFrame(step=step, kl_mean=0.0, kl_slope=0.0, kl_accel=kl_accel,
                        entropy_mean=0.0, entropy_trend=entropy_trend,
                        adv_var=0.0, adv_skew=0.0, adv_kurtosis=0.0, adv_drift=adv_drift,
                        train_reward=0.0, train_reward_slope=0.0, gen_len_mean=0.0,
                        gen_len_var=0.0, ngram_diversity=0.0, logprob_concentration=0.0)


def _drive(det, healthy_steps=30, shock=True):
    rng = np.random.default_rng(0)
    fired = None
    for t in range(healthy_steps):
        st = det.update(_frame(t, kl_accel=rng.normal(0, 0.05)))
        if st.onset and fired is None:
            fired = st.onset_step
    if shock:
        for t in range(healthy_steps, healthy_steps + 20):
            st = det.update(_frame(t, kl_accel=2.0 + rng.normal(0, 0.05),
                                   entropy_trend=-1.0, adv_drift=3.0))
            if st.onset and fired is None:
                fired = st.onset_step
    return fired


def test_tube_detector_fires_after_shock_not_during_healthy():
    det = ContractionTubeDetector(warmup=20)
    fired = _drive(det)
    assert fired is not None and fired >= 30


def test_tube_detector_silent_on_healthy_only():
    det = ContractionTubeDetector(warmup=20)
    fired = _drive(det, shock=False)
    assert fired is None


def test_baselines_fire_on_shock():
    assert _drive(ThresholdDetector(warmup=20)) is not None
    assert _drive(CusumDetector(warmup=20)) is not None


def test_onset_is_sticky():
    det = ContractionTubeDetector(warmup=20)
    _drive(det)
    st = det.update(_frame(100, kl_accel=2.0, entropy_trend=-1.0, adv_drift=3.0))
    assert st.onset is False and st.onset_step is not None  # not re-declared
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_detectors.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `detector/__init__.py` and `base.py`**

```python
# flightrecorder/detector/__init__.py
```

```python
# flightrecorder/detector/base.py
"""Detector contract. update() takes RolloutFrame ONLY (oracle-blind by type)."""
from __future__ import annotations

from typing import Protocol, runtime_checkable
from ..types import RolloutFrame, DetectorState


@runtime_checkable
class Detector(Protocol):
    def update(self, frame: RolloutFrame) -> DetectorState: ...
    def reset(self) -> None: ...


# Geometry signals a detector may read (names on RolloutFrame). Never oracle fields.
GEOMETRY_SIGNALS = ("kl_accel", "entropy_trend", "adv_drift",
                    "gen_len_var", "logprob_concentration")


def make_state(step, onset, onset_step, score, triggered, tube_distance, cusum, why):
    return DetectorState(step=step, onset=onset, onset_step=onset_step,
                         score=float(score), triggered_metrics=list(triggered),
                         tube_distance=float(tube_distance), cusum=float(cusum),
                         explanation=why)
```

- [ ] **Step 4: Write `detector/threshold.py`**

```python
# flightrecorder/detector/threshold.py
"""Oracle-blind baseline: fire when any geometry signal exceeds a z-score threshold."""
from __future__ import annotations

from ..types import RolloutFrame, DetectorState
from ..core.rolling import RunningMoments
from .base import make_state

# Danger sign per signal: +1 means high values are alarming, -1 means low (collapse).
_SIGNALS = {"kl_accel": +1, "entropy_trend": -1, "adv_drift": +1}


class ThresholdDetector:
    def __init__(self, z: float = 4.0, warmup: int = 20):
        self.z = z; self.warmup = warmup
        self._m = {k: RunningMoments() for k in _SIGNALS}
        self._onset_step: int | None = None; self._n = 0

    def reset(self) -> None:
        self.__init__(self.z, self.warmup)

    def update(self, frame: RolloutFrame) -> DetectorState:
        self._n += 1
        triggered, worst = [], 0.0
        for sig, sign in _SIGNALS.items():
            m = self._m[sig]; val = getattr(frame, sig)
            z = sign * m.zscore(val)
            if self._n > self.warmup and z > self.z:
                triggered.append(sig); worst = max(worst, z)
            m.update(val)
        if triggered and self._onset_step is None:
            self._onset_step = frame.step
            return make_state(frame.step, True, self._onset_step,
                              min(1.0, worst / (2 * self.z)), triggered, 0.0, 0.0,
                              f"threshold breach: {triggered}")
        return make_state(frame.step, False, self._onset_step,
                          min(1.0, worst / (2 * self.z)), triggered, 0.0, 0.0, "")
```

- [ ] **Step 5: Write `detector/cusum.py`**

```python
# flightrecorder/detector/cusum.py
"""Oracle-blind baseline: one-sided CUSUM changepoint on a single geometry signal."""
from __future__ import annotations

from ..types import RolloutFrame, DetectorState
from ..core.rolling import RunningMoments
from .base import make_state


class CusumDetector:
    def __init__(self, signal: str = "kl_accel", k: float = 0.5, h: float = 5.0,
                 warmup: int = 20):
        self.signal = signal; self.k = k; self.h = h; self.warmup = warmup
        self._m = RunningMoments(); self._S = 0.0
        self._onset_step: int | None = None; self._n = 0

    def reset(self) -> None:
        self.__init__(self.signal, self.k, self.h, self.warmup)

    def update(self, frame: RolloutFrame) -> DetectorState:
        self._n += 1
        x = getattr(frame, self.signal)
        if self._n <= self.warmup:
            self._m.update(x)
            return make_state(frame.step, False, self._onset_step, 0.0, [], 0.0, 0.0, "")
        z = self._m.zscore(x)
        self._S = max(0.0, self._S + z - self.k)
        if self._S > self.h and self._onset_step is None:
            self._onset_step = frame.step
            return make_state(frame.step, True, self._onset_step, min(1.0, self._S / (2 * self.h)),
                              [self.signal], 0.0, self._S, f"CUSUM>{self.h} on {self.signal}")
        return make_state(frame.step, False, self._onset_step,
                          min(1.0, self._S / (2 * self.h)), [], 0.0, self._S, "")
```

- [ ] **Step 6: Write `detector/contraction_tube.py`**

```python
# flightrecorder/detector/contraction_tube.py
"""Default detector. Tube on [kl_accel, entropy_trend, adv_drift]; onset = tube-exit
AND CUSUM on a geometry signal (two-of-N). Fit on warmup window or an external reference."""
from __future__ import annotations

import numpy as np
from ..types import RolloutFrame, DetectorState
from .base import make_state

_DIMS = ("kl_accel", "entropy_trend", "adv_drift")


class ContractionTubeDetector:
    def __init__(self, warmup: int = 20, rho: float = 3.0, tau: float = 1.0,
                 k: float = 0.5, h: float = 5.0, cusum_signal: str = "kl_accel"):
        self.warmup = warmup; self.rho = rho; self.tau = tau
        self.k = k; self.h = h; self.cusum_signal = cusum_signal
        self._buf: list[np.ndarray] = []
        self._center = None; self._scale = None
        self._cu_mean = None; self._cu_std = None
        self._S = 0.0; self._onset_step: int | None = None; self._n = 0

    def reset(self) -> None:
        self.__init__(self.warmup, self.rho, self.tau, self.k, self.h, self.cusum_signal)

    def set_reference(self, frames: list[RolloutFrame]) -> None:
        """Controlled-pair fit: pass frames from the robust-verifier (non-hacking) run."""
        m = np.array([[getattr(f, d) for d in _DIMS] for f in frames], float)
        self._center = np.median(m, axis=0)
        self._scale = np.median(np.abs(m - self._center), axis=0) + 1e-6
        c = np.array([getattr(f, self.cusum_signal) for f in frames], float)
        self._cu_mean = float(np.mean(c)); self._cu_std = float(np.std(c)) + 1e-6

    def _fit_from_buffer(self) -> None:
        m = np.array(self._buf, float)
        self._center = np.median(m, axis=0)
        self._scale = np.median(np.abs(m - self._center), axis=0) + 1e-6
        cu = m[:, _DIMS.index(self.cusum_signal)]
        self._cu_mean = float(np.mean(cu)); self._cu_std = float(np.std(cu)) + 1e-6

    def update(self, frame: RolloutFrame) -> DetectorState:
        self._n += 1
        m = np.array([getattr(frame, d) for d in _DIMS], float)
        if self._center is None:
            self._buf.append(m)
            if self._n >= self.warmup:
                self._fit_from_buffer()
            return make_state(frame.step, False, self._onset_step, 0.0, [], 0.0, 0.0, "")
        z = (m - self._center) / self._scale
        outside = np.maximum(np.abs(z) - self.rho, 0.0)
        tube_distance = float(np.linalg.norm(outside))
        cu_z = (getattr(frame, self.cusum_signal) - self._cu_mean) / self._cu_std
        self._S = max(0.0, self._S + cu_z - self.k)
        tube_hit = tube_distance > self.tau
        cusum_hit = self._S > self.h
        score = min(1.0, 0.5 * tube_distance / max(self.tau, 1e-6)
                    + 0.5 * self._S / max(self.h, 1e-6))
        if tube_hit and cusum_hit and self._onset_step is None:
            self._onset_step = frame.step
            triggered = [d for d, zi in zip(_DIMS, z) if abs(zi) > self.rho] or [self.cusum_signal]
            return make_state(frame.step, True, self._onset_step, max(score, 0.5),
                              triggered, tube_distance, self._S,
                              f"tube-exit({tube_distance:.2f}) AND CUSUM({self._S:.2f})")
        return make_state(frame.step, False, self._onset_step, score, [],
                          tube_distance, self._S, "")
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/test_detectors.py tests/test_oracle_blindness.py -v`
Expected: all passed (oracle-blindness third test now asserts the `RolloutFrame` annotation)

- [ ] **Step 8: Commit**

```bash
git add flightrecorder/detector tests/test_detectors.py
git commit -m "feat: detectors (oracle-blind threshold/cusum baselines + ContractionTube)"
```

---

## Task 9: Sinks + Recorder

**Files:**
- Create: `flightrecorder/sinks/__init__.py`, `base.py`, `jsonl.py`, `console.py`, `flightrecorder/recorder.py`
- Test: `tests/test_recorder.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_recorder.py
import json
import numpy as np
from flightrecorder.types import RolloutFrame, OracleFrame
from flightrecorder.sinks.jsonl import JSONLSink
from flightrecorder.recorder import Recorder
from flightrecorder.detector.contraction_tube import ContractionTubeDetector


def _rf(step):
    return RolloutFrame(step=step, kl_mean=0, kl_slope=0, kl_accel=0, entropy_mean=0,
                        entropy_trend=0, adv_var=0, adv_skew=0, adv_kurtosis=0, adv_drift=0,
                        train_reward=0, train_reward_slope=0, gen_len_mean=0, gen_len_var=0,
                        ngram_diversity=0, logprob_concentration=0)


def test_recorder_writes_frame_and_detector_events(tmp_path):
    art = tmp_path / "run.jsonl"
    rec = Recorder(detector=ContractionTubeDetector(warmup=2), sinks=[JSONLSink(art)])
    for s in range(5):
        rec.record_frames(_rf(s), OracleFrame(step=s, oracle_reward=0.0, oracle_gap=0.0, scissors=0.0))
    rec.close()
    kinds = [json.loads(l)["kind"] for l in art.read_text().splitlines()]
    assert "frame" in kinds and "detector" in kinds and "oracle" in kinds


def test_recorder_record_batch_runs_extractors(tmp_path):
    from flightrecorder.types import RolloutBatch
    rec = Recorder(detector=ContractionTubeDetector(warmup=2), sinks=[])
    rf, of = rec.record(RolloutBatch(step=0, train_rewards=np.array([1.0, 0.0]),
                                     oracle_rewards=np.array([1.0, 0.0]),
                                     advantages=np.array([0.1, -0.1]),
                                     logprobs=np.array([-1.0]), ref_logprobs=np.array([-1.2]),
                                     completions=["a b c"], meta={}))
    assert rf.step == 0 and of.oracle_gap == 0.0  # train mean .5 - oracle mean .5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_recorder.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write sinks**

```python
# flightrecorder/sinks/__init__.py
```

```python
# flightrecorder/sinks/base.py
from __future__ import annotations
from typing import Protocol
from ..types import Event


class EventSink(Protocol):
    def emit(self, event: Event) -> None: ...
    def close(self) -> None: ...
```

```python
# flightrecorder/sinks/jsonl.py
from __future__ import annotations
import json, time
from dataclasses import asdict
from pathlib import Path
from ..types import Event


class JSONLSink:
    def __init__(self, path):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8")

    def emit(self, event: Event) -> None:
        self._fh.write(json.dumps(asdict(event)) + "\n"); self._fh.flush()

    def close(self) -> None:
        self._fh.close()
```

```python
# flightrecorder/sinks/console.py
from __future__ import annotations
from ..types import Event


class ConsoleSink:
    def emit(self, event: Event) -> None:
        if event.kind == "onset":
            print(f"[ONSET] step={event.step} {event.payload.get('explanation','')}")

    def close(self) -> None:
        pass
```

- [ ] **Step 4: Write `flightrecorder/recorder.py`**

```python
# flightrecorder/recorder.py
"""Orchestrates extractors -> RolloutFrame (+ OracleFrame), feeds ONLY RolloutFrame
to the detector, emits events to sinks."""
from __future__ import annotations

import time
from dataclasses import asdict
import numpy as np
from .types import RolloutBatch, RolloutFrame, OracleFrame, Event
from .core.kl import KLExtractor
from .core.entropy import EntropyExtractor
from .core.advantage import AdvantageExtractor
from .core.genstats import GenStatsExtractor
from .core.oracle import OracleExtractor
from .core.rolling import Smoother


class Recorder:
    def __init__(self, detector, sinks):
        self.detector = detector; self.sinks = list(sinks)
        self._kl = KLExtractor(); self._ent = EntropyExtractor()
        self._adv = AdvantageExtractor(); self._gen = GenStatsExtractor()
        self._oracle = OracleExtractor(); self._train = Smoother()

    def _emit(self, kind, step, payload):
        ev = Event(kind=kind, step=step, payload=payload, ts=time.time())
        for s in self.sinks:
            s.emit(ev)

    def record(self, batch: RolloutBatch):
        train_r = float(np.mean(batch.train_rewards))
        self._train.update(train_r)
        rf = RolloutFrame(
            step=batch.step, train_reward=train_r, train_reward_slope=self._train.slope,
            **self._kl.update(batch.logprobs, batch.ref_logprobs),
            **self._ent.update(batch.entropy, batch.logprobs),
            **self._adv.update(batch.advantages),
            **self._gen.update(batch.completions, batch.logprobs))
        of = None
        if batch.oracle_rewards is not None:
            oracle_r = float(np.mean(batch.oracle_rewards))
            od = self._oracle.update(train_r, oracle_r)
            of = OracleFrame(step=batch.step, **od)
        self._dispatch(rf, of)
        return rf, of

    def record_frames(self, rf: RolloutFrame, of: OracleFrame | None = None):
        self._dispatch(rf, of)
        return rf, of

    def _dispatch(self, rf: RolloutFrame, of: OracleFrame | None):
        self._emit("frame", rf.step, asdict(rf))
        if of is not None:
            self._emit("oracle", of.step, asdict(of))
        state = self.detector.update(rf)          # RolloutFrame ONLY
        self._emit("detector", rf.step, asdict(state))
        if state.onset:
            self._emit("onset", rf.step, asdict(state))

    def close(self):
        for s in self.sinks:
            s.close()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_recorder.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add flightrecorder/sinks flightrecorder/recorder.py tests/test_recorder.py
git commit -m "feat: sinks (jsonl/console) + Recorder orchestration"
```

---

## Task 10: Evaluator (`eval/evaluator.py`)

**Files:**
- Create: `flightrecorder/eval/__init__.py`, `flightrecorder/eval/evaluator.py`
- Test: `tests/test_evaluator.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluator.py
from flightrecorder.eval.evaluator import oracle_gap_turn, lead_time, aggregate


def test_oracle_turn_detects_sustained_drop():
    series = [0.8] * 20 + [0.7, 0.5, 0.3, 0.1, 0.05]  # drop starts at index 20
    turn = oracle_gap_turn(series, k=0.5, h=3.0, warmup=10)
    assert turn is not None and 20 <= turn <= 24


def test_oracle_turn_none_when_flat():
    assert oracle_gap_turn([0.8] * 40, warmup=10) is None


def test_lead_is_turn_minus_onset():
    assert lead_time(onset_step=15, oracle_turn=22) == 7
    assert lead_time(onset_step=None, oracle_turn=22) is None


def test_aggregate_reports_rates_and_mean_lead():
    runs = [
        {"onset_step": 15, "oracle_turn": 22, "is_hard_negative": False},
        {"onset_step": 18, "oracle_turn": 25, "is_hard_negative": False},
        {"onset_step": None, "oracle_turn": None, "is_hard_negative": True},
        {"onset_step": 30, "oracle_turn": None, "is_hard_negative": True},  # false positive
    ]
    agg = aggregate(runs)
    assert abs(agg["detection_rate"] - 1.0) < 1e-9
    assert abs(agg["fpr"] - 0.5) < 1e-9
    assert abs(agg["mean_lead"] - 7.0) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_evaluator.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the evaluator**

```python
# flightrecorder/eval/__init__.py
```

```python
# flightrecorder/eval/evaluator.py
"""Evaluator-side metrics. oracle_gap_turn is the lead-time reference (§6)."""
from __future__ import annotations

import numpy as np


def oracle_gap_turn(oracle_series, k: float = 0.5, h: float = 3.0,
                    warmup: int = 10) -> int | None:
    """First step a downward CUSUM on the held-out oracle reward fires."""
    x = np.asarray(oracle_series, float)
    if x.size <= warmup:
        return None
    mean = float(np.mean(x[:warmup])); std = float(np.std(x[:warmup])) + 1e-6
    S = 0.0
    for t in range(warmup, x.size):
        z = (mean - x[t]) / std          # downward deviation
        S = max(0.0, S + z - k)
        if S > h:
            return t
    return None


def lead_time(onset_step, oracle_turn) -> int | None:
    if onset_step is None or oracle_turn is None:
        return None
    return oracle_turn - onset_step


def aggregate(runs: list[dict]) -> dict:
    onset_runs = [r for r in runs if not r["is_hard_negative"]]
    hard_negs = [r for r in runs if r["is_hard_negative"]]
    detected = [r for r in onset_runs if r["onset_step"] is not None]
    fps = [r for r in hard_negs if r["onset_step"] is not None]
    leads = [lead_time(r["onset_step"], r["oracle_turn"]) for r in detected]
    leads = [l for l in leads if l is not None]
    return {
        "n_onset": len(onset_runs), "n_hard_neg": len(hard_negs),
        "detection_rate": len(detected) / len(onset_runs) if onset_runs else 0.0,
        "fpr": len(fps) / len(hard_negs) if hard_negs else 0.0,
        "mean_lead": float(np.mean(leads)) if leads else 0.0,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_evaluator.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add flightrecorder/eval tests/test_evaluator.py
git commit -m "feat: evaluator (oracle-gap turn, lead time, FPR/detection rate)"
```

---

## Task 11: Synthetic generator (`repro/synthetic.py`)

**Files:**
- Create: `flightrecorder/repro/__init__.py`, `flightrecorder/repro/synthetic.py`
- Test: `tests/test_synthetic.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_synthetic.py
from flightrecorder.repro.synthetic import generate_run


def test_onset_run_has_geometry_break_after_tstar():
    run = generate_run(seed=1, n_steps=120, tstar=60, hard_negative=False)
    rfs = run["rollout_frames"]
    early = sum(rf.kl_accel for rf in rfs[20:55]) / 35
    late = sum(rf.kl_accel for rf in rfs[65:100]) / 35
    assert late > early + 0.5                 # kl accelerates after onset
    assert run["oracle_frames"][-1].scissors > 1.0  # scissors opened
    assert run["tstar"] == 60


def test_hard_negative_has_no_sustained_break_and_no_oracle_turn():
    run = generate_run(seed=2, n_steps=120, hard_negative=True)
    assert run["tstar"] is None
    oracle_series = [of.oracle_reward for of in run["oracle_frames"]]
    from flightrecorder.eval.evaluator import oracle_gap_turn
    assert oracle_gap_turn(oracle_series, warmup=20) is None


def test_runs_are_deterministic():
    a = generate_run(seed=7, n_steps=50, tstar=30)
    b = generate_run(seed=7, n_steps=50, tstar=30)
    assert a["rollout_frames"][40].kl_accel == b["rollout_frames"][40].kl_accel
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_synthetic.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `repro/synthetic.py`**

```python
# flightrecorder/repro/__init__.py
```

```python
# flightrecorder/repro/synthetic.py
"""HARNESS VALIDATION ONLY (§8). Authored RolloutFrame/OracleFrame trajectories with a
ground-truth onset (or t*=inf hard negatives). NOT evidence the signal exists in real runs."""
from __future__ import annotations

import numpy as np
from ..types import RolloutFrame, OracleFrame
from ..core.oracle import OracleExtractor


def _logistic(t, t0, sharp):
    return 1.0 / (1.0 + np.exp(-sharp * (t - t0)))


def generate_run(seed: int, n_steps: int = 200, tstar: int | None = 100,
                 hard_negative: bool = False, sharpness: float = 0.3,
                 noise: float = 0.05) -> dict:
    """Returns {rollout_frames, oracle_frames, tstar, is_hard_negative}."""
    rng = np.random.default_rng(seed)
    oracle_ext = OracleExtractor(alpha=0.4)
    rollout_frames, oracle_frames = [], []
    spike_at = rng.integers(40, n_steps - 40) if hard_negative else -1

    for t in range(n_steps):
        on = 0.0 if (hard_negative or tstar is None) else _logistic(t, tstar, sharpness)
        # transient benign spike for hard negatives (recovers, no oracle turn)
        spike = 0.0
        if hard_negative and 0 <= spike_at <= t < spike_at + 6:
            spike = 1.5 * np.exp(-(t - spike_at))

        kl_accel = on * 2.5 + spike + rng.normal(0, noise)
        entropy_trend = -on * 1.2 + rng.normal(0, noise)
        adv_drift = on * 3.0 + spike + abs(rng.normal(0, noise))
        train_reward = 0.2 + 0.6 * _logistic(t, n_steps * 0.4, 0.05) + rng.normal(0, noise)
        # oracle keeps up until onset, then stalls/declines
        oracle_reward = train_reward - on * (0.5 + 0.3 * _logistic(t, tstar or 0, sharpness)) \
            if not hard_negative else train_reward + rng.normal(0, noise)

        rf = RolloutFrame(
            step=t, kl_mean=float(on * 5 + spike), kl_slope=float(on),
            kl_accel=float(kl_accel), entropy_mean=float(3.0 - on * 2),
            entropy_trend=float(entropy_trend), adv_var=float(1 + on),
            adv_skew=float(on), adv_kurtosis=float(on * 2), adv_drift=float(adv_drift),
            train_reward=float(train_reward), train_reward_slope=0.0,
            gen_len_mean=float(50 - on * 20), gen_len_var=float(5 + on * 10),
            ngram_diversity=float(max(0.0, 0.8 - on * 0.5)),
            logprob_concentration=float(min(1.0, 0.3 + on * 0.6)))
        od = oracle_ext.update(float(train_reward), float(oracle_reward))
        of = OracleFrame(step=t, synthetic_tstar=(None if hard_negative else tstar), **od)
        rollout_frames.append(rf); oracle_frames.append(of)

    return {"rollout_frames": rollout_frames, "oracle_frames": oracle_frames,
            "tstar": (None if hard_negative else tstar),
            "is_hard_negative": hard_negative}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_synthetic.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add flightrecorder/repro tests/test_synthetic.py
git commit -m "feat: synthetic harness (onset runs + hard negatives)"
```

---

## Task 12: Harness-level thesis test (oracle-blind)

Encodes the §13 property: the oracle-blind tube detector leads the oracle-gap turn and beats the oracle-blind baselines at fixed FPR over a seed set. **Harness-level, not the thesis.**

**Files:**
- Test: `tests/test_eval_property.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_eval_property.py
from flightrecorder.repro.synthetic import generate_run
from flightrecorder.eval.evaluator import oracle_gap_turn, aggregate
from flightrecorder.detector.contraction_tube import ContractionTubeDetector
from flightrecorder.detector.cusum import CusumDetector
from flightrecorder.detector.threshold import ThresholdDetector


def _run_detector(make_det, run):
    det = make_det()
    onset = None
    for rf in run["rollout_frames"]:
        st = det.update(rf)
        if st.onset and onset is None:
            onset = st.onset_step
    turn = oracle_gap_turn([of.oracle_reward for of in run["oracle_frames"]], warmup=20)
    return {"onset_step": onset, "oracle_turn": turn,
            "is_hard_negative": run["is_hard_negative"]}


def _suite():
    runs = [generate_run(seed=s, n_steps=200, tstar=100) for s in range(10)]
    runs += [generate_run(seed=100 + s, n_steps=200, hard_negative=True) for s in range(10)]
    return runs


def test_tube_detector_positive_lead_and_beats_baselines():
    runs = _suite()
    tube = aggregate([_run_detector(lambda: ContractionTubeDetector(warmup=30), r) for r in runs])
    cusum = aggregate([_run_detector(lambda: CusumDetector(warmup=30), r) for r in runs])
    thr = aggregate([_run_detector(lambda: ThresholdDetector(warmup=30), r) for r in runs])

    assert tube["mean_lead"] > 0, "tube must fire before the oracle-gap turn"
    assert tube["fpr"] <= 0.2, "tube FPR ceiling on hard negatives"
    # tube dominates at fixed FPR: not worse lead while keeping FPR no higher
    assert tube["mean_lead"] >= cusum["mean_lead"] or tube["fpr"] < cusum["fpr"]
    assert tube["fpr"] <= thr["fpr"], "naive threshold should false-positive at least as much"
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/test_eval_property.py -v`
Expected: PASS. If `mean_lead` is marginal or FPR too high, tune detector defaults (`warmup`, `tau`, `h`, `rho`) — these are the thresholds frozen before any real run (§14 threshold provenance). Record chosen defaults in a comment.

- [ ] **Step 3: Commit**

```bash
git add tests/test_eval_property.py
git commit -m "test: harness-level thesis (oracle-blind tube leads + beats baselines)"
```

---

## Task 13: CLI (`flr synth`, `flr eval`)

**Files:**
- Create: `flightrecorder/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
import json
from flightrecorder.cli import main


def test_synth_writes_artifact(tmp_path):
    out = tmp_path / "run.jsonl"
    rc = main(["synth", "--seed", "1", "--steps", "120", "--tstar", "60", "--out", str(out)])
    assert rc == 0 and out.exists()
    kinds = {json.loads(l)["kind"] for l in out.read_text().splitlines()}
    assert {"frame", "oracle", "detector"} <= kinds


def test_eval_prints_summary(tmp_path, capsys):
    rc = main(["eval", "--seeds", "4", "--steps", "150", "--tstar", "80"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "mean_lead" in out and "fpr" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError`

- [ ] **Step 3: Write `flightrecorder/cli.py`**

```python
# flightrecorder/cli.py
"""flr CLI. V1 subcommands: synth (write an artifact), eval (harness summary)."""
from __future__ import annotations

import argparse, json, sys
from .repro.synthetic import generate_run
from .recorder import Recorder
from .sinks.jsonl import JSONLSink
from .eval.evaluator import oracle_gap_turn, aggregate
from .detector.contraction_tube import ContractionTubeDetector
from .detector.cusum import CusumDetector
from .detector.threshold import ThresholdDetector

_DETECTORS = {"tube": ContractionTubeDetector, "cusum": CusumDetector,
              "threshold": ThresholdDetector}


def _cmd_synth(a) -> int:
    run = generate_run(seed=a.seed, n_steps=a.steps, tstar=a.tstar,
                       hard_negative=a.hard_negative)
    rec = Recorder(detector=ContractionTubeDetector(warmup=a.warmup), sinks=[JSONLSink(a.out)])
    for rf, of in zip(run["rollout_frames"], run["oracle_frames"]):
        rec.record_frames(rf, of)
    rec.close()
    print(f"wrote {a.out} ({a.steps} steps, tstar={run['tstar']})")
    return 0


def _eval_one(make_det, run, warmup):
    det = make_det(warmup=warmup); onset = None
    for rf in run["rollout_frames"]:
        st = det.update(rf)
        if st.onset and onset is None:
            onset = st.onset_step
    turn = oracle_gap_turn([of.oracle_reward for of in run["oracle_frames"]], warmup=20)
    return {"onset_step": onset, "oracle_turn": turn,
            "is_hard_negative": run["is_hard_negative"]}


def _cmd_eval(a) -> int:
    runs = [generate_run(seed=s, n_steps=a.steps, tstar=a.tstar) for s in range(a.seeds)]
    runs += [generate_run(seed=1000 + s, n_steps=a.steps, hard_negative=True)
             for s in range(a.seeds)]
    summary = {name: aggregate([_eval_one(cls, r, a.warmup) for r in runs])
               for name, cls in _DETECTORS.items()}
    print(json.dumps(summary, indent=2))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="flr")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="write a synthetic run artifact")
    s.add_argument("--seed", type=int, default=0); s.add_argument("--steps", type=int, default=200)
    s.add_argument("--tstar", type=int, default=100); s.add_argument("--warmup", type=int, default=30)
    s.add_argument("--hard-negative", action="store_true"); s.add_argument("--out", default="run.jsonl")
    s.set_defaults(func=_cmd_synth)

    e = sub.add_parser("eval", help="harness-level detector comparison")
    e.add_argument("--seeds", type=int, default=10); e.add_argument("--steps", type=int, default=200)
    e.add_argument("--tstar", type=int, default=100); e.add_argument("--warmup", type=int, default=30)
    e.set_defaults(func=_cmd_eval)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: 2 passed

- [ ] **Step 5: Run the full suite + the CLI end to end**

Run: `uv run pytest -q && uv run flr eval --seeds 6 --steps 160 --tstar 80`
Expected: all tests green; JSON summary shows `tube.mean_lead > 0` and `tube.fpr` ≤ baselines.

- [ ] **Step 6: Commit**

```bash
git add flightrecorder/cli.py tests/test_cli.py
git commit -m "feat: flr CLI (synth + eval)"
```

---

## Task 14: README quickstart

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
# Flight Recorder

Reward-hacking **onset detection** for RL post-training (GRPO). The detector reads
**rollout geometry only** (KL acceleration, entropy collapse, advantage drift) and is
**oracle-blind by construction** — it never sees the held-out oracle. The novel claim:
geometry fires *before* the oracle gap becomes visible.

## Install
\`\`\`bash
uv venv && uv pip install -e ".[dev]"
\`\`\`

## Quickstart (GPU-free harness validation)
\`\`\`bash
flr synth --seed 1 --steps 200 --tstar 100 --out run.jsonl   # write an artifact
flr eval --seeds 10 --steps 200 --tstar 100                  # compare detectors
\`\`\`
`eval` reports, per detector, detection rate, FPR on hard negatives, and **mean lead
time vs the oracle-gap turn**. The synthetic harness validates plumbing and tunes FPR;
it is **not** evidence the signal exists in real runs — that is gated on a real
TRL+Qwen run (see the spec, §14).

## Status
Core complete, harness-validated, real run launch-ready; **thesis: pending real run.**

License: Apache-2.0
\`\`\`
```

- [ ] **Step 2: Verify quickstart commands run**

Run: `uv run flr synth --seed 1 --steps 120 --tstar 60 --out /tmp/run.jsonl && uv run flr eval --seeds 4 --steps 120 --tstar 60`
Expected: artifact written; summary printed.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: README quickstart"
```

---

## Self-Review

**1. Spec coverage (Plan 1 scope):**
- §4 data model → Task 2 (frames), Task 3 (boundary). ✓
- §5 extractors: KL→T5, entropy/advantage/genstats→T6, oracle→T7. ✓
- §6 detectors: base+threshold+cusum+contraction_tube→T8; oracle-blind interface, geometry-only vector, CUSUM-on-geometry, set_reference for controlled pair. ✓
- §6 lead time vs oracle-gap turn → Task 10 (`oracle_gap_turn`, `lead_time`). ✓
- §8 synthetic + hard negatives + FPR → Task 11; eval → Task 10/12/13. ✓ (reward_testhack, onset_label, sandbox, trl_grpo_qwen are **Plan 2**, correctly deferred.)
- §13 boundary test→T3; harness property→T12; extractor/detector/evaluator unit tests→T4–T11. ✓
- §14 Tier-1 #1 (`flr synth` onset+hard-neg)→T11/T13; #2 (`flr eval` lead+FPR beating baselines)→T12/T13; threshold provenance noted in T12. ✓ (Tier-1 #3,#4 server/UI = Plans 3–4; #5,#6 adapters/sandbox/labeler = Plan 2; Tier-2 #7 real run = next prompt.)

**2. Placeholder scan:** No TBD/TODO; every code/test step shows full code; every run step states expected output. ✓

**3. Type consistency:** `RolloutFrame`/`OracleFrame` field names identical across types.py, extractors, recorder `**dict` unpacking (extractor dict keys match frame fields: kl_mean/kl_slope/kl_accel, entropy_mean/entropy_trend, adv_var/adv_skew/adv_kurtosis/adv_drift, gen_len_mean/gen_len_var/ngram_diversity/logprob_concentration, oracle_reward/oracle_gap/scissors). `DetectorState` built only via `make_state`. Detector ctor kwarg `warmup` consistent across all three (used by `_eval_one(cls, r, a.warmup)` and tests). `generate_run` returns the dict keys used by evaluator/CLI/tests (`rollout_frames`, `oracle_frames`, `tstar`, `is_hard_negative`). ✓

**Note on T12 tuning:** if the harness property test is marginal on first run, the fix is tuning frozen detector defaults (warmup/tau/h/rho) — not changing the test. Those defaults become the pre-registered thresholds per §14.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-06-flight-recorder-core.md`.
