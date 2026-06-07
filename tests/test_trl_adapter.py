import numpy as np
from types import SimpleNamespace
from flightrecorder.adapters.base import group_normalized_advantages
from flightrecorder.adapters.trl import TRLCoordinator, wrap_reward_fns, FlightRecorderCallback
from flightrecorder.recorder import Recorder
from flightrecorder.detector.contraction_tube import ContractionTubeDetector


class CapturingSink:
    def __init__(self): self.events = []
    def emit(self, e): self.events.append(e)
    def close(self): pass


def test_group_normalized_advantages():
    adv = group_normalized_advantages([0.0, 1.0, 2.0, 3.0], group_size=2)
    assert adv is not None and adv.shape == (4,)
    assert abs(adv[0] + adv[1]) < 1e-6        # each group is mean-zero
    assert group_normalized_advantages([1, 2, 3], group_size=2) is None  # uneven


def test_wrap_reward_fns_returns_train_and_stashes_oracle():
    coord = TRLCoordinator()
    train_fn = lambda prompts, completions, **kw: [1.0, 0.0]
    oracle_fn = lambda prompts, completions, **kw: [1.0, 1.0]
    wrapped = wrap_reward_fns(coord, train_fn, oracle_fn)
    out = wrapped(prompts=["p", "p"], completions=["a", "b"])
    assert out == [1.0, 0.0]                   # TRAIN rewards returned to the trainer
    c, t, o = coord.drain()
    assert list(t) == [1.0, 0.0] and list(o) == [1.0, 1.0]  # oracle captured, not returned


def test_callback_records_batch_with_logged_kl_and_oracle_gap():
    sink = CapturingSink()
    rec = Recorder(detector=ContractionTubeDetector(warmup=2), sinks=[sink])
    coord = TRLCoordinator()
    wrapped = wrap_reward_fns(
        coord,
        train_fn=lambda prompts, completions, **kw: [1.0, 1.0, 0.0, 0.0],
        oracle_fn=lambda prompts, completions, **kw: [0.0, 0.0, 0.0, 0.0])
    wrapped(prompts=["p"] * 4, completions=["x", "y", "z", "w"])
    state = SimpleNamespace(global_step=7, log_history=[{"kl": 0.5, "entropy": 1.2}])
    FlightRecorderCallback(rec, coord, group_size=2).on_step_end(state=state, control="CTRL")
    frames = [e for e in sink.events if e.kind == "frame"]
    oracles = [e for e in sink.events if e.kind == "oracle"]
    assert frames and frames[0].payload["step"] == 7
    assert frames[0].payload["kl_mean"] == 0.5                  # from the logged scalar
    assert oracles and abs(oracles[0].payload["oracle_gap"] - 0.5) < 1e-9  # .5 train - 0 oracle


def test_callback_noop_when_nothing_stashed():
    cb = FlightRecorderCallback(
        Recorder(detector=ContractionTubeDetector(warmup=2), sinks=[CapturingSink()]),
        TRLCoordinator())
    assert cb.on_step_end(state=SimpleNamespace(global_step=1, log_history=[]), control="C") == "C"
