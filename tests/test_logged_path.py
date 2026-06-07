import numpy as np
from flightrecorder.core.kl import KLExtractor
from flightrecorder.core.entropy import EntropyExtractor
from flightrecorder.types import RolloutBatch
from flightrecorder.recorder import Recorder
from flightrecorder.detector.contraction_tube import ContractionTubeDetector


def test_kl_update_scalar_tracks_accel():
    ext = KLExtractor(alpha=1.0)
    for v in (0.0, 1.0, 3.0):
        out = ext.update_scalar(v)
    assert out["kl_mean"] == 3.0 and out["kl_accel"] > 0


def test_entropy_update_scalar():
    ext = EntropyExtractor(alpha=1.0)
    out = ext.update_scalar(2.5)
    assert out["entropy_mean"] == 2.5


def test_recorder_uses_logged_kl_when_no_logprobs():
    rec = Recorder(detector=ContractionTubeDetector(warmup=2), sinks=[])
    rf, _ = rec.record(RolloutBatch(step=0, train_rewards=np.array([1.0]),
                                    logged={"kl": 0.42, "entropy": 1.7}, meta={}))
    assert rf.kl_mean == 0.42 and rf.entropy_mean == 1.7


def test_recorder_kl_nan_when_nothing_available():
    rec = Recorder(detector=ContractionTubeDetector(warmup=2), sinks=[])
    rf, _ = rec.record(RolloutBatch(step=0, train_rewards=np.array([1.0]), meta={}))
    assert np.isnan(rf.kl_mean)
