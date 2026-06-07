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
