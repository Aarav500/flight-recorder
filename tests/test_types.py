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
