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
    # base.py is written in Task 8; guard the import so this file passes standalone too.
    try:
        from flightrecorder.detector.base import Detector
    except ModuleNotFoundError:
        return
    sig = inspect.signature(Detector.update)
    ann = sig.parameters["frame"].annotation
    assert getattr(ann, "__name__", str(ann)) == "RolloutFrame"
