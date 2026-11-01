import os
from flightrecorder.config import (
    load_thresholds, detector_from_thresholds, check_threshold_provenance)


def test_load_committed_thresholds_match_pre_registered_values():
    t = load_thresholds("configs/thresholds.yaml")
    assert t.tube_tau == 1.0 and t.cusum_h == 5.0 and t.persistence == 5
    assert t.warmup == 30 and t.cusum_signal == "kl_accel"
    assert t.source.endswith("thresholds.yaml")


def test_detector_built_from_thresholds():
    det = detector_from_thresholds(load_thresholds("configs/thresholds.yaml"))
    assert det.tau == 1.0 and det.h == 5.0 and det.persistence == 5 and det.warmup == 30


def test_defaults_when_path_missing():
    t = load_thresholds(None)
    assert t.source == "defaults" and t.tube_tau == 1.0 and t.persistence == 5


def test_provenance_clean_when_thresholds_predate_artifact(tmp_path):
    th = tmp_path / "thresholds.yaml"; th.write_text("tube_tau: 1.0\n")
    art = tmp_path / "run.jsonl"; art.write_text("{}\n")
    os.utime(th, (1000, 1000)); os.utime(art, (2000, 2000))
    assert check_threshold_provenance(str(th), str(art)) == []


def test_provenance_warns_when_thresholds_modified_after_artifact(tmp_path):
    th = tmp_path / "thresholds.yaml"; th.write_text("tube_tau: 1.0\n")
    art = tmp_path / "run.jsonl"; art.write_text("{}\n")
    os.utime(art, (2000, 2000)); os.utime(th, (3000, 3000))
    warns = check_threshold_provenance(str(th), str(art))
    assert len(warns) == 1 and "MODIFIED AFTER" in warns[0]


def test_provenance_warns_when_config_missing(tmp_path):
    warns = check_threshold_provenance(str(tmp_path / "nope.yaml"), None)
    assert len(warns) == 1 and "MISSING" in warns[0]
