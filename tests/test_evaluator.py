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
