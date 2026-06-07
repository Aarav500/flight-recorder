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
