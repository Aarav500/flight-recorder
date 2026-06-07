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
