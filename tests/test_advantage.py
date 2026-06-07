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
