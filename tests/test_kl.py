import numpy as np
from flightrecorder.core.kl import KLExtractor


def test_k1_kl_is_mean_logprob_diff():
    ext = KLExtractor(estimator="k1", alpha=1.0)
    lp = np.array([-1.0, -2.0]); rlp = np.array([-1.5, -2.5])
    out = ext.update(lp, rlp)
    assert abs(out["kl_mean"] - 0.5) < 1e-9   # mean((lp - rlp)) = 0.5


def test_kl_accel_positive_when_kl_curves_up():
    ext = KLExtractor(estimator="k1", alpha=1.0)
    # kl_mean = lp - rlp; feed an upward-curving series 0, 1, 3 (slope 1 then 2)
    for kl in (0.0, 1.0, 3.0):
        ext.update(np.array([kl]), np.array([0.0]))
    out = ext.update(np.array([6.0]), np.array([0.0]))  # series ->6, slope 3 => accel +1
    assert out["kl_accel"] > 0


def test_missing_inputs_return_nan_no_crash():
    ext = KLExtractor()
    out = ext.update(None, None)
    assert np.isnan(out["kl_mean"])
