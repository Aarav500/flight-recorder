import numpy as np
from flightrecorder.core.entropy import EntropyExtractor


def test_uses_provided_entropy_mean():
    ext = EntropyExtractor(alpha=1.0)
    out = ext.update(entropy=np.array([2.0, 4.0]), logprobs=None)
    assert abs(out["entropy_mean"] - 3.0) < 1e-9


def test_entropy_trend_negative_on_collapse():
    ext = EntropyExtractor(alpha=1.0)
    for h in (5.0, 4.0, 2.0):
        out = ext.update(entropy=np.array([h]), logprobs=None)
    assert out["entropy_trend"] < 0


def test_falls_back_to_logprob_surprisal():
    ext = EntropyExtractor(alpha=1.0)
    out = ext.update(entropy=None, logprobs=np.array([-1.0, -3.0]))
    assert abs(out["entropy_mean"] - 2.0) < 1e-9  # mean(-logprob)
