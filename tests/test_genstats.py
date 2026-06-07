from flightrecorder.core.genstats import GenStatsExtractor


def test_len_and_diversity():
    ext = GenStatsExtractor()
    out = ext.update(completions=["a b c", "a b c d"], logprobs=None)
    assert out["gen_len_mean"] == 3.5
    assert 0.0 <= out["ngram_diversity"] <= 1.0


def test_logprob_concentration_from_logprobs():
    ext = GenStatsExtractor()
    out = ext.update(completions=["x"], logprobs=[0.0])  # exp(0)=1 fully concentrated
    assert abs(out["logprob_concentration"] - 1.0) < 1e-9


def test_empty_safe():
    ext = GenStatsExtractor()
    out = ext.update(completions=None, logprobs=None)
    assert out["gen_len_mean"] == 0.0
