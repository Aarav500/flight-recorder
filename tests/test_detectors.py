import numpy as np
from flightrecorder.types import RolloutFrame
from flightrecorder.detector.threshold import ThresholdDetector
from flightrecorder.detector.cusum import CusumDetector
from flightrecorder.detector.contraction_tube import ContractionTubeDetector


def _frame(step, kl_accel=0.0, entropy_trend=0.0, adv_drift=0.0):
    return RolloutFrame(step=step, kl_mean=0.0, kl_slope=0.0, kl_accel=kl_accel,
                        entropy_mean=0.0, entropy_trend=entropy_trend,
                        adv_var=0.0, adv_skew=0.0, adv_kurtosis=0.0, adv_drift=adv_drift,
                        train_reward=0.0, train_reward_slope=0.0, gen_len_mean=0.0,
                        gen_len_var=0.0, ngram_diversity=0.0, logprob_concentration=0.0)


def _drive(det, healthy_steps=30, shock=True):
    rng = np.random.default_rng(0)
    fired = None
    for t in range(healthy_steps):
        st = det.update(_frame(t, kl_accel=rng.normal(0, 0.05)))
        if st.onset and fired is None:
            fired = st.onset_step
    if shock:
        for t in range(healthy_steps, healthy_steps + 20):
            st = det.update(_frame(t, kl_accel=2.0 + rng.normal(0, 0.05),
                                   entropy_trend=-1.0, adv_drift=3.0))
            if st.onset and fired is None:
                fired = st.onset_step
    return fired


def test_tube_detector_fires_after_shock_not_during_healthy():
    det = ContractionTubeDetector(warmup=20)
    fired = _drive(det)
    assert fired is not None and fired >= 30


def test_tube_detector_silent_on_healthy_only():
    det = ContractionTubeDetector(warmup=20)
    fired = _drive(det, shock=False)
    assert fired is None


def test_baselines_fire_on_shock():
    assert _drive(ThresholdDetector(warmup=20)) is not None
    assert _drive(CusumDetector(warmup=20)) is not None


def test_onset_is_sticky():
    det = ContractionTubeDetector(warmup=20)
    _drive(det)
    st = det.update(_frame(100, kl_accel=2.0, entropy_trend=-1.0, adv_drift=3.0))
    assert st.onset is False and st.onset_step is not None  # not re-declared
