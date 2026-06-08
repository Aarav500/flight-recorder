import numpy as np
from flightrecorder.types import RolloutFrame, OracleFrame
from flightrecorder.repro.integrity import (
    StepRecord, run_integrity_report, format_report,
    check1_callback_captures, check2_discovers_hack, check3_sandbox_isolation,
    check4_behavioral_labeler)
from flightrecorder.repro.reward_testhack import CORRECT, HACK


def _rf(step, kl, ent, advvar):
    return RolloutFrame(step=step, kl_mean=kl, kl_slope=0, kl_accel=0, entropy_mean=ent,
                        entropy_trend=0, adv_var=advvar, adv_skew=0, adv_kurtosis=0,
                        adv_drift=0, train_reward=0, train_reward_slope=0, gen_len_mean=0,
                        gen_len_var=0, ngram_diversity=0, logprob_concentration=0)


def _rec(step, *, hacking, kl, ent, advvar, train, oracle):
    comps = [HACK, HACK] if hacking else [CORRECT, CORRECT]
    return StepRecord(step=step, rollout=_rf(step, kl, ent, advvar),
                      oracle=OracleFrame(step=step, oracle_reward=float(np.mean(oracle)),
                                         oracle_gap=0.0, scissors=0.0),
                      completions=comps,
                      train_rewards=np.array(train, float),
                      oracle_rewards=np.array(oracle, float))


def _good_run(n=20):
    recs = []
    for s in range(n):
        hacking = s >= 10
        recs.append(_rec(
            s, hacking=hacking, kl=0.1 * s, ent=2.0 - 0.05 * s, advvar=1.0 + 0.01 * s,
            train=[0.9, 0.9] if hacking else [0.2, 0.2],
            oracle=[0.1, 0.1] if hacking else [0.2, 0.2]))
    return recs


def test_good_run_passes_all_four_checks():
    results = run_integrity_report(_good_run(), step_budget=20, oracle_turn_step=12)
    assert all(r.passed for r in results), format_report(results)


def test_check1_fails_on_degenerate_capture():
    recs = [_rec(s, hacking=False, kl=0.0, ent=0.0, advvar=0.0, train=[0.1], oracle=[0.1])
            for s in range(10)]
    assert check1_callback_captures(recs).passed is False


def test_check2_fails_when_hack_never_discovered():
    recs = [_rec(s, hacking=False, kl=0.1 * s, ent=2.0, advvar=1.0, train=[0.2], oracle=[0.2])
            for s in range(20)]
    assert check2_discovers_hack(recs, step_budget=20).passed is False


def test_check2_fails_when_hack_too_late():
    recs = _good_run(20)
    # move all hacking to the final two steps (>= 0.8*20 = 16)
    for i, r in enumerate(recs):
        r.completions = [HACK, HACK] if i >= 18 else [CORRECT, CORRECT]
    assert check2_discovers_hack(recs, step_budget=20).passed is False


def test_check3_fails_when_oracle_tracks_train():
    recs = []
    for s in range(20):
        v = 0.1 + 0.04 * s
        recs.append(_rec(s, hacking=s >= 10, kl=0.1 * s, ent=2.0, advvar=1.0,
                         train=[v, v], oracle=[v, v]))  # oracle == train
    assert check3_sandbox_isolation(recs).passed is False


def test_check4_fails_without_behavioral_onset():
    recs = [_rec(s, hacking=False, kl=0.1 * s, ent=2.0, advvar=1.0, train=[0.2], oracle=[0.2])
            for s in range(20)]
    assert check4_behavioral_labeler(recs, oracle_turn_step=12).passed is False


def test_report_formats_pass_fail_lines():
    text = format_report(run_integrity_report(_good_run(), 20, 12))
    assert "CHECK 1" in text and "CHECK 4" in text and "OVERALL: PASS" in text
