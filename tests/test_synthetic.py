from flightrecorder.repro.synthetic import generate_run
from flightrecorder.eval.evaluator import oracle_gap_turn


def test_onset_run_has_geometry_break_after_tstar():
    run = generate_run(seed=1, n_steps=120, tstar=60, hard_negative=False)
    rfs = run["rollout_frames"]
    early = sum(rf.kl_accel for rf in rfs[20:55]) / 35
    late = sum(rf.kl_accel for rf in rfs[65:100]) / 35
    assert late > early + 0.5                 # kl accelerates after onset
    assert run["oracle_frames"][-1].scissors > 0.5  # scissors opened materially
    assert run["tstar"] == 60


def test_hard_negative_has_no_sustained_break_and_no_oracle_turn():
    run = generate_run(seed=2, n_steps=120, hard_negative=True)
    assert run["tstar"] is None
    oracle_series = [of.oracle_reward for of in run["oracle_frames"]]
    assert oracle_gap_turn(oracle_series, warmup=20) is None


def test_runs_are_deterministic():
    a = generate_run(seed=7, n_steps=50, tstar=30)
    b = generate_run(seed=7, n_steps=50, tstar=30)
    assert a["rollout_frames"][40].kl_accel == b["rollout_frames"][40].kl_accel
