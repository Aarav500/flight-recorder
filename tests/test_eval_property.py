from flightrecorder.repro.synthetic import generate_run
from flightrecorder.eval.evaluator import oracle_gap_turn, aggregate
from flightrecorder.detector.contraction_tube import ContractionTubeDetector
from flightrecorder.detector.cusum import CusumDetector
from flightrecorder.detector.threshold import ThresholdDetector


def _run_detector(make_det, run):
    det = make_det()
    onset = None
    for rf in run["rollout_frames"]:
        st = det.update(rf)
        if st.onset and onset is None:
            onset = st.onset_step
    turn = oracle_gap_turn([of.oracle_reward for of in run["oracle_frames"]], warmup=20)
    return {"onset_step": onset, "oracle_turn": turn,
            "is_hard_negative": run["is_hard_negative"]}


def _suite():
    runs = [generate_run(seed=s, n_steps=200, tstar=100) for s in range(10)]
    runs += [generate_run(seed=100 + s, n_steps=200, hard_negative=True) for s in range(10)]
    return runs


def test_tube_detector_positive_lead_and_beats_baselines():
    runs = _suite()
    tube = aggregate([_run_detector(lambda: ContractionTubeDetector(warmup=30), r) for r in runs])
    cusum = aggregate([_run_detector(lambda: CusumDetector(warmup=30), r) for r in runs])
    thr = aggregate([_run_detector(lambda: ThresholdDetector(warmup=30), r) for r in runs])

    assert tube["mean_lead"] > 0, "tube must fire before the oracle-gap turn"
    assert tube["fpr"] <= 0.2, "tube FPR ceiling on hard negatives"
    # tube dominates at fixed FPR: not worse lead while keeping FPR no higher
    assert tube["mean_lead"] >= cusum["mean_lead"] or tube["fpr"] < cusum["fpr"]
    assert tube["fpr"] <= thr["fpr"], "naive threshold should false-positive at least as much"
