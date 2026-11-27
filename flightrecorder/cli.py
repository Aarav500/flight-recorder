"""flr CLI. Subcommands: synth (write an artifact), eval (harness summary), serve (API)."""
from __future__ import annotations

import argparse, json, sys
from .repro.synthetic import generate_run
from .recorder import Recorder
from .sinks.jsonl import JSONLSink
from .eval.evaluator import oracle_gap_turn, aggregate
from .detector.cusum import CusumDetector
from .detector.threshold import ThresholdDetector
from .config import load_thresholds, detector_from_thresholds, check_threshold_provenance


def _cmd_synth(a) -> int:
    run = generate_run(seed=a.seed, n_steps=a.steps, tstar=a.tstar,
                       hard_negative=a.hard_negative)
    detector = detector_from_thresholds(load_thresholds(a.thresholds))
    rec = Recorder(detector=detector, sinks=[JSONLSink(a.out)])
    for rf, of in zip(run["rollout_frames"], run["oracle_frames"]):
        rec.record_frames(rf, of)
    rec.close()
    print(f"wrote {a.out} ({a.steps} steps, tstar={run['tstar']})")
    return 0


def _eval_one(make_det, run):
    det = make_det()
    onset = None
    for rf in run["rollout_frames"]:
        st = det.update(rf)
        if st.onset and onset is None:
            onset = st.onset_step
    turn = oracle_gap_turn([of.oracle_reward for of in run["oracle_frames"]], warmup=20)
    return {"onset_step": onset, "oracle_turn": turn,
            "is_hard_negative": run["is_hard_negative"]}


def _cmd_eval(a) -> int:
    # GATE 1: detector thresholds come from the pre-registered config, never tuned here.
    t = load_thresholds(a.thresholds)
    print(f"thresholds: {t.source}  (tube_tau={t.tube_tau}, cusum_h={t.cusum_h}, "
          f"persistence={t.persistence}, warmup={t.warmup})")
    for w in check_threshold_provenance(a.thresholds, None):
        print("!!! " + w)
    factories = {
        "tube": lambda: detector_from_thresholds(t),
        "cusum": lambda: CusumDetector(warmup=t.warmup, k=t.cusum_k, h=t.cusum_h, signal=t.cusum_signal),
        "threshold": lambda: ThresholdDetector(warmup=t.warmup),
    }
    runs = [generate_run(seed=s, n_steps=a.steps, tstar=a.tstar) for s in range(a.seeds)]
    runs += [generate_run(seed=1000 + s, n_steps=a.steps, hard_negative=True)
             for s in range(a.seeds)]
    summary = {name: aggregate([_eval_one(mk, r) for r in runs]) for name, mk in factories.items()}
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_serve(a) -> int:
    try:
        import uvicorn
        from .server.app import create_app
    except Exception as e:
        print(f"flr serve needs extras: pip install 'flightrecorder[server]' ({e})")
        return 1
    print(f"serving runs from {a.runs} on http://{a.host}:{a.port}")
    uvicorn.run(create_app(a.runs), host=a.host, port=a.port)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="flr")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="write a synthetic run artifact")
    s.add_argument("--seed", type=int, default=0); s.add_argument("--steps", type=int, default=200)
    s.add_argument("--tstar", type=int, default=100)
    s.add_argument("--thresholds", default="configs/thresholds.yaml")
    s.add_argument("--hard-negative", action="store_true"); s.add_argument("--out", default="run.jsonl")
    s.set_defaults(func=_cmd_synth)

    e = sub.add_parser("eval", help="harness-level detector comparison")
    e.add_argument("--seeds", type=int, default=10); e.add_argument("--steps", type=int, default=200)
    e.add_argument("--tstar", type=int, default=100)
    e.add_argument("--thresholds", default="configs/thresholds.yaml")
    e.set_defaults(func=_cmd_eval)

    sv = sub.add_parser("serve", help="serve the REST + WebSocket API over saved runs")
    sv.add_argument("--runs", default="runs"); sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(func=_cmd_serve)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
