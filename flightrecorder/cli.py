"""flr CLI. V1 subcommands: synth (write an artifact), eval (harness summary)."""
from __future__ import annotations

import argparse, json, sys
from .repro.synthetic import generate_run
from .recorder import Recorder
from .sinks.jsonl import JSONLSink
from .eval.evaluator import oracle_gap_turn, aggregate
from .detector.contraction_tube import ContractionTubeDetector
from .detector.cusum import CusumDetector
from .detector.threshold import ThresholdDetector

_DETECTORS = {"tube": ContractionTubeDetector, "cusum": CusumDetector,
              "threshold": ThresholdDetector}


def _cmd_synth(a) -> int:
    run = generate_run(seed=a.seed, n_steps=a.steps, tstar=a.tstar,
                       hard_negative=a.hard_negative)
    rec = Recorder(detector=ContractionTubeDetector(warmup=a.warmup), sinks=[JSONLSink(a.out)])
    for rf, of in zip(run["rollout_frames"], run["oracle_frames"]):
        rec.record_frames(rf, of)
    rec.close()
    print(f"wrote {a.out} ({a.steps} steps, tstar={run['tstar']})")
    return 0


def _eval_one(make_det, run, warmup):
    det = make_det(warmup=warmup); onset = None
    for rf in run["rollout_frames"]:
        st = det.update(rf)
        if st.onset and onset is None:
            onset = st.onset_step
    turn = oracle_gap_turn([of.oracle_reward for of in run["oracle_frames"]], warmup=20)
    return {"onset_step": onset, "oracle_turn": turn,
            "is_hard_negative": run["is_hard_negative"]}


def _cmd_eval(a) -> int:
    runs = [generate_run(seed=s, n_steps=a.steps, tstar=a.tstar) for s in range(a.seeds)]
    runs += [generate_run(seed=1000 + s, n_steps=a.steps, hard_negative=True)
             for s in range(a.seeds)]
    summary = {name: aggregate([_eval_one(cls, r, a.warmup) for r in runs])
               for name, cls in _DETECTORS.items()}
    print(json.dumps(summary, indent=2))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="flr")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="write a synthetic run artifact")
    s.add_argument("--seed", type=int, default=0); s.add_argument("--steps", type=int, default=200)
    s.add_argument("--tstar", type=int, default=100); s.add_argument("--warmup", type=int, default=30)
    s.add_argument("--hard-negative", action="store_true"); s.add_argument("--out", default="run.jsonl")
    s.set_defaults(func=_cmd_synth)

    e = sub.add_parser("eval", help="harness-level detector comparison")
    e.add_argument("--seeds", type=int, default=10); e.add_argument("--steps", type=int, default=200)
    e.add_argument("--tstar", type=int, default=100); e.add_argument("--warmup", type=int, default=30)
    e.set_defaults(func=_cmd_eval)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
