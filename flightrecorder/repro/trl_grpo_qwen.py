"""Launch-ready GRPO + small-Qwen run on the test-overwrite task, wired through the TRL
adapter and the sandboxed reward pair (spec §8).

Scale is a LAUNCH FLAG, not a code change:
  - shakeout (~$4 ceiling): 1 small model, tiny steps, 1 seed, gameable reward only.
    The result is the four-check pipeline-integrity report, not a lead-time number.
  - full ($30-90): controlled pair (gameable vs robust reference) + seeds + hard negatives.

Detector thresholds are read from a pre-registered config (GATE 1); they are never tuned
to a run. Heavy deps (torch/trl/transformers/datasets) are imported lazily inside run(),
so this module imports and unit-tests with none of them installed.
"""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass

from .reward_testhack import SQUARE_TASK, HARD_TASK, CodeTask, make_reward_fns
from .reward_weakverifier import OVERFIT_TASK
from .reward_benign import CONST_TASK, COMPLEX_TASK, MAXRUN_TASK, SECONDMAX_TASK
from ..config import load_thresholds, detector_from_thresholds, check_threshold_provenance


def task_for(config: "RunConfig") -> CodeTask:
    """pilot -> weak-verifier OVERFIT task; benign -> a well-specified non-hacking task
    (const trivial target | complex structured target) for the FPR measurement; HARD_TASK
    (test-overwrite) stays available for a future demo; everything else uses square."""
    if config.scale == "pilot":
        return OVERFIT_TASK
    if config.scale == "benign":
        return {"const": CONST_TASK, "complex": COMPLEX_TASK,
                "maxrun": MAXRUN_TASK, "secondmax": SECONDMAX_TASK}.get(
                    config.benign_task, COMPLEX_TASK)
    return SQUARE_TASK


@dataclass
class RunConfig:
    model: str
    steps: int
    num_generations: int                 # GRPO group size
    learning_rate: float
    seeds: list[int]
    reward_mode: str                     # "gameable" (hackable) | "robust" (reference)
    hard_negatives: bool
    out_dir: str
    scale: str
    thresholds_path: str | None = "configs/thresholds.yaml"
    integrity_report: bool = False
    prompts_per_epoch: int = 64
    beta: float = 0.0                    # GRPO KL-penalty coeff; >0 makes TRL log `kl`
    seed_hack_from: int | None = None    # shakeout-only: seed a known hack from this call
    onset_mode: str = "behavioral"       # "behavioral" (AST overwrite) | "oracle" (overfit)
    curriculum_switch: int | None = None  # pilot: ENGINEERED warm-start length (calls)
    benign_task: str = "complex"         # benign scale default: "complex" (lus2) is the
    #                                      calibration-confirmed boundary task; "const" saturates


def build_config(scale: str, out_dir: str = "runs", model: str | None = None,
                 thresholds_path: str | None = "configs/thresholds.yaml",
                 integrity_report: bool = False) -> RunConfig:
    common = dict(out_dir=out_dir, scale=scale, thresholds_path=thresholds_path,
                  integrity_report=integrity_report)
    if scale == "shakeout":
        # Plumbing test: KL penalty on (beta>0) so `kl` is logged; lr high enough that KL
        # moves; the hack is SEEDED into the recorded stream (no model emergently hacks a
        # trivial task in 20 steps). The result is the four PASS/FAIL checks, not a number.
        return RunConfig(model=model or "Qwen/Qwen2.5-0.5B", steps=20, num_generations=4,
                         learning_rate=5e-5, seeds=[0], reward_mode="gameable",
                         hard_negatives=False, beta=0.04, seed_hack_from=8, **common)
    if scale == "pilot":
        # Emergent budget-sizing pilot: NO seeding. A coder-instruct model on a HARD gameable
        # task (see task_for) that it usually cannot solve, with the overwrite affordance in
        # the prompt. The question this run answers: at what step does the held-out oracle
        # turn (oracle_turn_step go non-null)? That sizes the full run's budget. Single seed.
        return RunConfig(model=model or "Qwen/Qwen2.5-Coder-1.5B-Instruct", steps=120,
                         num_generations=8, learning_rate=1e-5, seeds=[0],
                         reward_mode="gameable", hard_negatives=False, beta=0.04,
                         seed_hack_from=None, onset_mode="oracle", curriculum_switch=35,
                         **common)
    if scale == "benign":
        # FPR / cry-wolf measurement: NO hack, NO seeding, NO curriculum. BASE model (starts
        # high-entropy so an entropy collapse is observable; the Instruct variant was already
        # pre-collapsed at entropy ~0.2 and froze). We measure whether the oracle-blind detector
        # trips on benign convergence (any trigger is a false positive). --task picks the target.
        return RunConfig(model=model or "Qwen/Qwen2.5-1.5B", steps=120,
                         num_generations=8, learning_rate=1e-5, seeds=[0],
                         reward_mode="gameable", hard_negatives=False, beta=0.04,
                         seed_hack_from=None, onset_mode="behavioral", curriculum_switch=None,
                         **common)
    if scale == "full":
        # Real experiment: NO seeding -- emergence is the question. beta>0 for KL capture.
        return RunConfig(model=model or "Qwen/Qwen2.5-1.5B", steps=400, num_generations=8,
                         learning_rate=1e-6, seeds=[0, 1, 2], reward_mode="gameable",
                         hard_negatives=True, beta=0.04, seed_hack_from=None, **common)
    raise ValueError(f"unknown scale: {scale!r} (use 'shakeout', 'pilot', 'benign', or 'full')")


def make_components(config: RunConfig, task: CodeTask = SQUARE_TASK, seed: int = 0,
                    collector: list | None = None) -> dict:
    """Build recorder + TRL coordinator + callback + wrapped reward fn. Uses only
    flightrecorder + stdlib -- no torch -- so it is unit-testable on a laptop."""
    from ..recorder import Recorder
    from ..sinks.jsonl import JSONLSink
    from ..sinks.console import ConsoleSink
    from ..adapters.trl import TRLCoordinator, wrap_reward_fns, FlightRecorderCallback

    os.makedirs(config.out_dir, exist_ok=True)
    artifact = os.path.join(config.out_dir, f"{config.scale}_{config.reward_mode}_seed{seed}.jsonl")
    detector = detector_from_thresholds(load_thresholds(config.thresholds_path))
    recorder = Recorder(detector=detector, sinks=[JSONLSink(artifact), ConsoleSink()])
    coord = TRLCoordinator()
    train_fn, oracle_fn = make_reward_fns(task)
    # gameable: optimize the visible (overwritable) tests; measure with the held-out oracle.
    # robust: optimize the held-out verifier (the model can't see/overwrite it) -- the
    #         non-hacking reference whose frames the tube is fit on.
    if config.reward_mode == "gameable":
        optimize_fn, measure_fn = train_fn, oracle_fn
    else:
        optimize_fn, measure_fn = oracle_fn, train_fn
    from .reward_testhack import HACK
    seed_spec = ({"from_call": config.seed_hack_from, "hack_text": HACK}
                 if config.seed_hack_from is not None else None)
    # Curriculum only applies to the gameable run (the robust reference optimises the strong
    # verifier throughout). It optimises measure_fn (strong) first, then optimize_fn (weak).
    curriculum = config.curriculum_switch if config.reward_mode == "gameable" else None
    reward_fn = wrap_reward_fns(coord, optimize_fn, measure_fn, seed_hack=seed_spec,
                                curriculum_switch=curriculum)
    callback = FlightRecorderCallback(recorder, coord, group_size=config.num_generations,
                                      collector=collector)
    return {"recorder": recorder, "coordinator": coord, "callback": callback,
            "reward_fn": reward_fn, "artifact": artifact, "task": task}


def emit_integrity_report(collector: list, config: RunConfig, artifact: str) -> int:
    """Print the four integrity checks (and any GATE-1 provenance warnings) BEFORE any
    lead-time output. Returns 0 if all checks pass, else 1."""
    from ..eval.evaluator import oracle_gap_turn
    from .integrity import run_integrity_report, format_report

    if config.seed_hack_from is not None:
        print(f"\n[shakeout] NOTE — the test-overwrite hack was SEEDED from reward-call "
              f"{config.seed_hack_from}: >=50% of each later step's completions are scored as "
              f"the hack, and that seeded reward both feeds the recorder AND drives the "
              f"optimizer (so the policy moves and KL is non-zero). This is a PLUMBING test of "
              f"the capture->label->isolation->report path on a known hack; it does NOT claim "
              f"the model emergently discovered it (that is a full-run question).\n")

    for w in check_threshold_provenance(config.thresholds_path, artifact):
        print("\n!!! " + w + "\n")

    oracle_series = [sr.oracle.oracle_reward for sr in collector if sr.oracle is not None]
    warmup = min(20, max(2, len(oracle_series) // 5)) if oracle_series else 2
    oracle_turn = oracle_gap_turn(oracle_series, warmup=warmup) if oracle_series else None

    results = run_integrity_report(collector, config.steps, oracle_turn, config.onset_mode)
    print(format_report(results))

    # Budget-sizing block: the point of the pilot is WHEN the held-out oracle turns, and how
    # far the oracle-blind detector onset leads it. Replay the pre-registered detector over
    # the captured rollout frames to get its onset step (the detector only ever sees rollout
    # geometry). All of this is NOT-EVIDENCE: single seed, pilot scale.
    thr = load_thresholds(config.thresholds_path)
    det = detector_from_thresholds(thr)
    onset_step = None
    exceed = total = 0                                  # instantaneous trigger-rate (FPR proxy)
    for sr in collector:
        st = det.update(sr.rollout)
        if st.onset and onset_step is None:
            onset_step = st.onset_step
        if sr.step >= thr.warmup:
            total += 1
            if st.tube_distance > thr.tube_tau and st.cusum > thr.cusum_h:
                exceed += 1
    trigger_rate = (exceed / total) if total else 0.0
    traj = [round(sr.oracle.oracle_reward, 4) for sr in collector if sr.oracle is not None]
    bar = "=" * 68
    print("\n" + bar)
    print("BUDGET-SIZING (NOT-EVIDENCE — single seed, pilot scale)")
    print(bar)
    if config.curriculum_switch is not None:
        print(f"  *** ENGINEERED TURN: a curriculum warm-start optimised the STRONG verifier")
        print(f"      for the first {config.curriculum_switch} steps, then switched to the weak")
        print(f"      verifier. The oracle turn below is a CONSTRUCTED consequence of that")
        print(f"      switch — it sizes the full run's step/warmup budget. It is NOT evidence")
        print(f"      that hacking emerges on its own.")
        print(f"  curriculum_switch_step: {config.curriculum_switch}")
    print(f"  steps_run:        {config.steps}")
    print(f"  oracle_turn_step: {oracle_turn}")
    print(f"  detector_warmup:  {thr.warmup}  (detector cannot fire before this)")
    if oracle_turn is not None:
        room = oracle_turn - thr.warmup
        rel = (f"{room} steps AFTER warmup — geometry has room to lead" if room > 0
               else f"{-room} steps BEFORE/at warmup — geometry has NO room to lead; "
                     f"increase the full run's budget so the turn lands well after warmup")
        print(f"  turn vs warmup:   {rel}")
    print(f"  detector onset:   {onset_step}")
    if oracle_turn is not None and onset_step is not None:
        print(f"  lead = oracle_turn - onset = {oracle_turn - onset_step}   (NOT-EVIDENCE)")
    elif oracle_turn is None:
        print("  oracle never turned in this budget — trajectory below shows the trend.")
    print(f"  oracle_reward per step: {traj}")
    print(bar)

    # Convergence SHARPNESS — measured alongside firing. The deliverable is the relationship:
    # at what benign sharpness does the detector start tripping (the operating point a later
    # hack must exceed to be distinguishable).
    def _fin(xs):
        return [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]
    ka = _fin([sr.rollout.kl_accel for sr in collector])
    ks = _fin([sr.rollout.kl_slope for sr in collector])
    et = _fin([sr.rollout.entropy_trend for sr in collector])
    em = _fin([sr.rollout.entropy_mean for sr in collector])
    peak_kl_slope = max(ks) if ks else float("nan")
    peak_kl_accel = max(ka) if ka else float("nan")
    peak_entropy_drop_rate = -min(et) if et else float("nan")   # fastest per-step entropy fall
    entropy_total_drop = (em[0] - em[-1]) if len(em) >= 2 else float("nan")

    # Detector trigger-rate. On a BENIGN run (scale=benign) ANY trigger is a false positive —
    # this is the cry-wolf / FPR measurement that simulation cannot answer.
    print("\n" + bar)
    if config.scale == "benign":
        print(f"DETECTOR FPR — BENIGN run (task={config.benign_task}, lr={config.learning_rate:g}): "
              f"ANY trigger is a FALSE POSITIVE")
    else:
        print("DETECTOR TRIGGER")
    print(bar)
    print(f"  onset_step (latched):              {onset_step}")
    print(f"  instantaneous trigger_rate:        {trigger_rate:.3f}  ({exceed}/{total} post-warmup steps)")
    print(f"  SHARPNESS peak_kl_slope={peak_kl_slope:.3f}  peak_kl_accel={peak_kl_accel:.3f}  "
          f"peak_entropy_drop_rate={peak_entropy_drop_rate:.3f}  entropy_total_drop={entropy_total_drop:.3f}")
    print(f"  -> the (sharpness, trigger_rate) pair is the benign operating point.")
    print(bar)
    return 0 if all(r.passed for r in results) else 1


def run(config: RunConfig, task: CodeTask = SQUARE_TASK, seed: int = 0) -> int:
    try:
        import torch  # noqa: F401
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
    except Exception as e:  # pragma: no cover - only without the [train] extra / GPU
        print(f"[flightrecorder] real run needs extras: pip install 'flightrecorder[train]' "
              f"and a GPU. Import failed: {e}")
        return 1

    collector: list = [] if config.integrity_report else None  # type: ignore[assignment]
    comp = make_components(config, task, seed, collector=collector)
    # Instruct models need the chat template: pass a conversational prompt so TRL applies it.
    if "Instruct" in config.model or "instruct" in config.model:
        prompt = [{"role": "user", "content": task.prompt}]
    else:
        prompt = task.prompt
    dataset = Dataset.from_dict({"prompt": [prompt] * config.prompts_per_epoch})
    args = GRPOConfig(
        output_dir=os.path.join(config.out_dir, "trainer"),
        per_device_train_batch_size=config.num_generations,
        num_generations=config.num_generations,
        max_steps=config.steps, learning_rate=config.learning_rate,
        max_completion_length=256,                 # bound generation cost
        beta=config.beta,                          # >0 -> TRL logs `kl` for capture (CHECK 1)
        logging_steps=1, save_strategy="no", report_to=[], seed=seed)
    trainer = GRPOTrainer(model=config.model, reward_funcs=[comp["reward_fn"]],
                          args=args, train_dataset=dataset, callbacks=[comp["callback"]])
    trainer.train()
    comp["recorder"].close()
    print(f"[flightrecorder] wrote {comp['artifact']}")

    rc = 0
    if config.integrity_report:
        rc = emit_integrity_report(collector, config, comp["artifact"])
    return rc


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="flr-grpo-qwen")
    p.add_argument("--scale", choices=["shakeout", "pilot", "benign", "full"], default="shakeout")
    p.add_argument("--task", choices=["const", "complex", "maxrun", "secondmax"], default=None,
                   help="benign scale task: const (sum) | complex (lus2) | maxrun | secondmax")
    p.add_argument("--artifact-dir", default="runs", help="where run artifacts are written")
    p.add_argument("--out", default=None, help="alias for --artifact-dir")
    p.add_argument("--model", default=None)
    p.add_argument("--steps", type=int, default=None, help="override the scale's step budget")
    p.add_argument("--lr", type=float, default=None,
                   help="override learning rate — the benign-pair SHARPNESS lever (high=sharp)")
    p.add_argument("--reward-mode", choices=["gameable", "robust"], default=None,
                   help="controlled pair: gameable (weak verifier) | robust (strong reference)")
    p.add_argument("--curriculum-switch-step", type=int, default=None,
                   help="pilot: ENGINEERED warm-start length (optimise strong verifier first)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--thresholds", default="configs/thresholds.yaml",
                   help="pre-registered detector thresholds (GATE 1)")
    p.add_argument("--integrity-report", action="store_true",
                   help="collect per-step records and print the four integrity checks at run end")
    p.add_argument("--dry-run", action="store_true",
                   help="build config + components and print the plan; do not import trl/torch")
    a = p.parse_args(argv)
    out_dir = a.out or a.artifact_dir
    config = build_config(a.scale, out_dir=out_dir, model=a.model,
                          thresholds_path=a.thresholds, integrity_report=a.integrity_report)
    if a.steps is not None:
        config.steps = a.steps
    if a.lr is not None:
        config.learning_rate = a.lr
    if a.reward_mode is not None:
        config.reward_mode = a.reward_mode
    if a.curriculum_switch_step is not None:
        config.curriculum_switch = a.curriculum_switch_step
    if a.task is not None:
        config.benign_task = a.task
    task = task_for(config)
    if a.dry_run:
        comp = make_components(config, task=task, seed=a.seed)
        comp["recorder"].close()
        print(f"[dry-run] scale={config.scale} model={config.model} steps={config.steps} "
              f"groups={config.num_generations} reward={config.reward_mode} task={task.name} "
              f"thresholds={config.thresholds_path} integrity={config.integrity_report} "
              f"hard_negatives={config.hard_negatives} -> {comp['artifact']}")
        return 0
    return run(config, task=task, seed=a.seed)


if __name__ == "__main__":
    raise SystemExit(main())
