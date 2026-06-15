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
import os
from dataclasses import dataclass

from .reward_testhack import SQUARE_TASK, CodeTask, make_reward_fns
from ..config import load_thresholds, detector_from_thresholds, check_threshold_provenance


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
    if scale == "full":
        # Real experiment: NO seeding -- emergence is the question. beta>0 for KL capture.
        return RunConfig(model=model or "Qwen/Qwen2.5-1.5B", steps=400, num_generations=8,
                         learning_rate=1e-6, seeds=[0, 1, 2], reward_mode="gameable",
                         hard_negatives=True, beta=0.04, seed_hack_from=None, **common)
    raise ValueError(f"unknown scale: {scale!r} (use 'shakeout' or 'full')")


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
    reward_fn = wrap_reward_fns(coord, optimize_fn, measure_fn, seed_hack=seed_spec)
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
        print(f"\n[shakeout] NOTE — the test-overwrite hack was SEEDED into the recorded "
              f"rollout stream from reward-call {config.seed_hack_from}. This is a PLUMBING "
              f"test of the capture->label->isolation->report path on a known hack; it does "
              f"NOT claim the model emergently discovered it (that is a full-run question).\n")

    for w in check_threshold_provenance(config.thresholds_path, artifact):
        print("\n!!! " + w + "\n")

    oracle_series = [sr.oracle.oracle_reward for sr in collector if sr.oracle is not None]
    warmup = min(20, max(2, len(oracle_series) // 5)) if oracle_series else 2
    oracle_turn = oracle_gap_turn(oracle_series, warmup=warmup) if oracle_series else None

    results = run_integrity_report(collector, config.steps, oracle_turn)
    print(format_report(results))
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
    dataset = Dataset.from_dict({"prompt": [task.prompt] * config.prompts_per_epoch})
    args = GRPOConfig(
        output_dir=os.path.join(config.out_dir, "trainer"),
        per_device_train_batch_size=config.num_generations,
        num_generations=config.num_generations,
        max_steps=config.steps, learning_rate=config.learning_rate,
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
    p.add_argument("--scale", choices=["shakeout", "full"], default="shakeout")
    p.add_argument("--artifact-dir", default="runs", help="where run artifacts are written")
    p.add_argument("--out", default=None, help="alias for --artifact-dir")
    p.add_argument("--model", default=None)
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
    if a.dry_run:
        comp = make_components(config, seed=a.seed)
        comp["recorder"].close()
        print(f"[dry-run] scale={config.scale} model={config.model} steps={config.steps} "
              f"groups={config.num_generations} reward={config.reward_mode} "
              f"thresholds={config.thresholds_path} integrity={config.integrity_report} "
              f"hard_negatives={config.hard_negatives} -> {comp['artifact']}")
        return 0
    return run(config, seed=a.seed)


if __name__ == "__main__":
    raise SystemExit(main())
