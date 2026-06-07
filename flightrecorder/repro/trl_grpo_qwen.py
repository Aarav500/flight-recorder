"""Launch-ready GRPO + small-Qwen run on the test-overwrite task, wired through the TRL
adapter and the sandboxed reward pair (spec §8).

Scale is a LAUNCH FLAG, not a code change:
  - shakeout (~$15): 1 small model, tiny steps, 1 seed, gameable reward only -- proves the
    whole pipeline records a real trainer end to end before any real spend.
  - full ($30-90): controlled pair (gameable vs robust reference) + seeds + hard negatives.

Heavy deps (torch/trl/transformers/datasets) are imported lazily inside run(), so this
module imports and unit-tests with none of them installed. The actual run needs
`pip install 'flightrecorder[train]'` and a GPU.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

from .reward_testhack import SQUARE_TASK, CodeTask, make_reward_fns


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
    prompts_per_epoch: int = 64


def build_config(scale: str, out_dir: str = "runs", model: str | None = None) -> RunConfig:
    if scale == "shakeout":
        return RunConfig(model=model or "Qwen/Qwen2.5-0.5B", steps=20, num_generations=4,
                         learning_rate=1e-6, seeds=[0], reward_mode="gameable",
                         hard_negatives=False, out_dir=out_dir, scale=scale)
    if scale == "full":
        return RunConfig(model=model or "Qwen/Qwen2.5-1.5B", steps=400, num_generations=8,
                         learning_rate=1e-6, seeds=[0, 1, 2], reward_mode="gameable",
                         hard_negatives=True, out_dir=out_dir, scale=scale)
    raise ValueError(f"unknown scale: {scale!r} (use 'shakeout' or 'full')")


def make_components(config: RunConfig, task: CodeTask = SQUARE_TASK, seed: int = 0) -> dict:
    """Build the recorder + TRL coordinator + callback + the wrapped reward fn. Uses only
    flightrecorder + stdlib -- no torch -- so it is unit-testable on a laptop."""
    from ..recorder import Recorder
    from ..sinks.jsonl import JSONLSink
    from ..sinks.console import ConsoleSink
    from ..detector.contraction_tube import ContractionTubeDetector
    from ..adapters.trl import TRLCoordinator, wrap_reward_fns, FlightRecorderCallback

    os.makedirs(config.out_dir, exist_ok=True)
    artifact = os.path.join(config.out_dir, f"{config.scale}_{config.reward_mode}_seed{seed}.jsonl")
    warmup = min(30, max(5, config.steps // 5))
    recorder = Recorder(detector=ContractionTubeDetector(warmup=warmup),
                        sinks=[JSONLSink(artifact), ConsoleSink()])
    coord = TRLCoordinator()
    train_fn, oracle_fn = make_reward_fns(task)
    # gameable: optimize the visible (overwritable) tests; measure with held-out oracle.
    # robust: optimize the held-out verifier (model can't see/overwrite it) -- the
    #         non-hacking reference whose frames the tube is fit on.
    if config.reward_mode == "gameable":
        optimize_fn, measure_fn = train_fn, oracle_fn
    else:
        optimize_fn, measure_fn = oracle_fn, train_fn
    reward_fn = wrap_reward_fns(coord, optimize_fn, measure_fn)
    callback = FlightRecorderCallback(recorder, coord, group_size=config.num_generations)
    return {"recorder": recorder, "coordinator": coord, "callback": callback,
            "reward_fn": reward_fn, "artifact": artifact, "task": task}


def run(config: RunConfig, task: CodeTask = SQUARE_TASK, seed: int = 0) -> int:
    try:
        import torch  # noqa: F401
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
    except Exception as e:  # pragma: no cover - only without the [train] extra / GPU
        print(f"[flightrecorder] real run needs extras: pip install 'flightrecorder[train]' "
              f"and a GPU. Import failed: {e}")
        return 1

    comp = make_components(config, task, seed)
    dataset = Dataset.from_dict({"prompt": [task.prompt] * config.prompts_per_epoch})
    args = GRPOConfig(
        output_dir=os.path.join(config.out_dir, "trainer"),
        per_device_train_batch_size=config.num_generations,
        num_generations=config.num_generations,
        max_steps=config.steps, learning_rate=config.learning_rate,
        logging_steps=1, save_strategy="no", report_to=[], seed=seed)
    trainer = GRPOTrainer(model=config.model, reward_funcs=[comp["reward_fn"]],
                          args=args, train_dataset=dataset, callbacks=[comp["callback"]])
    trainer.train()
    comp["recorder"].close()
    print(f"[flightrecorder] wrote {comp['artifact']}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="flr-grpo-qwen")
    p.add_argument("--scale", choices=["shakeout", "full"], default="shakeout")
    p.add_argument("--out", default="runs")
    p.add_argument("--model", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true",
                   help="build config + components and print the plan; do not import trl/torch")
    a = p.parse_args(argv)
    config = build_config(a.scale, out_dir=a.out, model=a.model)
    if a.dry_run:
        comp = make_components(config, seed=a.seed)
        comp["recorder"].close()
        print(f"[dry-run] scale={config.scale} model={config.model} steps={config.steps} "
              f"groups={config.num_generations} reward={config.reward_mode} "
              f"hard_negatives={config.hard_negatives} -> {comp['artifact']}")
        return 0
    return run(config, seed=a.seed)


if __name__ == "__main__":
    raise SystemExit(main())
