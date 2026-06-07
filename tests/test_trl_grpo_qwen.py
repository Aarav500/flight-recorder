import json
from flightrecorder.repro.trl_grpo_qwen import build_config, make_components, main
from flightrecorder.repro.reward_testhack import CORRECT, HACK


def test_presets():
    sk = build_config("shakeout")
    full = build_config("full")
    assert sk.steps == 20 and sk.seeds == [0] and sk.hard_negatives is False
    assert full.hard_negatives is True and len(full.seeds) >= 2


def test_make_components_builds_without_torch(tmp_path):
    cfg = build_config("shakeout", out_dir=str(tmp_path))
    comp = make_components(cfg, seed=0)
    # the wrapped reward fn returns the TRAIN signal and stashes into the coordinator
    out = comp["reward_fn"](prompts=["p", "p"], completions=[CORRECT, HACK])
    assert out == [1.0, 1.0]                       # both pass visible (HACK overwrites)
    _, train, oracle = comp["coordinator"].drain()
    assert list(oracle) == [1.0, 0.0]              # held-out oracle catches the hack
    comp["recorder"].close()


def test_dry_run_returns_zero_and_writes_artifact_path(tmp_path, capsys):
    rc = main(["--dry-run", "--scale", "shakeout", "--out", str(tmp_path)])
    assert rc == 0
    assert "dry-run" in capsys.readouterr().out


def test_callback_records_through_to_artifact(tmp_path):
    from types import SimpleNamespace
    cfg = build_config("shakeout", out_dir=str(tmp_path))
    comp = make_components(cfg, seed=1)
    comp["reward_fn"](prompts=["p"] * 4, completions=[CORRECT, CORRECT, HACK, HACK])
    state = SimpleNamespace(global_step=3, log_history=[{"kl": 0.1}])
    comp["callback"].on_step_end(state=state, control="C")
    comp["recorder"].close()
    kinds = {json.loads(l)["kind"] for l in open(comp["artifact"]).read().splitlines()}
    assert {"frame", "oracle", "detector"} <= kinds
