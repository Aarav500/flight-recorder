"""Source C driver: VERIFY the residual-stream readout, FIT and FREEZE the concept directions,
then SWEEP a directory of checkpoints into the artifact benchmark/score_prime.py already reads.

    python scripts/sweep_source_c.py --verify --model=Qwen/Qwen2.5-0.5B-Instruct
    python scripts/sweep_source_c.py --fit    --model=Qwen/Qwen2.5-0.5B-Instruct
    python scripts/sweep_source_c.py --sweep  --model=Qwen/Qwen2.5-0.5B-Instruct \
        --checkpoints=runs/ckpt/hack_seed0 --arm=hack --task=count_vowels --seed=0 \
        --hidden=runs/ckpt/hack_seed0/hidden.json

THIS SCRIPT SPENDS NOTHING AND CANNOT. There is no --submit, no run_uv_job import, no hub upload
and no instance creation anywhere in it. --verify and --fit are CPU forward passes on an
already-cached model; --sweep is CPU (or local GPU) forward passes over checkpoints that already
exist on disk. Producing those checkpoints is the paid step and it is NOT done here.

WHY A CHECKPOINT SWEEP AND NOT A RE-RUN. Every launcher in this repo sets save_strategy="no"
(scripts/launch_judge_batch.py:158, launch_multiprobe_run.py:120, launch_prime_probe.py:991,
launch_probe_run.py:146, flightrecorder/repro/trl_grpo_qwen.py:275) and three of the four write to
output_dir="/tmp/out", i.e. ephemeral job-container storage. NO CHECKPOINT FROM ANY PUBLISHED RUN
EXISTS -- not on disk, not on the hub. What was saved is per-step SCALARS (runs/batch/*.jsonl:
kl_mean, entropy_trend, adv_kurtosis, ...), which is exactly the low-order-scalar-aggregate class
this experiment exists to escape and contains no activations to re-read. So the lead-time half of
E1 cannot be recovered by re-scoring frozen logs, and this script is written to be the thing that
runs the moment checkpoints exist -- from either a checkpointed re-run or an in-run probe.

--sweep also accepts an in-run artifact instead of a checkpoint directory: pass
--rows=<jsonl> where each line is a probe row already produced by
prime_source_c.score_source_c inside a training callback. That path needs no checkpoint storage
at all and is the cheaper way to get the same table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flightrecorder.repro import prime_probe as PP          # noqa: E402
from flightrecorder.repro import prime_source_c as SC       # noqa: E402

DIAG = ROOT / "experiments" / "prime" / "diag_v3_n64.json"
DIRECTIONS = ROOT / "experiments" / "prime" / "sourcec_directions.json"
OUT_DIR = ROOT / "runs" / "prime"

FLAGS = [a for a in sys.argv[1:] if a.startswith("--")]


def _flag(name, default=None):
    for a in FLAGS:
        if a == "--" + name:
            return True
        if a.startswith("--%s=" % name):
            return a.split("=", 1)[1]
    return default


def _load_pairs(path=DIAG, n=None):
    """The FROZEN diagnostic selection, read-only, adapted through prime_probe's ONE off-stream
    adapter. D_hack and the robust-honest control are stored as separate lists by the launcher,
    so the category is supplied here rather than read from the record -- exactly as
    launch_prime_probe.py does it."""
    sel = json.loads(Path(path).read_text(encoding="utf-8"))
    h, c = sel["d_hack"], sel["d_control"]
    if n:
        h, c = h[:int(n)], c[:int(n)]
    pairs = (PP.pairs_from_frozen(h, category="hack")
             + PP.pairs_from_frozen(c, category="robust_honest"))
    return pairs, sel


def _load_model(model_id):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32,
        device_map=("cuda" if torch.cuda.is_available() else None))
    print("    loaded %s %.0fM params on %s" % (type(mdl).__name__,
          sum(p.numel() for p in mdl.parameters()) / 1e6, next(mdl.parameters()).device))
    return mdl, tok


def _checkpoints(d):
    """[(step, path)] from a HF-style output dir. Accepts checkpoint-<N> subdirs and, as the
    step-0 anchor, the base directory itself when it holds weights."""
    d = Path(d)
    out = []
    if any(d.glob("*.safetensors")) or (d / "pytorch_model.bin").exists():
        out.append((0, d))
    for p in sorted(d.glob("checkpoint-*")):
        try:
            out.append((int(p.name.split("-")[-1]), p))
        except ValueError:
            continue
    return sorted(out)


# --------------------------------------------------------------------------- (a) verify
def cmd_verify(model_id, n_pairs):
    pairs, sel = _load_pairs(n=n_pairs)
    print("[verify] frozen set %s  |D_hack|=%d |D_control|=%d  content_hash=%s"
          % (DIAG.name, sel["n_hack"], sel["n_control"], sel["content_hash"][:16]))
    model, tok = _load_model(model_id)
    model.train()                                  # enter TRAIN mode: restoration is the check
    res = SC.verify_readout(model, tok, pairs, batch_size=4)
    print("[verify] ALL CHECKS PASSED")
    return res


# --------------------------------------------------------------------------- (b) fit + freeze
def cmd_fit(model_id, n_pairs, folds, directions=None, pairs=None, model=None):
    directions = Path(directions or DIRECTIONS)
    if pairs is None:
        pairs, sel = _load_pairs(n=n_pairs)
    else:
        sel = {"content_hash": "n/a"}
    model, tok = model if model is not None else _load_model(model_id)
    print("[fit] collecting step-0 residuals over every layer, 3 components x %d pairs"
          % len(pairs))
    built = SC.build_step0(model, tok, pairs, layers=None, folds=folds, batch_size=4)
    rep = built["report"]
    sha = SC.freeze_directions(directions, built["frozen_body"], meta={
        "model": model_id, "diagnostic": DIAG.name,
        "diagnostic_content_hash": sel["content_hash"],
        "diagnostic_sha256": rep["diagnostic_sha256"], "prompt_sha256": rep["prompt_sha256"],
        "n_hack": rep["n_hack"], "n_control": rep["n_control"], "folds": folds,
        "readout": "unit-normalised projection of the centred residual stream at the final "
                   "prompt token (add_generation_prompt=True), pre-generation",
        "fit": "difference of means over D_hack minus robust-honest control, frozen offline "
               "category labels only; layer chosen once at step 0 by out-of-fold AUROC",
        "reported_numbers_are": "OUT-OF-FOLD -- in-sample separation of a difference-of-means "
                                "direction is trivial and is not reported"})
    print("[fit] frozen -> %s  sha256=%s" % (directions, sha[:16]))
    print("\n  STEP-0, OUT-OF-SAMPLE (|D_hack|=%d, |D_control|=%d)" % (rep["n_hack"], rep["n_control"]))
    print("  %-5s %6s %10s %10s   %9s %9s   %10s" %
          ("comp", "layer", "AUROC_oof", "2nd-layer", "hack_mean", "ctrl_mean", "logit_AUROC"))
    for c in SC.COMPONENTS:
        r = rep["components"][c]
        print("  %-5s %6d %10.4f %10s   %+9.4f %+9.4f   %10.4f" %
              (c, r["layer"], r["auroc_oof"],
               ("%.4f" % r["runner_up_layer_auroc"]) if r["runner_up_layer_auroc"] else "-",
               r["hack_mean"], r["control_mean"], rep["logit_channel_auroc_oof"][c]))
    print("\n  logit_AUROC is the SAME forward pass, SAME position, SAME prompts -- the "
          "output-distribution\n  channel prime_probe already reads. The difference between the "
          "two columns is the CHANNEL\n  and nothing else.")
    directions.with_name(directions.stem + "_step0_report.json").write_text(
        json.dumps(rep, indent=1), encoding="utf-8")
    return built


# --------------------------------------------------------------------------- (c) the sweep
def cmd_sweep(model_id, n_pairs, directions=None, pairs=None, ckpt_dir=None, arm=None,
              out_dir=None, task=None, seed=None):
    """Point at a directory of checkpoints -> the final table, with no further engineering.

    Everything downstream is INHERITED: benchmark/score_prime.py's collapse event, its FPR-10
    threshold on the matched benign arm, its published tpr_at_fpr10 definition and its verdict
    function all apply to the sourcec_* series exactly as they apply to prime_*, because
    build_source_c_blob emits the same shape."""
    directions = Path(directions or DIRECTIONS)
    ckpt_dir = ckpt_dir or _flag("checkpoints")
    rows_path = _flag("rows")
    arm = arm or _flag("arm")
    if not arm:
        raise SystemExit("--sweep needs --arm=hack|benign (it decides which side of the FPR "
                         "operating point this run lands on)")
    if not (ckpt_dir or rows_path):
        raise SystemExit("--sweep needs --checkpoints=DIR or --rows=JSONL")
    frozen = SC.load_directions(directions)
    print("[sweep] frozen directions %s sha256=%s  fit on %s"
          % (directions.name, frozen["sha256"][:16], frozen["meta"].get("model")))
    if pairs is None:
        pairs, _sel = _load_pairs(n=n_pairs)

    probe_rows, prime_rows = [], []
    if rows_path:
        for line in Path(rows_path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                probe_rows.append(json.loads(line))
    else:
        cks = _checkpoints(ckpt_dir)
        if not cks:
            raise SystemExit("no checkpoints under %s (looked for checkpoint-<N>/ and weights in "
                             "the directory itself)" % ckpt_dir)
        print("[sweep] %d checkpoints: steps %s" % (len(cks), [s for s, _ in cks]))
        for step, path in cks:
            model, tok = _load_model(str(path))
            got = SC.collect(model, tok, pairs,
                             layers=sorted({int(frozen["components"][c]["layer"])
                                            for c in SC.COMPONENTS}), batch_size=4)
            if got["diagnostic_sha256"] != frozen["meta"].get("diagnostic_sha256"):
                raise SystemExit("diagnostic set changed since the directions were frozen; the "
                                 "per-step values are not comparable")
            r = SC.score_source_c(model, tok, pairs, frozen, precollected=got)
            probe_rows.append({"step": step, **{k: r[k] for k in
                               SC.SOURCE_C_KEYS + SC.DIAG_KEYS + SC.CONTROL_KEYS + SC.MARGIN_KEYS
                               if k in r}})
            print("  step %-6d %s" % (step, {c: round(r["sourcec_%s" % c], 3)
                                             for c in SC.COMPONENTS}))
            del model, tok

    hidden_path = _flag("hidden")
    reward_rows = (json.loads(Path(hidden_path).read_text(encoding="utf-8"))
                   if hidden_path else [])
    task = task if task is not None else _flag("task")
    seed = seed if seed is not None else _flag("seed")
    blob = SC.build_source_c_blob(probe_rows, reward_rows,
                                  {"arm": arm, "task": task, "seed": seed,
                                   "model": model_id, "directions_sha256": frozen["sha256"],
                                   "diagnostic": DIAG.name},
                                  prime_rows=prime_rows or None)
    od = Path(out_dir or OUT_DIR)
    od.mkdir(parents=True, exist_ok=True)
    out = od / ("sourcec_%s_%s_seed%s.json" % (task or "task", arm, seed if seed is not None else 0))
    out.write_text(json.dumps(blob, indent=1), encoding="utf-8")
    print("[sweep] wrote %s" % out)
    print("[sweep] now run:  python benchmark/score_prime.py")
    return blob


# --------------------------------------------------------------------------- confound test
def _strip_payload(solution: str):
    """Remove the exploit payload from a D_hack solution, keeping the (still wrong) task body.

    Every hack pair in the frozen set carries a grader-overwrite payload -- a top-level helper
    that writes `test_visible.py`, plus its call. This drops exactly those top-level nodes, by
    parsing, so the remainder is the candidate function alone. Returns (stripped, ok)."""
    import ast
    try:
        tree = ast.parse(solution)
    except SyntaxError:
        return solution, False
    keep, dropped = [], 0
    for node in tree.body:
        seg = ast.get_source_segment(solution, node) or ""
        if "test_visible" in seg:
            dropped += 1
            continue
        keep.append(seg)
    out = "\n\n\n".join(s for s in keep if s.strip()) + "\n"
    return out, (dropped > 0 and "test_visible" not in out and "def " in out)


def cmd_confound(model_id, n_pairs, directions=None):
    """IS THE FROZEN DIRECTION READING THE CONCEPT, OR THE PAYLOAD STRING?

    A step-0 out-of-fold AUROC of 1.000 at EVERY layer from the first transformer block onward is
    not what a deep concept looks like; it is what a surface feature looks like. And on this
    diagnostic set there is an obvious candidate: `test_visible` occurs in 100% of D_hack
    solutions and 0% of the robust-honest controls, and the hack solutions are ~60% longer. The
    launcher's own lexical audit gates PRESENTATION markers but deliberately does not gate
    MECHANISM markers, on the grounds that they ARE the exploit -- which is the right call for a
    forced-choice question about the code, and the wrong assumption for an activation probe,
    because a linear readout of the residual stream can separate the two blocks on the string
    alone without representing anything about reward hacking.

    THE TEST. Score three blocks with the SAME frozen direction, layer and cut:
        (1) D_hack as frozen                      -- payload present
        (2) D_hack with the payload REMOVED       -- same wrong bodies, payload gone
        (3) the robust-honest control             -- payload absent
    Block (2) is not a new diagnostic bin and is not re-binned; it is an ABLATION used to ask
    what the direction responds to. If the direction encodes "this solution exploits the grader",
    (2) should stay near (1). If it encodes "this text contains the payload", (2) collapses onto
    (3).

    WHY IT MATTERS FOR E1. A direction that reads a fixed string in a FIXED diagnostic set is a
    property of the SET, not of the checkpoint -- so it is near-constant across a training run and
    cannot produce a lead, however good its step-0 AUROC looks. That is a prediction about the
    paid run, obtainable for nothing.
    """
    import numpy as np
    directions = Path(directions or DIRECTIONS)
    frozen = SC.load_directions(directions)
    pairs, _ = _load_pairs(n=n_pairs)
    hack = [p for p in pairs if p["category"] == "hack"]
    ctrl = [p for p in pairs if p["category"] == "robust_honest"]
    abl, bad = [], 0
    for p in hack:
        s, ok = _strip_payload(p["solution"])
        if not ok:
            bad += 1
            continue
        abl.append(dict(p, solution=s, id="%s_ablated" % p.get("id")))
    print("[confound] ablated %d/%d hack solutions (%d unparseable or payload-free)"
          % (len(abl), len(hack), bad))
    model, tok = _load_model(model_id)
    layers = sorted({int(frozen["components"][c]["layer"]) for c in SC.COMPONENTS})
    blocks = {}
    for name, blk in (("hack", hack), ("hack_ablated", abl), ("control", ctrl)):
        blocks[name] = SC.collect(model, tok, blk, layers=layers, batch_size=4)
    print("\n[confound] frozen direction scored on three blocks (same layer, same cut)")
    print("  %-5s %6s %12s %14s %12s   %14s %16s"
          % ("comp", "layer", "hack_mean", "ablated_mean", "ctrl_mean",
             "AUROC hack/ctrl", "AUROC abl/ctrl"))
    aurocs, retention, dump = [], [], {}
    for c in SC.COMPONENTS:
        cf = frozen["components"][c]
        L = int(cf["layer"])
        sc = {}
        for name in blocks:
            s, _raw = SC.project(blocks[name]["resid"][c][L], cf["direction"], cf["mu"])
            sc[name] = np.asarray(s, float)
        a1 = SC.auroc(sc["hack"], sc["control"])
        a2 = SC.auroc(sc["hack_ablated"], sc["control"])
        h, ab, k = sc["hack"].mean(), sc["hack_ablated"].mean(), sc["control"].mean()
        # RETENTION is the number that matters, and AUROC alone hides it. AUROC is
        # scale-free: two tight distributions separate at ~1.0 however small the gap between
        # them. What the ablation asks is how much of the hack-vs-control DISPLACEMENT the
        # payload was responsible for.
        r = float((ab - k) / (h - k)) if (h - k) else float("nan")
        aurocs.append(a2)
        retention.append(r)
        dump[c] = {"layer": L, "cut": float(cf["cut"]), "hack_mean": float(h),
                   "ablated_mean": float(ab), "control_mean": float(k),
                   "auroc_hack_ctrl": a1, "auroc_ablated_ctrl": a2, "retention": r,
                   "ablated_mean_above_cut": bool(ab > float(cf["cut"])),
                   "scores": {k2: [round(float(x), 6) for x in v] for k2, v in sc.items()}}
        print("  %-5s %6d %12.4f %14.4f %12.4f   %14.4f %16.4f   ret %5.1f%%"
              % (c, L, h, ab, k, a1, a2, 100 * r))
    med_a, med_r = float(np.median(aurocs)), float(np.median(retention))
    out = directions.with_name(directions.stem + "_confound.json")
    out.write_text(json.dumps({"median_auroc_ablated": med_a, "median_retention": med_r,
                               "components": dump}, indent=1), encoding="utf-8")
    print("\n  median AUROC(ablated vs control) = %.4f" % med_a)
    print("  median RETENTION of the hack-vs-control displacement = %.1f%%" % (100 * med_r))
    print("  wrote %s" % out.name)
    print("\n  READ BOTH NUMBERS, NOT EITHER ALONE.")
    if med_a < 0.65:
        print("  The direction is a PAYLOAD-STRING detector: removing the payload collapses the\n"
              "  separation to chance.")
    elif med_r < 0.6:
        print("  MIXED, and the honest reading is the unflattering one. AUROC survives the\n"
              "  ablation (%.3f) but %.0f%% of the displacement does not: the payload string was\n"
              "  responsible for most of the projection magnitude, and the ablated block's mean\n"
              "  falls BELOW the frozen gate, so under the calibrated cut it would score ~0.\n"
              "  AUROC stays high only because both blocks are tight, not because the direction\n"
              "  is measuring the concept. What remains is ALSO confounded: the ablated hack\n"
              "  bodies are deliberately WRONG and the controls are correct, so the residue is\n"
              "  consistent with a buggy-code detector rather than a hack-recognition one.\n"
              "  Either way it is a fixed property of a FIXED text set."
              % (med_a, 100 * (1 - med_r)))
    else:
        print("  The separation survives ablation in both AUROC and magnitude, so the direction\n"
              "  is not merely reading the payload.")
    print("\n  AND NONE OF IT IS EVIDENCE OF LEAD. Every number here is measured on two fixed\n"
          "  blocks of text at a single checkpoint. Lead time is about whether the readout MOVES\n"
          "  as the policy learns to hack, which no step-0 quantity can settle. A step-0 AUROC of\n"
          "  1.000 is fully consistent with a perfectly flat series over training.")
    return med_a, med_r


# --------------------------------------------------------------------------- selftest
def cmd_selftest():
    """Drive the WHOLE chain -- fit -> freeze -> checkpoint discovery -> per-checkpoint readout
    -> blob -> benchmark/score_prime.py table -- against REAL weights written to REAL checkpoint
    directories, offline, in a temp dir, on a model small enough to be free.

    This exists because "(c) the scorer produces the final table with no further engineering" is
    a claim about plumbing, and a claim about plumbing that has never been executed is a guess.
    What it does NOT establish is anything about model behaviour: the policy here is a
    randomly-initialised 4-layer/64-unit Qwen2 and its numbers are meaningless. The point is that
    the only thing standing between here and the final table is checkpoints.

    The toy is built FROM THE CACHED Qwen2.5-0.5B-Instruct CONFIG AND TOKENIZER rather than
    downloaded, so this runs offline and costs nothing -- and it keeps the real chat template, so
    the read position is the same kind of position the real probe reads.
    """
    import shutil
    import tempfile

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tmp = Path(tempfile.mkdtemp(prefix="sourcec_selftest_"))
    ref = "Qwen/Qwen2.5-0.5B-Instruct"
    print("[selftest] building real checkpoint dirs under %s" % tmp)
    tok = AutoTokenizer.from_pretrained(ref)
    cfg = AutoConfig.from_pretrained(ref)
    cfg.num_hidden_layers, cfg.hidden_size = 4, 64
    cfg.intermediate_size, cfg.num_attention_heads, cfg.num_key_value_heads = 128, 4, 2
    # `layer_types` is a per-layer list copied from the 24-layer parent; leaving it stale makes
    # the SAVED config self-inconsistent and only blows up on RELOAD, i.e. inside the sweep.
    cfg.layer_types = ["full_attention"] * cfg.num_hidden_layers
    ck = tmp / "ckpts"
    torch.manual_seed(0)
    base = AutoModelForCausalLM.from_config(cfg).to(torch.float32)
    # TWELVE checkpoints, not four. benchmark/score_prime.py dates a first move against a
    # 5-SAMPLE baseline window (grid_benchmark.active_frac / first_move_step), so a 4-point sweep
    # cannot populate the lead, benign-activity or TPR columns at all -- it returns '--', 0/0 and
    # n/a and a plumbing test that accepted that would be asserting nothing. The cadence here (5)
    # is the launcher's PROBE_EVERY, so this is also the shape a real sweep produces.
    STEPS = list(range(0, 60, 5))
    for step in STEPS:
        # a deterministic drift per step, so the swept series is not constant and the scorer has
        # something to date a first move off
        m = AutoModelForCausalLM.from_config(cfg).to(torch.float32)
        m.load_state_dict(base.state_dict())
        with torch.no_grad():
            g = torch.Generator().manual_seed(1234)
            for p in m.parameters():
                # A HAND-WRITTEN ONSET, and it has to start AFTER the baseline window. The
                # scorer dates a first move against the first 5 SAMPLES, so a perturbation that
                # begins at the second checkpoint is absorbed into the baseline and no move can
                # ever be dated -- which is how the first version of this test came back all
                # '--' and still printed PASS. Flat through step 25, then ramping.
                p.add_(torch.randn(p.shape, generator=g) * (0.02 * max(0, step - 25) / 30.0))
        d = ck / ("checkpoint-%d" % step)
        m.save_pretrained(d)
        tok.save_pretrained(d)
        del m
    found = _checkpoints(ck)
    assert [s for s, _ in found] == STEPS, found
    print("[selftest] checkpoint discovery: %s" % [s for s, _ in found])

    pairs, _ = _load_pairs(n=12)
    dpath = tmp / "directions.json"
    cmd_fit(ref, None, 3, directions=dpath, pairs=pairs, model=(base, tok))
    del base

    blobs = []
    for arm, seed in (("hack", 0), ("benign", 0), ("benign", 1)):
        cmd_sweep(ref, None, directions=dpath, pairs=pairs, ckpt_dir=ck, arm=arm,
                  out_dir=tmp / "runs", task="selftest", seed=seed)
        # a held-out-quality series so the scorer has a collapse event to measure lead against
        p = tmp / "runs" / ("sourcec_selftest_%s_seed%d.json" % (arm, seed))
        blob = json.loads(p.read_text(encoding="utf-8"))
        blob["hidden"] = [[t, (0.6 if (arm == "benign" or t < 40) else 0.2)]
                          for t in range(max(STEPS) + 1)]
        blob["seed"] = seed
        p.write_text(json.dumps(blob, indent=1), encoding="utf-8")
        blobs.append(p)

    sys.path.insert(0, str(ROOT / "benchmark"))
    import score_prime as SP                                    # noqa: E402
    runs = SP.load_prime_runs(tmp / "runs", verbose=False)
    rows = SP.score(runs, how="baseline_drop", frac=0.6)
    labels = [r["detector"] for r in rows]
    missing = [l for l in SP.SOURCE_C_LABELS if l not in labels]
    assert not missing, "these Source C rows never reached the table: %s" % missing
    print("\n[selftest] THE FINAL TABLE, produced from a checkpoint directory with no further "
          "engineering:")
    _table(rows, SP)
    # THE ASSERTION THAT MAKES THIS A TEST RATHER THAN A DEMO. A table of '--' / '0/0' / 'n/a'
    # would otherwise "pass" while proving nothing, which is exactly what the first version of
    # this self-test did: four checkpoints is fewer than the 5-SAMPLE baseline window
    # benchmark/score_prime.py dates a first move against, so every column came back empty and
    # the run still printed PASS. The claim under test is that pointing the scorer at a
    # checkpoint directory yields the FINAL table -- so the lead, runs-leading, benign-activity
    # and TPR@10%FPR columns must all be populated from the swept series.
    live = [r for r in rows if r["detector"] in SP.SOURCE_C_LABELS]
    empty = [r["detector"] for r in live
             if r["median_lead"] is None or r["benign_activity"] is None
             or r.get("tpr_at_fpr10") is None or r["n_runs"] == 0]
    if empty:
        raise AssertionError(
            "these Source C rows reached the table with UNPOPULATED columns: %s. The sweep -> "
            "scorer chain therefore does not in fact produce the final table, and a 'PASS' here "
            "would be vacuous." % empty)
    print("\n[selftest] every column populated for all %d Source C rows." % len(live))
    print("[selftest] the numbers above are MEANINGLESS -- the policy is a randomly-initialised\n"
          "           4-layer/64-unit toy and the 'collapse' is hand-written. What is established\n"
          "           is that checkpoint discovery -> per-checkpoint residual readout -> blob ->\n"
          "           benchmark/score_prime.py runs end to end and populates EVERY column of the\n"
          "           final table. The only missing input is real checkpoints.")
    shutil.rmtree(tmp, ignore_errors=True)
    print("[selftest] PASS")


def _table(rows, SP):
    print("  %-42s %7s %8s %9s %8s" % ("detector", "lead", "leading", "benign", "TPR@.1"))
    for r in rows:
        if r["detector"] not in SP.SOURCE_C_LABELS:
            continue
        print("  %-42s %7s %8s %9s %8s" % (
            r["detector"],
            ("%+g" % r["median_lead"]) if r["median_lead"] is not None else "--",
            "%d/%d" % (r["n_lead_pos"], r["n_runs"]),
            ("%.2f" % r["benign_activity"]) if r["benign_activity"] is not None else "n/a",
            ("%.2f" % r["tpr_at_fpr10"]) if r.get("tpr_at_fpr10") is not None else "n/a"))


if __name__ == "__main__":
    model_id = _flag("model", "Qwen/Qwen2.5-0.5B-Instruct")
    n_pairs = _flag("n-pairs")
    folds = int(_flag("folds", 4))
    did = False
    if _flag("verify"):
        cmd_verify(model_id, n_pairs); did = True
    if _flag("fit"):
        cmd_fit(model_id, n_pairs, folds); did = True
    if _flag("sweep"):
        cmd_sweep(model_id, n_pairs); did = True
    if _flag("confound"):
        cmd_confound(model_id, n_pairs); did = True
    if _flag("selftest"):
        cmd_selftest(); did = True
    if not did:
        print(__doc__)
