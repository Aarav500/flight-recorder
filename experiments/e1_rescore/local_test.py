"""Local CPU test of the E1 rescoring job. No AWS call, no GPU, no 7B download.

Imports the job baked with bake_rescore_job.py local_test (target judge = Qwen2.5-0.5B-Instruct, the only model
this test loads) and checks, in order:
  1. chunk predicates cover steps 0-399 exactly once;
  2. score_rows with a FAKE scorer that OOMs above 5 sequences: every row scored once, each score lands on its
     own row, the token budget halves;
  3. job.main(...) end to end on CPU in float32: validation gate on steps 0-4 of the two validation runs (80
     completions, per-step means against the logged float16 judge_small), then the "target" pass over the same
     80 completions through the batched path; S3 uploads go to a local folder instead.
  4. optionally (--abort-test) the gate's abort path with the tolerance forced negative.

Run under the job's pinned packages (CPU torch):
    uv run --no-project --python 3.12 --with torch==2.11.0 --with transformers==5.19.0 --with numpy==2.5.3 \
        --with accelerate==1.15.0 --with tokenizers==0.23.2 --with huggingface-hub==1.33.0 \
        --with safetensors==0.8.0 python experiments/e1_rescore/local_test.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
SCRATCH = Path(__import__("os").environ.get("E1_RESCORE_SCRATCH", str(Path(__file__).resolve().parents[2] / "experiments" / "e1_rescore" / "_work")))
JOBDIR = SCRATCH / "job"


def load_job(path):
    spec = importlib.util.spec_from_file_location("rescore_job", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_chunks(job):
    hits = [sum(bool(pred(s)) for _n, _r, pred in job.CHUNKS) for s in range(400)]
    assert all(h == 1 for h in hits), [s for s, h in enumerate(hits) if h != 1]
    sizes = {n: sum(bool(pred(s)) for s in range(400)) for n, _r, pred in job.CHUNKS}
    assert sizes == {"edges": 50, "every5": 70, "rest": 280}, sizes
    print("chunks OK: steps per chunk %s -> completions %s" % (sizes, {k: v * 48 * 8 for k, v in sizes.items()}))


def test_fake_oom(job, rows):
    import torch

    class Fake:
        max_length = 1024
        calls = []

        def score(self, problem, texts):
            self.calls.append(len(texts))
            if len(texts) > 5:
                raise torch.OutOfMemoryError("CUDA out of memory (simulated)")
            return [(len(t) % 997) / 997.0 for t in texts]

    sub = [r for r in rows if r["step"] < 3][:300]          # mixed runs and tasks
    lens = [100 + (len(r["text"]) % 50) for r in sub]
    st = {"budget": 4096, "oom_halvings": 0, "batches": 0, "tokens": 0, "padded_tokens": 0}
    got = {}

    def sink(r, s):
        k = (r["run"], r["step"], r["i"])
        assert k not in got, "row scored twice"
        got[k] = s

    job.score_rows(Fake(), sub, lens, st, sink)
    assert len(got) == len(sub)
    bad = [r for r in sub if got[(r["run"], r["step"], r["i"])] != (len(r["text"]) % 997) / 997.0]
    assert not bad, "scores landed on the wrong rows"
    assert st["oom_halvings"] >= 1 and max(c for c in Fake.calls if c <= 5) <= 5
    print("fake-OOM OK: %d rows, %d batches, %d halvings, budget 4096 -> %d" % (
        len(sub), st["batches"], st["oom_halvings"], st["budget"]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default=str(JOBDIR / "rescore_local_test.py"))
    ap.add_argument("--data", default=str(SCRATCH / "e1_completions.jsonl.gz"))
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--abort-test", action="store_true")
    a = ap.parse_args(argv)

    job = load_job(a.job)
    assert job.JUDGE == "Qwen/Qwen2.5-0.5B-Instruct", "the local test must only ever load the 0.5B model"
    test_chunks(job)
    rows = job.load_rows(a.data)
    test_fake_oom(job, rows)

    fake_s3 = JOBDIR / "fake_s3"
    shutil.rmtree(fake_s3, ignore_errors=True)
    fake_s3.mkdir(parents=True)
    uploads = []

    def upload(local, name):
        shutil.copyfile(local, fake_s3 / name)
        uploads.append((round(time.time() - job.T0), name))
        return True

    vruns = set(job.VAL_LOGGED)
    steps = tuple(range(a.steps))
    t = time.time()
    rc = job.main(device="cpu", val_dtype="float32", target_dtype="float32", val_steps=steps,
                  data_path=a.data, fetch=False, upload=upload,
                  result_path=str(JOBDIR / "local_result.json"), partial_path=str(JOBDIR / "local_partial.json"),
                  row_filter=lambda r: r["run"] in vruns and r["step"] in steps, partial_every_s=60)
    print("job.main rc=%d in %.0fs; uploads %s" % (rc, time.time() - t, uploads))
    res = json.loads((JOBDIR / "local_result.json").read_text())
    assert rc == 0 and res["complete"] and res["n_scored"] == len(vruns) * len(steps) * 8, (rc, res["n_scored"])
    for k in ("judge", "revision", "dtype", "data_sha256", "validation", "scores", "n_scored", "order", "timing"):
        assert k in res, k
    assert [u[1] for u in uploads][-2:] == ["partial.json", "result.json"]

    v = res["validation"]
    print("\nVALIDATION (CPU float32 rescoring vs logged GPU float16 judge_small), steps %s" % list(steps))
    print("  %-42s %4s %9s %9s %9s" % ("run", "step", "rescored", "logged", "diff"))
    for run, st in v["per_step"].items():
        for s, e in st.items():
            print("  %-42s %4s %9.5f %9.5f %+9.5f" % (run, s, e["rescored"], e["logged"], e["diff"]))
    print("  max |diff| %.5f, mean |diff| %.5f; batched vs per-step path max %.2e; passed %s" % (
        v["max_abs_diff"], v["mean_abs_diff"], v["batched_path_max_abs_diff"], v["passed"]))

    # the target pass (same 0.5B model, float32) against the validation per-step scores, per completion
    d_comp, d_step = [], []
    for run, st in v["per_step"].items():
        for s, e in st.items():
            tgt = res["scores"][run][s]
            d_comp += [abs(x - y) for x, y in zip(tgt, e["scores"])]
            d_step.append(abs(sum(tgt) / 8 - e["logged"]))
    print("TARGET PASS (0.5B fp32 stand-in): per-completion max |target - validation| %.2e; per-step mean max "
          "|target - logged| %.5f" % (max(d_comp), max(d_step)))
    print("judge_facts: %s" % json.dumps({k: res["judge_facts"][k] for k in ("yes_tokens", "no_tokens", "pad_token",
                                                                  "padding_side", "max_length", "param_dtype",
                                                                  "prompt_tokens_mean", "prompt_tokens_max",
                                                                  "n_prompts_over_max_length")}))
    print("timing: %s" % {k: (round(x, 1) if isinstance(x, float) else x) for k, x in res["timing"].items()})
    print("order: %s" % res["order"])

    if a.abort_test:
        job.VAL_TOL_STEP = -1.0
        rc2 = job.main(device="cpu", val_dtype="float32", target_dtype="float32", val_steps=(0,),
                       data_path=a.data, fetch=False, upload=upload,
                       result_path=str(JOBDIR / "local_result_abort.json"),
                       partial_path=str(JOBDIR / "local_partial_abort.json"),
                       row_filter=lambda r: False, partial_every_s=60)
        r2 = json.loads((JOBDIR / "local_result_abort.json").read_text())
        assert rc2 == 2 and r2.get("aborted") and not r2["validation"]["passed"] and r2["n_scored"] == 0
        assert r2["judge_facts"] is None, "target judge must not load after a failed gate"
        print("abort path OK: rc=%d, %s" % (rc2, r2["aborted"]))
    print("\nLOCAL TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
