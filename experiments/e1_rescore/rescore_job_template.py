# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "torch==2.11.0+cu128",
#   "transformers==5.19.0",
#   "numpy==2.5.3",
#   "accelerate==1.15.0",
#   "tokenizers==0.23.2",
#   "huggingface-hub==1.33.0",
#   "safetensors==0.8.0",
# ]
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# [tool.uv.sources]
# torch = [{ index = "pytorch-cu128" }]
# ///
"""E1 third-judge rescoring job. ONE GPU instance, ONE target judge, every stored E1 completion. ASCII only.

Template: the launcher substitutes the six placeholders on the constants block below (judge id, HF revision,
S3 data URL, S3 result prefix, data SHA-256, base64 tar.gz of the committed flightrecorder package);
experiments/e1_rescore/bake_rescore_job.py does it, compiles the result and refuses leftovers.

THE JUDGE IS THE REPO'S OWN. The package is unpacked and the scoring goes through
flightrecorder.repro.reward_judge.JudgeScorer.score, unchanged: prompt = _JUDGE_TEMPLATE.format(
problem=task.prompt, code=extract_code(completion)), right padding, truncation at 1024 tokens,
score = sigmoid(logsumexp(logits[Yes variants]) - logsumexp(logits[No variants])) at the last real token.
The revision is pinned by downloading that exact HF commit (snapshot_download) and handing JudgeScorer the
local snapshot directory, so JudgeScorer.__init__ also runs unchanged; dtype is passed as its argument
(bfloat16 for the 7B targets). Task prompts are reward_codeset.CODE_TASKS by name. The job asserts the
template string and the three judge modules against SHA-256s recorded at bake time (CRLF -> LF normalized).

ORDER
  1. aws s3 cp the data file, verify its SHA-256.
  2. VALIDATION GATE: Qwen/Qwen2.5-0.5B-Instruct (the logged hacking-arm judge) in float16 via JudgeScorer,
     all 8 completions of steps 0-9 of one hacking and one benign run, scored exactly as training did (one
     score() call per step on its 8 completions). Per-step means are compared with the logged judge_small.
     The same 160 completions are then scored again through this job's batched path (task-grouped,
     length-sorted, adaptive batch) and compared per completion; that checks the bookkeeping the target
     run relies on. Either check over tolerance aborts the job before the target model is downloaded.
     The 0.5B model is then freed.
  3. Target judge, pinned revision, bfloat16. All 153,600 completions in priority order:
       edges   steps 0-24 and 375-399 of every run
       every5  the remaining steps divisible by 5 (25..370)
       rest    everything else
     Within a chunk: grouped by task (score() takes one problem), sorted by prompt token length (longest
     first, so an OOM shows early), batches of B sequences with B x longest <= token budget. An OOM halves
     the budget and retries the same rows.
  4. partial.json (same structure, unscored entries null, "complete": false) goes to S3 after validation
     and then every ~10 min; /tmp/result.json at the end, plus a final complete partial.json and
     result.json under the same prefix. An exception or SIGTERM (the instance timeout) during step 3
     publishes partial.json and result.json with what is scored, marked "aborted", then exits non-zero.
"""
import base64
import gc
import gzip
import hashlib
import io
import json
import math
import os
import platform
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone

# ------------------------------------------------------------------------------------------------
# BAKED CONSTANTS (the launcher substitutes these six)
# ------------------------------------------------------------------------------------------------
JUDGE = "__JUDGE__"
REVISION = "__REVISION__"
S3_DATA = "__S3_DATA__"
S3_PREFIX = "__S3_PREFIX__"
DATA_SHA = "__DATA_SHA__"
_PKG_B64 = "__PKG_B64__"

# ------------------------------------------------------------------------------------------------
# FIXED CONSTANTS (recorded 2026-10-09)
# ------------------------------------------------------------------------------------------------
# main's commit sha from https://huggingface.co/api/models/<id>, fetched 2026-10-09 (all ungated)
KNOWN_REVISIONS = {
    "allenai/OLMo-2-1124-7B-Instruct": "470b1fba1ae01581f270116362ee4aa1b97f4c84",  # lastModified 2025-01-06
    "Qwen/Qwen2.5-7B-Instruct": "a09a35458c702b33eeacc393d103063234e8bc28",         # lastModified 2025-01-12
    "Qwen/Qwen2.5-0.5B-Instruct": "7ae557604adf67be50417f59c2c2f167def9a775",       # lastModified 2024-09-25,
}                                                                                    # i.e. what E1 loaded as main
TARGET_DTYPE = "bfloat16"
S3_REGION = "us-east-1"                   # the bucket's region, whatever region the instance is in
DATA_PATH = "/opt/e1/data/e1_completions.jsonl.gz"
RESULT_PATH = "/tmp/result.json"
PARTIAL_PATH = "/tmp/partial.json"
PARTIAL_EVERY_S = 600
N_GEN = 8
MAX_BS = 256
MAX_LENGTH = 1024                          # JudgeScorer's default; used here only to estimate batch sizes

# JudgeScorer's prompt template and the three modules that define the judge, at the commit E1 ran
# (reward_testhack.py as committed in 94eb433 is the version the E1 jobs imported; the others are unchanged
# since before E1). sha256 of the LF-normalized file bytes / of the template string.
TEMPLATE_SHA256 = "bf20ed637511280df73720c819d7975fcc5d3ba755a983f79711f1552d024b58"
MODULE_SHA256 = {
    "flightrecorder/repro/reward_judge.py": "33d57e3db1bcc925a2822f8f98130b0c0ae9d15edcd7c1f511d5ceccd4c2a81e",
    "flightrecorder/repro/reward_testhack.py": "b0ebfc08495c081b5236cb9203aac789587d01a311e3f8d97092faf6cf32ad47",
    "flightrecorder/repro/reward_codeset.py": "f2961c61d628da9bdaf787e992a7473da9d9aca55e6627d1ab2fafe9e8fda279",
}

# Validation gate. judge_small per log row, copied from runs/e1/main/<run>.json log[0..9] (full precision).
VAL_JUDGE = "Qwen/Qwen2.5-0.5B-Instruct"
VAL_DTYPE = "float16"                      # what the E1 jobs used (JudgeScorer default)
VAL_STEPS = tuple(range(10))
VAL_LOGGED = {
    "judge_judge05_count_vowels_s200_lr1e-6": [   # hacking arm
        0.7479248046875, 0.76702880859375, 0.7489013671875, 0.7015380859375, 0.75799560546875,
        0.7188720703125, 0.74359130859375, 0.697021484375, 0.76611328125, 0.7412109375],
    "judge_judge15_count_vowels_s200_lr1e-6": [   # benign arm
        0.7479248046875, 0.73651123046875, 0.74627685546875, 0.7352294921875, 0.7244873046875,
        0.69573974609375, 0.739990234375, 0.698760986328125, 0.74761962890625, 0.71795654296875],
}
VAL_TOL_STEP = 0.02     # |rescored per-step mean - logged judge_small|, max over the 20 steps
VAL_TOL_PATH = 0.01     # |per-step path - batched path|, max over the 160 completions

CHUNKS = (
    ("edges", "steps 0-24 and 375-399", lambda s: s <= 24 or s >= 375),
    ("every5", "remaining steps with step % 5 == 0 (25..370)", lambda s: 25 <= s <= 374 and s % 5 == 0),
    ("rest", "all other steps (25..374, step % 5 != 0)", lambda s: 25 <= s <= 374 and s % 5 != 0),
)

T0 = time.time()
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
for _k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):       # both judges are ungated: never send a token
    os.environ.pop(_k, None)


def log(msg):
    print("[%7.0fs] %s" % (time.time() - T0, msg), flush=True)


def _norm_sha(b):
    return hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest()


def _f(x):
    """float, or None when not finite (result JSON is written with allow_nan=False)."""
    x = float(x)
    return x if math.isfinite(x) else None


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------
# the repo's package, unpacked and checked
# ------------------------------------------------------------------------------------------------
_unbaked = [k for k, v in (("JUDGE", JUDGE), ("REVISION", REVISION), ("S3_DATA", S3_DATA),
                           ("S3_PREFIX", S3_PREFIX), ("DATA_SHA", DATA_SHA), ("PKG_B64", _PKG_B64[:64]))
            if v.startswith("__") and v.endswith("__")]
if _unbaked:
    raise SystemExit("template not baked: " + ", ".join(_unbaked))
assert KNOWN_REVISIONS.get(JUDGE) == REVISION, "revision for %s is not the pinned one" % JUDGE

PKG_DIR = os.path.join(tempfile.gettempdir(), "fr_rescore_pkg")
shutil.rmtree(PKG_DIR, ignore_errors=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(_PKG_B64)), mode="r:gz") as _tf:
    PKG_COMMIT = _tf.pax_headers.get("comment")          # git archive records the commit id here
    _tf.extractall(PKG_DIR, filter="data")
sys.path.insert(0, PKG_DIR)

from flightrecorder.repro.reward_judge import _JUDGE_TEMPLATE, JudgeScorer  # noqa: E402
from flightrecorder.repro.reward_testhack import extract_code  # noqa: E402
from flightrecorder.repro.reward_codeset import CODE_TASKS  # noqa: E402

TASK_BY_NAME = {t.name: t for t in CODE_TASKS}
MODULE_SHA_SEEN = {}
for _rel, _want in MODULE_SHA256.items():
    with open(os.path.join(PKG_DIR, *_rel.split("/")), "rb") as _fh:
        MODULE_SHA_SEEN[_rel] = _norm_sha(_fh.read())
    assert MODULE_SHA_SEEN[_rel] == _want, "judge module differs from the E1 version: " + _rel
assert hashlib.sha256(_JUDGE_TEMPLATE.encode()).hexdigest() == TEMPLATE_SHA256, "judge template changed"


# ------------------------------------------------------------------------------------------------
# I/O
# ------------------------------------------------------------------------------------------------
def fetch_data(path=DATA_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    part = path + ".part"
    subprocess.run(["aws", "s3", "cp", S3_DATA, part, "--only-show-errors", "--region", S3_REGION],
                   check=True, timeout=1800)
    os.replace(part, path)
    return path


def s3_upload(local, name):
    """Best effort: an upload failure is printed and never stops the scoring."""
    dest = S3_PREFIX.rstrip("/") + "/" + name
    try:
        r = subprocess.run(["aws", "s3", "cp", local, dest, "--only-show-errors", "--region", S3_REGION],
                           timeout=600, capture_output=True, text=True)
        ok = r.returncode == 0
        log("UPLOAD %s %s%s" % ("ok" if ok else "FAILED rc=%d" % r.returncode, dest,
                                "" if ok else " " + (r.stderr or "")[-300:].strip()))
        return ok
    except Exception as e:  # noqa: BLE001
        log("UPLOAD FAILED %s %r" % (dest, e))
        return False


def load_rows(path):
    with gzip.open(path, "rt", encoding="ascii") as f:
        rows = [json.loads(line) for line in f]
    assert len(rows) == 153_600, len(rows)
    keys = [(r["run"], r["step"], r["i"]) for r in rows]
    assert keys == sorted(keys) and len(set(keys)) == len(keys), "data not sorted/unique by (run, step, i)"
    runs = sorted({r["run"] for r in rows})
    assert len(runs) == 48, len(runs)
    for r in rows:
        assert r["task"] in TASK_BY_NAME, r["task"]
    return rows


def write_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, allow_nan=False)
    os.replace(tmp, path)


def snapshot(model_id, revision):
    from huggingface_hub import snapshot_download
    t = time.time()
    p = snapshot_download(model_id, revision=revision, token=False,
                          allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"])
    log("snapshot %s@%s -> %s (%.0fs)" % (model_id, revision[:12], p, time.time() - t))
    return p


def free(scorer):
    import torch
    del scorer.model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def judge_facts(scorer):
    tok, m = scorer.tok, scorer.model
    return {"yes_ids": scorer._yes_ids, "no_ids": scorer._no_ids,
            "yes_tokens": tok.convert_ids_to_tokens(scorer._yes_ids),
            "no_tokens": tok.convert_ids_to_tokens(scorer._no_ids),
            "pad_token": tok.pad_token, "padding_side": tok.padding_side, "max_length": scorer.max_length,
            "bos_prepended": bool(tok("x")["input_ids"][:1] == [tok.bos_token_id]) if tok.bos_token_id is not None else False,
            "vocab_size": int(m.get_output_embeddings().weight.shape[0]),
            "param_dtype": str(next(m.parameters()).dtype).replace("torch.", ""),
            "model_class": type(m).__name__}


# ------------------------------------------------------------------------------------------------
# scoring
# ------------------------------------------------------------------------------------------------
def token_lengths(scorer, rows, bs=2048):
    """Length of each row's judge prompt (the exact string JudgeScorer.score builds), capped at max_length."""
    out = []
    for a in range(0, len(rows), bs):
        prompts = [_JUDGE_TEMPLATE.format(problem=TASK_BY_NAME[r["task"]].prompt, code=extract_code(r["text"]))
                   for r in rows[a:a + bs]]
        ids = scorer.tok(prompts)["input_ids"]
        out += [len(x) for x in ids]
    return out


def initial_budget(scorer, device):
    """Tokens per batch (B x longest), half the free memory. Per token: the full [B, T, V] logits, one layer's
    MLP/hidden activations, and the KV cache of ALL layers (the forward builds a DynamicCache because use_cache
    defaults to config.use_cache=True; for OLMo-2-7B, 32 layers x 32 KV heads, that is ~0.5 MB per token, more
    than the logits). The KV term only sizes batches; it changes no score."""
    if device == "cpu":
        return 8192
    import torch
    free_b, _total = torch.cuda.mem_get_info()
    cfg = scorer.model.config
    el = torch.finfo(next(scorer.model.parameters()).dtype).bits // 8
    v = scorer.model.get_output_embeddings().weight.shape[0]
    nh = getattr(cfg, "num_attention_heads", 1) or 1
    nkv = getattr(cfg, "num_key_value_heads", None) or nh
    hd = getattr(cfg, "head_dim", None) or getattr(cfg, "hidden_size", 0) // nh
    kv = 2 * getattr(cfg, "num_hidden_layers", 0) * nkv * hd
    per_tok = el * (v + 2 * getattr(cfg, "intermediate_size", 0) + 8 * getattr(cfg, "hidden_size", 0) + kv)
    return int(max(MAX_LENGTH, min(131072, 0.5 * free_b / per_tok)))


def _is_oom(e):
    import torch
    s = str(e)
    return isinstance(e, torch.OutOfMemoryError) or "out of memory" in s.lower() or "CUBLAS_STATUS_ALLOC_FAILED" in s


class Scored:
    """scores[run][str(step)] = [float|None] * 8, filled in place."""

    def __init__(self):
        self.scores = {}
        self.n = 0
        self.nonfinite = 0

    def put(self, row, s):
        s = float(s)
        if not math.isfinite(s):
            self.nonfinite += 1
            s = None
        slot = self.scores.setdefault(row["run"], {}).setdefault(str(row["step"]), [None] * N_GEN)
        if slot[row["i"]] is None and s is not None:
            self.n += 1
        slot[row["i"]] = s


def score_rows(scorer, rows, lens, state, sink, tick=None):
    """Score rows through JudgeScorer.score in task-grouped, length-sorted batches. state carries the token
    budget and counters across calls; sink(row, score) records; tick(n_rows, n_tokens) reports progress."""
    import torch
    by_task = defaultdict(list)
    for r, n in zip(rows, lens):
        by_task[r["task"]].append((min(n, scorer.max_length), r))
    for task in sorted(by_task):
        grp = sorted(by_task[task], key=lambda x: -x[0])
        problem = TASK_BY_NAME[task].prompt
        j = 0
        while j < len(grp):
            longest = grp[j][0]
            n = max(1, min(MAX_BS, state["budget"] // longest, len(grp) - j))
            batch = grp[j:j + n]
            oom = False
            try:
                out = scorer.score(problem, [r["text"] for _n, r in batch])
            except (RuntimeError, torch.OutOfMemoryError) as e:
                if not _is_oom(e):
                    raise
                oom = True
            if oom:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if n == 1:
                    raise RuntimeError("OOM on a single %d-token prompt" % longest)
                state["budget"] = max(1, (n * longest) // 2)
                state["oom_halvings"] += 1
                log("OOM at B=%d x %d tokens; token budget -> %d" % (n, longest, state["budget"]))
                continue
            assert len(out) == len(batch)
            for (_n, r), s in zip(batch, out):
                sink(r, s)
            state["batches"] += 1
            state["tokens"] += sum(x for x, _r in batch)
            state["padded_tokens"] += n * longest
            j += n
            if tick:
                tick(n, sum(x for x, _r in batch))


# ------------------------------------------------------------------------------------------------
# validation gate
# ------------------------------------------------------------------------------------------------
def run_validation(rows, device, dtype, steps=VAL_STEPS):
    t = time.time()
    idx = {(r["run"], r["step"], r["i"]): r for r in rows if r["run"] in VAL_LOGGED and r["step"] in steps}
    vrows = [idx[(run, s, i)] for run in VAL_LOGGED for s in steps for i in range(N_GEN)]
    path = snapshot(VAL_JUDGE, KNOWN_REVISIONS[VAL_JUDGE])
    sc = JudgeScorer(path, device=device, dtype=dtype)          # the repo's class, E1's defaults otherwise
    facts = judge_facts(sc)

    # (a) exactly as training scored them: one score() call per step on its 8 completions
    per_step, per_comp_a, diffs = {}, {}, []
    for run in VAL_LOGGED:
        per_step[run] = {}
        for s in steps:
            rs = [idx[(run, s, i)] for i in range(N_GEN)]
            out = sc.score(TASK_BY_NAME[rs[0]["task"]].prompt, [r["text"] for r in rs])
            for r, x in zip(rs, out):
                per_comp_a[(run, s, r["i"])] = float(x)
            m = sum(out) / len(out)
            d = m - VAL_LOGGED[run][s]
            diffs.append(abs(d))
            per_step[run][str(s)] = {"rescored": _f(m), "logged": VAL_LOGGED[run][s], "diff": _f(d),
                                     "scores": [_f(x) for x in out], "batched": None}
            log("VALIDATE %s step %d rescored %.5f logged %.5f diff %+.5f" % (run, s, m, VAL_LOGGED[run][s], d))

    # (b) the same completions through the batched path the target run uses
    got = {}
    st = {"budget": initial_budget(sc, device), "oom_halvings": 0, "batches": 0, "tokens": 0, "padded_tokens": 0}
    score_rows(sc, vrows, token_lengths(sc, vrows), st,
               lambda r, x: got.__setitem__((r["run"], r["step"], r["i"]), float(x)))
    assert set(got) == set(per_comp_a), "batched path lost or invented rows"
    pdiff = max(abs(got[k] - per_comp_a[k]) for k in per_comp_a)
    for run in VAL_LOGGED:
        for s in steps:
            per_step[run][str(s)]["batched"] = [_f(got[(run, s, i)]) for i in range(N_GEN)]
    free(sc)
    del sc

    finite = all(math.isfinite(x) for x in list(per_comp_a.values()) + list(got.values()))
    mstep = max(diffs)
    passed = bool(finite and mstep <= VAL_TOL_STEP and pdiff <= VAL_TOL_PATH)
    v = {"model": VAL_JUDGE, "revision": KNOWN_REVISIONS[VAL_JUDGE], "dtype": dtype, "device": device,
         "runs": list(VAL_LOGGED), "steps": list(steps), "n_completions": len(vrows),
         "per_step": per_step, "max_abs_diff": _f(mstep), "mean_abs_diff": _f(sum(diffs) / len(diffs)),
         "batched_path_max_abs_diff": _f(pdiff), "batched_path_batches": st["batches"],
         "tolerance_step": VAL_TOL_STEP, "tolerance_path": VAL_TOL_PATH, "all_finite": finite,
         "passed": passed, "judge_facts": facts, "seconds": time.time() - t}
    log("VALIDATION %s: max |rescored - logged| per-step mean %.5f (tol %.3f); batched vs per-step path "
        "max %.2e (tol %.3f)" % ("PASSED" if passed else "FAILED", mstep, VAL_TOL_STEP, pdiff, VAL_TOL_PATH))
    return v


# ------------------------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------------------------
def env_info(device):
    import numpy
    import torch
    import transformers
    try:
        import accelerate
        acc = accelerate.__version__
    except Exception:  # noqa: BLE001
        acc = None
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "numpy": numpy.__version__, "accelerate": acc,
            "cuda": torch.version.cuda, "gpu": gpu, "device": device, "host": platform.node()}


def main(device="cuda", val_dtype=VAL_DTYPE, target_dtype=TARGET_DTYPE, val_steps=VAL_STEPS,
         data_path=DATA_PATH, fetch=True, upload=s3_upload, result_path=RESULT_PATH,
         partial_path=PARTIAL_PATH, row_filter=None, partial_every_s=PARTIAL_EVERY_S):
    """Production: main() with the defaults. The keyword overrides exist for the local CPU test only."""
    import torch
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    timing = {"start_utc": started}
    if device == "cuda":
        assert torch.cuda.is_available(), "no CUDA device"
    log("E1 rescore: judge %s@%s dtype %s; package commit %s" % (JUDGE, REVISION, target_dtype, PKG_COMMIT))

    t = time.time()
    if fetch:
        fetch_data(data_path)
    sha = sha256_file(data_path)
    assert sha == DATA_SHA, "data sha256 %s != baked %s" % (sha, DATA_SHA)
    rows = load_rows(data_path)
    timing["data_s"] = time.time() - t
    log("data: %d completions, sha256 ok" % len(rows))
    runs = {}
    for r in rows:
        runs.setdefault(r["run"], {"arm": r["arm"], "task": r["task"], "seed": r["seed"]})

    out = {"judge": JUDGE, "revision": REVISION, "dtype": target_dtype, "data_sha256": sha,
           "validation": None, "scores": {}, "n_scored": 0, "order": None, "timing": timing,
           "complete": False, "runs": runs, "env": env_info(device), "judge_facts": None,
           "provenance": {"package_commit": PKG_COMMIT, "template_sha256": TEMPLATE_SHA256,
                          "module_sha256": MODULE_SHA_SEEN, "known_revisions": KNOWN_REVISIONS,
                          "s3_data": S3_DATA, "s3_prefix": S3_PREFIX},
           "mode": "production" if (device == "cuda" and row_filter is None) else "local_test"}

    def publish(final=False):
        write_json(out, partial_path)
        upload(partial_path, "partial.json")
        if final:
            write_json(out, result_path)
            upload(result_path, "result.json")

    # ---- validation gate
    t = time.time()
    out["validation"] = val = run_validation(rows, device, val_dtype, tuple(val_steps))
    timing["validation_s"] = time.time() - t
    if not val["passed"]:
        out["aborted"] = "validation gate failed"
        timing["total_s"] = time.time() - T0
        publish(final=True)
        log("ABORT: validation gate failed; the target judge was not loaded")
        return 2
    publish()

    # ---- target judge. From here on, any exception (or SIGTERM from the instance's timeout, see __main__)
    # publishes what is scored so far, marked "aborted", before the job exits.
    sink = Scored()
    out["scores"] = sink.scores
    state = {"budget": None, "oom_halvings": 0, "batches": 0, "tokens": 0, "padded_tokens": 0}
    clock = {"last_print": time.time(), "last_upload": time.time(), "t_score": time.time()}
    try:
        t = time.time()
        path = snapshot(JUDGE, REVISION)
        sc = JudgeScorer(path, device=device, dtype=target_dtype)
        out["judge_facts"] = judge_facts(sc)
        out["judge_facts"]["snapshot_files"] = sorted(os.listdir(path))
        assert out["judge_facts"]["param_dtype"] == target_dtype, out["judge_facts"]["param_dtype"]
        timing["target_load_s"] = time.time() - t
        log("target loaded: %s" % json.dumps(out["judge_facts"]))

        todo = rows if row_filter is None else [r for r in rows if row_filter(r)]
        t = time.time()
        lens = token_lengths(sc, todo)
        timing["tokenize_s"] = time.time() - t
        out["judge_facts"]["n_prompts_over_max_length"] = sum(n > sc.max_length for n in lens)
        out["judge_facts"]["prompt_tokens_mean"] = sum(lens) / max(len(lens), 1)
        out["judge_facts"]["prompt_tokens_max"] = max(lens) if lens else 0
        log("tokenized %d prompts in %.0fs: mean %.0f, max %d, %d over %d (truncated, as in training)" % (
            len(lens), timing["tokenize_s"], out["judge_facts"]["prompt_tokens_mean"],
            out["judge_facts"]["prompt_tokens_max"], out["judge_facts"]["n_prompts_over_max_length"],
            sc.max_length))

        state["budget"] = initial_budget(sc, device)
        timing["token_budget_initial"] = state["budget"]
        plan = []
        for name, rule, pred in CHUNKS:
            sel = [(r, n) for r, n in zip(todo, lens) if pred(r["step"])]
            plan.append((name, rule, sel))
        out["order"] = {"chunks": [{"name": nm, "rule": rule, "n": len(sel)} for nm, rule, sel in plan],
                        "chunks_done": [],
                        "within_chunk": "grouped by task; sorted by prompt token length, longest first; "
                                        "B x longest <= token budget, B <= %d; OOM halves the budget" % MAX_BS}
        assert sum(len(sel) for _nm, _rule, sel in plan) == len(todo)
        log("plan: " + ", ".join("%s %d" % (nm, len(sel)) for nm, _rule, sel in plan) +
            "; token budget %d" % state["budget"])

        total = len(todo)
        clock.update({"last_print": time.time(), "last_upload": time.time(), "t_score": time.time()})

        def tick(n, ntok):
            now = time.time()
            if now - clock["last_print"] >= 60:
                clock["last_print"] = now
                el = now - clock["t_score"]
                rate = sink.n / max(el, 1e-9)
                log("PROGRESS %d/%d (%.1f%%) %.1f completions/s %.0f tok/s, eta %.2f h, batches %d, budget %d, "
                    "ooms %d" % (sink.n, total, 100.0 * sink.n / max(total, 1), rate,
                                 state["tokens"] / max(el, 1e-9), (total - sink.n) / max(rate, 1e-9) / 3600,
                                 state["batches"], state["budget"], state["oom_halvings"]))
            if now - clock["last_upload"] >= partial_every_s:
                clock["last_upload"] = now
                out["n_scored"] = sink.n
                timing["scoring_s"] = now - clock["t_score"]
                publish()

        for name, _rule, sel in plan:
            tc = time.time()
            score_rows(sc, [r for r, _n in sel], [n for _r, n in sel], state, sink.put, tick)
            timing["chunk_%s_s" % name] = time.time() - tc
            out["order"]["chunks_done"].append(name)
            out["n_scored"] = sink.n
            log("CHUNK %s done: %d rows in %.0fs; scored %d/%d" % (name, len(sel), time.time() - tc, sink.n,
                                                                  total))
            clock["last_upload"] = time.time()
            publish()
    except BaseException as e:  # noqa: BLE001  includes SystemExit from the SIGTERM handler
        out["aborted"] = "target stage stopped early: %s" % repr(e)[:300]
        out["n_scored"] = sink.n
        out["n_nonfinite"] = sink.nonfinite
        timing["scoring_s"] = time.time() - clock["t_score"]
        timing["total_s"] = time.time() - T0
        log("ABORT in the target stage after %d scores: %s; publishing what is scored" % (sink.n, out["aborted"]))
        publish(final=True)
        raise

    timing["scoring_s"] = time.time() - clock["t_score"]
    timing.update({"batches": state["batches"], "oom_halvings": state["oom_halvings"],
                   "token_budget_final": state["budget"], "tokens": state["tokens"],
                   "padded_tokens": state["padded_tokens"],
                   "completions_per_s": sink.n / max(timing["scoring_s"], 1e-9),
                   "tokens_per_s": state["tokens"] / max(timing["scoring_s"], 1e-9)})
    out["n_scored"] = sink.n
    out["n_nonfinite"] = sink.nonfinite
    missing = sum(x is None for st in sink.scores.values() for v in st.values() for x in v)
    out["complete"] = bool(sink.n == total and missing == 0 and sink.nonfinite == 0
                           and len([1 for st in sink.scores.values() for _ in st]) * N_GEN == total)
    timing["total_s"] = time.time() - T0
    publish(final=True)

    # descriptive only, so the job log shows whether the judge is alive (not an analysis)
    for arm in ("hacking", "benign"):
        xs = [x for run, st in sink.scores.items() if runs[run]["arm"] == arm for v in st.values() for x in v
              if x is not None]
        if xs:
            mu = sum(xs) / len(xs)
            log("SUMMARY %s: n %d, mean score %.4f, share in (0.01, 0.99) %.3f" % (
                arm, len(xs), mu, sum(0.01 < x < 0.99 for x in xs) / len(xs)))
    log("DONE complete=%s n_scored %d/%d in %.2f h; %s" % (out["complete"], sink.n, total,
                                                           timing["total_s"] / 3600, result_path))
    return 0 if out["complete"] else 3


def _on_sigterm(signum, frame):
    """The instance's `timeout` sends SIGTERM (SIGKILL 120 s later). Turn it into SystemExit so main() publishes
    the scores done so far; ignore any further SIGTERM while that upload runs."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise SystemExit(143)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _on_sigterm)
    sys.exit(main())
