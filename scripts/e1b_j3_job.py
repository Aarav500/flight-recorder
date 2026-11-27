"""E1b J3 SCORING JOB: template, data builder and bake(). Builds files only: nothing here launches, uploads or spends.
ASCII only.

WHAT THE JOB IS
  Pre-registration section 2 (the third judge, branch A), spec 3.6 and section 12's "J3 scoring" row: one offline
  job scores all 8 completions x 400 steps of every USED E1b run with J3 = Qwen/Qwen2.5-7B-Instruct at revision
  a09a35458c702b33eeacc393d103063234e8bc28 in bfloat16, through the repo's flightrecorder.repro.reward_judge.
  JudgeScorer (revision= passed to both of its from_pretrained calls): _JUDGE_TEMPLATE, reward_testhack.extract_code,
  the task prompt from reward_codeset.CODE_TASKS, right padding, truncation at 1024 tokens. That is the E1 third-
  judge rescoring's setup (experiments/e1_rescore/rescore_job_template.py, frozen; its patterns are rewritten here,
  nothing is imported from it): the same validation gate, chunk order, task-grouped length-sorted batches with an
  OOM-halving token budget, and the result format of experiments/e1_rescore/results/qwen25_7b.json, keyed here by
  E1b run ids (build contract section 3). The collector (scripts/launch_e1b_j3.py --collect) writes it to
  runs/e1b/j3/qwen25_7b.json, which experiments/e1b_primary.load_j3 reads.

THE DATA (build_data)
  The used runs come from experiments/e1b_primary.load_e1b (called, never re-implemented), after the data
  manifest check that the registered analysis makes first (spec 7.8). The data file is gzip (mtime 0, no file
  name, level 9, so the same runs give the same bytes and SHA-256) of JSON lines, ASCII: one header line
  {"header": {...}} (the used run ids with arm, task, seed and the analysis's run tag; the used tags per arm and the
  data manifest's SHA-256, in the form e1b_primary's results file records them; selection counts), then one row
  per stored completion, sorted by (run, step, i):
      {"run": run id, "arm", "task", "seed", "step", "i", "text": completions[step].texts[i] (the full stored
       text), "judge_small": completions[step].judge_small[i] (for the validation gate only)}
  prep_data(out_dir) builds it in a child process that first runs e1b_primary.verify_pins_b() (spec 7.6), before
  this module is imported, so the pins see only the analysis's own modules.

THE BAKED JOB (UV_J3 below; run on the instance as job.py next to job.py.lock and data.jsonl.gz)
  * Environment: the PEP 723 header is byte-identical to scripts/e1b_job.py's, so experiments/e1b_uv.lock is its
    lock (uv checks a script lock against the header only). Its torch, transformers, numpy, accelerate,
    tokenizers, huggingface-hub and safetensors are the versions the E1 rescoring job pinned. Run with
        uv run --no-config --locked --python 3.12.15 --managed-python job.py
  * Provenance, checked before any model loads (exit 1, "E1B J3 STOP: ..." on stderr): the embedded package's
    SHA-256, job.py.lock's SHA-256, the extracted reward_testhack.py and reward_codeset.py and the
    _JUDGE_TEMPLATE string against the E1 rescoring's SHA-256s, and the data file's SHA-256 and row count against
    the baked values. Measured and written to the result: job_file_sha256 (the job's own bytes), the SHA-256 of
    the three judge modules, and the Hub snapshot each model was loaded from (read from the cache after loading).
    As the training job does, after each judge loads and before it scores anything the job stops (exit 1, "E1B J3
    STOP: ..." on stderr; result.json marked aborted, nothing scored) unless the Hub cache holds exactly one snapshot
    of that judge's repo and it is the pinned revision (check_snapshot). The collector refuses a result whose
    recorded snapshots differ from the registered revisions.
  * Order: data; VALIDATION GATE (Qwen/Qwen2.5-0.5B-Instruct at 7ae55760, float16, the training judge, on all 8
    completions of steps 0-9 of the first used hacking run and the first used twin run: per-step means against
    the stored judge_small, then the same completions through the batched path; over tolerance -> rc 2 and the
    target judge is never loaded); target judge; all rows in chunk order edges (steps 0-24, 375-399), every5,
    rest.
  * Output: result.json and partial.json next to the job (partial.json rewritten after validation, every 10
    minutes and after each chunk; the instance uploads both). An exception or SIGTERM (the instance timeout) in the
    target stage writes result.json with what is scored, marked "aborted", then exits non-zero.
  * Monitoring: stdout carries only "J3 <phase> <scored>/<total>" lines (counts); stderr carries phases, counts
    and whole-second durations. No score value is printed anywhere. status.json (E1B_J3_STATUS_PATH, default next
    to the job) holds {run_id, phase, n_scored, n_total, rc, ts}, rc null (the instance writes the final rc).
  * Exit codes: 0 complete; 2 validation gate failed; 3 finished but incomplete; 1 provenance stop (a Hub
    snapshot that is not the pin included) or error; 143 SIGTERM.

BAKE
  bake(commit, data_path, run_id, out_dir) -> (job_path, provenance). It reads this template and the lock from the
  commit's blobs (CRLF refused), refuses unless this file and scripts/e1b_job.py equal their committed versions and
  the header equals e1b_job's, archives flightrecorder/ at the commit with scripts/e1b_job.package_tar (git
  archive, machine-independent settings), checks the archived judge modules against the E1 rescoring's SHA-256s,
  checks the data file, substitutes and checks every placeholder and py_compiles the job. It writes
  <out_dir>/<run_id>/job.py, job.py.lock and data.jsonl.gz. provenance = {run_id, git_commit, package_sha256,
  template_sha256, lock_sha256, job_sha256, data_sha256, n_completions, n_runs}.

    python scripts/e1b_j3_job.py --prep OUT_DIR                        # verify pins, load_e1b, write the data
    python scripts/e1b_j3_job.py --bake COMMIT DATA RUN_ID OUT_DIR      # bake one job, print the provenance
"""
from __future__ import annotations

import ast
import base64
import gzip
import hashlib
import io
import json
import py_compile
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True       # leave no new __pycache__ next to the pinned files
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import e1b_job as J  # noqa: E402  _git, _blob, package_tar, header, LOCK_PATH, TEMPLATE_PATH, PYTHON_VERSION

TEMPLATE_PATH = "scripts/e1b_j3_job.py"
LOCK_PATH = J.LOCK_PATH
JUDGE_SHORT = "qwen25_7b"
STAGE = "j3"
# J3 (pre-registration section 2; spec 3.6; build contract section 2). experiments/e1b_primary.J3 holds the same.
J3 = {"judge": "Qwen/Qwen2.5-7B-Instruct", "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
      "dtype": "bfloat16"}
VAL = {"judge": "Qwen/Qwen2.5-0.5B-Instruct", "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
       "dtype": "float16"}     # judge_small as the E1b jobs load it (scripts/e1b_job.py JUDGE_DTYPE)
RUN_ID_RE = r"e1b_j3_qwen25_7b_a[1-9][0-9]*"
STEPS, GEN = 400, 8
DATA_NAME = "data.jsonl.gz"
DATA_FORMAT = "e1b_j3_data v1"
ROW_KEYS = ("run", "arm", "task", "seed", "step", "i", "text", "judge_small")
REQUIRED = ("flightrecorder/__init__.py", "flightrecorder/repro/__init__.py", "flightrecorder/repro/reward_judge.py",
            "flightrecorder/repro/reward_testhack.py", "flightrecorder/repro/reward_codeset.py")
# The E1 rescoring's judge definition (experiments/e1_rescore/rescore_job_template.py, TEMPLATE_SHA256 and
# MODULE_SHA256; LF-normalized): extract_code and the task prompts must be E1's, and so must the judge template.
# reward_judge.py itself changed after E1 (the optional revision argument), so its SHA-256 is recorded, not checked.
E1_TEMPLATE_SHA256 = "bf20ed637511280df73720c819d7975fcc5d3ba755a983f79711f1552d024b58"
E1_MODULE_SHA256 = {
    "flightrecorder/repro/reward_testhack.py": "b0ebfc08495c081b5236cb9203aac789587d01a311e3f8d97092faf6cf32ad47",
    "flightrecorder/repro/reward_codeset.py": "f2961c61d628da9bdaf787e992a7473da9d9aca55e6627d1ab2fafe9e8fda279",
}


# ============================================================================================
# THE BAKED JOB. The header is scripts/e1b_job.py's, byte for byte (bake checks it).
# ============================================================================================
UV_J3 = r'''# /// script
# requires-python = "==3.12.*"
# dependencies = ["torch==2.11.0", "transformers==5.19.0", "trl==1.14.2", "datasets==5.1.0", "accelerate==1.15.0", "numpy==2.5.3"]
# [tool.uv]
# exclude-newer = "2026-10-08T02:35:00Z"
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# [tool.uv.sources]
# torch = [{ index = "pytorch-cu128" }]
# ///
# E1b J3 scoring job, baked by scripts/e1b_j3_job.py from a named commit. Run with uv 0.12.23, with job.py.lock
# (experiments/e1b_uv.lock) and data.jsonl.gz next to this file:
#     uv run --no-config --locked --python 3.12.15 --managed-python job.py
# stdout carries only "J3 <phase> <scored>/<total>" lines; stderr only phases, counts and whole-second durations.
# No score value is printed anywhere: scores go to result.json and partial.json only.
import base64, gzip, hashlib, io, json, math, os, platform, signal, sys, tarfile, tempfile, time  # noqa: E401
from collections import defaultdict
from datetime import datetime, timezone

_STDOUT = sys.stdout
sys.stdout = sys.stderr     # from here on any print, the libraries' included, goes to stderr

RUN_ID = "__RUN_ID__"
GIT_COMMIT, PACKAGE_SHA256 = "__GIT_COMMIT__", "__PACKAGE_SHA256__"
TEMPLATE_SHA256, LOCK_SHA256 = "__TEMPLATE_SHA256__", "__LOCK_SHA256__"
DATA_SHA256, N_ROWS = "__DATA_SHA256__", __N_ROWS__

TARGET = {"judge": "Qwen/Qwen2.5-7B-Instruct", "revision": "a09a35458c702b33eeacc393d103063234e8bc28", "dtype": "bfloat16"}
VALIDATION = {"judge": "Qwen/Qwen2.5-0.5B-Instruct", "revision": "7ae557604adf67be50417f59c2c2f167def9a775", "dtype": "float16"}
REGISTERED_TARGET = {"judge": "Qwen/Qwen2.5-7B-Instruct", "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
                     "dtype": "bfloat16"}
DEVICE = "cuda"
CPU_TOKEN_BUDGET = 8192
STEPS, GEN = 400, 8
VAL_STEPS = tuple(range(10))
VAL_TOL_STEP = 0.02     # |rescored per-step mean - stored judge_small mean|, max over the steps (E1 rescoring's)
VAL_TOL_PATH = 0.01     # |per-step path - batched path|, max over the completions (E1 rescoring's)
MAX_BS = 256
MAX_LENGTH = 1024       # JudgeScorer's default; here only to size batches
PARTIAL_EVERY_S = 600
STATUS_EVERY_S = 60
E1_TEMPLATE_SHA256 = "bf20ed637511280df73720c819d7975fcc5d3ba755a983f79711f1552d024b58"
E1_MODULE_SHA256 = {
    "flightrecorder/repro/reward_testhack.py": "b0ebfc08495c081b5236cb9203aac789587d01a311e3f8d97092faf6cf32ad47",
    "flightrecorder/repro/reward_codeset.py": "f2961c61d628da9bdaf787e992a7473da9d9aca55e6627d1ab2fafe9e8fda279",
}
RECORDED_MODULES = ("flightrecorder/repro/reward_judge.py", "flightrecorder/repro/reward_testhack.py",
                    "flightrecorder/repro/reward_codeset.py")
CHUNKS = (
    ("edges", "steps 0-24 and 375-399", lambda s: s <= 24 or s >= 375),
    ("every5", "remaining steps with step % 5 == 0 (25..370)", lambda s: 25 <= s <= 374 and s % 5 == 0),
    ("rest", "all other steps (25..374, step % 5 != 0)", lambda s: 25 <= s <= 374 and s % 5 != 0),
)
HERE = os.path.dirname(os.path.abspath(__file__))
STATUS_PATH = os.environ.get("E1B_J3_STATUS_PATH", os.path.join(HERE, "status.json"))
DATA_PATH = os.path.join(HERE, "data.jsonl.gz")
RESULT_PATH = os.path.join(HERE, "result.json")
PARTIAL_PATH = os.path.join(HERE, "partial.json")
T0 = time.time()
STATE = {"phase": "start", "n_scored": 0}
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
for _k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):      # both judges are ungated: never send a token
    os.environ.pop(_k, None)


def _stop(why):
    sys.exit("E1B J3 STOP: " + why)


def say(msg):
    """stderr: phases, counts and whole-second durations only, never a score."""
    sys.stderr.write("[%6ds] %s\n" % (int(time.time() - T0), msg))
    sys.stderr.flush()


def _status():
    tmp = STATUS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"run_id": RUN_ID, "phase": STATE["phase"], "n_scored": int(STATE["n_scored"]),
                   "n_total": int(N_ROWS), "rc": None,
                   "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}, f)
    os.replace(tmp, STATUS_PATH)


def progress(phase, n=None):
    """One stdout line "J3 <phase> <scored>/<total>" and a status.json rewrite."""
    STATE["phase"] = phase
    if n is not None:
        STATE["n_scored"] = int(n)
    _STDOUT.write("J3 %s %d/%d\n" % (phase, STATE["n_scored"], N_ROWS))
    _STDOUT.flush()
    _status()


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _lf_sha(b):
    return _sha(b.replace(b"\r\n", b"\n"))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# Provenance, checked before the package is imported or any model loads.
_left = [k for k, v in (("RUN_ID", RUN_ID), ("GIT_COMMIT", GIT_COMMIT), ("DATA_SHA256", DATA_SHA256))
         if v.startswith("__") and v.endswith("__")]
if _left:
    _stop("template not baked: " + ", ".join(_left))
with open(os.path.abspath(__file__), "rb") as _f:
    JOB_FILE_SHA256 = _sha(_f.read())       # measured: the bytes this instance actually runs
_TAR = gzip.decompress(base64.b64decode("__PKG_B64__"))
if _sha(_TAR) != PACKAGE_SHA256:
    _stop("the embedded package's SHA-256 differs from package_sha256")
_LOCK = os.path.abspath(__file__) + ".lock"
if not os.path.isfile(_LOCK):
    _stop("no job.py.lock next to the job")
with open(_LOCK, "rb") as _f:
    if _sha(_f.read()) != LOCK_SHA256:
        _stop("job.py.lock's SHA-256 differs from lock_sha256")
_FR = tempfile.mkdtemp(prefix="e1b_j3_fr_")    # a fresh directory, so only the package's own files can import
with tarfile.open(fileobj=io.BytesIO(_TAR), mode="r:") as _tf:
    _tf.extractall(_FR, filter="data")
MODULE_SHA256 = {}
for _rel in RECORDED_MODULES:
    with open(os.path.join(_FR, *_rel.split("/")), "rb") as _f:
        MODULE_SHA256[_rel] = _lf_sha(_f.read())
for _rel, _want in E1_MODULE_SHA256.items():
    if MODULE_SHA256[_rel] != _want:
        _stop("%s differs from the E1 rescoring's version" % _rel)
sys.path.insert(0, _FR)
_status()

import flightrecorder  # noqa: E402
from flightrecorder.repro.reward_judge import _JUDGE_TEMPLATE, JudgeScorer  # noqa: E402
from flightrecorder.repro.reward_testhack import extract_code  # noqa: E402
from flightrecorder.repro.reward_codeset import CODE_TASKS  # noqa: E402

if not os.path.abspath(flightrecorder.__file__).startswith(os.path.abspath(_FR) + os.sep):
    _stop("flightrecorder was imported from outside the embedded package")
if _sha(_JUDGE_TEMPLATE.encode("utf-8")) != E1_TEMPLATE_SHA256:
    _stop("the judge template differs from the E1 rescoring's")
TASK_BY_NAME = {t.name: t for t in CODE_TASKS}


def _f(x):
    """float, or None when not finite (the result is written with allow_nan=False)."""
    x = float(x)
    return x if math.isfinite(x) else None


def write_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="ascii", newline="\n") as f:
        json.dump(obj, f, allow_nan=False)
    os.replace(tmp, path)


def load_data(path):
    """(header, rows). Refuses a file that is not the baked one or not in the builder's shape."""
    sha = sha256_file(path)
    if sha != DATA_SHA256:
        _stop("data SHA-256 %s differs from the baked %s" % (sha, DATA_SHA256))
    with gzip.open(path, "rt", encoding="ascii") as f:
        header = json.loads(f.readline())["header"]
        rows = [json.loads(line) for line in f]
    if len(rows) != N_ROWS or header.get("n_rows") != N_ROWS:
        _stop("data has %d rows, header %s, baked %d" % (len(rows), header.get("n_rows"), N_ROWS))
    keys = [(r["run"], r["step"], r["i"]) for r in rows]
    if keys != sorted(keys) or len(set(keys)) != len(keys):
        _stop("data rows are not sorted and unique by (run, step, i)")
    runs = header["runs"]
    per = defaultdict(int)
    for r in rows:
        if r["task"] not in TASK_BY_NAME or r["run"] not in runs or runs[r["run"]]["task"] != r["task"]:
            _stop("row %s %s %s does not match the header" % (r["run"], r["step"], r["i"]))
        per[r["run"]] += 1
    if set(per) != set(runs) or any(n != STEPS * GEN for n in per.values()):
        _stop("data does not hold %d x %d completions of every header run" % (STEPS, GEN))
    return header, rows


def hub_cache(model_id):
    """The Hub cache entry of model_id after loading: snapshot directory names and refs (a refs/main would show a
    read that resolved the default branch)."""
    from huggingface_hub import constants
    d = os.path.join(constants.HF_HUB_CACHE, "models--" + model_id.replace("/", "--"))
    snaps = sorted(os.listdir(os.path.join(d, "snapshots"))) if os.path.isdir(os.path.join(d, "snapshots")) else []
    refs = {}
    if os.path.isdir(os.path.join(d, "refs")):
        for name in sorted(os.listdir(os.path.join(d, "refs"))):
            with open(os.path.join(d, "refs", name), encoding="utf-8") as fh:
                refs[name] = fh.read().strip()
    return {"snapshots": snaps, "refs": refs}


def check_snapshot(name, spec, cache):
    """After spec's judge has loaded: stop (E1B J3 STOP, exit 1) unless the Hub cache holds exactly one snapshot of
    its repo and that snapshot is its pinned revision, so every file the Hub served came from the pinned commit (the
    training job's rule, scripts/e1b_job.py). Called before anything is scored with that judge."""
    if cache["snapshots"] != [spec["revision"]]:
        _stop("the Hub snapshot(s) the %s %s was loaded from are [%s], not exactly its pinned revision %s"
              % (name, spec["judge"], ", ".join(cache["snapshots"]), spec["revision"]))


def free(scorer):
    import torch
    del scorer.model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def judge_facts(scorer):
    tok, m = scorer.tok, scorer.model
    bos = tok.bos_token_id
    return {"yes_ids": scorer._yes_ids, "no_ids": scorer._no_ids,
            "yes_tokens": tok.convert_ids_to_tokens(scorer._yes_ids),
            "no_tokens": tok.convert_ids_to_tokens(scorer._no_ids),
            "pad_token": tok.pad_token, "padding_side": tok.padding_side, "max_length": scorer.max_length,
            "revision": scorer.revision,
            "bos_prepended": bool(tok("x")["input_ids"][:1] == [bos]) if bos is not None else False,
            "vocab_size": int(m.get_output_embeddings().weight.shape[0]),
            "param_dtype": str(next(m.parameters()).dtype).replace("torch.", ""),
            "model_class": type(m).__name__}


def token_lengths(scorer, rows, bs=2048):
    """Length of each row's judge prompt (the exact string JudgeScorer.score builds)."""
    out = []
    for a in range(0, len(rows), bs):
        prompts = [_JUDGE_TEMPLATE.format(problem=TASK_BY_NAME[r["task"]].prompt, code=extract_code(r["text"]))
                   for r in rows[a:a + bs]]
        out += [len(x) for x in scorer.tok(prompts)["input_ids"]]
    return out


def initial_budget(scorer, device):
    """Tokens per batch (B x longest): half the free GPU memory over the per-token cost of the [B, T, V] logits,
    one layer's activations and the KV cache of all layers (the E1 rescoring's sizing). It sizes batches only."""
    if device == "cpu":
        return CPU_TOKEN_BUDGET
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
    """scores[run][str(step)] = [float|None] * GEN, filled in place."""

    def __init__(self):
        self.scores, self.n, self.nonfinite = {}, 0, 0

    def put(self, row, s):
        s = float(s)
        if not math.isfinite(s):
            self.nonfinite += 1
            s = None
        slot = self.scores.setdefault(row["run"], {}).setdefault(str(row["step"]), [None] * GEN)
        if slot[row["i"]] is None and s is not None:
            self.n += 1
        slot[row["i"]] = s


def score_rows(scorer, rows, lens, state, sink, tick=None):
    """Rows through JudgeScorer.score in task-grouped, length-sorted batches (longest first, B x longest <= the
    token budget, B <= MAX_BS); an OOM halves the budget and retries the same rows."""
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
            try:
                out = scorer.score(problem, [r["text"] for _n, r in batch])
            except (RuntimeError, torch.OutOfMemoryError) as e:
                if not _is_oom(e):
                    raise
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if n == 1:
                    raise RuntimeError("OOM on a single %d-token prompt" % longest)
                state["budget"] = max(1, (n * longest) // 2)
                state["oom_halvings"] += 1
                say("OOM at B=%d x %d tokens; token budget now %d" % (n, longest, state["budget"]))
                continue
            if len(out) != len(batch):
                raise RuntimeError("JudgeScorer returned %d scores for %d texts" % (len(out), len(batch)))
            for (_n, r), s in zip(batch, out):
                sink(r, s)
            state["batches"] += 1
            state["tokens"] += sum(x for x, _r in batch)
            state["padded_tokens"] += n * longest
            j += n
            if tick:
                tick()


def validation_runs(header):
    """The first used hacking run and the first used twin run, in run-id order."""
    out = []
    for arm in ("judge", "judgegate"):
        ids = sorted(rid for rid, v in header["runs"].items() if v["arm"] == arm)
        if ids:
            out.append(ids[0])
    return out


def run_validation(rows, header, device, dtype, steps=VAL_STEPS):
    """The E1 rescoring's gate with the training judge: (a) one score() call per step on its 8 completions, the
    per-step mean against the stored judge_small mean; (b) the same completions through score_rows. Returns the
    record (values go to the result file only)."""
    t = time.time()
    vruns = validation_runs(header)
    idx = {(r["run"], r["step"], r["i"]): r for r in rows if r["run"] in vruns and r["step"] in steps}
    vrows = [idx[(run, s, i)] for run in vruns for s in steps for i in range(GEN)]
    sc = JudgeScorer(VALIDATION["judge"], device=device, dtype=dtype, revision=VALIDATION["revision"])
    facts = judge_facts(sc)
    cache = hub_cache(VALIDATION["judge"])
    check_snapshot("validation judge", VALIDATION, cache)     # before it scores anything
    per_step, comp_a, diffs = {}, {}, []
    for run in vruns:
        per_step[run] = {}
        for s in steps:
            rs = [idx[(run, s, i)] for i in range(GEN)]
            got = sc.score(TASK_BY_NAME[rs[0]["task"]].prompt, [r["text"] for r in rs])
            for r, x in zip(rs, got):
                comp_a[(run, s, r["i"])] = float(x)
            m = sum(got) / len(got)
            stored = sum(float(r["judge_small"]) for r in rs) / GEN
            diffs.append(abs(m - stored))
            per_step[run][str(s)] = {"rescored": _f(m), "stored": _f(stored), "diff": _f(m - stored),
                                     "scores": [_f(x) for x in got], "batched": None}
    got_b = {}
    st = {"budget": initial_budget(sc, device), "oom_halvings": 0, "batches": 0, "tokens": 0, "padded_tokens": 0}
    score_rows(sc, vrows, token_lengths(sc, vrows), st,
               lambda r, x: got_b.__setitem__((r["run"], r["step"], r["i"]), float(x)))
    if set(got_b) != set(comp_a):
        raise RuntimeError("the batched path lost or invented rows")
    pdiff = max(abs(got_b[k] - comp_a[k]) for k in comp_a)
    for run in vruns:
        for s in steps:
            per_step[run][str(s)]["batched"] = [_f(got_b[(run, s, i)]) for i in range(GEN)]
    free(sc)
    del sc
    finite = all(math.isfinite(x) for x in list(comp_a.values()) + list(got_b.values()))
    mstep = max(diffs)
    passed = bool(finite and mstep <= VAL_TOL_STEP and pdiff <= VAL_TOL_PATH)
    return {"model": VALIDATION["judge"], "revision": VALIDATION["revision"], "dtype": dtype, "device": device,
            "runs": vruns, "steps": list(steps), "n_completions": len(vrows), "per_step": per_step,
            "max_abs_diff": _f(mstep), "mean_abs_diff": _f(sum(diffs) / len(diffs)),
            "batched_path_max_abs_diff": _f(pdiff), "batched_path_batches": st["batches"],
            "tolerance_step": VAL_TOL_STEP, "tolerance_path": VAL_TOL_PATH, "all_finite": finite, "passed": passed,
            "judge_facts": facts, "hub_cache": cache, "seconds": time.time() - t}


def env_info(device):
    import importlib.metadata as md
    import re as _re
    import numpy
    import torch
    import transformers
    pk = {}
    for d in md.distributions():
        name = d.metadata["Name"]
        if name:
            pk[_re.sub(r"[-_.]+", "-", name).lower()] = d.version
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
            "numpy": numpy.__version__, "accelerate": pk.get("accelerate"), "cuda": torch.version.cuda, "gpu": gpu,
            "device": device, "packages": dict(sorted(pk.items()))}


def main(device=None, val_steps=VAL_STEPS, data_path=DATA_PATH, result_path=RESULT_PATH, partial_path=PARTIAL_PATH,
         partial_every_s=PARTIAL_EVERY_S):
    """Production: main() with the defaults (the keyword arguments exist for the local tests)."""
    import torch
    device = device or DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        _stop("no CUDA device")
    timing = {"start_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    progress("start", 0)
    t = time.time()
    header, rows = load_data(data_path)
    timing["data_s"] = time.time() - t
    say("data: %d completions of %d runs, SHA-256 matches the baked value" % (len(rows), len(header["runs"])))
    out = {"judge": TARGET["judge"], "revision": TARGET["revision"], "dtype": TARGET["dtype"],
           "data_sha256": DATA_SHA256, "validation": None, "scores": {}, "n_scored": 0, "n_total": N_ROWS,
           "order": None, "timing": timing, "complete": False,
           "runs": {rid: {"arm": v["arm"], "task": v["task"], "seed": v["seed"]} for rid, v in header["runs"].items()},
           "env": env_info(device), "judge_facts": None, "snapshots": None,
           "run_id": RUN_ID, "stage": "j3", "job_file_sha256": JOB_FILE_SHA256,
           "provenance": {"git_commit": GIT_COMMIT, "package_sha256": PACKAGE_SHA256,
                          "template_sha256": TEMPLATE_SHA256, "lock_sha256": LOCK_SHA256,
                          "job_file_sha256": JOB_FILE_SHA256, "module_sha256": MODULE_SHA256,
                          "judge_template_sha256": E1_TEMPLATE_SHA256},
           "used": header.get("used"), "data_manifest": header.get("data_manifest"),
           "mode": "production" if (device == "cuda" and TARGET == REGISTERED_TARGET) else "test"}

    def publish(final=False):
        write_json(out, partial_path)
        if final:
            write_json(out, result_path)

    progress("data", 0)
    t = time.time()
    try:
        val = out["validation"] = run_validation(rows, header, device, VALIDATION["dtype"], tuple(val_steps))
    except BaseException as e:  # noqa: BLE001  includes SystemExit from the SIGTERM handler
        out["aborted"] = "validation stage stopped early: %s" % repr(e)[:300]
        timing["total_s"] = time.time() - T0
        publish(final=True)
        progress("aborted", 0)
        say("ABORT in the validation stage (%s); result.json says so, nothing is scored" % type(e).__name__)
        raise
    timing["validation_s"] = time.time() - t
    if not val["passed"]:
        out["aborted"] = "validation gate failed"
        timing["total_s"] = time.time() - T0
        publish(final=True)
        progress("validation_failed", 0)
        say("STOP: the validation gate failed (values in result.json); the target judge was not loaded")
        return 2
    publish()
    progress("validated", 0)

    sink = Scored()
    out["scores"] = sink.scores
    state = {"budget": None, "oom_halvings": 0, "batches": 0, "tokens": 0, "padded_tokens": 0}
    clock = {"status": time.time(), "partial": time.time(), "t_score": time.time()}
    try:
        t = time.time()
        sc = JudgeScorer(TARGET["judge"], device=device, dtype=TARGET["dtype"], revision=TARGET["revision"])
        out["judge_facts"] = judge_facts(sc)
        if out["judge_facts"]["param_dtype"] != TARGET["dtype"]:
            raise RuntimeError("the judge loaded in %s, not %s" % (out["judge_facts"]["param_dtype"], TARGET["dtype"]))
        out["snapshots"] = {"target": hub_cache(TARGET["judge"]), "validation": val["hub_cache"]}
        check_snapshot("target judge", TARGET, out["snapshots"]["target"])     # before any row is scored
        timing["target_load_s"] = time.time() - t
        progress("target_loaded", 0)
        t = time.time()
        lens = token_lengths(sc, rows)
        timing["tokenize_s"] = time.time() - t
        out["judge_facts"]["n_prompts_over_max_length"] = sum(n > sc.max_length for n in lens)
        out["judge_facts"]["prompt_tokens_mean"] = sum(lens) / max(len(lens), 1)
        out["judge_facts"]["prompt_tokens_max"] = max(lens) if lens else 0
        state["budget"] = timing["token_budget_initial"] = initial_budget(sc, device)
        plan = [(name, rule, [(r, n) for r, n in zip(rows, lens) if pred(r["step"])]) for name, rule, pred in CHUNKS]
        if sum(len(sel) for _nm, _rule, sel in plan) != len(rows):
            raise RuntimeError("the chunks do not cover every row exactly once")
        out["order"] = {"chunks": [{"name": nm, "rule": rule, "n": len(sel)} for nm, rule, sel in plan],
                        "chunks_done": [],
                        "within_chunk": "grouped by task; sorted by prompt token length, longest first; "
                                        "B x longest <= token budget, B <= %d; OOM halves the budget" % MAX_BS}
        say("tokenized %d prompts in %d s; %d over %d tokens (truncated, as in training)"
            % (len(lens), int(timing["tokenize_s"]), out["judge_facts"]["n_prompts_over_max_length"], sc.max_length))
        clock.update(status=time.time(), partial=time.time(), t_score=time.time())

        def tick():
            now = time.time()
            if now - clock["status"] >= STATUS_EVERY_S:
                clock["status"] = now
                progress("scoring", sink.n)
            if now - clock["partial"] >= partial_every_s:
                clock["partial"] = now
                out["n_scored"] = sink.n
                timing["scoring_s"] = now - clock["t_score"]
                publish()

        for name, _rule, sel in plan:
            tc = time.time()
            score_rows(sc, [r for r, _n in sel], [n for _r, n in sel], state, sink.put, tick)
            timing["chunk_%s_s" % name] = time.time() - tc
            out["order"]["chunks_done"].append(name)
            out["n_scored"] = sink.n
            clock["partial"] = time.time()
            publish()
            progress("chunk_" + name, sink.n)
            say("chunk %s: %d rows in %d s, %d batches so far, %d OOM halvings"
                % (name, len(sel), int(time.time() - tc), state["batches"], state["oom_halvings"]))
    except BaseException as e:  # noqa: BLE001  includes SystemExit from the SIGTERM handler
        out["aborted"] = "target stage stopped early: %s" % repr(e)[:300]
        out["n_scored"], out["n_nonfinite"] = sink.n, sink.nonfinite
        timing["scoring_s"] = time.time() - clock["t_score"]
        timing["total_s"] = time.time() - T0
        publish(final=True)
        progress("aborted", sink.n)
        say("ABORT in the target stage after %d scores (%s); result.json holds what is scored" % (sink.n,
                                                                                                  type(e).__name__))
        raise
    timing["scoring_s"] = time.time() - clock["t_score"]
    timing.update({"batches": state["batches"], "oom_halvings": state["oom_halvings"],
                   "token_budget_final": state["budget"], "tokens": state["tokens"],
                   "padded_tokens": state["padded_tokens"],
                   "completions_per_s": sink.n / max(timing["scoring_s"], 1e-9),
                   "tokens_per_s": state["tokens"] / max(timing["scoring_s"], 1e-9)})
    out["n_scored"], out["n_nonfinite"] = sink.n, sink.nonfinite
    missing = sum(x is None for st in sink.scores.values() for v in st.values() for x in v)
    out["complete"] = bool(sink.n == len(rows) and missing == 0 and sink.nonfinite == 0 and
                           sum(len(st) for st in sink.scores.values()) * GEN == len(rows))
    timing["total_s"] = time.time() - T0
    publish(final=True)
    progress("done", sink.n)
    say("done: complete %s, %d of %d scored, %d s" % (out["complete"], sink.n, len(rows), int(timing["total_s"])))
    return 0 if out["complete"] else 3


def _on_sigterm(signum, frame):
    """The instance's timeout sends SIGTERM (SIGKILL 120 s later): turn it into SystemExit so main() writes what is
    scored; ignore any further SIGTERM while that runs."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise SystemExit(143)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _on_sigterm)
    sys.exit(main())
'''

_PLACEHOLDERS = ("__PKG_B64__", "__RUN_ID__", "__GIT_COMMIT__", "__PACKAGE_SHA256__", "__TEMPLATE_SHA256__",
                 "__LOCK_SHA256__", "__DATA_SHA256__", "__N_ROWS__")


# ============================================================================================
# host side: the data file
# ============================================================================================
def _experiments():
    p = str(ROOT / "experiments")
    if p not in sys.path:
        sys.path.insert(0, p)
    import e1b_primary as E  # noqa: E402
    return E


def build_data(out_path, main_dir=None, **load_kw):
    """Write the J3 data file for the used runs of experiments/e1b_primary.load_e1b and return {path, sha256,
    n_rows, n_runs, header}. main_dir and load_kw (load_e1b's other keyword arguments: launch_records, registered,
    prereg, ...) default to load_e1b's own defaults, the registered paths; the tests pass synthetic ones. Refuses
    (SystemExit) a data-manifest failure, as the registered analysis does before it selects. Prints nothing."""
    E = _experiments()
    md = E.MAIN_B if main_dir is None else Path(main_dir)
    man = E.data_manifest(md)
    why = E._manifest_failure(man)
    if why:
        raise SystemExit("REFUSE: data manifest (spec 7.8): %s" % why)
    sel = E.load_e1b(md, **dict(load_kw, manifest=man))
    used = sel["used"]["hacking"] + sel["used"]["twin"]
    runs, rows = {}, []
    for r in used:
        rid = E._run_id(r)
        if rid in runs:
            raise SystemExit("REFUSE: two used runs share the run id %s" % rid)
        if r.get("arm") not in J.ARMS or r.get("task") not in J.TASKS:
            raise SystemExit("REFUSE: used run %s has arm %r, task %r" % (rid, r.get("arm"), r.get("task")))
        comps = r.get("completions") or []
        if len(comps) != STEPS:
            raise SystemExit("REFUSE: used run %s stores %d steps of completions, not %d" % (rid, len(comps), STEPS))
        runs[rid] = {"arm": r["arm"], "task": r["task"], "seed": int(r["seed"]), "tag": E._tag(r)}
        for s, c in enumerate(comps):
            texts, js = c.get("texts"), c.get("judge_small")
            if int(c.get("step", -1)) != s or not isinstance(texts, list) or len(texts) != GEN or \
                    not isinstance(js, list) or len(js) != GEN:
                raise SystemExit("REFUSE: used run %s step %d does not store %d texts and judge_small" % (rid, s, GEN))
            for i in range(GEN):
                if not isinstance(texts[i], str) or isinstance(js[i], bool) or not isinstance(js[i], (int, float)):
                    raise SystemExit("REFUSE: used run %s step %d completion %d is not a text with its judge_small"
                                     % (rid, s, i))
                rows.append({"run": rid, "arm": r["arm"], "task": r["task"], "seed": int(r["seed"]), "step": s,
                             "i": i, "text": texts[i], "judge_small": float(js[i])})
    rows.sort(key=lambda x: (x["run"], x["step"], x["i"]))
    header = {"format": DATA_FORMAT, "n_rows": len(rows), "n_runs": len(runs), "steps": STEPS, "generations": GEN,
              "runs": dict(sorted(runs.items())),
              "used": {a: [E._tag(r) for r in v] for a, v in sel["used"].items()},
              "data_manifest": {"sha256": man["sha256"], "n_files": man["n_files"]},
              "selection": {k: len(sel.get(k) or ()) for k in ("excluded", "unregistered", "surplus",
                                                               "replacements_needed", "replacements_blocked",
                                                               "main_not_launched", "main_to_launch")}}
    if "main_to_launch" not in sel:         # before launching can stop at an achieved n, every unlaunched run is due
        header["selection"]["main_to_launch"] = header["selection"]["main_not_launched"]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0,
                                                    compresslevel=9) as gz:
        gz.write((json.dumps({"header": header}, sort_keys=True) + "\n").encode("ascii"))
        for row in rows:
            gz.write((json.dumps(row, sort_keys=True) + "\n").encode("ascii"))
    return {"path": str(out_path), "sha256": _sha256_file(out_path), "n_rows": len(rows), "n_runs": len(runs),
            "header": header}


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# The child process of prep_data: the pins are verified before anything but the analysis's own modules is imported
# (verify_pins_b counts every repo module in sys.modules), then this file builds the data.
_PREP_CHILD = r'''
import json, sys
sys.dont_write_bytecode = True
root, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, root + "/experiments")
import e1b_primary as E
E.verify_pins_b()
sys.path.insert(0, root + "/scripts")
import e1b_j3_job as JJ
info = JJ.build_data(out)
info["header"] = {k: v for k, v in info["header"].items() if k != "runs"}
info["header"]["run_ids"] = sorted(JJ.read_header(out)["runs"])
sys.stdout.write("E1B_J3_DATA " + json.dumps(info, sort_keys=True) + "\n")
'''


def prep_data(out_dir):
    """Build <out_dir>/data.jsonl.gz from the registered paths in a child process that verifies the pins first
    (spec 7.6). Returns build_data's dict (the header without the per-run table, plus run_ids). Raises SystemExit
    with the child's last lines when the pins, the data manifest or the selection refuse."""
    out = Path(out_dir) / DATA_NAME
    r = subprocess.run([sys.executable, "-c", _PREP_CHILD, str(ROOT), str(out)], capture_output=True, text=True,
                       cwd=str(ROOT))
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("E1B_J3_DATA ")), None)
    if r.returncode != 0 or line is None:
        tail = (r.stderr.strip().splitlines() or r.stdout.strip().splitlines() or ["rc %d" % r.returncode])[-3:]
        raise SystemExit("data build refused: %s" % " | ".join(tail))
    return json.loads(line[len("E1B_J3_DATA "):])


def read_header(path):
    with gzip.open(path, "rt", encoding="ascii") as f:
        return json.loads(f.readline())["header"]


def check_data(path):
    """The builder's shape, read back: (sha256, header). Header first, then rows sorted and unique by (run, step, i),
    every key present, every run of the header with STEPS x GEN rows of its own task, n_rows as the header says."""
    sys.path.insert(0, str(ROOT))
    from flightrecorder.repro.reward_codeset import CODE_TASKS
    tasks = {t.name for t in CODE_TASKS}
    sha = _sha256_file(path)
    with gzip.open(path, "rt", encoding="ascii") as f:
        first = json.loads(f.readline())
        if not isinstance(first, dict) or "header" not in first or first["header"].get("format") != DATA_FORMAT:
            raise ValueError("%s: no %s header line" % (path, DATA_FORMAT))
        header = first["header"]
        last, n, per = None, 0, {}
        for line in f:
            d = json.loads(line)
            n += 1
            if tuple(sorted(d)) != tuple(sorted(ROW_KEYS)):
                raise ValueError("row %d keys %s" % (n, sorted(d)))
            key = (d["run"], d["step"], d["i"])
            if last is not None and key <= last:
                raise ValueError("row %d out of (run, step, i) order or duplicated" % n)
            last = key
            hr = header["runs"].get(d["run"])
            if hr is None or d["task"] not in tasks or hr["task"] != d["task"] or hr["arm"] != d["arm"]:
                raise ValueError("row %d does not match the header's run table" % n)
            per[d["run"]] = per.get(d["run"], 0) + 1
    if n != header["n_rows"] or n == 0:
        raise ValueError("%d rows; the header says %s" % (n, header["n_rows"]))
    if set(per) != set(header["runs"]) or any(v != STEPS * GEN for v in per.values()):
        raise ValueError("not %d x %d rows for every header run" % (STEPS, GEN))
    return sha, header


# ============================================================================================
# bake
# ============================================================================================
def header():
    """The PEP 723 block of the job (identical to scripts/e1b_job.py's)."""
    lines = UV_J3.split("\n")
    return "\n".join(lines[:lines.index("# ///", 1) + 1]) + "\n"


def _template_of(src):
    """_JUDGE_TEMPLATE's value in reward_judge.py source, read with ast (no import)."""
    for n in ast.parse(src).body:
        if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "_JUDGE_TEMPLATE" for t in n.targets):
            return ast.literal_eval(n.value)
    return None


def bake(commit, data_path, run_id, out_dir):
    """Write <out_dir>/<run_id>/job.py, job.py.lock and data.jsonl.gz from commit. Returns (job_path, provenance)."""
    if not isinstance(run_id, str) or not re.fullmatch(RUN_ID_RE, run_id):
        raise ValueError("run id %r is not a J3 run id (%s)" % (run_id, RUN_ID_RE))
    sha = J._git("rev-parse", "--verify", "%s^{commit}" % commit).decode().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise RuntimeError("commit %r did not resolve to a full sha: %r" % (commit, sha))
    template = J._blob(sha, TEMPLATE_PATH)
    if Path(__file__).read_bytes().replace(b"\r\n", b"\n") != template:
        raise RuntimeError("%s differs from its committed version at %s; bake only from the committed template"
                           % (TEMPLATE_PATH, sha[:12]))
    if Path(J.__file__).read_bytes().replace(b"\r\n", b"\n") != J._blob(sha, J.TEMPLATE_PATH):
        raise RuntimeError("%s differs from its committed version at %s (its header and package helpers are used)"
                           % (J.TEMPLATE_PATH, sha[:12]))
    if header() != J.header():
        raise RuntimeError("the J3 job's PEP 723 header differs from %s's, so %s would not be its lock"
                           % (J.TEMPLATE_PATH, LOCK_PATH))
    lock = J._blob(sha, LOCK_PATH)
    tar = J.package_tar(sha)
    with tarfile.open(fileobj=io.BytesIO(tar), mode="r:") as t:
        names = set(t.getnames())
        missing = [p for p in REQUIRED if p not in names]
        if missing:
            raise RuntimeError("the package at %s lacks %s" % (sha[:12], ", ".join(missing)))
        for rel, want in E1_MODULE_SHA256.items():
            if hashlib.sha256(t.extractfile(rel).read().replace(b"\r\n", b"\n")).hexdigest() != want:
                raise RuntimeError("%s at %s differs from the E1 rescoring's version" % (rel, sha[:12]))
        tmpl = _template_of(t.extractfile("flightrecorder/repro/reward_judge.py").read().decode("utf-8"))
        if tmpl is None or hashlib.sha256(tmpl.encode("utf-8")).hexdigest() != E1_TEMPLATE_SHA256:
            raise RuntimeError("the judge template at %s differs from the E1 rescoring's" % sha[:12])
    data_sha, head = check_data(data_path)
    prov = {"run_id": run_id, "git_commit": sha, "package_sha256": hashlib.sha256(tar).hexdigest(),
            "template_sha256": hashlib.sha256(template).hexdigest(), "lock_sha256": hashlib.sha256(lock).hexdigest(),
            "data_sha256": data_sha, "n_completions": head["n_rows"], "n_runs": head["n_runs"]}
    b64 = base64.b64encode(gzip.compress(tar, compresslevel=9, mtime=0)).decode("ascii")
    subs = {"__PKG_B64__": b64, "__RUN_ID__": run_id, "__GIT_COMMIT__": sha,
            "__PACKAGE_SHA256__": prov["package_sha256"], "__TEMPLATE_SHA256__": prov["template_sha256"],
            "__LOCK_SHA256__": prov["lock_sha256"], "__DATA_SHA256__": data_sha, "__N_ROWS__": str(head["n_rows"])}
    assert set(subs) == set(_PLACEHOLDERS)
    for k in _PLACEHOLDERS:
        if UV_J3.count(k) != 1:
            raise RuntimeError("placeholder %s appears %d times in the template" % (k, UV_J3.count(k)))
    src = UV_J3
    for k in _PLACEHOLDERS:
        src = src.replace(k, subs[k])
    left = re.findall(r"__[A-Z][A-Z0-9_]*__", src.replace(b64, ""))
    if left:
        raise RuntimeError("unsubstituted placeholders: " + ", ".join(sorted(set(left))))
    job_dir = Path(out_dir) / run_id
    job_dir.mkdir(parents=True, exist_ok=True)
    job_path = job_dir / "job.py"
    body = src.encode("ascii")                           # LF: the template string has LF line endings
    job_path.write_bytes(body)
    (job_dir / "job.py.lock").write_bytes(lock)
    if Path(data_path).resolve() != (job_dir / DATA_NAME).resolve():
        shutil.copyfile(data_path, job_dir / DATA_NAME)
    with tempfile.TemporaryDirectory() as d:             # fail here, not on the instance
        py_compile.compile(str(job_path), cfile=str(Path(d) / "job.pyc"), doraise=True)
    prov["job_sha256"] = hashlib.sha256(body).hexdigest()
    return job_path, prov


def main(argv):
    if argv[:1] == ["--prep"] and len(argv) == 2:
        info = prep_data(argv[1])
        print("wrote %s: %d completions of %d used runs, sha256 %s (no value printed)"
              % (info["path"], info["n_rows"], info["n_runs"], info["sha256"]))
        return 0
    if argv[:1] == ["--bake"] and len(argv) == 5:
        commit, data, rid, out_dir = argv[1:]
        job_path, prov = bake(commit, data, rid, out_dir)
        print(json.dumps(dict(prov, job_path=str(job_path)), indent=1))
        return 0
    print(__doc__.strip().splitlines()[0])
    print("usage: python scripts/e1b_j3_job.py --prep OUT_DIR | --bake COMMIT DATA RUN_ID OUT_DIR")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
