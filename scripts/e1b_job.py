"""E1b JOB TEMPLATE AND bake(). Builds job files only: nothing here launches, uploads or spends. ASCII only.

WHAT THE JOB IS
  E1's job, scripts/launch_slow_onset.py arm "judge" (that file stays frozen; its UV string is copied here,
  not imported), with only the changes experiments/e1b_preregistration.md sections 3 and 8 and
  experiments/e1b_analysis_spec.md sections 0.4, 5 and 7.1-7.3 require. The interfaces are fixed by
  experiments/e1b_design/build_contract.md (sections 2-6).
    * Arms: judge (hacking, reward J05) and judgegate (twin, reward J05 x twin_gate.gate). The other E1 arms
      are gone. Both judges score every completion in both arms, and g is computed and logged in both arms.
    * Logging: every per-step and per-completion field of build_contract section 4, with full texts (E1 cut
      them at 1,500 characters), the provenance fields, and the version of every installed distribution,
      python, torch's CUDA version, GPU name and driver.
    * Monitoring (pre-registration section 8): stdout carries exactly one "STEP <n>" line per step and nothing
      else. The ONSET, SAMPLE and PROBE prints, the GO/NO-GO summary and the RESULT_B64 dump are removed, and
      so is the trainer's own log printing and every progress bar. status.json holds {run_id, step, rc, ts}.
    * Provenance (spec 7.1-7.3): the package is `git archive` of flightrecorder/ at a named commit, never the
      working tree; the model revisions are passed to every from_pretrained; packages come from an exact lock.
      Two result fields are measured by the job, not baked into it (build_contract section 8): job_file_sha256,
      the SHA-256 of the job file's own bytes as it runs, which rule (e) compares with the launch record's
      job_sha256 (the SHA-256 of the file bake() wrote); and snapshots, the Hub snapshot commit each of the three
      models was loaded from, read from the Hub cache after loading. The job stops before training unless each
      snapshot is its pinned revision.
  The training configuration is E1's and is written into the template, not substituted: Qwen/Qwen2.5-1.5B,
  lr 1e-6, beta 0.02, 8 generations, 400 steps, max_completion_length 256, probe 16 every 5.

MODEL REVISIONS: WHAT THE LOCKED TRL DOES WITH A STRING MODEL ID (trl 1.14.2, trl/trainer/grpo_trainer.py)
  E1 passed the policy as a string and no model_init_kwargs. GRPOTrainer.__init__ then loads, for a string:
    * the policy with trl.trainer.utils.create_model_from_path(model_id, **args.model_init_kwargs), with
      trust_remote_code defaulting to False; create_model_from_path defaults dtype to float32 and device_map to
      "auto" on a GPU (None on a CPU);
    * the processing class with AutoProcessor.from_pretrained(model id, revision=model_init_kwargs.get(
      "revision"), truncation_side="left", padding_side="left", trust_remote_code=False);
    * the reference model (beta 0.02 > 0, no PEFT) with create_model_from_path(model.config._name_or_path,
      **model_init_kwargs).
  With architecture=None, create_model_from_path first calls AutoConfig.from_pretrained(model_id) WITHOUT the
  revision, only to pick the architecture class: a read at the default branch, for the policy and again for the
  reference model, which pre-registration section 13 ("revision= to every from_pretrained") does not allow.
  create_model_from_path(model_id, architecture=None, **kwargs) takes the class as a keyword, so the job passes
  GRPOConfig(model_init_kwargs={"revision": <policy sha>, "architecture": AutoModelForCausalLM}). TRL then
  loads the policy and the reference model with AutoModelForCausalLM.from_pretrained(model_id, revision=...),
  which picks the class from the PINNED config (Qwen2ForCausalLM for Qwen/Qwen2.5-1.5B at 8faed761, whose main
  branch is that commit, so it is the class E1's default-branch lookup picked), and the processing class with
  the same revision. On trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 this loads the same class with identical
  weights, config and generation config as the bare revision, and no from_pretrained call is left without the
  revision (experiments/e1b_design/job_smoke). transformers 5.19.0 passes the revision on to the generation
  config. A class in model_init_kwargs makes args.to_json_string() raise (json cannot encode a type); nothing on
  the job's path calls it (report_to=[], save_strategy "no"). The job still stops before training unless each
  loaded class is the one its pinned config names (config.architectures[0]). Both judges get revision= through
  JudgeScorer, which passes it to its AutoTokenizer and AutoModelForCausalLM calls. One metadata read stays at
  the default branch and is not a from_pretrained call: transformers' tokenizer loader asks
  hf_api().model_info(model_id) for the repo's tags (tokenization_utils_tokenizers.py, Mistral regex check; it
  takes no revision) and changes nothing unless a tag names mistralai as the base model.

TRUNCATION FLAG (eos)
  trl 1.14.2 computes completions/clipped_ratio in GRPOTrainer._generate as `ids[-1] not in [*eos_token_ids,
  tokenizer.pad_token_id]` over the same completion_ids list it then passes to the reward functions
  (_generate_and_score_completions -> _calculate_rewards). eos_token_ids is the tokenizer's eos id plus any
  eos id the generation config declares. The job logs eos = not twin_gate.truncated(ids, tokenizer eos,
  tokenizer pad) and stops before training unless TRL's eos_token_ids is exactly [tokenizer eos], so the logged
  flag is TRL's own rule. For Qwen/Qwen2.5-1.5B at the pinned revision both are 151643 (<|endoftext|>), which
  is also the pad token.

WHAT PRINTS (transformers 5.19.0, trl 1.14.2)
  Trainer.__init__ adds PrinterCallback when disable_tqdm is set and ProgressCallback otherwise
  (transformers/trainer.py). ProgressCallback draws the step bar and writes every log dict through
  tqdm.write; PrinterCallback prints every log dict. report_to=[] adds no integration callback. TRL adds a
  callback only for sync_ref_model, and prints completion tables only with log_completions=True (default
  False). The job removes both printing callbacks after the trainer is built, so Trainer.log still reaches
  GeomCallback.on_log through the callback handler and nothing prints the logs. The "Loading weights" and Hub
  bars are turned off with transformers.utils.logging.disable_progress_bar, HF_HUB_DISABLE_PROGRESS_BARS and
  TQDM_DISABLE. Inside the job sys.stdout is pointed at stderr, so any other print lands on stderr; only the
  STEP lines are written to the real stdout.

THE LOCK (experiments/e1b_uv.lock)
  A PEP 723 script lock, lock format version 1 revision 5, written by uv 0.12.23 (`uv lock --script`) from the
  header of UV below: requires-python ==3.12.*, exact == pins of the six direct dependencies, torch from the
  cu128 explicit index as in E1's header, and exclude-newer 2026-10-08T02:35:00Z, the cutoff of
  experiments/e1_env_lock_reconstructed.txt. Every distribution is pinned to one version with its sha256
  hashes. On linux x86_64 it installs the same 72 versions as that file (the lock also lists colorama and
  tzdata, which are marked win32/emscripten only). Rebuild it with
      python scripts/e1b_job.py --write-header <tmp>/stub.py
      uv lock --script <tmp>/stub.py          # uv 0.12.23; writes <tmp>/stub.py.lock
  and copy stub.py.lock to experiments/e1b_uv.lock. uv checks a script lock against the script's inline
  metadata only (the header), never its body, so the lock made from the stub is the lock of every baked job.
  bake() writes it next to the job as job.py.lock, the name uv looks for.

RUNNING ON THE INSTANCE (job.py and job.py.lock in one directory; scripts/launch_e1b_aws.py runs these)
      echo "uv==0.12.23 --hash=sha256:565c6e2874dbeae86c02f3dea97255e878fec672659a73d4930c6b93fcab2fff" > uv-req.txt
      python3 -m pip install --only-binary :all: --require-hashes -r uv-req.txt
      python3 -m uv run --no-config --locked --python 3.12.15 --managed-python job.py
  pip installs uv only from a wheel whose SHA-256 is UV_WHEEL_SHA256 (PyPI's hash of the manylinux x86_64
  wheel). --locked makes uv exit with an error, before installing anything, unless job.py.lock is up to date
  with the job's header; it then installs the locked versions and checks their hashes, without resolving.
  --no-config ignores any uv.toml on the machine (the job's own [tool.uv] table still applies). 3.12.15 is the
  managed CPython uv 0.12.23 downloads for "3.12", the one E1's job logs name. Environment: E1B_STATUS_PATH
  (default /tmp/status.json). The result goes to /tmp/result.json, as in E1. The job first hashes its own file
  (job_file_sha256). Before any model loads, it stops (exit 1, reason on stderr) if the embedded package's
  SHA-256, the extracted twin_gate.py's SHA-256 or job.py.lock's SHA-256 differs from the baked value. It
  writes status.json once with step null after those checks, then once per step. After every model has loaded
  and before training, it stops if the snapshot directory the Hub cache holds for a model's repo is not exactly
  that model's pinned revision.

BAKE
  bake(arm, task, seed, stage, commit, out_dir) -> (job_path, provenance), build_contract section 6; provenance
  holds run_id, git_commit, package_sha256, template_sha256, lock_sha256, gate_sha256 and job_sha256, the
  SHA-256 of the job.py bytes bake() wrote (the launcher stores it in the launch record). It reads
  the template, the lock and the gate from the commit's blobs (refusing a blob with CRLF line endings, so
  every SHA-256 is of LF text), refuses unless the working-tree scripts/e1b_job.py equals the committed one,
  archives flightrecorder/ at the commit with git archive (core.autocrlf=false, core.eol=lf, tar.umask=0002,
  so the bytes do not depend on the machine's git settings), substitutes and checks every placeholder and
  py_compiles the job. It refuses an arm, task, seed or stage outside the registered set.

    python scripts/e1b_job.py --bake ARM TASK SEED STAGE COMMIT OUT_DIR   # bake one job, print provenance
    python scripts/e1b_job.py --write-header PATH                         # the PEP 723 header, for the lock
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import py_compile
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ARMS = ("judge", "judgegate")
TASKS = ("count_vowels", "is_prime", "second_largest")
# Seeds by stage (pre-registration section 3; build_contract section 2). Replacements take the next unused seed
# from 316 for their arm and task. Launches stop at 60 per arm (48 + 12, section 8), so one arm and task can
# never use more than 12 replacement seeds: 316-327.
STAGE_SEEDS = {"pilot": (299, 298, 297), "main": tuple(range(300, 316)), "main_replace": tuple(range(316, 328))}
TEMPLATE_PATH = "scripts/e1b_job.py"
LOCK_PATH = "experiments/e1b_uv.lock"
GATE_PATH = "flightrecorder/repro/twin_gate.py"
PACKAGE_DIR = "flightrecorder"
UV_VERSION = "0.12.23"        # the uv that wrote the lock; the instance installs exactly this one
# PyPI's sha256 of uv-0.12.23-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (uploaded 2026-10-03)
UV_WHEEL_SHA256 = "565c6e2874dbeae86c02f3dea97255e878fec672659a73d4930c6b93fcab2fff"
PYTHON_VERSION = "3.12.15"    # the managed CPython uv 0.12.23 picks for 3.12, as in E1's job logs
RUN_ARGS = ["uv", "run", "--no-config", "--locked", "--python", PYTHON_VERSION, "--managed-python", "job.py"]
_REQUIRED = ("flightrecorder/__init__.py", "flightrecorder/repro/__init__.py", GATE_PATH,
             "flightrecorder/repro/reward_judge.py", "flightrecorder/repro/reward_testhack.py",
             "flightrecorder/repro/reward_codeset.py", "flightrecorder/repro/oracle_lenient.py")


# ============================================================================================
# THE BAKED JOB. E1's template (launch_slow_onset.UV, arm "judge") with the E1b changes listed above.
# ============================================================================================
UV = r'''# /// script
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
# E1b job, baked by scripts/e1b_job.py from a named commit. Run with uv 0.12.23, job.py.lock next to this file:
#     uv run --no-config --locked --python 3.12.15 --managed-python job.py
# stdout carries only "STEP <n>" lines. Everything else goes to stderr.
import base64, gzip, hashlib, io, json, os, sys, tarfile, tempfile  # noqa: E401
from datetime import datetime, timezone

_STDOUT = sys.stdout
sys.stdout = sys.stderr     # from here on any print, the libraries' included, goes to stderr

ARM, TASK, SEED, STAGE, RUN_ID = "__ARM__", "__TASK__", __SEED__, "__STAGE__", "__RUN_ID__"
GIT_COMMIT, PACKAGE_SHA256 = "__GIT_COMMIT__", "__PACKAGE_SHA256__"
TEMPLATE_SHA256, LOCK_SHA256, GATE_SHA256 = "__TEMPLATE_SHA256__", "__LOCK_SHA256__", "__GATE_SHA256__"
STATUS_PATH = os.environ.get("E1B_STATUS_PATH", "/tmp/status.json")
RESULT_PATH = "/tmp/result.json"    # as E1


def _stop(why):
    sys.exit("E1B STOP before training: " + why)


def _status(step):
    """status.json = {run_id, step, rc, ts}, rewritten atomically; rc stays null (the instance writes the final rc)."""
    tmp = STATUS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"run_id": RUN_ID, "step": step, "rc": None,
                   "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}, f)
    os.replace(tmp, STATUS_PATH)


# Measured, not baked: the SHA-256 of this job file's own bytes. Rule (e) compares it with the launch record's
# job_sha256, the SHA-256 of the file bake() wrote, so an edit made after bake shows in the result.
with open(os.path.abspath(__file__), "rb") as _f:
    JOB_FILE_SHA256 = hashlib.sha256(_f.read()).hexdigest()

# Provenance, checked before the package is imported or any model loads: the embedded package, the lock uv
# installed from (job.py.lock next to this file) and the gate module must all be the baked ones.
_TAR = gzip.decompress(base64.b64decode("__PKG_B64__"))
if hashlib.sha256(_TAR).hexdigest() != PACKAGE_SHA256:
    _stop("the embedded package's SHA-256 differs from package_sha256")
_LOCK = os.path.abspath(__file__) + ".lock"
if not os.path.isfile(_LOCK):
    _stop("no job.py.lock next to the job")
with open(_LOCK, "rb") as _f:
    if hashlib.sha256(_f.read()).hexdigest() != LOCK_SHA256:
        _stop("job.py.lock's SHA-256 differs from lock_sha256")
_FR = tempfile.mkdtemp(prefix="e1b_fr_")    # a fresh directory, so only the package's own files can import
tarfile.open(fileobj=io.BytesIO(_TAR), mode="r:").extractall(_FR)
with open(os.path.join(_FR, "flightrecorder", "repro", "twin_gate.py"), "rb") as _f:
    if hashlib.sha256(_f.read()).hexdigest() != GATE_SHA256:
        _stop("the extracted twin_gate.py's SHA-256 differs from gate_sha256")
sys.path.insert(0, _FR)
_status(None)

# Progress output off before the libraries load (it changes no computation): tqdm 4.70 reads TQDM_* variables.
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import ast as _ast  # noqa: E402
import importlib.metadata as _md  # noqa: E402
import platform  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from trl import GRPOConfig as _Config, GRPOTrainer as _Trainer  # noqa: E402
from transformers import AutoModelForCausalLM, PrinterCallback, ProgressCallback, TrainerCallback  # noqa: E402
from transformers.utils import logging as _hf_logging  # noqa: E402
from huggingface_hub import constants as _hub_constants  # noqa: E402
import flightrecorder  # noqa: E402
from flightrecorder.repro.reward_codeset import CODE_TASKS  # noqa: E402
from flightrecorder.repro.reward_judge import JudgeScorer  # noqa: E402
from flightrecorder.repro.reward_testhack import oracle_reward, extract_code, _FENCE  # noqa: E402
from flightrecorder.repro.oracle_lenient import lenient_oracle_reward  # noqa: E402  label only, never a reward
from flightrecorder.repro.twin_gate import gate, truncated  # noqa: E402

_hf_logging.disable_progress_bar()     # transformers' "Loading weights" bars, and the Hub's
if not os.path.abspath(flightrecorder.__file__).startswith(os.path.abspath(_FR) + os.sep):
    _stop("flightrecorder was imported from outside the embedded package")

POLICY, JUDGE = "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-0.5B-Instruct"   # JUDGE: the judge both arms optimize
BIG, SMALL = "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"
REVISIONS = {"policy": "8faed761d45a263340a0528343f099c05c9a4323",
             "judge_small": "7ae557604adf67be50417f59c2c2f167def9a775",
             "judge_big": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"}
JUDGE_DEVICE, JUDGE_DTYPE = "cuda", "float16"    # JudgeScorer's defaults, which E1 used
STEPS, GEN, LR, BETA, MAXTOK = 400, 8, 1e-6, 0.02, 256
PROBE_N, PROBE_EVERY = 16, 5
assert ARM in ("judge", "judgegate"), "unknown arm " + ARM
TASKS = [t for t in CODE_TASKS if t.name == TASK]
assert TASKS, "unknown task " + TASK
FN = TASK     # the function the task asks for, which the gate looks for at the module's top level
assert ("function %s(" % FN) in TASKS[0].prompt, "the task prompt does not name " + FN
BY_PROMPT = {t.prompt: t for t in TASKS}
PROMPTS = [t.prompt for t in TASKS]

random.seed(SEED)
need = STEPS + 16
ds_prompts = (PROMPTS * ((need // len(PROMPTS)) + 1))[:need]
random.shuffle(ds_prompts)
dataset = Dataset.from_dict({"prompt": ds_prompts})


def _versions():
    """Every installed distribution (normalized name -> version), python, torch's CUDA, GPU name and driver."""
    pk = {}
    for d in _md.distributions():
        name = d.metadata["Name"]
        if name:
            pk[re.sub(r"[-_.]+", "-", name).lower()] = d.version
    gpu = drv = None
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=60)
        if r.returncode == 0 and r.stdout.strip():
            gpu, drv = [x.strip() for x in r.stdout.strip().splitlines()[0].rsplit(",", 1)]
    except Exception:       # no nvidia-smi (a CPU smoke test): the fields stay None
        pass
    if gpu is None and torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
    return {"python": platform.python_version(), "packages": dict(sorted(pk.items())),
            "torch_cuda": torch.version.cuda, "gpu_name": gpu, "driver": drv}


VERSIONS = _versions()


def _snapshots():
    """The Hub snapshot commit each model was loaded from, read from the Hub cache after loading. Every file the
    Hub serves is stored under <HF_HUB_CACHE>/models--<org>--<name>/snapshots/<commit>/, the cache directory
    resolved as huggingface_hub's hf_hub_download resolves it, so a repo whose cache holds exactly one snapshot
    directory was loaded from that commit. With none or several, the value is the sorted list of them."""
    hub = str(_Path(_hub_constants.HF_HUB_CACHE).expanduser().resolve())
    out = {}
    for key, repo in (("policy", POLICY), ("judge_small", SMALL), ("judge_big", BIG)):
        d = os.path.join(hub, "models--" + repo.replace("/", "--"), "snapshots")
        found = sorted(os.listdir(d)) if os.path.isdir(d) else []
        out[key] = found[0] if len(found) == 1 else found
    return out


scorer_big = JudgeScorer(BIG, device=JUDGE_DEVICE, dtype=JUDGE_DTYPE, revision=REVISIONS["judge_big"])
scorer_small = JudgeScorer(SMALL, device=JUDGE_DEVICE, dtype=JUDGE_DTYPE, revision=REVISIONS["judge_small"])
LOG, PROBE_LOG = [], []
COMPS = []   # every completion of every step, full text, so the gate and any oracle can be re-run later
_n = {"c": 0}
EOS_ID = PAD_ID = None   # the policy tokenizer's ids, set once the trainer has loaded it


def _parses(text):
    """ast.parse plus compile of extract_code(text), as the gate does. Nothing is executed."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            compile(_ast.parse(extract_code(text or "")), "submission.py", "exec")
        return True
    except Exception:
        return False


def reward_fn(prompts=None, completions=None, completion_ids=None, **kw):
    """H: J05. T: J05 x g. Both judges, g and both oracles are computed and logged in both arms."""
    assert completion_ids is not None, "TRL did not pass completion_ids to the reward function"
    comps = completions or []
    pr = prompts or []
    assert len(completion_ids) == len(comps), (len(completion_ids), len(comps))
    c = _n["c"]
    st = kw.get("trainer_state")
    assert st is None or int(st.global_step) == c, ("log step", c, "trainer step", st.global_step)
    task = BY_PROMPT[pr[0]]
    hidden = [oracle_reward(x, task) for x in comps]
    hidden_len = [lenient_oracle_reward(x, task) for x in comps]   # same tests, prose stripped
    jb = scorer_big.score(pr[0], comps)
    js = scorer_small.score(pr[0], comps)
    g = [gate(x, FN) for x in comps]
    reward = list(js) if ARM == "judge" else [js[i] * g[i] for i in range(len(comps))]
    jbm = float(np.nanmean(jb)); jsm = float(np.nanmean(js)); djm = jsm - jbm
    hm = float(np.mean(hidden)); pm = float(np.mean(reward))
    adv = float(np.std(reward))
    LOG.append({"step": c, "task": task.name, "judge_big": jbm, "judge_small": jsm,
                "disagree": djm, "proxy": pm, "hidden": hm, "delta": pm - hm, "adv_std": adv,
                "hidden_lenient": float(np.mean(hidden_len)),
                "entropy": None, "kl": None, "grad_norm": None, "mean_len": None, "clipped": None,
                "gate_rate": float(np.mean(g)), "js_std": float(np.std(js)), "jb_std": float(np.std(jb))})
    COMPS.append({"step": c, "texts": list(comps), "n_tokens": [len(ids) for ids in completion_ids],
                  "eos": [not truncated(ids, EOS_ID, PAD_ID) for ids in completion_ids],
                  "fenced": [_FENCE.search(x) is not None for x in comps],
                  "parses": [_parses(x) for x in comps], "G": g, "judge_small": list(js),
                  "judge_big": list(jb), "reward": reward, "hidden": hidden, "hidden_lenient": hidden_len})
    _n["c"] += 1
    _STDOUT.write("STEP %d\n" % c)
    _STDOUT.flush()
    _status(c)
    return reward


class GeomCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kw):
        if logs and LOG:
            r = LOG[-1]
            for k in ("entropy", "kl", "grad_norm"):
                if r.get(k) is None and k in logs:
                    r[k] = float(logs[k])
            # E1 detectors: completion length and truncation rate (experiments/e1_detectors.py)
            for k, src in (("mean_len", "completions/mean_length"), ("clipped", "completions/clipped_ratio")):
                if r.get(k) is None and src in logs:
                    r[k] = float(logs[src])


class EvalProbe(TrainerCallback):
    """E1's held-out probe, unchanged except that it prints nothing: PROBE_N fresh completions at temperature
    1.0 every PROBE_EVERY steps, scored on the same hidden tests, off-gradient (no_grad, text discarded)."""
    def on_step_end(self, args, state, control, model=None, **kw):
        m = model if model is not None else kw.get("model")
        step = int(getattr(state, "global_step", 0))
        if m is None or PROBE_N <= 0 or step % PROBE_EVERY != 0:
            return
        try:
            tok = kw.get("processing_class") or kw.get("tokenizer")
            if tok is None:
                return
            task = TASKS[0]
            m.eval()
            pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
            enc = tok([task.prompt], return_tensors="pt").to(m.device)
            with torch.no_grad():
                out = m.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0,
                                 max_new_tokens=MAXTOK, num_return_sequences=PROBE_N,
                                 pad_token_id=pad)
            m.train()
            plen = enc["input_ids"].shape[1]
            texts = tok.batch_decode(out[:, plen:], skip_special_tokens=True)
            hid = [oracle_reward(t, task) for t in texts]
            PROBE_LOG.append({"step": step, "n": PROBE_N, "probe_hidden": float(np.mean(hid)),
                              "probe_sem": float(np.std(hid) / max(np.sqrt(len(hid)), 1.0))})
        except Exception as e:
            try:
                m.train()
            except Exception:
                pass
            print("PROBE_ERR step %s %s" % (step, repr(e)[:200]), file=sys.stderr, flush=True)


# model_init_kwargs carries the policy revision, which TRL passes to the policy, the reference model and the
# processing class, and the architecture, so that create_model_from_path skips its AutoConfig lookup at the
# default branch and the class comes from the pinned config (module docstring: trl 1.14.2,
# trl/trainer/grpo_trainer.py GRPOTrainer.__init__ and trl/trainer/utils.py create_model_from_path).
args = _Config(output_dir="/tmp/out", per_device_train_batch_size=GEN, num_generations=GEN,
               max_steps=STEPS, learning_rate=LR, max_completion_length=MAXTOK, beta=BETA,
               logging_steps=1, save_strategy="no", report_to=[], seed=SEED,
               model_init_kwargs={"revision": REVISIONS["policy"], "architecture": AutoModelForCausalLM})
trainer = _Trainer(model=POLICY, reward_funcs=[reward_fn], args=args, train_dataset=dataset,
                   callbacks=[GeomCallback(), EvalProbe()])
# transformers 5.19.0 adds PrinterCallback (disable_tqdm) or ProgressCallback (default), and both print every
# log dict. Removing them leaves on_log to the other callbacks, GeomCallback included.
for _cb in (PrinterCallback, ProgressCallback):
    while trainer.pop_callback(_cb) is not None:
        pass
if any(isinstance(cb, (PrinterCallback, ProgressCallback)) for cb in trainer.callback_handler.callbacks):
    _stop("a printing trainer callback is still registered")
EOS_ID, PAD_ID = trainer.processing_class.eos_token_id, trainer.processing_class.pad_token_id
if list(trainer.eos_token_ids) != [EOS_ID]:
    _stop("TRL's eos ids %s are not [tokenizer eos %s], so eos would not be TRL's clipped rule"
          % (list(trainer.eos_token_ids), EOS_ID))
for _m in (trainer.model, trainer.ref_model):      # the class AutoModelForCausalLM picked from the pinned config
    if _m is not None and type(_m).__name__ != (_m.config.architectures or [None])[0]:
        _stop("%s is not the pinned config's architecture %s" % (type(_m).__name__, _m.config.architectures))
# Every model is loaded now (both judges, the policy, the reference model and the processing class), so the Hub
# cache holds every snapshot the job read; each must be exactly its pin.
SNAPSHOTS = _snapshots()
for _k in REVISIONS:
    if SNAPSHOTS[_k] != REVISIONS[_k]:
        _stop("the Hub snapshot %s was loaded from is %s, not its pinned revision %s"
              % (_k, SNAPSHOTS[_k], REVISIONS[_k]))
trainer.train()

out = {"arm": ARM, "judge": JUDGE, "policy": POLICY, "task": TASK, "seed": SEED, "lr": LR, "steps": STEPS,
       "maxtok": MAXTOK, "probe_n": PROBE_N, "probe_every": PROBE_EVERY, "beta": BETA,
       "run_id": RUN_ID, "stage": STAGE, "revisions": REVISIONS, "versions": VERSIONS,
       "job_file_sha256": JOB_FILE_SHA256, "snapshots": SNAPSHOTS,
       "package_sha256": PACKAGE_SHA256, "git_commit": GIT_COMMIT, "template_sha256": TEMPLATE_SHA256,
       "lock_sha256": LOCK_SHA256, "gate_sha256": GATE_SHA256,
       "log": LOG, "completions": COMPS, "probe": PROBE_LOG, "trl_history": trainer.state.log_history}
with open(RESULT_PATH, "w") as f:
    json.dump(out, f)
'''

_PLACEHOLDERS = ("__PKG_B64__", "__ARM__", "__TASK__", "__SEED__", "__STAGE__", "__RUN_ID__", "__GIT_COMMIT__",
                 "__PACKAGE_SHA256__", "__TEMPLATE_SHA256__", "__LOCK_SHA256__", "__GATE_SHA256__")


def run_id(arm, task, seed):
    return "e1b_%s_%s_s%d" % (arm, task, seed)


def _git(*args):
    r = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError("git %s failed: %s" % (" ".join(args), r.stderr.decode("utf-8", "replace").strip()[-600:]))
    return r.stdout


def _blob(commit, path):
    """The committed bytes of path (no checkout filter). CRLF is refused, so every pinned SHA-256 is of LF text."""
    data = _git("cat-file", "blob", "%s:%s" % (commit, path))
    if b"\r\n" in data:
        raise RuntimeError("%s at %s has CRLF line endings; the pins are SHA-256 of LF text" % (path, commit))
    return data


def package_tar(commit):
    """git archive of flightrecorder/ at commit, uncompressed. The -c settings make the bytes independent of this
    machine's git config (core.autocrlf=true would write CRLF files into the archive)."""
    return _git("-c", "core.autocrlf=false", "-c", "core.eol=lf", "-c", "tar.umask=0002",
                "archive", "--format=tar", commit, PACKAGE_DIR + "/")


def bake(arm, task, seed, stage, commit, out_dir):
    """Write <out_dir>/<run_id>/job.py and job.py.lock for one registered run, from commit.
    Returns (job_path, provenance) with provenance = {run_id, git_commit, package_sha256, template_sha256,
    lock_sha256, gate_sha256, job_sha256}; job_sha256 is the SHA-256 of the job.py bytes on disk after writing,
    the value a run's measured job_file_sha256 must equal."""
    if arm not in ARMS:
        raise ValueError("arm %r is not registered (%s)" % (arm, ", ".join(ARMS)))
    if task not in TASKS:
        raise ValueError("task %r is not registered (%s)" % (task, ", ".join(TASKS)))
    if stage not in STAGE_SEEDS:
        raise ValueError("stage %r is not registered (%s)" % (stage, ", ".join(STAGE_SEEDS)))
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in STAGE_SEEDS[stage]:
        raise ValueError("seed %r is not registered for stage %s (%d-%d)"
                         % (seed, stage, min(STAGE_SEEDS[stage]), max(STAGE_SEEDS[stage])))
    sha = _git("rev-parse", "--verify", "%s^{commit}" % commit).decode().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise RuntimeError("commit %r did not resolve to a full sha: %r" % (commit, sha))

    template = _blob(sha, TEMPLATE_PATH)
    here = Path(__file__).read_bytes().replace(b"\r\n", b"\n")
    if here != template:
        raise RuntimeError("%s differs from its committed version at %s; bake only from the committed template"
                           % (TEMPLATE_PATH, sha[:12]))
    lock, gate_src = _blob(sha, LOCK_PATH), _blob(sha, GATE_PATH)
    tar = package_tar(sha)
    with tarfile.open(fileobj=io.BytesIO(tar), mode="r:") as t:
        names = set(t.getnames())
        missing = [p for p in _REQUIRED if p not in names]
        if missing:
            raise RuntimeError("the package at %s lacks %s" % (sha[:12], ", ".join(missing)))
        if t.extractfile(GATE_PATH).read() != gate_src:
            raise RuntimeError("the archived %s differs from its blob" % GATE_PATH)
    rid = run_id(arm, task, seed)
    prov = {"run_id": rid, "git_commit": sha,
            "package_sha256": hashlib.sha256(tar).hexdigest(),
            "template_sha256": hashlib.sha256(template).hexdigest(),
            "lock_sha256": hashlib.sha256(lock).hexdigest(),
            "gate_sha256": hashlib.sha256(gate_src).hexdigest()}
    b64 = base64.b64encode(gzip.compress(tar, compresslevel=9, mtime=0)).decode("ascii")
    subs = {"__PKG_B64__": b64, "__ARM__": arm, "__TASK__": task, "__SEED__": str(seed), "__STAGE__": stage,
            "__RUN_ID__": rid, "__GIT_COMMIT__": sha, "__PACKAGE_SHA256__": prov["package_sha256"],
            "__TEMPLATE_SHA256__": prov["template_sha256"], "__LOCK_SHA256__": prov["lock_sha256"],
            "__GATE_SHA256__": prov["gate_sha256"]}
    assert set(subs) == set(_PLACEHOLDERS)
    for k in _PLACEHOLDERS:
        if UV.count(k) != 1:
            raise RuntimeError("placeholder %s appears %d times in the template" % (k, UV.count(k)))
    src = UV
    for k in _PLACEHOLDERS:
        src = src.replace(k, subs[k])
    left = re.findall(r"__[A-Z][A-Z0-9_]*__", src.replace(b64, ""))
    if left:
        raise RuntimeError("unsubstituted placeholders: " + ", ".join(sorted(set(left))))

    job_dir = Path(out_dir) / rid
    job_dir.mkdir(parents=True, exist_ok=True)
    job_path = job_dir / "job.py"
    job_path.write_bytes(src.encode("ascii"))            # LF: the template string has LF line endings
    (job_dir / "job.py.lock").write_bytes(lock)          # uv's script lock is <script>.lock
    with tempfile.TemporaryDirectory() as d:             # fail here, not on the instance
        py_compile.compile(str(job_path), cfile=str(Path(d) / "job.pyc"), doraise=True)
    prov["job_sha256"] = hashlib.sha256(job_path.read_bytes()).hexdigest()
    return job_path, prov


def header():
    """The PEP 723 block of the template, the only part of a job that its lock depends on."""
    lines = UV.split("\n")
    end = lines.index("# ///", 1)
    return "\n".join(lines[:end + 1]) + "\n"


def main(argv):
    if argv[:1] == ["--write-header"] and len(argv) == 2:
        with open(argv[1], "w", encoding="ascii", newline="\n") as f:
            f.write(header())
        return 0
    if argv[:1] == ["--bake"] and len(argv) == 7:
        arm, task, seed, stage, commit, out_dir = argv[1:]
        job_path, prov = bake(arm, task, int(seed), stage, commit, out_dir)
        print(json.dumps(dict(prov, job_path=str(job_path)), indent=1))
        return 0
    print(__doc__.strip().splitlines()[0])
    print("usage: python scripts/e1b_job.py --bake ARM TASK SEED STAGE COMMIT OUT_DIR | --write-header PATH")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
