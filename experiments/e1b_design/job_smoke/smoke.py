"""E1b job smoke test on CPU: bake a real job from a scratch-clone commit, patch it for a CPU run, run both arms.

A design record for scripts/e1b_job.py (build_contract sections 4-6), not a registered analysis: no value it
prints feeds any decision. Nothing here launches, uploads or spends, and nothing lands in the repository: every
environment and output goes under WORK, which must lie outside it. Network use is read-only (PyPI, the PyTorch
CPU index, and public tiny models from the Hugging Face Hub, read anonymously).

    python experiments/e1b_design/job_smoke/smoke.py WORK          # exit 0 when every check passes

STEPS
  1. Clone the repository into WORK/clone (core.autocrlf=false), copy the working-tree job template, lock and
     the job's package import closure (bake's _REQUIRED) into it with LF line endings, commit there, and bake
     judge and judgegate jobs for count_vowels, pilot seed 299, with the clone's scripts/e1b_job.py:bake.
  2. Build WORK/v: uv 0.12.23 (the lock's uv, from its PyPI wheel), CPython 3.12.15 (uv-managed), and every
     locked distribution by hash from `uv export --script --locked` of the baked job, except torch, which is the
     CPU build of the locked version (torch==2.11.0+cpu from download.pytorch.org/whl/cpu).
  3. Patch each baked job only where a CPU run needs it (cpu_patch below), and run it with the venv's python
     through this file's --run-job mode. That mode runs the job as `python job.py` would and records, without
     changing anything the job computes: every from_pretrained call (installed when the job imports
     flightrecorder.repro.reward_judge, after transformers and trl are loaded, so the job's own import order and
     its progress-bar settings are untouched), every Hub download transformers makes, TRL's reward call
     (_calculate_rewards: the completion_ids it passes and the rewards it gets back), and every status.json
     write (os.replace onto E1B_STATUS_PATH).
  4. Check the outputs, run the stop cases, write WORK/report.json and print one line per check.

PATCHES (cpu_patch; each target string must occur exactly once in the baked job)
  * models: tiny public models at their current Hub revisions (read from the Hub API at run time). The policy
    and judge_small are trl-internal-testing/tiny-Qwen2ForCausalLM-2.5; judge_big is
    trl-internal-testing/tiny-Qwen3ForCausalLM, a different model, so a swap of the two judge columns would
    show. The JUDGE label (the result's "judge" field) stays "Qwen/Qwen2.5-0.5B-Instruct".
  * JUDGE_DEVICE, JUDGE_DTYPE = "cpu", "float32"; GRPOConfig gets use_cpu=True and bf16=False (float32
    training; the real job keeps GRPOConfig's bf16 default, as E1 did).
  * STEPS 3 (generations 8, lr, beta and max_completion_length 256 unchanged); PROBE_N, PROBE_EVERY = 2, 1.
  * RESULT_PATH and output_dir under the case folder; E1B_STATUS_PATH is set in the environment.
  * The eos guard. The tiny policy's generation config declares eos [151645, 151643] while its tokenizer eos is
    151645 (pad 151643), so TRL's eos_token_ids is [151645, 151643] and the job's guard
    `list(trainer.eos_token_ids) != [EOS_ID]` stops it (the eosguard case below shows this). The patch replaces
    the guard with the exact condition under which the logged flag is TRL's rule:
    set(eos_token_ids) | {pad} == {tokenizer eos, pad}. It holds for the tiny policy, so eos is still checked
    against TRL's clipped_ratio completion by completion.
  On Windows the venv, uv's cache, HF_HOME and TEMP use the 8.3 short form of WORK, because the long scratch
  path breaks transformers' imports and the Hub cache at MAX_PATH.

CASES
  main_judge, main_judgegate   both arms with the patches above.
  eosmix_judge, eosmix_judgegate
                               the same plus GRPOConfig(generation_kwargs={"sequence_bias": +6.0 on token ids
                               151643 and 151645}), a sampling-only change, so that a random-weight policy ends
                               some completions early and eos, n_tokens and clipped_ratio are not all at the cap.
  stop_package                 one byte of the embedded package changed: the package check must stop the job.
  stop_gate                    twin_gate.py in the package changed and package_sha256 re-pinned to the changed
                               package: the gate check must stop the job.
  stop_lock, stop_nolock       job.py.lock with one byte appended, or absent: the lock check must stop the job.
  eosguard                     the CPU patches without the eos-guard patch: the guard must stop the job.
  stop_snapshot                judge_big loaded at the tiny model's real revision while REVISIONS pins another
                               commit (its last hex digit changed): the snapshot check, which reads the Hub cache
                               after every model has loaded, must stop the job before training.
  matched_main, matched_eosmix the judge and judgegate runs of a kind share seed 299, and nothing before the first
                               generation depends on ARM: their step-0 texts, n_tokens and eos must be identical.
  doc                          the module docstring's uv install is the hash-pinned form, and its run command is
                               RUN_ARGS.
  arch                         GRPOTrainer built twice on the tiny policy (--compare-arch, no job): with
                               model_init_kwargs {"revision"} (the job as first built) and {"revision",
                               "architecture": AutoModelForCausalLM} (the job now). Both must load the same class,
                               weights, config, generation config and eos ids; only the second may be free of
                               from_pretrained calls without the revision.
  A stop case must exit 1 with its "E1B STOP before training: ..." line on stderr and print no STEP line; a
  provenance stop must also come before any status.json write and any from_pretrained call.

LIMITS
  Random-weight models write no code, so g is 0 on every completion: reward = judge_small x g is checked only
  at g = 0 in the twin arm, and the offline-gate comparison only on gate failures. Both judges are random too,
  so their scores carry no meaning. The pins equal each tiny model's current main, so a Hub read that ignored
  the revision would fetch the same files: only the instrumentation, and a refs/main left in the Hub cache by a
  read that resolved the branch, show which reads were pinned.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TASK, SEED, STAGE, ARMS = "count_vowels", 299, "pilot", ("judge", "judgegate")
TINY_POLICY = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"     # policy and judge_small
TINY_BIG = "trl-internal-testing/tiny-Qwen3ForCausalLM"            # judge_big
JUDGE_LABEL = "Qwen/Qwen2.5-0.5B-Instruct"
STEPS, GEN, PROBE_N, PROBE_EVERY, MAXTOK = 3, 8, 2, 1, 256
EOS_BIAS, BIAS_IDS = 6.0, (151643, 151645)    # eosmix: about 0.5% eos per token on a 151,936-token uniform draw
TORCH_CPU, TORCH_CPU_INDEX = "torch==2.11.0+cpu", "https://download.pytorch.org/whl/cpu"
DIRECT = {"torch": "2.11.0", "transformers": "5.19.0", "trl": "1.14.2", "datasets": "5.1.0",
          "accelerate": "1.15.0", "numpy": "2.5.3"}
ABSENT = "<absent>"
TOP_KEYS = ("arm", "judge", "policy", "task", "seed", "lr", "steps", "maxtok", "probe_n", "probe_every", "beta",
            "run_id", "stage", "revisions", "versions", "job_file_sha256", "snapshots", "package_sha256",
            "git_commit", "template_sha256", "lock_sha256", "gate_sha256", "log", "completions", "probe",
            "trl_history")
PROV_KEYS = ("run_id", "git_commit", "package_sha256", "template_sha256", "lock_sha256", "gate_sha256",
             "job_sha256")    # build_contract section 6: bake()'s provenance
MODEL_REPOS = {"policy": TINY_POLICY, "judge_small": TINY_POLICY, "judge_big": TINY_BIG}
LOG_KEYS = ("step", "task", "judge_big", "judge_small", "disagree", "proxy", "hidden", "hidden_lenient", "delta",
            "adv_std", "entropy", "kl", "grad_norm", "mean_len", "clipped", "gate_rate", "js_std", "jb_std")
COMP_KEYS = ("step", "texts", "n_tokens", "eos", "fenced", "parses", "G", "judge_small", "judge_big", "reward",
             "hidden", "hidden_lenient")
STOP = "E1B STOP before training: "
METRIC_WORDS = re.compile(r"(loss|reward|kl|entropy|grad_norm|clipped|mean_len|judge|hidden|proxy|probe|"
                          r"disagree|gate_rate|adv|advantage)", re.I)


# ============================================================================================
# --run-job: runs inside the job's venv. Standard library only, and nothing printed to stdout.
# ============================================================================================
def run_job(job, trace):
    """Run JOB as `python JOB` would, appending one JSON line per recorded event to TRACE."""
    import importlib.abc
    import runpy
    import threading

    status_path = os.path.abspath(os.environ["E1B_STATUS_PATH"])
    seq = [0]
    tls = threading.local()

    def rec(event, **kw):
        seq[0] += 1
        kw.update(seq=seq[0], event=event)
        with open(trace, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(kw, default=repr) + "\n")

    real_replace = os.replace

    def replace(src, dst, *a, **k):
        out = real_replace(src, dst, *a, **k)
        if os.path.abspath(dst) == status_path:
            with open(dst, encoding="utf-8") as f:
                rec("status", text=f.read())
        return out
    os.replace = replace

    def stack():
        return list(getattr(tls, "stack", []))

    def wrap_from_pretrained(owner, label):
        raw = owner.__dict__.get("from_pretrained")
        if not isinstance(raw, classmethod):
            rec("wrap_missing", label=label, kind=type(raw).__name__)
            return
        fn = raw.__func__

        def from_pretrained(cls, *a, **k):
            st = stack()
            ident = str(a[0] if a else k.get("pretrained_model_name_or_path"))
            rec("from_pretrained", label=label, cls=cls.__name__, id=ident, depth=len(st),
                revision=k.get("revision", ABSENT), keys=sorted(k), outer=st[:1])
            tls.stack = st + [[label, cls.__name__, ident]]
            try:
                obj = fn(cls, *a, **k)
            finally:
                tls.stack = st
            rec("loaded", label=label, cls=cls.__name__, id=ident, depth=len(st), type=type(obj).__name__)
            return obj
        setattr(owner, "from_pretrained", classmethod(from_pretrained))

    def wrap_hub(owner, name):
        real = getattr(owner, name)

        def hub(*a, **k):
            args = a[1:] if isinstance(owner, type) else a      # HfApi methods: drop self
            rec("hub", fn=name, repo=str(args[0] if args else k.get("repo_id")),
                filename=k.get("filename", args[1] if len(args) > 1 else None),
                revision=k.get("revision", ABSENT), stack=stack())
            return real(*a, **k)
        setattr(owner, name, hub)

    def install():
        import huggingface_hub
        import transformers
        import transformers.utils.hub as th
        from transformers.models.auto.auto_factory import _BaseAutoModelClass
        from trl.trainer import grpo_trainer as gt
        for label, owner in (("PreTrainedModel", transformers.PreTrainedModel),
                             ("AutoModel*", _BaseAutoModelClass),      # auto_class_update gives every Auto
                             ("AutoModelForCausalLM", transformers.AutoModelForCausalLM),   # class its own copy
                             ("AutoConfig", transformers.AutoConfig),
                             ("PreTrainedConfig", transformers.PretrainedConfig),
                             ("AutoTokenizer", transformers.AutoTokenizer),
                             ("PreTrainedTokenizerBase", transformers.PreTrainedTokenizerBase),
                             ("AutoProcessor", transformers.AutoProcessor),
                             ("ProcessorMixin", transformers.ProcessorMixin),
                             ("GenerationConfig", transformers.GenerationConfig),
                             ("AutoFeatureExtractor", transformers.AutoFeatureExtractor),
                             ("FeatureExtractionMixin", transformers.FeatureExtractionMixin)):
            wrap_from_pretrained(owner, label)
        for name in ("hf_hub_download", "snapshot_download"):
            wrap_hub(th, name)
        for name in ("list_repo_tree", "list_repo_files", "model_info", "repo_info", "file_exists",
                     "hf_hub_download", "snapshot_download"):
            if hasattr(huggingface_hub.HfApi, name):
                wrap_hub(huggingface_hub.HfApi, name)
        real_calc = gt.GRPOTrainer._calculate_rewards

        def calc(self, inputs, prompts, completions, completion_ids_list, *a, **k):
            out = real_calc(self, inputs, prompts, completions, completion_ids_list, *a, **k)
            ids = completion_ids_list
            eos_pad = [*self.eos_token_ids, self._tokenizer.pad_token_id]
            rec("reward_call", step=int(self.state.global_step), ids_type=type(ids).__name__,
                elem_types=sorted({type(x).__name__ for x in ids}),
                int_ids=all(isinstance(t, int) for x in ids for t in x),
                n_tokens=[len(x) for x in ids], last=[x[-1] if len(x) else None for x in ids],
                trl_truncated=[(not len(x)) or x[-1] not in eos_pad for x in ids],     # TRL's clipped rule
                completions=list(completions), rewards=[float(v) for v in out[:, 0].tolist()],
                eos_token_ids=list(self.eos_token_ids), pad=self._tokenizer.pad_token_id)
            return out
        gt.GRPOTrainer._calculate_rewards = calc
        rec("installed")

    class Hook(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == "flightrecorder.repro.reward_judge":
                sys.meta_path.remove(self)
                install()
            return None

    sys.meta_path.insert(0, Hook())
    sys.argv = [job]
    sys.path[0] = os.path.dirname(os.path.abspath(job))
    g = runpy.run_path(job, run_name="__main__")
    tr = g["trainer"]

    def dev_dtype(m):
        p = next(m.parameters())
        return [str(p.device), str(p.dtype)]
    rec("globals", eos_id=g["EOS_ID"], pad_id=g["PAD_ID"], trl_eos_token_ids=list(tr.eos_token_ids),
        callbacks=[type(cb).__name__ for cb in tr.callback_handler.callbacks],
        bf16=tr.args.bf16, use_cpu=tr.args.use_cpu, generation_kwargs=tr.args.generation_kwargs,
        policy=dev_dtype(tr.model), ref=dev_dtype(tr.ref_model) if tr.ref_model is not None else None,
        judge_small=dev_dtype(g["scorer_small"].model), judge_big=dev_dtype(g["scorer_big"].model),
        policy_class=type(tr.model).__name__, ref_class=type(tr.ref_model).__name__)


def compare_arch(model_id, revision, out_dir):
    """--compare-arch, in the venv: GRPOTrainer with model_init_kwargs {"revision"} and then {"revision",
    "architecture": AutoModelForCausalLM}; print one JSON line on stdout."""
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    from datasets import Dataset
    from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
    from trl import GRPOConfig, GRPOTrainer
    calls = []

    def wrap(owner, label):
        fn = owner.__dict__["from_pretrained"].__func__

        def from_pretrained(cls, *a, **k):
            calls.append([label, cls.__name__, str(k.get("revision", ABSENT))])
            return fn(cls, *a, **k)
        owner.from_pretrained = classmethod(from_pretrained)
    for label, owner in (("AutoConfig", AutoConfig), ("AutoModelForCausalLM", AutoModelForCausalLM),
                         ("PreTrainedModel", PreTrainedModel)):
        wrap(owner, label)

    def build(kw):
        calls.clear()
        args = GRPOConfig(output_dir=out_dir, per_device_train_batch_size=GEN, num_generations=GEN, max_steps=1,
                          learning_rate=1e-6, max_completion_length=8, beta=0.02, logging_steps=1,
                          save_strategy="no", report_to=[], seed=0, use_cpu=True, bf16=False, model_init_kwargs=kw)
        tr = GRPOTrainer(model=model_id, args=args, train_dataset=Dataset.from_dict({"prompt": ["x"] * 16}),
                         reward_funcs=[lambda prompts=None, completions=None, **k: [0.0] * len(completions)])
        return tr, list(calls)
    old, c_old = build({"revision": revision})
    new, c_new = build({"revision": revision, "architecture": AutoModelForCausalLM})
    same = {}
    for name in ("model", "ref_model"):
        a, b = getattr(old, name).state_dict(), getattr(new, name).state_dict()
        same[name + "_weights"] = sorted(a) == sorted(b) and all(bool((a[k] == b[k]).all()) for k in a)
        same[name + "_config"] = getattr(old, name).config.to_dict() == getattr(new, name).config.to_dict()
        same[name + "_generation_config"] = (getattr(old, name).generation_config.to_dict()
                                             == getattr(new, name).generation_config.to_dict())
        same[name + "_class"] = [type(getattr(old, name)).__name__, type(getattr(new, name)).__name__]
    same["eos_token_ids"] = [list(old.eos_token_ids), list(new.eos_token_ids)]
    print(json.dumps({"same": same, "calls_old": c_old, "calls_new": c_new}), file=sys.__stdout__, flush=True)


# ============================================================================================
# Harness (the repository's Python, numpy available)
# ============================================================================================
def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _short(p):
    """The 8.3 form of an existing path on Windows (MAX_PATH); the path itself elsewhere."""
    p = str(p)
    if os.name != "nt":
        return p
    import ctypes
    buf = ctypes.create_unicode_buffer(1024)
    n = ctypes.windll.kernel32.GetShortPathNameW(p, buf, 1024)
    return buf.value if 0 < n < 1024 else p


def _run(cmd, **kw):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError("%s failed (%d): %s" % (" ".join(map(str, cmd[:4])), r.returncode,
                                                   (r.stderr or r.stdout)[-1500:]))
    return r


def _rmtree(p):
    def onexc(func, path, exc):      # git's pack files are read-only on Windows
        os.chmod(path, stat.S_IWRITE)
        func(path)
    if Path(p).exists():
        if sys.version_info >= (3, 12):
            shutil.rmtree(p, onexc=onexc)
        else:
            shutil.rmtree(p, onerror=lambda f, q, e: onexc(f, q, e))


def _load(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Ctx:
    def __init__(self, work):
        self.work = Path(work).resolve()
        self.work.mkdir(parents=True, exist_ok=True)
        self.short = Path(_short(self.work))
        for d in ("uvc", "py", "hf", "tmp", "runs"):
            (self.work / d).mkdir(exist_ok=True)
        self.uv = self.work / "uv" / ("uv.exe" if os.name == "nt" else "uv")
        self.vpy = self.short / "v" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        # huggingface_hub resolves its cache dir to the long path (Path.resolve), so on Windows HF_HOME takes the
        # extended-length form, which file APIs accept beyond MAX_PATH
        self.hf_home = (r"\\?" + "\\" if os.name == "nt" else "") + str(self.work / "hf")
        self.uvenv = dict(os.environ, UV_CACHE_DIR=str(self.short / "uvc"),
                          UV_PYTHON_INSTALL_DIR=str(self.short / "py"), UV_NO_CONFIG="1")
        self.rows = []
        self.info = {}

    def check(self, case, name, ok, detail=None):
        self.rows.append({"case": case, "check": name, "ok": bool(ok), "detail": detail})
        return ok


def setup_uv(ctx, version):
    if not ctx.uv.is_file():
        d = ctx.work / "uvwhl"
        _run([sys.executable, "-m", "pip", "download", "--only-binary", ":all:", "--no-deps",
              "uv==" + version, "-d", d])
        whl = sorted(d.glob("uv-%s-*.whl" % version))[0]
        ctx.uv.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(whl) as z:
            ctx.uv.write_bytes(z.read("uv-%s.data/scripts/%s" % (version, ctx.uv.name)))
        if os.name != "nt":
            ctx.uv.chmod(0o755)
    got = _run([ctx.uv, "--version"]).stdout.split()[1]
    if got != version:
        raise RuntimeError("uv %s, wanted %s" % (got, version))


def clone_and_bake(ctx):
    """A scratch clone holding the working tree's job files, committed; both arms baked from that commit."""
    wt = _load("e1b_job_worktree", REPO / "scripts" / "e1b_job.py")
    files = (wt.TEMPLATE_PATH, wt.LOCK_PATH) + tuple(wt._REQUIRED)
    clone = ctx.work / "clone"
    _rmtree(clone)
    _run(["git", "clone", "-q", "--no-hardlinks", "-c", "core.autocrlf=false", REPO, clone])
    copied = {}
    for rel in files:
        b = (REPO / rel).read_bytes().replace(b"\r\n", b"\n")
        head = subprocess.run(["git", "-C", str(REPO), "cat-file", "blob", "HEAD:" + rel], capture_output=True)
        (clone / rel).parent.mkdir(parents=True, exist_ok=True)
        (clone / rel).write_bytes(b)
        copied[rel] = {"sha256": _sha(b), "vs_repo_HEAD": "absent at HEAD" if head.returncode else
                       ("same" if head.stdout.replace(b"\r\n", b"\n") == b else "differs")}
    _run(["git", "-C", clone, "add", "--", *files])
    _run(["git", "-C", clone, "-c", "user.name=e1b-smoke", "-c", "user.email=e1b-smoke@example.invalid",
          "commit", "-q", "-m", "e1b smoke: working-tree job files"])
    J = _load("e1b_job_clone", clone / "scripts" / "e1b_job.py")
    assert J.ROOT == clone.resolve(), (J.ROOT, clone)
    head = _run(["git", "-C", clone, "rev-parse", "HEAD"]).stdout.strip()
    _rmtree(ctx.work / "baked")
    baked = {}
    for arm in ARMS:
        jp, prov = J.bake(arm, TASK, SEED, STAGE, "HEAD", ctx.work / "baked")
        baked[arm] = (Path(jp), prov)
    ctx.info["clone"] = {"commit": head, "copied": copied}
    ctx.info["provenance"] = {a: p for a, (_, p) in baked.items()}
    return J, baked, files


def setup_env(ctx, J, job):
    """WORK/v: CPython 3.12.15, torch==2.11.0+cpu, and every other locked distribution by hash."""
    reqs = ctx.work / "env_reqs.txt"
    _run([ctx.uv, "export", "--no-config", "--script", job, "--format", "requirements.txt", "--locked",
          "--no-emit-package", "torch", "-o", reqs], env=ctx.uvenv)
    lock = (job.parent / "job.py.lock").read_bytes()
    want = {"lock_sha256": _sha(lock), "reqs_sha256": _sha(reqs.read_bytes()), "torch": TORCH_CPU,
            "python": J.PYTHON_VERSION, "uv": J.UV_VERSION}
    marker = ctx.work / "v_ok.json"
    if not (marker.is_file() and json.loads(marker.read_text()) == want and Path(ctx.vpy).is_file()):
        _rmtree(ctx.work / "v")
        _run([ctx.uv, "venv", "--no-config", "--python", J.PYTHON_VERSION, "--managed-python", ctx.short / "v"],
             env=ctx.uvenv)
        _run([ctx.uv, "pip", "install", "--no-config", "--python", ctx.vpy, "--no-deps", "--index-url",
              TORCH_CPU_INDEX, TORCH_CPU], env=ctx.uvenv)
        _run([ctx.uv, "pip", "install", "--no-config", "--python", ctx.vpy, "--no-deps", "--require-hashes",
              "-r", reqs], env=ctx.uvenv)
        marker.write_text(json.dumps(want), encoding="utf-8")
    chk = _run([ctx.uv, "pip", "check", "--python", ctx.vpy], env=ctx.uvenv)
    ctx.info["env"] = dict(want, uv_pip_check=(chk.stdout + chk.stderr).strip().splitlines()[-1])


def hub_sha(model_id):
    with urllib.request.urlopen("https://huggingface.co/api/models/" + model_id, timeout=60) as r:
        return json.load(r)["sha"]


# --------------------------------------------------------------------------------------------
# Job text: patches and tampering
# --------------------------------------------------------------------------------------------
def _sub(src, old, new):
    n = src.count(old)
    if n != 1:
        raise RuntimeError("patch target occurs %d times: %r" % (n, old[:90]))
    return src.replace(old, new)


def cpu_patch(src, case_dir, revs, eos_guard=True, eos_bias=False):
    d = str(case_dir).replace("\\", "/")
    gen = (' generation_kwargs={"sequence_bias": [[[%d], %r], [[%d], %r]]},'
           % (BIAS_IDS[0], EOS_BIAS, BIAS_IDS[1], EOS_BIAS)) if eos_bias else ""
    subs = [('POLICY, JUDGE = "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-0.5B-Instruct"',
             'POLICY, JUDGE = "%s", "%s"' % (TINY_POLICY, JUDGE_LABEL)),
            ('BIG, SMALL = "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"',
             'BIG, SMALL = "%s", "%s"' % (TINY_BIG, TINY_POLICY)),
            ('"policy": "8faed761d45a263340a0528343f099c05c9a4323"', '"policy": "%s"' % revs["policy"]),
            ('"judge_small": "7ae557604adf67be50417f59c2c2f167def9a775"', '"judge_small": "%s"' % revs["judge_small"]),
            ('"judge_big": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"', '"judge_big": "%s"' % revs["judge_big"]),
            ('JUDGE_DEVICE, JUDGE_DTYPE = "cuda", "float16"', 'JUDGE_DEVICE, JUDGE_DTYPE = "cpu", "float32"'),
            ('STEPS, GEN, LR, BETA, MAXTOK = 400, 8, 1e-6, 0.02, 256',
             'STEPS, GEN, LR, BETA, MAXTOK = %d, %d, 1e-6, 0.02, %d' % (STEPS, GEN, MAXTOK)),
            ('PROBE_N, PROBE_EVERY = 16, 5', 'PROBE_N, PROBE_EVERY = %d, %d' % (PROBE_N, PROBE_EVERY)),
            ('RESULT_PATH = "/tmp/result.json"', 'RESULT_PATH = "%s/result.json"' % d),
            ('args = _Config(output_dir="/tmp/out",',
             'args = _Config(output_dir="%s/out", use_cpu=True, bf16=False,%s' % (d, gen))]
    if eos_guard:
        subs.append(("if list(trainer.eos_token_ids) != [EOS_ID]:",
                     "if set(trainer.eos_token_ids) | {PAD_ID} != {EOS_ID, PAD_ID}:"))
    for old, new in subs:
        src = _sub(src, old, new)
    return src, subs


_B64 = re.compile(r'_TAR = gzip\.decompress\(base64\.b64decode\("([A-Za-z0-9+/=]*)"\)\)')


def embedded_tar(src):
    return gzip.decompress(base64.b64decode(_B64.search(src).group(1)))


def with_tar(src, tar):
    m = _B64.search(src)
    return src[:m.start(1)] + base64.b64encode(gzip.compress(tar, mtime=0)).decode("ascii") + src[m.end(1):]


def flip_case_in_member(tar, name):
    """Change the case of the 4th byte of a package file (inside its docstring); the tar stays valid."""
    with tarfile.open(fileobj=io.BytesIO(tar), mode="r:") as t:
        off = t.getmember(name).offset_data + 3
    b = bytearray(tar)
    assert chr(b[off]).isalpha(), (name, chr(b[off]))
    b[off] ^= 0x20
    return bytes(b)


# --------------------------------------------------------------------------------------------
# Running a case
# --------------------------------------------------------------------------------------------
def run_case(ctx, name, src, lock_bytes, timeout=3600):
    d = ctx.work / "runs" / name
    _rmtree(d)
    d.mkdir(parents=True)
    sd = Path(_short(d))
    (d / "job.py").write_text(src, encoding="ascii", newline="\n")
    if lock_bytes is not None:
        (d / "job.py.lock").write_bytes(lock_bytes)
    env = {k: v for k, v in os.environ.items() if k not in (
        "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HUB_OFFLINE", "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP",
        "VIRTUAL_ENV", "UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR")}
    env.update(HF_HOME=ctx.hf_home, TEMP=str(ctx.short / "tmp"), TMP=str(ctx.short / "tmp"),
               TMPDIR=str(ctx.short / "tmp"), PYTHONUNBUFFERED="1", E1B_STATUS_PATH=str(sd / "status.json"))
    cmd = [str(ctx.vpy), str(Path(__file__).resolve()), "--run-job", str(sd / "job.py"), str(sd / "trace.jsonl")]
    t0 = time.time()
    with open(d / "stdout.txt", "wb") as so, open(d / "stderr.txt", "wb") as se:
        rc = subprocess.run(cmd, stdout=so, stderr=se, env=env, cwd=str(sd), timeout=timeout).returncode
    trace = [json.loads(x) for x in (d / "trace.jsonl").read_text(encoding="utf-8").splitlines()] \
        if (d / "trace.jsonl").is_file() else []
    res = {"rc": rc, "seconds": round(time.time() - t0, 1), "dir": d, "trace": trace,
           "stdout": (d / "stdout.txt").read_bytes(), "stderr": (d / "stderr.txt").read_bytes(),
           "status": json.loads((d / "status.json").read_text()) if (d / "status.json").is_file() else None,
           "result": None}
    if (d / "result.json").is_file():
        with open(d / "result.json", encoding="ascii") as f:
            res["result"] = json.load(f)
    return res


def _lines(b):
    return b.decode("utf-8", "replace").splitlines()


def _close(a, b, tol=1e-9):
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol


def _is_float(x):
    return isinstance(x, float)


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


# --------------------------------------------------------------------------------------------
# Checks on a training case
# --------------------------------------------------------------------------------------------
def check_training(ctx, case, arm, res, prov, revs, ids_by_model, lock_versions, eos_mix):
    import numpy as np
    offline_gate = _load("e1b_offline_gate", REPO / "experiments" / "e1b_offline_gate.py").offline_gate
    from flightrecorder.repro.reward_testhack import _FENCE     # e1b_offline_gate put REPO on sys.path
    C = lambda name, ok, detail=None: ctx.check(case, name, ok, detail)   # noqa: E731
    C("exit code 0", res["rc"] == 0, {"rc": res["rc"], "seconds": res["seconds"],
                                     "stderr_tail": _lines(res["stderr"])[-5:] if res["rc"] else None})
    if res["rc"] != 0 or res["result"] is None:
        return
    R, tr = res["result"], res["trace"]

    # stdout and stderr
    out = _lines(res["stdout"])
    other = [x for x in out if not re.fullmatch(r"STEP \d+", x)]
    C("stdout is exactly STEP 0..%d" % (STEPS - 1), out == ["STEP %d" % i for i in range(STEPS)],
      {"lines": out, "other_lines_verbatim": other, "crlf": res["stdout"].count(b"\r\n")})
    err = _lines(res["stderr"])
    uniq = {}
    for x in err:
        uniq[x] = uniq.get(x, 0) + 1
    flagged = [x for x in uniq if METRIC_WORDS.search(x) and re.search(r"\d\.\d|[:=]\s*-?\d", x)]
    C("no PROBE_ERR or E1B STOP line on stderr", not any(x.startswith(("PROBE_ERR", STOP)) for x in err),
      {"stderr_lines": len(err), "unique": len(uniq)})
    ctx.info.setdefault("stderr_unique", {})[case] = uniq
    ctx.info.setdefault("stderr_metric_like", {})[case] = flagged

    # result file: keys, types, values, lengths
    C("result top-level keys are exactly contract section 4's", sorted(R) == sorted(TOP_KEYS),
      {"extra": sorted(set(R) - set(TOP_KEYS)), "missing": sorted(set(TOP_KEYS) - set(R))})
    types_ok = (all(isinstance(R.get(k), str) for k in ("arm", "judge", "policy", "task", "run_id", "stage"))
                and all(_is_int(R.get(k)) for k in ("seed", "steps", "maxtok", "probe_n", "probe_every"))
                and all(_is_float(R.get(k)) for k in ("lr", "beta"))
                and all(isinstance(R.get(k), list) for k in ("log", "completions", "probe", "trl_history")))
    C("top-level field types", types_ok)
    want = {"arm": arm, "judge": JUDGE_LABEL, "policy": TINY_POLICY, "task": TASK, "seed": SEED, "lr": 1e-6,
            "steps": STEPS, "maxtok": MAXTOK, "probe_n": PROBE_N, "probe_every": PROBE_EVERY, "beta": 0.02,
            "run_id": prov["run_id"], "stage": STAGE, "revisions": revs}
    C("top-level values match the patched job", all(R.get(k) == v for k, v in want.items()),
      {k: [R.get(k), v] for k, v in want.items() if R.get(k) != v})
    hexok = lambda v, n: isinstance(v, str) and re.fullmatch("[0-9a-f]{%d}" % n, v) is not None   # noqa: E731
    C("revisions: exactly policy, judge_small, judge_big, 40-hex",
      sorted(R["revisions"]) == ["judge_big", "judge_small", "policy"] and all(hexok(v, 40) for v in R["revisions"].values()))
    V = R["versions"]
    C("versions: keys and types", isinstance(V, dict) and sorted(V) == ["driver", "gpu_name", "packages", "python", "torch_cuda"]
      and isinstance(V["python"], str) and re.fullmatch(r"3\.12\.\d+", V["python"]) is not None
      and isinstance(V["packages"], dict) and V["packages"] and all(isinstance(a, str) and isinstance(b, str) for a, b in V["packages"].items())
      and all(V[k] is None or isinstance(V[k], str) for k in ("torch_cuda", "gpu_name", "driver")),
      {"python": V.get("python"), "n_packages": len(V.get("packages") or {}), "torch_cuda": V.get("torch_cuda"),
       "gpu_name_type": type(V.get("gpu_name")).__name__, "driver_type": type(V.get("driver")).__name__})
    pk = V["packages"]
    bad = {}
    for n, v in pk.items():
        lv = lock_versions.get(n)
        if n == "torch":
            if v != TORCH_CPU.split("==")[1] or (lv or "").split("+")[0] != v.split("+")[0]:
                bad[n] = [v, lv]
        elif lv != v:
            bad[n] = [v, lv]
    C("every installed distribution has the lock's version (torch: the CPU build of it)",
      not bad and all(pk.get(k, "").split("+")[0] == v for k, v in DIRECT.items()),
      {"mismatch": bad, "direct": {k: pk.get(k) for k in DIRECT}, "python": V["python"]})
    C("provenance fields equal bake()'s", all(R[k] == prov[k] for k in prov if k not in ("run_id", "job_sha256"))
      and all(hexok(R[k], 64) for k in ("package_sha256", "template_sha256", "lock_sha256", "gate_sha256"))
      and hexok(R["git_commit"], 40))
    # Measured provenance (build_contract section 8). The job hashes its own file; this run executed the CPU-patched
    # job, so the measured hash must be that file's and must differ from bake()'s job_sha256, as an edited job would.
    ran = _sha((res["dir"] / "job.py").read_bytes())
    C("job_file_sha256 = SHA-256 of the job.py this run executed (patched, so not bake()'s job_sha256)",
      hexok(R.get("job_file_sha256"), 64) and R.get("job_file_sha256") == ran and ran != prov.get("job_sha256"),
      {"job_file_sha256": R.get("job_file_sha256"), "ran": ran, "bake_job_sha256": prov.get("job_sha256")})
    hub_dir = ctx.work / "hf" / "hub"
    cached = {}
    for k, repo in MODEL_REPOS.items():
        sd = hub_dir / ("models--" + repo.replace("/", "--")) / "snapshots"
        cached[k] = sorted(x.name for x in sd.iterdir()) if sd.is_dir() else []
    snaps = R.get("snapshots")
    C("snapshots: exactly policy, judge_small, judge_big, each the pin and the Hub cache's one snapshot directory",
      isinstance(snaps, dict) and sorted(snaps) == sorted(MODEL_REPOS) and snaps == revs
      and all(cached[k] == [snaps[k]] for k in MODEL_REPOS), {"snapshots": snaps, "pins": revs, "cache": cached})

    log, comps, probe, hist = R["log"], R["completions"], R["probe"], R["trl_history"]
    C("len(log) == len(completions) == STEPS", len(log) == len(comps) == STEPS, [len(log), len(comps)])
    C("log rows: keys exactly E1's 15 + gate_rate, js_std, jb_std",
      all(sorted(r) == sorted(LOG_KEYS) for r in log), [sorted(set(r) ^ set(LOG_KEYS)) for r in log])
    C("log rows: step = index, task, every value a float (trainer fields filled)",
      all(r["step"] == i and _is_int(r["step"]) and r["task"] == TASK
          and all(_is_float(r[k]) for k in LOG_KEYS if k not in ("step", "task")) for i, r in enumerate(log)),
      [{k: type(r[k]).__name__ for k in LOG_KEYS if not _is_float(r[k])} for r in log])
    C("completions: keys exactly section 4's", all(sorted(c) == sorted(COMP_KEYS) for c in comps))
    lens_ok = all(c["step"] == i and all(isinstance(c[k], list) and len(c[k]) == GEN for k in COMP_KEYS if k != "step")
                  for i, c in enumerate(comps))
    C("completions: step = index, 8 values per column", lens_ok)
    C("completions: column types (str, int, bool x3, float x6)",
      all(all(isinstance(x, str) for x in c["texts"]) and all(_is_int(x) for x in c["n_tokens"])
          and all(isinstance(x, bool) for k in ("eos", "fenced", "parses") for x in c[k])
          and all(_is_float(x) for k in ("G", "judge_small", "judge_big", "reward", "hidden", "hidden_lenient") for x in c[k])
          and all(x in (0.0, 1.0) for x in c["G"]) for c in comps))
    C("probe: one row per step (every %d), n = %d, keys step/n/probe_hidden/probe_sem" % (PROBE_EVERY, PROBE_N),
      [p["step"] for p in probe] == list(range(1, STEPS + 1)) and all(sorted(p) == ["n", "probe_hidden", "probe_sem", "step"]
                                                                      and p["n"] == PROBE_N for p in probe),
      [p.get("step") for p in probe])
    C("trl_history is a list of dicts", all(isinstance(h, dict) for h in hist), len(hist))

    # the reward call TRL made
    calls = [e for e in tr if e["event"] == "reward_call"]
    C("TRL called the reward function once per step", [e["step"] for e in calls] == list(range(STEPS)),
      [e["step"] for e in calls])
    C("the reward function received completion_ids: a list of %d lists of int" % GEN,
      all(e["ids_type"] == "list" and e["elem_types"] == ["list"] and e["int_ids"] and len(e["n_tokens"]) == GEN
          for e in calls), [(e["ids_type"], e["elem_types"], e["int_ids"]) for e in calls])
    if len(calls) == STEPS:
        C("texts are TRL's completions in full", all(c["texts"] == e["completions"] for c, e in zip(comps, calls)))
        C("n_tokens = len(completion_ids) per completion", all(c["n_tokens"] == e["n_tokens"] for c, e in zip(comps, calls)))
        C("eos = not TRL's clipped rule (ids[-1] in eos_token_ids + pad), per completion",
          all(c["eos"] == [not t for t in e["trl_truncated"]] for c, e in zip(comps, calls)),
          {"eos_token_ids": calls[0]["eos_token_ids"], "pad": calls[0]["pad"]})
        C("TRL received the logged reward (float32, 1e-6)",
          all(all(abs(a - b) <= 1e-6 for a, b in zip(c["reward"], e["rewards"])) for c, e in zip(comps, calls)))
    th = sorted((h for h in hist if "completions/clipped_ratio" in h), key=lambda h: h["step"])
    C("trl_history has one metrics row per step, at global steps 1..%d" % STEPS,
      [h["step"] for h in th] == list(range(1, STEPS + 1)), [h["step"] for h in th])
    eos_all = [x for c in comps for x in c["eos"]]
    ntok = [x for c in comps for x in c["n_tokens"]]
    ctx.info.setdefault("eos_mix", {})[case] = {"eos_true": sum(eos_all), "of": len(eos_all),
                                                "n_tokens_min": min(ntok), "n_tokens_max": max(ntok)}
    if len(th) == STEPS:
        C("1 - mean(eos) == trl_history completions/clipped_ratio, and == log.clipped",
          all(_close(1 - np.mean(c["eos"]), h["completions/clipped_ratio"]) and _close(r["clipped"], h["completions/clipped_ratio"])
              for c, h, r in zip(comps, th, log)),
          [[1 - float(np.mean(c["eos"])), h["completions/clipped_ratio"], r["clipped"]] for c, h, r in zip(comps, th, log)])
        C("mean(n_tokens) == trl_history completions/mean_length, and == log.mean_len",
          all(_close(np.mean(c["n_tokens"]), h["completions/mean_length"], 1e-4) and _close(r["mean_len"], h["completions/mean_length"])
              for c, h, r in zip(comps, th, log)))
        C("log entropy, kl, grad_norm equal trl_history's",
          all(all(_close(r[k], h.get(k)) for k in ("entropy", "kl", "grad_norm")) for r, h in zip(log, th)))
    if eos_mix:
        C("eosmix: both eos values occur, so the eos checks are not all at the cap",
          0 < sum(eos_all) < len(eos_all), ctx.info["eos_mix"][case])

    # reward and per-step means, against the completion columns
    bad_r, bad_s = [], []
    for i, (r, c) in enumerate(zip(log, comps)):
        js, jb, rw, g = (np.array(c[k], dtype=float) for k in ("judge_small", "judge_big", "reward", "G"))
        want_r = js if arm == "judge" else js * g
        if not np.all(np.abs(rw - want_r) <= 1e-6):
            bad_r.append(i)
        exp = {"judge_small": np.nanmean(js), "judge_big": np.nanmean(jb),
               "disagree": np.nanmean(js) - np.nanmean(jb), "proxy": np.mean(rw), "adv_std": np.std(rw),
               "hidden": np.mean(c["hidden"]), "hidden_lenient": np.mean(c["hidden_lenient"]),
               "delta": np.mean(rw) - np.mean(c["hidden"]), "gate_rate": np.mean(g), "js_std": np.std(js),
               "jb_std": np.std(jb)}
        bad_s += [[i, k, r[k], float(v)] for k, v in exp.items() if not _close(r[k], v)]
    C("reward == %s per completion (1e-6)" % ("judge_small" if arm == "judge" else "judge_small x G"), not bad_r, bad_r)
    C("log means and SDs (gate_rate, js_std, jb_std, proxy, adv_std, ...) agree with the columns", not bad_s, bad_s)
    C("judge_small and judge_big columns differ (two distinct judges)",
      any(a != b for c in comps for a, b in zip(c["judge_small"], c["judge_big"])))
    texts = [t for c in comps for t in c["texts"]]
    py, G_off = offline_gate([(t, TASK) for t in texts], V["python"])
    G_log = [x for c in comps for x in c["G"]]
    C("logged G == experiments/e1b_offline_gate.offline_gate (under Python %s)" % py, list(G_off) == G_log,
      {"python": py, "G_rate": float(np.mean(G_log))})
    fen = [_FENCE.search(t) is not None for t in texts]
    C("fenced == reward_testhack._FENCE.search(text) is not None", fen == [x for c in comps for x in c["fenced"]])
    ctx.info.setdefault("G_rate", {})[case] = float(np.mean(G_log))

    # status.json, from the recorded writes
    sts = [(e["seq"], json.loads(e["text"])) for e in tr if e["event"] == "status"]
    fp = [e for e in tr if e["event"] == "from_pretrained"]
    C("status.json written with step null, then once per step 0..%d" % (STEPS - 1),
      [s["step"] for _, s in sts] == [None] + list(range(STEPS)), [s["step"] for _, s in sts])
    C("every status.json: keys exactly run_id, step, rc, ts; rc null; run_id; ts UTC",
      all(sorted(s) == ["rc", "run_id", "step", "ts"] and s["rc"] is None and s["run_id"] == prov["run_id"]
          and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", s["ts"]) for _, s in sts))
    C("the step-null status.json comes before the first from_pretrained call",
      bool(sts) and bool(fp) and sts[0][0] < fp[0]["seq"])
    C("final status.json on disk: step %d, rc null" % (STEPS - 1),
      res["status"] is not None and res["status"]["step"] == STEPS - 1 and res["status"]["rc"] is None, res["status"])

    # from_pretrained and Hub reads
    pin = lambda e: ids_by_model.get(e["id"])   # noqa: E731
    bad_fp = [e for e in fp if e["revision"] != pin(e)]
    C("every from_pretrained call, at every depth, received revision = the pin",
      bool(fp) and not bad_fp, [[e["label"], e["cls"], e["id"], e["revision"], e["depth"], e["outer"]] for e in bad_fp])
    top = sorted((e["label"], e["id"]) for e in fp if e["depth"] == 0)
    want_top = sorted([("AutoTokenizer", TINY_BIG), ("AutoModelForCausalLM", TINY_BIG),        # judge_big
                       ("AutoTokenizer", TINY_POLICY), ("AutoModelForCausalLM", TINY_POLICY),  # judge_small
                       ("AutoModelForCausalLM", TINY_POLICY), ("AutoModelForCausalLM", TINY_POLICY),  # policy, ref
                       ("AutoProcessor", TINY_POLICY)])                                         # policy tokenizer
    C("top-level loads are exactly both judges (tokenizer + model), policy, reference model, policy processor",
      top == want_top, top)
    counts = {}
    for e in fp:
        k = "%s d%d" % (e["label"], e["depth"])
        counts[k] = counts.get(k, 0) + 1
    ctx.info.setdefault("from_pretrained_calls", {})[case] = counts
    C("no instrumentation target was missing", not [e for e in tr if e["event"] == "wrap_missing"],
      [e for e in tr if e["event"] == "wrap_missing"])
    # Hub reads. Every file read must be at the pin. A default-branch metadata read is allowed only of one known
    # kind: transformers' Mistral-tag check in the tokenizer loader (tokenization_utils_tokenizers.py,
    # hf_api().model_info(model_id), which takes no revision).
    hub = [e for e in tr if e["event"] == "hub"]
    row = lambda e: [e["fn"], e["repo"], e["filename"], e["revision"], (e["stack"][:1] or [[None]])[0][0]]   # noqa: E731
    unpinned = [e for e in hub if e["revision"] != ids_by_model.get(e["repo"])]
    files_bad = [e for e in unpinned if e["fn"] not in ("model_info", "repo_info")]
    meta_bad = [e for e in unpinned if e["fn"] in ("model_info", "repo_info") and not (
        e["fn"] == "model_info" and e["stack"] and e["stack"][0][0] in ("AutoTokenizer", "AutoProcessor"))]
    C("every Hub file read is at the pinned revision", bool(hub) and not files_bad, [row(e) for e in files_bad])
    C("the only default-branch Hub reads are the tokenizer loader's Mistral-tag model_info", not meta_bad,
      [row(e) for e in meta_bad])
    ctx.info.setdefault("hub_unpinned", {})[case] = [row(e) for e in unpinned]

    # what ran
    gl = [e for e in tr if e["event"] == "globals"]
    if gl:
        g = gl[0]
        C("printing callbacks removed (no PrinterCallback, no ProgressCallback); GeomCallback, EvalProbe kept",
          not {"PrinterCallback", "ProgressCallback"} & set(g["callbacks"]) and {"GeomCallback", "EvalProbe"} <= set(g["callbacks"]),
          g["callbacks"])
        C("cpu float32: policy, reference model and both judges; bf16 off",
          all(g[k] == ["cpu", "torch.float32"] for k in ("policy", "ref", "judge_small", "judge_big"))
          and g["bf16"] is False and g["use_cpu"] is True,
          {k: g[k] for k in ("policy", "ref", "judge_small", "judge_big", "bf16", "use_cpu")})
        ctx.info.setdefault("ids", {})[case] = {k: g[k] for k in ("eos_id", "pad_id", "trl_eos_token_ids",
                                                                  "policy_class", "ref_class", "generation_kwargs")}
        if eos_mix:
            C("eosmix: the biased ids are the tokenizer's eos and pad", {g["eos_id"], g["pad_id"]} == set(BIAS_IDS),
              [g["eos_id"], g["pad_id"]])


def check_stop(ctx, case, res, why, provenance_stop):
    C = lambda name, ok, detail=None: ctx.check(case, name, ok, detail)   # noqa: E731
    err = _lines(res["stderr"])
    stops = [x for x in err if x.startswith(STOP)]
    C("exit code 1", res["rc"] == 1, res["rc"])
    C("stderr: %s%s..." % (STOP, why[:40]), len(stops) == 1 and stops[0].startswith(STOP + why), stops)
    C("stdout is empty (no STEP line)", res["stdout"] == b"", _lines(res["stdout"]))
    C("no result.json", res["result"] is None)
    sts = [json.loads(e["text"]) for e in res["trace"] if e["event"] == "status"]
    fp = [e for e in res["trace"] if e["event"] == "from_pretrained"]
    if provenance_stop:
        C("stopped before any status.json write and any from_pretrained call",
          not sts and res["status"] is None and not fp, {"status_writes": len(sts), "from_pretrained": len(fp)})
    else:
        ld = sorted((e["label"], e["id"]) for e in res["trace"] if e["event"] == "loaded" and e["depth"] == 0
                    and e["label"] == "AutoModelForCausalLM")
        C("stopped after both judges, the policy and the reference model loaded, with only the step-null status",
          [s["step"] for s in sts] == [None] and ld == sorted([("AutoModelForCausalLM", TINY_BIG)]
                                                             + [("AutoModelForCausalLM", TINY_POLICY)] * 3),
          {"status": sts, "loaded": ld})


def check_provenance(ctx, J, baked, files):
    """The job's embedded package, lock and gate against bake()'s values and the working tree."""
    for arm, (jp, prov) in baked.items():
        case = "bake_" + arm
        src = jp.read_text(encoding="ascii")
        tar = embedded_tar(src)
        with tarfile.open(fileobj=io.BytesIO(tar), mode="r:") as t:
            gate = t.extractfile(J.GATE_PATH).read()
            names = sorted(t.getnames())
        wt = {rel: (REPO / rel).read_bytes().replace(b"\r\n", b"\n") for rel in files}
        ctx.check(case, "package_sha256 = SHA-256 of the embedded, decompressed package", _sha(tar) == prov["package_sha256"])
        ctx.check(case, "gate_sha256 = embedded twin_gate.py = working-tree twin_gate.py",
                  _sha(gate) == prov["gate_sha256"] == _sha(wt[J.GATE_PATH]))
        ctx.check(case, "lock_sha256 = job.py.lock = working-tree experiments/e1b_uv.lock",
                  _sha((jp.parent / "job.py.lock").read_bytes()) == prov["lock_sha256"] == _sha(wt[J.LOCK_PATH]))
        ctx.check(case, "template_sha256 = working-tree scripts/e1b_job.py", prov["template_sha256"] == _sha(wt[J.TEMPLATE_PATH]))
        ctx.check(case, "provenance keys are exactly build_contract section 6's", sorted(prov) == sorted(PROV_KEYS),
                  {"extra": sorted(set(prov) - set(PROV_KEYS)), "missing": sorted(set(PROV_KEYS) - set(prov))})
        ctx.check(case, "job_sha256 = SHA-256 of the job.py bytes bake() wrote", prov.get("job_sha256") == _sha(jp.read_bytes()),
                  {"job_sha256": prov.get("job_sha256"), "file": _sha(jp.read_bytes())})
        ctx.check(case, "job is LF and ASCII", b"\r" not in jp.read_bytes() and max(jp.read_bytes()) < 128)
        ctx.check(case, "every package import-closure file matches the working tree",
                  all(t_ok for t_ok in [wt[rel] == _member(tar, rel) for rel in J._REQUIRED]))
        ctx.info.setdefault("package_files", {})[arm] = len(names)


def check_arch(ctx, rev):
    """The job's architecture keyword against the bare revision, on the tiny policy (case arch)."""
    d = Path(_short(ctx.work / "runs")) / "arch"
    # its own HF_HOME: the bare-revision build resolves main and would leave refs/main in the jobs' Hub cache
    env = dict(os.environ, HF_HOME=ctx.hf_home + "_arch", TEMP=str(ctx.short / "tmp"), TMP=str(ctx.short / "tmp"))
    env.pop("HF_TOKEN", None)
    r = subprocess.run([str(ctx.vpy), str(Path(__file__).resolve()), "--compare-arch", TINY_POLICY, rev, str(d)],
                       capture_output=True, text=True, env=env, timeout=1800)
    if not ctx.check("arch", "--compare-arch ran", r.returncode == 0, r.stderr[-1500:] if r.returncode else None):
        return
    out = json.loads(r.stdout.strip().splitlines()[-1])
    same = out["same"]
    ctx.check("arch", "policy and reference model: same class, weights, config and generation config both ways",
              all(same[n + k] is True for n in ("model", "ref_model") for k in ("_weights", "_config", "_generation_config"))
              and all(len(set(same[n + "_class"])) == 1 for n in ("model", "ref_model"))
              and same["eos_token_ids"][0] == same["eos_token_ids"][1], same)
    bare = [c for c in out["calls_old"] if c[2] != rev]
    ctx.check("arch", "the bare revision leaves from_pretrained calls without it (TRL's AutoConfig lookup)",
              bool(bare) and all(c[0] == "AutoConfig" for c in bare), bare)
    ctx.check("arch", "with the architecture keyword every from_pretrained call has the revision",
              bool(out["calls_new"]) and all(c[2] == rev for c in out["calls_new"]), out["calls_new"])


def check_matched(ctx, kind):
    """Case matched_<kind>: the two arms' runs of one kind share seed 299, and nothing before the first generation
    depends on ARM, so their step-0 completions must be identical. A missing result fails the check."""
    got = {}
    for arm in ARMS:
        p = ctx.work / "runs" / ("%s_%s" % (kind, arm)) / "result.json"
        got[arm] = json.loads(p.read_text(encoding="ascii")) if p.is_file() else None
    a, b = got["judge"], got["judgegate"]
    both = bool(a and b and a.get("completions") and b.get("completions"))
    same = {k: both and a["completions"][0].get(k) == b["completions"][0].get(k)
            for k in ("texts", "n_tokens", "eos", "judge_small", "judge_big", "G")}
    ctx.info.setdefault("matched_seed_step0_identical_texts", {})[kind] = same["texts"] if both else None
    ctx.info.setdefault("matched_seed_step0_identical_scores", {})[kind] = (
        {k: same[k] for k in ("judge_small", "judge_big", "G")} if both else None)
    ctx.check("matched_" + kind, "judge and judgegate step-0 completions identical at shared seed %d (texts, n_tokens, "
              "eos)" % SEED, both and same["texts"] and same["n_tokens"] and same["eos"], {"both_results": both, "same": same})


def check_doc(ctx, J):
    """Case doc: the module docstring documents the instance commands as the launcher runs them."""
    doc = J.__doc__ or ""
    pinned = "uv==%s --hash=sha256:%s" % (J.UV_VERSION, J.UV_WHEEL_SHA256)
    bare = [x.strip() for x in doc.splitlines() if re.search(r"pip install\b.*\buv==", x)]
    ctx.check("doc", "the documented uv install is hash-pinned: a requirements line '%s' installed with "
              "--require-hashes -r, and no pip install of a bare uv==" % pinned[:16],
              pinned in doc and re.search(r"pip install --only-binary :all: --require-hashes -r \S+", doc) is not None
              and not bare, {"pip_install_with_bare_uv": bare})
    run = "python3 -m " + " ".join(J.RUN_ARGS)
    ctx.check("doc", "the documented run command is python3 -m RUN_ARGS", run in doc, run)


def _member(tar, name):
    with tarfile.open(fileobj=io.BytesIO(tar), mode="r:") as t:
        return t.extractfile(name).read()


def main(argv):
    if argv[:1] == ["--run-job"] and len(argv) == 3:
        run_job(argv[1], argv[2])
        return 0
    if argv[:1] == ["--compare-arch"] and len(argv) == 4:
        compare_arch(*argv[1:])
        return 0
    if len(argv) != 1:
        print(__doc__.strip().splitlines()[0])
        print("usage: python experiments/e1b_design/job_smoke/smoke.py WORK")
        return 2
    work = Path(argv[0]).resolve()
    if work == REPO or REPO in work.parents:
        print("WORK must lie outside the repository")
        return 2
    ctx = Ctx(work)
    import tomllib
    # experiments/e1b_offline_gate.py runs `uv run --python 3.12.15 --managed-python --no-python-downloads` (the
    # run's full logged version; no other patch release) from PATH: this uv, this cache, these Pythons
    os.environ.update(PATH=str(ctx.uv.parent) + os.pathsep + os.environ["PATH"], UV_NO_CONFIG="1",
                      UV_CACHE_DIR=str(ctx.short / "uvc"), UV_PYTHON_INSTALL_DIR=str(ctx.short / "py"))
    wt = _load("e1b_job_wt", REPO / "scripts" / "e1b_job.py")
    setup_uv(ctx, wt.UV_VERSION)
    J, baked, files = clone_and_bake(ctx)
    setup_env(ctx, J, baked["judge"][0])
    lock_versions = {p["name"]: p["version"] for p in tomllib.loads((REPO / J.LOCK_PATH).read_text("utf-8"))["package"]}
    revs = {"policy": hub_sha(TINY_POLICY), "judge_big": hub_sha(TINY_BIG)}
    revs["judge_small"] = revs["policy"]
    ids_by_model = {TINY_POLICY: revs["policy"], TINY_BIG: revs["judge_big"]}
    ctx.info["models"] = {"policy": TINY_POLICY, "judge_small": TINY_POLICY, "judge_big": TINY_BIG, "revisions": revs}
    check_provenance(ctx, J, baked, files)

    patches = {}
    for arm in ARMS:
        jp, prov = baked[arm]
        base = jp.read_text(encoding="ascii")
        lock = (jp.parent / "job.py.lock").read_bytes()
        for kind, mix in (("main", False), ("eosmix", True)):
            case = "%s_%s" % (kind, arm)
            src, subs = cpu_patch(base, Path(_short(ctx.work / "runs")) / case, revs, eos_bias=mix)
            patches[case] = subs
            res = run_case(ctx, case, src, lock)
            ctx.info.setdefault("seconds", {})[case] = res["seconds"]
            check_training(ctx, case, arm, res, prov, revs, ids_by_model, lock_versions, mix)
    for kind in ("main", "eosmix"):
        check_matched(ctx, kind)

    check_arch(ctx, revs["policy"])

    # stop cases, each on top of the CPU patches so that a check that failed to stop could only run tiny models
    jp, prov = baked["judgegate"]
    lock = (jp.parent / "job.py.lock").read_bytes()
    base = jp.read_text(encoding="ascii")
    sp = lambda case, **kw: cpu_patch(base, Path(_short(ctx.work / "runs")) / case, revs, **kw)[0]   # noqa: E731
    src = sp("stop_package")
    res = run_case(ctx, "stop_package", with_tar(src, flip_case_in_member(embedded_tar(src), "flightrecorder/repro/reward_judge.py")), lock, 600)
    check_stop(ctx, "stop_package", res, "the embedded package's SHA-256 differs from package_sha256", True)
    src = sp("stop_gate")
    tar = flip_case_in_member(embedded_tar(src), J.GATE_PATH)
    src = _sub(with_tar(src, tar), '"%s"' % prov["package_sha256"], '"%s"' % _sha(tar))
    res = run_case(ctx, "stop_gate", src, lock, 600)
    check_stop(ctx, "stop_gate", res, "the extracted twin_gate.py's SHA-256 differs from gate_sha256", True)
    res = run_case(ctx, "stop_lock", sp("stop_lock"), lock + b"\n", 600)
    check_stop(ctx, "stop_lock", res, "job.py.lock's SHA-256 differs from lock_sha256", True)
    res = run_case(ctx, "stop_nolock", sp("stop_nolock"), None, 600)
    check_stop(ctx, "stop_nolock", res, "no job.py.lock next to the job", True)
    res = run_case(ctx, "eosguard", sp("eosguard", eos_guard=False), lock, 1800)
    check_stop(ctx, "eosguard", res, "TRL's eos ids [151645, 151643] are not [tokenizer eos 151645]", False)
    # judge_big loads at the model's real revision (so the shared Hub cache gains no snapshot and no ref), while
    # REVISIONS pins another commit: the snapshot the job reads from the cache then differs from the pin.
    real = revs["judge_big"]
    other = real[:-1] + ("1" if real[-1] == "0" else "0")
    src = _sub(sp("stop_snapshot"), 'revision=REVISIONS["judge_big"])', 'revision="%s")' % real)
    src = _sub(src, '"judge_big": "%s"' % real, '"judge_big": "%s"' % other)
    res = run_case(ctx, "stop_snapshot", src, lock, 1800)
    check_stop(ctx, "stop_snapshot", res, "the Hub snapshot judge_big was loaded from is %s, not its pinned revision %s"
               % (real, other), False)
    check_doc(ctx, J)

    # The pins equal main here, so snapshots cannot tell a pinned read from an unpinned one, but a file read
    # that resolves the default branch leaves refs/main behind (the job as first built left one).
    hub = ctx.work / "hf" / "hub"
    ctx.info["hub_cache"] = {p.name: {"snapshots": sorted(x.name for x in (p / "snapshots").iterdir()),
                                      "refs": sorted(x.name for x in (p / "refs").iterdir()) if (p / "refs").is_dir() else []}
                             for p in sorted(hub.glob("models--*"))}
    ctx.check("hub_cache", "only the pinned snapshots, and no refs (no file read resolved a branch)",
              ctx.info["hub_cache"] and all(v["snapshots"] == [ids_by_model[k.split("--", 1)[1].replace("--", "/")]]
                                            and not v["refs"] for k, v in ctx.info["hub_cache"].items()),
              ctx.info["hub_cache"])
    ctx.info["patches"] = {k: [[o, n.replace(str(_short(ctx.work)).replace("\\", "/"), "<WORK>")] for o, n in v]
                           for k, v in patches.items()}
    n_ok = sum(r["ok"] for r in ctx.rows)
    report = {"ok": n_ok == len(ctx.rows), "n_checks": len(ctx.rows), "n_ok": n_ok, "rows": ctx.rows, "info": ctx.info}
    with open(ctx.work / "report.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=1, default=str)
    for r in ctx.rows:
        print("%-4s %-18s %s" % ("OK" if r["ok"] else "FAIL", r["case"], r["check"]))
    print("%d of %d checks OK; report: %s" % (n_ok, len(ctx.rows), ctx.work / "report.json"))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
