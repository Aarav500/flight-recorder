"""Multi-task benign GRPO batch: 3 tasks x 6 seeds = 18 runs, identical config to
gentle_seed0 (base Qwen2.5-1.5B, lr 1e-6, 120 steps), varying only task + seed. Each
run's per-step geometry artifact is persisted to an HF dataset immediately, so a timeout
never loses completed seeds. One HF Job per task (seeds looped inside). ASCII only."""
from __future__ import annotations

import base64
import io
import tarfile
from pathlib import Path

from huggingface_hub import get_token, run_uv_job

import sys
ROOT = Path(__file__).resolve().parents[1]
TASKS = ["complex", "maxrun", "secondmax"]   # lus2 anchor + two new mid-difficulty tasks
# argv: model algo tag seeds(comma) lr   (defaults reproduce the original Qwen/GRPO batch)
MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B"
ALGO = sys.argv[2] if len(sys.argv) > 2 else "grpo"
TAG = sys.argv[3] if len(sys.argv) > 3 else "qwen_grpo"
SEED_LIST = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [0, 1]
LR = sys.argv[5] if len(sys.argv) > 5 else "1e-6"
REPO = "Aarav500/fr-benign-batch"
print(f"model={MODEL} algo={ALGO} tag={TAG} seeds={SEED_LIST} lr={LR}")

buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as t:
    t.add(ROOT / "flightrecorder", arcname="flightrecorder")
    t.add(ROOT / "configs", arcname="configs")
B64 = base64.b64encode(buf.getvalue()).decode()
print(f"baked package+configs: {len(B64)} b64 chars")

TEMPLATE = '''# /// script
# dependencies = [
#   "trl>=0.15","transformers>=4.47","datasets>=3.2","accelerate>=1.2",
#   "torch>=2.6","pyyaml>=6.0","numpy>=1.26","scipy>=1.12","huggingface-hub>=0.25",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
#
# [tool.uv.sources]
# torch = [{ index = "pytorch-cu128" }]
# ///
import base64, io, os, sys, glob, tarfile

_PKG_B64 = "__PKG_B64__"
tarfile.open(fileobj=io.BytesIO(base64.b64decode(_PKG_B64)), mode="r:gz").extractall("/tmp/fr")
sys.path.insert(0, "/tmp/fr")

from flightrecorder.repro.trl_grpo_qwen import main  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402

TASK = "__TASK__"
SEED_LIST = __SEED_LIST__
REPO = "__REPO__"
MODEL = "__MODEL__"
ALGO = "__ALGO__"
TAG = "__TAG__"
LR = "__LR__"
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(REPO, repo_type="dataset", exist_ok=True)
os.makedirs("/tmp/fr/runs", exist_ok=True)

for seed in SEED_LIST:
    for f in glob.glob("/tmp/fr/runs/*.jsonl"):
        os.remove(f)
    print(f"=== RUN model={MODEL} algo={ALGO} task={TASK} seed={seed} ===", flush=True)
    try:
        rc = main(["--scale", "benign", "--task", TASK, "--model", MODEL, "--algo", ALGO,
                   "--lr", LR, "--steps", "120",
                   "--integrity-report", "--thresholds", "/tmp/fr/configs/thresholds.yaml",
                   "--seed", str(seed), "--artifact-dir", "/tmp/fr/runs"])
    except Exception as e:
        print(f"RUN FAILED task={TASK} seed={seed}: {e}", flush=True)
        continue
    arts = sorted(glob.glob("/tmp/fr/runs/*.jsonl"), key=os.path.getmtime)
    if arts:
        dest = f"{TAG}_{TASK}_seed{seed}.jsonl"
        api.upload_file(path_or_fileobj=arts[-1], path_in_repo=dest, repo_id=REPO, repo_type="dataset")
        print(f"PERSISTED {dest} (rc={rc})", flush=True)
    else:
        print(f"NO ARTIFACT task={TASK} seed={seed} (rc={rc})", flush=True)

print(f"BATCH_TASK_DONE {TASK}", flush=True)
'''

jobs = {}
for task in TASKS:
    script = (TEMPLATE.replace("__PKG_B64__", B64).replace("__TASK__", task)
              .replace("__SEED_LIST__", repr(SEED_LIST)).replace("__REPO__", REPO)
              .replace("__MODEL__", MODEL).replace("__ALGO__", ALGO)
              .replace("__TAG__", TAG).replace("__LR__", LR))
    out_path = ROOT / "scripts" / f"_batch_{TAG}_{task}.py"
    out_path.write_text(script, encoding="utf-8")
    job = run_uv_job(str(out_path), flavor="a100-large", timeout="5h",
                     secrets={"HF_TOKEN": get_token()})
    jid = getattr(job, "id", str(job))
    jobs[task] = jid
    print(f"FIRED task={task} job={jid}")

print("\nJOBS:", jobs)
Path(ROOT / "scripts" / "_batch_jobs.txt").write_text(
    "\n".join(f"{t}\t{j}" for t, j in jobs.items()), encoding="utf-8")
