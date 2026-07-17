"""Real reward-hack runs for the direct comparison: GRPO on the gameable OVERFIT_TASK
(broken visible verifier), reward_mode=gameable, NO curriculum, NO seed-injection -- the
policy games the verifier on its own (train/visible reward rises, held-out oracle drops).
Same base model / lr / steps as the benign batch, so geometry is directly comparable. One
a100 job, seeds 0-2, incremental persist. ASCII only."""
from __future__ import annotations

import base64
import io
import sys
import tarfile
from pathlib import Path

from huggingface_hub import get_token, run_uv_job

ROOT = Path(__file__).resolve().parents[1]
LR = sys.argv[1] if len(sys.argv) > 1 else "1e-6"   # match benign batch by default
SEEDS = [0, 1, 2]
REPO = "Aarav500/fr-benign-batch"

buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as t:
    t.add(ROOT / "flightrecorder", arcname="flightrecorder")
    t.add(ROOT / "configs", arcname="configs")
B64 = base64.b64encode(buf.getvalue()).decode()
print(f"baked: {len(B64)} b64 chars  lr={LR}")

UV = '''# /// script
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

SEEDS = __SEEDS__
LR = "__LR__"
REPO = "__REPO__"
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(REPO, repo_type="dataset", exist_ok=True)
os.makedirs("/tmp/fr/runs", exist_ok=True)

for seed in SEEDS:
    for f in glob.glob("/tmp/fr/runs/*.jsonl"):
        os.remove(f)
    print(f"=== HACK RUN task=overfit gameable seed={seed} lr={LR} ===", flush=True)
    try:
        rc = main(["--scale", "benign", "--task", "overfit", "--reward-mode", "gameable",
                   "--lr", LR, "--steps", "120",
                   "--integrity-report", "--thresholds", "/tmp/fr/configs/thresholds.yaml",
                   "--seed", str(seed), "--artifact-dir", "/tmp/fr/runs"])
    except Exception as e:
        print(f"RUN FAILED seed={seed}: {e}", flush=True)
        continue
    arts = sorted(glob.glob("/tmp/fr/runs/*.jsonl"), key=os.path.getmtime)
    if arts:
        dest = f"hack_seed{seed}.jsonl"
        api.upload_file(path_or_fileobj=arts[-1], path_in_repo=dest, repo_id=REPO, repo_type="dataset")
        print(f"PERSISTED {dest} (rc={rc})", flush=True)
    else:
        print(f"NO ARTIFACT seed={seed} (rc={rc})", flush=True)

print("HACK_BATCH_DONE", flush=True)
'''

script = (UV.replace("__PKG_B64__", B64).replace("__SEEDS__", repr(SEEDS))
          .replace("__LR__", LR).replace("__REPO__", REPO))
out_path = ROOT / "scripts" / "_hack_run.py"
out_path.write_text(script, encoding="utf-8")
job = run_uv_job(str(out_path), flavor="a100-large", timeout="3h", secrets={"HF_TOKEN": get_token()})
print("JOB_ID:", getattr(job, "id", job))
