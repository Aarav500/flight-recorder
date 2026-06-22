"""Benign-task calibration that MATCHES the training path: samples each task's actual
instruction prompt and scores the RAW completion with train_reward (exactly as
make_reward_fns does). Clearing here predicts the GRPO run will have gradient (unlike the
code-completion calibration, which mispredicted for base FIM models like starcoder).
argv[1] = model id. ~$0.3 on l4x1."""
from __future__ import annotations

import base64
import io
import sys
import tarfile
from pathlib import Path

from huggingface_hub import get_token, run_uv_job

ROOT = Path(__file__).resolve().parents[1]
MODEL = sys.argv[1] if len(sys.argv) > 1 else "HuggingFaceTB/SmolLM2-1.7B"

buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as t:
    t.add(ROOT / "flightrecorder", arcname="flightrecorder")
B64 = base64.b64encode(buf.getvalue()).decode()
print(f"baked: {len(B64)} b64 chars  model={MODEL}")

UV = '''# /// script
# dependencies = ["torch>=2.6","transformers>=4.47","numpy>=1.26"]
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# [tool.uv.sources]
# torch = [{ index = "pytorch-cu128" }]
# ///
import base64, io, sys, tarfile
_PKG_B64 = "__PKG_B64__"
tarfile.open(fileobj=io.BytesIO(base64.b64decode(_PKG_B64)), mode="r:gz").extractall("/tmp/fr")
sys.path.insert(0, "/tmp/fr")

from flightrecorder.repro.reward_benign import COMPLEX_TASK, MAXRUN_TASK, SECONDMAX_TASK  # noqa: E402
from flightrecorder.repro.reward_testhack import train_reward  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

MODEL = "__MODEL__"
GROUPS, G, TEMP, MAXTOK = 6, 8, 1.0, 256
TASKS = [("complex", COMPLEX_TASK), ("maxrun", MAXRUN_TASK), ("secondmax", SECONDMAX_TASK)]

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to("cuda").eval()


@torch.no_grad()
def gen(prompt, n):
    enc = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**enc, do_sample=True, temperature=TEMP, top_p=1.0,
                         max_new_tokens=MAXTOK, num_return_sequences=n, pad_token_id=tok.pad_token_id)
    return tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)

print(f"BENIGN-CALIB model={MODEL} (instruction prompt + raw scoring = training path)", flush=True)
for name, task in TASKS:
    all_r, stds = [], []
    for grp in range(GROUPS):
        rs = [float(train_reward(c, task)) for c in gen(task.prompt, G)]
        all_r += rs; stds.append(float(np.std(rs)))
    a = np.array(all_r); s = np.array(stds)
    clears = (s > 0).mean() >= 0.5 and 0.02 < a.mean() < 0.97
    print(f"TASK {name}: mean={a.mean():.3f} groups_with_gradient={int((s>0).sum())}/{GROUPS} "
          f"-> {'CLEARS' if clears else 'DOES NOT CLEAR'}", flush=True)
'''
out = ROOT / "scripts" / "_benign_calib.py"
out.write_text(UV.replace("__PKG_B64__", B64).replace("__MODEL__", MODEL), encoding="utf-8")
job = run_uv_job(str(out), flavor="l4x1", timeout="30m", secrets={"HF_TOKEN": get_token()})
print("JOB_ID:", getattr(job, "id", job))
