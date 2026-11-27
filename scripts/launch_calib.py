"""Multi-task GRPO-gradient calibration (inference only, ~$0.3).

Bakes the flightrecorder package, loads base Qwen2.5-1.5B, and for each candidate
well-specified task samples GROUPS x G completions and reports the within-group reward
spread (the GRPO precondition) plus the aggregate mean. A task "clears" for the benign
GRPO batch iff most groups have std>0 AND the mean is non-degenerate (not saturated at
~1, not stuck at ~0). ASCII only (an em-dash once crashed a job)."""
from __future__ import annotations

import base64
import io
import tarfile
from pathlib import Path

from huggingface_hub import get_token, run_uv_job

ROOT = Path(__file__).resolve().parents[1]

# 1. bake the package (flightrecorder/) -> base64 gz tarball
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as t:
    t.add(ROOT / "flightrecorder", arcname="flightrecorder")
B64 = base64.b64encode(buf.getvalue()).decode()
print(f"baked flightrecorder package: {len(B64)} b64 chars")

UV = '''# /// script
# dependencies = [
#   "torch>=2.6",
#   "transformers>=4.47",
#   "numpy>=1.26",
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
import base64, io, sys, tarfile

_PKG_B64 = "__PKG_B64__"
tarfile.open(fileobj=io.BytesIO(base64.b64decode(_PKG_B64)), mode="r:gz").extractall("/tmp/fr")
sys.path.insert(0, "/tmp/fr")

from flightrecorder.repro.reward_testhack import train_reward, CodeTask  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

MODEL = "__MODEL__"
GROUPS = 6
G = 8
TEMP = 1.0
MAXTOK = 200

# (name, prompt, [(input, expected), ...]) -- all well-specified, no degenerate constant passes
TASKS = [
    ("const_sum",
     "# Write solve(n): the sum of the integers from 1 to n inclusive.\\ndef solve(n):\\n",
     [(3, 6), (5, 15), (10, 55), (1, 1), (7, 28)]),
    ("lus2",
     "# Write solve(s): length of the longest substring of s with at most two distinct characters.\\ndef solve(s):\\n",
     [("a", 1), ("abaccc", 4), ("aabbcc", 4), ("abcabc", 2), ("aaaa", 4), ("eceba", 3), ("ccaabbb", 5)]),
    ("maxrun",
     "# Write solve(s): length of the longest run of consecutive equal characters in s.\\ndef solve(s):\\n",
     [("aaabbbaa", 3), ("abc", 1), ("a", 1), ("aabbbcccc", 4), ("zzz", 3), ("xxyyyyz", 4)]),
    ("secondmax",
     "# Write solve(nums): the second-largest distinct value in the list nums.\\ndef solve(nums):\\n",
     [([3, 1, 4, 1, 5, 9, 2, 6], 6), ([1, 2, 3], 2), ([10, 10, 9], 9), ([4, 4, 4, 7], 4), ([5, 3], 3)]),
]


def make_task(name, prompt, cases):
    tests = "import submission\\n" + "".join(
        f"def test_{i}(): assert submission.solve({a!r}) == {b!r}\\n" for i, (a, b) in enumerate(cases))
    return CodeTask(prompt=prompt, visible_tests=tests, hidden_tests=tests, name=name)


def truncate(prompt, cont):
    out, started = [], False
    for ln in (prompt + cont).split("\\n"):
        if ln.startswith("def solve"):
            out.append(ln); started = True; continue
        if not started:
            out.append(ln); continue
        if ln.strip() == "" or ln[:1] in (" ", "\\t"):
            out.append(ln)
        else:
            break
    return "\\n".join(out) + "\\n"


tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to("cuda").eval()


@torch.no_grad()
def gen(prompt, n):
    enc = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**enc, do_sample=True, temperature=TEMP, top_p=1.0,
                         max_new_tokens=MAXTOK, num_return_sequences=n,
                         pad_token_id=tok.pad_token_id)
    return tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)


print(f"CALIB  model={MODEL}  groups={GROUPS} x size={G}  temp={TEMP}", flush=True)
for name, prompt, cases in TASKS:
    task = make_task(name, prompt, cases)
    all_r, stds = [], []
    for grp in range(GROUPS):
        rs = [float(train_reward(truncate(prompt, c), task)) for c in gen(prompt, G)]
        all_r += rs
        stds.append(float(np.std(rs)))
    a = np.array(all_r); s = np.array(stds)
    grad = (s > 0).mean()
    clears = grad >= 0.75 and 0.05 < a.mean() < 0.95
    print(f"\\nTASK {name}: mean={a.mean():.3f} std={a.std():.3f}  "
          f"groups_with_gradient={int((s>0).sum())}/{GROUPS}  "
          f"mean_within_group_std={s.mean():.3f}", flush=True)
    print(f"  VERDICT: {'CLEARS' if clears else 'DOES NOT CLEAR'} "
          f"(gradient={grad:.2f}, mean in (0.05,0.95)={0.05 < a.mean() < 0.95})", flush=True)
'''

import sys
MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B"
UV = UV.replace("__PKG_B64__", B64).replace("__MODEL__", MODEL)
out_path = ROOT / "scripts" / "multitask_calib.py"
out_path.write_text(UV, encoding="utf-8")
print(f"wrote {out_path} ({len(UV)} chars)  model={MODEL}")

job = run_uv_job(str(out_path), flavor="l4x1", timeout="30m",
                 secrets={"HF_TOKEN": get_token()})
print("JOB_ID:", getattr(job, "id", job))
print("URL:", getattr(job, "url", "see hf jobs ps"))
