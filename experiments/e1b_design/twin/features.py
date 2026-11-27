"""E1b twin reward: per-completion features on E1's stored completions (offline, CPU, read-only).

Reads runs/e1/main/*.json (never writes there). For every stored completion it records the
static properties the candidate gates use, plus the strict and lenient oracle values the job
stored. Output: features.jsonl.gz next to this file (one row per completion).

Token counts: E1 did not store completion_ids, so the completion is re-encoded with the cached
Qwen2.5 tokenizer (0.5B-Instruct snapshot 7ae5576; the Qwen2.5 family shares one vocabulary).
Re-encoding decoded text can differ from the generated ids by a few tokens, so the truncation
flag is checked against the job's logged clipped ratio (completions/clipped_ratio) in report.py.
Texts were stored cut at 1500 chars; those are flagged `cut1500`.
"""
from __future__ import annotations

import ast
import gzip
import json
import re
import sys
from pathlib import Path

REPO = Path(r"<home>/OneDrive/Desktop/Fligh recorder")
sys.path.insert(0, str(REPO))
from flightrecorder.repro.reward_testhack import _FENCE, extract_code  # noqa: E402

from tokenizers import Tokenizer  # noqa: E402

TOK = Tokenizer.from_file(str(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct"
                              / "snapshots/7ae557604adf67be50417f59c2c2f167def9a775/tokenizer.json"))
FN = {"count_vowels": "count_vowels", "is_prime": "is_prime", "second_largest": "second_largest"}
OUT = Path(__file__).resolve().parent / "features.jsonl.gz"


def parse_info(code: str, fn: str) -> dict:
    """Static facts about the extracted code. Never executes it."""
    try:
        tree = ast.parse(code)
    except Exception:          # SyntaxError, and MemoryError/RecursionError on pathological input
        return {"parses": False, "defines": False, "defines_def": False, "trivial_body": None,
                "toplevel_exec": None,
                "empty_code": not code.strip()}
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn]
    lam = [n for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Lambda)
           and any(isinstance(t, ast.Name) and t.id == fn for t in n.targets)]
    trivial = None
    if defs:
        body = [b for b in defs[-1].body
                if not (isinstance(b, ast.Expr) and isinstance(getattr(b, "value", None), ast.Constant)
                        and isinstance(b.value.value, str))]           # drop the docstring
        trivial = (not body) or all(isinstance(b, ast.Pass) or
                                    (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant)
                                     and b.value.value is Ellipsis) for b in body)
        # a body that only returns a constant (return 0 / return None / return True)
        if not trivial and len(body) == 1 and isinstance(body[0], ast.Return) and \
                (body[0].value is None or isinstance(body[0].value, ast.Constant)):
            trivial = True
    ok_top = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)
    toplevel_exec = any(not isinstance(n, ok_top) and not (
        isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)) for n in tree.body)
    if not defs and lam:       # `fn = lambda ...`: the strict oracle accepts it, so the gate must too
        b = lam[-1].value.body
        trivial = isinstance(b, ast.Constant)
    return {"parses": True, "defines": bool(defs) or bool(lam), "defines_def": bool(defs),
            "trivial_body": trivial,
            "toplevel_exec": toplevel_exec, "empty_code": not code.strip()}


def outside_text(text: str) -> str:
    """Text outside the first closed fenced block ('' if there is none outside)."""
    m = _FENCE.search(text)
    if not m:
        return text
    return text[:m.start()] + text[m.end():]


def main():
    files = sorted((REPO / "runs/e1/main").glob("judge_judge*_s2*_lr1e-6.json"))
    assert len(files) == 48, len(files)
    n = 0
    with gzip.open(OUT, "wt", encoding="utf-8") as fo:
        for f in files:
            d = json.loads(f.read_text())
            arm = "hack" if d["judge"].endswith("0.5B-Instruct") else "benign"
            task, seed, fn = d["task"], d["seed"], FN[d["task"]]
            assert d["maxtok"] == 256 and d["steps"] == 400
            clipped = {r["step"]: r.get("clipped") for r in d["log"]}
            for c in d["completions"]:
                texts = c["texts"]
                enc = TOK.encode_batch(texts, add_special_tokens=False)
                for i, t in enumerate(texts):
                    code = extract_code(t)
                    m = _FENCE.search(t)
                    pi = parse_info(code, fn)
                    outside = outside_text(t) if m else None
                    row = {"arm": arm, "task": task, "seed": seed, "step": c["step"], "i": i,
                           "strict": c["hidden"][i], "lenient": c["hidden_lenient"][i],
                           "fence_closed": m is not None, "fence_any": "```" in t,
                           "ntok": len(enc[i].ids), "nchar": len(t), "cut1500": len(t) >= 1500,
                           "outside_nonws": (outside is not None and bool(outside.strip())),
                           "outside_len": (len(outside.strip()) if outside is not None else None),
                           "starts_space_def": t.lstrip("\n").startswith(" def ") or t.startswith(" def "),
                           "step_clipped": clipped.get(c["step"]), **pi}
                    fo.write(json.dumps(row) + "\n")
                    n += 1
            print(f.name, arm, task, seed, flush=True)
    print("rows", n)


if __name__ == "__main__":
    main()
