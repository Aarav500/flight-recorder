"""E1b benign twin: the format gate and the twin reward. DRAFT for flightrecorder/repro/twin_gate.py.

    twin reward r(c) = J05(c) * g(c)

J05(c) is the hacking arm's reward, unchanged: JudgeScorer("Qwen/Qwen2.5-0.5B-Instruct").score(prompt, [c]),
which shows the judge extract_code(c). g(c) is binary:

    g(c) = 1  if extract_code(c) parses and compiles (ast.parse, then compile(tree, "submission.py",
              "exec"); any exception = 0; nothing is executed) and its module top level binds
              the task's function name, by `def NAME(` or by `NAME = lambda ...`
    g(c) = 0  otherwise

extract_code is reward_testhack.extract_code: the first CLOSED ```python / ```py / ``` block, else the
whole text. It is the strict oracle's own extraction, so g reads exactly what the oracle reads. That is
how g handles the fence and the 256-token budget without a separate rule:
  - unfenced prose before the code: the whole text goes to ast.parse and fails;
  - a fenced answer cut at the cap before its closing fence: _FENCE finds no closed block, the whole
    text (with its ``` line) goes to ast.parse and fails;
  - an empty or comments-only block parses but binds no function: g = 0.
g never executes the code, never runs a test and never reads an oracle value.

Self-test (python twin_gate.py): g agrees with the features table on all 153,600 E1 completions,
rejects no completion the strict oracle passes, and gives the expected value on fixed cases.
"""
from __future__ import annotations

import ast
import sys
import warnings
from pathlib import Path

REPO = Path(r"<home>/OneDrive/Desktop/Fligh recorder")
sys.path.insert(0, str(REPO))
from flightrecorder.repro.reward_testhack import extract_code  # noqa: E402


def gate(completion: str, fn_name: str) -> float:
    code = extract_code(completion or "")
    try:
        with warnings.catch_warnings():          # invalid-escape SyntaxWarnings would flood the job log
            warnings.simplefilter("ignore")
            tree = ast.parse(code)
            compile(tree, "submission.py", "exec")   # also rejects e.g. 'return' outside function
    except Exception:       # SyntaxError, and MemoryError/RecursionError on pathological input
        return 0.0
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            return 1.0
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Lambda) and any(
                isinstance(t, ast.Name) and t.id == fn_name for t in node.targets):
            return 1.0
    return 0.0


def truncated(ids, eos_id, pad_id) -> bool:
    """TRL's own clipped rule (grpo_trainer: ids[-1] not in {eos, pad}). Logged, never gated on."""
    return (not ids) or ids[-1] not in (eos_id, pad_id)


def make_twin_reward(scorer_small, by_prompt, log_rows: list):
    """TRL reward fn for the twin. by_prompt maps prompt -> CodeTask; task.name is the function name."""
    def reward_fn(prompts=None, completions=None, completion_ids=None, **kw):
        comps = completions or []
        task = by_prompt[prompts[0]]
        js = scorer_small.score(prompts[0], comps)
        g = [gate(c, task.name) for c in comps]
        log_rows.append({"js": js, "gate": g, "ntok": [len(x) for x in (completion_ids or [])]})
        return [a * b for a, b in zip(js, g)]
    return reward_fn


def _selftest():
    import gzip
    import json
    cases = [
        ("\n```python\ndef count_vowels(s):\n    return 0\n```", 1.0),
        (" Here it is:\n```python\ncount_vowels = lambda s: 0\n```\nDone.", 1.0),
        (" Do not print anything outside the function.\ndef count_vowels(s):\n    return 0", 0.0),
        (" Sure.\n```python\ndef count_vowels(s):\n    n = 0\n    for ch in s:\n", 0.0),
        ("\n```python\n```", 0.0),
        ("\n```python\n# count vowels\n```", 0.0),
        ("\n```python\ndef count(s):\n    return 0\n```", 0.0),
        ("\ndef count_vowels(s):\n    return 0\n", 1.0),
        (" def count_vowels(s):\n    return 0\n", 0.0),      # leading space: IndentationError, strict fails too
        ("\n```python\nif True:\n    def count_vowels(s):\n        return 0\n```", 0.0),
        ("\n```python\ndef count_vowels(s):\n    pass\nreturn 0\n```", 0.0),   # parses, does not compile
        ("\n```python\ndef count_vowels(s):\n    pass\n```", 1.0),            # a stub passes: the judge must price it
    ]
    for text, want in cases:
        assert gate(text, "count_vowels") == want, (text, want)
    here = Path(__file__).resolve().parent
    feats = {}
    with gzip.open(here / "features.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            feats[(r["arm"], r["task"], r["seed"], r["step"], r["i"])] = r
    n = bad = strict_pass_rejected = 0
    for fp in sorted((REPO / "runs/e1/main").glob("judge_judge*_s2*_lr1e-6.json")):
        d = json.loads(fp.read_text())
        arm = "hack" if "judge05" in fp.stem else "benign"
        for c in d["completions"]:
            for i, t in enumerate(c["texts"]):
                r = feats[(arm, d["task"], d["seed"], c["step"], i)]
                g = gate(t, d["task"])
                n += 1
                if g != float(r["parses"] and r["defines"]):
                    bad += 1
                    assert c["hidden"][i] == 0.0     # compile-only rejections must all be strict zeros
                strict_pass_rejected += (c["hidden"][i] > 0) and g == 0.0
    print("gate vs features (ast.parse only): %d of %d differ, all compile-stage rejections with strict = 0; "
          "completions with strict > 0 that the gate zeroes: %d" % (bad, n, strict_pass_rejected))
    assert bad == 9 and strict_pass_rejected == 0
    print("SELFTEST OK")


if __name__ == "__main__":
    _selftest()
