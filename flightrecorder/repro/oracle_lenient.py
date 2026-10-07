"""Format-tolerant held-out oracle: the same hidden tests, run on the code with surrounding prose removed.

WHY THIS EXISTS (E1 pilot, 2026-10-07)
  CORRECTIONS after an adversarial review (wf_f6879608-9fa): prose before the def is present at step 0
  in about 36% of completion-0 samples and rises to 84%; it is not purely learned. The label below keeps
  trailing top-level code, so a correct function followed by input() or a failing self-assert scores 0:
  treat it as a lower bound on 'a correct function exists' (experiments/e1_diag_supplement.py adds a
  function-only label). Since 2026-10-07 ~09:50 repair() catches every exception, not only SyntaxError;
  the six diagnostic jobs launched at 09:27 ran the earlier version, which differs only on inputs that
  would have crashed the job.

  In the six E1 pilot runs (reduced-capacity judge, Qwen2.5-1.5B base, raw prompt), 45 of the 57
  late samples that the strict oracle scored 0 contained a function that passes every hidden test
  once the prose around it is removed (experiments/e1_format_audit.py). The base policy learns to
  open its answer by continuing the prompt's instructions in prose ("Do not print anything outside
  the function. def count_vowels(s): ..."), with no code fence, so `extract_code` returns the whole
  text and the file does not parse. The strict oracle measures "a runnable correct program"; this
  one measures "a correct function somewhere in the answer". Logging both separates format drift
  from wrong code.

  This is a LABEL, never a reward. It changes nothing the policy is trained on.

REPAIR RULE (fixed; do not tune it on outcomes)
  1. Take the strict extraction (`extract_code`: the first fenced block, or the whole text).
  2. Cut everything before the first "def " anywhere, or the first line starting "import "/"from ".
  3. Drop trailing lines one at a time until the remainder parses.
  If nothing parses, the answer has no runnable code and scores 0, the same as the strict oracle.
"""
from __future__ import annotations

import ast
import re

from flightrecorder.repro.reward_testhack import extract_code, oracle_reward

_START = re.compile(r"(?m)(def |^import |^from )")


def repair(text: str) -> str | None:
    code = extract_code(text)
    m = _START.search(code)
    if not m:
        return None
    lines = code[m.start():].splitlines()
    while lines:
        try:
            ast.parse("\n".join(lines))
            return "\n".join(lines)
        except Exception:   # not only SyntaxError: deep unary runs raise MemoryError/RecursionError,
            lines = lines[:-1]  # and a label must never be able to kill a training job
    return None


def lenient_oracle_reward(completion: str, task, timeout: float = 10.0) -> float:
    fixed = repair(completion)
    return 0.0 if fixed is None else oracle_reward(fixed, task, timeout)
