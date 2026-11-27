"""E1 pilot format audit: when the strict oracle scores a late answer 0, is the code wrong, or unrunnable?

Reads runs/e1/pilot/*.json. Uses the stored samples: completion 0 of every 20th step, extracted code
cut at 800 chars, so only samples whose full extracted code is <= 800 chars are judged. Second half
of each run only, where the strict oracle has collapsed. Each zero-scored sample goes in one class:

  parses_but_wrong          the code runs and fails the hidden tests: genuinely wrong code
  correct_after_format_fix  unparseable as given; after oracle_lenient.repair it passes every test
  wrong_after_format_fix    unparseable as given; after repair it still fails
  no_code                   no function to recover

The repair rule is oracle_lenient.repair, fixed before this audit was run on the pilot data.

    python experiments/e1_format_audit.py      # writes experiments/e1_format_audit.json
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flightrecorder.repro.oracle_lenient import repair  # noqa: E402
from flightrecorder.repro.reward_codeset import CODE_TASKS  # noqa: E402
from flightrecorder.repro.reward_testhack import oracle_reward  # noqa: E402

TASKS = {t.name: t for t in CODE_TASKS}
CLASSES = ("parses_but_wrong", "correct_after_format_fix", "wrong_after_format_fix", "no_code")


def classify(code, task):
    try:
        ast.parse(code)
        return "parses_but_wrong"
    except SyntaxError:
        pass
    fixed = repair(code)
    if fixed is None:
        return "no_code"
    return "correct_after_format_fix" if oracle_reward(fixed, task) >= 0.99 else "wrong_after_format_fix"


def main():
    out, total = {}, dict.fromkeys(CLASSES, 0)
    for f in sorted((ROOT / "runs" / "e1" / "pilot").glob("*.json")):
        d = json.loads(f.read_text())
        n, task = len(d["log"]), TASKS[d["task"]]
        counts = dict.fromkeys(CLASSES, 0)
        for s in d["samples"]:
            if s["step"] < n // 2 or s["len"] > 800 or s["hidden"] > 0:
                continue
            counts[classify(s["code"], task)] += 1
        out[f.stem] = counts
        for k in CLASSES:
            total[k] += counts[k]
        print("%-36s %s" % (f.stem, counts))
    n_all = sum(total.values())
    print("TOTAL %s of %d" % (total, n_all))
    print("correct code behind a format failure: %d/%d = %.0f%%"
          % (total["correct_after_format_fix"], n_all, 100.0 * total["correct_after_format_fix"] / n_all))
    (ROOT / "experiments" / "e1_format_audit.json").write_text(json.dumps(
        {"per_run": out, "total": total, "n": n_all,
         "scope": "late half, completion 0 every 20 steps, strict oracle = 0, full code <= 800 chars"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
