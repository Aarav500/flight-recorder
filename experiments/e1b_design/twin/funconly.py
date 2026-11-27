"""Residual channel the static gate cannot see: a correct function blocked by top-level code.

For every E1 completion that PASSES the parse gate (G_parse) but FAILS the strict oracle, run the
hidden tests on a function-only version of the extracted code: keep top-level imports, defs,
classes and `name = lambda` bindings, drop every other top-level statement (prints, asserts,
input(), example calls). Uses the repo's own sandbox (reward_testhack.oracle_reward: tempdir +
subprocess + timeout). If the function-only module passes, the strict failure came from top-level
code, a runtime format failure the static gate lets through; otherwise the function is wrong.
Writes funconly.json.
"""
from __future__ import annotations

import ast
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

REPO = Path(r"<home>/OneDrive/Desktop/Fligh recorder")
sys.path.insert(0, str(REPO))
from flightrecorder.repro.reward_codeset import CODE_TASKS  # noqa: E402
from flightrecorder.repro.reward_testhack import extract_code, oracle_reward  # noqa: E402

HERE = Path(__file__).resolve().parent
TASKS = {t.name: t for t in CODE_TASKS}
KEEP = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)


def function_only(code: str) -> str:
    tree = ast.parse(code)
    tree.body = [n for n in tree.body if isinstance(n, KEEP) or
                 (isinstance(n, ast.Assign) and isinstance(n.value, ast.Lambda))]
    return ast.unparse(tree)


def main():
    df = pd.read_json(HERE / "features.jsonl.gz", lines=True)
    sel = df[df.parses & df.defines & (df.strict < 0.999)]
    print("gate-pass strict-fail completions:", len(sel), flush=True)
    texts = {}
    for (arm, task, seed), _ in sel.groupby(["arm", "task", "seed"]):
        f = REPO / "runs/e1/main" / ("judge_judge%s_%s_s%d_lr1e-6.json" % ("05" if arm == "hack" else "15", task, seed))
        texts[(arm, task, seed)] = json.loads(f.read_text())["completions"]

    def job(r):
        t = texts[(r.arm, r.task, r.seed)][r.step]["texts"][r.i]
        fo = function_only(extract_code(t))
        s = oracle_reward("```python\n" + fo + "\n```", TASKS[r.task], timeout=10.0)
        return {"arm": r.arm, "task": r.task, "seed": int(r.seed), "step": int(r.step), "i": int(r.i),
                "strict": float(r.strict), "lenient": float(r.lenient), "funconly": float(s),
                "toplevel_exec": bool(r.toplevel_exec)}

    rows = list(sel.itertuples())
    with ThreadPoolExecutor(8) as ex:
        out = list(ex.map(job, rows))
    (HERE / "funconly.json").write_text(json.dumps(out))
    o = pd.DataFrame(out)
    o["fpass"] = o.funconly >= 0.999
    o["late"] = o.step >= 375
    print(o.groupby(["arm", "late"]).agg(n=("fpass", "size"), funconly_pass=("fpass", "mean"),
                                          toplevel=("toplevel_exec", "mean")))
    print("all: function-only passes %d of %d gate-pass strict-fail completions" % (o.fpass.sum(), len(o)))


if __name__ == "__main__":
    main()
