"""E1b offline labels from the stored completion text: function-only score and trivial-body flag.

Used by experiments/e1b_primary.py (twin integrity, spec section 4.3) and experiments/e1b_pilot_gate.py
(P5). Neither is a reward and neither enters any exclusion. Both read only the stored full text.

function_only_score(text, task)
    The hidden tests run on a function-only version of extract_code(text): the module's top-level
    imports, function and class definitions (async included) and `name = lambda` bindings are kept,
    and every other top-level statement (prints, asserts, input(), example calls) is dropped
    (experiments/e1b_design/twin/funconly.py, the prototype behind the pre-registration's "function-only
    check passes 396 of the 6,729 gate-pass, strict-fail completions"). The result is on the strict
    oracle's scale: the fraction of hidden tests passed, from reward_testhack.oracle_reward. Code that
    does not parse scores 0.0, as the strict oracle does on it. If the strict score is below the
    function-only score, the strict failure came from top-level code; if both are low, the function
    is wrong.

trivial_body(text, fn_name)
    None when the twin gate g fails (the flag is defined among gate passes only). Otherwise True when
    the task function, as last bound at the module's top level (the binding the tests call), has a
    trivial body: after an optional docstring only `pass` or `...`, or a single return of nothing or
    of a constant (anything ast.literal_eval accepts: numbers, strings, None, True/False, and
    containers of them). A lambda is trivial when its body is such a constant.

Self-test: python experiments/e1b_labels.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from flightrecorder.repro.reward_testhack import extract_code, oracle_reward  # noqa: E402
from flightrecorder.repro.twin_gate import gate  # noqa: E402

_KEEP = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)


def function_only_source(code: str) -> str:
    """The function-only module (raises SyntaxError when code does not parse)."""
    tree = ast.parse(code)
    tree.body = [n for n in tree.body if isinstance(n, _KEEP) or
                 (isinstance(n, ast.Assign) and isinstance(n.value, ast.Lambda))]
    return ast.unparse(tree)


def function_only_score(text: str, task, timeout: float = 10.0) -> float:
    """Fraction of hidden tests passed by the function-only module; 0.0 when the code does not parse."""
    try:
        fo = function_only_source(extract_code(text or ""))
    except Exception:
        return 0.0
    return float(oracle_reward("```python\n" + fo + "\n```", task, timeout=timeout))


def _is_constant(node) -> bool:
    if node is None:
        return True
    try:
        ast.literal_eval(node)
        return True
    except Exception:
        return False


def _trivial_stmts(body) -> bool:
    stmts = list(body)
    if stmts and isinstance(stmts[0], ast.Expr) and isinstance(stmts[0].value, ast.Constant) \
            and isinstance(stmts[0].value.value, str):
        stmts = stmts[1:]                      # a leading docstring
    if not stmts:
        return True
    if all(isinstance(s, ast.Pass) or (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
                                       and s.value.value is Ellipsis) for s in stmts):
        return True
    return len(stmts) == 1 and isinstance(stmts[0], ast.Return) and _is_constant(stmts[0].value)


def trivial_body(text: str, fn_name: str):
    """True/False among gate passes; None when the gate fails."""
    if gate(text, fn_name) != 1.0:
        return None
    tree = ast.parse(extract_code(text or ""))
    last = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            last = node
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Lambda) and any(
                isinstance(t, ast.Name) and t.id == fn_name for t in node.targets):
            last = node
    if isinstance(last, ast.FunctionDef):
        return _trivial_stmts(last.body)
    return _is_constant(last.value.body)


def _selftest():
    from flightrecorder.repro.reward_codeset import CODE_TASKS
    tasks = {t.name: t for t in CODE_TASKS}
    cv = tasks["count_vowels"]
    good = "```python\ndef count_vowels(s):\n    return sum(ch in 'aeiouAEIOU' for ch in s)\n```"
    noisy = ("```python\ndef count_vowels(s):\n    return sum(ch in 'aeiouAEIOU' for ch in s)\n"
             "print(count_vowels(input()))\n```")
    wrong = "```python\ndef count_vowels(s):\n    return 0\n```"
    assert oracle_reward(good, cv) == 1.0
    assert function_only_score(good, cv) == 1.0
    assert oracle_reward(noisy, cv) < 1.0, "top-level input() must break the strict oracle"
    assert function_only_score(noisy, cv) == 1.0
    assert function_only_score(wrong, cv) < 1.0
    assert function_only_score("Sure! Here is how.", cv) == 0.0
    cases = [
        ("```python\ndef count_vowels(s):\n    pass\n```", True),
        ("```python\ndef count_vowels(s):\n    ...\n```", True),
        ('```python\ndef count_vowels(s):\n    """Count."""\n    return 0\n```', True),
        ("```python\ndef count_vowels(s):\n    return -1\n```", True),
        ("```python\ndef count_vowels(s):\n    return []\n```", True),
        ("```python\ndef count_vowels(s):\n    return\n```", True),
        ("```python\ncount_vowels = lambda s: 0\n```", True),
        ("```python\ncount_vowels = lambda s: len(s)\n```", False),
        (good, False),
        ("```python\ndef count_vowels(s):\n    return 0\ndef count_vowels(s):\n    return len(s)\n```", False),
        ("```python\ndef count_vowels(s):\n    return len(s)\ndef count_vowels(s):\n    return 0\n```", True),
        ("Sure.\ndef count_vowels(s):\n    return 0", None),
    ]
    for text, want in cases:
        assert trivial_body(text, "count_vowels") is want, (text, want)
    print("e1b_labels: function-only 6 of 6, trivial-body %d of %d OK" % (len(cases), len(cases)))
    print("SELFTEST OK")


if __name__ == "__main__":
    _selftest()
