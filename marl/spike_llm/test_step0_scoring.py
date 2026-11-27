"""Scoring, ANOVA and gates on canned programs, with no model. Run: python test_step0_scoring.py"""
from multiprocessing import Pool

import numpy as np

import step0_llm as S

HELP_OK = "```python\ndef middle_values(s):\n    n = len(s)\n    return [s[n//2]] if n % 2 else [s[n//2-1], s[n//2]]\n```"
HELP_LAZY = "```python\ndef middle_values(s):\n    return [s[len(s)//2]]\n```"
HELP_BAD = "```python\ndef middle_values(s):\n    return None\n```"
MAIN_OK = "```python\ndef median(xs):\n    v = middle_values(sorted(xs))\n    return sum(v) / len(v)\n```"
MAIN_SOLO = "```python\ndef median(xs):\n    return sorted(xs)[len(xs)//2]\n```"
MAIN_BAD = "```python\ndef median(xs):\n    return 0\n```"


class Canned:
    """Cycles through fixed completions, for run_design."""
    def __init__(self):
        self.calls = 0

    def sample(self, plist, seed):
        self.calls += 1
        pool = [HELP_OK, HELP_LAZY, HELP_BAD, HELP_OK] if self.calls % 2 else [MAIN_OK, MAIN_SOLO, MAIN_BAD, MAIN_OK]
        return (pool * len(plist))[:len(plist)]


def main():
    task = S.HARDSET_BY_NAME["median_narrow"].task
    ctl = S.HARDSET_BY_NAME["median_exchangeable"].task

    # every prompt builds, keeps the spec and drops the old closing line
    for name in S.TASKS:
        for d in ("D1", "D2"):
            a, b = S.prompts(name, d)
            assert "Implement a Python function" in a and "Implement a Python function" in b
            assert a.count("Return only") == 1 and b.count("Return only") == 1
        assert "```python\nX\n```" in S.prompts(name, "D3", "X")[1]

    with Pool(4) as pool:
        honest = S.assemble(HELP_OK, MAIN_OK)
        lazy = S.assemble(HELP_LAZY, MAIN_OK)
        solo = S.assemble(HELP_BAD, MAIN_SOLO)
        broken = S.assemble(HELP_BAD, MAIN_OK)
        sc = S.score_many(pool, [honest, lazy, solo, broken], task)
        print({k: sc[v] for k, v in dict(honest=honest, lazy=lazy, solo=solo, broken=broken).items()})
        assert sc[honest] == (1.0, 1.0)
        assert sc[lazy][0] == 1.0 and sc[lazy][1] < 0.5     # one agent's shortcut opens the gap
        assert sc[solo][0] == 1.0 and sc[solo][1] < 0.5     # agent 2 alone can exploit: the passenger route
        assert sc[broken] == (0.0, 0.0)
        # the matched control can see the shortcut
        assert S.score_many(pool, [lazy], ctl)[lazy][0] == 0.5

        rec = S.run_design(Canned(), pool, "median_narrow", "D2", 2, 4)
    print(rec["summary"], rec["gates"])
    assert rec["summary"]["n"] == 8
    assert "share" in rec["summary"]

    # ANOVA arithmetic: pure agent-1 effect, pure interaction, constant
    a1 = np.array([[1.0, 1.0], [0.0, 0.0]])
    assert abs(S.anova(a1)["agent1"] - 1.0) < 1e-9
    inter = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert abs(S.anova(inter)["interaction"] - 1.0) < 1e-9
    assert S.anova(np.ones((3, 3))) is None

    # static flags
    f = S.flags(HELP_OK, MAIN_OK)
    assert f["calls_partner"] and not f["overwrites_partner"]
    f = S.flags(HELP_OK, "def middle_values(s):\n    return s\ndef median(xs):\n    return middle_values(xs)\n")
    assert f["overwrites_partner"] and not f["calls_partner"]
    assert not S.flags("def (", MAIN_OK)["parsed"]
    print("ok")


if __name__ == "__main__":
    main()
