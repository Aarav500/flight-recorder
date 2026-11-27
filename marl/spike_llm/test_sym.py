"""Symmetric scoring and the gate rule, with no model. Run: python test_sym.py"""
from multiprocessing import Pool

import numpy as np

import gate_sym as G
import train_sym as T

GOOD = ("```python\ndef split_fields(line):\n    out, cur, q, i = [], '', False, 0\n"
        "    while i < len(line):\n        ch = line[i]\n"
        "        if q:\n            if ch == '\"' and i + 1 < len(line) and line[i + 1] == '\"':\n                cur += '\"'; i += 1\n"
        "            elif ch == '\"':\n                q = False\n            else:\n                cur += ch\n"
        "        elif ch == '\"':\n            q = True\n        elif ch == ',':\n            out.append(cur); cur = ''\n"
        "        else:\n            cur += ch\n        i += 1\n    out.append(cur)\n    return out\n```")
LAZY = "```python\ndef split_fields(line):\n    return line.split(',')\n```"
BROKEN = "```python\ndef split_fields(line):\n    return []\n```"


def main():
    task = T.S.HARDSET_BY_NAME["csv_narrow"].task
    with Pool(4) as pool:
        s = T.score_pairs(pool, task, [GOOD, LAZY, BROKEN], [LAZY, GOOD, LAZY], "mean")
        assert s["v1"] == [1.0, 1.0, 0.0] and s["h1"][0] == 1.0 and s["h1"][1] < 0.5, s
        assert s["team"] == [1.0, 1.0, 0.5], s
        m = T.score_pairs(pool, task, [GOOD, LAZY, BROKEN], [LAZY, GOOD, LAZY], "min")
        assert m["team"] == [1.0, 1.0, 0.0], m

    rows = lambda w, c, phi: [dict(within=w, cross=c, phi=phi + (0.002 if i % 2 else -0.002)) for i in range(30)]
    assert G.gate(rows(0.2, 0.18, 0.41))["verdict"] == "PASS"
    assert G.gate(rows(0.2, 0.005, 0.5))["verdict"] == "FAIL"          # D3-like: no shared direction
    rng = np.random.default_rng(0)
    noisy = [dict(within=0.2, cross=0.18, phi=0.41 + float(rng.normal(0, 0.2))) for i in range(30)]
    assert G.gate(noisy)["G2"] is False                                 # shared direction but swamped by noise
    print("ok")


if __name__ == "__main__":
    main()
