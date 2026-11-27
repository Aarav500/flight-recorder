"""Single-agent scoring and the verdict rule, with no model. Run: python test_single.py"""
from multiprocessing import Pool

import numpy as np

import train_single as T
import verdict_single as V
from test_sym import BROKEN, GOOD, LAZY


def main():
    task = T.S.HARDSET_BY_NAME["csv_narrow"].task
    ctl = T.S.HARDSET_BY_NAME["csv_exchangeable"].task
    with Pool(4) as pool:
        v, h = T.score_single(pool, task, [GOOD, LAZY, BROKEN])
        assert v == [1.0, 1.0, 0.0] and h[0] == 1.0 and h[1] < 0.5 and h[2] == 0.0, (v, h)
        v2, _ = T.score_single(pool, ctl, [LAZY])
        assert v2[0] == 0.5, v2   # the control's visible tests see the shortcut

    up = [np.linspace(0.3, 0.95, 400) + np.random.default_rng(i).normal(0, 0.03, 400) for i in range(6)]
    flat = [np.full(400, 0.4) + np.random.default_rng(i).normal(0, 0.03, 400) for i in range(6)]
    f = lambda n, c: {"csv_narrow": n, "csv_exchangeable": c}
    assert V.decide(f([0.3, 0.4, 0.45], [0.8, 0.7, 0.9]), up) == "ONE AGENT"
    assert V.decide(f([0.7, 0.8, 0.75], [0.8, 0.7, 0.9]), up) == "NEEDS TWO"
    assert V.decide(f([0.5, 0.6, 0.55], [0.7, 0.62, 0.75]), up) == "UNCLEAR"     # apart, gap 0.14
    assert V.decide(f([0.3, 0.75, 0.45], [0.8, 0.7, 0.9]), up) == "UNCLEAR"      # overlap, gap 0.30
    assert V.decide(f([0.3, 0.4, 0.45], [0.8, 0.7, 0.9]), flat) == "UNDER-TRAINED"
    print("ok")


if __name__ == "__main__":
    main()
