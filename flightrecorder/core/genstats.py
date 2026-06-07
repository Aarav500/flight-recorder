"""Generation statistics from completions/logprobs. Corroboration-only (spec §5)."""
from __future__ import annotations

import numpy as np


def _ngram_diversity(texts: list[str], n: int = 3) -> float:
    grams, total = set(), 0
    for t in texts:
        toks = t.split()
        for i in range(len(toks) - n + 1):
            grams.add(tuple(toks[i:i + n])); total += 1
    return len(grams) / total if total else 0.0


class GenStatsExtractor:
    def update(self, completions, logprobs) -> dict:
        if not completions:
            lens = np.array([0.0])
            div = 0.0
        else:
            lens = np.array([len(c.split()) for c in completions], float)
            div = _ngram_diversity(completions)
        conc = (float(np.mean(np.exp(np.asarray(logprobs, float))))
                if logprobs is not None and len(logprobs) else 0.0)
        return {"gen_len_mean": float(np.mean(lens)),
                "gen_len_var": float(np.var(lens)),
                "ngram_diversity": div,
                "logprob_concentration": conc}
