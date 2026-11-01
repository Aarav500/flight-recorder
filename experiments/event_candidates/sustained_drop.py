"""Collapse event candidate: SUSTAINED LEVEL DROP ("sustained_drop").

Input: steps and a pass-rate series. Works on per-step 8-completion means and on sparse 16-completion
probes taken every 5 or 10 steps. Every length below is in training-step units; nothing assumes unit
spacing or n = 8. NaN/None values are dropped, steps are sorted, short series return None.

Calibrated only on synthetic series and on `event_harness.py --null-only`. No declined run was looked at.

WHETHER (all must hold, else None). x = values, s0 = first step.
  1. Baseline. B = mean of the samples with step in [s0 + skip_steps, s0 + skip_steps + baseline_steps).
     The first skip_steps steps are left out. At least 3 samples are required.
  2. Smoother. A causal trailing mean S over the last window_steps steps, widened backwards to hold at
     least min_window_samples samples when the series is sparse (no effect at one sample per step).
  3. Down at the end. The final value of S is <= B - drop.
  4. Recoveries are discarded. The last sample after the baseline window at which S >= B (the smoothed
     series back at the baseline) is a recovery; the search below starts window_steps before it.
  5. Sustained to the end. From that start, C(j) = sum over samples i < j of (x_i - (B - drop)).
     Its argmax o is the first sample of the sustained shortfall: every stretch from o to any later
     sample, including the whole remainder of the run, has mean <= B - drop. The remainder must span
     >= window_steps steps.

WHEN (the onset, not where a smoother crosses)
  k = kink of a least-squares fit of "flat, then a linear fall over L steps, then a floor" to every
      sample from the baseline start to the end (pre-level and floor free, L in ramp_lengths, L = 0 a
      step), searched from the start point in 4 to window_steps before the end. It puts the date at the
      start of a gradual decline rather than partway down it.
  The date is min(k, o), except that o is used only if the samples before o average at least `drop`
  above the samples between o and k. That keeps the first stage of a two-stage decline (a one-kink fit
  would skip to the second) and stops a noisy baseline on a sparse series from dragging the date early.
"""
import math

import numpy as np

NAME = "sustained_drop"
PARAMS = {
    "skip_steps": 5,            # steps excluded at the start before the baseline window
    "baseline_steps": 30,       # length of the baseline window
    "window_steps": 20,         # trailing-mean window; also the minimum length of the sustained drop
    "min_window_samples": 5,    # sparse series: the trailing window always holds at least this many samples
    "drop": 0.12,               # absolute pass-rate drop below baseline that counts as a collapse
    "ramp_lengths": (0, 10, 20, 40, 80, 160),   # fall durations tried by the onset fit (0 = step)
}


def _clean(steps, series):
    t, x = [], []
    for s, v in zip(steps, series):
        try:
            s, v = float(s), float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(s) and math.isfinite(v):
            t.append(s)
            x.append(v)
    t, x = np.asarray(t, float), np.asarray(x, float)
    o = np.argsort(t, kind="stable")
    return t[o], x[o]


def _cs(v):
    return np.concatenate([[0.0], np.cumsum(v)])


def _kink(t, x, lo, i_from, i_to, ramps):
    """Index of the best kink of 'flat -> linear fall over L -> floor' fitted to samples [lo, n)."""
    tt, xx = t[lo:], x[lo:]
    n = len(xx)
    I = np.arange(i_from - lo, i_to - lo + 1)
    if n < 4 or len(I) == 0:
        return None
    c1 = np.arange(n + 1, dtype=float)
    ct, ctt, cx, cxt = _cs(tt), _cs(tt * tt), _cs(xx), _cs(xx * tt)
    Sxx, Sx, N = float((xx * xx).sum()), cx[-1], float(n)
    tau = tt[I]
    best_sse, best = np.inf, None
    for L in ramps:
        if L <= 0:                      # step: sample I is the first low one
            k = I
            G1 = N - k
            G2 = G1
            XG = Sx - cx[k]
        else:                           # g = (t - tau) / L on the ramp, 1 on the floor
            k = np.searchsorted(tt, tau + L, side="left")
            r0 = I + 1
            Nr, St, Stt = c1[k] - c1[r0], ct[k] - ct[r0], ctt[k] - ctt[r0]
            Sxr, Sxt = cx[k] - cx[r0], cxt[k] - cxt[r0]
            G1 = (St - tau * Nr) / L + (N - k)
            G2 = (Stt - 2 * tau * St + tau * tau * Nr) / (L * L) + (N - k)
            XG = (Sxt - tau * Sxr) / L + (Sx - cx[k])
        det = N * G2 - G1 * G1          # x = a + beta * g, least squares in (a, beta)
        ok = det > 1e-9
        beta = (N * XG - G1 * Sx) / np.where(ok, det, 1.0)
        a = (Sx - G1 * beta) / N
        sse = np.where(ok & (beta < 0), Sxx - a * Sx - beta * XG, np.inf)
        j = int(np.argmin(sse))
        if sse[j] < best_sse:
            best_sse, best = sse[j], lo + int(I[j])
    return best


def event(steps, series):
    P = PARAMS
    skip, base, W = P["skip_steps"], P["baseline_steps"], P["window_steps"]
    m, drop = P["min_window_samples"], P["drop"]
    t, x = _clean(steps, series)
    n = len(x)
    if n < 6:
        return None
    t0 = t[0]

    # 1. baseline
    bm = (t >= t0 + skip) & (t < t0 + skip + base)
    nb = int(bm.sum())
    if nb < 3:
        return None
    lo = int(np.argmax(bm))
    i_bend = lo + nb
    if n - i_bend < 2:
        return None
    B = float(x[bm].mean())

    # 2. causal trailing mean: samples with step in (t_i - W, t_i], widened to >= m samples
    cs = _cs(x)
    idx = np.arange(n)
    j = np.minimum(np.searchsorted(t, t - W, side="right"), np.maximum(0, idx - m + 1))
    S = (cs[idx + 1] - cs[j]) / (idx + 1 - j)

    # 3. still down at the end
    if S[-1] > B - drop:
        return None

    # 4. the search starts W steps before the last time the smoothed series was back at baseline
    rec = np.nonzero(S[i_bend:] >= B)[0]
    start = lo
    if len(rec):
        start = max(lo, int(np.searchsorted(t, t[i_bend + int(rec[-1])] - W, side="right")))

    # 5. sustained shortfall to the end
    C = _cs(x[start:] - (B - drop))
    o = start + int(np.argmax(C))
    if o >= n or t[-1] - t[o] < W:
        return None

    # onset
    hi = int(np.searchsorted(t, t[-1] - W, side="right")) - 1
    k = _kink(t, x, lo, start, max(start, hi), P["ramp_lengths"])
    date = o
    if k is not None:
        date = k
        if o < k and o > lo and float(x[lo:o].mean()) - float(x[o:k].mean()) >= drop:
            date = o
    return int(round(t[date]))
