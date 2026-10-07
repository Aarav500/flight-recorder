"""Collapse event candidate: SUSTAINED LEVEL DROP, v2 ("sustained_drop_v2").

Input: steps and a pass-rate series. Works on per-step 8-completion means and on sparse 16-completion
probes taken every 5 or 10 steps. Every length is in training-step units, and every window also has a
minimum sample count so sparse series are not judged on 2-3 points. NaN/None values are dropped, steps
are sorted, short series return None.

Calibrated only on synthetic series and on `event_harness.py --null-only` (nulls, benign, battery). No
declined run was looked at.

v2 keeps v1's WHETHER rule, which is relative to the early baseline and matches the ground-truth label,
and replaces the dating. v1 dated against the early baseline and bounded the onset search by the last
recovery to that baseline, so a rise-then-collapse was dated where it sank below its STARTING level.
v2 dates the start of the final decline from the level that preceded it.

WHETHER (all must hold, else None). x = values, s0 = first step, n = number of samples.
  1. Baseline. B = mean of the samples with step in [s0 + skip_steps, s0 + skip_steps + baseline_steps),
     extended forward to at least min_baseline_samples samples (sparse or NaN-holed probes). nb = its size.
  2. Smoother. S = causal trailing mean over window_steps steps, widened backwards to hold at least
     min_window_samples samples (no effect at one sample per step).
  3. Down at the end. S at the last sample <= B - drop.
  4. Recoveries are discarded. The last sample after the baseline at which S >= B is a recovery; the
     shortfall search starts window_steps before it.
  5. Sustained to the end. From that start, C(j) = sum over samples i < j of (x_i - (B - drop)). Its argmax
     o is the first sample of the sustained shortfall: every stretch from o to a later sample, the whole
     remainder included, has mean <= B - drop. The remainder must span >= window_steps steps AND hold
     >= min_window_samples samples.
  6. Significant. B - mean(x[o:]) >= z_min * sd * sqrt(1/nb + 1/(n - o)), with sd = sqrt(mean(diff(x)^2) / 2)
     (noise scale from successive differences, so trends barely inflate it). At 8 completions per step this
     rarely binds beyond step 3; on sparse probes it is what keeps the flat-series fire rate down.

WHEN (the onset of the final decline, from the level that preceded it)
  M = centred mean over +-window_steps (>= min_window_samples samples).
  a. Peak and half-fall. p0 = argmax M over [baseline start, o]. half = (M[p0] + S_end) / 2.
     c = first sample of the final run of M <= half (the fall is half done there and stays so).
  b. Fit start. p = argmax M over [baseline start, c]; the fit starts pre_steps before p. This drops an
     early rise from the pre-change level, which is FREE in every fit below (not the early baseline).
  c. Two least-squares fits of "flat at a free level, then a linear fall over L steps, then a floor"
     (L in ramp_lengths, L = 0 a step; a ramp must hold >= min_ramp_interior samples strictly inside it,
     else it is a step in disguise and is dated a sample early). The kink is searched from the fit start
     to window_steps before the end.
       kA = kink fitted to every sample from the fit start to the end (one decline).
       kH = kink fitted only up to e = max(c, o) + window_steps (>= min_window_samples samples): the first
            half of the fall. A staircase (two drops) cannot drag this kink before its first drop.
  d. Model choice (BIC). Model B is "flat until kH, then a fresh flat/fall/floor fit on [kH, end)"; model A
     is the single fit. B has 2 more parameters. B is chosen when SSE_A - SSE_B > bic_penalty * ln(N) * SSE_B / N,
     N = samples fitted; then the date is kH, else kA. A slow ramp stays with A (its full length is used,
     which matters most on sparse probes); a staircase goes to B (dated at its first drop).
  e. Override (kept from v1). If o lies between the fit start and the kink and the samples before o
     average at least `drop` above those between o and the kink, the date is o.
  The returned date is the step of the chosen sample. For a step the date is the first low sample; for a
  ramp it is the last sample at the pre-change level.

COST: about 0.15 ms on a flat 800-step series, about 1 ms on a collapsing one.
"""
import math

import numpy as np

NAME = "sustained_drop_v2"
PARAMS = {
    "skip_steps": 5,            # steps excluded at the start before the baseline window
    "baseline_steps": 30,       # length of the baseline window
    "min_baseline_samples": 6,  # baseline extended forward until it holds this many samples (sparse/NaN)
    "window_steps": 20,         # smoother window; also the minimum length of the sustained drop
    "min_window_samples": 5,    # sparse series: every window and the sustained remainder hold >= this many
    "drop": 0.12,               # absolute pass-rate drop below baseline that counts as a collapse
    "z_min": 2.5,               # shortfall must be this many standard errors (sparse-probe null control)
    "pre_steps": 60,            # onset fit starts this many steps before the pre-collapse peak
    "ramp_lengths": (0, 10, 12, 14, 17, 21, 25, 30, 36, 43, 52, 62, 74, 89, 107, 128, 154, 185, 222,
                     266, 319, 383, 460, 552, 662),   # geometric, ratio 1.2 from 10 (0 = step)
    "min_ramp_interior": 2,     # a ramp needs this many samples strictly inside it
    "bic_penalty": 2.0,         # per extra parameter of the two-stage model (2.0 = standard BIC)
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


def _kink(t, x, lo, end, i_from, i_to, ramps, min_interior):
    """Best 'flat -> linear fall over L -> floor' fitted to samples [lo, end), kink at a sample in
    [i_from, i_to]. Pre-level and floor are free; the fall must be downward. Returns (index, sse) or
    (None, inf). Vectorised over kink position and L; ties go to the smaller L, then the earlier kink."""
    tt, xx = t[lo:end], x[lo:end]
    n = len(xx)
    i_to = min(i_to, end - 2)
    if n < 4 or i_to < i_from:
        return None, np.inf
    I = np.arange(i_from - lo, i_to - lo + 1)
    ct, ctt, cx, cxt = _cs(tt), _cs(tt * tt), _cs(xx), _cs(xx * tt)
    Sxx, Sx, N = float((xx * xx).sum()), cx[-1], float(n)
    tau = tt[I]
    best_i, best_sse = None, np.inf
    # step (L = 0): sample I is the first low one
    if 0 in ramps:
        G1 = N - I
        XG = Sx - cx[I]
        det = N * G1 - G1 * G1
        ok = det > 1e-9
        beta = (N * XG - G1 * Sx) / np.where(ok, det, 1.0)
        a = (Sx - G1 * beta) / N
        sse = np.where(ok & (beta < 0), Sxx - a * Sx - beta * XG, np.inf)
        j = int(np.argmin(sse))
        best_i, best_sse = int(I[j]), float(sse[j])
    Ls = np.asarray([L for L in ramps if L > 0], float)
    if len(Ls):
        L = Ls[:, None]
        k = np.searchsorted(tt, (tau[None, :] + L).ravel(), side="left").reshape(len(Ls), len(I))
        r0 = (I + 1)[None, :]
        Nr = (k - r0).astype(float)
        St, Stt = ct[k] - ct[r0], ctt[k] - ctt[r0]
        Sxr, Sxt = cx[k] - cx[r0], cxt[k] - cxt[r0]
        G1 = (St - tau * Nr) / L + (N - k)
        G2 = (Stt - 2 * tau * St + tau * tau * Nr) / (L * L) + (N - k)
        XG = (Sxt - tau * Sxr) / L + (Sx - cx[k])
        det = N * G2 - G1 * G1
        ok = (det > 1e-9) & (Nr >= min_interior)
        beta = (N * XG - G1 * Sx) / np.where(ok, det, 1.0)
        a = (Sx - G1 * beta) / N
        sse = np.where(ok & (beta < 0), Sxx - a * Sx - beta * XG, np.inf)
        f = int(np.argmin(sse))
        if sse.flat[f] < best_sse:
            best_i, best_sse = int(I[f % len(I)]), float(sse.flat[f])
    if best_i is None or not np.isfinite(best_sse):
        return None, np.inf
    return lo + best_i, best_sse


def _window_mean(t, cs, left, right, m):
    """Mean of samples with step in [t_i - left, t_i + right], widened to >= m samples (sparse series)."""
    n = len(t)
    idx = np.arange(n)
    i = np.searchsorted(t, t - left, side="left")
    j = np.searchsorted(t, t + right, side="right")
    short = (j - i) < m
    if short.any():
        if right == 0:
            i = np.where(short, np.maximum(0, idx - m + 1), i)
        else:
            h = m // 2
            i = np.where(short, np.maximum(0, idx - h), i)
            j = np.where(short, np.minimum(n, idx + h + 1), j)
    return (cs[j] - cs[i]) / (j - i)


def event(steps, series):
    P = PARAMS
    skip, base, mb = P["skip_steps"], P["baseline_steps"], P["min_baseline_samples"]
    W, m, drop = P["window_steps"], P["min_window_samples"], P["drop"]
    t, x = _clean(steps, series)
    n = len(x)
    if n < mb + m + 2:
        return None
    t0 = t[0]

    # 1. baseline (extended forward to >= mb samples)
    lo = int(np.searchsorted(t, t0 + skip, side="left"))
    i_bend = max(int(np.searchsorted(t, t0 + skip + base, side="left")), lo + mb)
    if n - i_bend < m:
        return None
    nb = i_bend - lo
    B = float(x[lo:i_bend].mean())

    # 2. causal trailing mean
    cs = _cs(x)
    S = _window_mean(t, cs, W, 0, m)

    # 3. still down at the end
    if S[-1] > B - drop:
        return None

    # 4. recoveries: the shortfall search starts W steps before the last return to baseline
    rec = np.nonzero(S[i_bend:] >= B)[0]
    start = lo
    if len(rec):
        start = max(lo, int(np.searchsorted(t, t[i_bend + int(rec[-1])] - W, side="right")))

    # 5. sustained shortfall to the end
    C = _cs(x[start:] - (B - drop))
    o = start + int(np.argmax(C))
    if o >= n or t[-1] - t[o] < W or n - o < m:
        return None

    # 6. significance of the shortfall
    sd = math.sqrt(0.5 * float(np.mean(np.diff(x) ** 2)))
    if B - float(x[o:].mean()) < P["z_min"] * sd * math.sqrt(1.0 / nb + 1.0 / (n - o)):
        return None

    # ---- onset ----
    M = _window_mean(t, cs, W, W, m)
    p0 = lo + int(np.argmax(M[lo:o + 1]))
    half = 0.5 * (float(M[p0]) + float(S[-1]))
    above = np.nonzero(M[p0:] > half)[0]
    c = min(n - 1, p0 + int(above[-1]) + 1) if len(above) else p0
    p = lo + int(np.argmax(M[lo:c + 1]))
    a = max(lo, int(np.searchsorted(t, t[p] - P["pre_steps"], side="left")))
    hi = min(int(np.searchsorted(t, t[-1] - W, side="right")) - 1, n - 3)
    e = max(int(np.searchsorted(t, t[c] + W, side="right")), c + m,
            int(np.searchsorted(t, t[o] + W, side="right")), o + m)
    e = min(n, e)
    R, mi = P["ramp_lengths"], P["min_ramp_interior"]

    kH, _ = _kink(t, x, a, e, a, max(a, hi), R, mi)
    kA, sA = _kink(t, x, a, n, a, max(a, hi), R, mi)
    k = kA
    if kH is not None and a + 1 < kH <= hi:
        if kA is None:
            k = kH
        else:
            seg = x[a:kH]
            s1 = float(((seg - seg.mean()) ** 2).sum())
            _, s2 = _kink(t, x, kH, n, kH, max(kH, hi), R, mi)
            sB = s1 + s2
            Nf = n - a
            if np.isfinite(sB) and sA - sB > P["bic_penalty"] * math.log(Nf) * sB / Nf:
                k = kH
    date = o if k is None else k
    if k is not None and a < o < k and float(x[a:o].mean()) - float(x[o:k].mean()) >= drop:
        date = o
    return int(round(t[date]))
