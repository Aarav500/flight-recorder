"""twowin_bonf: fixed baseline window vs. trailing windows, Bonferroni over scanned positions,
practical-drop floor, a later disjoint confirmation window, onset dated at the start of the terminal
significantly-low stretch.

All lengths are in training-step units, so the same function applies to the per-step stream
(8 completions, every step) and to the probe (16 completions, every 5 or 10 steps). Noise is estimated
from the series itself; nothing assumes n=8, n=16 or unit spacing.

DEFINITION (W, alpha, delta, floor_lag_frac are PARAMS)
  s0 = first observed step.
  Baseline     B    = observations with step in [s0, s0 + W).
  Trailing     T(e) = observations with step in (e - W, e], one per observed step e whose window shares no
                      observation with B. These are the K scanned positions.
  Noise        sigma^2 = 0.5 * mean(first differences^2)    (per-observation noise; immune to level shifts)
               f = robust sd of a W-step mean that also includes slow wander / autocorrelation:
                   second differences of overlapping L-step window means at lags L and 2L (L = floor_lag_frac*W),
                   f_L = 1.4826 * median|d2| / sqrt(6),  scaled to W as f = f_L * sqrt(L / W)
               s_w^2 = the window's own sample variance (level-dependent binomial noise, within-window trend)
               Var(mean of a window holding n observations) = max(sigma^2 / n, s_w^2 / n, f^2)
  Test         z(e) = (mean B - mean T(e)) / sqrt(Var_B + Var_T(e))       one-sided: "T(e) is below B"
               drop(e) = mean B - mean T(e)
  FLAGGED      z(e) >= Phi^-1(1 - alpha / K)  (Bonferroni over all K positions)  AND  drop(e) >= delta
  CONFIRMED    the FINAL trailing window (ending at the last observation) has z >= Phi^-1(1 - alpha)
               AND drop >= delta: the series is still materially below its baseline when training ends.
  Terminal episode: positions after the last position whose window mean was back at/above the baseline
               mean (z <= 0); the whole scan if there is none.
  FIRES iff CONFIRMED and some FLAGGED position in the terminal episode whose window ENDS BEFORE the final
               window STARTS (a separate, later confirmation window: >= 2W steps of evidence below baseline).
  ONSET DATE   the first position e of the terminal episode with z >= Phi^-1(1 - alpha) (the stretch's first
               significantly-low window), reported as that window's nominal START step, e - W + 1 (step units,
               so stream and probe are anchored identically whatever the probe spacing). Dating at the
               window's end would lag the onset by construction.
  Returns None when it does not fire, and for series too short to hold a baseline plus a window.
"""
import math
from statistics import NormalDist

import numpy as np

NAME = "twowin_bonf"
PARAMS = {"W": 25, "alpha": 0.05, "delta": 0.15, "floor_lag_frac": 0.5}

_ND = NormalDist()


def _clean(steps, series):
    """Finite (step, value) pairs, sorted by step, duplicate steps averaged. None/NaN/garbage dropped."""
    ss, xs = [], []
    for s, x in zip(steps, series):
        try:
            s = float(s)
            x = float(x)
        except (TypeError, ValueError):
            continue
        if math.isfinite(s) and math.isfinite(x):
            ss.append(s)
            xs.append(x)
    if not ss:
        return None, None
    s = np.asarray(ss, float)
    x = np.asarray(xs, float)
    if len(s) > 1 and np.any(np.diff(s) <= 0):
        o = np.argsort(s, kind="stable")
        s, x = s[o], x[o]
        if np.any(np.diff(s) == 0):
            u, inv = np.unique(s, return_inverse=True)
            x = np.bincount(inv, weights=x) / np.bincount(inv)
            s = u
    return s, x


def _wander_sd(s, cs, W, L):
    """Robust sd of a W-step mean including low-frequency noise, from second differences of overlapping
    L-step window means at lags L and 2L (a linear trend cancels; a level shift touches few terms)."""
    lo = np.searchsorted(s, s - L, side="right")
    e = np.nonzero((s - L) >= (s[0] - 1))[0]          # windows (e-L, e] inside the observed range
    if len(e) < 3:
        return 0.0
    m = (cs[e + 1] - cs[lo[e]]) / (e + 1 - lo[e])
    se_ = s[e]
    i1 = np.searchsorted(se_, se_ - L, side="right") - 1
    i2 = np.searchsorted(se_, se_ - 2 * L, side="right") - 1
    ok = i2 >= 0
    if ok.sum() < 3:
        return 0.0
    d2 = m[ok] - 2.0 * m[i1[ok]] + m[i2[ok]]
    return 1.4826 * float(np.median(np.abs(d2))) / math.sqrt(6.0) * math.sqrt(L / W)


def scan(steps, series, W=None, alpha=None, delta=None, floor_lag_frac=None):
    """Return (fires, onset_step_or_None, details). event() is the harness entry point."""
    W = float(PARAMS["W"] if W is None else W)
    alpha = PARAMS["alpha"] if alpha is None else alpha
    delta = PARAMS["delta"] if delta is None else delta
    flf = PARAMS["floor_lag_frac"] if floor_lag_frac is None else floor_lag_frac
    s, x = _clean(steps, series)
    if s is None or len(s) < 6:
        return False, None, None
    nb = int(np.searchsorted(s, s[0] + W, side="left"))          # baseline = indices [0, nb)
    if nb < 1 or nb >= len(s):
        return False, None, None
    cs = np.concatenate([[0.0], np.cumsum(x)])
    lo = np.searchsorted(s, s - W, side="right")                 # T(s_j) = indices [lo_j, j]
    j = np.nonzero(lo >= nb)[0]                                  # no overlap with the baseline
    if len(j) == 0:
        return False, None, None
    lo = lo[j]
    nw = (j + 1 - lo).astype(float)
    mw = (cs[j + 1] - cs[lo]) / nw
    mb = cs[nb] / nb
    d = np.diff(x)
    sig2 = 0.5 * float(np.mean(d * d))
    f2 = _wander_sd(s, cs, W, flf * W) ** 2
    c2 = np.concatenate([[0.0], np.cumsum(x * x)])
    s2b = max(c2[nb] / nb - (cs[nb] / nb) ** 2, 0.0) * nb / max(nb - 1, 1)
    s2w = np.maximum((c2[j + 1] - c2[lo]) / nw - mw * mw, 0.0) * nw / np.maximum(nw - 1, 1)
    se = np.sqrt(max(sig2 / nb, f2, s2b / nb) + np.maximum(np.maximum(sig2 / nw, f2), s2w / nw))
    drop = mb - mw
    z = np.where(se > 0, drop / np.where(se > 0, se, 1.0), np.where(drop > 0, np.inf, 0.0))
    K = len(j)
    z_scan = _ND.inv_cdf(1.0 - alpha / K)                        # Bonferroni over the K scanned positions
    z_one = _ND.inv_cdf(1.0 - alpha)                             # single pre-specified comparison
    det = {"K": K, "z_scan": z_scan, "z_final": float(z[-1]), "drop_final": float(drop[-1]),
           "baseline": float(mb), "sigma": math.sqrt(sig2), "wander_sd": math.sqrt(f2)}
    if not (z[-1] >= z_one and drop[-1] >= delta):               # confirmation window = final window
        return False, None, det
    rec = np.nonzero(z <= 0)[0]
    start = int(rec[-1]) + 1 if len(rec) else 0                  # terminal episode
    flagged = (z >= z_scan) & (drop >= delta) & (s[j] < s[lo[-1]])   # ends before the final window starts
    if not flagged[start:].any():
        return False, None, det
    first = start + int(np.argmax(z[start:] >= z_one))           # first significantly-low window of the episode
    return True, int(round(s[j[first]] - W + 1)), det            # its nominal start step, e - W + 1


def event(steps, series):
    fires, onset, _ = scan(steps, series)
    return onset if fires else None
