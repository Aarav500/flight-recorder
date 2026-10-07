"""Collapse event v2: WHETHER from v1's early-baseline Page CUSUM; WHEN from a free-level piecewise fit.

v1 dated the collapse at the last zero of a CUSUM referenced to the EARLY baseline mu0. That date is
where the series sinks below mu0 - k, not where it turns down: +45 steps late on slow ramps and +50 to
+90 late on rise-then-collapse. v2 keeps v1's test for WHETHER a collapse happened and dates the onset
separately, as the start of the final sustained decline from the level just before it.

INPUT  steps, series: a per-step pass-rate stream (k/8) or a probe (16-sample means every 5 or 10 steps).
       Non-finite pairs are dropped, the rest sorted by step. All lengths are in training-step units
       except where a count of observations is stated.

WHETHER (v1, plus two changes marked *)
  1. Fewer than min_obs observations -> None. Noise sd: sigma = sqrt(sum of squared successive
     differences / (2 (n-1))), floored at sigma_floor.
  2. Baseline window = observations in the first base_frac of the step span; final window = observations
     in the last base_frac of the span; each holds at least base_min_obs observations. mu0 = baseline mean.
     level = mu0 - min_drop. If mean(final window) > level -> None.
  3. Page CUSUM for a downward shift from the first post-baseline observation:
        C_i = max(0, C_{i-1} + (mu0 - x_i) - k),  k = min_drop / 2.
     An excursion is a maximal run of C > 0 starting at observation tau; it must start at or before the
     first observation of the final window. It alarms when
        C > h * sigma * sqrt(min(1, n / h_n_ref))                                              (*)
     (*) The square-root factor only lowers the threshold for series with fewer than h_n_ref observations
     (coarse probes). It is 1 for every series in the harness nulls (T >= 200, one observation per step).
  4. The first alarming excursion with mean(x[tau:]) <= level is a collapse. An alarming excursion that
     fails is skipped; a later excursion (after C returns to 0) can still qualify.
  5. (*) The onset date below must be at or before the first step of the final window, i.e. the dated
     decline must cover the whole final window. A dip confined to the final window is not a collapse.

WHEN (new). Two least-squares models are fit to a window [L, end] of the series:
  TRAPEZOID  flat a up to t0, linear from a at t0 to b at t1, flat b after (continuous; a > b; a, b free).
             t0 < t1 are searched in step units (a grid of span/80 steps, then every step within 2 grid
             cells of the best pair); exact ties in t0 (a step between two sparse observations) take the
             middle tied t0. A pure step is the case t1 - t0 below the observation spacing. At least
             base_min_obs observations at or before t0 and one at or after t1.
  STAIRCASE  level a, then c, then b, with a > c > b; the middle level holds >= base_min_obs observations
             (searched on an observation grid of n/80, then every observation within 2 grid cells).
  Window: start with the whole series. Fit both; L = (the earlier of the trapezoid t0 and the last
  observation before the staircase's first jump) - pre_steps; refit on [L, end]; repeat until the window
  stops changing (at most 3 fits). The pre-change level is therefore free and local: a rise early in the
  run does not drag it down, and the first stage of a two-stage decline stays inside the window.
  Choice: the staircase replaces the trapezoid only if it explains more than stair_pen * ln(m) * s^2 of
  extra sum of squares (m = observations in the window; s = the larger of sigma and the same successive-
  difference sd computed on the window's observations before t1, so a noise-free collapsed tail cannot
  shrink the penalty). This catches two-stage declines, which a single ramp fits by starting it ~25 steps
  too early; ramps and single steps keep the trapezoid.
  DATE  trapezoid: t0, or, when the ramp holds no observation (a plain step), the midpoint between the
        last observation before and the first after the step. Staircase: the midpoint of its first jump.
        Rounded half up to an integer step. (If no fit exists, the date falls back to step[tau].)

CALIBRATION (synthetic series and --null-only only; no declined or real-outcome series were looked at)
  h = 15 and the rest of WHETHER are v1's. h_n_ref = 200: on synthetic flat 16-sample probes every 5 or 10
  steps (T 200/400/800, p 0.2-0.9, 500 reps a cell) the full event fires in <= 2.2% of runs in every cell.
  stair_pen = 3: in units of ln(m) s^2, the staircase's excess fit has p98 <= ~2.5 (max ~3.7) on synthetic
  single steps, ramps and rise-then-fall shapes, and p10 ~ 4.1 on the battery's two-stage shape (stream).
  pre_steps = 100: the battery is unchanged between 50 and 150; a rise straight into the collapse (no
  plateau) is dated about +5 / +10 / +15 steps late at 50 / 100 / 150.
"""
import math

import numpy as np

NAME = "cusum_page_down_v2"

PARAMS = {
    "base_frac": 0.125,    # baseline window = first 12.5% of the step span; final window = last 12.5%
    "base_min_obs": 4,     # min observations in each window, before the onset, and in a staircase middle level
    "min_drop": 0.15,      # absolute drop below mu0 that counts as a collapse; CUSUM reference k = min_drop / 2
    "h": 15.0,             # CUSUM alarm threshold in units of the noise sd
    "h_n_ref": 200,        # threshold scaled by sqrt(min(1, n / h_n_ref)): lower only for sparse series
    "sigma_floor": 0.02,   # lower bound on the noise sd estimate
    "min_obs": 10,         # fewer finite observations than this -> None
    "pre_steps": 100,      # dating: the fit window starts this many steps before the earliest candidate onset
    "stair_pen": 3.0,      # staircase replaces the trapezoid if it explains > stair_pen * ln(m) * sigma^2 more
}


def _clean(steps, series):
    """Finite (step, value) pairs sorted by step (stable), as two lists."""
    try:
        s = np.asarray(steps, dtype=float)
        x = np.asarray(series, dtype=float)
        fast = s.ndim == 1 and s.shape == x.shape
    except (TypeError, ValueError):
        fast = False
    if not fast:                                   # ragged or non-numeric input: pairwise, like v1
        pts = []
        for a, b in zip(steps, series):
            try:
                a, b = float(a), float(b)
            except (TypeError, ValueError):
                continue
            pts.append((a, b))
        s = np.asarray([p[0] for p in pts], dtype=float)
        x = np.asarray([p[1] for p in pts], dtype=float)
    ok = np.isfinite(s) & np.isfinite(x)
    s, x = s[ok], x[ok]
    o = np.argsort(s, kind="stable")
    return s[o].tolist(), x[o].tolist()


def _detect(S, X):
    """WHETHER. Returns (tau, sigma, e) for the qualifying excursion, or None. e = first final-window obs."""
    P = PARAMS
    n = len(X)
    d2 = 0.0
    for i in range(1, n):
        d2 += (X[i] - X[i - 1]) ** 2
    sigma = max(math.sqrt(d2 / (2.0 * (n - 1))), P["sigma_floor"])
    span = S[-1] - S[0]
    cutoff = S[0] + P["base_frac"] * span
    m = 0
    while m < n and S[m] < cutoff:
        m += 1
    m = max(m, P["base_min_obs"])
    e = n
    while e > 0 and S[e - 1] > S[-1] - P["base_frac"] * span:
        e -= 1
    e = min(e, n - P["base_min_obs"])
    if e < m:
        return None
    mu0 = sum(X[:m]) / m
    suf = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suf[i] = suf[i + 1] + X[i]
    level = mu0 - P["min_drop"]
    if suf[e] / (n - e) > level:          # not collapsed at the end: no excursion can qualify
        return None
    k = 0.5 * P["min_drop"]
    thr = P["h"] * sigma * math.sqrt(min(1.0, n / float(P["h_n_ref"])))
    c = 0.0
    start = -1
    checked = False
    for i in range(m, n):
        c += mu0 - X[i] - k
        if c <= 0.0:
            c, start, checked = 0.0, -1, False
            continue
        if start < 0:
            if i > e:                     # an excursion starting inside the final window cannot qualify
                break
            start = i
        if c > thr and not checked:
            checked = True
            if suf[start] / (n - start) <= level:
                return start, sigma, e
    return None


def _prefix(s, x):
    z = np.zeros(1)
    return (np.concatenate([z, np.cumsum(s)]), np.concatenate([z, np.cumsum(s * s)]),
            np.concatenate([z, np.cumsum(x)]), np.concatenate([z, np.cumsum(s * x)]))


def _trap_grid(s, cs, g0, g1, min_pre):
    """Trapezoid fit for every pair t0 < t1 from the grids (step units), in closed form from prefix sums.
    Returns (explained SS, t0, t1, i0, i1) of the best pair with a > b; i0 = #obs <= t0, i1 = #obs < t1."""
    T0, T1 = np.meshgrid(g0, g1, indexing="ij")
    ok = T1 > T0
    T0 = T0[ok]
    T1 = T1[ok]
    if T0.size == 0:
        return None
    n = s.size
    cS, cS2, cX, cSX = cs
    i0 = np.searchsorted(s, T0, side="right")       # A: s <= t0        model a
    i1 = np.searchsorted(s, T1, side="left")        # B: t0 < s < t1    model a r + b q, r = (t1-s)/D, q = 1-r
    nA = i0.astype(float)                           # C: s >= t1        model b
    xA = cX[i0]
    nB = (i1 - i0).astype(float)
    sB = cS[i1] - cS[i0]
    s2B = cS2[i1] - cS2[i0]
    xB = cX[i1] - cX[i0]
    sxB = cSX[i1] - cSX[i0]
    nC = n - i1.astype(float)
    xC = cX[n] - cX[i1]
    D = T1 - T0
    sr2 = (T1 * T1 * nB - 2 * T1 * sB + s2B) / (D * D)
    srx = (T1 * xB - sxB) / D
    sq2 = (s2B - 2 * T0 * sB + T0 * T0 * nB) / (D * D) + nC
    sqx = (sxB - T0 * xB) / D + xC
    srq = ((T1 + T0) * sB - s2B - T1 * T0 * nB) / (D * D)
    su2 = nA + sr2                                  # basis u = 1 on A, r on B ; basis q = q on B, 1 on C
    sux = xA + srx
    with np.errstate(divide="ignore", invalid="ignore"):
        det = su2 * sq2 - srq * srq
        a = (sq2 * sux - srq * sqx) / det
        b = (su2 * sqx - srq * sux) / det
        expl = a * sux + b * sqx
    expl = np.where((i0 >= min_pre) & (i1 < n) & (a > b) & np.isfinite(expl), expl, -np.inf)
    j = int(np.argmax(expl))
    if not np.isfinite(expl[j]):
        return None
    # exact ties (a step between two sparse observations leaves t0 free inside the gap): take the middle one
    tied = np.flatnonzero(expl >= expl[j] - 1e-9 * max(1.0, abs(expl[j])))
    if tied.size > 1:
        order = tied[np.argsort(T0[tied], kind="stable")]
        j = int(order[(order.size - 1) // 2])
    return float(expl[j]), float(T0[j]), float(T1[j]), int(i0[j]), int(i1[j])


def _trap_fit(s, cs, min_pre):
    lo, hi = s[0], s[-1]
    g = max(1.0, math.floor((hi - lo) / 80.0))
    grid = np.arange(lo, hi + 0.5 * g, g)
    r = _trap_grid(s, cs, grid, grid, min_pre)
    if r is None:
        return None
    _, t0, t1, _, _ = r
    f0 = np.arange(max(lo, t0 - 2 * g), min(hi, t0 + 2 * g) + 0.5, 1.0)
    f1 = np.arange(max(lo, t1 - 2 * g), min(hi, t1 + 2 * g) + 0.5, 1.0)
    r2 = _trap_grid(s, cs, f0, f1, min_pre)
    return r2 if (r2 is not None and r2[0] >= r[0]) else r


def _stair_grid(cs, n, j0s, j1s, min_pre, min_mid):
    """Staircase: level a on obs [0, j0), c on [j0, j1), b on [j1, n), a > c > b. Returns (explained, j0, j1)."""
    J0, J1 = np.meshgrid(j0s, j1s, indexing="ij")
    ok = (J1 - J0 >= min_mid) & (J0 >= min_pre) & (J1 < n)
    J0 = J0[ok]
    J1 = J1[ok]
    if J0.size == 0:
        return None
    cX = cs[2]
    xA = cX[J0]
    xB = cX[J1] - cX[J0]
    xC = cX[n] - cX[J1]
    mA = xA / J0
    mB = xB / (J1 - J0)
    mC = xC / (n - J1)
    expl = np.where((mA > mB) & (mB > mC), xA * mA + xB * mB + xC * mC, -np.inf)
    j = int(np.argmax(expl))
    if not np.isfinite(expl[j]):
        return None
    return float(expl[j]), int(J0[j]), int(J1[j])


def _stair_fit(cs, n, min_pre, min_mid):
    g = max(1, n // 80)
    idx = np.arange(0, n + 1, g)
    r = _stair_grid(cs, n, idx, idx, min_pre, min_mid)
    if r is None:
        return None
    _, j0, j1 = r
    f0 = np.arange(max(0, j0 - 2 * g), min(n, j0 + 2 * g) + 1)
    f1 = np.arange(max(0, j1 - 2 * g), min(n, j1 + 2 * g) + 1)
    r2 = _stair_grid(cs, n, f0, f1, min_pre, min_mid)
    return r2 if (r2 is not None and r2[0] >= r[0]) else r


def onset(S, X, sigma):
    """WHEN: start of the final sustained decline from the level just before it (step units, float)."""
    P = PARAMS
    s = np.asarray(S, float)
    x = np.asarray(X, float)
    mp = P["base_min_obs"]
    keep = np.ones(s.size, bool)
    fit = None
    for _ in range(3):
        ss, xx = s[keep], x[keep]
        cs = _prefix(ss, xx)
        tr = _trap_fit(ss, cs, mp)
        if tr is None:
            break
        st = _stair_fit(cs, ss.size, mp, mp)
        fit = (tr, st, ss, xx)
        first = tr[1] if st is None else min(tr[1], ss[st[1] - 1])
        k2 = s >= first - P["pre_steps"]
        if (k2 == keep).all():
            break
        keep = k2
    if fit is None:
        return None
    tr, st, ss, xx = fit
    # the penalty's noise sd: the larger of the whole-series sigma and the sigma of the window before the
    # fitted decline ends (a noise-free collapsed tail must not shrink the penalty for the pre-change part)
    pre = xx[ss < tr[2]]
    if pre.size > 2:
        sigma = max(sigma, math.sqrt(float(np.sum(np.diff(pre) ** 2)) / (2.0 * (pre.size - 1))))
    if st is not None and st[0] - tr[0] > P["stair_pen"] * math.log(ss.size) * sigma * sigma:
        return 0.5 * (ss[st[1] - 1] + ss[st[1]])          # two-stage decline: its first jump
    if tr[3] == tr[4]:                                     # empty ramp: a plain step between two observations
        return 0.5 * (ss[tr[3] - 1] + ss[tr[3]])
    return tr[1]


def event(steps, series):
    S, X = _clean(steps, series)
    if len(S) < PARAMS["min_obs"]:
        return None
    det = _detect(S, X)
    if det is None:
        return None
    tau, sigma, e = det
    t0 = onset(S, X, sigma)
    if t0 is None:
        t0 = S[tau]
    if t0 > S[e]:          # the dated decline must cover the whole final window, else it is a terminal dip
        return None
    return int(math.floor(t0 + 0.5))
