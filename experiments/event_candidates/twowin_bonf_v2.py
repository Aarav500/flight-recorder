"""twowin_bonf_v2: WHETHER from the early baseline (two terminal windows); WHEN from a least-squares fit
whose pre-collapse level is free.

WHY v2. v1 dated the collapse with the same baseline-relative windows it used to decide that a collapse
happened. So it dated the moment the series sank below its STARTING level: late on rise-then-fall (+135)
and on slow ramps. On steps it dated pre-collapse noise excursions (an episode reset only when a window
returned to a high-by-chance baseline). v2 keeps a baseline-relative rule for WHETHER, which is robust
and matches the ground-truth label (first 25 vs last 25), and dates the onset separately.

Lengths are in training-step units and every window holds at least m_min observations, so the same
function serves the per-step stream (8 completions, every step) and the probe (16 completions, every 5 or
10 steps). Noise is estimated from the series itself; nothing assumes n=8, n=16 or unit spacing.
Calibrated only on synthetic series and on `event_harness.py --null-only`.

DEFINITION (PARAMS: W, m_min, alpha, delta)
  Input   finite (step, value) pairs, sorted, duplicate steps averaged. n observations, dt = median gap.
          m = max(ceil(W / dt), m_min) observations per window (W steps on the stream; m_min*dt on a sparse
          probe). n < 4m -> None.
  Noise   sigma^2 = 0.5 * mean(first differences^2)            per-observation noise
          wander(r) = 1.4826 * median|d2| / sqrt(6), d2 = M[i] - 2 M[i+m] + M[i+2m], M[i] = mean of r[i:i+m]
                    (three disjoint m-observation windows, index lags m and 2m on any grid; a linear trend
                    cancels; Var(d2) = 6 Var(mean) under white noise) = sd of an m-observation mean
          e^2     = max(0, wander(r)^2 - sigma^2 / m) with r = the residuals of the fitted collapse model
                    (M1 below, or M2 when its stage is accepted): the slow-wander variance left after the
                    collapse itself is removed (the raw series would count the collapse as wander)
          Var(mean of a segment of k observations) = max(sigma^2, segment sample variance) / k + e^2

  WHETHER  B = the first m observations; T2, T1 = the last two disjoint m-observation windows (T1 ends the
           run). For EACH of T2 and T1:  drop = mean B - mean T >= delta  AND
           z = drop / sqrt(Var B + Var T) >= z_(1 - alpha). Two fixed windows: no scan correction needed.

  WHEN     Least squares on ALL observations. Columns: level h; an optional RISE rho * max(0, a - t) with
           rho <= 0; the DECLINE beta * g(t) with beta < 0, g = clip((t - tau) / L, 0, 1) (a linear fall
           from the plateau starting at tau, lasting L steps, then a floor) or g = [t >= tau] (a step, L = 0).
    M1     tau (>= the 3rd observation's step), L by grid search: the step at every observed step; ramps
           of L >= 4 steps holding >= 3 observations, on a coarse grid (L x1.3 from 4 to the span; tau every
           max(4, L/8) steps) refined around its optimum (L x1.03 within one coarse ratio; tau every
           max(1, L/60) steps within +-max(6, L/5)).
           A ramp replaces the step only if SSE_ramp + ln(n) sigma^2 < SSE_step (BIC: one more parameter).
           Then a over all observed steps <= tau, then (tau, L) again given a.
    M2     M1 plus an earlier sustained STAGE gamma * [t >= c], gamma < 0, a < c < tau, with >= m
           observations on the plateau before c and >= m at the stage level. c and the decline are fitted
           alternately from three starts: M1's decline; the best single step as the decline; the best single
           step as the stage. The stage is accepted iff
             SSE(M1) - SSE(M2) > 2 ln(n) sigma^2                      (BIC: two more parameters)
             d1 = mean(plateau obs in [a, c)) - mean(obs from c to the last one before the decline) >= delta
             d1 / sqrt(Var plateau + Var stage) >= z_(1 - alpha / K)  (Bonferroni over the K candidate c;
                                                                         e^2 from M2's residuals)
    ONSET  c if the stage is accepted, else M1's tau: the start of the final sustained decline, measured
           from the level that precedes it. A step's change lies between two observations; it is reported
           as the first low observation's step moved back by floor(gap / 2) (no shift at unit spacing).

  ORDER    first step of T2 >= ONSET > step of the last baseline observation: the dated decline has lasted
           >= 2m observations (2W steps on the stream), all inside the two confirmation windows, and it
           began after the baseline was measured (else the baseline is not a pre-collapse level).

  FIRES iff WHETHER and ORDER. event() returns int(round(ONSET)) or None.

NOT DETECTED, by design: a collapse from a peak to a level still above the early baseline. The ground-truth
label (last 25 vs first 25) calls such a run not declined. A variant that measured WHETHER from the fitted
pre-onset plateau detected those collapses and kept the harness nulls, but it roughly doubled the
false-fire rate on synthetic slow-wander nulls (logit AR(1)), so it is not used.
"""
import math
from statistics import NormalDist

import numpy as np

NAME = "twowin_bonf_v2"
PARAMS = {"W": 20, "m_min": 4, "alpha": 0.05, "delta": 0.15}

_ND = NormalDist()
_COARSE_RATIO, _COARSE_TAU = 1.3, 4.0      # numerical search grid (accuracy, not a modelling choice)
_FINE_RATIO, _FINE_TAU = 1.03, 6


# ----------------------------------------------------------------------------------------------- input
def _clean(steps, series):
    try:                                               # fast path: plain numbers (None becomes NaN)
        s = np.asarray(steps, dtype=float)
        x = np.asarray(series, dtype=float)
        if s.ndim != 1 or x.ndim != 1:
            raise ValueError
        k = min(len(s), len(x))
        s, x = s[:k], x[:k]
        ok = np.isfinite(s) & np.isfinite(x)
        return _sort_dedup(s[ok], x[ok])
    except (TypeError, ValueError):
        pass
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
    return _sort_dedup(np.asarray(ss, float), np.asarray(xs, float))


def _sort_dedup(s, x):
    if len(s) == 0:
        return None, None
    if len(s) > 1 and np.any(np.diff(s) <= 0):
        o = np.argsort(s, kind="stable")
        s, x = s[o], x[o]
        if np.any(np.diff(s) == 0):
            u, inv = np.unique(s, return_inverse=True)
            x = np.bincount(inv, weights=x) / np.bincount(inv)
            s = u
    return s, x


def _cs(v):
    return np.concatenate([[0.0], np.cumsum(v)])


_GEOM_CACHE = {}


def _geom(lo, hi, ratio):
    key = (round(lo, 6), round(hi, 6), ratio)
    if key not in _GEOM_CACHE:
        if len(_GEOM_CACHE) > 4096:
            _GEOM_CACHE.clear()
        _GEOM_CACHE[key] = _geom_build(lo, hi, ratio)
    return _GEOM_CACHE[key]


def _geom_build(lo, hi, ratio):
    out, L = [], lo
    while L <= hi * (1 + 1e-9):
        out.append(L)
        L *= ratio
    return np.unique(np.round(np.asarray(out, float), 3))


# ----------------------------------------------------------------------------------------------- least squares
class _LS:
    """Least squares of x on structured columns, one column scanned at a time (Frisch-Waugh-Lovell).
    Columns: 1;  u_a = max(0, a - t);  s_c = [t >= c];  g = clip((t - tau) / L, 0, 1) or [t >= tau].
    With a <= c <= tau: u.s = u.g = 0 and s.g = sum g."""

    def __init__(self, t, x):
        self.t, self.x, self.n = t, x, len(x)
        self.c1 = np.arange(self.n + 1, dtype=float)
        self.ct, self.ctt, self.cx, self.cxt = _cs(t), _cs(t * t), _cs(x), _cs(x * t)
        self.Sx, self.Sxx = float(self.cx[-1]), float((x * x).sum())

    def g_sums(self, tau, L):
        """tau, L broadcastable arrays (L > 0 everywhere). Returns G1 = sum g, G2 = sum g^2, XG = sum x g."""
        t, n = self.t, self.n
        tau, L = np.broadcast_arrays(np.asarray(tau, float), np.asarray(L, float))
        r0 = np.searchsorted(t, tau.ravel(), side="right").reshape(tau.shape)
        k = np.searchsorted(t, (tau + L).ravel(), side="left").reshape(tau.shape)
        Nr = self.c1[k] - self.c1[r0]
        St = self.ct[k] - self.ct[r0]
        Stt = self.ctt[k] - self.ctt[r0]
        Sxr = self.cx[k] - self.cx[r0]
        Sxt = self.cxt[k] - self.cxt[r0]
        G1 = (St - tau * Nr) / L + (n - k)
        G2 = (Stt - 2.0 * tau * St + tau * tau * Nr) / (L * L) + (n - k)
        XG = (Sxt - tau * Sxr) / L + (self.Sx - self.cx[k])
        self.last_inside = Nr                          # observations strictly inside the ramp
        return G1, G2, XG

    def step_sums(self, c):
        k = np.searchsorted(self.t, c, side="left")
        S = (self.n - k).astype(float)
        return S, S, self.Sx - self.cx[k]

    def col_sums(self, col):
        if col[0] == "1":
            return float(self.n), float(self.n), self.Sx
        if col[0] == "u":
            a = col[1]
            ia = int(np.searchsorted(self.t, a, side="left"))
            return (ia * a - self.ct[ia], ia * a * a - 2.0 * a * self.ct[ia] + self.ctt[ia],
                    a * self.cx[ia] - self.cxt[ia])
        if col[0] == "s" or col[2] <= 0:
            S, _, XS = self.step_sums(np.array([col[1]]))
            return float(S[0]), float(S[0]), float(XS[0])
        G1, G2, XG = self.g_sums(np.array([col[1]]), np.array([col[2]]))
        return float(G1[0]), float(G2[0]), float(XG[0])

    def gram(self, cols):
        sums = [self.col_sums(c) for c in cols]
        p = len(cols)
        M, v = np.zeros((p, p)), np.zeros(p)
        for i in range(p):
            M[i, i], v[i] = sums[i][1], sums[i][2]
            for j in range(i):
                ki, kj = cols[i][0], cols[j][0]
                if kj == "1":
                    val = sums[i][0]
                elif {ki, kj} == {"s", "g"}:
                    val = sums[i][0] if ki == "g" else sums[j][0]
                else:
                    val = 0.0
                M[i, j] = M[j, i] = val
        return M, v

    def prep(self, fixed):
        ZtZ, Ztx = self.gram(fixed)
        try:
            A = np.linalg.inv(ZtZ)
        except np.linalg.LinAlgError:
            A = np.linalg.pinv(ZtZ)
        Ax = A @ Ztx
        return A, Ax, self.Sxx - float(Ztx @ Ax)

    def fwl(self, pre, Ztc, cc, cx):
        """Candidate column c (vectorised: Z'c (K,p), c'c (K,), c'x (K,)) added to the fixed columns
        (pre = prep(fixed)). Returns SSE (K,), the candidate's coefficient (K,), the fixed coefficients (K,p)."""
        A, Ax, sse_z = pre
        q = Ztc @ A
        ct = cc - np.einsum("kp,kp->k", q, Ztc)
        xt = cx - Ztc @ Ax
        ok = ct > 1e-9 * self.n
        coef = xt / np.where(ok, ct, 1.0)
        sse = sse_z - xt * coef
        sse = np.where(ok & (sse > -1e-9), sse, np.inf)  # an SSE below 0 is round-off: reject it
        return sse, coef, Ax[None, :] - q * coef[:, None]


def _signs_ok(fixed, gam):
    ok = np.ones(gam.shape[0], bool)
    for i, col in enumerate(fixed):
        if col[0] == "u":
            ok &= gam[:, i] <= 0
        elif col[0] in ("s", "g"):
            ok &= gam[:, i] < 0
    return ok


def _ramp_candidates(lo, hi, Ls, base, frac):
    """Concatenated (tau, L) candidates: for each L, tau = lo, lo + st, ... <= hi with st = max(base, L * frac)."""
    st = np.maximum(base, Ls * frac)
    cnt = np.floor((hi - lo) / st + 1e-9).astype(int) + 1
    cnt = np.where(hi >= lo, cnt, 0)
    tot = int(cnt.sum())
    if tot == 0:
        return np.zeros(0), np.zeros(0)
    off = np.repeat(np.cumsum(cnt) - cnt, cnt)
    idx = np.arange(tot) - off
    return lo + idx * np.repeat(st, cnt), np.repeat(Ls, cnt)


def _scan_decline(F, fixed, lo, ramps=True):
    """Best decline column g (tau >= lo) given the fixed columns: (sse, tau, L), or (inf, None, None).
    Step model at every observed step; ramps on a coarse grid (L x1.3 from 4 steps to the span, tau every
    max(4, L/8) steps), then a fine grid around the coarse optimum (L x1.03 within one coarse ratio,
    tau every max(1, L/60) steps within +-max(6, L/5))."""
    t = F.t
    kinds = [c[0] for c in fixed]
    pre = F.prep(fixed)

    def evaluate(tau, L, step):
        G1, G2, XG = F.step_sums(tau) if step else F.g_sums(tau, L)
        Ztc = np.stack([G1 if k in ("1", "s") else np.zeros_like(G1) for k in kinds], axis=-1)
        sse, beta, gam = F.fwl(pre, Ztc, G2, XG)
        ok = (beta < 0) & _signs_ok(fixed, gam)
        if not step:
            ok &= F.last_inside >= 3                   # a ramp must hold >= 3 observations (sparse grids)
        return np.where(ok, sse, np.inf)

    step = (np.inf, None, None)
    tau_obs = t[t >= lo]
    if len(tau_obs):
        s = evaluate(tau_obs, None, True)
        j = int(np.argmin(s))
        if np.isfinite(s[j]):
            step = (float(s[j]), float(tau_obs[j]), 0.0)
    hi = t[-1] - 1.0
    if not ramps or hi < lo:
        return step
    span = t[-1] - t[0]
    tau, L = _ramp_candidates(lo, hi, _geom(4.0, max(span, 4.0), _COARSE_RATIO), _COARSE_TAU, 1.0 / 8.0)
    s = evaluate(tau, L, False)
    j = int(np.argmin(s))
    if not np.isfinite(s[j]):
        return step
    ramp = (float(s[j]), float(tau[j]), float(L[j]))
    tau0, L0 = ramp[1], ramp[2]
    w = max(float(_FINE_TAU), L0 / 5.0)
    tau, L = _ramp_candidates(max(lo, tau0 - w), min(hi, tau0 + w),
                              _geom(max(4.0, L0 / _COARSE_RATIO), L0 * _COARSE_RATIO, _FINE_RATIO), 1.0, 1.0 / 60.0)
    if len(tau):
        s = evaluate(tau, L, False)
        j = int(np.argmin(s))
        if s[j] < ramp[0]:
            ramp = (float(s[j]), float(tau[j]), float(L[j]))
    # a ramp has one more parameter than a step (its length): BIC, ln(n) sigma^2
    return ramp if ramp[0] + F.ramp_pen < step[0] else step


def _scan_rise(F, fixed, agrid):
    t = F.t
    ia = np.searchsorted(t, agrid, side="left")
    U1 = ia * agrid - F.ct[ia]
    U2 = ia * agrid * agrid - 2.0 * agrid * F.ct[ia] + F.ctt[ia]
    XU = agrid * F.cx[ia] - F.cxt[ia]
    Ztc = np.stack([U1 if c[0] == "1" else np.zeros_like(U1) for c in fixed], axis=-1)
    sse, rho, gam = F.fwl(F.prep(fixed), Ztc, U2, XU)
    sse = np.where((rho <= 0) & _signs_ok(fixed, gam), sse, np.inf)
    j = int(np.argmin(sse))
    return float(sse[j]), float(agrid[j])


def _scan_stage(F, fixed, cgrid):
    g = [c for c in fixed if c[0] == "g"][0]
    G1 = F.col_sums(g)[0]
    S, _, XS = F.step_sums(cgrid)
    Ztc = np.stack([S if c[0] == "1" else (np.full_like(S, G1) if c[0] == "g" else np.zeros_like(S))
                    for c in fixed], axis=-1)
    sse, gamma, gam = F.fwl(F.prep(fixed), Ztc, S, XS)
    sse = np.where((gamma < 0) & _signs_ok(fixed, gam), sse, np.inf)
    j = int(np.argmin(sse))
    return float(sse[j]), float(cgrid[j])


# ----------------------------------------------------------------------------------------------- noise
def _wander_sd(x, m):
    n = len(x)
    if n < 3 * m + 3:
        return 0.0
    cs = _cs(x)
    M = (cs[m:] - cs[:-m]) / m                       # M[i] = mean of x[i : i + m]
    d2 = M[2 * m:] - 2.0 * M[m:-m] + M[:-2 * m]      # windows [i, i+m), [i+m, i+2m), [i+2m, i+3m)
    return 1.4826 * float(np.median(np.abs(d2))) / math.sqrt(6.0)


def _seg_var(v, sigma2, e2):
    k = len(v)
    s2 = float(v.var(ddof=1)) if k > 1 else 0.0
    return max(sigma2, s2) / k + e2


def _design(t, cols):
    out = []
    for c in cols:
        if c[0] == "1":
            out.append(np.ones_like(t))
        elif c[0] == "u":
            out.append(np.maximum(0.0, c[1] - t))
        elif c[0] == "s" or c[2] <= 0:
            out.append((t >= c[1]).astype(float))
        else:
            out.append(np.clip((t - c[1]) / c[2], 0.0, 1.0))
    return np.stack(out, axis=1)


def _resid_excess(t, x, cols, m, sigma2):
    """Wander excess e^2 of the residuals of the fitted model (the collapse itself removed)."""
    Z = _design(t, cols)
    r = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    f = _wander_sd(r, m)
    return max(0.0, f * f - sigma2 / m)


def _half_gap(t, c):
    i = int(np.searchsorted(t, c, side="left"))
    g = t[i] - t[i - 1] if i > 0 else 0.0
    return t[i] - math.floor(g / 2.0)


# ----------------------------------------------------------------------------------------------- onset
def _onset(t, x, m, sigma2, alpha, delta):
    n = len(x)
    F = _LS(t, x)
    F.ramp_pen = math.log(n) * sigma2
    lo0 = t[2]
    one = [("1",)]
    # M1
    sse1, tau, L = _scan_decline(F, one, lo0)
    if tau is None:
        return None, None
    a = None
    agrid = t[(t > t[1]) & (t <= tau)]
    if len(agrid):
        sse_a, a_new = _scan_rise(F, one + [("g", tau, L)], agrid)
        if sse_a < sse1 - 1e-12:
            sse_n, tau_n, L_n = _scan_decline(F, one + [("u", a_new)], max(lo0, a_new))
            if tau_n is not None and sse_n <= sse_a + 1e-9:
                a, sse1, tau, L = a_new, sse_n, tau_n, L_n
    info = {"a": a, "tau": tau, "L": L, "stage": None}
    date = _half_gap(t, tau) if L <= 0 else tau          # a ramp always holds >= 3 observations
    rise = [("u", a)] if a is not None else []
    info["e2"] = _resid_excess(t, x, one + rise + [("g", tau, L)], m, sigma2)
    # M2
    i_a = int(np.searchsorted(t, a, side="left")) if a is not None else 0
    _, tau_s, _ = _scan_decline(F, one + rise, max(lo0, a or lo0), ramps=False)

    def stage_given_decline(tg, Lg):
        i_t = int(np.searchsorted(t, tg, side="right" if Lg > 0 else "left"))
        ci = np.arange(i_a + m, i_t - m + 1)
        if len(ci) == 0:
            return None
        sse_c, c = _scan_stage(F, one + rise + [("g", tg, Lg)], t[ci])
        return (sse_c, c, len(ci), tg, Lg) if np.isfinite(sse_c) else None

    def decline_given_stage(c, K):
        k_c = int(np.searchsorted(t, c, side="left"))
        if k_c - i_a < m or k_c + m >= n:
            return None
        sse_g, tg, Lg = _scan_decline(F, one + rise + [("s", c)], t[k_c + m])
        return (sse_g, c, K, tg, Lg) if tg is not None else None

    # alternate stage <-> decline from three starts: decline = M1's; decline = the best single step;
    # stage = the best single step (then the decline after it)
    best = (np.inf, None, None, None, None)
    starts = [("g", tau, L)]
    if tau_s is not None:
        if (tau_s, 0.0) != (tau, L):
            starts.append(("g", tau_s, 0.0))
        starts.append(("s", tau_s))
    for st in starts:
        if st[0] == "g":
            s1 = stage_given_decline(st[1], st[2])
            s2 = decline_given_stage(s1[1], s1[2]) if s1 else None
            s3 = stage_given_decline(s2[3], s2[4]) if s2 else None
            cands = (s1, s2, s3)
        else:
            s1 = decline_given_stage(st[1], 0)
            cands = (stage_given_decline(s1[3], s1[4]) if s1 else None,)   # K comes from a stage scan
        for s_ in cands:
            if s_ is not None and s_[0] < best[0]:
                best = s_
    sse2, c, K, tau2, L2 = best
    if c is not None and sse1 - sse2 > 2.0 * math.log(n) * sigma2:
        k_c = int(np.searchsorted(t, c, side="left"))
        i_t = int(np.searchsorted(t, tau2, side="right" if L2 > 0 else "left"))
        pre, mid = x[i_a:k_c], x[k_c:i_t]
        if len(pre) >= m and len(mid) >= m:
            e2r = _resid_excess(t, x, one + rise + [("s", c), ("g", tau2, L2)], m, sigma2)
            d1 = float(pre.mean() - mid.mean())
            z1 = d1 / math.sqrt(_seg_var(pre, sigma2, e2r) + _seg_var(mid, sigma2, e2r))
            z_need = _ND.inv_cdf(1.0 - alpha / max(K, 1))
            info["stage"] = {"c": c, "d1": d1, "z1": z1, "z_need": z_need, "tau2": tau2, "L2": L2}
            if d1 >= delta and z1 >= z_need:
                date = _half_gap(t, c)
                info["e2"] = e2r
    return int(round(date)), info


# ----------------------------------------------------------------------------------------------- entry
def scan(steps, series, W=None, m_min=None, alpha=None, delta=None):
    """Return (fires, onset_or_None, details). event() is the harness entry point."""
    W = float(PARAMS["W"] if W is None else W)
    m_min = int(PARAMS["m_min"] if m_min is None else m_min)
    alpha = PARAMS["alpha"] if alpha is None else alpha
    delta = PARAMS["delta"] if delta is None else delta
    t, x = _clean(steps, series)
    if t is None or len(t) < 8:
        return False, None, None
    n = len(t)
    dt = float(np.median(np.diff(t)))
    m = max(int(math.ceil(W / dt - 1e-9)), m_min)
    if n < 4 * m:
        return False, None, None
    d = np.diff(x)
    sigma2 = 0.5 * float(np.mean(d * d))
    B, T2, T1 = x[:m], x[n - 2 * m:n - m], x[n - m:]
    det = {"m": m, "baseline": float(B.mean()), "sigma": math.sqrt(sigma2)}
    drops = {"T2": float(B.mean() - T2.mean()), "T1": float(B.mean() - T1.mean())}
    det.update({"drop_" + k: v for k, v in drops.items()})
    if min(drops.values()) < delta:
        return False, None, det
    date, info = _onset(t, x, m, sigma2, alpha, delta)
    det["fit"], det["onset"] = info, date
    if date is None:
        return False, None, det
    e2 = info["e2"]
    det["wander_excess"] = math.sqrt(e2)
    vB = _seg_var(B, sigma2, e2)
    z_one = _ND.inv_cdf(1.0 - alpha)
    for name, Tk in (("T2", T2), ("T1", T1)):
        se = math.sqrt(vB + _seg_var(Tk, sigma2, e2))
        z = drops[name] / se if se > 0 else math.inf
        det["z_" + name] = z
        if z < z_one:
            return False, None, det
    if date > t[n - 2 * m] or date <= t[m - 1]:
        return False, None, det
    return True, date, det


def event(steps, series):
    fires, onset, _ = scan(steps, series)
    return onset if fires else None
