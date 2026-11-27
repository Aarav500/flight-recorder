"""Collapse event: one-sided (downward) Page CUSUM against an early-window baseline.

DEFINITION
  1. Keep the finite (step, value) pairs, sorted by step. Fewer than min_obs -> None.
  2. Noise sd: sigma = sqrt(sum of squared successive differences / (2 (n-1))), floored at sigma_floor.
     (Mean squared successive difference: barely moved by a level shift or a slow trend, and it needs
     no assumption about 8 vs 16 completions or step spacing.)
  3. Baseline window = observations in the first base_frac of the step span; final window = observations
     in the last base_frac of the span; each holds at least base_min_obs observations.
     mu0 = mean of the baseline window.
  4. Page CUSUM for a downward shift, run from the first post-baseline observation:
         C_i = max(0, C_{i-1} + (mu0 - x_i) - k),   k = min_drop / 2  (textbook reference for a shift of min_drop)
     An excursion is a maximal run of C > 0; it starts at observation tau (the first observation after the
     last zero). It alarms when C > h * sigma.
  5. An alarming excursion is a collapse iff ALL of
         tau is at or before the start of the final window      (the collapse covers the whole final window)
         mean(x[tau:])          <= mu0 - min_drop               (post-change mean)
         mean(final window)     <= mu0 - min_drop               (still collapsed at the end)
     The first such excursion is the event. If an alarming excursion fails, scanning continues; a later
     excursion (after C has returned to 0) can still qualify.
  6. DATE = the CUSUM change-point estimate, observation tau (the first observation after the last time C
     was 0 before the alarm), moved back by half the gap to the previous observation so that a coarsely
     sampled probe is not dated systematically late:
         date = step[tau] - floor((step[tau] - step[tau-1]) / 2)
     For unit-spaced series this is step[tau]; for a 10-step probe it is step[tau] - 5.

CALIBRATION (synthetic nulls only)
  h = 15 sd: on flat Binomial(8, p)/8 series, the CUSUM ALONE (before the conditions in 5) raises any
  alarm in at most 2.5% of runs at every p in 0.4..0.7 (the worst levels) and T in {200, 400, 800}
  (worst cell T=200, p=0.5: 0.025, 100k reps); other levels are lower. The conditions in 5 only remove fires.
"""
import math

NAME = "cusum_page_down"

PARAMS = {
    "base_frac": 0.125,    # baseline window = first 12.5% of the step span; final window = last 12.5%
    "base_min_obs": 4,     # each window holds at least this many observations
    "min_drop": 0.15,      # absolute drop that counts as a collapse; CUSUM reference k = min_drop / 2
    "h": 15.0,             # alarm threshold on the CUSUM, in units of the estimated noise sd
    "sigma_floor": 0.02,   # lower bound on the noise sd estimate
    "min_obs": 10,         # fewer finite observations than this -> None
}


def event(steps, series):
    P = PARAMS
    pts = []
    for s, x in zip(steps, series):
        try:
            s, x = float(s), float(x)
        except (TypeError, ValueError):
            continue
        if math.isfinite(s) and math.isfinite(x):
            pts.append((s, x))
    n = len(pts)
    if n < P["min_obs"]:
        return None
    pts.sort(key=lambda t: t[0])
    S = [p[0] for p in pts]
    X = [p[1] for p in pts]

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
    thr = P["h"] * sigma
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
                gap = S[start] - S[start - 1]
                return int(round(S[start] - math.floor(gap / 2.0)))
    return None
