"""Independent recompute of the E1b n = 48 design numbers. numpy/scipy only; no e1_primary import."""
import json
import math
from pathlib import Path

import numpy as np
from scipy import integrate, stats

HERE = Path(__file__).resolve().parent
N = 48


def fp_at(n):
    v = np.arange(n, dtype=float)              # distinct maxima
    thr = np.percentile(v, 90)
    return int((v > thr).sum())


def fisher_p(tp, nh, fp, nt):
    # one-sided P(X >= tp), X ~ hypergeom(total nh+nt, successes tp+fp, draws nh)
    return float(stats.hypergeom(nh + nt, tp + fp, nh).sf(tp - 1))


def hits_needed(nh, nt, fp, alpha):
    return min(x for x in range(nh + 1) if fisher_p(x, nh, fp, nt) < alpha)


def power(nh, nt, fp, tpr, alpha=0.05):
    k = hits_needed(nh, nt, fp, alpha)
    return float(stats.binom.sf(k - 1, nh, tpr))


def mde(nh, nt, fp, target=0.80, alpha=0.05, step=0.01):
    for t in np.arange(step, 1 + 1e-9, step):
        if power(nh, nt, fp, round(t, 4), alpha) >= target:
            return round(t, 4)


def wilson(x, n, z):
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z / d * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return c - h, c + h


def newcombe_upper(x1, n1, x2, n2, conf=0.90):
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    p1, p2 = x1 / n1, x2 / n2
    l1, u1 = wilson(x1, n1, z)
    l2, u2 = wilson(x2, n2, z)
    return (p1 - p2) + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def nonsup_max(nh, nt, fp, delta):
    ok = [x for x in range(nh + 1) if newcombe_upper(x, nh, fp, nt) < delta]
    return max(ok)


def thr_estimated_prob(n, event_fn):
    """thr = U(j) + fr (U(j+1) - U(j)), j = floor(0.9 (n-1)) + 1 (1-indexed). U(j) ~ Beta(j, n-j+1);
    given U(j) = u, U(j+1) = u + (1-u) W, W ~ Beta(1, n-j). Integrate event(1 - thr) over (u, w)."""
    pos = 0.9 * (n - 1)
    j = math.floor(pos) + 1
    fr = pos - math.floor(pos)
    bu = stats.beta(j, n - j + 1)
    bw = stats.beta(1, n - j)

    def f(w, u):
        thr = u + fr * (1 - u) * w
        return bu.pdf(u) * bw.pdf(w) * event_fn(1 - thr)
    val, _ = integrate.dblquad(f, 0, 1, 0, 1, epsabs=1e-11, epsrel=1e-11)
    return val


def thr_estimated_mc(n, nsim, rng):
    pos = 0.9 * (n - 1)
    j = math.floor(pos) + 1
    fr = pos - math.floor(pos)
    u = rng.beta(j, n - j + 1, nsim)
    w = rng.beta(1, n - j, nsim)
    thr = u + fr * (1 - u) * w
    tp = rng.binomial(n, 1 - thr)
    return tp


def main():
    out = []
    P = out.append
    fp = fp_at(N)
    P("FP at n=48 (numpy 90th pct on distinct values): %d  -> %d/48 = %.4f" % (fp, fp, fp / N))
    P("fixed_fp for n 24..48: %s" % {n: fp_at(n) for n in range(24, 49)})
    k = hits_needed(N, N, fp, 0.05)
    P("hits needed p<0.05: %d (p at %d = %.4f, at %d = %.4f); scipy fisher_exact check at %d: %.4f" % (
        k, k, fisher_p(k, N, fp, N), k - 1, fisher_p(k - 1, N, fp, N), k,
        stats.fisher_exact([[k, N - k], [fp, N - fp]], alternative="greater")[1]))
    P("kill at TPR %d/48 = %.3f" % (k, k / N))
    for t in (0.32, 0.33, 0.35, 0.375, 0.40, 0.5):
        P("  Fisher power at TPR %.3f: %.4f ; x 0.963 = %.4f" % (t, power(N, N, fp, t), power(N, N, fp, t) * 0.963))
    P("MDE 80%% (0.01 grid) %.2f ; 90%% %.2f ; 80%% (0.005 grid) %.3f ; 90%% (0.005) %.3f" % (
        mde(N, N, fp), mde(N, N, fp, 0.90), mde(N, N, fp, step=0.005), mde(N, N, fp, 0.90, step=0.005)))
    P("size, rate known: P(Bin(48, 5/48) >= %d) = %.5f" % (k, stats.binom.sf(k - 1, N, fp / N)))
    # non-superiority
    for x in range(8, 14):
        P("  Newcombe 90%% upper at TP %d (FP 5/48): %.5f" % (x, newcombe_upper(x, N, fp, N)))
    ns = nonsup_max(N, N, fp, 0.25)
    ns20 = nonsup_max(N, N, fp, 0.20)
    P("non-sup at 0.25 holds iff TP <= %d; at 0.20 iff TP <= %d" % (ns, ns20))
    P("P(non-sup shown | TPR = FPR, rate known): 0.25 -> %.4f ; 0.20 -> %.4f" % (
        stats.binom.cdf(ns, N, fp / N), stats.binom.cdf(ns20, N, fp / N)))
    for t in (0.25, 0.30):
        inc25 = stats.binom.cdf(k - 1, N, t) - stats.binom.cdf(ns, N, t)
        inc20 = stats.binom.cdf(k - 1, N, t) - stats.binom.cdf(ns20, N, t)
        P("  inconclusive at TPR %.2f: delta 0.25 %.4f ; delta 0.20 %.4f" % (t, inc25, inc20))
    # threshold estimated
    tot = thr_estimated_prob(N, lambda q: 1.0)
    size_est = thr_estimated_prob(N, lambda q: stats.binom.sf(k - 1, N, q))
    ns_est = thr_estimated_prob(N, lambda q: stats.binom.cdf(ns, N, q))
    ns20_est = thr_estimated_prob(N, lambda q: stats.binom.cdf(ns20, N, q))
    m1 = thr_estimated_prob(N, lambda q: N * q)
    m2 = thr_estimated_prob(N, lambda q: N * q * (1 - q) + (N * q) ** 2)
    P("threshold estimated (quadrature, Beta order-stat route): total %.6f size %.4f P(ns 0.25) %.4f P(ns 0.20) %.4f mean TP %.3f var %.3f (binomial var at mean %.3f)" % (
        tot, size_est, ns_est, ns20_est, m1, m2 - m1 ** 2, m1 * (1 - m1 / N)))
    rng = np.random.default_rng(777)
    tp = thr_estimated_mc(N, 20_000_000, rng)
    P("threshold estimated (MC 2e7): size %.4f P(ns 0.25) %.4f P(ns 0.20) %.4f mean %.3f var %.3f" % (
        (tp >= k).mean(), (tp <= ns).mean(), (tp <= ns20).mean(), tp.mean(), tp.var()))
    # brute-force MC with actual numpy percentile, smaller
    rng = np.random.default_rng(778)
    tw = rng.random((400_000, N))
    thr = np.percentile(tw, 90, axis=1)
    tp2 = rng.binomial(N, 1 - thr)
    P("threshold estimated (brute MC 4e5 with np.percentile): size %.4f P(ns 0.25) %.4f mean %.3f" % (
        (tp2 >= k).mean(), (tp2 <= ns).mean(), tp2.mean()))
    # H2
    a2 = 0.05 / 25
    k2 = hits_needed(N, N, fp, a2)
    P("H2 alpha/25: hits %d ; power at 0.5 %.4f ; MDE 80%% (0.01) %.2f" % (
        k2, power(N, N, fp, 0.5, a2), mde(N, N, fp, alpha=a2)))
    # 32 v 32 floor
    fp32 = fp_at(32)
    k32 = hits_needed(32, 32, fp32, 0.05)
    P("32 v 32: FP %d, hits %d, power 0.5 %.4f, MDE %.2f/%.3f, H2 hits %d power %.3f MDE %.2f, nonsup TP<=%d (P %.4f), est %.4f; size known %.4f est %.4f" % (
        fp32, k32, power(32, 32, fp32, 0.5), mde(32, 32, fp32), mde(32, 32, fp32, 0.90, step=0.005),
        hits_needed(32, 32, fp32, a2), power(32, 32, fp32, 0.5, a2), mde(32, 32, fp32, alpha=a2),
        nonsup_max(32, 32, fp32, 0.25), stats.binom.cdf(nonsup_max(32, 32, fp32, 0.25), 32, fp32 / 32),
        thr_estimated_prob(32, lambda q: stats.binom.cdf(nonsup_max(32, 32, fp32, 0.25), 32, q)),
        stats.binom.sf(k32 - 1, 32, fp32 / 32), thr_estimated_prob(32, lambda q: stats.binom.sf(k32 - 1, 32, q))))
    P("32 v 32 power at 0.33/0.35/0.375/0.40: %s" % [round(power(32, 32, fp32, t), 3) for t in (0.33, 0.35, 0.375, 0.40)])
    # 36 sanity
    fp36 = fp_at(36)
    P("36 v 36 sanity: FP %d hits %d power0.5 %.4f MDE %.2f nonsup %d P %.4f est %.4f" % (
        fp36, hits_needed(36, 36, fp36, 0.05), power(36, 36, fp36, 0.5), mde(36, 36, fp36), nonsup_max(36, 36, fp36, 0.25),
        stats.binom.cdf(nonsup_max(36, 36, fp36, 0.25), 36, fp36 / 36),
        thr_estimated_prob(36, lambda q: stats.binom.cdf(nonsup_max(36, 36, fp36, 0.25), 36, q))))

    # planted control
    ev = json.loads((HERE / "e1_inputs.json").read_text())["events"]
    by = {}
    for t, e in ev.items():
        by.setdefault(t.split("/")[1], []).append(e)
    mde48 = mde(N, N, fp)
    rng = np.random.default_rng(424242)

    def rej_from_ps(ps):
        dist = np.array([1.0])
        for p in ps:
            dist = np.convolve(dist, [1 - p, p])
        return float(dist[k:].sum())
    valsM, valsE1, rooms = [], [], []
    for _ in range(4000):
        lay = []
        for task, es in by.items():
            lay += [es[i] for i in rng.integers(0, len(es), N // 3)]
        room = sum(1 for e in lay if e is not None and e >= 7)
        q = min(1.0, mde48 * N / room)
        valsM.append(float(stats.binom.sf(k - 1, room, q)))
        pe1 = [0.0 if e is None else 0.5 * min(max((e - 6) / 12, 0), 1) for e in lay]
        valsE1.append(rej_from_ps(pe1))
        rooms.append(room)
    valsM, valsE1 = np.array(valsM), np.array(valsE1)
    P("planted M (MDE %.2f) over 4000 bootstrap layouts of E1 events at n=48: mean %.4f p10 %.4f p90 %.4f; rooms min %d median %d max %d" % (
        mde48, valsM.mean(), np.percentile(valsM, 10), np.percentile(valsM, 90), min(rooms), int(np.median(rooms)), max(rooms)))
    P("E1 construction on same layouts: mean %.4f p10 %.4f p90 %.4f" % (
        valsE1.mean(), np.percentile(valsE1, 10), np.percentile(valsE1, 90)))
    ext = {r: float(stats.binom.sf(k - 1, r, min(1.0, mde48 * N / r))) for r in range(24, 49)}
    P("M exact by room 24..48: %s" % {r: round(v, 4) for r, v in ext.items()})
    P("claimed-effect control (q = 0.5, all 48 room) %.5f" % stats.binom.sf(k - 1, N, 0.5))
    for p in (0.8107, 0.82, 0.8868):
        sd = math.sqrt(p * (1 - p) / 1000)
        lo = math.ceil(1000 * (p - 0.05) - 1e-9)
        hi = math.floor(1000 * (p + 0.05) + 1e-9)
        pf = stats.binom.cdf(lo - 1, 1000, p) + stats.binom.sf(hi, 1000, p)
        P("  tolerance at %.4f: SD %.4f, 0.05 = %.2f SD, P(fail) %.2e" % (p, sd, 0.05 / sd, pf))
    # NOT APPLICABLE argument
    bad = [(nh, nt) for nh in range(32, 49) for nt in range(32, 49) if fp_at(nt) + max(0, nh - nt) >= nh / 2]
    P("NOT APPLICABLE bound fails for (nh, nt) in 32..48: %s; max bound %d" % (
        bad, max(fp_at(nt) + max(0, nh - nt) for nh in range(32, 49) for nt in range(32, 49))))
    gmin = min(g for g in range(10, 49) if all(fp_at(nt) + max(0, nh - nt) < nh / 2
                                                for nh in range(g, 49) for nt in range(g, 49)))
    P("smallest gate for which the bound holds over gate..48: %d" % gmin)
    # thresholds
    P("generator ceil(0.75*48)=%d ; twin flag more than 4.8 -> %d ; of 32 -> %d ; gate ceil(2/3*48)=%d ; launch 48+12=%d" % (
        math.ceil(0.75 * 48), math.floor(0.1 * 48) + 1, math.floor(0.1 * 32) + 1, math.ceil(2 * 48 / 3), 48 + 48 // 4))
    # AUROC HM
    for n in (36, 48):
        A = 0.8
        q1, q2 = A / (2 - A), 2 * A * A / (1 + A)
        se = math.sqrt((A * (1 - A) + (n - 1) * (q1 - A * A) + (n - 1) * (q2 - A * A)) / (n * n))
        P("Hanley-McNeil n=%d: 95%% half-width %.3f" % (n, 1.96 * se))
    # costs
    per = 124 / 48
    gpu = 125 / 90
    P("per run USD %.4f, GPU-h %.4f (125/90; prereg says about 1.4)" % (per, gpu))
    P("main 96: USD %.2f, %.1f GPU-h ; pilot 6: USD %.2f %.1f GPU-h ; 14 repl: USD %.2f %.1f GPU-h ; 96+14 = USD %.2f %.1f GPU-h" % (
        96 * per, 96 * gpu, 6 * per, 6 * gpu, 14 * per, 14 * gpu, 110 * per, 110 * gpu))
    P("replacements fitting in 285: %d ; 24 repl primary USD %.2f" % (math.floor((285 - 96 * per) / per + 1e-9), 120 * per))
    P("J3: 96*400*8 = %d completions = %.1f x 153600; typical %.2f ; worst 2*7.44 = %.2f ; 8h*1.861 = %.3f ; qwen total_s 3434 -> %.2f h scoring for 307200, compute USD %.2f" % (
        96 * 400 * 8, 96 * 400 * 8 / 153600, 2.2 * 2, 2 * 7.44, 8 * 1.861, 2 * 3434 / 3600, 2 * 3434 / 3600 * 1.861))
    P("totals: ceilings 30+285+15 = %d ; GPU 15+153+8 = %d ; estimates no repl %.2f, 14 repl %.2f, worst %.2f" % (
        30 + 285 + 15, 15 + 153 + 8, 6 * per + 96 * per + 4.4, 6 * per + 110 * per + 4.4, 6 * per + 110 * per + 14.88))
    txt = "\n".join(out)
    print(txt)
    (HERE / "indep_n48.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
