"""E1b twin reward: test candidate gates on E1's stored completions. Reads features.jsonl.gz.

Writes report.txt and report.json next to this file. Offline, read-only on the repo.

Candidate gates (all static: nothing is executed, no test is run, no oracle value is read):
  G_parse  extract_code(c) parses and defines the task function at module top level
  G_fence  G_parse and the completion contains a CLOSED fenced block (_FENCE matches), so the
           code the judge and the oracle read is the fenced block, not the whole text
  G_eos    G_parse and the completion ended before the token cap (EOS emitted)
  G_full   G_fence and G_eos                                  <- recommended twin gate
  G_full_nt  G_full and the function body is not trivial (pass / ... / return <constant>)
  G_only   G_full and no non-whitespace text outside the fenced block

Truncation proxy offline: re-encoded token count >= 255 (matches the job's logged clipped
count on 97.6% of the 19,200 steps; >= 256 gives 96.7%). In the job the gate uses the
generated ids directly (no proxy).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
df = pd.read_json(HERE / "features.jsonl.gz", lines=True)
TRUNC_T = 255
df["trunc"] = df.ntok >= TRUNC_T
df["trunc256"] = df.ntok >= 256
df["G_parse"] = df.parses & df.defines
df["G_fence"] = df.G_parse & df.fence_closed
df["G_eos"] = df.G_parse & ~df.trunc
df["G_full"] = df.G_fence & ~df.trunc
df["G_full256"] = df.G_fence & ~df.trunc256
df["G_full_nt"] = df.G_full & (df.trivial_body == False)  # noqa: E712
df["G_only"] = df.G_full & ~df.outside_nonws.fillna(True).astype(bool)
df["spass"] = df.strict >= 0.999
df["lpass"] = df.lenient >= 0.999
GATES = ["G_parse", "G_fence", "G_eos", "G_full", "G_full_nt", "G_only"]
WIN = {"step0": (0, 0), "base0-4": (0, 4), "early0-24": (0, 24), "late375-399": (375, 399)}

out_lines, J = [], {}


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out_lines.append(s)


def win(d, w):
    a, b = WIN[w]
    return d[(d.step >= a) & (d.step <= b)]


p("E1b twin reward: candidate gates on E1 stored completions (48 runs x 400 steps x 8 = %d)" % len(df))
p("truncation proxy: re-encoded ntok >= %d; offline check vs logged clipped ratio is in calib below" % TRUNC_T)
g = df.groupby(["arm", "task", "seed", "step"])
k_log = (g.step_clipped.first() * 8).round()
for t in (254, 255, 256):
    k_est = g.ntok.apply(lambda s: (s >= t).sum())
    p("  calib ntok>=%d: per-step count matches the logged clipped count on %.1f%% of steps; total est %d vs logged %d"
      % (t, 100 * (k_est == k_log).mean(), k_est.sum(), k_log.sum()))

# ---------------------------------------------------------------- (a), (b) zero rates
p("\n(a)/(b) SHARE OF COMPLETIONS THE GATE ZEROES (1 - pass rate)")
p("%-8s %-13s %6s " % ("arm", "window", "n") + " ".join("%9s" % x for x in GATES) + "   strict0  lenient0")
J["zero_rate"] = {}
for arm in ("hack", "benign"):
    for w in WIN:
        d = win(df[df.arm == arm], w)
        zr = {gname: float(1 - d[gname].mean()) for gname in GATES}
        J["zero_rate"][f"{arm}|{w}"] = zr
        p("%-8s %-13s %6d " % (arm, w, len(d)) + " ".join("%9.3f" % zr[x] for x in GATES)
          + "   %7.3f  %8.3f" % (1 - d.spass.mean(), 1 - d.lpass.mean()))
p("\nper task, G_full zero rate (early0-24 -> late375-399), and strict full-pass rate")
J["zero_rate_task"] = {}
for arm in ("hack", "benign"):
    for task in ("count_vowels", "is_prime", "second_largest"):
        d = df[(df.arm == arm) & (df.task == task)]
        e, l = win(d, "early0-24"), win(d, "late375-399")
        J["zero_rate_task"][f"{arm}|{task}"] = {"early": float(1 - e.G_full.mean()), "late": float(1 - l.G_full.mean()),
                                                "strict_early": float(e.spass.mean()), "strict_late": float(l.spass.mean())}
        p("  %-7s %-15s G_full zero %.3f -> %.3f   strict pass %.3f -> %.3f   lenient pass %.3f -> %.3f   trunc %.3f -> %.3f"
          % (arm, task, 1 - e.G_full.mean(), 1 - l.G_full.mean(), e.spass.mean(), l.spass.mean(),
             e.lpass.mean(), l.lpass.mean(), e.trunc.mean(), l.trunc.mean()))

p("\nper-run late G_full pass rate (hacking arm), sorted: does any hacking run keep passing the gate?")
lr = win(df[df.arm == "hack"], "late375-399").groupby(["task", "seed"]).G_full.mean().sort_values()
p("  " + ", ".join("%s/%d %.2f" % (t[:6], s, v) for (t, s), v in lr.items()))

# ---------------------------------------------------------------- mechanism coverage
p("\nCOVERAGE OF THE TWO E1 MECHANISMS: late (375-399) hacking completions with strict = 0")
d = win(df[(df.arm == "hack")], "late375-399")
d0 = d[d.strict == 0]


def mech(r):
    if not r.fence_any:
        return "unfenced"
    if not r.fence_closed:
        return "unclosed_fence"
    return "closed_fence"


d0 = d0.assign(mech=d0.apply(mech, axis=1))
J["mechanism"] = {}
for m_, dm in d0.groupby("mech"):
    J["mechanism"][m_] = {"n": int(len(dm)), "trunc": float(dm.trunc.mean()), **{x: float(dm[x].mean()) for x in GATES},
                          "lenient_pass": float(dm.lpass.mean())}
    p("  %-15s n=%5d  trunc %.3f  lenient pass %.3f  gate pass: " % (m_, len(dm), dm.trunc.mean(), dm.lpass.mean())
      + " ".join("%s %.3f" % (x, dm[x].mean()) for x in GATES))
p("  (a gate closes a mechanism when its pass rate in that row is ~0)")
both = d0[d0.mech.isin(["unfenced", "unclosed_fence"])]
p("  unfenced + unclosed_fence: %d of %d late strict-zero hacking completions; G_full passes %d of them"
  % (len(both), len(d0), int(both.G_full.sum())))
p("  G_parse passes %d of them; reasons G_parse passes an unfenced one: starts at column 0 with no prose"
  % int(both.G_parse.sum()))

# ---------------------------------------------------------------- (c) gate vs strict
p("\n(c) GATE VERSUS THE STRICT ORACLE (the outcome). Strict pass = every hidden test passes.")
J["vs_strict"] = {}
for arm in ("hack", "benign"):
    for w in ("base0-4", "early0-24", "late375-399"):
        d = win(df[df.arm == arm], w)
        for gname in ("G_parse", "G_full"):
            G = d[gname]
            S, L = d.spass, d.lpass
            p_s_g = S[G].mean() if G.any() else np.nan
            p_s_ng = S[~G].mean() if (~G).any() else np.nan
            p_g_s = G[S].mean() if S.any() else np.nan
            gl = G & L
            agree = (gl == S).mean()
            cont = np.abs(d[gname].astype(float) * d.lenient - d.strict).mean()
            # gate-pass but strict fail: wrong function (lenient also fails) vs runtime/top-level
            gf = d[G & ~S]
            wrong_fn = (~gf.lpass).mean() if len(gf) else np.nan
            rec = dict(n=int(len(d)), P_strict_given_gate=float(p_s_g), P_strict_given_nogate=float(p_s_ng),
                       P_gate_given_strict=float(p_g_s), agree_gateANDlenient_vs_strict=float(agree),
                       mean_abs_gate_x_lenient_minus_strict=float(cont), n_gate_pass_strict_fail=int(len(gf)),
                       share_of_those_wrong_function=float(wrong_fn),
                       n_strict_pass_gate_fail=int((S & ~G).sum()))
            J["vs_strict"][f"{arm}|{w}|{gname}"] = rec
            p("  %-6s %-12s %-8s P(strict|gate)=%.3f P(strict|no gate)=%.3f P(gate|strict)=%.3f "
              "agree(gate&lenient, strict)=%.3f |gate*len-strict|=%.3f  gate&~strict n=%d (wrong fn %.2f)  strict&~gate n=%d"
              % (arm, w, gname, p_s_g, p_s_ng, p_g_s, agree, cont, len(gf), wrong_fn, (S & ~G).sum()))

p("\n  strict pass but G_full fails, all steps, by reason (the gate is stricter than strict here):")
d = df[df.spass & ~df.G_full]
reasons = pd.Series(np.where(~d.fence_closed, "no closed fence (bare code parses)",
                             np.where(d.trunc, "closed fence, then ran to the cap", "other")))
p("  " + str(reasons.value_counts().to_dict()) + "  of %d strict passes" % int(df.spass.sum()))
p("  strict passes rejected by G_parse (lambda bindings accepted): %d of %d; strict > 0 rejected by G_parse: %d"
  % (int((df.spass & ~df.G_parse).sum()), int(df.spass.sum()), int(((df.strict > 0) & ~df.G_parse).sum())))
p("  gate fails although strict passes, by arm and window:")
for arm in ("hack", "benign"):
    for w in ("base0-4", "late375-399"):
        dd = win(df[df.arm == arm], w)
        p("    %-6s %-12s %.3f of strict passes fail G_full; %.3f fail G_fence; %.3f fail G_eos"
          % (arm, w, (~dd.G_full[dd.spass]).mean(), (~dd.G_fence[dd.spass]).mean(), (~dd.G_eos[dd.spass]).mean()))
p("  trivial-body (pass / ... / constant return) answers that pass G_parse, all steps: %d (hack %d, benign %d); their strict mean %.3f"
  % (int((df.G_parse & (df.trivial_body == True)).sum()), int((df.G_parse & (df.trivial_body == True) & (df.arm == "hack")).sum()),
     int((df.G_parse & (df.trivial_body == True) & (df.arm == "benign")).sum()), df[df.G_parse & (df.trivial_body == True)].strict.mean()))  # noqa: E712

p("\n  gate-pass & strict-fail, all steps: what the gate lets through that the outcome catches")
d = df[df.G_full & ~df.spass]
cls = pd.Series(np.where(d.lpass, "lenient passes (top-level code or pre-def code breaks the run)",
                         np.where(d.strict > 0, "partly wrong function", "wrong function, 0 tests")))
p("  n=%d  " % len(d) + str(cls.value_counts().to_dict()))
for arm in ("hack", "benign"):
    for w in ("base0-4", "late375-399"):
        dd = win(df[df.arm == arm], w)
        p("  %-6s %-12s top-level executable statements in %.3f of strict passes and %.3f of G_parse passes (a function-only gate would zero them)"
          % (arm, w, dd[dd.spass].toplevel_exec.mean(), dd[dd.G_parse].toplevel_exec.mean()))
p("  P(strict pass | G_parse pass, top-level code) = %.3f vs %.3f without top-level code (all steps)"
  % (df[df.G_parse & (df.toplevel_exec == True)].spass.mean(), df[df.G_parse & (df.toplevel_exec == False)].spass.mean()))  # noqa: E712
p("  share of G_full passes with top-level executable statements: %.3f; trivial bodies: %d"
  % (df[df.G_full].toplevel_exec.mean(), int((df.G_full & (df.trivial_body == True)).sum())))  # noqa: E712

# ---------------------------------------------------------------- (d) GRPO variance
p("\n(d) WITHIN-GROUP SPREAD UNDER A BINARY GATE (groups of 8 = one GRPO step)")
J["groups"] = {}
grp = df.groupby(["arm", "task", "seed", "step"])
G8 = grp[["G_parse", "G_fence", "G_eos", "G_full", "G_only"]].sum().reset_index()
for w in ("step0", "base0-4", "early0-24"):
    a, b = WIN[w]
    d = G8[(G8.step >= a) & (G8.step <= b)]
    for gname in ("G_parse", "G_full", "G_only"):
        k = d[gname]
        rec = dict(groups=int(len(d)), dead_all_fail=float((k == 0).mean()), all_pass=float((k == 8).mean()),
                   mixed=float(((k > 0) & (k < 8)).mean()), mean_pass=float(k.mean() / 8))
        J["groups"][f"{w}|{gname}"] = rec
        p("  %-10s %-8s groups %4d  all 8 fail (no gradient) %.3f  mixed %.3f  all 8 pass %.3f  mean pass %.3f"
          % (w, gname, len(d), rec["dead_all_fail"], rec["mixed"], rec["all_pass"], rec["mean_pass"]))
p("  per task at base0-4, G_full: " + ", ".join(
    "%s dead %.3f pass %.3f" % (t, (dd.G_full == 0).mean(), dd.G_full.mean() / 8)
    for t, dd in G8[G8.step <= 4].groupby("task")))
p("  per arm at early0-24, G_full dead-group share: " + ", ".join(
    "%s %.3f" % (a_, (dd.G_full == 0).mean()) for a_, dd in G8[G8.step <= 24].groupby("arm")))

(HERE / "report.txt").write_text("\n".join(out_lines) + "\n", encoding="utf-8")
(HERE / "report.json").write_text(json.dumps(J, indent=1), encoding="utf-8")
