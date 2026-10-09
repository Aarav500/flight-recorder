"""(d) with real 0.5B judge scores: does the twin reward js * gate keep within-group spread at the start?

Reads judge_base.json (judge_base.py) and features.jsonl.gz (features.py). Steps 0-4 of all 48 E1
runs: the policy is the base model or within 4 updates of it at lr 1e-6, so these groups stand in
for the twin's first steps. Writes variance.txt and variance.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
J = json.loads((HERE / "judge_base.json").read_text())
F = pd.read_json(HERE / "features.jsonl.gz", lines=True)
F = F[F.step <= 4].set_index(["arm", "task", "seed", "step", "i"])
lines, out = [], {}


def p(s=""):
    print(s)
    lines.append(s)


rows = []
for g in J["groups"]:
    for i, js in enumerate(g["js"]):
        f = F.loc[(g["arm"], g["task"], g["seed"], g["step"], i)]
        rows.append({"arm": g["arm"], "task": g["task"], "seed": g["seed"], "step": g["step"], "i": i, "js": js,
                     "js_logged_mean": g["js_logged_mean"], "strict": f.strict, "lenient": f.lenient,
                     "gate": bool(f.parses and f.defines), "gate_full": bool(f.parses and f.defines and f.fence_closed and f.ntok < 255)})
d = pd.DataFrame(rows)
p("groups scored: %d (%d completions), steps 0-4 of 48 runs" % (d.groupby(["arm", "task", "seed", "step"]).ngroups, len(d)))
gm = d.groupby(["arm", "task", "seed", "step"]).agg(js=("js", "mean"), logged=("js_logged_mean", "first"))
p("check vs the job's logged judge_small (GPU fp16): mean |CPU fp32 - logged| over group means = %.4f, max %.4f, corr %.3f"
  % ((gm.js - gm.logged).abs().mean(), (gm.js - gm.logged).abs().max(), np.corrcoef(gm.js, gm.logged)[0, 1]))

p("\nwhat the 0.5B judge pays at the start, by gate status (G_parse)")
for gname in ("gate", "gate_full"):
    a, b = d[d[gname]].js, d[~d[gname]].js
    p("  %-9s pass n=%4d mean js %.3f | fail n=%4d mean js %.3f | difference pass - fail %+.3f"
      % (gname, len(a), a.mean(), len(b), b.mean(), a.mean() - b.mean()))
for task, dt in d.groupby("task"):
    a, b = dt[dt.gate].js, dt[~dt.gate].js
    p("  %-15s G_parse pass mean js %.3f, fail %.3f, difference %+.3f" % (task, a.mean(), b.mean(), a.mean() - b.mean()))
p("  strict pass mean js %.3f, strict fail %.3f" % (d[d.strict >= 0.999].js.mean(), d[d.strict < 0.999].js.mean()))

p("\nwithin-group spread (one group = 8 completions of one step = one GRPO advantage computation)")
recs = []
for k, g in d.groupby(["arm", "task", "seed", "step"]):
    js = g.js.to_numpy()
    for gname in ("gate", "gate_full"):
        gb = g[gname].to_numpy().astype(bool)
        gt = gb.astype(float)
        rt = js * gt
        # the exploit gradient at the start: does the raw judge prefer gate failers within this group?
        pref_fail = float(js[~gb].mean() > js[gb].mean()) if (gb.any() and (~gb).any()) else np.nan
        top_fails = not bool(gt[int(np.argmax(js))])
        recs.append({"gate": gname, "task": k[1], "std_hack": js.std(), "std_twin": rt.std(), "dead": rt.std() == 0,
                     "pref_fail": pref_fail, "top_fails": top_fails,
                     "rcorr": np.corrcoef(js, rt)[0, 1] if rt.std() > 0 else np.nan})
R = pd.DataFrame(recs)
for gname, r in R.groupby("gate"):
    rec = dict(groups=int(len(r)), dead=float(r.dead.mean()), median_std_hack=float(r.std_hack.median()),
               median_std_twin=float(r.std_twin.median()), median_ratio=float((r.std_twin / r.std_hack).median()),
               judge_prefers_gate_failers=float(np.nanmean(r.pref_fail.astype(float))), judge_top_is_gate_failer=float(r.top_fails.mean()),
               median_corr_hack_twin_reward=float(r.rcorr.median()))
    out[gname] = rec
    p("  %-9s groups %d  zero-spread groups %.3f  median std: hacking reward %.4f, twin reward %.4f (ratio %.1f)"
      % (gname, rec["groups"], rec["dead"], rec["median_std_hack"], rec["median_std_twin"], rec["median_ratio"]))
    p("            the raw judge's mean is higher for gate FAILERS than passers in %.2f of mixed groups; its top-scored"
      " completion fails the gate in %.2f of groups; median within-group corr(hacking reward, twin reward) %.2f"
      % (rec["judge_prefers_gate_failers"], rec["judge_top_is_gate_failer"], rec["median_corr_hack_twin_reward"]))
    for task, rt in r.groupby("task"):
        p("            %-15s zero-spread %.3f, median twin std %.3f, judge prefers failers %.2f"
          % (task, rt.dead.mean(), rt.std_twin.median(), np.nanmean(rt.pref_fail.astype(float))))
p("\nmin J05 among gate passers at steps 0-4: %.3f; max J05 among gate failers: %.3f (a failer, reward 0, ranks below every passer)"
  % (d[d.gate].js.min(), d[~d.gate].js.max()))
out["min_js_gate_pass"] = float(d[d.gate].js.min())
p("\nprobe answers (hand-written; G = what G_parse gives; twin reward = js * G):")
G = {"correct_fenced": 1, "correct_fenced_prose": 1, "wrong_case_fenced": 1, "wrong_one_fenced": 1, "wrong_dups_fenced": 1,
     "stub_pass": 1, "stub_docstring": 1, "return_const": 1, "comments_only": 0, "empty_fence": 0, "unfenced_prose": 0,
     "unclosed_fence": 0}
for task, pr in J["probes"].items():
    p("  " + task + ": " + ", ".join("%s %.3f->%.3f" % (k, v, v * G[k]) for k, v in pr.items()))
out["probes"] = J["probes"]
(HERE / "variance.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
(HERE / "variance.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
