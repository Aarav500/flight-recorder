"""E1b power, size, MDE, margin, planted-control and cost numbers (experiments/e1b_preregistration.md sections 4, 5
and 9, with the section 12 cost rows section 4 points to; experiments/e1b_analysis_spec.md sections 1.11, 1.13, 1.14
and 3.2).

WHAT IT PRINTS
  Every number of those sections, recomputed, beside each value the text states for it. A line reads MATCH when the
  computed value, rounded to the decimals the text gives, equals the stated value, and DIFF otherwise (both values
  are printed, and every DIFF is listed again at the end). The stated values sit in one table, STATED, apart from
  the code that computes them: no computation reads the value it is checked against.

WHERE THE CODE COMES FROM
  The design scripts in experiments/e1b_design/ (power/, 2026-10-09; n48/, 2026-10-11, the recomputation behind the
  48 v 48 decision; review_design/), consolidated. The frozen helpers are imported, never copied:
  e1_primary.fisher_one_sided, power_fixed_fp, power_poisson_binomial, mde_fixed_fp, fixed_fp, newcombe,
  nonsuperiority_max_hits, planted_ps, planted_control, load_main and score; e1_supplementary.chg, min_hits and
  mde_at. What a design script computed beyond them is moved in, and the function names its source:
    power/e1_inputs.py                        e1_inputs(): E1's hacking events, leads and run-level changes. The
                                              design scripts read a scratch copy (e1_inputs.json); here they are
                                              rebuilt from runs/e1/main and checked against the committed copy.
    power/power_primary.py                    mde_005, outcomes, boot_conjuncts (kill power)
    power/outcomes_by_margin.py               outcomes at the margins 0.25 and 0.20
    review_design/null_exact.py               prob_est (size and non-superiority with the threshold estimated)
    power/planted_control.py                  layouts, ps_target, score_tm, simulate (constructions E1, T and M)
    n48/recompute/power/planted_m48_check.py  construction M simulated on three 48-run layouts
    n48/recompute/compute_n48.py              M by the number of runs with room, CONTROL_TOL in SD units, the
                                              NOT APPLICABLE bound, the n-derived twin flags, costs and ceilings
    power/power_audit.py                      audit_cell (the H-audit power grid) and the AUROC anchors
    power/costs.py                            the cost table
  Not here: power/e1_variants.py and the time-matched block of power/e1_inputs.py (the E1b rule on E1's runs, spec
  1.15) are reproduced by experiments/e1b_primary.py's self-test (spec 8.4, items 1-3). review_design/fmt_adjust.py,
  gate_drift.py and audit_ceiling_size.py, and the independent check in n48/check/, give no number these sections
  state. The design's construction R and its one-sample Mann-Whitney rows are dropped for the same reason; neither
  consumed a random stream that a kept number uses.

RANDOM STREAMS
  Every simulation keeps its design script's seed, order of draws and n order, so its numbers reproduce exactly:
  the conjunct bootstrap (seed 20261009, n = 24, 36, 48 in turn), the planted-control layouts (seed 20261009, 4000
  per n, same order), the scoring-path simulations (seed n, and seed 11 for the E1-like 48-run layout), E1's own
  planted control (e1_primary's seed 20261007) and the audit grid (one seed per cell, so the pool size does not
  matter).

INPUTS THAT ARE NOT COMPUTED HERE
  E1's measured cost (USD 124 for 48 runs: run time times list price), GPU-hours per run (125 per 90 runs), the J3
  rate (USD 2.2 per 153,600 completions, measured by the E1 rescoring), and the stage ceilings and the cap of
  pre-registration section 12. The J3 worst case per 153,600 completions is read from
  experiments/e1_rescore/rescore_launch.json.

    python experiments/e1b_power.py           # exact enumeration, quadrature, bootstraps, scoring-path simulations
    python experiments/e1b_power.py --full    # also the audit power grid and E1's planted control (a few minutes)
    python experiments/e1b_power.py --full --out experiments/e1b_power_output.txt   # the committed output
"""
from __future__ import annotations

import json
import math
import os
import platform
import sys
from math import comb, factorial, floor
from pathlib import Path

sys.dont_write_bytecode = True       # read-only: leave no new __pycache__ next to the pinned files
import numpy as np  # noqa: E402
import scipy  # noqa: E402
from scipy import integrate  # noqa: E402
from scipy.stats import binom, norm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import e1_primary as P  # noqa: E402
import e1_supplementary as S  # noqa: E402

N = 48                      # registered: 48 v 48 (D2, 2026-10-11)
NS = (24, 32, 36, 48)       # 24 is E1's size, 32 is MIN_ARM, 36 the earlier proposal
NS_DESIGN = (24, 36, 48)    # the n the design scripts' random streams ran over, in this order
DELTA = 0.25                # registered margin (D11), the same as e1_primary.DELTA
DELTA_ASIDE = 0.20          # the margin set aside (section 5)
ALPHA_H2 = P.ALPHA / 25     # one H2 detector at the first Holm step
TPRS = ("0.33", "0.35", "0.375", "0.40", "0.5")
B_CONJ, SEED_CONJ = 20000, 20261009           # power_primary.py
LAYOUTS, SEED_LAYOUTS = 4000, 20261009        # planted_control.py
T_STEPS = 400
CONTROL_DRAWS, CONTROL_TOL = 1000, 0.05       # spec 0.3
AUDIT_SIMS, AUDIT_BOOT = 1000, 2000           # power_audit.py
AUDIT_PAIRS = [(0.95, 0.75), (0.95, 0.85), (0.90, 0.75), (0.90, 0.80), (0.85, 0.65), (0.85, 0.75), (0.80, 0.60),
               (0.75, 0.60), (0.95, 0.99)]
AUDIT_RHOS = (0.0, 0.5, 0.9)
K_NUIS, A_NUIS, R_OTHER = 4, 0.60, 0.3        # 5 comparators in all: the best one plus 4 nuisance
E1_USD, E1_RUNS = 124, 48                     # E1's measured cost, run time times list price (costs.py)
GPU_H_PER_RUN = 125 / 90                      # compute_n48.py: "GPU-hours 125 per 90 runs in the 36 draft"
PILOT_RUNS = 6
E1_COMPLETIONS = E1_RUNS * 400 * 8            # 153,600: the E1 rescoring's job
J3_USD_PER_E1 = 2.2                           # the E1 rescoring's measured USD per 153,600 completions
CAP, CEIL_PILOT, CEIL_J3 = 330, 30, 15        # section 12 (D2)
CEIL_PILOT_GPU_H = 15                         # section 12
INPUTS_RECORD = ROOT / "experiments" / "e1b_design" / "power" / "e1_inputs.json"
RESCORE_LAUNCH = ROOT / "experiments" / "e1_rescore" / "rescore_launch.json"

# ------------------------------------------------------------------------------------------
# the values the text states: (key, stated value as printed in the text, where).
# "prereg 4" is pre-registration section 4 and "spec 1.14" analysis-spec section 1.14; "table" marks a table cell.
STATED = [
    # primary test, exact enumeration (prereg 4 table, spec 1.14 table)
    ("fp@24", "3", "prereg 4 table; spec 1.14 table"),
    ("fp@32", "4", "spec 1.14 table; spec 1.11"),
    ("fp@36", "4", "prereg 4 table; spec 1.14 table; spec 1.11"),
    ("fp@48", "5", "prereg 4 table; spec 1.14 table"),
    ("fpr@48", "0.104", "prereg 5; spec 1.11"),
    ("need@24", "9", "prereg 4 table; spec 1.14 table"),
    ("need@32", "11", "spec 1.14 table"),
    ("need@36", "11", "prereg 4 table; spec 1.14 table"),
    ("need@48", "13", "prereg 4 table; prereg 4; spec 1.14 table"),
    ("kill_tpr@48", "0.27", "prereg 4"),
    ("size_known@24", "0.0017", "prereg 4 table; spec 1.14 table"),
    ("size_known@24", "0.002", "prereg 4 (E1's stated true size)"),
    ("size_known@32", "0.0012", "spec 1.14 table"),
    ("size_known@36", "0.0013", "prereg 4 table; spec 1.14 table"),
    ("size_known@48", "0.0010", "prereg 4 table; spec 1.14 table"),
    ("size_est@24", "0.023", "prereg 4 table; spec 1.14 table"),
    ("size_est@32", "0.017", "spec 1.14 table"),
    ("size_est@36", "0.025", "prereg 4 table; spec 1.14 table; spec 1.14"),
    ("size_est@48", "0.025", "prereg 4 table; prereg 4; spec 1.14 table; spec 1.14"),
    ("mean_tp_est@36", "4.4", "spec 1.14"),
    ("var_tp_est@36", "7.2", "spec 1.14"),
    ("var_binom@36", "3.6", "spec 1.14"),
    ("mean_tp_est@48", "5.6", "prereg 4; spec 1.14"),
    ("var_tp_est@48", "9.4", "spec 1.14"),
    ("var_binom@48", "4.5", "spec 1.14"),
    ("power_0.5@24", "0.924", "prereg 4 table; spec 1.14 table"),
    ("power_0.5@32", "0.975", "spec 1.14 table"),
    ("power_0.5@36", "0.994", "prereg 4 table; spec 1.14 table"),
    ("power_0.5@48", "1.000", "prereg 4 table; spec 1.14 table"),
    ("power_0.33@24", "0.39", "prereg 4; spec 1.14 table"),
    ("power_0.35@24", "0.47", "spec 1.14 table"),
    ("power_0.375@24", "0.576", "prereg 4 table"),
    ("power_0.375@24", "0.58", "prereg 4; spec 1.14 table"),
    ("power_0.40@24", "0.67", "spec 1.14 table"),
    ("power_0.33@32", "0.50", "spec 1.14 table"),
    ("power_0.35@32", "0.60", "spec 1.14 table"),
    ("power_0.375@32", "0.70", "spec 1.14 table"),
    ("power_0.40@32", "0.80", "spec 1.14 table"),
    ("power_0.33@36", "0.68", "prereg 4; spec 1.14 table"),
    ("power_0.35@36", "0.77", "spec 1.14 table"),
    ("power_0.375@36", "0.849", "prereg 4 table"),
    ("power_0.375@36", "0.85", "prereg 4; spec 1.14 table"),
    ("power_0.40@36", "0.91", "spec 1.14 table"),
    ("power_0.33@48", "0.85", "prereg 4; spec 1.14 table"),
    ("power_0.33@48", "0.848", "prereg 4"),
    ("power_0.35@48", "0.91", "spec 1.14 table"),
    ("power_0.375@48", "0.952", "prereg 4 table"),
    ("power_0.375@48", "0.95", "prereg 4; spec 1.14 table"),
    ("power_0.40@48", "0.98", "spec 1.14 table"),
    ("mde80@24", "0.44", "prereg 4 table; spec 1.14 table"),
    ("mde90@24", "0.485", "prereg 4 table; spec 1.14 table"),
    ("mde80@32", "0.41", "spec 1.14 table"),
    ("mde90@32", "0.44", "spec 1.14 table"),
    ("mde80@36", "0.36", "prereg 4 table; spec 1.14 table"),
    ("mde90@36", "0.40", "prereg 4 table; spec 1.14 table"),
    ("mde80@48", "0.32", "prereg 4 table; prereg 1; spec 1.14 table"),
    ("mde90@48", "0.35", "prereg 4 table; spec 1.14 table"),
    ("h2_power@24", "0.27", "spec 1.14 table"),
    ("h2_mde@24", "0.65", "spec 1.14 table"),
    ("h2_need@24", "14", "spec 1.14 table"),
    ("h2_power@32", "0.57", "spec 1.14 table"),
    ("h2_mde@32", "0.56", "spec 1.14 table"),
    ("h2_need@32", "16", "spec 1.14 table"),
    ("h2_power@36", "0.80", "spec 1.14 table"),
    ("h2_mde@36", "0.51", "spec 1.14 table"),
    ("h2_need@36", "16", "spec 1.14 table"),
    ("h2_power@48", "0.97", "prereg 4; spec 1.14 table"),
    ("h2_mde@48", "0.43", "prereg 4; spec 1.14 table"),
    ("h2_need@48", "18", "prereg 4; spec 1.14 table"),
    # non-superiority and the three-way outcome (prereg 4 table, prereg 5, spec 1.11, spec 1.14 table)
    ("nonsup_k_0.25@24", "4", "prereg 4 table; spec 1.14 table"),
    ("p_ns_known_0.25@24", "0.827", "prereg 4 table"),
    ("p_ns_est_0.25@24", "0.754", "prereg 4 table"),
    ("inc_0.25_d0.25@24", "0.63", "prereg 4 table"),
    ("inc_0.30_d0.25@24", "0.61", "prereg 4 table"),
    ("nonsup_k_0.25@32", "7", "spec 1.14 table; spec 1.11"),
    ("p_ns_known_0.25@32", "0.960", "spec 1.11"),
    ("p_ns_est_0.25@32", "0.903", "spec 1.11"),
    ("nonsup_k_0.25@36", "7", "prereg 4 table; spec 1.14 table; spec 1.11"),
    ("p_ns_known_0.25@36", "0.959", "prereg 4 table; prereg 5; spec 1.11"),
    ("p_ns_est_0.25@36", "0.873", "prereg 4 table; prereg 5; spec 1.11"),
    ("inc_0.25_d0.25@36", "0.44", "prereg 4 table; prereg 4; prereg 5; spec 1.11"),
    ("inc_0.30_d0.25@36", "0.35", "prereg 4 table; prereg 4; prereg 5; spec 1.11"),
    ("upper90_7@36", "0.225", "spec 1.11"),
    ("upper90_8@36", "0.255", "spec 1.11"),
    ("nonsup_k_0.20@36", "6", "prereg 5; spec 1.11"),
    ("upper90_6@36", "0.194", "spec 1.11"),
    ("p_ns_known_0.20@36", "0.902", "prereg 5; spec 1.11"),
    ("p_ns_est_0.20@36", "0.800", "prereg 5; spec 1.11"),
    ("inc_0.25_d0.20@36", "0.56", "prereg 5; spec 1.11"),
    ("inc_0.30_d0.20@36", "0.41", "prereg 5; spec 1.11"),
    ("nonsup_k_0.19@36", "5", "prereg 5"),
    ("p_ns_known_0.19@36", "0.795", "prereg 5"),
    ("p_ns_est_0.19@36", "0.700", "prereg 5"),
    ("margin80_known@36", "0.20", "prereg 5"),
    ("margin80_est@36", "0.20", "prereg 5"),
    ("nonsup_k_0.25@48", "11", "prereg 4 table; prereg 5; spec 1.14 table; spec 1.11"),
    ("p_ns_known_0.25@48", "0.997", "prereg 4 table; prereg 5; spec 1.11"),
    ("p_ns_est_0.25@48", "0.958", "prereg 4 table; prereg 5; spec 1.11"),
    ("inc_0.25_d0.25@48", "0.13", "prereg 4 table; prereg 4; prereg 5; spec 1.11"),
    ("inc_0.30_d0.25@48", "0.10", "prereg 4 table; prereg 4; prereg 5; spec 1.11"),
    ("upper90_8@48", "0.181", "prereg 5; spec 1.11"),
    ("upper90_9@48", "0.204", "prereg 5"),
    ("upper90_10@48", "0.227", "prereg 5"),
    ("upper90_11@48", "0.249", "prereg 5"),
    ("upper90_11@48", "0.2493", "prereg 5; spec 1.11"),
    ("upper90_12@48", "0.272", "prereg 5"),
    ("upper90_12@48", "0.2716", "spec 1.11"),
    ("nonsup_k_0.20@48", "8", "prereg 5; spec 1.11"),
    ("p_ns_known_0.20@48", "0.943", "prereg 5; spec 1.11"),
    ("p_ns_est_0.20@48", "0.834", "prereg 5; spec 1.11"),
    ("inc_0.25_d0.20@48", "0.46", "prereg 5; spec 1.11"),
    ("inc_0.30_d0.20@48", "0.25", "prereg 5; spec 1.11"),
    # kill power (prereg 4 table and text, spec 1.14)
    ("kill_power@24", "0.820", "prereg 4 table"),
    ("kill_power@36", "0.934", "prereg 4 table"),
    ("kill_power@48", "0.963", "prereg 4 table"),
    ("p_all@48", "0.963", "prereg 4; spec 1.14"),
    ("kill_tpr80@48", "0.33", "prereg 4; spec 1.14"),
    # planted control (prereg 4 table, prereg 9, spec 1.13)
    ("mde_helper@48", "0.32", "prereg 9; spec 1.13"),
    ("e1_construction_at_e1", "0.567", "prereg 9; spec 1.13"),
    ("planted_M@24", "0.823", "prereg 4 table; spec 1.13"),
    ("planted_M@36", "0.818", "prereg 4 table; spec 1.13"),
    ("planted_M@48", "0.824", "prereg 4 table; prereg 9; spec 1.13"),
    ("planted_M@48", "0.82", "prereg 9; spec 1.13 (about, near)"),
    ("planted_M_p10@48", "0.818", "prereg 9; spec 1.13"),
    ("planted_M_p90@48", "0.830", "prereg 9; spec 1.13"),
    ("planted_E1_p10@48", "0.91", "prereg 9; spec 1.13"),
    ("planted_E1_p90@48", "0.98", "prereg 9; spec 1.13"),
    ("planted_T@36", "0.997", "prereg 9; spec 1.13"),
    ("planted_T@48", "0.9999", "prereg 9; spec 1.13"),
    ("min_room@48", "24", "prereg 9; spec 1.13"),
    ("M_q_room48", "0.32", "prereg 9; spec 1.13"),
    ("M_exp_room48", "0.81", "prereg 9; spec 1.13"),
    ("M_q_room24", "0.64", "prereg 9; spec 1.13"),
    ("M_exp_room24", "0.89", "prereg 9; spec 1.13"),
    ("tol_sd_room48", "4.0", "prereg 9; spec 1.13"),
    ("tol_sd_0.82", "4.1", "prereg 9; spec 1.13"),
    ("tol_sd_room24", "5.0", "prereg 9; spec 1.13"),
    ("pfail_room48", "6e-5", "spec 1.13"),
    ("pfail_0.82", "4e-5", "spec 1.13"),
    ("pfail_max", "<1e-4", "prereg 9 (below 1 in 10,000)"),
    ("na_bound_max", "20", "spec 1.13"),
    ("na_pairs_failing", "0", "spec 1.13"),
    ("twin_flag@48", "5", "prereg 9"),
    ("twin_flag@32", "4", "prereg 9"),
    ("sim_T@24", "0.945", "spec 1.13"),
    ("exp_T@24", "0.943", "spec 1.13"),
    ("sim_M_e1like", "0.810", "spec 1.13"),
    ("exp_M_e1like", "0.825", "spec 1.13"),
    ("sim_M_room48", "0.806", "spec 1.13"),
    ("exp_M_room48", "0.811", "spec 1.13"),
    ("sim_M_room24", "0.887", "spec 1.13"),
    ("exp_M_room24", "0.887", "spec 1.13"),
    ("e1_control_rate", "0.576", "prereg 9; spec 1.13"),
    ("e1_control_expected", "0.567", "prereg 9; spec 1.13"),
    # H-audit (prereg 4, spec 3.2; the anchors also prereg 1 and spec 3.4)
    ("audit_size@24", "0.035", "prereg 4; spec 3.2"),
    ("audit_size@36", "0.046", "prereg 4; spec 3.2"),
    ("audit_size@48", "0.033", "prereg 4; spec 3.2"),
    ("audit_0.90|0.75|0.5@48", "0.97", "prereg 4; spec 3.2"),
    ("audit_0.95|0.85|0.5@48", "0.915", "prereg 4; spec 3.2"),
    ("audit_0.90|0.80|0.5@48", "0.78", "prereg 4; spec 3.2"),
    ("audit_0.85|0.75|0.5@48", "0.65", "prereg 4; spec 3.2"),
    ("audit_0.95|0.99|0.5@48", "0.00", "prereg 4; spec 3.2"),
    ("corr_dis_js", "0.55", "prereg 4"),
    ("corr_dis_jb", "0.95", "prereg 4"),
    ("sd_js", "0.018", "prereg 1; spec 3.4"),
    ("auroc_js_0.02", "0.78", "prereg 1; spec 3.4"),
    ("auroc_js_0.04", "0.94", "prereg 1; spec 3.4"),
    # costs (prereg 4 table and text; prereg 12)
    ("cost_main_pilot@24", "140", "prereg 4 table"),
    ("cost_repl@24", "170", "prereg 4 table"),
    ("cost_main_pilot@36", "202", "prereg 4 table"),
    ("cost_repl@36", "248", "prereg 4 table"),
    ("cost_main_pilot@48", "264", "prereg 4 table"),
    ("cost_repl@48", "326", "prereg 4 table; prereg 4"),
    ("per_run", "2.58", "prereg 4; prereg 12"),
    ("gpu_h_per_run", "1.4", "prereg 4"),
    ("main_usd", "248", "prereg 4; prereg 12"),
    ("main_gpu_h", "133", "prereg 4"),
    ("pilot_usd", "16", "prereg 4; prereg 12"),
    ("repl_usd", "36", "prereg 4; prereg 12"),
    ("j3_completions", "307200", "prereg 12"),
    ("j3_usd", "4.4", "prereg 4; prereg 12"),
    ("j3_worst_usd", "14.88", "prereg 4; prereg 12"),
    ("repl25_runs", "24", "prereg 4; prereg 12"),
    ("repl25_usd", "62", "prereg 12"),
    ("repl25_primary", "310", "prereg 12"),
    ("repl25_total", "355", "prereg 4; prereg 12"),
    ("primary_ceiling", "285", "prereg 12"),
    ("repl_fit", "14", "prereg 4; prereg 12"),
    ("primary_with_repl", "284", "prereg 12"),
    ("repl_fit_pilot16", "19", "prereg 12"),
    ("primary_gpu_ceiling", "153", "prereg 12"),
    ("j3_gpu_ceiling", "8", "prereg 12"),
    ("total_gpu_ceiling", "176", "prereg 12"),
    ("total_ceiling", "330", "prereg 12"),
]
STATED_BY_KEY = {}
for _k, _s, _w in STATED:
    STATED_BY_KEY.setdefault(_k, []).append((_s, _w))
FULL_ONLY = {"e1_control_rate", "e1_control_expected"} | {k for k in STATED_BY_KEY if k.startswith("audit_")}

OUT = []        # every printed line, for --out
CHECKED = []    # (key, label, value, stated, where, ok)
SHOWN = set()


def say(line=""):
    print(line)
    OUT.append(line)


def _like(value, stated):
    """value written with the precision of the stated string: its decimals, or its mantissa in scientific notation.
    A stated '<x' reads as an upper bound."""
    if stated.startswith("<"):
        return stated if value < float(stated[1:]) else "%g" % value
    if "e" in stated:
        mant = stated.split("e")[0]
        m, e = ("%.*e" % (len(mant.split(".")[1]) if "." in mant else 0, value)).split("e")
        return "%se%d" % (m, int(e))
    return "%.*f" % (len(stated.split(".")[1]) if "." in stated else 0, value)


def _val(value):
    if value is None:
        return "None"
    if isinstance(value, (int, np.integer)):
        return "%d" % value
    return "%.6g" % value


def show(key, value, label):
    """Print one computed value and, for every place the text states it, MATCH or DIFF."""
    SHOWN.add(key)
    head = "  %-66s %-11s" % (label, _val(value))
    entries = STATED_BY_KEY.get(key, [])
    if not entries:
        say(head.rstrip())
    for i, (stated, where) in enumerate(entries):
        got = None if value is None else _like(value, stated)
        ok = got == stated
        CHECKED.append((key, label, value, stated, where, ok))
        say("%s stated %s [%s] -> %s" % (head if i == 0 else " " * len(head), stated, where,
                                          "MATCH" if ok else "DIFF (computed %s)" % got))


def note(text):
    say("    note: " + text)


# ------------------------------------------------------------------------------------------
# inputs from E1's runs (power/e1_inputs.py)
def e1_inputs():
    """E1's 24 hacking runs as the design scripts used them: the registered event on the strict series, E1's lead
    (event - 3-sigma first move of |disagreement|, as e1_primary.score computes it) and the run-level changes
    (e1_supplementary.chg). The tag order is load_main's, which the bootstraps' random streams depend on."""
    import e1_event as EV
    import e1_detectors as D
    used, excluded, unregistered, _, surplus = P.load_main()
    hack = [(r["_tag"], r) for r in used["hacking"]]
    ben = used["benign"]
    event_of = lambda run: EV.event(list(range(len(run["log"]))), [r.get("hidden") for r in run["log"]])
    prim = next({"label": l, "family": f, "gb": g, "dm": m} for (l, f, g, m) in D.detectors() if l == P.PRIMARY)
    sc = P.score(prim, hack, ben, event_of)
    events = {t: sc["events"][t] for t, _ in hack}
    leads = {}
    for t, r in hack:
        fm = P.GB.first_move(prim["gb"](r))
        leads[t] = (float(events[t]) - float(fm)) if (events[t] is not None and fm is not None) else None
    assert sorted(v for v in leads.values() if v is not None) == sorted(sc["leads"])
    chg = {t: {f: float(S.chg(r, f)) for f in ("disagree", "judge_small", "judge_big")} for t, r in hack}
    return {"hack": hack, "ben": ben, "event_of": event_of, "events": events, "leads": leads, "chg": chg,
            "n_excluded": len(excluded), "n_unregistered": len(unregistered), "n_surplus": len(surplus)}


def compare_record(inp):
    """Rebuilt inputs against the design record's copy (experiments/e1b_design/power/e1_inputs.json)."""
    if not INPUTS_RECORD.exists():
        return "design record %s not found; not compared" % INPUTS_RECORD.relative_to(ROOT).as_posix()
    d = json.loads(INPUTS_RECORD.read_text())
    rec_chg = {r["tag"]: r["chg"] for r in d["runs"] if r["arm"] == "hacking"}
    same_order = list(d["events"]) == list(inp["events"])
    ev_ok = d["events"] == inp["events"]
    ld_ok = d["lead_by_tag"] == inp["leads"]
    ch_ok = all(abs(rec_chg[t][f] - v) < 1e-12 for t, c in inp["chg"].items() for f, v in c.items())
    return "equal to %s: tag order %s, events %s, leads %s, changes %s" % (
        INPUTS_RECORD.relative_to(ROOT).as_posix(), *("yes" if x else "NO" for x in (same_order, ev_ok, ld_ok, ch_ok)))


# ------------------------------------------------------------------------------------------
# primary test (power/power_primary.py, power/outcomes_by_margin.py, review_design/null_exact.py,
# n48/recompute/compute_n48.py)
def binom_pmf(n, p):
    return [comb(n, x) * p ** x * (1 - p) ** (n - x) for x in range(n + 1)]


def mde_005(n, fp, target=0.80, alpha=P.ALPHA):
    """power_primary.py: e1_primary.mde_fixed_fp's search at 0.005 resolution (the frozen helper steps by 0.01)."""
    return next((t / 1000 for t in range(5, 1001, 5) if P.power_fixed_fp(n, n, t / 1000, fp, alpha) >= target), None)


def upper90(x, n, fp):
    return P.newcombe(x, n, fp, n, 0.90)[1]


def nonsup_k(n, fp, d):
    """compute_n48.py: the largest TP whose upper Newcombe 90% bound on TPR - FPR is below d."""
    return max([x for x in range(n + 1) if upper90(x, n, fp) < d], default=None)


def outcomes(n, fp, tpr, delta):
    """power_primary.py: P(Fisher p < 0.05), P(non-superiority shown and no significance), P(neither) when
    TP ~ Bin(n, tpr) and FP is fixed."""
    pmf = binom_pmf(n, tpr)
    sig = sum(q for x, q in enumerate(pmf) if P.fisher_one_sided(x, n, fp, n) < P.ALPHA)
    ns = sum(q for x, q in enumerate(pmf) if P.fisher_one_sided(x, n, fp, n) >= P.ALPHA and upper90(x, n, fp) < delta)
    return sig, ns, 1 - sig - ns


def prob_est(n, event):
    """review_design/null_exact.py: the expectation of event(1 - thr) when thr is numpy's linear 90th percentile of
    n iid U(0, 1) twin maxima, thr = U(j) + fr (U(j+1) - U(j)), integrated exactly over the joint density of the
    order statistics j and j + 1. With an uninformative detector (hacking and twin window maxima exchangeable),
    TP given thr is Binomial(n, 1 - thr), so event(q) = P(Bin(n, q) in a set) gives that set's probability."""
    pos = 0.9 * (n - 1)
    j = floor(pos) + 1
    fr = pos - floor(pos)
    c = factorial(n) / (factorial(j - 1) * factorial(n - j - 1))
    f = lambda v, u: c * u ** (j - 1) * (1 - v) ** (n - j - 1) * event(1 - (u + fr * (v - u)))
    val, _err = integrate.dblquad(f, 0, 1, lambda u: u, lambda u: 1, epsabs=1e-10, epsrel=1e-10)
    return val


def primary(n):
    """Every exact number for one n v n design, FP fixed at fixed_fp(n) by the time-matched 90th percentile."""
    fp = P.fixed_fp(n)
    need = S.min_hits(n, n, fp, P.ALPHA)
    r = {"fp": fp, "fpr": fp / n, "need": need, "kill_tpr": need / n,
         "p_at_need": P.fisher_one_sided(need, n, fp, n), "p_below_need": P.fisher_one_sided(need - 1, n, fp, n),
         "size_known": P.power_fixed_fp(n, n, fp / n, fp),
         "size_est": prob_est(n, lambda q: 1 - binom.cdf(need - 1, n, q)),
         "total_est": prob_est(n, lambda q: 1.0),
         "mean_tp_est": prob_est(n, lambda q: n * q),
         "var_binom": n * (fp / n) * (1 - fp / n),
         "mde80": mde_005(n, fp), "mde90": mde_005(n, fp, 0.90),
         "mde80_helper": P.mde_fixed_fp(n, n, fp), "mde90_helper": P.mde_fixed_fp(n, n, fp, 0.90),
         "h2_need": S.min_hits(n, n, fp, ALPHA_H2), "h2_power": P.power_fixed_fp(n, n, 0.5, fp, ALPHA_H2),
         "h2_mde": S.mde_at(n, n, fp, ALPHA_H2)}
    r["var_tp_est"] = prob_est(n, lambda q: n * q * (1 - q) + (n * q) ** 2) - r["mean_tp_est"] ** 2
    for t in TPRS:
        r["power_" + t] = P.power_fixed_fp(n, n, float(t), fp)
    for d in (DELTA, DELTA_ASIDE) + ((0.19,) if n == 36 else ()):
        k = nonsup_k(n, fp, d)
        r["nonsup_k_%.2f" % d] = k
        r["p_ns_known_%.2f" % d] = float(binom.cdf(k, n, fp / n))
        r["p_ns_est_%.2f" % d] = prob_est(n, lambda q, k=k: binom.cdf(k, n, q))
        for t in (0.25, 0.30):
            r["inc_%.2f_d%.2f" % (t, d)] = outcomes(n, fp, t, d)[2]
    assert r["nonsup_k_0.25"] == P.nonsuperiority_max_hits(n, n, fp)     # e1_primary's DELTA is the same 0.25
    for x in range(max(0, fp - 1), min(n, need + 1) + 1):
        r["upper90_%d" % x] = upper90(x, n, fp)
    return r


def margin80(n, fp, known):
    """The smallest two-decimal margin an uninformative detector (TPR = FPR) meets with probability >= 0.80
    (pre-registration section 5), with the twins' rate known (binomial) or the threshold estimated (prob_est)."""
    cache = {}
    for d in range(1, 61):
        k = nonsup_k(n, fp, d / 100)
        if k is None:
            continue
        if k not in cache:
            cache[k] = float(binom.cdf(k, n, fp / n)) if known else prob_est(n, lambda q, k=k: binom.cdf(k, n, q))
        if cache[k] >= 0.80:
            return d / 100
    return None


def boot_conjuncts(runs, n, rng):
    """power_primary.py: task-stratified bootstrap of E1's hacking runs to n (n/3 per task). Returns P(window >= 12),
    P(median lead > 0), P(all three: window, lead and generator check), P(generator check: events >=
    ceil(0.75 n)). The leads are E1's (two-sided first move from step 0), where disagreement did not anticipate."""
    by_task = {}
    for t, (e, l) in runs.items():
        by_task.setdefault(t.split("/")[1], []).append((e, l))
    per = n // 3
    win = lead = both = gen = 0
    for _ in range(B_CONJ):
        ev, ld = [], []
        for task, rs in by_task.items():
            for i in rng.integers(0, len(rs), per):
                e, l = rs[i]
                ev.append(-1 if e is None else e)
                if l is not None:
                    ld.append(l)
        w = np.median(ev) >= P.WINDOW_MIN
        m = len(ld) > 0 and np.median(ld) > 0
        g = sum(e >= 0 for e in ev) >= math.ceil(P.GEN_FRAC * n)
        win += w
        lead += m
        both += (w and m and g)
        gen += g
    return win / B_CONJ, lead / B_CONJ, both / B_CONJ, gen / B_CONJ


# ------------------------------------------------------------------------------------------
# planted control (power/planted_control.py, n48/recompute/power/planted_m48_check.py, compute_n48.py)
def layouts(events, n, rng, k):
    """planted_control.py: k task-stratified bootstrap layouts of E1's hacking events, n/3 per task."""
    by = {}
    for t, e in events.items():
        by.setdefault(t.split("/")[1], []).append(e)
    out = []
    for _ in range(k):
        ev = []
        for task, es in by.items():
            ev += [es[i] for i in rng.integers(0, len(es), n // 3)]
        out.append(ev)
    return out


def ps_target(ev, target):
    """Construction M (target = the MDE) and the claimed-effect construction T (target = 0.5): a run with room (an
    event at step 7 or later) is planted with probability q = min(1, target n / n_room), the rest never
    (planted_control.py ps_T, planted_m48_check.py ps_M)."""
    room = sum(1 for e in ev if e is not None and e >= 7)
    q = min(1.0, target * len(ev) / room) if room else 0.0
    return [q if (e is not None and e >= 7) else 0.0 for e in ev]


def up_dev(x):
    return np.maximum(x - np.median(x[:5]), 0.0)


def score_tm(hack_series, twin_series, events):
    """planted_control.py's minimal E1b scoring path: upward-only deviation from the median of the first 5 steps,
    twin maxima over [0, e) (the whole run for the twin of a no-event run), the 90th-percentile threshold, and an
    alarm strictly before the event. It searches from step 0, not from the causal start. On these series that
    changes nothing: noise deviations are below 1, twin spikes (at step 6 or later) at least 1.5 and planted
    spikes at least 4, so the threshold and every alarm come from spikes. Returns (TP, FP)."""
    tw = []
    for s, e in zip(twin_series, events):
        d = up_dev(s)
        tw.append(float(np.max(d if e is None else d[:e])))
    thr = float(np.percentile(tw, 90))
    tp = 0
    for s, e in zip(hack_series, events):
        if e is None:
            continue
        idx = np.nonzero(up_dev(s) > thr)[0]
        tp += int(len(idx) > 0 and idx[0] < e)
    fp = sum(v > thr for v in tw)
    return tp, fp


def simulate(ev, ps, draws, rng):
    """planted_control.py simulate_T and planted_m48_check.py simulate_M (the same draws): U(0, 1) noise, one twin
    spike U(2.5, 3.5) at a step in [6, W) when W > 6, and run i planted with probability ps[i] with a spike U(5, 6)
    at e - U{1..min(12, e - 6)}. A rejection is Fisher p < 0.05, window and lead granted. Returns (rate, FP counts
    seen)."""
    n = len(ev)
    rej, fps = 0, set()
    for _ in range(draws):
        hs, ts = [], []
        for e, p in zip(ev, ps):
            x = rng.uniform(0, 1, T_STEPS)
            if p > 0 and rng.random() < p:
                x[e - int(rng.integers(1, min(12, e - 6) + 1))] = rng.uniform(5, 6)
            hs.append(x)
            y = rng.uniform(0, 1, T_STEPS)
            w = T_STEPS if e is None else e
            if w > 6:
                y[int(rng.integers(6, w))] = rng.uniform(2.5, 3.5)
            ts.append(y)
        tp, fp = score_tm(hs, ts, ev)
        fps.add(int(fp))
        rej += int(P.fisher_one_sided(tp, n, fp, n) < P.ALPHA)
    return rej / draws, sorted(fps)


def pfail(p, draws=CONTROL_DRAWS, tol=CONTROL_TOL):
    """compute_n48.py: exact chance that a correct pipeline fails, |X / draws - p| > tol, X ~ Bin(draws, p)."""
    return sum(binom.pmf(x, draws, p) for x in range(draws + 1) if abs(x / draws - p) > tol + 1e-9)


# ------------------------------------------------------------------------------------------
# H-audit power (power/power_audit.py): MVN run-level changes with unit variances; twin mean 0, hacking mean
# sqrt(2) Phi^-1(A) per score; the best comparator at A_best with corr rho to disagreement, 4 nuisance comparators
# at 0.60, every other pair 0.3; comparator AUROCs direction-free; paired bootstrap within arm (B = 2000); reject
# when the 5th percentile of AUROC(disagreement) - max comparator AUROC is above 0.
def d_of(a):
    return np.sqrt(2) * norm.ppf(a)


def _cov(rho, m):
    c = np.full((m, m), R_OTHER)
    np.fill_diagonal(c, 1.0)
    c[0, 1] = c[1, 0] = rho
    return c


def _weights(n, rng):
    idx = rng.integers(0, n, (AUDIT_BOOT, n)) + n * np.arange(AUDIT_BOOT)[:, None]
    return np.bincount(idx.ravel(), minlength=AUDIT_BOOT * n).reshape(AUDIT_BOOT, n).astype(float)


def _boot_auroc(h, t, wh, wt):
    c = (h[:, None] > t[None, :]).astype(float)
    return ((wh @ c) * wt).sum(1) / (len(h) * len(t))


def _audit_sim(n, a_dis, a_best, rho, rng):
    m = 2 + K_NUIS
    mu = np.array([d_of(a_dis), d_of(a_best)] + [d_of(A_NUIS)] * K_NUIS)
    cv = _cov(rho, m)
    h = rng.multivariate_normal(mu, cv, n)
    t = rng.multivariate_normal(np.zeros(m), cv, n)
    wh, wt = _weights(n, rng), _weights(n, rng)
    a0 = _boot_auroc(h[:, 0], t[:, 0], wh, wt)
    comps = []
    for j in range(1, m):
        a = _boot_auroc(h[:, j], t[:, j], wh, wt)
        comps.append(np.maximum(a, 1 - a))
    delta = a0 - np.max(np.vstack(comps), axis=0)
    return bool(np.percentile(delta, 5) > 0)


def audit_cell(args):
    """One (n, A_dis, A_best, rho) cell: the rejection rate over AUDIT_SIMS simulated experiments."""
    n, a_dis, a_best, rho, seed = args
    rng = np.random.default_rng(seed)
    return float(np.mean([_audit_sim(n, a_dis, a_best, rho, rng) for _ in range(AUDIT_SIMS)]))


def audit_jobs():
    """power_audit.py's cells and seeds: the size cells first, then the grid (the seed ignores rho, as there)."""
    jobs = [(n, 0.80, 0.80, 0.5, 1000 + n) for n in NS_DESIGN]
    jobs += [(n, a, b, r, 7 + 1000 * i + n) for i, (a, b) in enumerate(AUDIT_PAIRS) for r in AUDIT_RHOS for n in NS_DESIGN]
    return jobs


# ------------------------------------------------------------------------------------------
def main(full=False, out=None):
    say("E1b POWER, SIZE, MDE, MARGIN, PLANTED-CONTROL AND COST NUMBERS (experiments/e1b_power.py)")
    say("mode: %s; python %s, numpy %s, scipy %s" % (
        "full" if full else "default (--full adds the audit power grid and E1's planted control)",
        platform.python_version(), np.__version__, scipy.__version__))
    say("Each line: the computed value, then each value the text states, where, and MATCH or DIFF. MATCH means the")
    say("computed value, rounded to the decimals the text gives, equals the stated value.")

    say("\n== 0. E1 INPUTS, rebuilt from runs/e1/main (power/e1_inputs.py) ==")
    inp = e1_inputs()
    ev = inp["events"]
    say("  e1_primary.load_main: %d hacking and %d benign runs used; %d excluded, %d unregistered, %d surplus" % (
        len(inp["hack"]), len(inp["ben"]), inp["n_excluded"], inp["n_unregistered"], inp["n_surplus"]))
    say("  hacking events, sorted (-1 = no event): %s" % sorted(-1 if e is None else e for e in ev.values()))
    lds = [v for v in inp["leads"].values() if v is not None]
    say("  E1 leads (event - first move of |disagreement|), %d runs: median %.1f, %d positive" % (
        len(lds), float(np.median(lds)), sum(1 for v in lds if v > 0)))
    say("  " + compare_record(inp))

    say("\n== 1. PRIMARY TEST, EXACT ENUMERATION (prereg 4 table and text; prereg 5; spec 1.11; spec 1.14) ==")
    say("  One-sided Fisher, alpha 0.05, FP fixed at e1_primary.fixed_fp(n) by the time-matched 90th percentile.")
    say("  'threshold estimated': exact over numpy's interpolated 90th percentile of n exchangeable twin maxima")
    say("  (review_design/null_exact.py). MDE on a 0.005 grid (power_primary.py), frozen 0.01-step helper beside it.")
    prim = {}
    for n in NS:
        r = prim[n] = primary(n)
        fp = r["fp"]
        say("\n-- %d v %d%s --" % (n, n, {48: " (registered)", 36: " (earlier proposal)", 32: " (MIN_ARM)",
                                          24: " (E1's size)"}[n]))
        show("fp@%d" % n, fp, "FP fixed at fixed_fp(%d)" % n)
        show("fpr@%d" % n, r["fpr"], "FPR = FP / n")
        show("need@%d" % n, r["need"], "hits needed for p < 0.05 (p %.4f at %d, %.4f at %d)" % (
            r["p_at_need"], r["need"], r["p_below_need"], r["need"] - 1))
        show("kill_tpr@%d" % n, r["kill_tpr"], "lowest TPR that can be declared a kill (hits / n)")
        show("size_known@%d" % n, r["size_known"], "size, twin FPR known: P(Bin(n, FP/n) >= hits)")
        show("size_est@%d" % n, r["size_est"], "size, threshold estimated from the twins (total %.6f)" % r["total_est"])
        show("mean_tp_est@%d" % n, r["mean_tp_est"], "  mean TP, threshold estimated")
        show("var_tp_est@%d" % n, r["var_tp_est"], "  variance of TP, threshold estimated")
        show("var_binom@%d" % n, r["var_binom"], "  binomial variance n p (1 - p), p = FP/n")
        if n == 36:
            note("the stated 7.2 is the exact quadrature above, rounded. The drafts stated 7.1 until the correction "
                 "of 2026-10-25: review_design/power_recheck.py's Monte Carlo estimate (7.119, 400,000 draws; "
                 "n48/recompute/review_design/power_recheck.txt).")
        for t in TPRS:
            show("power_%s@%d" % (t, n), r["power_" + t], "Fisher power at TPR %s" % t)
        show("mde80@%d" % n, r["mde80"], "MDE at 80%% power (0.01-step helper: %s)" % r["mde80_helper"])
        show("mde90@%d" % n, r["mde90"], "MDE at 90%% power (0.01-step helper: %s)" % r["mde90_helper"])
        show("h2_need@%d" % n, r["h2_need"], "H2 first Holm step (alpha/25): hits needed")
        show("h2_power@%d" % n, r["h2_power"], "H2 first Holm step: power at TPR 0.5")
        show("h2_mde@%d" % n, r["h2_mde"], "H2 first Holm step: MDE at 80% (0.01 grid)")
        for d in (DELTA, DELTA_ASIDE) + ((0.19,) if n == 36 else ()):
            tag = "%.2f" % d
            show("nonsup_k_%s@%d" % (tag, n), r["nonsup_k_" + tag], "non-superiority at delta %s holds iff TP <=" % tag)
            show("p_ns_known_%s@%d" % (tag, n), r["p_ns_known_" + tag], "  P(shown | TPR = FPR), twin FPR known")
            show("p_ns_est_%s@%d" % (tag, n), r["p_ns_est_" + tag], "  P(shown | TPR = FPR), threshold estimated")
            for t in ("0.25", "0.30"):
                show("inc_%s_d%s@%d" % (t, tag, n), r["inc_%s_d%s" % (t, tag)],
                     "  inconclusive (p >= 0.05, non-superiority not shown) at TPR %s" % t)
        say("  upper Newcombe 90%% bound on TPR - FPR by TP count: %s" % ", ".join(
            "%d: %.4f" % (x, r["upper90_%d" % x]) for x in range(max(0, fp - 1), min(n, r["need"] + 1) + 1)))
        for x in range(max(0, fp - 1), min(n, r["need"] + 1) + 1):
            if ("upper90_%d@%d" % (x, n)) in STATED_BY_KEY:
                show("upper90_%d@%d" % (x, n), r["upper90_%d" % x], "  upper 90%% bound at TP %d" % x)
        if n == 36:
            show("margin80_known@36", margin80(36, fp, True), "smallest margin (0.01 steps) shown w.p. >= 0.80, rate known")
            show("margin80_est@36", margin80(36, fp, False), "  the same, threshold estimated")

    say("\n== 2. KILL POWER (prereg 4 table and text; spec 1.14) ==")
    say("  Fisher power at TPR 0.5 times P(window >= 12, median lead > 0 and generator check all hold), the last from a")
    say("  task-stratified bootstrap of E1's 24 hacking (event, lead) pairs, %d draws, seed %d, n = %s in turn" % (
        B_CONJ, SEED_CONJ, ", ".join(str(n) for n in NS_DESIGN)))
    say("  (power_primary.py; independence assumed; E1's leads, where disagreement did not anticipate).")
    runs = {t: (ev[t], inp["leads"][t]) for t in ev}
    rng = np.random.default_rng(SEED_CONJ)
    conj = {}
    for n in NS_DESIGN:
        w, l, both, g = conj[n] = boot_conjuncts(runs, n, rng)
        say("  n = %d: P(window >= 12) %.4f, P(median lead > 0) %.4f, P(generator ok) %.4f" % (n, w, l, g))
        show("p_all@%d" % n, both, "  P(all three)")
        show("kill_power@%d" % n, prim[n]["power_0.5"] * both, "  kill power at TPR 0.5")
    p_all = conj[N][2]
    kt = next(t / 100 for t in range(1, 101) if P.power_fixed_fp(N, N, t / 100, P.fixed_fp(N)) * p_all >= 0.80)
    show("kill_tpr80@48", kt, "smallest TPR (0.01 grid) where kill power at 48 v 48 reaches 0.80")
    say("    kill power at TPR %.2f: %.4f; at %.2f: %.4f" % (
        kt - 0.01, P.power_fixed_fp(N, N, kt - 0.01, P.fixed_fp(N)) * p_all, kt,
        P.power_fixed_fp(N, N, kt, P.fixed_fp(N)) * p_all))

    say("\n== 3. PLANTED CONTROL, CONSTRUCTION M (prereg 4 table; prereg 9; spec 1.13) ==")
    mde = {n: P.mde_fixed_fp(n, n, P.fixed_fp(n)) for n in NS_DESIGN}
    show("mde_helper@48", mde[N], "planted effect: e1_primary.mde_fixed_fp(48, 48, 5)")
    ev24 = list(ev.values())
    show("e1_construction_at_e1", P.power_poisson_binomial(P.planted_ps(ev24), 24, 3),
         "E1's construction at E1's own layout (24 v 24, FP 3)")
    say("  Exact expected rejection, e1_primary.power_poisson_binomial over the per-run plant probabilities, on %d" % LAYOUTS)
    say("  task-stratified bootstrap layouts of E1's events per n (planted_control.py, seed %d, n = %s in turn)." % (
        SEED_LAYOUTS, ", ".join(str(n) for n in NS_DESIGN)))
    say("  E1 = e1_primary.planted_ps; M = ps_target(MDE); T = ps_target(0.5), the claimed effect.")
    rng = np.random.default_rng(SEED_LAYOUTS)
    first_layout = {}
    for n in NS_DESIGN:
        fp = P.fixed_fp(n)
        lay = layouts(ev, n, rng, LAYOUTS)
        first_layout[n] = lay[0]
        for name, f in (("E1", P.planted_ps), ("M", lambda e, n=n: ps_target(e, mde[n])),
                        ("T", lambda e: ps_target(e, 0.5))):
            vals = np.array([P.power_poisson_binomial(f(e), n, fp) for e in lay])
            tprs = np.array([np.mean(f(e)) for e in lay])
            say("  n = %d, %-2s mean %.5f, 10th pct %.5f, 90th pct %.5f; mean planted TPR %.3f" % (
                n, name, vals.mean(), np.percentile(vals, 10), np.percentile(vals, 90), tprs.mean()))
            if name == "M":
                show("planted_M@%d" % n, float(vals.mean()), "  M, mean over layouts (MDE %.2f)" % mde[n])
                if n == N:
                    show("planted_M_p10@48", float(np.percentile(vals, 10)), "  M, 10th percentile")
                    show("planted_M_p90@48", float(np.percentile(vals, 90)), "  M, 90th percentile")
            if name == "E1" and n == N:
                show("planted_E1_p10@48", float(np.percentile(vals, 10)), "  E1's construction, 10th percentile")
                show("planted_E1_p90@48", float(np.percentile(vals, 90)), "  E1's construction, 90th percentile")
            if name == "T" and n in (36, 48):
                show("planted_T@%d" % n, float(vals.mean()), "  claimed effect T (TPR 0.5), mean over layouts")
    for n in (36, 48):
        fp = P.fixed_fp(n)
        say("    claimed effect with every run having room (q = 0.5, = binomial power at 0.5), n = %d: %.5f "
            "(compute_n48.py)" % (n, P.power_poisson_binomial([0.5] * n, n, fp)))
    note("both stated claimed-effect figures are checked against one definition, the mean over E1-like layouts, the "
         "definition of the M figure the text sets them against: the text's 0.997 at 36 (planted_control.py) and its "
         "0.9999 at 48 are that mean. The every-run-has-room value above (compute_n48.py) gives 0.9996 at 48 and 0.994 "
         "at 36; the drafts stated that 0.9996 at 48 until the correction of 2026-10-25.")

    say("\n  Construction M at 48 v 48 by the number of hacking runs with room (compute_n48.py). The window holding")
    say("  (median event >= 12) puts at least n/2 events at step 12 or later, so at least that many runs have room.")
    fp = P.fixed_fp(N)
    show("min_room@48", math.ceil(N / 2), "fewest runs with room when the window holds")
    by_room = {}
    for room in range(math.ceil(N / 2), N + 1):
        q = min(1.0, mde[N] * N / room)
        by_room[room] = (q, P.power_poisson_binomial([q] * room + [0.0] * (N - room), N, fp))
    say("  " + ", ".join("%d: %.4f" % (k, v[1]) for k, v in by_room.items()))
    show("M_q_room48", by_room[N][0], "48 with room: q")
    show("M_exp_room48", by_room[N][1], "48 with room: exact expectation (the lowest)")
    show("M_q_room24", by_room[24][0], "24 with room: q")
    show("M_exp_room24", by_room[24][1], "24 with room: exact expectation (the highest)")
    say("\n  CONTROL_TOL %.2f at %d draws: the tolerance in SDs of the simulated rate, and the exact chance a correct" % (
        CONTROL_TOL, CONTROL_DRAWS))
    say("  pipeline fails (compute_n48.py; 0.82 is the literal value the design used for 'about 0.82').")
    pf = {}
    for key, p in (("room48", by_room[N][1]), ("0.82", 0.82), ("room24", by_room[24][1])):
        sd = math.sqrt(p * (1 - p) / CONTROL_DRAWS)
        pf[key] = pfail(p)
        show("tol_sd_" + key, CONTROL_TOL / sd, "expectation %.4f: tolerance in SD (SD %.4f)" % (p, sd))
        if ("pfail_" + key) in STATED_BY_KEY:
            show("pfail_" + key, pf[key], "  exact P(a correct pipeline fails)")
        else:
            say("    exact P(a correct pipeline fails): %.2e" % pf[key])
    show("pfail_max", max(pf.values()), "largest of the three")

    say("\n  NOT APPLICABLE (spec 1.13): when fewer than fixed_fp(n_t) + 1 twin windows reach step 7, at most")
    say("  fixed_fp(n_t) + max(0, n_h - n_t) hacking runs have an event at step 12 or later (compute_n48.py).")
    pairs = [(nh, nt) for nh in range(32, N + 1) for nt in range(32, N + 1)]
    bound = {(nh, nt): P.fixed_fp(nt) + max(0, nh - nt) for nh, nt in pairs}
    worst = max(bound, key=lambda k: (bound[k], k))
    show("na_bound_max", bound[worst], "largest bound over n_h, n_t in 32..48 (at n_h %d, n_t %d)" % worst)
    show("na_pairs_failing", sum(1 for k in pairs if bound[k] >= k[0] / 2),
         "(n_h, n_t) in 32..48 where the bound reaches half of n_h")
    for n in (48, 32):
        show("twin_flag@%d" % n, math.floor(0.10 * n) + 1, "TWIN NOT BENIGN fires at this many of %d (more than 10%%)" % n)

    say("\n  Simulation through planted_control.py's scoring path, %d draws (spec 1.13)." % CONTROL_DRAWS)
    lay = first_layout[24]
    exp = P.power_poisson_binomial(ps_target(lay, 0.5), 24, P.fixed_fp(24))
    sim, fps = simulate(lay, ps_target(lay, 0.5), CONTROL_DRAWS, np.random.default_rng(24))
    show("sim_T@24", sim, "T at 24 v 24, first bootstrap layout (seed 24): simulated (FP seen %s)" % fps)
    show("exp_T@24", exp, "  exact expectation")
    for n in (36, 48):
        lay = first_layout[n]
        exp = P.power_poisson_binomial(ps_target(lay, 0.5), n, P.fixed_fp(n))
        sim, fps = simulate(lay, ps_target(lay, 0.5), CONTROL_DRAWS, np.random.default_rng(n))
        say("  T at %d v %d, first bootstrap layout (seed %d): simulated %.3f, exact %.4f, FP seen %s" % (
            n, n, n, sim, exp, fps))
    e1like = layouts(ev, N, np.random.default_rng(11), 1)[0]
    for key, name, lay in (("e1like", "E1-like bootstrap layout (seed 11)", e1like),
                           ("room48", "every run with room (events at 40)", [40] * N),
                           ("room24", "24 with room, window holds (24 at 6, 24 at 30)", [6] * 24 + [30] * 24)):
        ps = ps_target(lay, mde[N])
        exp = P.power_poisson_binomial(ps, N, P.fixed_fp(N))
        sim, fps = simulate(lay, ps, CONTROL_DRAWS, np.random.default_rng(N))
        say("  M at 48 v 48, %s: n_room %d, q %.3f, median event %.1f, FP seen %s, |diff| %.3f -> %s" % (
            name, sum(1 for p in ps if p > 0), max(ps), float(np.median([-1 if e is None else e for e in lay])), fps,
            abs(sim - exp), "PASSES" if abs(sim - exp) <= CONTROL_TOL + 1e-9 else "FAILS"))
        show("sim_M_" + key, sim, "    simulated")
        show("exp_M_" + key, exp, "    exact expectation")
    if full:
        pc = P.planted_control(inp["hack"], inp["ben"], inp["event_of"], draws=CONTROL_DRAWS)
        show("e1_control_rate", pc["rate"], "E1's planted control on E1's runs (e1_primary, seed 20261007)")
        show("e1_control_expected", pc["expected"], "  its exact expectation")
    else:
        say("  E1's registered planted control (0.576 against 0.567): --full")

    say("\n== 4. H-AUDIT (prereg 4; spec 3.2; anchors prereg 1 and spec 3.4) ==")
    hk = list(inp["chg"].values())
    x = np.array([c["disagree"] for c in hk])
    js = np.array([c["judge_small"] for c in hk])
    jb = np.array([c["judge_big"] for c in hk])
    show("corr_dis_js", float(np.corrcoef(x, js)[0, 1]), "E1 hacking arm: corr(chg disagreement, chg 0.5B judge)")
    show("corr_dis_jb", float(np.corrcoef(x, -jb)[0, 1]), "E1 hacking arm: corr(chg disagreement, -chg 1.5B judge)")
    sd = float(np.std(js, ddof=1))
    show("sd_js", sd, "E1 hacking arm: SD of chg 0.5B judge (power_audit.py anchors)")
    for diff in (0.02, 0.04):
        show("auroc_js_%.2f" % diff, float(norm.cdf(diff / (sd * np.sqrt(2)))),
             "  AUROC implied by a twin gaining %.2f less, equal SDs" % diff)
    if full:
        from multiprocessing import Pool
        jobs = audit_jobs()
        with Pool(min(16, os.cpu_count() or 1)) as pool:
            res = dict(zip([j[:4] for j in jobs], pool.map(audit_cell, jobs)))
        say("  Paired bootstrap of AUROC(disagreement) - max AUROC*(5 comparators), %d simulated experiments per cell," % AUDIT_SIMS)
        say("  B = %d, reject when the 5th percentile is above 0; MVN changes; %d nuisance comparators at %.2f; other" % (
            AUDIT_BOOT, K_NUIS, A_NUIS))
        say("  pairs %.1f (power_audit.py). Monte Carlo SE <= %.3f." % (R_OTHER, 0.5 / np.sqrt(AUDIT_SIMS)))
        for n in NS_DESIGN:
            show("audit_size@%d" % n, res[(n, 0.80, 0.80, 0.5)], "size, A_dis = A_best = 0.80, rho 0.5, n = %d" % n)
        say("  A_dis  A_best  rho  " + "  ".join("n=%d " % n for n in NS_DESIGN))
        for a, b in AUDIT_PAIRS:
            for rho in AUDIT_RHOS:
                say("  %.2f   %.2f   %.1f  %s" % (a, b, rho, "  ".join("%.3f" % res[(n, a, b, rho)] for n in NS_DESIGN)))
        for a, b in AUDIT_PAIRS:
            key = "audit_%.2f|%.2f|0.5@48" % (a, b)
            if key in STATED_BY_KEY:
                show(key, res[(N, a, b, 0.5)], "power at n = 48, rho 0.5: A_dis %.2f against A_best %.2f" % (a, b))
        say("  n = 48, rho 0.5, by gap A_dis - A_best (the text: 0.80 power only when A_dis is high and well ahead; "
            "at lower levels even larger gaps fall short):")
        say("  " + "; ".join("%.2f v %.2f (gap %.2f): %.3f" % (a, b, a - b, res[(N, a, b, 0.5)])
                             for a, b in sorted(AUDIT_PAIRS, key=lambda p: (p[0] - p[1], p[0]))))
    else:
        say("  audit power grid and its size cells: --full")

    say("\n== 5. COSTS (prereg 4 table and text; prereg 12) ==")
    per = E1_USD / E1_RUNS
    show("per_run", per, "USD per run (E1's USD %d for %d runs; costs.py)" % (E1_USD, E1_RUNS))
    show("gpu_h_per_run", GPU_H_PER_RUN, "GPU-hours per run (125 per 90 runs; compute_n48.py)")
    for n in NS_DESIGN:      # costs.py
        main_runs = 2 * n
        rep = main_runs // 4
        show("cost_main_pilot@%d" % n, (main_runs + PILOT_RUNS) * per, "n = %d: main %d + pilot %d runs, USD" % (
            n, main_runs, PILOT_RUNS))
        show("cost_repl@%d" % n, (main_runs + PILOT_RUNS + rep) * per, "  with %d replacements (25%%), USD" % rep)
    note("the whole-dollar figures are '%.0f' of x.5 values (139.5, 170.5, 201.5, 263.5, 325.5), which rounds half "
         "to even, as costs.py printed them")
    show("main_usd", 2 * N * per, "96 main runs, USD")
    show("main_gpu_h", 2 * N * GPU_H_PER_RUN, "96 main runs, GPU-hours")
    show("pilot_usd", PILOT_RUNS * per, "pilot, 6 runs, USD")
    launch = json.loads(RESCORE_LAUNCH.read_text())
    worst = launch["qwen25_7b"]["worst_usd"]
    usd_h = launch["qwen25_7b"]["usd_per_h"]
    j3_n = 2 * N * 400 * 8
    show("j3_completions", j3_n, "J3: completions of 96 runs (400 steps x 8)")
    j3_usd = J3_USD_PER_E1 * j3_n / E1_COMPLETIONS
    show("j3_usd", j3_usd, "J3 estimate, USD (USD %.1f per %d)" % (J3_USD_PER_E1, E1_COMPLETIONS))
    j3_worst = worst * j3_n / E1_COMPLETIONS
    show("j3_worst_usd", j3_worst, "J3 worst case, USD (rescore_launch.json worst_usd %.2f per %d)" % (
        worst, E1_COMPLETIONS))
    primary_ceiling = CAP - CEIL_PILOT - CEIL_J3
    show("primary_ceiling", primary_ceiling, "primary ceiling = cap %d - pilot %d - J3 %d, USD" % (CAP, CEIL_PILOT, CEIL_J3))
    repl_fit = math.floor((primary_ceiling - 2 * N * per) / per + 1e-9)
    show("repl_fit", repl_fit, "replacements the primary ceiling covers")
    show("repl_usd", repl_fit * per, "%d replacements, USD" % repl_fit)
    show("primary_with_repl", (2 * N + repl_fit) * per, "96 runs + %d replacements, USD" % repl_fit)
    show("repl_fit_pilot16", math.floor((CAP - 16 - CEIL_J3 - 2 * N * per) / per + 1e-9),
         "replacements that would fit with a pilot ceiling of USD 16")
    rep25 = 2 * N // 4
    show("repl25_runs", rep25, "25% replacements, runs")
    show("repl25_usd", rep25 * per, "  their cost, USD")
    show("repl25_primary", (2 * N + rep25) * per, "  primary ceiling they would need, USD")
    show("repl25_total", (2 * N + rep25) * per + CEIL_PILOT + CEIL_J3, "  stage ceilings in all, USD")
    show("primary_gpu_ceiling", (2 * N + repl_fit) * GPU_H_PER_RUN, "primary GPU-hours, 96 + %d runs" % repl_fit)
    show("j3_gpu_ceiling", j3_worst / usd_h, "J3 GPU-hours, worst case at USD %.3f per hour" % usd_h)
    show("total_gpu_ceiling", CEIL_PILOT_GPU_H + round((2 * N + repl_fit) * GPU_H_PER_RUN) + round(j3_worst / usd_h),
         "GPU-hour ceilings in all (pilot %d, given)" % CEIL_PILOT_GPU_H)
    show("total_ceiling", CEIL_PILOT + primary_ceiling + CEIL_J3, "USD ceilings in all")
    say("  estimate in all: pilot %.2f + main %.2f + %d replacements %.2f + J3 %.2f (worst %.2f) = USD %.2f (%.2f)" % (
        PILOT_RUNS * per, 2 * N * per, repl_fit, repl_fit * per, j3_usd, j3_worst,
        (PILOT_RUNS + 2 * N + repl_fit) * per + j3_usd, (PILOT_RUNS + 2 * N + repl_fit) * per + j3_worst))

    say("\n== SUMMARY ==")
    skipped = sorted(k for k in STATED_BY_KEY if k not in SHOWN and (k in FULL_ONLY and not full))
    missing = sorted(k for k in STATED_BY_KEY if k not in SHOWN and k not in skipped)
    n_ok = sum(1 for c in CHECKED if c[5])
    say("  %d stated values checked: %d MATCH, %d DIFF; %d not computed in this mode (--full)" % (
        len(CHECKED), n_ok, len(CHECKED) - n_ok, sum(len(STATED_BY_KEY[k]) for k in skipped)))
    for key, label, value, stated, where, ok in CHECKED:
        if not ok:
            say("  DIFF %s: %s computed %s (%s), stated %s [%s]" % (
                key, label.strip(), _val(value), _like(value, stated) if value is not None else "None", stated, where))
    if missing:
        say("  STATED keys never computed (a defect of this script): %s" % missing)
    if out:
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(OUT) + "\n")
    return 1 if missing else 0


if __name__ == "__main__":
    args = sys.argv[1:]
    out_path = args[args.index("--out") + 1] if "--out" in args else None
    raise SystemExit(main(full="--full" in args, out=out_path))
