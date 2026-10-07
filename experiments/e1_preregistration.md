# 02 Pre-registration: E1, slow-onset arm

Kit file 02 of 09. Paper: Audit or Early Warning (AaravOS/papers/audit_or_early_warning.md). Version: 1 DRAFT. Frozen: no.

DRAFT by the assistant on 2026-10-07. Under kit 02, Aarav writes sections 1 and 11 himself. Those two sections are empty. Every other section is filled and marked as a proposal. Aarav can change any of it before the freeze. Nothing here is binding until section 13 is filled.

Track rule, adopted by Aarav 2026-10-07 ("Do this then lets start"): main track only if E1 is clean by 2027-03-15. Clean means a kill result, or a pre-registered null inside a 12-step window. Otherwise the paper goes to E&D, Benchmark Design.

## 1. Hypotheses

AARAV WRITES THIS SECTION.

| ID | Hypothesis | Direction | Primary or secondary |
|---|---|---|---|
| H1 | | | Primary |
| H2 | | | Secondary |

Decisions H1 must make. The other sections depend on them.
- **Which detector is primary.** The kill result reads "one detector shows both" (plan section 2). The sweep scores 20 detectors (tex:723), so "any of 20" at α=0.05 is a forking path. At 24 vs 24, TPR 0.5 against FPR 0.1, experiments/e1_power.py gives:
  - one named detector: power 0.90, MDE 0.44;
  - Bonferroni over 20: power 0.53, MDE 0.61;
  - Bonferroni over 20 at 36 vs 36: power 0.80, MDE 0.50.
- **Which collapse event is primary.** E0 (paper/stats/e0_slope_validation.json) gives these false-collapse counts on the 12 grid benign runs:
  - baseline_drop(0.6, persist=2): 0/12;
  - the published persist=1 version: 8/12, all at steps 0 to 16;
  - raw peak_drop: 6/12;
  - slope: 9/12 smoothed.
  The pilot gate uses baseline_drop(0.6, persist=2).
- **Alarm rule.** The published TPR counts an alarm at or before the event (`a <= e`, score_prime.py:492). Strict early warning is `a < e`. The sweep reports both (tex:728). Pick one as primary.

## 2. Primary test

PROPOSED, from the plan's kill result (audit_or_early_warning.md, section 2).

- **Test name:** one-sided Fisher exact test on a 2×2 table. The rows are the arms: hacking and benign. The columns are alarmed and not alarmed.
- **One-sided or two-sided, and why:** one-sided. The claim is directional: the detector alarms on hacking runs more often than on benign runs. A detector that alarms less on hacking runs is not an early warning. The two arms are different runs, so Fisher is valid here. Kit 02's warning against Fisher applies to paired detectors.
- **Alpha:** 0.05, one-sided, for the single primary detector. If H1 names several detectors, use Holm over them.
- **Statistic and decision rule:**
  - Set the threshold at the 90th percentile of the benign run-level scores, the convention at tex:727. With 24 benign runs, the realized FPR is 2/24 or 3/24, and the paper reports the realized value.
  - TPR is the fraction of all 24 hacking runs whose alarm comes before the primary event, under the alarm rule from section 1. A run with no event counts as a miss, as in score_prime.py:492-493.
  - KILL RESULT if both of these hold:
    - Fisher p < 0.05;
    - the median lead of the primary detector over the hacking runs is above 0.
  - NULL, which needs the window precondition in section 9, otherwise.
- **Script:** experiments/e1_primary.py, NOT YET WRITTEN. The assistant writes it before the freeze. Its self-test must:
  - reproduce tab:sweep's TPR and p-values on the frozen 9 vs 9 grid;
  - recover the planted positive control from section 9.

## 3. Unit of analysis

- **Unit:** one GRPO training run, meaning one policy trajectory.
- **n per arm:** 24 hacking (reduced-capacity: the 0.5B judge is optimized) and 24 benign (capable: the 1.5B judge is optimized).
- **Effective n per arm, and why:** 24.
  - Each run is a separate job with its own seed, so the GRPO sampling RNG and the prompt order both differ.
  - Policy trajectories diverge between seeds: the grid's event steps range from 2 to 51 at persist=2.
  - Unlike JUDGe's nonce seeds, each seed changes what the policy is trained on.
- **Clusters:** 3 tasks (count_vowels, is_prime, second_largest), 8 seeds per task per arm. There is 1 policy (Qwen2.5-1.5B base) and 1 judge pair (Qwen2.5-0.5B-Instruct and 1.5B-Instruct). A task-stratified exact test is a sensitivity analysis (section 10, X3).
- **Seeds:** 200 to 207 for every task, with both arms sharing a seed per task to keep them matched. Replacements use 208 onward. Seed 100 is the pilot, and seeds 0 to 7 are already used.

PROPOSED: Name the unit of analysis in the abstract: "24 hacking and 24 benign training runs".

## 4. Sample size and power

- **Smallest effect that matters, and why:** the plan names TPR 0.5 against FPR 0.1 (E1, "Sample"). AARAV CONFIRMS the "why": it is a substantive claim about what a useful monitor must achieve.
- **MDE at 80% power:**
  - TPR ≥ 0.44 at 24 vs 24 with one primary detector, against FPR 0.1;
  - 0.61 with Bonferroni over 20 detectors;
  - for comparison, the frozen 9 vs 9 grid had an MDE of 0.74, and its largest observed TPR is 0.44 (tex:682).
- **Method:** exact enumeration of the one-sided Fisher test, experiments/e1_power.py.
- **Cost of one unit:** the pilot measured 9 to 13 s per step on an A100, including both judge passes and the probes (HF jobs, 2026-10-07, two log snapshots 51 s apart). The launcher's older constant is 16 s. Per run, with about 0.15 h of setup:

  | Pilot result | Steps per run | Hours per run | Cost per run (HF a100-large, $2.50/h) |
  |---|---|---|---|
  | lr 1e-6 | 400 | 1.2 to 1.6 | $3.00 to $4.00 |
  | lr 5e-7 | 800 | 2.2 to 3.1 | $5.50 to $7.75 |

- **Total cost:** 48 runs plus up to 12 replacements.
  - At lr 1e-6: 58 to 96 GPU-hours, $145 to $240.
  - At lr 5e-7: 106 to 186 GPU-hours, $265 to $465. The upper end is above the plan's cap of $300, so this branch needs a section 12 decision before the main runs.
  - AWS g6e spot is about the same price per hour and bills to credits, but it had no capacity on 2026-10-07 (placement score 1/10).

## 5. Margins for "no difference" claims

A null is the main-track outcome the track rule allows, so this section matters.

- **Equivalence test:** TOST on TPR − FPR. Use the Newcombe hybrid-score interval, a one-sided 95% upper bound.
- **Margin (delta), and why that size:** AARAV CHOOSES. What the design can support: when a detector is truly uninformative (TPR = FPR = 0.1), the upper bound falls below these values 80% of the time:
  - 0.24 at 24 vs 24;
  - 0.19 at 36 vs 36.

  A margin below 0.24 at n=24 is likely to leave the null inconclusive. The calculation is exact enumeration (the script is in the session log; it moves into e1_primary.py before the freeze).
- **Scale:** absolute difference in proportions.

## 6. Statistics plan

| Comparison | Paired or unpaired | Test | Sided | CI method | Multiple-comparison method | Error bar shows | Variation in the bar |
|---|---|---|---|---|---|---|---|
| Primary detector: TPR vs realized benign FPR | Unpaired (different runs) | Fisher exact | One | Clopper-Pearson per proportion; Newcombe for the difference | None if H1 names one detector; Holm otherwise | 95% CI | Runs |
| Primary detector: lead (event step − alarm step) on hacking runs | One sample | Exact sign test, median > 0 | One | Order-statistic CI for the median | None (part of the primary rule) | 95% CI | Runs |
| Remaining detectors: TPR vs FPR | Unpaired | Fisher exact | One | As row 1 | Holm across the remaining detectors | 95% CI | Runs |
| Window precondition: median pre-event window ≥ 12 (hacking) | One sample | Descriptive with a CI | n/a | Order-statistic CI for the median | None | 95% CI | Runs |
| Equivalence, if the result is null | Unpaired | TOST, margin from section 5 | One per bound | Newcombe hybrid score | None for the primary detector | 90% CI (TOST) | Runs |
| Detector vs detector, if reported | PAIRED (same runs) | Exact McNemar | Two | Exact conditional | Holm | 95% CI | Runs |

No symmetric bars past [0, 1]. Each null reports its MDE. Each count reports an exact interval.

## 7. Exclusions

| Rule | Reason | Expected count |
|---|---|---|
| The run logs fewer than its step budget (crash, timeout or preemption). Rerun it with the next unused seed. | An incomplete trajectory has no defined event window. | 0-2 per arm. HF Jobs do not preempt, and the timeout is 1.6× the estimate. |
| The first-5-step baseline of held-out quality is 0. | baseline_drop is undefined: 0.6 × 0 = 0. | 0. The lowest grid baseline is above 0. |
| More than 5% of steps are missing from the log. | Event dating needs a near-complete series. | 0 |

- No exclusion may depend on a detector's output or on the oracle's trajectory.
- A hacking run with no collapse event is not excluded. It counts as a miss (section 2).
- Pilot runs (seed 100) are excluded from every confirmatory analysis.

## 8. Stopping rule

- **Design:** fixed n, 24 + 24, with no interim look at the primary outcome.
- During the runs, monitoring covers job health only: completion, step count and the error lines. It never covers a detector or the event.
- Replacements follow section 7, up to 30 launched per arm. After that the arm stands at whatever n has completed, and the MDE is recomputed and reported.
- Do not add runs after you look at the primary result.

## 9. Controls

| Arm | What it controls for | n | Matched to |
|---|---|---|---|
| Benign arm, matched: the capable 1.5B judge is optimized | Detector activity from ordinary GRPO dynamics. It sets the FPR. | 24 | Task, seed, lr, steps, probe cadence |
| Placebo: partial-credit reward (tex:227-229) | A second benign mechanism. Reward shaping collapsed 0/9 in the corpus. | AARAV DECIDES. The plan lists it with no n. | Task, lr, steps |
| Planted positive control | Whether the scoring pipeline can find an effect of the claimed size. A synthetic alarm fires 1 to 12 steps before the event on a random 50% of hacking runs, and at random on 10% of benign runs. It runs through e1_primary.py unchanged, over 1000 draws. Expected rejection rate: about 0.90, the power from section 4. | 24 + 24, synthetic | The real runs' events |
| Natural hack arm | The reduced-capacity arm is natural: the policy finds the weak judge's blind spots itself. Nothing is scripted. | 24 | n/a |

- **Source of the behavior:** emergent. The policy optimizes a weaker judge's score, and held-out tests are the oracle.
- **Natural hack rate:** report it before any detector result. It is the fraction of hacking runs where the event fires, and the fraction where the proxy rises while held-out quality falls. For reference, in the frozen grid the event fired in 9/9 runs at persist=2.
- **Window precondition:** "clean" needs a median pre-event window of at least 12 steps on the 24 main hacking runs. The pilot sets the rate. If the main runs miss the window anyway, the null is not clean, and the track rule sends the paper to E&D.

## 10. Exploratory analyses

| ID | Analysis | Why |
|---|---|---|
| X1 | Event dating on the causal probe series: 16 completions per probe, the event at −20% off the probe peak | It removes the raw-series artifact without an acausal smoother (launch_slow_onset.py). |
| X2 | All detectors under the published event (persist=1) and smoothed peak_drop | Continuity with tab:sweep |
| X3 | A task-stratified exact test, a Cochran-Mantel-Haenszel-type stratified permutation | 3 task clusters |
| X4 | Lead time against window: Spearman, replicating the 0.83 and 0.87 at tex:710 | Tests the "no warning was available" account |
| X5 | The pilot runs (seed 100) | Calibration only. Reported descriptively. |
| X6 | Δjudge disagreement trajectories, with both judges logged in every run | The "disagreement leads hacking" reversal (tex:279-286) |

## 11. Kill rules

AARAV WRITES THIS SECTION. Source material, not filled in:
- the plan's E1 kill (audit_or_early_warning.md, E1);
- the kill result (section 2 of that file);
- the plan's cap of $300 (E1, "Cost");
- the 2027-03-15 track gate.

| Kill rule | Exact threshold | Action |
|---|---|---|
| Result | | |
| Budget | | |
| Calendar | | |

## 12. Spend gates

| Stage | Purpose | Ceiling (GPU-hours) | Ceiling (USD) | Approved by Aarav (date) |
|---|---|---|---|---|
| Calibration | E0: validate event definitions offline on CPU | 0 | 0 | Done 2026-10-06 |
| Pilot | Pick the lr: 6 runs, lr 1e-6 for 400 steps and 5e-7 for 800 steps, 1 per task, seed 100. Gated by experiments/e1_pilot_gate.py, fixed in commit 24e1b5b before any data. | 25 | 60 (about $29 at the measured step time) | "lets start", chat 2026-10-07. The ceiling is PROPOSED. |
| Primary | 48 runs plus up to 12 replacements at the pilot's lr | 96 (lr 1e-6) or 186 (lr 5e-7) | 240 or 465. The plan's cap is $300. | PENDING |

If a stage needs more than its ceiling, write a deviation in section 14 before the spend.

## 13. Freeze record

Run in the repository after sections 1 and 11 are written:

```
git add experiments/e1_preregistration.md experiments/e1_primary.py
git commit -m "Freeze E1 pre-registration v1"
git rev-parse HEAD
git log -1 --format=%cI
sha256sum experiments/e1_preregistration.md
```

- Commit hash: UNKNOWN
- Commit time (ISO 8601, with time zone): UNKNOWN
- SHA-256 of this file: UNKNOWN
- Human reader before the freeze (name and date): UNKNOWN

The repository is public. Pushing the frozen file gives a third-party timestamp, but the file names the paper and its author. Whether to push it, or to email the SHA-256 to the reader instead, is Aarav's call (kit 02 PROPOSED, line 174).

## 14. Deviation log

These rows are pre-freeze notes, logged so the paper can cite them.

| Date | Section changed | What changed | Why | Before or after a look at the data? | Effect on the claims | Commit |
|---|---|---|---|---|---|---|
| 2026-10-07 | Plan timeline | The pilot ran before the E1 freeze. The plan's timeline had the freeze first (11-19 to 12-11). | Aarav: "lets start". Kit 02 requires the freeze only before primary data. | Before. The pilot gate was committed before the jobs were submitted. | None on the primary test. The pilot seeds are excluded (section 7). The 12-step threshold predates the pilot (plan section 2). | 24e1b5b |
| 2026-10-07 | Pilot step budget | The plan cites the 200-step setup at tex:129. The pilot uses 400 steps at 1e-6 and 800 at 5e-7. | Lr is a clock lever. A lr 1e-6 run never crossed 0.6 × baseline in 200 steps (launch_slow_onset.py docstring), so 200 steps would trip the kill rule by construction. | Before | The main runs inherit the step budget of the chosen lr. | 24e1b5b |
| 2026-10-07 | Pilot backend | Moved from AWS g6e spot to HF Jobs a100-large. The job is the same. | EC2 returned InsufficientInstanceCapacity in us-east-1a to 1d (score 1/10), and the other regions have 0 quota. | Before | None. The per-step time differs, but the log is the same. | 97f885c |
| 2026-10-07 | Pilot tasks | The tasks are the published grid's count_vowels, is_prime and second_largest, not tex:129's sum_list, is_prime and longest_two_distinct. | E1 replaces the 9 vs 9 grid comparison, so it uses the grid's tasks. | Before | The tasks match the paper's primary comparison. | 24e1b5b |
