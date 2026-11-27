# E1 third-judge rescoring: analysis plan

Written 2026-10-09. No third-judge score exists yet. The analysis is `experiments/e1_rescore/analyze_rescore.py`,
which implements this file.

## Status

E1's data have been seen: the registered analysis (2026-10-08), the exploratory analyses X1-X7, and the
audit-baseline check (`experiments/audit_baselines_report.txt`). So every reading below is EXPLORATORY and none
of it can change E1's registered verdict. What this file does fix, before any score exists, is the readings, the
sign of every score and the criteria. If any of that changes after scores exist, the change goes in the deviation
log at the bottom with a date. The text above the log stays as written.

## Why

In E1 the hacking arm trains on the 0.5B judge and the benign arm trains on the 1.5B judge. The audit-baseline
check found the 0.5B judge alone separates the arms at AUROC 1.00, both under max|dev| and under the signed
change, because it is the hacking arm's reward. The training reward alone gives 0.91 (signed) and 0.80 (max|dev|).
On these runs no audit number can tell a hacking detector from a detector of which judge a run trained on.

A third judge that neither arm trained on removes that confound. What we want to know is whether a judge nobody
optimized sees the hacking arm's quality fall, and whether training reward minus that judge separates the arms
better than the training reward alone does.

## Judges

| role | model | revision (main's commit sha, HF API, 2026-10-09) |
|---|---|---|
| PRIMARY | `allenai/OLMo-2-1124-7B-Instruct` (different family) | `470b1fba1ae01581f270116362ee4aa1b97f4c84` |
| SECONDARY | `Qwen/Qwen2.5-7B-Instruct` (same family, larger) | `a09a35458c702b33eeacc393d103063234e8bc28` |

Both are ungated. Both run in bfloat16. The logged E1 judges ran in float16 (the `JudgeScorer` default).

The judge is `flightrecorder/repro/reward_judge.py` `JudgeScorer`, unchanged: prompt
`_JUDGE_TEMPLATE.format(problem=task.prompt, code=extract_code(completion))` with `extract_code` from
`reward_testhack` and the task prompt from `reward_codeset.CODE_TASKS`; right padding; truncation at 1024 tokens;
score = sigmoid(logsumexp(logits[Yes variants]) - logsumexp(logits[No variants])) at the last real token.

The analysis prints each result file's judge, revision and dtype and flags any that differ from this table.

## Data

The 48 runs `e1_primary.load_main` uses (24 hacking, 24 benign, 0 excluded), 400 steps x 8 stored completions,
153,600 texts. The job stored each text cut at 1,500 characters. 2 of 153,600 hit the cut, both in hacking runs,
while the logged judges saw the full text. Each completion has `hidden` (strict: fraction of hidden tests passed)
and `hidden_lenient`. A strict pass is `hidden == 1.0`. A lenient pass is `hidden_lenient == 1.0`.

## Definitions

- J3(c): the third judge's score of completion c.
- A run's J3 series: at each step, the mean J3 over that step's 8 completions. A step enters only when all 8 are
  scored (see Partial data); on a complete file that is every step. 400 values.
- chg(x): mean of the last 25 steps minus mean of the first 25 (`e1_supplementary.chg`, the same as
  e1_primary.py:500-502).
- maxdev(x): max over steps of `detection_metrics.dev(x)`, the published audit score.
- A run is falling if chg(J3 series) < 0.
- Run-level AUROC: hacking is positive everywhere. The direction of every score is fixed below and is never flipped
  after looking. An AUROC under 0.5 is reported as it comes out. CI = `detection_metrics.auroc_ci` (bootstrap,
  2000, seed 0). p = `detection_metrics.perm_p` (10,000 label shuffles, seed 0), one-sided in the fixed direction.
  Where scores tie across arms the midrank AUROC is printed too.
- Counts get Clopper-Pearson 95% intervals (`e1_primary.clopper_pearson`).
- Completion-level statistics are clustered by run. Their CIs come from a run bootstrap: resample runs within arm,
  2000 draws, seed 0. Completion-level AUROCs are midrank, because bf16 scores tie often.

## Readings

**R0, validation.** The job also re-scores with the 0.5B judge through the same code (as built on 2026-10-09:
float16, steps 0-9 of one hacking and one benign run, and the job aborts before loading the target judge if its own
check fails). For each (run, step) it re-scores, the mean of the re-scores is compared with the logged
`judge_small`, which the analysis reads from `runs/e1/main` itself. Reported: steps compared, max and
mean absolute difference. R0 FLAG if the max absolute per-step difference is above 0.02. Where a logged
per-completion 0.5B score exists (`samples`: completion 0 at every 20th step) that difference is printed too, with
no flag attached. If the job gives only its own summary number, that number is used and labelled job-reported.
A flag, or no R0 at all, doesn't stop the analysis. It is attached to the verdict line.

**R1, judge sanity gate.** Completion-level AUROC of J3 for strict pass against strict fail, pooled over both arms,
with a run-bootstrap CI. The same for lenient pass. Per-arm values are printed without CIs. A judge whose pooled
strict-pass AUROC is below 0.60 is reported but not interpreted. Its verdict reads NOT INTERPRETED.

**R2, J3 change per arm.** chg(J3 series) for each run. Per arm: the mean change (run-bootstrap CI, descriptive),
the number falling out of n with a Clopper-Pearson interval, and the mean J3 level over the first 25 and the last
25 steps. Per task: mean change and number falling, descriptive.

**R3, audit AUROC of J3 alone.** PRIMARY score: -chg(J3 series). The sign is fixed now: J3 is expected to fall in
hacking runs, so a larger score means hacking. Secondary score: maxdev(J3 series). The logged 0.5B judge and
disagreement are printed on the same runs for reference.

**R4, the auditor's gap.** gap = training reward (the logged `proxy`: the 0.5B score in hacking runs, the 1.5B
score in benign runs) minus the J3 series, step by step. Signed score: +chg(gap), sign fixed now (the gap is
expected to rise in hacking runs). Secondary: maxdev(gap). Both are compared with the training reward alone,
+chg(proxy) and maxdev(proxy), computed from `runs/e1/main` (the audit-baseline report has 0.91 and 0.80). The
comparison is a paired bootstrap of AUROC(gap) - AUROC(reward alone): resample runs within arm, 2000 draws,
seed 0, the same draws for both scores, midrank AUROC, percentile 95% CI.

**R5, does J3 track format or function.** Completion level, pooled: Pearson r of J3 with `hidden` and with
`hidden_lenient` (run-bootstrap CIs), with Spearman printed alongside. Two conditional AUROCs. Among lenient
passes, J3 for strict pass against strict fail: does J3 mark down working code whose format broke? Among strict
fails, J3 for lenient pass against lenient fail: does J3 still credit working code whose format broke? Run level:
R3's two J3 scores against the logged 1.5B judge alone, -chg(judge_big) and maxdev(judge_big) (the audit-baseline
report has 0.95 reversed and 0.63), with the same paired bootstrap. No criterion uses R5.

## Criteria

From the interpretation review. Each judge gets a verdict. The PRIMARY judge's verdict is the result and the
SECONDARY's is reported alongside. n is the number of hacking runs with a J3 change (24 when complete). The checks
run in this order: R1 gate, coverage, KILL, LIMITED RESCUE.

- **KILL** for the audit claim. J3 is flat in the hacking arm: |mean change| < 0.02, or falling in at most n/3
  runs (8/24). Then reward minus J3 is just the reward.
- **LIMITED RESCUE.** All three hold. (a) The hacking arm falls: mean change < -0.02 and falling in at least 2n/3
  runs (16/24). (b) The benign arm is flat or rising, which this plan fixes as benign mean change >= -0.02. (c)
  R4's signed paired difference, gap minus reward alone, has a 95% lower bound above 0. Then the supportable claim
  is about a held-out evaluator, not disagreement as such.
- Otherwise **INCONCLUSIVE**, with the numbers.

KILL and LIMITED RESCUE can't both hold. The review said only "flat or rising" for (b); the -0.02 bound on the
benign mean is my reading of it, set to the same 0.02 band the KILL criterion uses, and fixed here before scoring.

## Partial data

The analysis reads `result.json` or `partial.json` and prints how many of the 153,600 completions are scored.
Only fully scored steps (all 8 completions) enter any reading, R1 and R5 included. The job scores each chunk in
batches grouped by task and sorted longest prompt first, so a partly scored step holds only its longest
completions, and length is not neutral in E1. Those completions are counted and left out. A
run enters R2-R4 only if both its first-25 and its last-25 window have at least one fully scored step. With fewer than
16 such runs in either arm (`e1_primary.MIN_ARM`) the verdict is NO VERDICT (coverage). With fewer than 153,600
scored, any verdict is marked PROVISIONAL. The count thresholds stay at n/3 and 2n/3 of the runs that qualify.
maxdev takes its baseline from the first 5 scored steps, so on partial data it is provisional as well.

## What this plan does not do

It registers nothing and corrects for nothing; p values are descriptive. It changes no frozen or pinned E1 file.
A verdict marked with any flag (R0, revision or dtype differing from this plan, data SHA mismatch, partial, input
issues) is not final until the flag is explained in the deviation log.

## Files and how to run

- `prep_data.py` builds `e1_completions.jsonl.gz` and its SHA-256.
- The GPU job scores it with one judge and writes `result.json` (and `partial.json` every ~10 min) to
  `s3://fr-deploy-325527186655/e1/rescore/<olmo2_7b|qwen25_7b>/`.
- `python experiments/e1_rescore/analyze_rescore.py OLMO_RESULT.json QWEN7B_RESULT.json --data-sha e1_completions.sha256`
  writes `rescore_report.txt` and `rescore_results.json` here. Every line is labelled EXPLORATORY.

## Pre-score amendment

2026-10-09, pre-launch review, before any third-judge score existed: Definitions and Partial data now admit only
fully scored steps (above), because the job's length-sorted batching makes a partly scored step a length-biased
sample. A complete result is unaffected.

## Deviation log

- 2026-10-09, after scoring (no change to any reading, sign or criterion). Both jobs finished complete (153,600 of 153,600, mode production, data SHA, revisions and dtype as in this plan; R0 reproduced the logged 0.5B scores exactly). An independent recompute (workflow wf_0eede913-d79) reproduced every point estimate and both verdicts. Two notes. (a) `analyze_rescore.py` rounds to 4 decimals and then prints 3, so four descriptive values in the report are 0.001 low (for example Qwen R2 second_largest +0.086, true +0.0865). No criterion uses them, and the script was not changed after scoring. (b) The data file's gzip SHA-256 reproduces only with a stock-zlib Python (3.13 here). A zlib-ng build gives the same uncompressed content with a different gzip hash.
