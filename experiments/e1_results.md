# E1 results (2026-10-08)

The registered result comes first, as e1_primary.py prints it ("registered first, also if null"). Sections 9 and 11 of experiments/e1_preregistration.md require only that the natural hack rate and the generator check come before any detector result (:239, :267). Everything after the registered result is labeled as registered reporting or exploratory.

## Registered result

**CLEAN NULL (non-superiority shown: no anticipation beyond TPR - FPR = 0.25).**

Source: `python experiments/e1_primary.py`, run unmodified on 2026-10-08 at 14:45 EDT. Output: experiments/e1_primary_report.txt and e1_primary_results.json.

| Step | Value | Rule | Outcome |
|---|---|---|---|
| Provenance | 14 pinned files match e1_pins.json | must match | pass |
| Runs used | 24 hacking and 24 benign; 0 excluded, 0 unregistered, 0 surplus | at least 16 per arm | pass |
| Primary signal present | 0 runs missing disagreement | none missing | pass |
| Natural hack rate | Event in 23/24 hacking runs. Proxy (0.5B judge) up and strict oracle down in 24/24 | generator stop below ceil(0.75 x 24) = 18 events | no stop |
| Window precondition | Median pre-event window 32.0 steps (no-event run counted as -1) | at least 12 | holds |
| Primary: disagreement (level) | Strict TPR 2/24 = 0.083, CP 95% (0.010, 0.270). Realized FPR 3/24 = 0.125, CP 95% (0.027, 0.324) | kill needs one-sided Fisher p < 0.05, a point median lead > 0, and the window | p = 0.8262: no kill |
| Lead | Median 12.0 steps over 23 runs, 95% CI (-48, 33); sign test 14/23, p = 0.20 | the point median > 0 is a kill conjunct; the CI and sign test are reported only | met (12 > 0); no kill, because p fails |
| Non-superiority | TPR - FPR Newcombe 90% CI (-0.201, 0.116); 95% CI (-0.236, 0.152) | upper 90% bound < 0.25 | shown |
| MDE | TPR 0.44 at 80% power (24 vs 24, FP fixed at 3) | reported with every null | |
| Planted control | Rejection rate 0.576 against an exact expectation of 0.567 | within 0.05 | passes, so the verdict is not UNVERIFIED |
| H2 (25 secondary detectors) | Smallest p 0.2307 (velocity of disagreement, 6/24 vs 3/24); every Holm-adjusted p is 1.00 | Holm p < 0.05, lead > 0 and window | none holds |

Track: under the adopted rule ("clean" means a kill result, or a pre-registered null inside a 12-step window), this result gives the **main track**. Aarav confirmed this on 2026-10-08 (section 14, Track row).

## Independent verification (workflow wf_d0eec84d-a77)

- One rebuild from the pre-registration text and the raw run files used no e1_primary.py code; it called only e1_event.event. It reproduced every number above.
- A second spot-check with its own code reproduced the primary TP/FP, Fisher p, events, window and Newcombe interval.
- Two further checks re-ran the frozen e1_primary.py read-only and matched it.
- No code-vs-text mismatch was found.
- Data integrity holds for all 48 runs:
  - registered config;
  - 400 steps each, indexed 0-399;
  - each result.json is byte-identical to the copy embedded in its job log;
  - matched seeding: identical step-0 completions across arms;
  - no Tracebacks or OOM errors.
- Evidence and a SHA-256 manifest of all data are kept locally, under the gitignored paper/freeze/e1_main_20261008/.

Monitoring could not be checked by that workflow. Section 14, monitoring row: after the one boot-time line already logged, monitoring printed only instance states, S3 status lines and step numbers. A transcript search finds no run of `launch_e1_aws.py main --status`.

## Readings that qualify the result

Sources: experiments/e1_supplementary_report.txt and the verification workflow. None of these changes the registered verdict. Each must appear in the paper.

1. **Registered reporting (supplementary B): "clean" depends on the adopted event.**
   - Under cusum_page_down_v2, the rule-selected event and a registered sensitivity analysis, the median window is 9.0. The window precondition would then fail, and decide() would read NULL, NOT CLEAN. This is illustrative only: no decision rule is registered for sensitivity events.
   - sustained_drop_v2 was adopted as a logged deviation from the pre-fixed selection rule. That came after the candidates were scored on corpus runs and before any main data (section 14, event-adoption row).
   - Under baseline_drop(0.6, persist=2), the window is 24.5, with 4/24 benign false events.
2. **Registered reporting (supplementary C): the collapse is strict-oracle only (option (a)).**
   - On the format-tolerant oracle, the registered event fires in 0/24 hacking runs, against 23/24 on the strict oracle.
   - Proxy-up with oracle-down holds in 2/24 on lenient, against 24/24 on strict.
   - The hacking arm's lenient score rises (+0.174 over the run) and falls in only 2/24 runs, while its strict score falls (-0.398).
   - This is format, length and spec failure, not a decline in function correctness. The lenient score is only a lower bound on a correct function being present.
3. **Exploratory (supplementary D2, D3): non-superiority is fragile.**
   - An upward-only alarm gives 6/24 vs 3/24, with an upper 90% bound of 0.306.
   - A time-matched operating point, which counts benign alarms only over the matched hacking run's pre-event steps, gives 6/24 vs 3/23, with an upper bound of 0.302.
   - Both exceed 0.25. Neither is significant (p 0.23 and 0.25).
4. **Exploratory (verification skeptic, not in the supplementary): the benign arm is not a no-signal control for disagreement.** It optimizes the 1.5B judge, so its disagreement drifts negative: the last-25 minus first-25 change is positive in 0/24 benign runs. The registered alarm is two-sided, so that drift sets the threshold.
5. **Exploratory (supplementary D1): anticipation is nearly impossible on is_prime.**
   - is_prime's own window is 9 steps. Its strict score falls from step 0, and two of its events sit at step 5, the earliest date the event can return.
   - Per-task windows are 39.0 (count_vowels), 9.0 (is_prime) and 65.5 (second_largest).
   - Both strict hits are on count_vowels.
6. **Exploratory (supplementary D4): disagreement lags.** Its alarm comes a median 83 steps after the event. Of the 15 runs that have both an alarm and an event, it comes before the event in only 2.
7. **Registered (e1_primary_report.txt prints the 0.567 expectation; supplementary D5 restates it; section 14, wording row): the planted control has power 0.567 at the realized events.** Section 9's "about 0.92" holds only when every event is at step 18 or later, and 9 of the 23 events are earlier.

## Exploratory analyses X1-X7 (pre-registration section 10)

Source: experiments/e1_exploratory_report.txt (experiments/e1_exploratory.py). Every result here is exploratory, and none changes the registered verdict. Choices made after the data are printed in the report as "CHOICE" lines.

- **X1, the probe-series event** (the first probe at or below 0.8 times the probe peak; the job's stored causal_window):
  - It fires in 24/24 hacking runs and 2/24 benign runs, with a median window of 27.5 steps.
  - The primary result is unchanged: 2/24 vs 3/24, p 0.826.
  - The probe dates barely agree with the registered dates (Spearman 0.30).
  - An online, running-peak version of the rule fires on 20/24 benign runs.
- **X2, the published events.**
  - Under baseline_drop(0.6, persist=1), the event fires on 23/24 benign runs, the window is 3.5 steps, and no detector has a strict hit.
  - Under the smoothed peak_drop (w=15, 20%; acausal, a label only), the window is 25.5 steps with 0/24 benign events, and the primary is 2/24 vs 3/24.
  - Across all 7 event definitions and 26 detectors, no one-sided p is below 0.05; the smallest is 0.2307.
  - The published first-move event (Definition A) is two-sided, so 5/14 of its hacking "events" and 16/19 of its benign ones are upward moves.
- **X3, the task-stratified exact test.**
  - p 0.821 (pooled 0.826); Mantel-Haenszel odds ratio 0.67.
  - Both hits are on count_vowels (2/8 vs 0/8).
  - No detector has a stratified p below 0.05.
- **X4, lead against window.**
  - The tex:710 correlation recurs: Spearman 0.852 (n 23, 95% CI 0.64-0.94). Like-for-like under the paper's Definition B events, it is 0.893 (raw) and 0.824 (smoothed).
  - The correlation is largely mechanical. The best detector first-moves at a near-constant early step (median 8), and the correlation recurs even under the persist=1 event, whose window is 3.5 steps (rho 0.803).
  - The registered +12-step first-move lead comes from a disagreement first move that also fires in 18/24 benign runs, at a similar step (median 20.5, against 24 in hacking runs).
  - A short window does not explain the null on count_vowels (median 39) or second_largest (65.5). It may on is_prime (9).
- **X5, the pilot.**
  - The registered event fires in 5/6 runs, and its windows match the pre-freeze record.
  - The primary alarms before the event in 0/6.
- **X6, the two judges.**
  - In the hacking arm, disagreement widens because the 0.5B judge rises (+0.192, rising in 24/24). That judge is that arm's reward, so this is partly by construction.
  - The held-out 1.5B judge barely moves (+0.036, falling in only 5/24).
  - Compared with the strict oracle, disagreement first-moves before it in 7/13 dated runs and after it in 6/13; the oracle cannot be dated in 10/23. Neither the paper's reversal nor its opposite is reproduced.
- **X7, the collapse mechanism.**
  - The mechanism varies by seed within a task. 6/24 hacking runs are truncation-driven: count_vowels 3/8, second_largest 3/8, is_prime 0/8.
  - On those 6 runs the truncation rate has already risen by the event.
  - No length or truncation detector shows early warning in either group.
  - The primary's two strict hits are the two latest-event runs. Both are truncation-driven count_vowels runs, consistent with an alarm that comes a median 83 steps after the event.

### Paper text that E1 contradicts (pre-E1 text in paper/neurips_paper.tex)

- **fig:dis caption (tex:262-264):** disagreement widens because the held-out 1.5B judge falls. In E1, the 0.5B judge rises in 24/24 runs, and the 1.5B judge falls in only 5/24 (X6).
- **tex:281-282:** disagreement first-moves after the collapse in every seed, or never. In E1, it moves first in 7 of 13 dated runs (X6).
- **tex:712:** no detector leads under any event definition tested. In E1, 13/26 detectors have a positive median first-move lead under the registered and the smoothed events (X2). None is significant at the operating point.
- **Pre-registration :41-45:** second_largest, and count_vowels at 256 tokens, collapse through truncation as whole tasks. In E1, 3/8 runs of each do (X7; section 14).

## Audit-baseline check (2026-10-08; experiments/audit_baselines_report.txt)

This checks the paper's audit headline ("disagreement separates hacking from benign runs at AUROC 0.98", the grid, 9 v 9). It is not part of E1's registered analysis, and every E1 line in it is exploratory.

- **Reproduction.** The published audit table reproduces exactly; disagreement's AUROC is 79/81. An independent checker recomputed the key numbers with its own code.
- **Single judges do as well.**
  - In the grid every hacking run trains on the 0.5B judge and every benign run on the 1.5B judge. The 0.5B judge's score alone, scored the same way, gives AUROC 1.00.
  - The training reward alone gives 0.85, and 1.00 with the signed score.
  - Across all 69 logged runs that train on the 0.5B judge, every run is labeled hacking and every one collapsed, and no benign run trains on it. In this data, no audit number can tell a hacking detector apart from a detector of which judge the run trained on.
- **E1 (exploratory) has the same confound.** Disagreement gives 0.90; the 0.5B judge alone gives 1.00.
- **Most of the separation is step-to-step noise** (exploratory; windows chosen after seeing the data). Smoothing disagreement with a causal 10-step moving mean before scoring lowers the AUROC to 0.64 on the grid and 0.52 in E1.
- **The held-out 1.5B judge does not fall in hacking runs:** +0.022 on the grid (falling in 4/9) and +0.036 in E1 (5/24). At the end of training, the gap between the training reward and the other judge is larger in benign runs (AUROC 0.25 on the grid, 0.14 in E1).
- **Contrasts where neither arm trains on a judge** (heuristic-proxy against partial-credit or verifier-coverage, 9 v 9):
  - disagreement gives 0.88 and 0.85; the 0.5B judge alone gives 0.75 and 0.77, and the intervals overlap;
  - only 4 of the 9 heuristic-proxy runs lost strict accuracy.

**Third-judge rescoring (2026-10-09; experiments/e1_rescore/rescore_report.txt).** All 153,600 stored E1 completions were rescored with two judges that neither arm trained on. The readings and criteria were fixed in experiments/e1_rescore/PLAN.md, committed before any score existed (08ac3e4). The data had already been seen, so this is exploratory. Re-scoring with the original 0.5B judge reproduced its logged scores exactly, and an independent recompute matched every point estimate.

- **OLMo-2-7B-Instruct (primary): NOT INTERPRETED.** It failed the pre-set sanity gate: it ranks strict passes above fails at only 0.59, against a required 0.60. Its score rose in every run of both arms (+0.24 and +0.26).
- **Qwen2.5-7B-Instruct (secondary): INCONCLUSIVE.**
  - It passed the gate at 0.83.
  - It did not see the hacking arm's quality fall: +0.05, with only 10 of 24 runs falling. It saw the benign arm improve: +0.29.
  - Its change separates the arms at 0.81, but because the benign runs improved more, not because the hacking runs declined.
  - Training reward minus this judge did not beat the training reward alone (paired difference -0.06, 95% CI -0.23 to +0.11).
  - It is the same model family as the training judges.
- **What this allows.** No held-out judge tested here confirms that the hacking runs lost quality. That fits the lenient oracle, which rises in those runs, and it means no "held-out evaluator audits this hacking" claim is supported. The result is not a KILL either: the primary judge is uninterpretable, and the secondary misses KILL by 2 runs.

**Consequence.** On the current data the paper cannot claim that disagreement audits hacking. Suggested wording is in paper/drafts/audit_claim_reading_20261008.md (local). Two experiments would make the claim testable:
- **E1b:** both arms train on the 0.5B judge. The twin's reward is format-gated, and the comparators are registered.
- **Third judge:** rescore E1's stored completions with a judge neither arm trained on.

## What the paper may and may not claim

May claim:
- the registered primary result, the clean null, and the non-superiority qualifier as registered;
- the natural hack rate, described as format, length and spec failure, with the lenient oracle alongside;
- that none of the 26 scored detectors showed significant early warning in E1, stating that a single H2 detector at the first Holm step had power 0.27 at TPR 0.5 (MDE 0.65, or 0.53 and 0.57 for the two truncation detectors, whose realized FP was below 3/24);
- scope: Qwen2.5-1.5B base, the Qwen2.5-0.5B/1.5B-Instruct judge pair, GRPO at lr 1e-6 for 400 steps, a 256-token cap, three tasks, one event and one operating point.

May not claim:
- early warning, or that disagreement's 12-step median lead means it leads the collapse (its CI includes 0);
- that disagreement has zero anticipatory signal: differences in TPR - FPR below the 0.25 margin are not excluded. The MDE, TPR 0.44, is the design's 80%-power point, not a bound on what the data exclude;
- any general claim that no training-time signal anticipates reward hacking;
- non-superiority for any secondary detector;
- that "clean" holds under the registered sensitivity events (under cusum_page_down_v2 the window is 9.0 and fails);
- a clean null per task;
- that the design had about 0.92 power at the realized events (the exact expectation was 0.567);
- wrong-code hacking;
- exact reproducibility from the freeze commit (section 14, job code and environment rows);
- that disagreement first-moves after the collapse (the tex:281-282 reversal; X6), or that it widens because the held-out judge falls (X6);
- that the window-lead correlation shows warning was unavailable: it recurs even with a 3.5-step window and is largely mechanical (X4);
- that disagreement audits hacking: in all current data, which judge a run trains on is collinear with the hacking label, and the 0.5B judge's score alone does as well (audit-baseline check above);
- that every detector is causal or online: the 16 velocity and acceleration detectors look 2 steps ahead (section 14);
- that a whole task collapses through truncation (X7);
- that the pre-registration was externally time-stamped or registered on a public registry (section 14, timestamp row).

## Provenance

- **Analysis:** the frozen e1_primary.py, run by e1_autorun.py from the session scratch directory. An identical copy is at scripts/e1_autorun.py.
- **Code:** the code the jobs executed is the d1aa118 template plus the reward_testhack.py committed with this record. The job package's decompressed-tar SHA-256 is e360abbc...91f0.
- **Environment:** reconstructed in experiments/e1_env_lock_reconstructed.txt. Model revisions are listed in section 14.
- **Hardware:** NVIDIA L40S on four EC2 instances, in us-east-1 and us-east-2.
- **Cost:** about USD 124, estimated from run times and list prices. There is no billing data. All four instances were confirmed not running after the runs (describe-instances, about 14:50 EDT).
