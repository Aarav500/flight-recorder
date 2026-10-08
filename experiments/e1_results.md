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
- that the pre-registration was externally time-stamped or registered on a public registry (section 14, timestamp row).

## Provenance

- **Analysis:** the frozen e1_primary.py, run by e1_autorun.py from the session scratch directory. An identical copy is at scripts/e1_autorun.py.
- **Code:** the code the jobs executed is the d1aa118 template plus the reward_testhack.py committed with this record. The job package's decompressed-tar SHA-256 is e360abbc...91f0.
- **Environment:** reconstructed in experiments/e1_env_lock_reconstructed.txt. Model revisions are listed in section 14.
- **Hardware:** NVIDIA L40S on four EC2 instances, in us-east-1 and us-east-2.
- **Cost:** about USD 124, estimated from run times and list prices. There is no billing data. All four instances were confirmed not running after the runs (describe-instances, about 14:50 EDT).
