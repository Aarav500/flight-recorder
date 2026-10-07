# Replacement collapse event: selection result (2026-10-07)

- Harness, metrics and rule: experiments/event_harness.py. Fixed in b7474cf; amendment 1, the synthetic dating battery, in 6b481f9. Both predate any outcome scoring.
- Candidates: frozen in 34f5620 before scoring.
- Full numbers: experiments/event_harness_results.json.
- Corpus: 108 series (40 declined, 60 not declined, 8 ambiguous).

## Registered outcome (report this first)

| Candidate | Passes | Null synth | Null perm | Benign fire | Hit (n=40) | Battery score | Stream vs probe (n=8) | Grid window |
|---|---|---|---|---|---|---|---|---|
| ref baseline_drop(0.6, p=2) | no | 1.000 | 0.601 | 0.317 | 1.000 | 59.4 | 61.0 | 11 |
| sustained_drop v1 | no (battery) | 0.007 | 0.018 | 0.000 | 0.975 | 18.6 | 19.5 | 22 |
| cusum_page_down v1 | no (battery) | 0.003 | 0.019 | 0.033 | 1.000 | 16.7 | 9.5 | 34.5 |
| twowin_bonf v1 | no (battery) | 0.003 | 0.003 | 0.000 | 0.975 | 22.2 | 18.5 | 28 |
| **sustained_drop_v2** | **yes** | 0.007 | 0.020 | 0.000 | 0.975 | 3.9 | **17.0** | 29 |
| **cusum_page_down_v2** | **yes, SELECTED** | 0.003 | 0.017 | 0.033 | 1.000 | **3.4** | 43.5 | 10 |
| twowin_bonf_v2 | no (hit) | 0.003 | 0.002 | 0.000 | **0.475** | 3.2 | 39.0 (n=5) | 37 |

The fixed rule ranks first on the battery score, so it selects cusum_page_down_v2. The lead over sustained_drop_v2 is 3.4 against 3.9 steps.

## Why the selection should not be adopted as is (diagnostic, after the outcome)

These checks were run after seeing the outcome. They are a diagnostic, not a new selection rule.

1. **The battery score is in-sample.** The v2 designers chose among their fixes with the battery. twowin_bonf_v2 shows how far the battery overfits: it has the best battery score (3.2), but on real declines its hit rate is 0.475.
2. **cusum_page_down_v2 has a boundary-dating artifact on real runs.** In 4 of the 8 declined runs with probes, its stream date is step 3, the earliest it can return. Its own probe dates for those runs are 20-55 (per-run table in the session log). The 10-step bins of the pass rate place those onsets later:

   | Run | Bins over steps 0-40 | Onset from the bins | cusum_v2 date | sustained_v2 date |
   |---|---|---|---|---|
   | pilot count_vowels lr 1e-6 | 0.47, 0.54, 0.51, 0.40 | ~30 | 3 | 32 |
   | pilot count_vowels lr 5e-7 | 0.52, 0.49, 0.52, 0.47 | ~30-40 | 3 | 11 |
   | pilot is_prime lr 1e-6 | 0.52, 0.46, 0.39, 0.35 | ~10-15 | 3 | 11 |
   | pilot is_prime lr 5e-7 | 0.53, 0.49, 0.40, 0.55 | ~20-40 | 3 | 5 |

   The battery's earliest onset is step 40, so it never tested this. An onset dated too early destroys the pre-event window E1 needs. The grid median window is 10 under cusum_v2 and 29 under sustained_v2.
3. **Both candidates are weak on very early collapses.** The baseline window overlaps the collapse. Both have weak spots outside the battery, found by the re-break skeptics:
   - A sub-threshold first stage before the main collapse makes sustained_v2's dating bimodal; 13-19% of dates fall 100-150 steps late.
   - cusum_v2 dates small early shifts, drifts and recovered dips as the onset, sometimes hundreds of steps early.
   - Both are vague about where smooth, accelerating declines begin.
   - Both count collapse only against the run's starting level, so a collapse from a later peak back toward the start does not fire.

## Recommendation (Aarav decides; pre-registration section 1)

- Adopt **sustained_drop_v2** as the primary event. It is the other candidate that passes every hard constraint, and its stream and probe dates agree best on real runs (17.0 against 43.5).
- Log the choice as a deviation: the rule's first criterion was in-sample, and the selected candidate shows a boundary artifact on real data.
- Report cusum_page_down_v2 and the reference event as sensitivity analyses.
- Any further revision of a candidate now counts as post-outcome, because the candidates have been scored on the real declined runs. It would need fresh probe-equipped runs to validate it.
