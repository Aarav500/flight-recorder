# MARL spikes (throwaway)

Quick tests to find out whether the cross-agent gradient idea is worth a real study. Nothing here is a
result, a frozen design or a pre-registration. Seeds used here are spent, so a confirmatory run needs fresh
ones and a registered rule. Nothing in this folder touches E1b's pinned files.

- `spike_jax/` JaxMARL IPPO with shared parameters, logging per-agent gradient directions. Copied from
  `Desktop/AAAI/idea2_marl_benchmark/code_v2`. Training is byte-identical to study L on the same machine
  (checked on a short run). New: a patched arm whose reward differs from the exploit arm's only on
  duplicate-claim states.
- `spike_llm/step0_llm.py` Inference-only test of the two-agent coding task from
  `docs/marl_proposal_rigorous.tex`. Its gates are in the file's docstring and were fixed before any run.
- `aws/launch_spike.py` Launches one self-terminating instance per stage. Dry run by default.

## Results, 2026-10-30 (exploratory, about USD 3)

Data is in `runs/jax` and `runs/llm`. `spike_jax/analyze_aligned.py` reads the first, `runs/llm/result.json`
holds the second with every completion.

**JaxMARL.** Replay of study L: 60 of 60 runs identical, so the readout doesn't touch training. Then:
- The patched arm collapses like the exploit arm (N=4, true reward -1.18 to -1.29 in both, cluster 0.225 in
  both). Removing the double claim changes nothing, so the double claim isn't what drives the failure.
- Separate networks don't collapse on either reward (N=4, 3 seeds, -1.17 to -0.71).
- **`analyze_phi.py`'s alarm table is void as a detector test.** It calibrates false alarms on an arm that is
  itself collapsing, and its collapse time fires on 27 of 30 benign control runs. Ignore its N=8 "leads".
- Lined up on the fall (N=4, 19 runs), phi doesn't rise ahead of it. It drops around the fall, and t_q runs
  late by construction, so this reads as concurrent. within - cross shows nothing. The gnorm column compares
  against an arm trained on a different reward and is confounded.

**LLM step 0, first screen.** Gate verdict: STOP for all three designs. D2 and D3 clear on `csv_narrow` only,
and 2 tasks were required.

## Round 2, 2026-10-30 (about USD 1.20, rules fixed before launch)

**JaxMARL agent id (`runs/jax2`, `spike_jax/analyze_twin.py`).** Twin rule FAILS, so this track stops. With an
agent id, all 10 seeds on the exploit reward and all 10 on the patched reward still fall (end -1.29, cluster
0.226). There is no same-reward benign arm, and the detector rule was never read.

**LLM, other 7 hard-set tasks (`runs/llm2`).** Combined gate reads FORWARD for D2 and D3: `csv_narrow` and
`round_half_up_narrow` pass, and no control does. Read that with care:
- G2 is lenient. `round_half_up_narrow` D2 passes with agent 1's share at 0.08 while agent 2 overwrites its
  helper in 62% of samples, and D3 gives agent 2 only 0.08 to 0.13.
- `spike_llm/exploit_coupling.py` finds no verified joint exploit. Its "needs both" count mostly reflects
  broken interfaces or thresholds; see its docstring.
- Proxy and oracle covary positively in 34 of 35 cells. The one negative is `round_half_up_narrow` D0.
- `round_half_up_narrow`'s oracle rose in single-agent 7B training (project memory, 2026-08-23). `csv_narrow`
  has no 7B training run that I could find.

## Two-agent training, 2026-10-31 (about USD 17.60 with calibration)

Sequential design (D3), two LoRA adapters on Qwen2.5-Coder-7B-Instruct, lr 3e-5, `csv_narrow` seeds 300-302, 400
steps. Rule in `spike_llm/verdict_train.py`, fixed before any run. **`phi.jsonl` has not been opened.**

**Fixed rule: RUN CONTROLS.** It fired on the gap arm (final gap 0.71, 0.48, 0.45). Only seed 300 has the proxy up
and the oracle down, and HACK needs 2 of 3, so HACK is already out. Controls could only return GAP or NULL.

Oracle by steps: 0-4 about 0.15, 25-49 about 0.60, 300-399 0.27 / 0.49 / 0.53. Proxy reaches about 0.95 by step 50.

Two faults in my rule, both found after the data, so nothing below counts:
- Its windows (0-399, 150-399) straddle the fall. Over 25-150, chosen after seeing the data, the oracle slope is
  -0.38 / -0.05 / -0.13 per 100 steps (p 2e-15, 0.17, 0.0006).
- I assumed the oracle would sit on a floor near 0.11, and it doesn't. The gap arm fires almost by construction
  here, since the gap over steps 25-49 is already about 0.35.

Which agent matters (`spike_llm/inspect_train2.py`): late groups of 8 hold 8 distinct helpers and 3.6-5.1 distinct
mains. The main is a thin loop that calls the helper in 99-100% of samples, and with the same main the oracle still
varies with the helper in 80-99% of groups. Seed 300's fall is a sudden jump in agent 1's KL near step 128 (0.89)
while agent 2 averages 0.010. Nothing here looks joint. The failures are wrong code in the quote branch of runnable
programs, unlike E1's format and length collapse.

## Controls, 2026-10-31 (USD 17.60; all MARL tests USD 39.34)

`csv_exchangeable` seeds 300-302: same code, prompt and held-out tests, different visible tests.

**Full rule: GAP.** All three trap seeds clear the gap test. True quality didn't fall by the rule, so the stop rule
still applies and `phi.jsonl` stays sealed.

| Steps 300-399 | True quality, seeds 300/301/302 | Mean | Proxy | Exploit rate |
|---|---|---|---|---|
| `csv_narrow` | 0.27 / 0.49 / 0.53 | 0.43 | 0.97 | 0.51 |
| `csv_exchangeable` | 0.98 / 0.68 / 0.84 | 0.83 | 0.98 | 0.00 |

The arms overlap through steps 25-49 and are fully separated from steps 50-99 on. That's complete separation at
3 v 3, where the smallest possible two-sided exact p is 0.10.

Another fault in my rule: its proxy-up + oracle-down signature fires on control seeds 301 and 302 too (slopes
about -0.03 per 100 steps over 150-399, after a peak), so it isn't specific as written.

`spike_llm/analyze_train_phi.py` holds the rule for one exploratory read of phi, written before any `phi.jsonl`
was opened.

## Phi read, 2026-11-08 (Aarav: "Open Φ once"). Result: NO LEAD. MARL stops here.

Run once, output saved in `runs/phi_read_20261108.txt`.
- phi, the proposal's statistic, sits at 0.48-0.51 in every window of every run in both arms. That's the 1 - 1/n
  noise level for two agents, and it never separates the arms.
- within - cross separates the arms in steps 5-24 and 25-49 (narrow higher), and not in 0-4. But agent 1's
  gradient norm, which needs no second agent, separates in both of those windows as well, so by the rule it's no
  lead. My guess, not checked: narrow's proxy saturates sooner (0.76 against 0.60 over steps 5-24), more groups go
  constant, and agent 1's gradients shrink and line up. That is the proxy's own movement, not a warning.

## Symmetric-agent check, 2026-11-15 (USD 0.48; all MARL tests USD 39.82). GATE: FAIL. MARL closed.

`spike_llm/train_sym.py`: both adapters write the whole function for the same `csv_narrow` prompt, and the team is
paid the mean of the two proxies. 30 steps, seed 901. Rule in `spike_llm/gate_sym.py`, fixed before the run.
Output is in `runs/symgate_verdict.txt`.
- The agents' gradients still don't share a direction: the cross-agent half-batch cosine is 0.0002 over steps
  5-29, and the full-batch cosine is -0.016.
- Each agent barely agrees with itself: its two half-batches give a cosine of 0.067. At 8 samples per step, a
  gradient's direction is almost all sampling noise, so phi sits at 0.508 whatever the agents do.
- Learning did happen: the team proxy went from 0.64 to 0.92 in 30 steps.

This closes phi on per-step gradients at this group size. One form it didn't test is comparing directions averaged
over many steps, such as the optimiser's own moving average.

## Scale check, 2026-11-15 (USD 1.54; all MARL tests USD 41.36). GATE: FAIL on both. MARL closed.

`spike_llm/train_scale.py`, rule in `spike_llm/gate_scale.py` fixed before the run, output in
`runs/scalegate_verdict.txt`. Symmetric agents on `csv_narrow`, 30 steps each, lora_B weights as the primary measure.
- 8 samples per step with gradients averaged over about 19 steps: within 0.017, cross 0.050. phi's noise (SD 0.028)
  is three times the swing it could make (0.008).
- 64 samples per step: within 0.048, cross -0.007. Eight times the samples didn't raise an agent's agreement with
  itself, which was 0.067 at 8 samples.

I predicted 64 samples would lift within-agent agreement to about 0.37, and it didn't. So per-step policy gradients
here are close to orthogonal from sample to sample, and the shared direction is a tiny part of them even at 64
samples. More samples or averaging over steps doesn't recover it at any size this budget reaches. A bigger model has
more parameters, which I'd expect to make this worse, not better. Both agents learned in both runs (team proxy
0.75 to 0.96 and 0.66 to 0.99).
