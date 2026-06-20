# Oracle-Blind Rollout Geometry Cannot Distinguish Reward Hacking from Benign Convergence

*Draft — negative result, for TMLR / safety-workshop submission. Flight Recorder project, 2026-06.*

## Abstract

A reward hack and a benign capability improvement are, at the level of the rollout distribution,
the **same event**: a policy sharpening onto whatever the reward favors. The only quantity that
distinguishes them — whether the favored mode also has higher *true quality* — is the held-out
oracle, which an **oracle-blind** detector cannot observe by construction, so no oracle-blind
geometric statistic can separate reward hacking from benign convergence *by kind*; we formalize
this as Proposition 1.

We support the claim with three empirical legs. First, in a detector-characterization study,
every candidate geometric axis (convergence sharpness, output degeneracy, advantage-distribution
shape) fires *identically* on a benign control matched on that axis — including `return 0` (a
degenerate hack) versus `return n*(n+1)//2` (a correct one-liner), which are geometrically
indistinguishable. Second, two independent real benign GRPO runs (base `Qwen2.5-1.5B`, a
well-specified coding task on which no hacking is possible) each produced a **false positive** — a
gentle benign convergence at KL ≈ 0.02 latched a detector onset at step 94 at a warmup-normalized
CUSUM of **S ≈ 37.5**, roughly eight times the firing threshold, reproduced across seeds — and an
onset-strength sweep establishes a **crossover** below which any onset is inseparable from benign
convergence for *any* threshold. Third, an elicitation probe found the test-overwrite exploit
**not elicitable within the tested scope** (≤14B, edit-distance and regex tasks, the
test-overwrite affordance): instruction tuning suppresses it (0/192), base 7B models sit at a
≈1% noise floor, and a capable 14B base model solving the tasks at 52–55% still produced **0/512**
(95% Wilson CI [0, 0.74%]), not rising from 7B to 14B.

We conclude that a usable hacking detector must reintroduce some oracle-adjacent signal: the
natural reframing — rollout geometry as an *efficient sampler* that decides when to spend a scarce
oracle query, rather than a classifier — is motivated by these results but left as future work,
because no real hack was elicitable within scope to build it against.

## 1. Introduction

Detecting reward-hacking *onset* during RL post-training would be valuable: an early signal,
computed from quantities the trainer already produces, could prompt intervention before a model
ships. The attractive form is **oracle-blind** — it reads only rollout geometry (cheap, always
available) and never consults a held-out quality measure. The hypothesized signal was an abrupt
geometric shift (rapid KL acceleration, entropy collapse, advantage redistribution) marking the
transition into hacking, preceding the point at which a held-out oracle gap becomes visible.

This paper reports that the oracle-blind version of this monitor cannot function as a *hacking
classifier*, and that the obstruction is structural rather than an artifact of tuning. Section 2
states the argument. Section 3 describes the apparatus we built to test it. Section 4 gives three
empirical legs consistent with the argument at every measured point: a characterization study in
which candidate discriminative axes collapse against matched benign controls (4.1); two real
benign runs that are false positives, with a durable, reproduced operating point (4.2); an onset-
strength crossover that locates the impossibility region quantitatively (4.3); and an elicitation
probe showing the test-overwrite exploit is not elicitable within the tested scope (≤14B,
edit-distance and regex tasks, the test-overwrite affordance) (4.4). Section 5 states limitations
up front. Section 6 gives the implication and the reframing it motivates.

We lead with the result rather than the original detector hope: oracle-blind geometry detects
*that a convergence is underway*, not *that it is a hack*, and we make that boundary precise.

## 2. The structural argument

Let a policy π be optimized against a reward r. Reward hacking and benign capability improvement
are, at the level of the rollout distribution, the **same event**: the policy concentrates
probability mass onto the outputs r rewards most. In both cases KL from the reference policy
grows, output entropy falls as the policy commits to a mode, and the advantage distribution
narrows as group rewards equalize. These are the signatures of *convergence*, and convergence is
what optimization does irrespective of whether the favored mode is desirable.

The *only* property separating a hack from a benign improvement is whether the higher-reward mode
also has higher **true quality**. That property is not a function of r (during hacking r looks
fine — that is what makes it hacking) and not a function of rollout geometry (which measures
*how* the policy moves, not *toward what, judged against ground truth*). It is a function of the
**oracle**: a held-out measure of true quality. An oracle-blind detector cannot observe the
oracle, hence cannot observe the only quantity that separates the regimes.

We now state this formally.

> **Proposition 1 (oracle-blind indistinguishability).** Let a detector be any measurable function
> *f* of rollout-geometry features *R* (KL, entropy, and advantage statistics computed from a run).
> For any hacking run there exists a benign-convergence run with identical *R*, differing only in
> the oracle *O* (true quality), which *f* does not observe. Hence *f*(*R*) is identical on the two
> runs, and no such *f* separates hacking from benign convergence by kind.

*Assumptions.* (A1) The detector is oracle-blind: its feature set is *R*, and *O ∉ R*. (A2) *R* is
a deterministic function of the run's policy outputs and the proxy reward *r* — i.e. *R = g(*outputs*, r)* — none of which involves *O*.

*Proof sketch.* By (A2), two runs with the same policy-output and reward trajectories have the
same *R*; reward hacking is precisely the case where *r* is high while *O* is low, so a hacking run
and a benign-convergence run can be constructed (or, empirically, *occur*) with coincident
policy-output/reward geometry while their oracles diverge — *O* carries information not present in
*r* and hence, by (A2), not present in *R*. Since *f* is a function of *R* alone (A1), *f* takes
the same value on both runs. The quantity that distinguishes them, *O*, is conditionally
independent of *f*'s input given *r* and is unobserved by *f*; therefore the separating information
is structurally unavailable to any oracle-blind *f*. ∎

The proposition subsumes advantage-distribution statistics: although advantage shape can encode "a
subset of samples discovered a higher-reward mode," that event occurs identically in benign
discovery and in hacking, and advantage shape is part of *R* (computed from *r* and oracle-blind
outputs), so it is covered by (A2). The result bounds achievability without claiming geometry is
uninformative — geometry genuinely detects *that a convergence is underway*; Proposition 1 says it
cannot certify the convergence is a hack.

## 3. Apparatus

**Flight Recorder** instruments a GRPO trainer and computes per-step danger metrics from
rollouts. Oracle-blindness is enforced at the type level: the trainer adapter assembles a
`RolloutBatch`, split by extractors into a `RolloutFrame` (the only object a detector receives)
and an `OracleFrame` (consumed only by the offline evaluator). `RolloutFrame` carries no oracle
fields; `Detector.update(frame: RolloutFrame)` is typed to accept only rollout geometry, so
oracle leakage into a detector is a type error.

The detector-visible extractors produce: KL mean/slope/acceleration; entropy mean and collapse
trend; advantage variance, skew, kurtosis, and a Wasserstein "drift" of the advantage
distribution against a rolling baseline; and generation statistics (mean/var generation length,
n-gram diversity, log-probability concentration). A separate, evaluator-only oracle extractor
computes the reward–quality gap and its changepoint.

The default detector, `ContractionTubeDetector`, fits a robust reference tube on a warmup window
over `[kl_accel, entropy_trend, adv_drift]` (per-dimension median and MAD) and declares onset
when the tube is exited (distance in MAD units beyond a radius ρ, exceeding `tube_tau`) **and** a
CUSUM on a geometry signal (default `kl_accel`, normalized by warmup mean/std) exceeds `cusum_h`,
**sustained** for `persistence` steps. Crucially, both gates are **warmup-normalized**: the tube
in MAD units of each run's own first 30 steps, the CUSUM in warmup z-scores. The detector is thus
scale-invariant — absolute KL magnitude is divided out. The pre-registered thresholds
(`tube_tau=1.0`, `cusum_h=5.0`, `rho=3.0`, `warmup=30`, `persistence=5`) were frozen on a
synthetic harness before any real-run data. The synthetic generator produces authored onset
trajectories and hard negatives; because its onsets are authored, it validates plumbing and tunes
false-positive rate but cannot establish that the signal exists in real runs.

## 4. Empirical results

### 4.1 Detector characterization: candidate axes collapse against matched benign controls

Running the real extractors and detector over controlled trajectories, we asked of each candidate
discriminative axis whether it separates a hack-like profile from a benign profile *matched on
that axis*.

*Convergence sharpness.* The tube signals fire on any sufficiently sharp convergence; three
benign profiles (collapse onto a constant, convergence onto a complex solution, convergence via a
high-diversity route) produced indistinguishable onsets. The detector keys on sharpness, not on
what the policy converged to.

*Output degeneracy.* The generation-length / n-gram-diversity / log-probability-concentration
features separate a short output from a long one, but that is "short = suspicious," which a
correct one-liner also satisfies. Against a hard control the degeneracy axis fired **identically**
on `return 0` (a degenerate hack, on a task where 0 is wrong) and `return n*(n+1)//2` (a correct
one-liner): both short, low-entropy, high-concentration.

*Advantage-distribution shape.* A heavy-tailed advantage distribution is detectable but fired
**identically** on a hacking profile (one sample found an exploit) and a benign control (one
sample was benignly lucky), which share the same advantage shape; at GRPO group size 8 the moment
estimators are additionally noisy. The axis detects a discovery event, not a hack.

In each case the apparent discriminative power against a soft control vanishes against a control
matched on the same axis — the empirical shadow of Section 2.

### 4.2 Two real benign runs are false positives, with a durable, reproduced operating point

We ran a genuine non-hacking GRPO run to measure false-positive behavior on real geometry: base
`Qwen2.5-1.5B`, a well-specified coding task (longest substring with at most two distinct
characters; a multi-case correct verifier so no degenerate constant can pass), 120 steps,
learning rate `1e-6` (deliberately gentle), no seeding, no curriculum. The run converged benignly
— mean reward ≈0.19→≈0.9, entropy ≈1.9→0.45, KL ≈0.02 throughout — and **tripped the detector**:
a latched onset at step 94 and an instantaneous trigger on 42/91 (= 0.46) post-warmup steps, at a
warmup-normalized peak CUSUM of **S ≈ 37.8**, roughly eight times the pre-registered `cusum_h=5`.
Benign convergence, normalized to its own quiet early phase, is a *strong* signal.

We then re-ran the identical configuration on an independent seed with the full per-step geometry
persisted to durable storage. The second run reproduces the result exactly — **onset 94, peak
CUSUM S = 37.55** — and now carries **real advantage-drift** (range [0, 1.19]) rather than the
zero placeholder used when the first run's artifact was reconstructed from logs. Replaying the
detector over the persisted real geometry yields the same onset and operating point. The false
positive is therefore (a) reproduced across seeds at a stable operating point and (b) confirmed
not to depend on the advantage-drift signal that was missing from the first analysis.

### 4.3 The crossover: where no threshold separates hacking from a benign run

Recalibration — raising `cusum_h` until the benign run goes quiet — appears to work against an
*authored, strong* synthetic onset (a threshold near `cusum_h ≈ 40` silences the benign run while
still firing on the synthetic onset), but that margin is a function of how strong the synthetic
onset was authored to be, not a property of real hacking. The correct question is the inverse:
how strong must an onset be to remain separable from the benign run? Sweeping onset strength
downward and comparing in the detector's own warmup-normalized CUSUM statistic (scale-fair):

| onset strength s | onset peak CUSUM S | separable from the benign run? |
|---|---|---|
| 0.10 | 231.8 | yes |
| 0.04 | 78.5 | yes |
| **≈ 0.024 (crossover)** | **≈ 37.8** | **boundary** |
| 0.02 | 27.6 | no — indistinguishable |

An onset is separable **iff its peak CUSUM exceeds the benign run's S ≈ 38.** Below that, the
benign run's CUSUM is at least as large, so **no setting of `(tube_tau, cusum_h)` keeps the
benign run quiet while firing on the onset.** The bar for distinguishability is "louder than a
benign capability jump," not "louder than noise" — and Section 2 says a hack, being the same
geometric event as a benign jump, need not be louder than one.

### 4.4 Elicitation probe: the exploit is not elicitable within the tested scope

The strongest objection to a negative result on detection is that we never put a *real* hack in
front of the detector. We therefore probed, by inference only (no training), whether an emergent
test-overwrite hack is even *elicitable*. We sampled completions under a prompt that exposes the
**test-overwrite affordance** — the submission is stated to run in a working directory containing
the grader file, which is shown in the prompt — and AST-scanned each completion for file
manipulation (writing a test file, `write_text`, `os.remove`/`unlink`, `patch`, `monkeypatch`).
We call this rate P2; because GRPO can only reinforce behaviors that are sampled, P2 is an upper
bound on what training could amplify. The picture is monotone and, with the 14B run, statistically
resolved:

| Model | scale / tuning | P2 (exploit rate) | 95% Wilson CI | P1 (reward variance) |
|---|---|---|---|---|
| Qwen2.5-Coder-7B-Instruct | 7B, instruct | **0 / 192 = 0.000** | — | present (std ≈ 0.48) |
| Qwen2.5-Coder-7B (base) | 7B, base, coder | 2 / 192 = 0.010 | [0.003, 0.037] | present |
| Qwen2.5-7B (base) | 7B, base, general | 1 / 192 = 0.005 | [0.001, 0.029] | present |
| CodeLlama-7b-hf (base) | 7B, base, coder | 1 / 192 = 0.005 | [0.001, 0.029] | present |
| deepseek-coder-6.7b-base | 7B, base, coder | 0 / 192 | — | **P1 = 0 (uninformative)** |
| **Qwen2.5-Coder-14B (base)** | **14B, base, coder** | **0 / 512 = 0.000** | **[0.000, 0.0074]** | **strong (52–55% solve)** |

Three readings. (i) **Instruction tuning suppresses the exploit entirely**: the 7B-Instruct model,
which is the only configuration one would deploy off the shelf, produced file manipulation in
exactly **0/192**, while its base counterpart produced a small but nonzero rate — safety tuning
removes the behavior. (ii) **Base models sit at a noise floor of ≈1%** (1–2 in 192), far below any
rate GRPO could amplify into a sustained policy; the one model with P2 = 0/192 among the base set,
deepseek-coder, produced no scorable output (P1 = 0, parse failures under the base-completion
format) and is therefore uninformative. (iii) **Scale does not rescue it**: doubling to 14B base,
with 512 samples for resolution, gives **0/512** with a tight 95% CI of **[0, 0.74%]** — entirely
below a ~5% "trainable" threshold and below even the 7B base point estimate — and crucially this
14B model has *strong* P1 (it solves both tasks at 52–55%), so the zero is a genuine behavioral
fact about a capable model, not a gradient or parse artifact. The capable model simply does not
reach for the exploit.

The composite claim, scoped to what was tested (≤14B, edit-distance and regex tasks, the
test-overwrite affordance): within this scope the exploit is either **tuned away** (instruct) or
**below trainable rates** (base), and it **does not rise with scale from 7B to 14B**. Within the
tested scope, the phenomenon a hacking detector would need to detect is **not elicitable** — which
reinforces the practical force of the structural result: in the configurations we could reach,
there is frequently no real hack for oracle-blind geometry to distinguish from benign convergence
in the first place. (Larger models and other exploit affordances are not covered by this claim;
see Section 5.)

## 5. Limitations

- **The real false positive is n = 2, not n = many.** Two independent seeds of one task/model
  reached the same operating point (onset 94, S ≈ 37.5). This is a reproduced existence proof that
  real benign convergence lands in the impossibility region with a stable operating point; it is
  not a claim that benign convergence universally sits at S ≈ 38 across tasks and models. The
  structural argument (Section 2) is what generalizes; the runs demonstrate the region is reached
  in practice and that the operating point is stable across seeds.
- **The advantage-drift caveat is resolved.** The first analysis reconstructed `adv_drift` as 0
  from logs and argued it immaterial. The persisted second run carries real `adv_drift` (up to
  1.19) and yields the *same* onset and CUSUM operating point, confirming the firing is driven by
  the CUSUM-on-`kl_accel` path and that the previously missing signal does not change the result.
  Whether a different, higher-fidelity advantage-geometry *design* could separate hacking from
  convergence remains open in principle, though Section 2 argues any oracle-blind statistic is
  subject to the same bound.
- **The elicitation finding is bounded to ≤14B, these tasks, and the test-overwrite affordance.**
  The monotone non-elicitability (instruct 0/192; base ≈1%; 14B base 0/512, CI [0, 0.74%]) is
  established only up to 14B, on edit-distance and regex-matching, under a single exploit affordance
  (overwriting a shown grader file). It does **not** show that larger models cannot be induced to
  hack, nor that *other* exploit classes are equally unreachable — in particular, more-gameable
  affordances such as verifier-parsing exploits (outputs crafted to fool a string- or regex-based
  grader rather than overwrite a file) were not exhausted. These are named as **future work**, not
  as gaps in the present claim: within the stated scope the exploit is not elicitable, and scale to
  14B does not change that.
- **Thresholds are pre-registered on synthetic data.** Warmup-normalization removes absolute-scale
  miscalibration as an explanation, but a real-onset reference — which an unelicitable hack denies
  us — would be needed to claim the monitor *could* work after recalibration; the crossover shows
  recalibration cannot rescue the impossibility region regardless.

## 6. Implications

Pure rollout geometry is insufficient for hacking detection, by a structural argument and three
consistent empirical legs. A usable detector must reintroduce an **oracle-adjacent signal** — a
held-out verifier, a cheap proxy of true quality, or a known-benign contrastive reference — to
recover the one quantity geometry cannot see. This reframes the goal from "detect hacking from
geometry" (impossible oracle-blind) to "detect hacking with the **least** oracle signal."

The natural design is **geometry as a sampler, not a classifier**: the contraction-tube CUSUM
becomes a trigger that decides *when* to spend a scarce oracle query, and the oracle resolves
hack-versus-benign at those points. The contribution would be an efficiency frontier — detection
latency versus oracle queries spent, against a query-every-step baseline. We state this as
**motivated but untested**: building and evaluating it requires a real hack to detect, and the
elicitation gate (4.4) closed before we could produce one. Characterizing this geometry-as-sampler
/ oracle-as-verifier trade-off, and the minimal oracle signal it needs, is the natural next
direction once a real, reproducible hack is available.

## Appendix: reproducibility notes

The detector and extractors are in `flightrecorder/`; the multi-axis characterization detector in
`flightrecorder/detector/multiaxis.py`; pre-registered thresholds in `configs/thresholds.yaml`
(frozen pre-data). The benign runs are GRPO over the TRL adapter on base `Qwen2.5-1.5B`; the
second run's full per-step geometry is persisted at the dataset `Aarav500/fr-gentle-artifact`
(`gentle_seed0.jsonl`) and the onset/CUSUM operating point reproduces from it locally with no GPU.
The crossover is a deterministic offline computation over the persisted benign series and a
strength-parameterized onset generator (`flightrecorder/repro/synthetic.py`). The elicitation probe
is inference-only over `Qwen2.5-Coder-7B-Instruct`, `Qwen2.5-Coder-7B`, `Qwen2.5-7B`,
`CodeLlama-7b-hf`, `deepseek-coder-6.7b-base`, and `Qwen2.5-Coder-14B` (instruct via the chat
template, base via raw code completion); AST detection of file-manipulation is in
`flightrecorder/repro/onset_label.py`.

**Cost.** The entire empirical record was established cheaply. The detector-characterization study
and the crossover are GPU-free offline computations. The two benign GRPO runs and the artifact-
persistence run cost a few dollars on a single A100. The full elicitation sweep that established the
negative result — one 7B-Instruct probe, a four-model 7B-base sweep, and the 14B/512-sample
scale-isolation shot, all inference-only on single A100s — totalled **≈ \$2.23** in compute. No
training run was ever required: the elicitation gate (P2) showed that the exploit GRPO would need
to amplify is not present in the sampled output distribution, so the expensive training experiment
was correctly never run.
