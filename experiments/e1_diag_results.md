# E1 format diagnostic: results (2026-10-07)

Six runs of the reduced-capacity arm, each logging a strict and a format-tolerant oracle on every completion (runs/e1/diag/):
- E1 config: lr 1e-6, 400 steps, seed 101, probe on;
- GRID config: lr 2e-6, 200 steps, seed 0, no probe. This is the published grid's config.
- Tasks: count_vowels, is_prime and second_largest, one run each per config.

Provenance:
- rule frozen in b3f6033;
- supplement pre-registered in 1341d7d, before any result was collected;
- data in 4d6c544.

An independent verification ran afterwards (workflow wf_b72c9513-48a). It used 3 lenses with a skeptic each, and all findings below survived. None of this is in the paper yet.

## Registered verdict (report this first)

| Config | Strict event | Lenient event | Frozen verdict |
|---|---|---|---|
| E1 | 3/3 | 0/3 | FORMAT ONLY |
| GRID | 3/3 | 1/3 | MIXED |

The one lenient fire (GRID count_vowels, step 37) has a null false-fire rate of 0.91 under block permutation. The pre-registered supplement marks it uninformative.

## Supplement and verification

- **The strict oracle collapses in every run.** The late-25 minus first-25 change is −0.20 to −0.50, with the CI excluding 0.
- **Function correctness does not decline in any run.** The format-tolerant labels rise in 5 of 6 runs. GRID count_vowels is flat:
  - lenient −0.030 [−0.094, +0.010];
  - function-only +0.034 [−0.057, +0.091].

  An independent extractor that does not import oracle_lenient agrees in sign in 6 of 6 runs, within about 0.06. On extended edge-case suites (16-18 extra cases per task), the fully-correct share of all completions never declines (verification lens "edgecase").
- **A small edge-case drift exists.** Among functions that pass the hidden tests, the share that also pass the extended cases falls a few points in the 3 GRID runs. Only GRID is_prime is significant (p = 0.017), on is_prime(0) and negative inputs. It is minor next to the strict collapse.
- **The level stays below benign.** Late function correctness:
  - count_vowels: about 0.72-0.84;
  - second_largest: about 0.63-0.74;
  - benign arm, strict: 0.97-0.99;
  - is_prime matches benign.

  So the weak judge gives a lower ceiling, not a decline.
- **The 0.5B judge rises in all 6 runs**, by +0.15 to +0.21. The 1.5B judge rises in 4, is flat in E1 count_vowels, and falls slightly in GRID count_vowels.

## Mechanism: two kinds of presentation failure

Both are failures of how the answer is presented, not of the code. The 0.5B judge rewards both.

1. **Unfenced prompt-continuation prose** (both is_prime runs). The answer opens by continuing the prompt's instructions ("Do not ...") and then writes `def` on the same line, with no code fence. extract_code returns the whole text, and it does not parse.
2. **A chatty answer cut off at the 256-token cap** (E1 count_vowels, both second_largest runs, about half of GRID count_vowels).
   - The answer is "Certainly! Below is..." plus an opening ```python fence plus code plus test prints.
   - It hits max_completion_length=256 before the closing fence. The clipped ratio rises from 0.35-0.57 to 0.67-0.995.
   - Mean length rises from 164-207 to 220-255 tokens.
   - Closing the fence rescues most of these answers (for example 351 of 396).

The "prose share" metric in e1_diag_supplement.py counts both kinds together.

## What this means

- Proxy up, true quality down holds only if "true quality" means **a runnable answer that follows the prompt's spec ("Return only the Python code") within the token budget**. It does not hold if true quality means correct code.
- This is reward hacking of the length/verbosity and format kind: a known reward-model bias. It is not the "plausible-but-wrong code that fools a weak judge" the paper describes.
- **The published grid:** the GRID config matches it at step 0, and is_prime and second_largest reproduce its strict trajectory closely. The published completions were never stored, so attributing the published collapse to the same cause is an inference. GRID count_vowels collapsed less (late strict 0.23) than published seed 0 (0.03).
- **Detectors:** the paper's 20 detectors (6 base signals × reference, velocity and acceleration, plus 2 CUSUM) include no completion-length or clipped-ratio signal. The published grid never logged length. In half the runs, length is the variable the collapse runs through. E1 must add length and clipped ratio as baseline detectors, or a reviewer will ask why the negative result omits them.
- **E1 as designed** would measure early warning of a format and length collapse. Aarav must decide the hacking definition (pre-reg section 1) before any main run.

## Token-cap test, 512 tokens (rule fixed in 3ebd273 before launch)

The published GRID config was rerun with max_completion_length 512, seed 0, on AWS EC2 g6e.4xlarge.

- **is_prime (2026-10-07): PROSE PERSISTS, as predicted before launch.**

  | | 256-token cap | 512-token cap |
  |---|---|---|
  | Strict change | −0.269 | −0.269 [−0.373, −0.140] |
  | Lenient change | +0.156 | +0.101 [+0.041, +0.168] |
  | Late clipped ratio | 0.01 | 0.01 |
  | Mean length | 134→180 | 162→241 |

  With twice the token budget the collapse is unchanged. Its mechanism is unfenced prose, not truncation.
- **count_vowels and second_largest: PENDING.** These are the two tasks whose 256-token collapse ran through truncation, so they decide whether that half is a cap artifact.
