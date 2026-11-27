# E1b job smoke test (CPU), 2026-10-23

A design record for scripts/e1b_job.py. It is not a registered analysis, and none of its values feed a decision.

    python experiments/e1b_design/job_smoke/smoke.py WORK

WORK must lie outside the repository. Every environment and output goes there, and the report is `WORK/report.json`. The exit code is 0 only when every check passes. The method is in the docstring of smoke.py. In short, the harness:

- clones the repo and commits the working-tree job files in the clone;
- bakes real judge and judgegate jobs (count_vowels, pilot seed 299) with `bake()`;
- builds a venv with uv 0.12.23, CPython 3.12.15, every locked distribution by hash, and torch 2.11.0+cpu;
- patches only what a CPU run needs and runs both arms for 3 steps;
- records from_pretrained calls, Hub reads, TRL's reward call and the status.json writes without changing them;
- checks the measured provenance fields, matched-seed step-0 identity and the docstring's instance commands.

Patches applied to the baked job:

- Tiny public models at their current Hub revisions:
  - the policy and judge_small are trl-internal-testing/tiny-Qwen2ForCausalLM-2.5;
  - judge_big is trl-internal-testing/tiny-Qwen3ForCausalLM, so the two judge columns differ.
- CPU and float32 (`use_cpu=True`, `bf16=False`).
- 3 steps, with a probe of 2 at every step.
- Scratch paths for the result and the output directory.
- The eos guard in the set form that the logged flag needs. With the tiny policy's two eos ids, the job's own guard stops the run, and the eosguard case shows this.

## Cases

| Case | What |
|---|---|
| main_judge, main_judgegate | both arms with the CPU patches |
| eosmix_judge, eosmix_judgegate | the same plus a sampling bias (+6.0 on ids 151643 and 151645) so that some completions end before the cap; it changes sampling only |
| stop_package, stop_gate, stop_lock, stop_nolock | a tampered embedded package, a tampered twin_gate.py (package_sha256 re-pinned), a changed job.py.lock, no job.py.lock |
| eosguard | the CPU patches without the eos-guard patch |
| stop_snapshot | judge_big loaded at the tiny model's real revision while REVISIONS pins another commit (last hex digit changed); the snapshot check must stop the job before training |
| arch | no job: GRPOTrainer built on the tiny policy with `model_init_kwargs` `{revision}` (the job as first built) and with `{revision, architecture: AutoModelForCausalLM}` (the job now), compared |
| bake_judge, bake_judgegate | no run: bake()'s provenance against the embedded package, the lock, the gate, the working tree and the job.py it wrote |
| matched_main, matched_eosmix | no run: the judge and judgegate results of a kind, which share seed 299 |
| doc | no run: the module docstring's uv install and run commands |

## Result

The last run (2026-10-25) was against scripts/e1b_job.py after the review fixes below (template_sha256 b8fd924529c1301772cb3c677be2dfd8e9db73383296878a83f76c16ba6222d3) and the current smoke.py. All 243 checks passed, and each training case took 61 to 74 s on a laptop CPU. Run 7, before the review, had 222 checks.

- **stdout** was exactly `STEP 0`, `STEP 1`, `STEP 2` in all four training runs, with no other line. stderr held only:
  - Python 3.12's tarfile DeprecationWarning;
  - on a cold Hub cache (run 7), huggingface_hub's unauthenticated-request and Windows-symlink warnings.
  
  No line on either stream carries a metric value.
- **Result file:**
  - It has exactly contract section 4's top-level keys, log keys and completion keys, with the right types.
  - It has 3 log rows and 3 completion entries of 8.
  - Its provenance equals `bake()`'s, which equals the working tree's SHA-256s.
  - Every installed distribution has the lock's version; torch is the CPU build of 2.11.0.
- **Measured provenance:**
  - bake()'s job_sha256 is the SHA-256 of the job.py it wrote.
  - Each run's job_file_sha256 is the SHA-256 of the file it executed. That file is the CPU-patched job, so the value differs from bake()'s job_sha256, as an edited job's would.
  - Each run's snapshots equals the pins, and equals the one snapshot directory the Hub cache holds for each model's repo.
- **TRL's reward call:**
  - It passed completion_ids as a list of 8 lists of int.
  - Texts are TRL's completions in full, and n_tokens = len(ids).
  - eos equals TRL's own clipped rule completion by completion.
  - 1 - mean(eos) equals trl_history's completions/clipped_ratio and log.clipped; mean(n_tokens) equals completions/mean_length.
  - In eosmix, 20 of 24 completions end before the cap (7 to 256 tokens).
- **Rewards and logged values:**
  - reward equals judge_small (judge), or judge_small x G (judgegate), to 1e-6, and TRL got that reward.
  - gate_rate, js_std, jb_std, proxy, adv_std and the other means agree with the completion columns.
  - Logged G equals `experiments/e1b_offline_gate.offline_gate`, run from the analysis Python 3.13 under uv-managed CPython 3.12.15, the runs' logged full version (the gate accepts no other patch release and downloads none).
- **status.json** was written with step null before any model loaded, then once per step. Each write had exactly {run_id, step, rc, ts}, with rc null.
- **from_pretrained:** every call received revision = the pin. This covers both judges' tokenizers and models, the policy, the reference model, the policy's processor, and every nested config, tokenizer and generation-config call.
- **Hub reads:** every file read was at the pin, and the Hub cache holds only the pinned snapshots and no `refs/main`. The one default-branch read is a metadata read: the tokenizer loader's Mistral-tag `model_info` (transformers, tokenization_utils_tokenizers.py), which takes no revision.
- **Stop cases:** each stop case exited 1 with its "E1B STOP before training" line, no STEP line and no result file. The four provenance stops came before any status.json write and any from_pretrained call. The eosguard and stop_snapshot stops came after both judges, the policy and the reference model had loaded.
- **Matched seeds (checked):** in main and in eosmix, the judge and judgegate runs share seed 299 and produced identical step-0 texts, n_tokens and eos. Both judges' scores and G were identical too (recorded, not checked). The check fails if they differ or a result is missing.
- **Docstring:** the uv install it documents is the hash-pinned form, and its run command is RUN_ARGS.

## Job changes made after the code review (2026-10-25)

- **Measured provenance (JOB-1).**
  - bake() returns job_sha256, the SHA-256 of the job.py it wrote.
  - The job hashes its own file at start and writes the value to the result as job_file_sha256.
  - After every model has loaded, the job reads the Hub cache's snapshot directory for each model's repo and writes the commits to the result as snapshots. It stops before training unless each is the pinned revision.
  - The PEP 723 header is byte-identical (`--write-header` output compared before and after), so experiments/e1b_uv.lock still applies. `uv lock --script --check --offline` accepts it for the baked job.
- **Matched seeds (JOB-2):** the step-0 identity is now a check, not an info value.
- **Docstring (JOB-5):** the instance install line is now `uv==0.12.23 --hash=sha256:<UV_WHEEL_SHA256>` in a requirements file, installed with `pip install --only-binary :all: --require-hashes -r`, as the launcher does.

The same smoke.py, run against the job as it was before these changes (template_sha256 8b3bf6ff...), passed 221 of 243 checks. The 22 failures were the new provenance, snapshot, stop_snapshot and docstring checks. A scratch test fed check_matched step-0 completions that differ, and then a missing result. The check failed both times. The old info assignment recorded neither.

## Job change made after run 3

Run 3, on the job as built, passed 217 of 217 checks, but only because one check had an exemption, and it showed what the exemption covered. trl's `create_model_from_path` called `AutoConfig.from_pretrained(model_id)` with no revision, once for the policy and once for the reference model. That is a default-branch read: it left `refs/main` in the Hub cache. Pre-registration section 13 says "revision= to every from_pretrained".

scripts/e1b_job.py now passes `model_init_kwargs={"revision": <policy sha>, "architecture": AutoModelForCausalLM}`. trl then loads both models with `AutoModelForCausalLM.from_pretrained(model_id, revision=...)`, which picks the class from the pinned config.

The arch case shows what this changes on the tiny policy:

- The class, weights, config, generation config and eos ids of the policy and the reference model are the same as with the bare revision.
- The two unrevisioned `AutoConfig` calls are gone.

The pinned Qwen/Qwen2.5-1.5B config names Qwen2ForCausalLM, and main is still that commit. Runs 4 to 7 have no exemption.

## Limits

- Random-weight models write no code, so G was 0 on every completion. The twin product and the offline gate were checked only at G = 0.
- The judges' scores mean nothing.
- The pins equal each tiny model's main, so the cached snapshots cannot tell a pinned read from an unpinned one. The instrumentation can, and so can a `refs/main` left by a read that resolved the branch.
- This was CPU on Windows. No GPU, bf16 autocast, 400-step run or 16-completion probe was run.
