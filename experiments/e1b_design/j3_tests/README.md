# E1b J3 scoring job: design record and tests

This folder records how the J3 scoring job (`scripts/e1b_j3_job.py`) and its launcher (`scripts/launch_e1b_j3.py`) were built and tested. It is not a registered analysis: nothing here decides anything, and no value it prints feeds a decision.

## What was missing (review finding C1, and AUD-3's job half)

Pre-registration section 2 (branch A), spec 3.6 and section 12's "J3 scoring" row register an offline job that scores every stored completion of every used E1b run with J3 = Qwen/Qwen2.5-7B-Instruct at revision a09a35458c702b33eeacc393d103063234e8bc28 in bfloat16. No such job existed. `experiments/e1b_primary.load_j3` read `runs/e1b/j3/qwen25_7b.json`, but nothing wrote it, so C6 and Q3 could never be computed. The other half of AUD-3 (an addendum mode in `e1b_primary.py`) belongs to the analysis track.

## What the job does

- **Runs scored.** The used runs come from `experiments/e1b_primary.load_e1b`, which the job calls rather than re-implementing the selection. The data manifest is checked first, as the registered analysis does. The launcher builds the data in a child process that runs `verify_pins_b()` before anything else is imported. This matters because `verify_pins_b` counts every repo module in `sys.modules`.
- **Data file.** A deterministic gzip of JSON lines. The header line holds the used run ids, the used tags per arm, the data manifest's SHA-256 (in the form the results file records them) and the selection counts. It is followed by one row per stored completion: the full stored text, plus its `judge_small` for the validation gate. The baked job checks the file's SHA-256 and row count.
- **Scoring.** Scores go through `reward_judge.JudgeScorer(revision=...)` unchanged. The setup is the E1 rescoring's: `_JUDGE_TEMPLATE`, `extract_code`, the task prompt from `CODE_TASKS`, right padding and truncation at 1024. The bake and the job both stop unless `reward_testhack.py`, `reward_codeset.py` and the template string have the E1 rescoring's SHA-256s. `reward_judge.py` changed after E1 (the revision argument), so its SHA-256 is recorded, not checked.
- **E1 patterns carried over** (rewritten, nothing imported from the frozen `experiments/e1_rescore/`):
  - the validation gate: the training judge Qwen2.5-0.5B-Instruct at 7ae55760, float16, on steps 0-9 of the first used hacking run and the first used twin run. It compares per-step means with the stored `judge_small`, then compares the per-step path with the batched path. Tolerances are 0.02 and 0.01; failing either gives rc 2, and the 7B model is never loaded;
  - the chunk order (edges, every5, rest);
  - task-grouped, length-sorted batches under an OOM-halving token budget;
  - the result format of `experiments/e1_rescore/results/qwen25_7b.json`, keyed by E1b run ids.
- **Fields added to the result:**
  - `run_id` and `stage: "j3"`;
  - `job_file_sha256`, the job's own bytes as measured (finding JOB-1's rule applied here);
  - `snapshots`, the Hub cache's snapshot directories and refs after loading;
  - `used` and `data_manifest`, copied from the data header so an addendum can compare them with the k = 0 results file;
  - every installed package version.
- **Environment.** The PEP 723 header is `scripts/e1b_job.py`'s, byte for byte, so `experiments/e1b_uv.lock` is the job's lock. That lock has the E1 rescoring's versions of torch, transformers, numpy, accelerate, tokenizers, huggingface-hub and safetensors.
- **Monitoring.**
  - stdout carries only `J3 <phase> <scored>/<total>` lines;
  - stderr carries phases, counts and whole-second durations;
  - no score value is printed anywhere;
  - `status.json` holds `{run_id, phase, n_scored, n_total, rc, ts}`;
  - `partial.json` is written after validation, every 10 minutes and after each chunk;
  - an exception or SIGTERM in either stage writes `result.json` marked `aborted`.
- **Snapshot stop (finding R8, 2026-10-25).** As the training job does, after each judge loads and before it scores anything, the job stops unless the Hub cache holds exactly one snapshot of that judge's repo and it is the pinned revision. The stop is exit 1 with `E1B J3 STOP: ...` on stderr, and `result.json` is marked aborted with nothing scored.

## What the launcher does

- **Paths.** S3 `e1/e1b/j3/<run_id>/`, where run_id is `e1b_j3_qwen25_7b_a<k>`. Launch records go to `runs/e1b/j3/launches.json`, measured hours to the shared `runs/e1b/instance_hours.json`, and the result to `runs/e1b/j3/qwen25_7b.json` with `runs/e1b/j3/MANIFEST.sha256`.
- **Launch records apart (finding R10).** The J3 records stay in `runs/e1b/j3/launches.json`, because their keys differ from build_contract section 3's. Every reader of main and pilot records reads `runs/e1b/launches.json` by its exact path: the main launcher, `e1b_primary.load_e1b`, the autorun and the pilot gate (pilot-stage records only). The result readers glob only `e1b_*_s<seed>.json` in `runs/e1b/main` and `runs/e1b/pilot`. So no J3 file is ever taken for a main or pilot record.
- **What --submit checks:**
  - the commit is registered: `e1b_registry.commit_registered` against HEAD's pre-registration;
  - every executed file equals its blob at the commit;
  - the freeze manifest's lines match the freeze commit's blobs, LF-normalized;
  - the receipt passes `launch_e1b_aws.receipt_stamps`;
  - the reader cell and the J3 approval cell are filled, by the main launcher's own rule (finding R7: `launch_e1b_aws.reader_problem` and `approval_problem`, imported, not copied). Each needs a valid YYYY-MM-DD date. The reader cell also needs "section", and the approval cell needs Aarav's words in straight or curly quotes. A cell saying "pending", "not", "TBD", "awaiting" or "none yet" is refused;
  - section 12's J3 ceilings are 8 GPU-h and USD 15;
  - the used runs are final: equal to the registered results file's when it exists; otherwise no replacement is needed and no main run is still to launch;
  - no earlier J3 instance is up;
  - nothing has been collected yet;
  - no other launcher holds `runs/e1b/j3/launch.lock`;
  - read-only role-policy and S3-prefix checks.
- **Spend.** This is e1b_registry's rule, imported (finding R6): `committed(records, "j3")` and `room(...)` for the registry's budget group `j3`, whose ceilings (8 GPU-h, USD 15) and planning hours (`J3_PLAN_H`) live in the registry. Per instance, the rule takes the larger of the planning hours (2.36 h, section 12's USD 4.4 estimate at g6e.xlarge) and the measured hours, at list price. The instance's backstop is the smallest of 8 h and the room left in USD and GPU-hours at its type's price. So the worst case can never exceed the ceiling: USD 14.89 on g6e.xlarge at 480 min, 14.98 on g6e.2xlarge at 401 min. A type with less than 240 minutes of room is skipped.
- **Instance.** The user data schedules the backstop shutdown first, then installs uv by hash and runs the job under `timeout` (backstop minus 20 minutes). A watchdog stops the job after 45 minutes without a `status.json` change. Every 5 minutes it pushes `status.json` and `partial.json`. It uploads the final rc last and powers off on any exit (terminate on shutdown).
- **Measured hours (finding R9).** Both launchers write `runs/e1b/instance_hours.json` through `e1b_registry.update_instance_hours`. It holds an exclusive lock file `instance_hours.json.lock` around the read-merge-write, reads the file again under the lock, and writes with an atomic replace. The merge keeps every entry, and a final entry never changes. So a J3 launcher and a main launcher writing at the same moment lose nothing.
- **--status and --collect** print counts only. Collect refuses a result whose judge, revision or dtype is not the registered J3's and the launch record's (finding R8). It also refuses one whose measured Hub snapshots (`snapshots.target`, the validation judge's) or measured dtype differ from the registered revisions and dtype, or whose data SHA-256, `job_file_sha256` or run id is not the launch record's. It names only the fields. It never replaces a collected file with different bytes, and takes `partial.json` only with `--accept-partial`.

## Tests

```
python experiments/e1b_design/j3_tests/run_tests.py --scratch DIR --venv PY --uv UV --hf-home HF
```

- `--venv`: a CPython 3.12.15 venv with the locked packages and CPU torch (the job smoke harness's `v`).
- `--uv`: uv 0.12.23.
- `--hf-home`: an HF cache that may already hold `trl-internal-testing/tiny-Qwen2ForCausalLM-2.5`.

The test sections:

- **data** uses the real `e1b_primary.py` on its own synthetic self-test runs. It also checks that the launch-record readers name `runs/e1b/launches.json` exactly, that no E1b reader walks `runs/e1b`, and that a J3 file name is not a result-file name (R10).
- **bake** builds in a scratch clone, commits there, and runs the `--bake` CLI.
- **stops** covers a tampered package, a tampered lock, a missing lock, no CUDA, and a tampered data file.
- **cpu** runs the job on the synthetic data with the tiny model, then checks:
  - stdout discipline and no score value printed;
  - E1's result keys;
  - measured provenance and the pinned snapshot, with no `refs/main`;
  - the validation arithmetic;
  - `load_j3` coverage and flags;
  - the gate's failure path;
  - aborts in the target stage and in the validation stage;
  - the snapshot stop (R8): a validation judge, or a target judge only, whose Hub cache shows a second snapshot stops with exit 1 before anything is scored.
- **launcher** runs in a scratch clone with a freeze record, `stub_e1b_primary.py` and `fake_aws.py`. The real `e1b_primary.py`'s pins name local-only files, so the stub stands in for it. Besides the dry runs, refusals, submit, status and collect, it covers:
  - the cells (R7): an impossible date, a reader cell without "section", quotes holding only a space, and "TBD" are refused; curly quotes are accepted;
  - collect's snapshot, dtype, judge and validation checks (R8), and a failed gate's file still being collected;
  - the J3 record staying out of the main launcher's records, status and Pilot ceiling (R10);
  - two launchers writing `runs/e1b/instance_hours.json` at once (R9).
- **userdata** runs the user data under Git bash with fakes. It covers ok, rc 3, stalled, no-job and signal.

Before the fix, the runner fails at its first section, because `scripts/e1b_j3_job.py` does not exist.

## Limits

- The CPU run uses random tiny weights on synthetic texts. The validation gate's per-step tolerance is opened for that run only. Its arithmetic and its failure path are checked separately.
- The CUDA path, bfloat16, the real 7B and 0.5B models, 307,200 completions, Linux and real AWS are not exercised. The E1 rescoring ran the same scoring path on an L40S, which covers them in part.
- The launcher tests use a stand-in `e1b_primary.py` and a fake aws CLI.
