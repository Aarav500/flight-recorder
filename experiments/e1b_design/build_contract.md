# E1b build contract (2026-10-23)

Shared interfaces for the E1b freeze-prerequisite code, so the files can be built in parallel and still fit together. experiments/e1b_preregistration.md and experiments/e1b_analysis_spec.md (both DRAFT, not frozen) define what the code must do. Where this file and they differ, they win. Report the difference; never change them silently.

## 1. Files and owners

| File | What | Status |
|---|---|---|
| flightrecorder/repro/twin_gate.py | `gate(completion, fn_name) -> 0.0/1.0`, `truncated(ids, eos_id, pad_id) -> bool` | built 2026-10-23; `python -m flightrecorder.repro.twin_gate` |
| experiments/e1b_labels.py | `function_only_score(text, task) -> float` (hidden-test fraction, strict scale), `trivial_body(text, fn_name) -> True/False/None` (None when the gate fails) | built 2026-10-23; `python experiments/e1b_labels.py` |
| flightrecorder/repro/reward_judge.py | `JudgeScorer(model_id, device, dtype, max_length, revision=None)`; revision goes to both from_pretrained calls; None leaves the calls unchanged | edited 2026-10-23 |
| experiments/e1b_primary.py, experiments/e1b_pins.json, experiments/e1b_selftest.txt | the registered analysis (spec sections 0-8) | analysis track |
| experiments/e1b_offline_gate.py | `offline_gate(items, python_version) -> (version used, [g])`, items = [(text, fn_name)]; python_version is the run's logged full version (versions.python; 3.12.15 for the job as built). Runs `twin_gate.gate` in-process only when this interpreter's full version equals it exactly, else under `uv run --no-project --python <full version> --managed-python --no-python-downloads`: a uv-managed interpreter of exactly that version, installed beforehand with `uv python install <full version>`; the gate never downloads one, and a different patch release is refused (spec 7.6). e1b_pins.json's `_offline_gate` block records the interpreter, and every analysis checks it. Called by e1b_primary.py and e1b_pilot_gate.py | built 2026-10-23; exact version 2026-10-25 |
| scripts/e1b_job.py, experiments/e1b_uv.lock | the job template and its `bake()`; the lock | job track |
| scripts/launch_e1b_aws.py, scripts/e1b_autorun.py | launcher and collector; autorun | launcher track |
| experiments/e1b_pilot_gate.py | P0-P5 (pre-registration section 9) | pilot track |
| experiments/e1b_power.py | the power and cost numbers of pre-registration sections 4, 5, 9 and spec 1.11, 1.13, 1.14, 3.2 | power track |

Never change: experiments/e1_primary.py, e1_event.py, e1_detectors.py, e1_pins.json, e1_preregistration.md, e1_supplementary.py, experiments/event_candidates/*, anticipatory/curvature_detectors.py, benchmark/*, paper/** (gitignored, local only, never committed or copied into a tracked path), scripts/launch_slow_onset.py, scripts/launch_e1_aws.py, scripts/launch_e1_batch.py, scripts/e1_autorun.py, runs/e1/**, experiments/e1_rescore/**. Import them; never copy their code.

## 2. Constants

- Policy `Qwen/Qwen2.5-1.5B` revision `8faed761d45a263340a0528343f099c05c9a4323`.
- judge_small `Qwen/Qwen2.5-0.5B-Instruct` revision `7ae557604adf67be50417f59c2c2f167def9a775`.
- judge_big `Qwen/Qwen2.5-1.5B-Instruct` revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- J3 `Qwen/Qwen2.5-7B-Instruct` revision `a09a35458c702b33eeacc393d103063234e8bc28`, bfloat16.
- Config as E1: lr 1e-6, beta 0.02, 8 generations, 400 steps, max_completion_length 256, probe 16 every 5, tasks count_vowels, is_prime, second_largest.
- Arms: `judge` (hacking, reward J05) and `judgegate` (twin, reward J05 x g). The `judge` field is `Qwen/Qwen2.5-0.5B-Instruct` in both.
- Seeds: main 300-315 per task and arm; replacements from 316 (at most 14 in all); pilot 299, reruns 298 then 297.
- Python in the job: CPython 3.12.15 exactly (uv-managed; scripts/e1b_job.py PYTHON_VERSION, the managed 3.12 that uv 0.12.23 picks, as in E1's job logs). The offline gate runs under this exact version (section 1).

## 3. Run id and paths

- run_id = `e1b_<arm>_<task>_s<seed>`, e.g. `e1b_judgegate_is_prime_s301`.
- S3: `s3://fr-deploy-325527186655/e1/e1b/<stage>/<run_id>/` holding job.py, job.py.lock, status.json, job.log, result.json, boot.log. Stages: `pilot`, `main`, `main_replace`.
- Local: pilot results go to runs/e1b/pilot/<run_id>.json; main and main_replace results both go to runs/e1b/main/<run_id>.json (load_e1b reads that one folder). Logs to runs/e1b/<stage>/logs/<run_id>.log, status files to runs/e1b/<stage>/status/<run_id>.json.
- Launch records: runs/e1b/launches.json, one list of records, each `{"stage", "run_id", "arm", "task", "seed", "git_commit", "package_sha256", "template_sha256", "lock_sha256", "gate_sha256", "instance_id", "region", "ami", "instance_type", "market", "submitted_utc", "s3", "job_sha256"}` (job_sha256: SHA-256 of the baked job.py bytes, from bake()). Readers: load_e1b, the launcher's guards and collector, and experiments/e1b_pilot_gate.py (pilot-stage records only). Replacements are also listed in runs/e1b/main_replace.json (spec 4.2).
- Data manifest (spec 7.8): the collector writes runs/e1b/<stage>/MANIFEST.sha256 (lines `<sha256>  <file name>`, sorted by name, LF) over every collected result file, before any analysis.
- J3 (section 8): launch records in runs/e1b/j3/launches.json, kept apart from runs/e1b/launches.json because their schema differs (load_e1b, the main launcher and collector, and the pilot gate read the latter); the result in runs/e1b/j3/qwen25_7b.json, with runs/e1b/j3/MANIFEST.sha256 beside it.
- Pilot gate report: runs/e1b/pilot_gate/pilot_gate_seed<seed>.txt and .json, outside runs/e1b/pilot, which holds only result files (section 8).

## 4. Result file (one per run; spec 0.4)

Top level:
- `arm`, `judge`, `policy`, `task`, `seed`, `lr`, `steps`, `maxtok`, `probe_n`, `probe_every`, `beta`, `run_id`, `stage`.
- `revisions`: `{"policy", "judge_small", "judge_big"}` (commit shas).
- `versions`: `{"python": "3.12.x", "packages": {name: version for every installed distribution}, "torch_cuda", "gpu_name", "driver"}`.
- `job_file_sha256` (SHA-256 of the job.py file the instance actually ran, measured by the job from its own bytes), `snapshots` (`{"policy", "judge_small", "judge_big"}`: the Hub snapshot commit each model was actually loaded from, read from the cache after loading).
- `package_sha256` (SHA-256 of the decompressed package tar), `git_commit`, `template_sha256` (SHA-256 of scripts/e1b_job.py at git_commit, LF), `lock_sha256` (experiments/e1b_uv.lock at git_commit), `gate_sha256` (flightrecorder/repro/twin_gate.py at git_commit).
- `log`: one row per step 0..399, E1's fields `step, task, judge_big, judge_small, disagree, proxy, hidden, hidden_lenient, delta, adv_std, entropy, kl, grad_norm, mean_len, clipped`, plus `gate_rate`, `js_std`, `jb_std`. `proxy` and `adv_std` are the mean and SD of the reward the arm optimizes (they include g in the twin). `js_std` and `jb_std` are population SDs (numpy default) of the 8 ungated scores.
- `completions`: one entry per step, column-wise as in E1: `{"step", "texts": [8 full texts, no cut], "n_tokens": [8], "eos": [8 bool], "fenced": [8 bool], "parses": [8 bool], "G": [8 float], "judge_small": [8], "judge_big": [8], "reward": [8], "hidden": [8], "hidden_lenient": [8]}`. `eos` is `not twin_gate.truncated(ids, eos, pad)` on the completion_ids TRL passes to the reward function. `fenced` is `reward_testhack._FENCE.search(text) is not None`. `parses` is ast.parse plus compile of extract_code(text) succeeding.
- `probe`: E1's probe rows `{step, n, probe_hidden, probe_sem}`.
- `trl_history`: trainer.state.log_history.
- No go/no-go block and no other outcome summary.

## 5. Monitoring

- stdout: exactly one line `STEP <n>` per step and no metric value anywhere (no ONSET, PROBE, SAMPLE or summary lines, no trainer log printing, no progress bar values).
- status.json: `{"run_id", "step", "rc", "ts"}` only. The job rewrites it each step with rc null; the instance script writes the final rc.

## 6. Interfaces between files

- `scripts/e1b_job.py`: `bake(arm, task, seed, stage, commit, out_dir) -> (job_path, provenance)` where provenance holds run_id, git_commit, package_sha256, template_sha256, lock_sha256, gate_sha256, job_sha256. The package is `git archive` of `flightrecorder/` at `commit`, never the working tree. `bake` refuses an arm, task, seed or stage outside the registered set.
- `scripts/launch_e1b_aws.py` imports `bake`. It never submits without `--submit`, and with `--submit` refuses unless the freeze record's timestamp receipt exists and matches experiments/e1b_freeze_manifest.txt, the human-reader field of the pre-registration's section 13 is filled, and every file in the job's import closure is unchanged from the commit.
- `experiments/e1b_primary.py`: `load_e1b(main_dir, launch_records)` reads section 3's records; `analyze_b(...)` as spec 8.3. CLI: no argument runs the registered analysis (or the next rerun, k read from the files present); `--selftest` runs `selftest_b` and writes its output to the path given by `--out` (the committed copy is experiments/e1b_selftest.txt); `--control-only` (spec 1.13 cause search); `--seal-audit`; `--exclusions-only` verifies the pins, runs load_e1b, prints only reasons (no values) and exits 3 when a replacement is needed and allowed, 4 when main runs are still unlaunched (and no replacement is needed), 0 otherwise (1 on a provenance failure); `--write-pins` rewrites every pin and is refused once experiments/e1b_freeze_manifest.txt exists; `--write-pins --files a,b` updates only the named pinned or closure files or pin blocks (`_env`, `_e1_data`, `_registered`, `_offline_gate`), before or after the freeze (spec 1.13: a fix updates the pins only for the files it changes); a write that includes the `_offline_gate` block is refused when the job's exact Python is not installed as a uv-managed interpreter (section 1); `--j3-addendum` (section 8 below) writes experiments/e1b_primary_audit_j3.txt and .json, never overwritten. Exit codes: 0 when a mode finishes (`--selftest` and `--selftest-items` only when every entry passes); 1 on a provenance failure, a refusal, a failed self-test, or a malformed command line (USAGE printed: one mode flag at most, `--out` only with `--selftest`, `--files` only with `--write-pins`, `--control-only` and `--seal-audit` only without a mode flag); `--exclusions-only` as above. scripts/e1b_autorun.py calls these as subprocesses.
- `experiments/e1b_pilot_gate.py` reads runs/e1b/pilot and section 4's schema, plus runs/e1b/launches.json and the pre-registration as committed at HEAD for rule (e) (section 8), which it applies to pilot runs as load_e1b does to main runs except the template, lock and gate blob-at-git_commit checks and the freeze-pin checks (the launcher's guard covers those before launch, and P0 checks gate_sha256 against the pinned twin_gate.py): a run that fails it is not the registered pilot run (data incomplete); it computes no detector. It first checks its own import closure against experiments/e1b_pins.json, writes its report to runs/e1b/pilot_gate/pilot_gate_seed<seed>.txt and .json (section 3), and exits 0 when P0-P5 all pass, 1 when any fails, 2 when the data are incomplete, and 3 when the gate cannot be evaluated (a provenance failure, a usage or environment error).

## 7. Repository rules

- The GitHub repo is public. No username, home directory or absolute local path in any file; use paths relative to the repo root (`Path(__file__).resolve().parents[k]`).
- Write text files with LF line endings (in Python: `open(p, "w", newline="\n")` or write bytes).
- No commits and no pushes; the session commits after review.
- No cloud resource is created and no money is spent. Read-only network use (PyPI metadata, wheel downloads into a scratch environment) is fine.
- Self-test and smoke-test output goes to a temporary directory, never to experiments/ or runs/, except the committed experiments/e1b_selftest.txt.

## 8. Revision 2 (2026-10-25, after the code review)

- S3 root moved under `e1/` (Aarav, 2026-10-25: "move E1b's storage path under e1/"): the instance role fr-e1-runner may read and write only `fr-deploy-325527186655/e1/*`. Every E1b S3 key is `e1/e1b/...`.
- experiments/e1b_registry.py (stdlib only) is the one source for the registered launch commits (`registered_commits(prereg_text)`: section 13's "Commit hash" cell and section 14's "Commit" column; `commit_registered(commit, tokens)`) and for the spend rule (`committed(records, group, hours)`, `room(...)`: per instance, the larger of runs x PLAN_H and the measured hours, at its type's list price; budget groups `"pilot"`, `"primary"` and `"j3"`, the last with section 12's "J3 scoring" ceilings and J3_PLAN_H per J3 instance). Both launchers, e1b_primary and the pilot gate import it; none keeps its own copy.
- runs/e1b/instance_hours.json: `{instance_id: {"hours": float, "final": bool}}`, written by both launchers (from describe-instances launch and termination times, or launch to now while running) only through `e1b_registry.update_instance_hours`, which holds the lock file runs/e1b/instance_hours.json.lock around its read-merge-write and replaces the file atomically. No outcome value.
- Registered provenance (rule (e)): a run is registered only if its git_commit is a full 40-hex id that `commit_registered` accepts at HEAD's pre-registration; its template_sha256, lock_sha256 and gate_sha256 equal the LF SHA-256 of those files' blobs at git_commit; its job_file_sha256 equals its launch record's job_sha256; its package_sha256 equals its launch record's; its revisions equal section 2's constants. The launcher refuses --submit for a --commit that `commit_registered` does not accept, and checks the freeze manifest lines against the freeze commit.
- J3: the result goes to runs/e1b/j3/qwen25_7b.json in the format of experiments/e1_rescore/results/qwen25_7b.json. The job is scripts/e1b_j3_job.py (template and bake) with its launcher scripts/launch_e1b_j3.py (stage `j3`, S3 `e1/e1b/j3/`, section 12 "J3 scoring" ceiling USD 15 / 8 GPU-h). Its launch records go to runs/e1b/j3/launches.json, not runs/e1b/launches.json: their schema differs, and load_e1b, the main launcher and collector, and the pilot gate read that file. `python experiments/e1b_primary.py --j3-addendum` runs after the registered run: it verifies the pins, refuses unless the data manifest, the used runs and the pairing table equal those in the k = 0 results file, runs only load_j3 and the audit, and writes experiments/e1b_primary_audit_j3.txt and .json, never overwritten, never touching an H1 output. If the audit was sealed (a failed control), the addendum goes to a sealed file the same way.
- CloudTrail export: the raw events go to the gitignored paper/freeze/e1b_<stage>/cloudtrail.json; a redacted copy (EventId, EventName, EventTime, EventSource, Resources) goes to runs/e1b/audit/<stage>_cloudtrail.json. Nothing but result files lives in runs/e1b/main or runs/e1b/pilot, and both readers glob only `e1b_*_s<seed>.json`. So the pilot gate writes its report to runs/e1b/pilot_gate/ (pilot_gate_seed<seed>.txt and .json).
- Offline gate (2026-10-25): it runs under the job's exact Python version (3.12.15), not its minor version, from a uv-managed interpreter it never downloads (section 1). e1b_pins.json's `_offline_gate` block records the interpreter, and every analysis checks it.
- Manifest hashes are LF-normalized (or compared as parsed {name: sha256}), so a CRLF checkout does not change them.
