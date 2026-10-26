"""E1b on AWS EC2: queued runs back to back on self-terminating g6e (1x L40S) instances. ASCII only.

WHAT
  The launcher and collector that experiments/e1b_preregistration.md (sections 8, 12, 13) and
  experiments/e1b_analysis_spec.md (sections 5, 7) ask for, with the run ids, S3 layout, local paths and launch
  records of experiments/e1b_design/build_contract.md section 3. It follows E1's scripts/launch_e1_aws.py,
  launch_e1_batch.py and e1_autorun.py, which stay frozen: their patterns are rewritten here, nothing is imported
  from them, so the E1b launcher pins nothing of E1's. Jobs are baked by scripts/e1b_job.py's bake() from a named
  commit; this file uploads them and starts the instances.

STAGES (fixed here, not on the command line, so what ran is what the commit says ran)
  pilot         6 runs: 3 tasks x seed 299 x {judge, judgegate}. --rerun=1 runs seed 298, --rerun=2 seed 297
                (pre-registration section 9: a pilot rerun uses 298, then 297).
  main          96 runs: seeds 300-315 x {judge, judgegate} x 3 tasks, seed-major, so any prefix of the queue is
                balanced across arm and task and keeps the matched pairs together.
  main_replace  the runs listed in runs/e1b/main_replace.json, as [arm, task, seed] (or {"arm", "task", "seed"}),
                each seed the next unused one from 316 for its arm and task, at most 14 in all (sections 7, 8, 12).
                Results join runs/e1b/main, where load_e1b reads them.

S3 (build_contract section 8: the instance role fr-e1-runner may read and write only fr-deploy-325527186655/e1/*)
  s3://fr-deploy-325527186655/e1/e1b/<stage>/<run_id>/ holds job.py, job.py.lock, status.json, job.log,
  result.json and boot.log. S3_ROOT below is the one place the e1/e1b root is written.

SPEND GUARDS
  The default is a dry run: bake, price, print the plan and every guard. Nothing is uploaded or launched and no
  AWS call is made. --submit spends, and refuses unless all of these hold:
    * no other --submit is running: runs/e1b/launch.lock is created exclusively for the whole submit and removed
      at its end (a second launcher stops with a message; a killed one leaves the file, to be removed by hand);
    * --commit names the commit the jobs are baked from, and it is registered at HEAD's pre-registration
      (experiments/e1b_registry.py: section 13's "Commit hash", or a commit in section 14's "Commit" column; spec
      7.1). A section 14 commit may differ from the freeze commit only in files its section 14 row names: every
      file of the job's import closure, the template, the lock, the gate, this launcher, the autorun, the registry
      and the manifest's files (the pre-registration itself excepted, since its sections 13 and 14 are filled
      after the freeze) is compared blob by blob;
    * every file in the job's import closure (computed from the template's own imports, followed through the
      package at --commit), the template, the lock, the gate, this launcher, scripts/e1b_autorun.py and
      experiments/e1b_registry.py are identical in the working tree and at --commit (spec 7.1: every executed
      file committed; pre-registration section 8: the autorun script committed before launch);
    * experiments/e1b_freeze_manifest.txt is committed at HEAD, lists every file of section 13's FILES, and each
      of its SHA-256 lines matches the blob at the freeze commit; a timestamp receipt is committed at HEAD and
      stamps the manifest's SHA-256 (section 13; spec 7.5): experiments/e1b_freeze_manifest.txt.ots, a complete
      OpenTimestamps proof holding a pending or Bitcoin attestation (a bare header is refused), or a .tsr, an
      RFC 3161 timestamp response with status granted (a timestamp request .tsq is refused), whose TSTInfo
      imprint is the manifest's SHA-256, also read with `openssl ts -reply -text` when openssl is on PATH;
    * section 13's "Human reader before the freeze" cell holds a YYYY-MM-DD date and says which sections were
      read (the word "section"), and section 12's "Approved by Aarav" cell of the stage's row (Pilot; Primary for
      main and main_replace) holds a date and Aarav's words in quotes; a cell saying "not", "TBD", "awaiting",
      "pending" or "none yet" is refused. Both are read from HEAD;
    * the stage's ceiling (section 12: Pilot USD 30 and 15 GPU-hours; Primary USD 285 and 153 GPU-hours, shared
      by main and main_replace) is not exceeded by the committed spend plus this launch. Committed spend is
      experiments/e1b_registry.committed: per launched instance, the larger of its runs x PLAN_H and its measured
      hours (runs/e1b/instance_hours.json), at its type's on-demand list price. A submit measures the hours again
      from describe-instances before it launches. The ceilings are checked against section 12's table;
    * LAUNCH_CAP (60 launches per arm over main and main_replace) and 14 replacements in all;
    * read-only AWS checks: the instance role allows s3:GetObject and s3:PutObject under the stage's prefix, and
      no run of this launch already has a status.json in S3.
  Before each instance, runs/e1b/launches.json is read again: runs launched meanwhile are dropped, and the ceiling,
  LAUNCH_CAP and the replacement allowance are checked again. Each instance is priced again at every type the
  fallback tries, and a type that would cross the ceiling is skipped. The dry run also prints the worst case
  (every instance runs to its backstop), which can exceed a ceiling: the ceilings of section 12 are planning
  estimates (USD 2.58 and about 1.4 GPU-hours per run).

INSTANCE (user data, one instance per batch of runs)
  Pinned AMI by id (Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) 20261002, the release E1 ran on),
  role fr-e1-runner, InstanceInitiatedShutdownBehavior=terminate, IMDSv2. At boot: a backstop `shutdown -h`
  (per-run timeout x runs + 30 min), then uv==0.12.23 from PyPI with its wheel SHA-256 (--require-hashes). Per run:
  job.py and job.py.lock into an empty directory, status.json {run_id, step: null, rc: null, ts}, then
      E1B_STATUS_PATH=<dir>/status.json timeout <per-run> python3 -m uv run --no-config --locked --python 3.12.15
          --managed-python job.py > job.log 2>&1
  (scripts/e1b_job.RUN_ARGS). A watchdog stops a job whose status.json has not changed for STALL_MIN minutes
  (rc "stalled"), and status.json is pushed every 5 minutes. At the end of a run: result.json, job.log and boot.log
  are uploaded, then status.json with the final rc (an int, or "stalled", "no-job"), last, so a final rc means the
  files are in place. On exit (all runs done, a signal, the backstop) every run without a final rc gets one
  ("interrupted" for the run in progress, "not-run" or "boot-failed" for the queued ones), boot.log goes to every
  run's prefix, and the instance shuts down, which terminates it. The instance never prints a metric value: it
  reads neither job.log nor result.json, and the job's stdout carries only "STEP <n>".

MONITORING AND COLLECTION (pre-registration section 8; spec 5)
  --status reads only each run's status.json (run_id, step, rc, ts) and the instance states, never job.log or
  result.json. A run is final when its status.json holds an rc, or lost when its instance is down (shutting-down,
  terminated, stopping, stopped, or unknown to AWS) without one; the instance state is read first, so a down
  instance has uploaded everything it ever will. --collect refuses while any launched run of the stage is neither,
  then downloads result.json, job.log, status.json and boot.log to the build_contract paths, never replaces a
  collected result file with different bytes, and writes runs/e1b/<stage>/MANIFEST.sha256 (`<sha256>  <file name>`,
  sorted, LF) over the stage's collected result files. It prints names and ok/missing only, never a value.
  Both write runs/e1b/instance_hours.json, {instance_id: {"hours", "final"}}: launch to termination from
  describe-instances (LaunchTime, and the time in StateTransitionReason), or launch to now while the instance runs;
  an instance that is down with no termination time is bounded by its backstop. No outcome value. The file is shared
  with scripts/launch_e1b_j3.py and written only through e1b_registry.update_instance_hours (an exclusive lock file
  around the read-merge-write, then an atomic replace), so two commands writing at once lose no entry.
  --audit-trail exports the CloudTrail events of the recorded instance ids (read-only, spec 7.9). The raw events
  hold the caller's identity ARN, user name, access key id and source IP address, so they go only to the
  gitignored paper/freeze/e1b_<stage>/cloudtrail.json (refused unless git ignores that path); a redacted copy,
  each event's EventId, EventName, EventTime, EventSource and Resources only, goes to
  runs/e1b/audit/<stage>_cloudtrail.json. Nothing but result files is written to runs/e1b/main or runs/e1b/pilot.

    python scripts/launch_e1b_aws.py pilot                                  # dry run at HEAD: bake, price, guards
    python scripts/launch_e1b_aws.py pilot --commit=<sha>                   # dry run at the freeze commit
    python scripts/launch_e1b_aws.py pilot --commit=<sha> --submit          # SPEND (on-demand)
    python scripts/launch_e1b_aws.py main --commit=<sha> --submit --max-runs=24 --region=us-east-2
    python scripts/launch_e1b_aws.py main --status                          # status.json + instance state only
    python scripts/launch_e1b_aws.py main --collect                         # -> runs/e1b/main/<run_id>.json, ...
    python scripts/launch_e1b_aws.py main --audit-trail                     # -> runs/e1b/audit/main_cloudtrail.json
  Options: --rerun=1|2 (pilot)  --spot  --region=us-east-1|us-east-2  --runs-per-instance=K  --max-runs=N
           --show-userdata
"""
from __future__ import annotations

import ast
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True       # leave no new __pycache__ next to the pinned files
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments"))
import e1b_job as J  # noqa: E402  bake(), ARMS, TASKS, STAGE_SEEDS, run_id(), RUN_ARGS, UV_VERSION, UV_WHEEL_SHA256
import e1b_registry as R  # noqa: E402  registered commits, and the spend rule shared with e1b_primary
from e1b_registry import BUDGET_GROUP, CEILINGS, PLAN_H, PRICE, TYPES  # noqa: E402,F401  one copy, in the registry

STAGES = ("pilot", "main", "main_replace")
BUCKET = "fr-deploy-325527186655"
BUCKET_REGION = "us-east-1"          # the bucket's region, whatever region an instance runs in
PROFILE = "fr-e1-runner"
S3_ROOT = "e1/e1b"                   # under e1/, the only prefix fr-e1-runner may read and write (build_contract 8)
RUNS_DIR = ROOT / "runs" / "e1b"
LAUNCHES = RUNS_DIR / "launches.json"
REPLACE_LIST = RUNS_DIR / "main_replace.json"
INSTANCE_HOURS = R.INSTANCE_HOURS    # runs/e1b/instance_hours.json, read by the registry's committed()
LOCK = RUNS_DIR / "launch.lock"      # held for the whole of a --submit
AUDIT_RAW = "paper/freeze/e1b_%s/cloudtrail.json"           # gitignored: the raw events hold identity and IP
AUDIT_REDACTED = "runs/e1b/audit/%s_cloudtrail.json"
AUDIT_KEYS = ("EventId", "EventName", "EventTime", "EventSource", "Resources")
PREREG = "experiments/e1b_preregistration.md"
MANIFEST = "experiments/e1b_freeze_manifest.txt"
RECEIPTS = (MANIFEST + ".ots", MANIFEST + ".tsr", "experiments/e1b_freeze_manifest.tsr")
LAUNCHER, AUTORUN = "scripts/launch_e1b_aws.py", "scripts/e1b_autorun.py"
REGISTRY = "experiments/e1b_registry.py"
RECORD_KEYS = ("stage", "run_id", "arm", "task", "seed", "git_commit", "package_sha256", "template_sha256",
               "lock_sha256", "gate_sha256", "instance_id", "region", "ami", "instance_type", "market",
               "submitted_utc", "s3", "job_sha256")     # build_contract section 3, exactly

# Regions and AMIs. Both AMIs are release 20261002 of the Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu
# 22.04), the one E1's main runs used (us-east-1: what E1's SSM lookup returned on 2026-10-08; us-east-2: E1's pin).
# Checked with describe-images on 2026-10-23: both available, x86_64, root /dev/sda1. Subnets are E1's.
REGIONS = {
    "us-east-1": {"ami": "ami-00803022f231f100e",
                  "subnets": {"us-east-1a": "subnet-0386677a3484341be", "us-east-1b": "subnet-050b222e93f526cd8",
                              "us-east-1c": "subnet-042747a96276b6f96", "us-east-1d": "subnet-028805185f69156e2",
                              "us-east-1e": "subnet-0ca989ad2aa4b4b89", "us-east-1f": "subnet-074261b937ec06406"}},
    "us-east-2": {"ami": "ami-0a2f3a3226fd42ce3",
                  "subnets": {"us-east-2a": "subnet-074b870b48d922f8c", "us-east-2b": "subnet-0c2d1e0b520d8ed3d",
                              "us-east-2c": "subnet-0c20f36c548bb8338"}},
}
REGION_ORDER = ("us-east-1", "us-east-2")
# Instance types and list prices (TYPES, PRICE), PLAN_H (the planning GPU-hours of one run), the section 12 CEILINGS
# and BUDGET_GROUP come from experiments/e1b_registry.py (imported above), shared with e1b_primary.
CAPACITY_CODES = ("InsufficientInstanceCapacity", "SpotMaxPriceTooLow", "Unsupported", "VcpuLimitExceeded",
                  "MaxSpotInstanceCountExceeded")

LAUNCH_CAP = 60              # launches per arm, main and main_replace together (sections 8, 12; spec 0.3)
MAX_REPLACEMENTS = 14        # in all (sections 7, 8, 12)
RUNS_PER_INSTANCE = {"pilot": 6, "main": 12, "main_replace": 14}   # default batch size; --runs-per-instance=K

# Instance limits. E1's 48 main runs trained for 1.09-1.17 h each (train_runtime in runs/e1/main, on g6e xlarge,
# 2xlarge and 4xlarge); the per-run timeout is about 2.5 times that, with the first run's install and downloads.
PER_RUN_MIN = 180            # hard timeout around `uv run job.py`
KILL_AFTER_S = 120           # SIGKILL this long after the timeout's SIGTERM
STALL_MIN = 45               # watchdog: stop a job whose status.json has not changed for this long
BACKSTOP_PAD_MIN = 30        # backstop shutdown = per-run timeout x runs + this
BOOT_SLACK_MIN = 15          # launch to boot, when the backstop is set: bounds the hours of a down, unmeasured instance
TICK_S, PUSH_S = 60, 300     # watchdog tick; status.json upload cadence (E1 pushed job.log every 5 minutes)
VOLUME_GB = 100
DOWN = ("shutting-down", "terminated", "stopping", "stopped", "gone")
ENDED = ("terminated", "stopped", "gone")                  # no further instance-hours
_m = re.search(r'^RESULT_PATH = "([^"]+)"', J.UV, re.M)
RESULT_PATH = _m.group(1) if _m else None     # where the baked job writes its result (/tmp/result.json)
_m = re.search(r"^STEPS, GEN, LR, BETA, MAXTOK = (\d+),", J.UV, re.M)
J_STEPS = int(_m.group(1)) if _m else None    # the template's step count, for the status column only

USER_DATA = r"""#!/bin/bash
# E1b batch instance, written by scripts/launch_e1b_aws.py: __N__ run(s) back to back, then power off. It never
# reads job.log or result.json, so it prints no metric value.
set -u
shutdown -h +__BACKSTOP_MIN__      # backstop first: whatever fails below, the instance powers off (and terminates)
export AWS_DEFAULT_REGION=__BUCKET_REGION__
export HOME="${HOME:-/root}"
W="__WORKDIR__"
RES="__RESULT_PATH__"
S3="__S3_STAGE__"
RUNS="__RUN_IDS__"
mkdir -p "$W" && cd "$W" || exit 1
exec > "$W/boot.log" 2>&1
echo "boot $(date -u +%FT%TZ); backstop +__BACKSTOP_MIN__ min; runs: $RUNS"
CUR=""; JOB=""; WHY=not-run
: > "$W/done.list"

put() { aws s3 cp "$1" "$2" --only-show-errors || echo "upload failed: $2"; }

set_status() {   # set_status FILE RUN_ID RC: {run_id, step, rc, ts}, keeping the step the job last wrote; RC null while running
python3 - "$1" "$2" "$3" <<'PY'
import json, os, sys, time
path, rid, rc = sys.argv[1:4]
step = None
try:
    with open(path) as f:
        d = json.load(f)
    if d.get("run_id") == rid and type(d.get("step")) is int:
        step = d["step"]
except Exception:
    pass
rc = None if rc == "null" else int(rc) if rc.lstrip("-").isdigit() else rc
with open(path + ".tmp", "w") as f:
    json.dump({"run_id": rid, "step": step, "rc": rc, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f)
os.replace(path + ".tmp", path)
PY
}

finish() {       # on any exit: a final rc for every run that has none, boot.log to every run, power off
  trap - EXIT TERM INT HUP
  for R in $RUNS; do
    grep -qx "$R" "$W/done.list" && continue
    if [ "$R" = "$CUR" ]; then
      [ -n "$JOB" ] && kill -TERM "$JOB" 2>/dev/null
      [ -f "$W/run/job.log" ] && put "$W/run/job.log" "$S3/$R/job.log"
      set_status "$W/run/status.json" "$R" interrupted && put "$W/run/status.json" "$S3/$R/status.json"
    else
      rm -f "$W/other.json"
      set_status "$W/other.json" "$R" "$WHY" && put "$W/other.json" "$S3/$R/status.json"
    fi
    echo "$R: final rc written on exit"
  done
  echo "exit $(date -u +%FT%TZ)"
  for R in $RUNS; do put "$W/boot.log" "$S3/$R/boot.log"; done
  shutdown -h now
}
trap finish EXIT
trap 'echo "signal $(date -u +%FT%TZ)"; exit 143' TERM INT HUP

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo "uv==__UV_VERSION__ --hash=sha256:__UV_SHA256__" > "$W/uv-req.txt"
install_uv() { python3 -m pip install -q --only-binary :all: --require-hashes -r "$W/uv-req.txt"; }
install_uv || { apt-get update -q && apt-get install -yq python3-pip && install_uv; } || { WHY=boot-failed; echo "uv install failed"; exit 1; }
python3 -m uv --version
export HF_HOME="$W/hf" UV_CACHE_DIR="$W/uvcache" PYTHONUNBUFFERED=1

for R in $RUNS; do
  CUR="$R"; JOB=""; B="$S3/$R"
  rm -rf "$W/run" "$RES" && mkdir -p "$W/run"
  if ! aws s3 cp "$B/job.py" "$W/run/job.py" --only-show-errors || ! aws s3 cp "$B/job.py.lock" "$W/run/job.py.lock" --only-show-errors; then
    echo "$R: job.py or job.py.lock not found"
    set_status "$W/run/status.json" "$R" no-job && put "$W/run/status.json" "$B/status.json"
    echo "$R" >> "$W/done.list"
    continue
  fi
  set_status "$W/run/status.json" "$R" null && put "$W/run/status.json" "$B/status.json"
  echo "$R: start $(date -u +%FT%TZ)"
  cd "$W/run"
  E1B_STATUS_PATH="$W/run/status.json" timeout --kill-after=__KILL_AFTER_S__ __PER_RUN_MIN__m python3 -m __RUN_ARGS__ > "$W/run/job.log" 2>&1 &
  JOB=$!
  cd "$W"
  STALLED=0; LAST=$(date +%s)
  while kill -0 "$JOB" 2>/dev/null; do
    sleep __TICK_S__ & wait $!
    NOW=$(date +%s)
    M=$(stat -c %Y "$W/run/status.json" 2>/dev/null || echo "$NOW")
    if [ "$STALLED" = 0 ] && [ $((NOW - M)) -gt __STALL_S__ ]; then
      echo "$R: watchdog: status.json unchanged for $((NOW - M)) s, stopping the job"
      STALLED=1; kill -TERM "$JOB" 2>/dev/null
    fi
    if [ $((NOW - LAST)) -ge __PUSH_S__ ]; then put "$W/run/status.json" "$B/status.json"; LAST=$NOW; fi
  done
  wait "$JOB"; RC=$?
  [ "$STALLED" = 1 ] && RC=stalled
  JOB=""
  echo "$R: end rc=$RC $(date -u +%FT%TZ)"
  [ -f "$RES" ] && put "$RES" "$B/result.json"
  put "$W/run/job.log" "$B/job.log"
  put "$W/boot.log" "$B/boot.log"
  set_status "$W/run/status.json" "$R" "$RC" && put "$W/run/status.json" "$B/status.json"
  echo "$R" >> "$W/done.list"
done
CUR=""
echo "all runs done $(date -u +%FT%TZ)"
"""
_UD_KEYS = ("__N__", "__BUCKET_REGION__", "__WORKDIR__", "__RESULT_PATH__", "__S3_STAGE__", "__RUN_IDS__",
            "__BACKSTOP_MIN__", "__UV_VERSION__", "__UV_SHA256__", "__KILL_AFTER_S__", "__PER_RUN_MIN__",
            "__RUN_ARGS__", "__TICK_S__", "__STALL_S__", "__PUSH_S__")


# ------------------------------------------------------------------------------------------
# small helpers
def _utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path, obj):
    """Atomic, LF, ASCII."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="ascii", newline="\n") as f:
        f.write(json.dumps(obj, indent=1) + "\n")
    os.replace(tmp, path)


def _git(*args):
    """-> (returncode, stdout bytes)."""
    r = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True)
    return r.returncode, r.stdout


def resolve(commit):
    rc, out = _git("rev-parse", "--verify", "--quiet", "%s^{commit}" % commit)
    sha = out.decode().strip()
    return sha if rc == 0 and re.fullmatch(r"[0-9a-f]{40}", sha) else None


def blob(rev, path):
    """The committed bytes of path at rev, or None."""
    rc, out = _git("cat-file", "blob", "%s:%s" % (rev, path))
    return out if rc == 0 else None


def same_as_commit(path, rev):
    """None when the working-tree file equals its blob at rev (a CRLF checkout of an LF blob counts as equal), else
    why not."""
    b = blob(rev, path)
    if b is None:
        return "not committed at %s" % rev[:12]
    f = ROOT / path
    if not f.is_file():
        return "missing from the working tree"
    wt = f.read_bytes()
    return None if wt == b or wt.replace(b"\r\n", b"\n") == b else "differs from its blob at %s" % rev[:12]


def _aws_exe():
    return shutil.which("aws") or "aws"      # which() honours PATHEXT, so a Windows aws.exe or aws.cmd is found


def aws(region, *args):
    """aws CLI -> parsed JSON (None for empty output). RuntimeError carries the CLI's error text."""
    r = subprocess.run([_aws_exe(), *args, "--region", region, "--output", "json"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("aws %s failed: %s" % (" ".join(args[:2]), (r.stderr or "").strip()[-600:]))
    return json.loads(r.stdout) if r.stdout.strip() else None


def s3(*args):
    """aws s3 ... against the bucket's region -> (returncode, stdout, stderr)."""
    r = subprocess.run([_aws_exe(), "s3", *args, "--region", BUCKET_REGION, "--only-show-errors"],
                       capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def _code(err):
    msg = str(err)
    return msg.split("An error occurred (")[-1].split(")")[0] if "An error occurred (" in msg else ""


def stage_key(stage):
    """The stage's S3 key prefix inside the bucket: e1/e1b/<stage>."""
    return "%s/%s" % (S3_ROOT, stage)


def prefix(stage, rid):
    return "s3://%s/%s/%s" % (BUCKET, stage_key(stage), rid)


def result_dir(stage):
    return RUNS_DIR / ("pilot" if stage == "pilot" else "main")      # main_replace joins main (build_contract 3)


# ------------------------------------------------------------------------------------------
# plan
def read_replacements():
    """runs/e1b/main_replace.json -> [(arm, task, seed)], checked: registered arm, task and seed (316-327), no
    duplicate, each arm and task's seeds 316, 317, ... in order (the next unused seed), at most 14 in all."""
    if not REPLACE_LIST.exists():
        return []
    try:
        data = json.loads(REPLACE_LIST.read_text(encoding="utf-8"))
    except ValueError as e:
        raise SystemExit("REFUSE: runs/e1b/main_replace.json is not JSON: %s" % e)
    if not isinstance(data, list):
        raise SystemExit("REFUSE: runs/e1b/main_replace.json must be a list of [arm, task, seed]")
    rows, nxt = [], {}
    for i, e in enumerate(data):
        if isinstance(e, dict):
            e = [e.get("arm"), e.get("task"), e.get("seed")]
        if not isinstance(e, (list, tuple)) or len(e) != 3:
            raise SystemExit("REFUSE: runs/e1b/main_replace.json entry %d is not [arm, task, seed]" % i)
        arm, task, seed = e
        if arm not in J.ARMS or task not in J.TASKS or isinstance(seed, bool) or not isinstance(seed, int) \
                or seed not in J.STAGE_SEEDS["main_replace"]:
            raise SystemExit("REFUSE: runs/e1b/main_replace.json entry %d %r is not a registered replacement "
                             "(arm %s, task %s, seed 316-327)" % (i, e, "/".join(J.ARMS), "/".join(J.TASKS)))
        want = nxt.get((arm, task), J.STAGE_SEEDS["main_replace"][0])
        if seed != want:
            raise SystemExit("REFUSE: runs/e1b/main_replace.json entry %d: %s %s seed %d, but the next unused seed "
                             "for that arm and task is %d" % (i, arm, task, seed, want))
        nxt[(arm, task)] = seed + 1
        rows.append((arm, task, seed))
    if len(rows) > MAX_REPLACEMENTS:
        raise SystemExit("REFUSE: runs/e1b/main_replace.json lists %d replacements; at most %d in all"
                         % (len(rows), MAX_REPLACEMENTS))
    return rows


def plan(stage, rerun=0):
    """[(arm, task, seed)] in launch order."""
    if stage == "pilot":
        seed = J.STAGE_SEEDS["pilot"][rerun]
        return [(a, t, seed) for a in J.ARMS for t in J.TASKS]
    if stage == "main":
        return [(a, t, s) for s in J.STAGE_SEEDS["main"] for a in J.ARMS for t in J.TASKS]
    return read_replacements()


def load_launches():
    if not LAUNCHES.exists():
        return []
    recs = json.loads(LAUNCHES.read_text(encoding="utf-8"))
    if not isinstance(recs, list):
        raise SystemExit("REFUSE: runs/e1b/launches.json is not a list of launch records")
    return recs


def stage_records(stage, recs=None):
    return [r for r in (load_launches() if recs is None else recs) if r.get("stage") == stage and r.get("instance_id")]


def launched_ids(recs):
    return {r["run_id"] for r in recs if r.get("instance_id")}


def batches(todo, k):
    return [todo[i:i + k] for i in range(0, len(todo), k)]


def backstop_min(n_runs, per_run_min=PER_RUN_MIN):
    return per_run_min * n_runs + BACKSTOP_PAD_MIN


def render(stage, run_ids, workdir="/opt/e1b", result_path=None, per_run_min=PER_RUN_MIN, kill_after_s=KILL_AFTER_S,
           stall_s=STALL_MIN * 60, tick_s=TICK_S, push_s=PUSH_S, backstop=None):
    """The user data of one instance. Every placeholder is used and none is left; ASCII."""
    if not (result_path or RESULT_PATH):
        raise RuntimeError("the template's RESULT_PATH was not found in scripts/e1b_job.py")
    subs = {"__N__": str(len(run_ids)), "__BUCKET_REGION__": BUCKET_REGION, "__WORKDIR__": workdir,
            "__RESULT_PATH__": result_path or RESULT_PATH, "__S3_STAGE__": "s3://%s/%s" % (BUCKET, stage_key(stage)),
            "__RUN_IDS__": " ".join(run_ids),
            "__BACKSTOP_MIN__": str(backstop if backstop is not None else backstop_min(len(run_ids), per_run_min)),
            "__UV_VERSION__": J.UV_VERSION, "__UV_SHA256__": J.UV_WHEEL_SHA256, "__KILL_AFTER_S__": str(kill_after_s),
            "__PER_RUN_MIN__": str(per_run_min), "__RUN_ARGS__": " ".join(J.RUN_ARGS), "__TICK_S__": str(tick_s),
            "__STALL_S__": str(stall_s), "__PUSH_S__": str(push_s)}
    assert set(subs) == set(_UD_KEYS)
    ud = USER_DATA
    for k in _UD_KEYS:
        if k not in ud:
            raise RuntimeError("user-data placeholder %s is unused" % k)
        ud = ud.replace(k, subs[k])
    left = re.findall(r"__[A-Z][A-Z0-9_]*__", ud)
    if left:
        raise RuntimeError("unsubstituted in user data: %s" % ", ".join(sorted(set(left))))
    ud.encode("ascii")
    return ud


# ------------------------------------------------------------------------------------------
# the pre-registration's freeze record (sections 12, 13), the manifest and its receipt
def freeze_record(text):
    """Section 12 rows {row: {"gpu_h", "usd", "approved"}}, section 13's human-reader value and FILES list."""
    out = {"rows": {}, "reader": None, "files": None}
    s12, s13 = R.section(text, 12), R.section(text, 13)
    if s12:
        rows = R.table_rows(s12)
        head = rows[0] if rows else []
        col = {name: next((i for i, c in enumerate(head) if c.startswith(name)), None)
               for name in ("Ceiling (GPU-hours)", "Ceiling (USD)", "Approved by Aarav")}
        for r in rows[1:]:
            def num(name):
                i = col[name]
                m = re.match(r"\s*([0-9]+(?:\.[0-9]+)?)", r[i]) if i is not None and i < len(r) else None
                return float(m.group(1)) if m else None
            i = col["Approved by Aarav"]
            out["rows"][r[0]] = {"gpu_h": num("Ceiling (GPU-hours)"), "usd": num("Ceiling (USD)"),
                                 "approved": r[i] if i is not None and i < len(r) else None}
    if s13:
        for r in R.table_rows(s13):
            if r[0].startswith("Human reader before the freeze") and len(r) > 1:
                out["reader"] = r[1]
        m = re.search(r'FILES="([^"]*)"', "\n".join(s13))
        out["files"] = m.group(1).split() if m else None
    return out


def deviation_rows(text, sha):
    """The section 14 rows whose "Commit" column names sha, by the registry's rule (a registered id, full or
    abbreviated, that sha starts with), each as one line of text."""
    rows = R.table_rows(R.section(text, 14))
    if not rows:
        return []
    col = next((i for i, c in enumerate(rows[0]) if c == "Commit"), None)
    if col is None:
        return []
    return [" | ".join(r) for r in rows[1:] if col < len(r) and R.commit_registered(sha, R._HEX.findall(r[col]))]


def names_path(text, path):
    """True when text names the repo path in full (not as part of a longer path)."""
    return re.search(r"(?<![\w./-])%s(?![\w/-]|\.\w)" % re.escape(path), text) is not None


# Section 13's reader cell and section 12's approval cell (F9): a real date and the right words, nothing that says the
# step has not happened yet. A false refusal only means the cell is reworded; a false pass launches without approval.
_NOT_YET = re.compile(r"\b(not|tbd|awaiting|pending|none yet)\b", re.I)
_QUOTED = re.compile('"[^"]*\\w[^"]*"|\u201c[^\u201d]*\\w[^\u201d]*\u201d')     # straight or curly quotes


def _has_date(cell):
    for m in re.finditer(r"\b(\d{4}-\d{2}-\d{2})\b", cell):
        try:
            datetime.strptime(m.group(1), "%Y-%m-%d")
            return True
        except ValueError:
            pass
    return False


def reader_problem(cell):
    """None when section 13's 'Human reader before the freeze' cell holds a YYYY-MM-DD date and the word 'section'
    (which sections were read), else why not."""
    cell = (cell or "").strip()
    if not cell:
        return "is empty"
    m = _NOT_YET.search(cell)
    if m:
        return "says %r" % m.group(0)
    if not _has_date(cell):
        return "has no YYYY-MM-DD date"
    if not re.search(r"\bsections?\b", cell, re.I):
        return "does not say which sections were read (no word 'section')"
    return None


def approval_problem(cell):
    """None when a section 12 'Approved by Aarav' cell holds a YYYY-MM-DD date and Aarav's words in quotes."""
    cell = (cell or "").strip()
    if not cell:
        return "is empty"
    m = _NOT_YET.search(cell)
    if m:
        return "says %r" % m.group(0)
    if not _has_date(cell):
        return "has no YYYY-MM-DD date"
    if not _QUOTED.search(cell):
        return "has no quoted words of Aarav's"
    return None


# ---- timestamp receipts (F3). A receipt must be a timestamp, not a request for one.
OTS_MAGIC = b"\x00OpenTimestamps\x00\x00Proof\x00\xbf\x89\xe2\xe8\x84\xe8\x92\x94"
OTS_ATTESTATIONS = {bytes.fromhex("83dfe30d2ef90c8e"): "pending", bytes.fromhex("0588960d73d71901"): "Bitcoin"}
OTS_UNARY = (0x02, 0x03, 0x08, 0x67, 0xf2, 0xf3)     # sha1, ripemd160, sha256, keccak256, reverse, hexlify
OTS_BINARY = (0xf0, 0xf1)                            # append, prepend: one varbytes argument
SHA256_OID = bytes.fromhex("0609608648016503040201")      # DER of 2.16.840.1.101.3.4.2.1
SIGNED_DATA_OID = bytes.fromhex("2a864886f70d010702")     # 1.2.840.113549.1.7.2, content octets
TSTINFO_OID = bytes.fromhex("060b2a864886f70d0109100104")  # DER of id-ct-TSTInfo 1.2.840.113549.1.9.16.1.4


def ots_problem(data, digest):
    """None when data is a complete OpenTimestamps proof of digest (magic, version 1, SHA-256 file digest, then a
    timestamp tree parsed to its last byte, as opentimestamps 0.4.5 reads it) holding at least one pending or Bitcoin
    attestation; else why not. The attestation itself (calendar or block header) is not verified here."""
    if data[:31] != OTS_MAGIC:
        return "is not an OpenTimestamps file"
    if data[31:33] != b"\x01\x08":
        return "is not an OpenTimestamps version 1 proof of a SHA-256 file digest"
    if data[33:65] != digest:
        return "stamps another file digest, not the manifest's sha256"
    if len(data) == 65:
        return "is a bare header with no timestamp operation or attestation after it"
    pos, found = [65], []

    def byte():
        if pos[0] >= len(data):
            raise ValueError("truncated")
        pos[0] += 1
        return data[pos[0] - 1]

    def varbytes(most, least=0):
        n, shift = 0, 0
        while True:
            b = byte()
            n |= (b & 0x7f) << shift
            if not b & 0x80:
                break
            shift += 7
            if shift > 63:
                raise ValueError("bad length")
        if not least <= n <= most or pos[0] + n > len(data):
            raise ValueError("bad length %d" % n)
        pos[0] += n

    def branch(tag, depth):
        if tag == 0x00:
            if pos[0] + 8 > len(data):
                raise ValueError("truncated attestation")
            found.append(data[pos[0]:pos[0] + 8])
            pos[0] += 8
            varbytes(8192)
        elif tag in OTS_BINARY:
            varbytes(4096, 1)
            timestamp(depth - 1)
        elif tag in OTS_UNARY:
            timestamp(depth - 1)
        else:
            raise ValueError("unknown operation 0x%02x at byte %d" % (tag, pos[0] - 1))

    def timestamp(depth):
        if depth <= 0:
            raise ValueError("nested too deeply")
        tag = byte()
        while tag == 0xff:
            branch(byte(), depth)
            tag = byte()
        branch(tag, depth)

    try:
        timestamp(256)
        if pos[0] != len(data):
            raise ValueError("%d byte(s) after the timestamp" % (len(data) - pos[0]))
    except ValueError as e:
        return "is not a well-formed OpenTimestamps proof (%s)" % e
    kinds = sorted({OTS_ATTESTATIONS[t] for t in found if t in OTS_ATTESTATIONS})
    if not kinds:
        return "holds no pending or Bitcoin attestation"
    return None


def _der(data, i, end=None):
    """The DER element at i -> (tag, content start, content end), or None. Definite lengths only."""
    end = len(data) if end is None else end
    if i + 2 > end:
        return None
    tag, n, j = data[i], data[i + 1], i + 2
    if n & 0x80:
        k = n & 0x7f
        if not 1 <= k <= 4 or j + k > end:
            return None
        n, j = int.from_bytes(data[j:j + k], "big"), j + k
    return (tag, j, j + n) if j + n <= end else None


def _der_children(data, start, end):
    out, i = [], start
    while i < end:
        e = _der(data, i, end)
        if e is None:
            return None
        out.append(e)
        i = e[2]
    return out


def tsr_problem(data, digest):
    """None when data is an RFC 3161 TimeStampResp: PKIStatus granted (0) or grantedWithMods (1), a timeStampToken
    of CMS signedData with id-ct-TSTInfo content, whose message imprint is SHA-256 of digest; and, when openssl is on
    PATH, `openssl ts -reply -in <file> -text` reads it with "Status: Granted". Else why not. A timestamp request
    (.tsq) fails the first test: it starts with its version INTEGER, not a PKIStatusInfo SEQUENCE. The TSA's
    signature is not verified here."""
    top = _der(data, 0)
    if top is None or top[0] != 0x30 or top[2] != len(data):
        return "is not one DER SEQUENCE"
    kids = _der_children(data, top[1], top[2])
    if not kids or kids[0][0] != 0x30:
        return "is not a timestamp response (no PKIStatusInfo; a timestamp request .tsq is not a receipt)"
    status = _der_children(data, kids[0][1], kids[0][2])
    if not status or status[0][0] != 0x02 or data[status[0][1]:status[0][2]] not in (b"\x00", b"\x01"):
        return "is a timestamp response whose PKIStatus is not granted or grantedWithMods"
    if len(kids) < 2 or kids[1][0] != 0x30:
        return "is a timestamp response without a timeStampToken"
    tok = _der_children(data, kids[1][1], kids[1][2])
    if not tok or tok[0][0] != 0x06 or data[tok[0][1]:tok[0][2]] != SIGNED_DATA_OID:
        return "has a timeStampToken that is not CMS signedData"
    t = data.find(TSTINFO_OID, kids[1][1], kids[1][2])
    if t < 0:
        return "has a timeStampToken without id-ct-TSTInfo content"
    s = data.find(SHA256_OID, t + len(TSTINFO_OID), kids[1][2])      # the TSTInfo's messageImprint algorithm
    if s < 0:
        return "has a TSTInfo whose message imprint is not SHA-256"
    j = s + len(SHA256_OID)
    j += 2 if data[j:j + 2] == b"\x05\x00" else 0                     # optional NULL parameters
    if data[j:j + 34] != b"\x04\x20" + digest:
        return "has a TSTInfo whose message imprint is not the manifest's sha256"
    exe = shutil.which("openssl")
    if exe:
        fd, tmp = tempfile.mkstemp(suffix=".tsr")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            r = subprocess.run([exe, "ts", "-reply", "-in", tmp, "-text"], capture_output=True, text=True,
                               errors="replace")
        finally:
            os.unlink(tmp)
        if r.returncode != 0 or "Status: Granted" not in r.stdout:
            return "is not read by `openssl ts -reply -text` as a granted response (rc %d)" % r.returncode
    return None


def receipt_problem(name, data, digest):
    """None when the receipt is a timestamp of digest (32 bytes): .ots by ots_problem, .tsr by tsr_problem."""
    return ots_problem(data, digest) if name.endswith(".ots") else tsr_problem(data, digest)


def receipt_stamps(name, data, digest):
    return receipt_problem(name, data, digest) is None


def import_closure(rev):
    """Repo files the job imports: the template's own flightrecorder imports, followed through the package at rev
    (absolute and relative imports, function-level ones included), plus every package __init__.py on the way."""
    rc, out = _git("ls-tree", "-r", "--name-only", rev, "flightrecorder/")
    files = set(out.decode().split()) if rc == 0 else set()

    def path_of(mod):
        base = mod.replace(".", "/")
        return next((p for p in (base + ".py", base + "/__init__.py") if p in files), None)

    def imports(tree, mod, is_pkg):
        pkg = mod.split(".") if is_pkg else mod.split(".")[:-1]
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                yield from (a.name for a in n.names)
            elif isinstance(n, ast.ImportFrom):
                if n.level:
                    base = pkg[:len(pkg) - n.level + 1] if n.level - 1 <= len(pkg) else []
                    target = ".".join(base + ([n.module] if n.module else []))
                else:
                    target = n.module or ""
                yield target
                yield from (target + "." + a.name for a in n.names)

    found, todo = set(), list(imports(ast.parse(J.UV), "__job__", False))
    while todo:
        mod = todo.pop()
        if not (mod == "flightrecorder" or mod.startswith("flightrecorder.")):
            continue
        parts = mod.split(".")
        for k in range(1, len(parts) + 1):
            sub = ".".join(parts[:k])
            p = path_of(sub)
            if p is None or p in found:
                continue
            found.add(p)
            src = blob(rev, p)
            if src is not None:
                todo.extend(imports(ast.parse(src), sub, p.endswith("__init__.py")))
    return sorted(found)


# ------------------------------------------------------------------------------------------
# guards
def guards(stage, commit, todo, recs, submit, hours=None):
    """-> (refusals, notes). Reads git and local files only; the AWS checks are aws_guards()."""
    fail, note = [], []
    if commit is None:
        (fail if submit else note).append("--commit is required with --submit (this dry run checks against HEAD)")
    sha = resolve(commit or "HEAD")
    head = resolve("HEAD")
    if sha is None:
        fail.append("commit %r is unknown" % (commit or "HEAD"))
        return fail, note
    note.append("commit %s%s" % (sha, "" if commit else " (HEAD)"))
    # every executed file committed: the import closure, the template, the lock, the gate, this launcher, autorun and
    # the registry (the spend rule)
    closure = import_closure(sha)
    note.append("job import closure at %s: %s" % (sha[:12], ", ".join(closure) or "NONE FOUND"))
    if not closure:
        fail.append("no flightrecorder module of the job's import closure is committed at %s" % sha[:12])
    executed = set(closure) | {J.TEMPLATE_PATH, J.LOCK_PATH, J.GATE_PATH, LAUNCHER, AUTORUN, REGISTRY}
    for p in sorted(executed):
        why = same_as_commit(p, sha)
        if why:
            fail.append("%s %s" % (p, why))
    # the pre-registration as committed at HEAD: section 13's freeze commit, FILES and reader field, section 12's row,
    # section 14's commits
    pre = blob("HEAD", PREREG)
    text = pre.decode("utf-8", "replace") if pre is not None else None
    rec = freeze_record(text) if text is not None else None
    if rec is None:
        fail.append("%s is not committed at HEAD" % PREREG)
    # commit identity (spec 7.1): the freeze commit, or a commit logged in section 14 before launch
    fc, sha_registered = None, False
    if text is not None:
        tok = R.freeze_commit(text)
        if tok is None:
            fail.append("section 13 'Commit hash' is not filled at HEAD: there is no freeze commit to launch from")
        else:
            fc = resolve(tok)
            if fc is None:
                fail.append("section 13's freeze commit %s is not a commit in this repository" % tok)
            else:
                note.append("freeze commit (section 13): %s" % fc)
        sha_registered = R.commit_registered(sha, R.registered_commits(text))
        if not sha_registered:
            fail.append("commit %s is neither the freeze commit (section 13 'Commit hash') nor a commit in section "
                        "14's Commit column at HEAD; jobs are built only from a registered commit (spec 7.1)"
                        % sha[:12])
    # the freeze manifest and its timestamp receipt, both committed at HEAD; the lines hold at the freeze commit
    man = blob("HEAD", MANIFEST)
    listed = {}
    if man is None:
        fail.append("%s is not committed at HEAD (section 13: committed before any launch)" % MANIFEST)
    else:
        if same_as_commit(MANIFEST, "HEAD"):
            fail.append("%s in the working tree differs from HEAD" % MANIFEST)
        bad = []
        for i, ln in enumerate(man.decode("utf-8", "replace").split("\n")):
            if not ln.strip():
                continue
            m = re.fullmatch(r"([0-9a-f]{64})  (\S+)", ln.rstrip("\r"))
            if not m:
                bad.append("line %d is not '<sha256>  <path>'" % (i + 1))
                continue
            listed[m.group(2)] = m.group(1)
            if fc is None:
                continue
            b = blob(fc, m.group(2))
            if b is None:
                bad.append("%s is not committed at the freeze commit %s" % (m.group(2), fc[:12]))
            elif hashlib.sha256(b).hexdigest() != m.group(1):
                bad.append("%s differs from its manifest line at the freeze commit %s" % (m.group(2), fc[:12]))
        if fc is None and listed:
            bad.append("its lines are not checked: no freeze commit")
        fail += ["manifest: " + b for b in bad]
        files = rec["files"] if rec else None
        if files is None:
            fail.append("section 13's FILES list was not found in %s" % PREREG)
        else:
            missing = [f for f in files if f not in listed]
            if missing:
                fail.append("manifest lacks section 13 file(s): %s" % ", ".join(missing))
        note.append("manifest: %d line(s), sha256 %s" % (len(listed), hashlib.sha256(man).hexdigest()))
        digest = hashlib.sha256(man).digest()
        got = [(p, blob("HEAD", p)) for p in RECEIPTS]
        got = [(p, b) for p, b in got if b is not None]
        if not got:
            fail.append("no timestamp receipt committed at HEAD (%s)" % " or ".join(RECEIPTS))
        else:
            why = [(p, receipt_problem(p, b, digest)) for p, b in got]
            ok = [p for p, w in why if w is None]
            if ok:
                note.append("receipt %s stamps the manifest's sha256" % ok[0])
            else:
                fail.append("no receipt stamps the manifest's sha256: %s"
                            % "; ".join("receipt %s %s" % (p, w) for p, w in why))
    # a section 14 commit differs from the freeze commit only in files its section 14 row names
    if text is not None and fc is not None and sha != fc and sha_registered:
        rows = deviation_rows(text, sha)
        scope = (executed | set(import_closure(fc)) | {J.TEMPLATE_PATH, J.LOCK_PATH, J.GATE_PATH} | set(listed)
                 | set((rec or {}).get("files") or [])) - {PREREG}
        changed = sorted(p for p in scope if blob(fc, p) != blob(sha, p))
        note.append("commit %s is a section 14 commit; against the freeze commit %s it changes: %s"
                    % (sha[:12], fc[:12], ", ".join(changed) or "nothing"))
        unnamed = [p for p in changed if not any(names_path(r, p) for r in rows)]
        if unnamed:
            fail.append("commit %s differs from the freeze commit %s in %s, which its section 14 row does not name"
                        % (sha[:12], fc[:12], ", ".join(unnamed)))
    if rec is not None:
        why = reader_problem(rec["reader"])
        if why:
            fail.append("section 13 'Human reader before the freeze' %s at HEAD: %r" % (why, rec["reader"] or ""))
        else:
            note.append("human reader: %s" % rec["reader"])
        label, ceil_h, ceil_usd = CEILINGS[BUDGET_GROUP[stage]]
        row = rec["rows"].get(label)
        if row is None:
            fail.append("section 12 has no %s row" % label)
        else:
            why = approval_problem(row["approved"])
            if why:
                fail.append("section 12 %s row: 'Approved by Aarav' %s at HEAD: %r"
                            % (label, why, (row["approved"] or "").strip()))
            else:
                note.append("section 12 %s approval: %s" % (label, row["approved"].strip()))
            if row["gpu_h"] != ceil_h or row["usd"] != ceil_usd:
                fail.append("section 12 %s ceilings (%s GPU-h, USD %s) differ from this launcher's (%s, %s)"
                            % (label, row["gpu_h"], row["usd"], ceil_h, ceil_usd))
    if head and head != sha:
        note.append("HEAD %s; the freeze record above is read from HEAD, the jobs from %s" % (head[:12], sha[:12]))
    fail += budget_guard(stage, todo, recs, note, R.load_instance_hours(INSTANCE_HOURS) if hours is None else hours)
    return fail, note


def budget_guard(stage, todo, recs, note, hours):
    """Ceiling (section 12), LAUNCH_CAP and the replacement allowance, for this launch at the cheapest type.
    Committed spend is e1b_registry.committed: per instance, the larger of runs x PLAN_H and its measured hours."""
    fail = []
    group = BUDGET_GROUP[stage]
    label, ceil_h, ceil_usd = CEILINGS[group]
    c_usd, c_h = R.committed(recs, group, hours)
    insts = {r["instance_id"] for r in recs if r.get("instance_id") and BUDGET_GROUP.get(r.get("stage")) == group}
    measured = sum(1 for i in insts if isinstance(hours.get(i), dict) and hours[i].get("hours") is not None)
    n = len(todo)
    add_usd, add_h = n * PLAN_H * TYPES[0][1], n * PLAN_H
    note.append("%s ceiling USD %.0f / %.0f GPU-h: committed USD %.2f / %.1f GPU-h (%d instance(s), %d measured; each "
                "at the larger of runs x %.3f h and its measured hours); this launch at the cheapest type USD %.2f / "
                "%.1f GPU-h (%d runs x %.3f h x $%.3f/h)"
                % (label, ceil_usd, ceil_h, c_usd, c_h, len(insts), measured, PLAN_H, add_usd, add_h, n, PLAN_H,
                   TYPES[0][1]))
    if c_usd + add_usd > ceil_usd + 1e-9 or c_h + add_h > ceil_h + 1e-9:
        fail.append("%s ceiling: committed USD %.2f + this launch USD %.2f > USD %.0f, or %.1f + %.1f GPU-h > %.0f"
                    % (label, c_usd, add_usd, ceil_usd, c_h, add_h, ceil_h))
    if stage in ("main", "main_replace"):
        for arm in J.ARMS:
            k = sum(1 for r in recs if r.get("instance_id") and r.get("stage") in ("main", "main_replace")
                    and r.get("arm") == arm) + sum(1 for a, _t, _s in todo if a == arm)
            if k > LAUNCH_CAP:
                fail.append("LAUNCH_CAP: %d launches in arm %s with this launch; at most %d" % (k, arm, LAUNCH_CAP))
        if stage == "main_replace":
            k = len(stage_records("main_replace", recs)) + n
            if k > MAX_REPLACEMENTS:
                fail.append("%d replacements with this launch; at most %d in all" % (k, MAX_REPLACEMENTS))
    return fail


def aws_guards(stage, todo):
    """Read-only AWS checks before any spend: the instance role reaches the stage's prefix, and no run of this launch
    already has a status.json in S3 (a launch whose record was lost)."""
    fail = []
    try:
        docs = []
        prof = aws(BUCKET_REGION, "iam", "get-instance-profile", "--instance-profile-name", PROFILE)
        for role in [r["RoleName"] for r in prof["InstanceProfile"]["Roles"]]:
            for name in aws(BUCKET_REGION, "iam", "list-role-policies", "--role-name", role)["PolicyNames"]:
                docs.append(aws(BUCKET_REGION, "iam", "get-role-policy", "--role-name", role,
                                "--policy-name", name)["PolicyDocument"])
            for p in aws(BUCKET_REGION, "iam", "list-attached-role-policies", "--role-name", role)["AttachedPolicies"]:
                arn = p["PolicyArn"]
                ver = aws(BUCKET_REGION, "iam", "get-policy", "--policy-arn", arn)["Policy"]["DefaultVersionId"]
                docs.append(aws(BUCKET_REGION, "iam", "get-policy-version", "--policy-arn", arn,
                                "--version-id", ver)["PolicyVersion"]["Document"])
        arn = "arn:aws:s3:::%s/%s/%s/job.py" % (BUCKET, stage_key(stage), J.run_id(*todo[0]) if todo else "x")
        for action in ("s3:GetObject", "s3:PutObject"):
            if not policy_allows(docs, action, arn):
                fail.append("instance role %s does not allow %s on %s (its policy must cover arn:aws:s3:::%s/%s/*; "
                            "the role has allowed arn:aws:s3:::%s/e1/* since E1)"
                            % (PROFILE, action, arn, BUCKET, S3_ROOT, BUCKET))
    except (RuntimeError, KeyError, TypeError) as e:
        fail.append("could not read the instance role's policies: %s" % str(e)[:300])
    uri = "s3://%s/%s/" % (BUCKET, stage_key(stage))
    rc, out, err = s3("ls", uri, "--recursive")
    if rc not in (0, 1):             # 1: nothing under the prefix yet
        fail.append("could not list %s: %s" % (uri, err.strip()[-300:]))
    have = {ln.split()[-1] for ln in out.splitlines() if ln.strip()} if rc == 0 else set()
    for a, t, s in todo:
        key = "%s/%s/status.json" % (stage_key(stage), J.run_id(a, t, s))
        if key in have:
            fail.append("%s already exists: this run was launched before" % key)
    return fail


def policy_allows(docs, action, arn):
    """An Allow statement covers action on arn and no Deny does (identity policies only; wildcards as fnmatch)."""
    def hits(st, key, value):
        vals = st.get(key, [])
        vals = [vals] if isinstance(vals, str) else vals
        return any(fnmatch.fnmatchcase(value.lower() if key == "Action" else value, v.lower() if key == "Action" else v)
                   for v in vals)
    allow = False
    for doc in docs:
        sts = doc.get("Statement", []) if isinstance(doc, dict) else []
        for st in [sts] if isinstance(sts, dict) else sts:
            if hits(st, "Action", action) and hits(st, "Resource", arn):
                if st.get("Effect") == "Deny":
                    return False
                allow = allow or st.get("Effect") == "Allow"
    return allow


# ------------------------------------------------------------------------------------------
# submit
def launch_batch(stage, k, ud_path, tmp, n_runs, room_usd, room_h, regions, market):
    """One instance through regions x types x subnets, skipping a type whose estimate would cross the ceiling.
    -> (instance, region, ami, type, misses)."""
    misses = []
    for region in regions:
        cfg = REGIONS[region]
        for itype, usd in TYPES:
            if n_runs * PLAN_H * usd > room_usd + 1e-9 or n_runs * PLAN_H > room_h + 1e-9:
                misses.append("%s/%s:OverCeiling" % (region, itype))
                continue
            spec = {"ImageId": cfg["ami"], "InstanceType": itype, "MinCount": 1, "MaxCount": 1,
                    "IamInstanceProfile": {"Name": PROFILE}, "InstanceInitiatedShutdownBehavior": "terminate",
                    "BlockDeviceMappings": [{"DeviceName": "/dev/sda1", "Ebs": {
                        "VolumeSize": VOLUME_GB, "VolumeType": "gp3", "DeleteOnTermination": True}}],
                    "MetadataOptions": {"HttpTokens": "required"},
                    "TagSpecifications": [{"ResourceType": "instance", "Tags": [
                        {"Key": "Name", "Value": "fr-e1b-%s-%d" % (stage, k)},
                        {"Key": "Project", "Value": "flight-recorder-e1b"}, {"Key": "Stage", "Value": stage}]}]}
            if market == "spot":
                spec["InstanceMarketOptions"] = {"MarketType": "spot", "SpotOptions": {
                    "SpotInstanceType": "one-time", "InstanceInterruptionBehavior": "terminate"}}
            quota = False
            for az, subnet in cfg["subnets"].items():
                spec["SubnetId"] = subnet
                sp = tmp / ("batch%d.json" % k)
                sp.write_text(json.dumps(spec), encoding="ascii")
                try:
                    inst = aws(region, "ec2", "run-instances", "--cli-input-json", "file://" + sp.as_posix(),
                               "--user-data", "file://" + ud_path.as_posix())["Instances"][0]
                    return inst, region, cfg["ami"], itype, misses
                except RuntimeError as e:
                    code = _code(e)
                    if code not in CAPACITY_CODES:
                        raise
                    misses.append("%s/%s/%s:%s" % (region, itype, az, code))
                    if code == "VcpuLimitExceeded":      # a regional quota: no other AZ or larger type helps
                        quota = True
                        break
            if quota:
                break
    return None, None, None, None, misses


def launch_record(stage, run, prov, inst, region, ami, itype, market, now):
    """One launch record, build_contract section 3's keys in its order."""
    arm, task, seed = run
    rid = J.run_id(arm, task, seed)
    rec = {"stage": stage, "run_id": rid, "arm": arm, "task": task, "seed": seed}
    rec.update((f, prov[f]) for f in ("git_commit", "package_sha256", "template_sha256", "lock_sha256", "gate_sha256"))
    rec.update({"instance_id": inst["InstanceId"], "region": region, "ami": ami, "instance_type": itype,
                "market": market, "submitted_utc": now, "s3": prefix(stage, rid), "job_sha256": prov["job_sha256"]})
    assert tuple(rec) == RECORD_KEYS
    return rec


def submit(stage, todo_batches, baked, regions, market, hours):
    """Launch the batches in order, each on one instance. Before each, runs/e1b/launches.json is read again: a run
    some other process launched meanwhile is dropped, and the ceiling, LAUNCH_CAP and the replacement allowance are
    checked again against the records as they are now."""
    group = BUDGET_GROUP[stage]
    tmp = Path(tempfile.mkdtemp(prefix="e1b_launch_"))     # no spaces: the aws exe takes file:// paths
    launched = 0
    try:
        for k, batch in enumerate(todo_batches):
            recs = load_launches()
            done = launched_ids(recs)
            meanwhile = [J.run_id(*r) for r in batch if J.run_id(*r) in done]
            if meanwhile:
                print("skip (launched meanwhile, now in %s): %s" % (LAUNCHES.relative_to(ROOT).as_posix(),
                                                                    ", ".join(meanwhile)))
                batch = [r for r in batch if J.run_id(*r) not in done]
            if not batch:
                continue
            why = budget_guard(stage, batch, recs, [], hours)
            if why:
                print("NOT LAUNCHED batch %d (%d runs): %s" % (k, len(batch), "; ".join(why)))
                print("stopping here: %d of %d batches launched" % (launched, len(todo_batches)))
                break
            room_usd, room_h = R.room(recs, group, hours)
            rids = [J.run_id(*r) for r in batch]
            for rid in rids:
                job = baked[rid][0]
                for name in ("job.py", "job.py.lock"):
                    rc, _, err = s3("cp", str(job.parent / name), prefix(stage, rid) + "/" + name)
                    if rc != 0:
                        raise SystemExit("upload failed for %s/%s: %s" % (rid, name, err.strip()[-300:]))
            ud = tmp / ("batch%d.sh" % k)
            with open(ud, "w", encoding="ascii", newline="\n") as f:
                f.write(render(stage, rids))
            inst, region, ami, itype, misses = launch_batch(stage, k, ud, tmp, len(batch), room_usd, room_h,
                                                            regions, market)
            if inst is None:
                print("NOT LAUNCHED batch %d (%d runs): %s" % (k, len(batch), ", ".join(misses)))
                print("stopping here: %d of %d batches launched" % (launched, len(todo_batches)))
                break
            now = _utc()
            new = [launch_record(stage, run, baked[rid][1], inst, region, ami, itype, market, now)
                   for run, rid in zip(batch, rids)]
            _write_json(LAUNCHES, load_launches() + new)
            launched += 1
            print("LAUNCHED batch %d on %s %s %s %s (%s): %s" % (k, inst["InstanceId"], region,
                                                               inst.get("Placement", {}).get("AvailabilityZone", "?"),
                                                               itype, market, ", ".join(rids)))
            if misses:
                print("    capacity misses before it: %s" % ", ".join(misses))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("launch records: %s" % LAUNCHES.relative_to(ROOT).as_posix())
    return 0


def acquire_lock():
    """runs/e1b/launch.lock, created exclusively: one --submit at a time. -> True, or SystemExit when it exists."""
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            held = LOCK.read_text(encoding="ascii", errors="replace").strip()
        except OSError:
            held = "unreadable"
        raise SystemExit("REFUSE: %s exists (%s): another launch_e1b_aws.py --submit is running. Wait for it to "
                         "finish. If none is running (one was killed), delete the file and run this again."
                         % (LOCK.relative_to(ROOT).as_posix(), held or "empty"))
    with os.fdopen(fd, "w", encoding="ascii", newline="\n") as f:
        f.write("pid %d, since %s\n" % (os.getpid(), _utc()))
    return True


def release_lock():
    try:
        LOCK.unlink()
    except FileNotFoundError:
        pass


# ------------------------------------------------------------------------------------------
# status, collect, audit trail
_GMT = re.compile(r"\((\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) GMT\)")     # StateTransitionReason's time


def _parse_utc(s):
    try:
        d = datetime.fromisoformat(str(s).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def instance_info(recs):
    """instance id -> {"state", "launch", "end"}, one describe call per instance (a vanished instance fails a
    multi-id call). state is 'gone' when AWS no longer knows the instance and 'unknown' when the call fails
    otherwise; launch is LaunchTime; end is the time in StateTransitionReason ("User initiated (... GMT)") once the
    instance is terminated or stopped, else None."""
    out = {}
    for iid, region in sorted({(r["instance_id"], r.get("region", BUCKET_REGION)) for r in recs}):
        info = {"state": "unknown", "launch": None, "end": None}
        try:
            d = aws(region, "ec2", "describe-instances", "--instance-ids", iid)
            i = d["Reservations"][0]["Instances"][0]
            info["state"] = i["State"]["Name"]
            info["launch"] = _parse_utc(i.get("LaunchTime"))
            m = _GMT.search(i.get("StateTransitionReason") or "")
            if m and info["state"] in ENDED:
                info["end"] = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        except RuntimeError as e:
            info["state"] = "gone" if _code(e) == "InvalidInstanceID.NotFound" else "unknown"
        except (KeyError, IndexError, TypeError, ValueError):
            info["state"] = "unknown"
        out[iid] = info
    return out


def instance_states(recs):
    """instance id -> state (see instance_info)."""
    return {k: v["state"] for k, v in instance_info(recs).items()}


def measure_hours(recs, info, prev, now=None):
    """{instance_id: {"hours", "final"}} for the instances of recs, from instance_info's answer. Running (or
    pending, stopping, shutting down): launch to now, not final. Terminated or stopped with its time: launch to that
    time, final. Down with no time AWS gives (gone, or no time in StateTransitionReason): the larger of the last
    measurement and launch to now capped at the backstop (runs x per-run timeout + pad, plus boot slack), final; the
    backstop bounds how long the instance could have run. A final measurement is never changed, and an instance whose
    state is unknown keeps its last measurement. No outcome value is read."""
    now = now or datetime.now(timezone.utc)
    runs = {}
    for r in recs:
        if r.get("instance_id"):
            runs.setdefault(r["instance_id"], []).append(r)
    out = {}
    for iid, rs in runs.items():
        old = prev.get(iid) if isinstance(prev.get(iid), dict) else None
        x = info.get(iid)
        if old and old.get("final"):
            out[iid] = old
            continue
        if x is None or x["state"] == "unknown":
            if old:
                out[iid] = old
            continue
        launch = x["launch"] or min((t for t in (_parse_utc(r.get("submitted_utc")) for r in rs) if t), default=None)
        if launch is None:
            continue
        if x["end"] is not None:
            h, final = (x["end"] - launch).total_seconds() / 3600.0, True
        elif x["state"] in ENDED:
            cap = (backstop_min(len(rs)) + BOOT_SLACK_MIN) / 60.0
            h = max(min((now - launch).total_seconds() / 3600.0, cap), float((old or {}).get("hours") or 0.0))
            final = True
        else:
            h, final = (now - launch).total_seconds() / 3600.0, False
        out[iid] = {"hours": round(max(h, 0.0), 4), "final": final}
    return out


def record_hours(recs, info=None):
    """Measure the instances of recs (describe-instances unless info is given, before any lock) and merge them into
    runs/e1b/instance_hours.json with e1b_registry.update_instance_hours, the one writer both launchers use: under an
    exclusive lock file the file is read again, measure_hours runs on that latest content, and the merge keeps every
    other entry and every final one, then an atomic replace. So a --status, --collect, autorun poll or J3 launcher
    writing at the same moment loses nothing. -> the whole {instance_id: {"hours", "final"}} as written."""
    if not recs:
        return R.load_instance_hours(INSTANCE_HOURS)
    info = instance_info(recs) if info is None else info
    return R.update_instance_hours(lambda prev: measure_hours(recs, info, prev), INSTANCE_HOURS)


def read_status(rec):
    """The run's status.json from S3 -> {run_id, step, rc, ts}, None when absent, or "malformed". Only these four
    fields are kept; no other file is read."""
    rc, out, _ = s3("cp", rec["s3"] + "/status.json", "-")
    if rc != 0 or not out.strip():
        return None
    try:
        d = json.loads(out)
    except ValueError:
        return "malformed"
    if not isinstance(d, dict) or d.get("run_id") != rec["run_id"]:
        return "malformed"
    st = {k: d.get(k) for k in ("run_id", "step", "rc", "ts")}
    if not (st["step"] is None or type(st["step"]) is int) or not (st["rc"] is None or type(st["rc"]) in (int, str)):
        return "malformed"
    return st


def status_rows(stage):
    """One row per launched run of the stage: run_id, instance_id, region, state, step, rc, ts, final, lost, and the
    instance's measured hours. Also records the stage's instance hours in runs/e1b/instance_hours.json."""
    recs = stage_records(stage)
    info = instance_info(recs)           # first: a down instance has uploaded everything it ever will
    hours = record_hours(recs, info)
    rows = []
    for r in recs:
        st = read_status(r)
        st = st if isinstance(st, dict) else {"step": None, "rc": None, "ts": None, "malformed": st == "malformed"}
        state = info.get(r["instance_id"], {}).get("state", "unknown")
        final = st["rc"] is not None
        rows.append({"run_id": r["run_id"], "instance_id": r["instance_id"], "region": r.get("region"),
                     "state": state, "step": st["step"], "rc": st["rc"], "ts": st["ts"], "final": final,
                     "lost": (not final) and state in DOWN, "malformed": bool(st.get("malformed")),
                     "hours": hours.get(r["instance_id"])})
    return rows


def summary(rows):
    ok = sum(1 for r in rows if r["final"] and r["rc"] == 0)
    failed = sum(1 for r in rows if r["final"] and r["rc"] != 0)
    lost = sum(1 for r in rows if r["lost"])
    return ("%d launched: finished rc=0 %d, finished rc!=0 %d, lost (instance down, no rc) %d, still running or "
            "queued %d" % (len(rows), ok, failed, lost, len(rows) - ok - failed - lost))


def print_status(stage):
    rows = status_rows(stage)
    if not rows:
        print("no launched run in stage %s (%s)" % (stage, LAUNCHES.relative_to(ROOT).as_posix()))
        return 0
    print("%-36s %-20s %-13s %-9s %-12s %s" % ("run", "instance", "state", "step", "rc", "ts"))
    for r in rows:
        step = "-" if r["step"] is None else "%d/%s" % (r["step"] + 1, J_STEPS)
        rc = "LOST" if r["lost"] else ("-" if r["rc"] is None else str(r["rc"]))
        bad = "  (status.json malformed)" if r["malformed"] else ""
        print("%-36s %-20s %-13s %-9s %-12s %s%s" % (r["run_id"], r["instance_id"], r["state"], step, rc,
                                                     r["ts"] or "-", bad))
    print(summary(rows))
    seen = {}
    for r in rows:
        seen.setdefault(r["instance_id"], r)
    for iid, r in sorted(seen.items()):
        h = r["hours"] or {}
        print("instance %-20s %-13s %s" % (iid, r["state"], "%.2f h %s" % (h["hours"], "(final)" if h.get("final")
                                                                         else "so far") if h else "hours unknown"))
    print("instance hours: %s" % INSTANCE_HOURS.relative_to(ROOT).as_posix())
    return 0


def _download(uri, dest):
    rc, _, _ = s3("cp", uri, str(dest))
    return rc == 0 and Path(dest).is_file()


def collect(stage):
    """Refuses until every launched run of the stage is final or lost; then downloads to the build_contract paths and
    writes runs/e1b/<stage>/MANIFEST.sha256. Prints names and ok/missing only."""
    rows = status_rows(stage)
    if not rows:
        raise SystemExit("REFUSE: no launched run in stage %s" % stage)
    waiting = [r for r in rows if not (r["final"] or r["lost"])]
    if waiting:
        raise SystemExit("REFUSE: %d of %d launched runs of %s have no final rc yet (%s); no result or log is read "
                         "before every run has one (pre-registration section 8)"
                         % (len(waiting), len(rows), stage, ", ".join(r["run_id"] for r in waiting[:6])
                            + (", ..." if len(waiting) > 6 else "")))
    rdir, sdir = result_dir(stage), RUNS_DIR / stage
    for d in (rdir, sdir / "logs", sdir / "status"):
        d.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="e1b_collect_"))
    conflicts, collected = [], []
    try:
        for r in rows:
            rid, uri = r["run_id"], prefix(stage, r["run_id"])
            got = {}
            for name, dest in (("result.json", rdir / (rid + ".json")), ("job.log", sdir / "logs" / (rid + ".log")),
                               ("boot.log", sdir / "logs" / (rid + ".boot.log")),
                               ("status.json", sdir / "status" / (rid + ".json"))):
                t = tmp / ("%s.%s" % (rid, name))
                got[name] = _download(uri + "/" + name, t)
                if not got[name]:
                    continue
                if name == "result.json":
                    if dest.exists() and dest.read_bytes() != t.read_bytes():
                        conflicts.append(rid)          # a collected result never changes silently
                        got[name] = "CONFLICT"
                        continue
                    collected.append(dest)
                shutil.copyfile(t, dest)
            why = "LOST: instance %s, no final rc" % r["state"] if r["lost"] else "rc=%s" % r["rc"]
            print("%-36s %-28s %s" % (rid, why, "  ".join("%s %s" % (n, {True: "ok", False: "missing"}.get(v, v))
                                                         for n, v in got.items())))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if conflicts:
        raise SystemExit("REFUSE: collected result file(s) differ from S3, left unchanged, no manifest written: %s"
                         % ", ".join(conflicts))
    write_manifest(sdir / "MANIFEST.sha256", collected)
    print("%d result file(s) in %s; manifest %s" % (len(collected), rdir.relative_to(ROOT).as_posix(),
                                                    (sdir / "MANIFEST.sha256").relative_to(ROOT).as_posix()))
    print(summary(rows))
    return 0


def write_manifest(path, files):
    """`<sha256>  <file name>` per result file, sorted by name, LF. A line already present must still hold (a
    collected file never changes); new files add lines."""
    lines = {}
    if path.exists():
        for ln in path.read_text(encoding="ascii").splitlines():
            if ln.strip():
                h, name = ln.split("  ", 1)
                lines[name] = h
    for f in files:
        h = hashlib.sha256(Path(f).read_bytes()).hexdigest()
        if lines.get(f.name, h) != h:
            raise SystemExit("REFUSE: %s changed since %s recorded it" % (f.name, path.name))
        lines[f.name] = h
    with open(path, "w", encoding="ascii", newline="\n") as fh:
        fh.write("".join("%s  %s\n" % (lines[n], n) for n in sorted(lines)))


def audit_trail(stage):
    """CloudTrail management events of every recorded instance of the stage (lookup-events by resource name, in the
    instance's region; CloudTrail keeps 90 days). Read-only. The raw events (caller identity ARN, user name, access
    key id, source IP) go only to the gitignored paper/freeze/e1b_<stage>/cloudtrail.json, and the run refuses
    unless git ignores that path. A redacted copy, each event cut to EventId, EventName, EventTime, EventSource and
    Resources, goes to runs/e1b/audit/<stage>_cloudtrail.json, never to a result folder (runs/e1b/main or pilot).
    The earliest RunInstances time (section 13's "First launch time") is printed and kept in the redacted copy."""
    recs = stage_records(stage)
    if not recs:
        raise SystemExit("no launched run in stage %s" % stage)
    raw_rel, red_rel = AUDIT_RAW % stage, AUDIT_REDACTED % stage
    rc, _ = _git("check-ignore", "-q", raw_rel)          # a tracked file is not ignored: refused too
    if rc != 0:
        raise SystemExit("REFUSE: %s is not ignored by git; the raw CloudTrail events hold the caller's identity and "
                         "IP address and must stay out of the public repo (add paper/ to .gitignore)" % raw_rel)
    exported = _utc()
    raw = {"exported_utc": exported, "stage": stage, "instances": {}}
    red = {"exported_utc": exported, "stage": stage, "fields": list(AUDIT_KEYS), "earliest_run_instances": None,
           "instances": {}}
    first = None
    for iid, region in sorted({(r["instance_id"], r.get("region", BUCKET_REGION)) for r in recs}):
        d = aws(region, "cloudtrail", "lookup-events", "--lookup-attributes",
                "AttributeKey=ResourceName,AttributeValue=" + iid)
        ev = (d or {}).get("Events", [])
        runs = sorted(r["run_id"] for r in recs if r["instance_id"] == iid)
        raw["instances"][iid] = {"region": region, "runs": runs, "events": ev}
        red["instances"][iid] = {"region": region, "runs": runs,
                                 "events": [{k: e[k] for k in AUDIT_KEYS if k in e} for e in ev]}
        for e in ev:
            t = _parse_utc(e.get("EventTime")) if e.get("EventName") == "RunInstances" else None
            if t is not None and (first is None or t < first[0]):
                first = (t, str(e["EventTime"]))
        print("%-20s %-9s %d event(s)" % (iid, region, len(ev)))
    red["earliest_run_instances"] = first[1] if first else None
    _write_json(ROOT / raw_rel, raw)
    _write_json(ROOT / red_rel, red)
    print("wrote %s (raw, gitignored, local only) and %s (redacted: %s)" % (raw_rel, red_rel, ", ".join(AUDIT_KEYS)))
    print("earliest RunInstances event: %s" % (first[1] if first else "none found"))
    return 0


# ------------------------------------------------------------------------------------------
def _integer(opts, name, default, lo):
    try:
        v = int(opts.get(name, default))
    except ValueError:
        v = lo - 1
    if v < lo:
        raise SystemExit("%s must be an integer >= %d" % (name, lo))
    return v


def main(argv):
    opts = {a.split("=", 1)[0]: (a.split("=", 1)[1] if "=" in a else True) for a in argv if a.startswith("--")}
    pos = [a for a in argv if not a.startswith("--")]
    known = {"--submit", "--commit", "--status", "--collect", "--audit-trail", "--rerun", "--spot",
             "--region", "--runs-per-instance", "--max-runs", "--show-userdata"}
    if len(pos) != 1 or pos[0] not in STAGES or set(opts) - known:
        raise SystemExit("usage: python scripts/launch_e1b_aws.py {%s} [options]; unknown: %s"
                         % ("|".join(STAGES), ", ".join(sorted(set(opts) - known)) or pos[1:] or "-"))
    stage = pos[0]
    valued = ("--commit", "--rerun", "--region", "--runs-per-instance", "--max-runs")
    flags = ("--submit", "--status", "--collect", "--audit-trail", "--spot", "--show-userdata")
    bad = [o for o in opts if (o in valued) != isinstance(opts[o], str) or (o in flags and opts[o] is not True)]
    if bad:
        raise SystemExit("option(s) %s: give %s as --name=value and the others without a value"
                         % (", ".join(bad), ", ".join(valued)))
    if "--status" in opts:
        return print_status(stage)
    if "--collect" in opts:
        return collect(stage)
    if "--audit-trail" in opts:
        return audit_trail(stage)
    if "--submit" not in opts:
        return launch(stage, opts)
    acquire_lock()                       # before launches.json is read: the whole submit is one at a time
    try:
        return launch(stage, opts)
    finally:
        release_lock()


def launch(stage, opts):
    """The dry run, or with --submit the launch (main() holds runs/e1b/launch.lock around it)."""
    submitting = "--submit" in opts
    rerun = _integer(opts, "--rerun", 0, 0)
    if rerun > 2 or (rerun and stage != "pilot"):
        raise SystemExit("--rerun=1 or 2, pilot only (seed 298, then 297)")
    commit = opts.get("--commit")
    regions = [opts["--region"]] if "--region" in opts else list(REGION_ORDER)
    if any(r not in REGIONS for r in regions):
        raise SystemExit("--region must be one of %s" % ", ".join(REGIONS))
    market = "spot" if "--spot" in opts else "ondemand"
    k = _integer(opts, "--runs-per-instance", RUNS_PER_INSTANCE[stage], 1)

    recs = load_launches()
    done = launched_ids(recs)
    rows = plan(stage, rerun)
    if rerun:
        prev = J.STAGE_SEEDS["pilot"][rerun - 1]
        if not all(J.run_id(a, t, prev) in done for a in J.ARMS for t in J.TASKS):
            raise SystemExit("REFUSE: --rerun=%d (seed %d) before every pilot run at seed %d was launched"
                             % (rerun, J.STAGE_SEEDS["pilot"][rerun], prev))
    todo = [r for r in rows if J.run_id(*r) not in done]
    for r in rows:
        if J.run_id(*r) in done:
            print("skip (already launched): %s" % J.run_id(*r))
    if "--max-runs" in opts:
        todo = todo[:_integer(opts, "--max-runs", 0, 1)]
    todo_batches = batches(todo, k)
    print("E1b %s%s: %d registered run(s), %d to launch in %d instance(s) of up to %d, %s, regions %s"
          % (stage, " rerun %d" % rerun if rerun else "", len(rows), len(todo), len(todo_batches), k, market,
             ", ".join(regions)))
    if not todo:
        print("NOTHING TO LAUNCH")
        return 0

    fail, note = guards(stage, commit, todo, recs, submitting)
    sha = resolve(commit or "HEAD")
    with tempfile.TemporaryDirectory(prefix="e1b_bake_") as td:
        baked = {}
        if sha:
            for arm, task, seed in todo:
                rid = J.run_id(arm, task, seed)
                try:
                    job, prov = J.bake(arm, task, seed, stage, sha, td)
                except (ValueError, RuntimeError) as e:
                    fail.append("bake %s: %s" % (rid, e))
                    break
                js = hashlib.sha256(Path(job).read_bytes()).hexdigest()     # the bytes that are uploaded
                if prov.get("job_sha256", js) != js:
                    fail.append("bake %s: its job_sha256 %s is not the SHA-256 of the job.py it wrote (%s)"
                                % (rid, prov.get("job_sha256"), js))
                    break
                baked[rid] = (Path(job), dict(prov, job_sha256=js))
        print("\n%-5s %-36s %s" % ("batch", "run", "s3 prefix"))
        for b, batch in enumerate(todo_batches):
            for arm, task, seed in batch:
                print("%-5d %-36s %s" % (b, J.run_id(arm, task, seed), prefix(stage, J.run_id(arm, task, seed))))
        if baked:
            p = next(iter(baked.values()))[1]
            print("\nbaked %d job(s) from %s: package %s, template %s, lock %s, gate %s; job.py %s (%s)"
                  % (len(baked), p["git_commit"][:12], p["package_sha256"][:16], p["template_sha256"][:16],
                     p["lock_sha256"][:16], p["gate_sha256"][:16], p["job_sha256"][:16], p["run_id"]))
        print("\nper instance: per-run timeout %d min, watchdog %d min without a status.json update, backstop "
              "shutdown at runs x %d + %d min, terminate on shutdown, %d GB gp3" % (
                  PER_RUN_MIN, STALL_MIN, PER_RUN_MIN, BACKSTOP_PAD_MIN, VOLUME_GB))
        est = {t: sum(len(b) for b in todo_batches) * PLAN_H * u for t, u in TYPES}
        worst = sum(backstop_min(len(b)) / 60.0 for b in todo_batches) * max(PRICE.values())
        print("estimate (planning, %.3f h per run): %s; worst case (every instance to its backstop at %s): USD %.2f"
              % (PLAN_H, ", ".join("USD %.2f at %s" % (v, t) for t, v in est.items()), TYPES[-1][0], worst))
        for n in note:
            print("  note: %s" % n)
        if "--show-userdata" in opts:
            print("----- user data, batch 0 -----\n%s-----" % render(stage, [J.run_id(*r) for r in todo_batches[0]]))
        if not submitting:
            for f in fail:
                print("  --submit would REFUSE: %s" % f)
            print("\nDRY RUN: nothing uploaded or launched, no launch record written. --submit (with --commit) spends.")
            return 0
        hours = None
        if not fail:
            fail = aws_guards(stage, todo)
        if not fail:                     # measure the group's instances now (read-only), then the ceiling again
            group = BUDGET_GROUP[stage]
            hours = record_hours([r for r in recs if r.get("instance_id")
                                  and BUDGET_GROUP.get(r.get("stage")) == group])
            measured = []
            fail = budget_guard(stage, todo, recs, measured, hours)
            for n in measured:
                print("  measured before launch: %s" % n)
        if fail:
            raise SystemExit("\n".join("REFUSE: %s" % f for f in fail))
        return submit(stage, todo_batches, baked, regions, market, hours)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
