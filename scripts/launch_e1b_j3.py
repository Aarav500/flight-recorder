"""E1b J3 scoring on AWS EC2: one self-terminating g6e (1x L40S) instance runs the J3 job. ASCII only.

WHAT
  Pre-registration section 2 (branch A), spec 3.6 and section 12's "J3 scoring" row: the job baked by
  scripts/e1b_j3_job.py scores every stored completion of every used E1b run with Qwen/Qwen2.5-7B-Instruct
  (a09a3545, bfloat16). This file builds the data (load_e1b's used runs, after the pins and the data manifest are
  checked), bakes the job from a registered commit, uploads job.py, job.py.lock and data.jsonl.gz to
  s3://<bucket>/e1/e1b/j3/<run_id>/ and starts one instance. Run ids are e1b_j3_qwen25_7b_a<k>, k = 1 for the first
  launch. It follows scripts/launch_e1_rescore.py (frozen; patterns rewritten, nothing imported from it) and imports
  the freeze-record, commit, receipt and cell (reader_problem, approval_problem) helpers of scripts/launch_e1b_aws.py
  and the spend rule (committed, room; budget group "j3", its ceilings and planning hours) and the measured-hours
  writer of experiments/e1b_registry.py, so the two launchers apply one rule each, from one copy.
  Launch records go to runs/e1b/j3/launches.json, a file of their own: their keys differ from build_contract section
  3's, and every reader of main and pilot launch records (launch_e1b_aws.load_launches, e1b_primary.load_e1b, the
  autorun, the pilot gate) reads runs/e1b/launches.json by its exact path, while the result readers glob only e1b_*_s<seed>.json in
  runs/e1b/main and runs/e1b/pilot, so no J3 file is taken for a main or pilot record.

SPEND GUARDS
  The default is a dry run: build the data, bake, price, print the plan and every guard. Nothing is uploaded or
  launched and no AWS call is made. --submit spends, and refuses unless all of these hold:
    * --commit names the commit the job is baked from, e1b_registry.commit_registered accepts it against HEAD's
      pre-registration (section 13's "Commit hash" or a section 14 "Commit"), and every file the launch runs is
      identical in the working tree and at that commit: the J3 template, this launcher, the files it imports
      (scripts/e1b_job.py, scripts/launch_e1b_aws.py, experiments/e1b_registry.py), the lock, the analysis that
      selects the runs (experiments/e1b_primary.py) and the job's package import closure (spec 7.1);
    * experiments/e1b_freeze_manifest.txt is committed at HEAD, lists every file of section 13's FILES, and each
      SHA-256 line matches its blob at the freeze commit (LF-normalized); a timestamp receipt committed at HEAD
      stamps the manifest's SHA-256 (section 13; spec 7.5);
    * section 13's human reader and section 12's "J3 scoring" approval are filled at HEAD, by the main launcher's
      rule (launch_e1b_aws.reader_problem and approval_problem): a valid YYYY-MM-DD date, and "section" in the
      reader cell or Aarav's words in straight or curly quotes in the approval cell; a cell saying "not", "TBD",
      "awaiting", "pending" or "none yet" is refused;
    * section 12's J3 ceilings are e1b_registry.CEILINGS["j3"] (8 GPU-hours, USD 15), and they hold for the worst
      case: the committed J3 spend (e1b_registry.committed(records, "j3"): per instance the larger of J3_PLAN_H and
      its measured hours, at its type's list price) plus this instance running to its backstop. The backstop is the
      smallest of 8
      hours and the room left in GPU-hours and in USD at the type's price; a type whose room gives less than
      4 hours (the E1 rescoring's backstop) is skipped;
    * the pins verify and the data manifest holds (the data build), and the used runs are final: when the
      registered results file (experiments/e1b_primary_results.json) exists, its used tags and data manifest
      SHA-256 equal the data's; otherwise no replacement is needed and no main run is still to launch;
    * no earlier J3 instance is still up, runs/e1b/j3/qwen25_7b.json has not been collected, no other launcher
      holds runs/e1b/j3/launch.lock, and the read-only AWS checks pass: the instance role allows s3:GetObject and
      s3:PutObject under e1/e1b/j3/<run_id>/, and nothing is stored there yet.
  The uploads come before run-instances, so a launch that finds no capacity (or fails otherwise) writes its record
  without an instance: the run id is spent, the next launch uses a fresh prefix, and nothing counts towards the
  ceiling.

INSTANCE (user data)
  The AMIs, subnets and role of scripts/launch_e1b_aws.py; g6e only, on-demand, 120 GB gp3,
  InstanceInitiatedShutdownBehavior=terminate, IMDSv2. At boot: `shutdown -h +<backstop>` first, then uv 0.12.23
  with its wheel SHA-256 (--require-hashes), then job.py, job.py.lock and data.jsonl.gz from S3, then
      E1B_J3_STATUS_PATH=<dir>/status.json timeout <backstop - 20 min> python3 -m uv run --no-config --locked
          --python 3.12.15 --managed-python job.py > job.log 2>&1
  A watchdog stops the job when status.json has not changed for 45 minutes (rc "stalled"); status.json and
  partial.json are pushed every 5 minutes. At the end: result.json, partial.json, job.log and boot.log are uploaded,
  then status.json with the final rc, last. On any exit (a signal, the backstop) the trap uploads what exists with rc
  "interrupted" (or "no-job", "boot-failed") and powers off, which terminates the instance. The instance reads
  neither job.log nor partial.json nor result.json, so it prints no score.

MONITORING AND COLLECTION
  --status reads only status.json ({run_id, phase, n_scored, n_total, rc, ts}: counts) and the instance states, and
  records measured hours in runs/e1b/instance_hours.json through e1b_registry.update_instance_hours (an exclusive
  lock file around the read-merge-write, shared with the main launcher). --collect refuses until the latest launch
  has a final rc or its instance is down, then downloads result.json (partial.json only with --accept-partial) to
  runs/e1b/j3/qwen25_7b.json, never replacing a collected file with different bytes, plus job.log, boot.log and
  status.json under runs/e1b/j3/, and writes runs/e1b/j3/MANIFEST.sha256. It refuses a file whose judge, revision
  or dtype is not the registered J3's and the launch record's, whose measured Hub snapshots (snapshots.target, the
  validation judge's) or measured dtype differ from the registered revisions and dtype, or whose data SHA-256,
  job_file_sha256 or run id is not the launch record's (result_problems), and prints names, ok/missing and counts
  only, never a score.

    python scripts/launch_e1b_j3.py                         # dry run at HEAD: data, bake, price, guards
    python scripts/launch_e1b_j3.py --commit=<sha>          # dry run at a registered commit
    python scripts/launch_e1b_j3.py --commit=<sha> --submit # SPEND (one on-demand instance)
    python scripts/launch_e1b_j3.py --status
    python scripts/launch_e1b_j3.py --collect [--accept-partial]
  Options: --region=us-east-1|us-east-2  --show-userdata
"""
from __future__ import annotations

import ast
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
for _p in (ROOT / "experiments", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import e1b_j3_job as JJ  # noqa: E402  bake, prep_data, read_header, J3, UV_J3, TEMPLATE_PATH, LOCK_PATH, RUN_ID_RE
import e1b_job as J  # noqa: E402  RUN_ARGS, UV_VERSION, UV_WHEEL_SHA256
import e1b_registry as REG  # noqa: E402  registered commits; CEILINGS, J3_PLAN_H, committed, room; the hours writer
import launch_e1b_aws as L  # noqa: E402  freeze_record, reader_problem, approval_problem, receipts, git, aws, s3

STAGE = "j3"
BUCKET, BUCKET_REGION, PROFILE = L.BUCKET, L.BUCKET_REGION, L.PROFILE
S3_ROOT = "e1/e1b"                   # build contract section 8: every E1b key is e1/e1b/...
J3_DIR = ROOT / "runs" / "e1b" / "j3"
LAUNCHES = J3_DIR / "launches.json"
RESULT = J3_DIR / ("%s.json" % JJ.JUDGE_SHORT)            # what experiments/e1b_primary.load_j3 reads
DATA_MANIFEST = J3_DIR / "MANIFEST.sha256"
LOCKFILE = J3_DIR / "launch.lock"
RESULTS_K0 = ROOT / "experiments" / "e1b_primary_results.json"
LAUNCHER = "scripts/launch_e1b_j3.py"
REGISTRY = "experiments/e1b_registry.py"
PRIMARY = "experiments/e1b_primary.py"
LAUNCHER_MAIN = "scripts/launch_e1b_aws.py"
GROUP = "j3"                         # e1b_registry's budget group of this stage (BUDGET_GROUP["j3"])
# section 12's "J3 scoring" row (its first cell starts with ROW): 8 GPU-hours, USD 15. One copy, in the registry.
ROW, CEIL_H, CEIL_USD = REG.CEILINGS[GROUP]
# Planning hours of the J3 instance (e1b_registry.J3_PLAN_H: section 12's USD 4.4 estimate at g6e.xlarge's price).
# The spend rule (e1b_registry.committed) counts the larger of this and the measured hours.
PLAN_H = REG.J3_PLAN_H
BACKSTOP_MAX_MIN = 480               # 8 h, the stage's GPU-hour ceiling (worst case USD 14.89 at g6e.xlarge)
BACKSTOP_MIN_MIN = 240               # the E1 rescoring's backstop (it scored 153,600 completions in 0.9 h)
JOB_PAD_MIN = 20                     # job timeout = backstop - this: time to write result.json and upload
KILL_AFTER_S = 120
STALL_MIN = 45
TICK_S, PUSH_S = 60, 300
VOLUME_GB = 120                      # AMI + uv cache + the 7B and 0.5B checkpoints (~16 GB) + the data
WORKDIR = "/opt/e1b_j3"
DOWN = ("shutting-down", "terminated", "stopping", "stopped", "gone")
RECORD_KEYS = ("stage", "run_id", "judge", "revision", "dtype", "git_commit", "package_sha256", "template_sha256",
               "lock_sha256", "job_sha256", "data_sha256", "n_completions", "n_runs", "run_ids",
               "data_manifest_sha256", "instance_id", "region", "ami", "instance_type", "market", "submitted_utc",
               "s3", "backstop_min", "job_timeout_min", "plan_h", "worst_usd")

USER_DATA = r"""#!/bin/bash
# E1b J3 scoring instance, written by scripts/launch_e1b_j3.py: one job, then power off. It never reads job.log,
# partial.json or result.json, so it prints no score.
set -u
shutdown -h +__BACKSTOP_MIN__      # backstop first: whatever fails below, the instance powers off (and terminates)
export AWS_DEFAULT_REGION=__BUCKET_REGION__
export HOME="${HOME:-/root}"
W="__WORKDIR__"
R="__RUN_ID__"
S3="__S3_PREFIX__"
mkdir -p "$W/run" && cd "$W" || exit 1
exec > "$W/boot.log" 2>&1
echo "boot $(date -u +%FT%TZ); backstop +__BACKSTOP_MIN__ min; job timeout __JOB_MIN__ min; run $R"
JOB=""; WHY=boot-failed; DONE=0

put() { aws s3 cp "$1" "$2" --only-show-errors || echo "upload failed: $2"; }

set_status() {   # set_status FILE RUN_ID RC: keeps the job's phase and counts; RC null while running
python3 - "$1" "$2" "$3" <<'PY'
import json, os, sys, time
path, rid, rc = sys.argv[1:4]
keep = {"phase": None, "n_scored": None, "n_total": None}
try:
    with open(path) as f:
        d = json.load(f)
    if d.get("run_id") == rid:
        keep["phase"] = d.get("phase") if isinstance(d.get("phase"), str) else None
        for k in ("n_scored", "n_total"):
            keep[k] = d.get(k) if type(d.get(k)) is int else None
except Exception:
    pass
rc = None if rc == "null" else int(rc) if rc.lstrip("-").isdigit() else rc
with open(path + ".tmp", "w") as f:
    json.dump({"run_id": rid, "phase": keep["phase"], "n_scored": keep["n_scored"], "n_total": keep["n_total"],
               "rc": rc, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f)
os.replace(path + ".tmp", path)
PY
}

finish() {       # on any exit: what exists is uploaded, a final rc if none was written, boot.log, power off
  trap - EXIT TERM INT HUP
  if [ "$DONE" = 0 ]; then
    if [ -n "$JOB" ]; then
      kill -TERM "$JOB" 2>/dev/null
      for _ in $(seq 1 90); do kill -0 "$JOB" 2>/dev/null || break; sleep 1; done
    fi
    for F in result.json partial.json job.log; do [ -f "$W/run/$F" ] && put "$W/run/$F" "$S3/$F"; done
    set_status "$W/run/status.json" "$R" "$WHY" && put "$W/run/status.json" "$S3/status.json"
    echo "final rc $WHY written on exit"
  fi
  echo "exit $(date -u +%FT%TZ)"
  put "$W/boot.log" "$S3/boot.log"
  shutdown -h now
}
trap finish EXIT
trap 'echo "signal $(date -u +%FT%TZ)"; exit 143' TERM INT HUP

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo "uv==__UV_VERSION__ --hash=sha256:__UV_SHA256__" > "$W/uv-req.txt"
install_uv() { python3 -m pip install -q --only-binary :all: --require-hashes -r "$W/uv-req.txt"; }
install_uv || { apt-get update -q && apt-get install -yq python3-pip && install_uv; } || { echo "uv install failed"; exit 1; }
python3 -m uv --version
for F in job.py job.py.lock data.jsonl.gz; do
  aws s3 cp "$S3/$F" "$W/run/$F" --only-show-errors || { echo "$F not found"; WHY=no-job; exit 1; }
done
set_status "$W/run/status.json" "$R" null && put "$W/run/status.json" "$S3/status.json"
export HF_HOME="$W/hf" UV_CACHE_DIR="$W/uvcache" PYTHONUNBUFFERED=1
WHY=interrupted
echo "$R: start $(date -u +%FT%TZ)"
cd "$W/run"
E1B_J3_STATUS_PATH="$W/run/status.json" timeout --kill-after=__KILL_AFTER_S__ __JOB_MIN__m python3 -m __RUN_ARGS__ > "$W/run/job.log" 2>&1 &
JOB=$!
cd "$W"
STALLED=0; LAST=$(date +%s)
while kill -0 "$JOB" 2>/dev/null; do
  sleep __TICK_S__ & wait $!
  NOW=$(date +%s)
  M=$(stat -c %Y "$W/run/status.json" 2>/dev/null || echo "$NOW")
  if [ "$STALLED" = 0 ] && [ $((NOW - M)) -gt __STALL_S__ ]; then
    echo "watchdog: status.json unchanged for $((NOW - M)) s, stopping the job"
    STALLED=1; kill -TERM "$JOB" 2>/dev/null
  fi
  if [ $((NOW - LAST)) -ge __PUSH_S__ ]; then
    put "$W/run/status.json" "$S3/status.json"
    [ -f "$W/run/partial.json" ] && put "$W/run/partial.json" "$S3/partial.json"
    LAST=$NOW
  fi
done
wait "$JOB"; RC=$?
[ "$STALLED" = 1 ] && RC=stalled
JOB=""
echo "$R: end rc=$RC $(date -u +%FT%TZ)"
for F in result.json partial.json job.log; do [ -f "$W/run/$F" ] && put "$W/run/$F" "$S3/$F"; done
put "$W/boot.log" "$S3/boot.log"
set_status "$W/run/status.json" "$R" "$RC" && put "$W/run/status.json" "$S3/status.json"
DONE=1
echo "done $(date -u +%FT%TZ)"
"""
_UD_KEYS = ("__BACKSTOP_MIN__", "__BUCKET_REGION__", "__WORKDIR__", "__RUN_ID__", "__S3_PREFIX__", "__UV_VERSION__",
            "__UV_SHA256__", "__KILL_AFTER_S__", "__JOB_MIN__", "__RUN_ARGS__", "__TICK_S__", "__STALL_S__",
            "__PUSH_S__")


# ------------------------------------------------------------------------------------------
# small helpers
def _utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rel(p):
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(p).name


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


def _code(err):
    msg = str(err)
    return msg.split("An error occurred (")[-1].split(")")[0] if "An error occurred (" in msg else ""


def prefix(rid):
    return "s3://%s/%s/%s/%s" % (BUCKET, S3_ROOT, STAGE, rid)


def load_launches():
    if not LAUNCHES.exists():
        return []
    recs = json.loads(LAUNCHES.read_text(encoding="utf-8"))
    if not isinstance(recs, list):
        raise SystemExit("REFUSE: %s is not a list of launch records" % _rel(LAUNCHES))
    return recs


def next_run_id(recs):
    return "e1b_j3_%s_a%d" % (JJ.JUDGE_SHORT, len(recs) + 1)


def committed(recs, hours=None):
    """(USD, GPU-hours) committed by J3 launches: e1b_registry.committed(recs, "j3", hours), the rule the main
    launcher and the analysis use, one entry per launched instance, the larger of J3_PLAN_H and its measured hours
    (runs/e1b/instance_hours.json), at its type's list price (an unknown type at the dearest)."""
    return REG.committed(recs, GROUP, hours)


def room(recs, hours=None):
    """(USD, GPU-hours) left under section 12's J3 ceilings: e1b_registry.room(recs, "j3", hours)."""
    return REG.room(recs, GROUP, hours)


def backstop_for(itype, room_usd, room_h):
    """Backstop minutes for one instance of itype: the smallest of BACKSTOP_MAX_MIN and the room in GPU-hours and in
    USD at the type's price, whole minutes; None when that is under BACKSTOP_MIN_MIN."""
    m = int(min(BACKSTOP_MAX_MIN, room_h * 60.0 + 1e-9, room_usd / REG.PRICE[itype] * 60.0 + 1e-9))
    return m if m >= BACKSTOP_MIN_MIN else None


def job_closure(rev):
    """Repo files the J3 job imports at rev: its flightrecorder imports followed through the package (absolute and
    relative, function-level included), plus each package __init__.py on the way."""
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

    found, todo = set(), list(imports(ast.parse(JJ.UV_J3.replace("__N_ROWS__", "0")), "__job__", False))
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
            src = L.blob(rev, p)
            if src is not None:
                todo.extend(imports(ast.parse(src), sub, p.endswith("__init__.py")))
    return sorted(found)


# ------------------------------------------------------------------------------------------
# guards
def guards(commit, submit, recs):
    """-> (refusals, notes, sha). Git and local files only; the data guards and the AWS checks come later."""
    fail, note = [], []
    if commit is None:
        (fail if submit else note).append("--commit is required with --submit (this dry run checks against HEAD)")
    sha = L.resolve(commit or "HEAD")
    if sha is None:
        fail.append("commit %r is unknown" % (commit or "HEAD"))
        return fail, note, None
    note.append("commit %s%s" % (sha, "" if commit else " (HEAD)"))
    closure = job_closure(sha)
    note.append("job import closure at %s: %s" % (sha[:12], ", ".join(closure) or "NONE FOUND"))
    if not closure:
        fail.append("no flightrecorder module of the job's import closure is committed at %s" % sha[:12])
    for p in sorted(set(closure) | {JJ.TEMPLATE_PATH, LAUNCHER, J.TEMPLATE_PATH, LAUNCHER_MAIN, REGISTRY, PRIMARY,
                                    JJ.LOCK_PATH}):
        why = L.same_as_commit(p, sha)
        if why:
            fail.append("%s %s" % (p, why))
    pre = L.blob("HEAD", L.PREREG)
    text = pre.decode("utf-8", "replace") if pre is not None else None
    if text is None:
        fail.append("%s is not committed at HEAD" % L.PREREG)
        return fail, note, sha
    tokens = REG.registered_commits(text)
    if REG.commit_registered(sha, tokens):
        note.append("commit %s is registered (section 13 or 14 of HEAD's pre-registration)" % sha[:12])
    else:
        fail.append("commit %s is neither section 13's freeze commit nor a section 14 commit at HEAD (registered: %s)"
                    % (sha[:12], ", ".join(tokens) or "none"))
    fail += freeze_guards(text, note)
    rec = L.freeze_record(text)
    # the cells by the main launcher's own rule (one copy: launch_e1b_aws.reader_problem and approval_problem)
    why = L.reader_problem(rec["reader"])
    if why:
        fail.append("section 13 'Human reader before the freeze' %s at HEAD: %r (name, YYYY-MM-DD date, sections "
                    "read)" % (why, (rec["reader"] or "").strip()))
    else:
        note.append("human reader: %s" % rec["reader"].strip())
    row = next((v for k, v in rec["rows"].items() if k.startswith(ROW)), None)
    if row is None:
        fail.append("section 12 has no '%s' row" % ROW)
    else:
        why = L.approval_problem(row["approved"])
        if why:
            fail.append("section 12 '%s' row: 'Approved by Aarav' %s at HEAD: %r (his quoted words and a YYYY-MM-DD "
                        "date)" % (ROW, why, (row["approved"] or "").strip()))
        else:
            note.append("section 12 %s approval: %s" % (ROW, row["approved"].strip()))
        if row["gpu_h"] != CEIL_H or row["usd"] != CEIL_USD:
            fail.append("section 12 '%s' ceilings (%s GPU-h, USD %s) differ from this launcher's (%s, %s)"
                        % (ROW, row["gpu_h"], row["usd"], CEIL_H, CEIL_USD))
    if RESULT.exists():
        fail.append("%s is already collected; a new J3 launch needs a section 14 row and the file moved aside first"
                    % _rel(RESULT))
    return fail, note, sha


def freeze_guards(text, note):
    """The freeze manifest committed at HEAD against the freeze commit's blobs (LF-normalized), section 13's FILES
    all listed, and a timestamp receipt committed at HEAD that stamps it."""
    fail = []
    fc = REG.freeze_commit(text)
    fsha = L.resolve(fc) if fc else None
    if fsha is None:
        fail.append("section 13's 'Commit hash' at HEAD names no commit of this repository (%r)" % fc)
    man = L.blob("HEAD", L.MANIFEST)
    if man is None:
        fail.append("%s is not committed at HEAD (section 13: committed before any launch)" % L.MANIFEST)
        return fail
    if L.same_as_commit(L.MANIFEST, "HEAD"):
        fail.append("%s in the working tree differs from HEAD" % L.MANIFEST)
    listed, bad = {}, []
    for i, ln in enumerate(man.decode("utf-8", "replace").split("\n")):
        if not ln.strip():
            continue
        m = re.fullmatch(r"([0-9a-f]{64})  (\S+)", ln.rstrip("\r"))
        if not m:
            bad.append("line %d is not '<sha256>  <path>'" % (i + 1))
            continue
        listed[m.group(2)] = m.group(1)
        if fsha is None:
            continue
        b = L.blob(fsha, m.group(2))
        if b is None:
            bad.append("%s is not committed at the freeze commit %s" % (m.group(2), fsha[:12]))
        elif m.group(1) not in (hashlib.sha256(b).hexdigest(), hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest()):
            bad.append("%s differs from its manifest line at the freeze commit %s" % (m.group(2), fsha[:12]))
    fail += ["manifest: " + b for b in bad]
    files = L.freeze_record(text)["files"]
    if files is None:
        fail.append("section 13's FILES list was not found in %s" % L.PREREG)
    else:
        missing = [f for f in files if f not in listed]
        if missing:
            fail.append("manifest lacks section 13 file(s): %s" % ", ".join(missing))
        for p in (JJ.TEMPLATE_PATH, LAUNCHER):
            note.append("%s %s the freeze manifest" % (p, "is in" if p in listed else "is NOT in"))
    note.append("manifest: %d line(s), sha256 %s" % (len(listed), hashlib.sha256(man).hexdigest()))
    digests = {hashlib.sha256(man).digest(), hashlib.sha256(man.replace(b"\r\n", b"\n")).digest()}
    got = [(p, L.blob("HEAD", p)) for p in L.RECEIPTS]
    got = [(p, b) for p, b in got if b is not None]
    if not got:
        fail.append("no timestamp receipt committed at HEAD (%s)" % " or ".join(L.RECEIPTS))
    else:
        ok = [p for p, b in got if any(L.receipt_stamps(p, b, d) for d in digests)]
        if ok:
            note.append("receipt %s stamps the manifest's sha256" % ok[0])
        else:
            fail.append("receipt %s does not stamp the manifest's sha256" % ", ".join(p for p, _ in got))
    return fail


def data_guards(info, note):
    """The used runs must be final: equal to the registered results file's when it exists; otherwise no
    replacement needed and no main run still to launch (load_e1b's main_to_launch: none once section 14 records
    that launching stopped at the achieved n)."""
    fail = []
    h = info["header"]
    if not info["n_runs"]:
        fail.append("load_e1b selected no used run: nothing to score")
    if RESULTS_K0.exists():
        try:
            res = json.loads(RESULTS_K0.read_text(encoding="utf-8"))
        except ValueError:
            res = {}
        if res.get("used") != h["used"]:
            fail.append("the used runs differ from %s's (the registered run's selection)" % _rel(RESULTS_K0))
        if (res.get("data_manifest") or {}).get("sha256") != h["data_manifest"]["sha256"]:
            fail.append("the data manifest differs from the one %s recorded" % _rel(RESULTS_K0))
        if not fail:
            note.append("used runs and data manifest equal %s's" % _rel(RESULTS_K0))
    else:
        s = h["selection"]
        if s["replacements_needed"] or s["main_to_launch"]:
            fail.append("the used runs are not final: %d replacement(s) needed, %d main run(s) still to launch (or "
                        "run the registered analysis first; J3 then scores its used runs)"
                        % (s["replacements_needed"], s["main_to_launch"]))
        note.append("no registered results file yet: the used runs are load_e1b's now")
    return fail


def aws_guards(rid):
    """Read-only: the instance role reaches e1/e1b/j3/<run_id>/, and nothing is stored there yet."""
    fail = []
    arn = "arn:aws:s3:::%s/%s/%s/%s/job.py" % (BUCKET, S3_ROOT, STAGE, rid)
    try:
        docs = []
        prof = L.aws(BUCKET_REGION, "iam", "get-instance-profile", "--instance-profile-name", PROFILE)
        for role in [r["RoleName"] for r in prof["InstanceProfile"]["Roles"]]:
            for name in L.aws(BUCKET_REGION, "iam", "list-role-policies", "--role-name", role)["PolicyNames"]:
                docs.append(L.aws(BUCKET_REGION, "iam", "get-role-policy", "--role-name", role,
                                  "--policy-name", name)["PolicyDocument"])
            for p in L.aws(BUCKET_REGION, "iam", "list-attached-role-policies", "--role-name", role)["AttachedPolicies"]:
                ver = L.aws(BUCKET_REGION, "iam", "get-policy", "--policy-arn", p["PolicyArn"])["Policy"]["DefaultVersionId"]
                docs.append(L.aws(BUCKET_REGION, "iam", "get-policy-version", "--policy-arn", p["PolicyArn"],
                                  "--version-id", ver)["PolicyVersion"]["Document"])
        for action in ("s3:GetObject", "s3:PutObject"):
            if not L.policy_allows(docs, action, arn):
                fail.append("instance role %s does not allow %s on %s" % (PROFILE, action, arn))
    except (RuntimeError, KeyError, TypeError) as e:
        fail.append("could not read the instance role's policies: %s" % str(e)[:300])
    rc, out, err = L.s3("ls", prefix(rid) + "/", "--recursive")
    if rc not in (0, 1):
        fail.append("could not list %s/: %s" % (prefix(rid), err.strip()[-300:]))
    elif rc == 0 and out.strip():
        fail.append("%s/ already holds objects: this run id was launched before" % prefix(rid))
    return fail


# ------------------------------------------------------------------------------------------
# instance states and measured hours
def _parse_ts(s):
    if not s:
        return None
    try:
        t = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def describe(rec):
    """-> (state, launch time, end time or None): 'gone' when AWS no longer knows the instance, 'unknown' when the
    call fails otherwise."""
    try:
        d = L.aws(rec.get("region", BUCKET_REGION), "ec2", "describe-instances", "--instance-ids", rec["instance_id"])
        if d is not None and d.get("Reservations") == []:      # AWS has purged it: rc 0, no reservations, no NotFound
            return "gone", None, None
        i = d["Reservations"][0]["Instances"][0]
    except RuntimeError as e:
        return ("gone" if _code(e) == "InvalidInstanceID.NotFound" else "unknown"), None, None
    except (KeyError, IndexError, TypeError):
        return "unknown", None, None
    m = re.search(r"\((\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) GMT\)", i.get("StateTransitionReason") or "")
    end = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc) if m else None
    return i["State"]["Name"], _parse_ts(i.get("LaunchTime")), end


def measured(rec, state, launch, end, now=None):
    """(hours, final) for runs/e1b/instance_hours.json: launch to its stop time when the instance is down and the
    time is known, else launch to now; an instance AWS no longer knows counts at its backstop."""
    now = now or datetime.now(timezone.utc)
    launch = launch or _parse_ts(rec.get("submitted_utc"))
    cap = float(rec.get("backstop_min", BACKSTOP_MAX_MIN)) / 60.0
    if state == "gone" or launch is None:
        return cap, state == "gone"
    if state in DOWN:           # stopped: its stop time, or (unknown) at most its backstop
        h = (end - launch).total_seconds() / 3600.0 if end else min(cap, (now - launch).total_seconds() / 3600.0)
        return max(0.0, h), state == "terminated"
    return max(0.0, (now - launch).total_seconds() / 3600.0), False      # running: launch to now, uncapped


def record_hours(recs):
    """Describe every J3 instance (read-only AWS calls, before any lock), then merge the measured hours into
    runs/e1b/instance_hours.json with e1b_registry.update_instance_hours, the one writer both launchers use (an
    exclusive lock file around the read-merge-write, so a main launcher writing at the same moment loses nothing;
    other instances' entries untouched; a final entry never changed). -> {instance_id: (state, hours, final)}."""
    seen = {}
    for r in recs:
        if r.get("instance_id"):
            state, launch, end = describe(r)
            h, final = measured(r, state, launch, end)
            seen[r["instance_id"]] = (state, {"hours": round(h, 4), "final": bool(final)})
    if not seen:
        return {}
    hours = REG.update_instance_hours({iid: v for iid, (_st, v) in seen.items()}, REG.INSTANCE_HOURS)
    return {iid: (st, hours[iid]["hours"], hours[iid]["final"]) for iid, (st, _v) in seen.items()}


# ------------------------------------------------------------------------------------------
# user data and launch
def render(rid, backstop_min, workdir=WORKDIR, kill_after_s=KILL_AFTER_S, stall_s=STALL_MIN * 60, tick_s=TICK_S,
           push_s=PUSH_S, job_min=None):
    """The instance's user data. Every placeholder is used and none is left; ASCII."""
    job_min = backstop_min - JOB_PAD_MIN if job_min is None else job_min
    if job_min <= 0:
        raise RuntimeError("job timeout %d min is not positive" % job_min)
    subs = {"__BACKSTOP_MIN__": str(backstop_min), "__BUCKET_REGION__": BUCKET_REGION, "__WORKDIR__": workdir,
            "__RUN_ID__": rid, "__S3_PREFIX__": prefix(rid), "__UV_VERSION__": J.UV_VERSION,
            "__UV_SHA256__": J.UV_WHEEL_SHA256, "__KILL_AFTER_S__": str(kill_after_s), "__JOB_MIN__": str(job_min),
            "__RUN_ARGS__": " ".join(J.RUN_ARGS), "__TICK_S__": str(tick_s), "__STALL_S__": str(stall_s),
            "__PUSH_S__": str(push_s)}
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


def launch(rid, k, tmp, room_usd, room_h, regions):
    """One on-demand instance through regions x g6e types x subnets; a type whose backstop would be under
    BACKSTOP_MIN_MIN is skipped. -> (instance, region, ami, type, backstop_min, misses)."""
    misses = []
    for region in regions:
        cfg = L.REGIONS[region]
        for itype, _usd in REG.TYPES:
            bmin = backstop_for(itype, room_usd, room_h)
            if bmin is None:
                misses.append("%s/%s:OverCeiling" % (region, itype))
                continue
            ud = tmp / ("j3_%s.sh" % itype)
            with open(ud, "w", encoding="ascii", newline="\n") as f:
                f.write(render(rid, bmin))
            spec = {"ImageId": cfg["ami"], "InstanceType": itype, "MinCount": 1, "MaxCount": 1,
                    "IamInstanceProfile": {"Name": PROFILE}, "InstanceInitiatedShutdownBehavior": "terminate",
                    "BlockDeviceMappings": [{"DeviceName": "/dev/sda1", "Ebs": {
                        "VolumeSize": VOLUME_GB, "VolumeType": "gp3", "DeleteOnTermination": True}}],
                    "MetadataOptions": {"HttpTokens": "required"},
                    "TagSpecifications": [{"ResourceType": "instance", "Tags": [
                        {"Key": "Name", "Value": "fr-e1b-j3-%d" % k}, {"Key": "Project", "Value": "flight-recorder-e1b"},
                        {"Key": "Stage", "Value": STAGE}]}]}       # no InstanceMarketOptions: on-demand only
            quota = False
            for az, subnet in cfg["subnets"].items():
                spec["SubnetId"] = subnet
                sp = tmp / "j3_spec.json"
                sp.write_text(json.dumps(spec), encoding="ascii")
                try:
                    inst = L.aws(region, "ec2", "run-instances", "--cli-input-json", "file://" + sp.as_posix(),
                                 "--user-data", "file://" + ud.as_posix())["Instances"][0]
                    return inst, region, cfg["ami"], itype, bmin, misses
                except RuntimeError as e:
                    code = _code(e)
                    if code not in L.CAPACITY_CODES:
                        raise
                    misses.append("%s/%s/%s:%s" % (region, itype, az, code))
                    if code == "VcpuLimitExceeded":
                        quota = True
                        break
            if quota:
                break
    return None, None, None, None, None, misses


def take_lock():
    J3_DIR.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(LOCKFILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit("REFUSE: %s exists: another J3 launcher is running (remove the file only if none is)"
                         % _rel(LOCKFILE))
    os.write(fd, ("pid %d %s\n" % (os.getpid(), _utc())).encode("ascii"))
    os.close(fd)


def submit(rid, k, job_dir, prov, info, regions):
    take_lock()
    tmp = Path(tempfile.mkdtemp(prefix="e1b_j3_launch_"))     # no spaces: the aws exe takes file:// paths
    try:
        recs = load_launches()
        if next_run_id(recs) != rid:
            raise SystemExit("REFUSE: %s changed while this launcher ran" % _rel(LAUNCHES))
        room_usd, room_h = room(recs)
        rec = {"stage": STAGE, "run_id": rid, "judge": JJ.J3["judge"], "revision": JJ.J3["revision"],
               "dtype": JJ.J3["dtype"], "git_commit": prov["git_commit"], "package_sha256": prov["package_sha256"],
               "template_sha256": prov["template_sha256"], "lock_sha256": prov["lock_sha256"],
               "job_sha256": prov["job_sha256"], "data_sha256": prov["data_sha256"],
               "n_completions": prov["n_completions"], "n_runs": prov["n_runs"],
               "run_ids": info["header"]["run_ids"], "data_manifest_sha256": info["header"]["data_manifest"]["sha256"],
               "instance_id": None, "region": None, "ami": None, "instance_type": None, "market": "ondemand",
               "submitted_utc": _utc(), "s3": prefix(rid), "backstop_min": None, "job_timeout_min": None,
               "plan_h": round(PLAN_H, 4), "worst_usd": None}
        assert tuple(rec) == RECORD_KEYS
        for name in ("job.py", "job.py.lock", JJ.DATA_NAME):
            rc, _, err = L.s3("cp", str(job_dir / name), prefix(rid) + "/" + name)
            if rc != 0:
                raise SystemExit("upload failed for %s: %s" % (name, err.strip()[-300:]))
        try:
            inst, region, ami, itype, bmin, misses = launch(rid, k, tmp, room_usd, room_h, regions)
        except RuntimeError as e:          # not a capacity miss: no instance, and this run id's prefix is used
            inst, misses = None, ["error: %s" % str(e)[:300]]
        if inst is None:
            # the prefix now holds the uploads, so the run id is spent: a record without an instance moves the next
            # launch to a fresh run id and prefix, and counts nothing towards the ceiling
            _write_json(LAUNCHES, load_launches() + [rec])
            print("NOT LAUNCHED %s: %s (recorded without an instance; the next launch uses %s)"
                  % (rid, ", ".join(misses), next_run_id(load_launches())))
            return 1
        rec.update(instance_id=inst["InstanceId"], region=region, ami=ami, instance_type=itype, backstop_min=bmin,
                   job_timeout_min=bmin - JOB_PAD_MIN, worst_usd=round(bmin / 60.0 * REG.PRICE[itype], 2))
        _write_json(LAUNCHES, load_launches() + [rec])
        print("LAUNCHED %s on %s %s %s %s: backstop %d min, worst case USD %.2f"
              % (rid, inst["InstanceId"], region, inst.get("Placement", {}).get("AvailabilityZone", "?"), itype, bmin,
                 rec["worst_usd"]))
        if misses:
            print("    capacity misses before it: %s" % ", ".join(misses))
        print("launch record: %s" % _rel(LAUNCHES))
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        try:
            LOCKFILE.unlink()
        except OSError:
            pass


# ------------------------------------------------------------------------------------------
# status and collect: counts only
def read_status(rec):
    """status.json from S3 -> {run_id, phase, n_scored, n_total, rc, ts}, None when absent, or "malformed"."""
    rc, out, _ = L.s3("cp", rec["s3"] + "/status.json", "-")
    if rc != 0 or not out.strip():
        return None
    try:
        d = json.loads(out)
    except ValueError:
        return "malformed"
    if not isinstance(d, dict) or d.get("run_id") != rec["run_id"]:
        return "malformed"
    st = {k: d.get(k) for k in ("run_id", "phase", "n_scored", "n_total", "rc", "ts")}
    if not all(st[k] is None or type(st[k]) is int for k in ("n_scored", "n_total")) or \
            not (st["rc"] is None or type(st["rc"]) in (int, str)) or not (st["phase"] is None or
                                                                         isinstance(st["phase"], str)):
        return "malformed"
    return st


def status_rows(recs):
    states = record_hours(recs)
    rows = []
    for r in recs:
        st = read_status(r)
        bad = st == "malformed"
        st = st if isinstance(st, dict) else {"phase": None, "n_scored": None, "n_total": None, "rc": None, "ts": None}
        state, h, final_h = states.get(r["instance_id"], ("unknown", None, False))
        final = st["rc"] is not None
        rows.append(dict(st, run_id=r["run_id"], instance_id=r["instance_id"], region=r.get("region"), state=state,
                         hours=h, usd=None if h is None else committed([r], {r["instance_id"]: {"hours": h}})[0],
                         final=final, lost=(not final) and state in DOWN, malformed=bad))
    return rows


def print_status():
    allr = load_launches()
    for r in allr:
        if not r.get("instance_id"):
            print("%-24s not launched (no instance; recorded so that its S3 prefix is not reused)" % r["run_id"])
    recs = [r for r in allr if r.get("instance_id")]
    if not recs:
        print("no J3 launch (%s)" % _rel(LAUNCHES))
        return 0
    print("%-24s %-20s %-13s %-15s %-17s %-12s %s" % ("run", "instance", "state", "phase", "scored", "rc", "hours"))
    for r in status_rows(recs):
        sc = "-" if r["n_scored"] is None else "%d/%s" % (r["n_scored"], r["n_total"])
        rc = "LOST" if r["lost"] else ("-" if r["rc"] is None else str(r["rc"]))
        print("%-24s %-20s %-13s %-15s %-17s %-12s %s%s" % (
            r["run_id"], r["instance_id"], r["state"], r["phase"] or "-", sc, rc,
            "-" if r["hours"] is None else "%.2f h (committed USD %.2f)" % (r["hours"], r["usd"]),
            "  (status.json malformed)" if r["malformed"] else ""))
    usd, h = committed(recs)
    print("J3 committed: USD %.2f / %.0f, %.2f / %.0f GPU-h (larger of plan %.2f h and measured, per instance)"
          % (usd, CEIL_USD, h, CEIL_H, PLAN_H))
    return 0


def coverage(doc, run_ids):
    """Counts only: completions scored (finite, in [0, 1]), steps with all 8 scored, runs present, unknown runs."""
    scores = doc.get("scores") or {}
    n = full = 0
    for rid in run_ids:
        for s in range(JJ.STEPS):
            v = (scores.get(rid) or {}).get(str(s))
            ok = [x for x in (v or []) if isinstance(x, (int, float)) and not isinstance(x, bool) and 0.0 <= x <= 1.0]
            n += len(ok)
            full += int(isinstance(v, list) and len(v) == JJ.GEN and len(ok) == JJ.GEN)
    return {"scored": n, "total": len(run_ids) * JJ.STEPS * JJ.GEN, "steps_full": full,
            "steps_total": len(run_ids) * JJ.STEPS, "runs_missing": sum(1 for r in run_ids if r not in scores),
            "unknown_runs": len(set(scores) - set(run_ids))}


def _snap(x):
    """The snapshot directory names of a hub_cache record ({"snapshots": [...], "refs": {...}}), or None."""
    return x.get("snapshots") if isinstance(x, dict) else None


def result_problems(doc, rec):
    """Names of the fields of a J3 result (result.json or partial.json) that differ from the launch record or from
    the registered J3 and validation judges (pre-registration section 2; build contract section 2), names only:
      * judge, revision, dtype: J3's (e1b_j3_job.J3), and the launch record's; data_sha256, job_file_sha256,
        run_id, stage: the launch record's;
      * snapshots.target: exactly [J3's revision], the one Hub snapshot the target was loaded from, whenever the
        target loaded or the file holds a score; judge_facts.param_dtype and judge_facts.revision, the measured
        dtype and the revision the scorer used, J3's once the target loaded;
      * validation (model, revision, hub_cache) and snapshots.validation: the validation judge's (e1b_j3_job.VAL)
        whenever the gate ran, exactly [its revision] as snapshot; and a file that holds a score must have passed it."""
    bad = [k for k in ("judge", "revision", "dtype") if doc.get(k) != JJ.J3[k] or doc.get(k) != rec.get(k)]
    bad += [k for k, want in (("data_sha256", rec["data_sha256"]), ("job_file_sha256", rec["job_sha256"]),
                              ("run_id", rec["run_id"]), ("stage", STAGE)) if doc.get(k) != want]
    scores = doc.get("scores") if isinstance(doc.get("scores"), dict) else {}
    scored = bool(doc.get("n_scored")) or any(x is not None for st in scores.values() if isinstance(st, dict)
                                              for v in st.values() if isinstance(v, list) for x in v)
    facts = doc.get("judge_facts")
    snaps = doc.get("snapshots") if isinstance(doc.get("snapshots"), dict) else {}
    if facts is not None or scored or "target" in snaps:
        if _snap(snaps.get("target")) != [JJ.J3["revision"]]:
            bad.append("snapshots.target")
    if facts is not None or scored:
        facts = facts if isinstance(facts, dict) else {}
        bad += ["judge_facts.%s" % k for k, want in (("param_dtype", JJ.J3["dtype"]), ("revision", JJ.J3["revision"]))
                if facts.get(k) != want]
    val = doc.get("validation")
    if val is not None or scored or "validation" in snaps:
        val = val if isinstance(val, dict) else {}
        bad += ["validation.%s" % k for k, want in (("model", JJ.VAL["judge"]), ("revision", JJ.VAL["revision"]))
                if val.get(k) != want]
        if _snap(val.get("hub_cache")) != [JJ.VAL["revision"]]:
            bad.append("validation.hub_cache")
        if "validation" in snaps and _snap(snaps.get("validation")) != [JJ.VAL["revision"]]:
            bad.append("snapshots.validation")
        if scored and val.get("passed") is not True:
            bad.append("validation.passed")
    return bad


def collect(accept_partial=False):
    recs = [r for r in load_launches() if r.get("instance_id")]
    if not recs:
        raise SystemExit("REFUSE: no launched J3 instance (%s)" % _rel(LAUNCHES))
    rec = recs[-1]
    row = next(r for r in status_rows([rec]) if r["run_id"] == rec["run_id"])
    if not (row["final"] or row["lost"]):
        raise SystemExit("REFUSE: %s has no final rc yet and its instance is %s; nothing is read before it ends"
                         % (rec["run_id"], row["state"]))
    tmp = Path(tempfile.mkdtemp(prefix="e1b_j3_collect_"))
    try:
        got = {}
        for name in ("result.json", "partial.json", "job.log", "boot.log", "status.json"):
            rc, _, _ = L.s3("cp", rec["s3"] + "/" + name, str(tmp / name))
            got[name] = rc == 0 and (tmp / name).is_file()
        logs, sdir = J3_DIR / "logs", J3_DIR / "status"
        for d in (logs, sdir):
            d.mkdir(parents=True, exist_ok=True)
        for name, dest in (("job.log", logs / (rec["run_id"] + ".log")),
                           ("boot.log", logs / (rec["run_id"] + ".boot.log")),
                           ("status.json", sdir / (rec["run_id"] + ".json"))):
            if got[name]:
                shutil.copyfile(tmp / name, dest)
        src = "result.json" if got["result.json"] else ("partial.json" if got["partial.json"] and accept_partial
                                                        else None)
        print("%-24s %-28s %s" % (rec["run_id"], "LOST: instance %s, no final rc" % row["state"] if row["lost"]
                                  else "rc=%s" % row["rc"],
                                  "  ".join("%s %s" % (n, "ok" if v else "missing") for n, v in got.items())))
        if src is None:
            raise SystemExit("REFUSE: no result.json%s; nothing written to %s" % (
                " (partial.json exists: --accept-partial collects it, and its C6 rows are PROVISIONAL)"
                if got["partial.json"] else "", _rel(RESULT)))
        raw = (tmp / src).read_bytes()
        try:
            doc = json.loads(raw)
        except ValueError:
            doc = None
        if not isinstance(doc, dict):
            raise SystemExit("REFUSE: %s is not a JSON object; nothing written" % src)
        checks = result_problems(doc, rec)
        if checks:
            raise SystemExit("REFUSE: %s differs from the launch record or the registered J3 in: %s; nothing written"
                             % (src, ", ".join(checks)))
        if RESULT.exists() and RESULT.read_bytes() != raw:
            raise SystemExit("REFUSE: %s exists with different bytes; a collected file never changes silently"
                             % _rel(RESULT))
        RESULT.parent.mkdir(parents=True, exist_ok=True)
        RESULT.write_bytes(raw)
        with open(DATA_MANIFEST, "w", encoding="ascii", newline="\n") as fh:
            fh.write("%s  %s\n" % (hashlib.sha256(raw).hexdigest(), RESULT.name))
        cov = coverage(doc, rec["run_ids"])
        label = ("PARTIAL" if src == "partial.json" else "ABORTED" if doc.get("aborted") else
                 "COMPLETE" if doc.get("complete") and cov["steps_full"] == cov["steps_total"] else "INCOMPLETE")
        print("%s -> %s: %s; %d of %d completions scored, %d of %d steps fully scored, %d run(s) missing, %d unknown"
              % (src, _rel(RESULT), label, cov["scored"], cov["total"], cov["steps_full"], cov["steps_total"],
                 cov["runs_missing"], cov["unknown_runs"]))
        print("judge, revision, dtype, Hub snapshots, data sha256, job sha256 and run id match the launch record and "
              "the registered J3; manifest %s" % _rel(DATA_MANIFEST))
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------------------------------
def main(argv):
    opts = {a.split("=", 1)[0]: (a.split("=", 1)[1] if "=" in a else True) for a in argv if a.startswith("--")}
    pos = [a for a in argv if not a.startswith("--")]
    known = {"--submit", "--commit", "--status", "--collect", "--accept-partial", "--region", "--show-userdata"}
    valued = ("--commit", "--region")
    bad = [o for o in opts if (o in valued) != isinstance(opts[o], str)]
    if pos or set(opts) - known or bad:
        raise SystemExit("usage: python scripts/launch_e1b_j3.py [--commit=SHA] [--submit] | --status | --collect "
                         "[--accept-partial]; options --region=R --show-userdata (got %s)" % " ".join(argv))
    if "--status" in opts:
        return print_status()
    if "--collect" in opts:
        return collect("--accept-partial" in opts)
    regions = [opts["--region"]] if "--region" in opts else list(L.REGION_ORDER)
    if any(r not in L.REGIONS for r in regions):
        raise SystemExit("--region must be one of %s" % ", ".join(L.REGIONS))
    commit, do_submit = opts.get("--commit"), "--submit" in opts
    recs = load_launches()
    rid, k = next_run_id(recs), len(recs) + 1
    print("E1b J3 scoring: %s, judge %s@%s %s, S3 %s/" % (rid, JJ.J3["judge"], JJ.J3["revision"][:12], JJ.J3["dtype"],
                                                         prefix(rid)))
    fail, note, sha = guards(commit, do_submit, recs)
    states = record_hours(recs) if recs and do_submit else {}
    up = [iid for iid, (st, _h, _f) in states.items() if st not in DOWN]
    if up:
        fail.append("earlier J3 instance(s) still up: %s" % ", ".join(up))
    c_usd, c_h = committed(recs)
    room_usd, room_h = room(recs)
    plan = [(t, backstop_for(t, room_usd, room_h)) for t, _u in REG.TYPES]
    note.append("J3 ceiling USD %.0f / %.0f GPU-h: committed USD %.2f / %.2f GPU-h (%d earlier launch(es)); room USD "
                "%.2f / %.2f GPU-h" % (CEIL_USD, CEIL_H, c_usd, c_h, len(recs), room_usd, room_h))
    for t, b in plan:
        note.append("%-12s %s" % (t, "backstop %d min, job timeout %d min, worst case USD %.2f" % (
            b, b - JOB_PAD_MIN, b / 60.0 * REG.PRICE[t]) if b else "skipped: less than %d min of room"
            % BACKSTOP_MIN_MIN))
    if not any(b for _t, b in plan):
        fail.append("J3 ceiling: no g6e type has %d minutes of room (committed USD %.2f, %.2f GPU-h); a further J3 "
                    "launch needs a section 14 deviation" % (BACKSTOP_MIN_MIN, c_usd, c_h))
    with tempfile.TemporaryDirectory(prefix="e1b_j3_bake_") as td:
        info = prov = job_path = None
        try:
            info = JJ.prep_data(td)
            print("data: %d completions of %d used runs (%s), sha256 %s"
                  % (info["n_rows"], info["n_runs"], ", ".join("%s %d" % (a, len(v)) for a, v in
                                                               info["header"]["used"].items()), info["sha256"]))
            fail += data_guards(info, note)
        except SystemExit as e:
            fail.append(str(e))
        if sha and info:
            try:
                job_path, prov = JJ.bake(sha, info["path"], rid, td)
                print("baked from %s: package %s, template %s, lock %s, job %s"
                      % (prov["git_commit"][:12], prov["package_sha256"][:16], prov["template_sha256"][:16],
                         prov["lock_sha256"][:16], prov["job_sha256"][:16]))
            except (ValueError, RuntimeError) as e:
                fail.append("bake: %s" % e)
        for n in note:
            print("  note: %s" % n)
        if "--show-userdata" in opts:
            b = next((b for _t, b in plan if b), BACKSTOP_MIN_MIN)
            print("----- user data (backstop %d min) -----\n%s-----" % (b, render(rid, b)))
        if not do_submit:
            for f in fail:
                print("  --submit would REFUSE: %s" % f)
            print("\nDRY RUN: nothing uploaded or launched, no launch record written. --submit (with --commit) spends.")
            return 0
        if not fail:
            fail = aws_guards(rid)
        if fail:
            raise SystemExit("\n".join("REFUSE: %s" % f for f in fail))
        return submit(rid, k, job_path.parent, prov, info, regions)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
