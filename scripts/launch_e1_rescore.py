"""E1 third-judge rescoring on EC2: one on-demand L40S instance per judge, cost-capped. ASCII only.

WHAT
  E1 stored every policy completion (48 runs x 400 steps x 8 = 153,600 texts). The audit headline is
  confounded by construction: the 0.5B judge alone separates the arms at AUROC 1.00 because it IS the
  hacking arm's reward (experiments/audit_baselines_report.txt). This rescores every completion with a
  THIRD judge that neither arm optimized, using the logged judges' exact definition (reward_judge.JudgeScorer):
      olmo2_7b   allenai/OLMo-2-1124-7B-Instruct   PRIMARY    (different family)
      qwen25_7b  Qwen/Qwen2.5-7B-Instruct          SECONDARY  (same family, larger)
  Each job is baked from experiments/e1_rescore/rescore_job_template.py, uploaded to
  s3://BUCKET/e1/rescore/<judge_short>/job.py and run by its own instance. The job fetches the data
  (s3://BUCKET/e1/rescore/data/e1_completions.jsonl.gz) itself, writes /tmp/result.json and pushes
  <prefix>/partial.json every ~10 min, so a timeout still leaves the scores done so far.

HOW
  Reuses launch_e1_aws (aws(), REGION_CFG, SUBNETS, AMI_PARAM, instance profile, bucket) and the
  launch_e1_batch pattern (uv user-data, status file, job.log upload loop, EXIT trap, scheduled backstop
  shutdown, InstanceInitiatedShutdownBehavior=terminate, launch_with_fallback). No secrets reach the
  instance: both judges are ungated, and the role can only touch s3://BUCKET/e1/*.

COST CAP (on-demand only)
  Types g6e.xlarge then g6e.2xlarge (1x L40S 48 GB; list $1.861 and $2.242/h). Regions us-east-2 (AMI
  pinned by id in REGION_CFG) then us-east-1 (SSM AMI parameter). Per-job timeout 225 min (3.75 h);
  backstop shutdown at +240 min. Worst case = jobs x 4 h x $2.242 = $17.94 for both; --submit refuses
  above CAP_USD ($20), counting instances already in the launch record.

    python scripts/launch_e1_rescore.py                # dry run: bake (git archive HEAD) + py_compile + the
                                                       # job's import-time checks in a child process + price. No AWS call.
    python scripts/launch_e1_rescore.py --preflight    # dry run + read-only checks (Hub, get/describe/head/ls)
    python scripts/launch_e1_rescore.py --submit       # SPEND: upload data + jobs, launch the instances
    python scripts/launch_e1_rescore.py --status       # instance state, status line, number of scores so far
    python scripts/launch_e1_rescore.py --collect      # -> experiments/e1_rescore/results/<judge_short>.json
  Options: --only=olmo2_7b[,qwen25_7b]  --data=PATH (.jsonl.gz; its .sha256 sits beside it)
           --template=PATH  --show-userdata
--status and --collect never print score values, only counts.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import py_compile
import re
import subprocess
import sys
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import launch_e1_aws as A        # noqa: E402  aws(), REGION_CFG, SUBNETS, AMI_PARAM, PROFILE, BUCKET
import launch_e1_batch as BATCH  # noqa: E402  launch_with_fallback()

# ---- third judges: fixed 2026-10-09, before any rescoring -------------------------------------------
# Revision = the Hub's `main` commit sha, read 2026-10-09 from https://huggingface.co/api/models/<id>
# (both ungated). They MUST equal the job builder's pins; the job loads with revision=__REVISION__.
JUDGE_OLMO2_7B = "allenai/OLMo-2-1124-7B-Instruct"
REVISION_OLMO2_7B = "470b1fba1ae01581f270116362ee4aa1b97f4c84"
JUDGE_QWEN25_7B = "Qwen/Qwen2.5-7B-Instruct"
REVISION_QWEN25_7B = "a09a35458c702b33eeacc393d103063234e8bc28"
JOBS = {   # judge_short -> (judge id, pinned revision, role). Launch order = this order (primary first).
    "olmo2_7b": (JUDGE_OLMO2_7B, REVISION_OLMO2_7B, "primary"),
    "qwen25_7b": (JUDGE_QWEN25_7B, REVISION_QWEN25_7B, "secondary"),
}
DTYPE = "bfloat16"     # chosen inside the job (and written to result.json); recorded here too
PARAMS_B = 7.6         # for the rough runtime estimate only (OLMo-2 7B 7.3B, Qwen2.5-7B 7.6B)

# ---- inputs and outputs -----------------------------------------------------------------------------
SCRATCH = Path(__import__("os").environ.get("E1_RESCORE_SCRATCH", str(Path(__file__).resolve().parents[1] / "experiments" / "e1_rescore" / "_work")))
DATA_GZ = SCRATCH / "e1_completions.jsonl.gz"          # written by experiments/e1_rescore/prep_data.py
N_EXPECTED = 48 * 400 * 8                                # runs x steps x stored completions
DATA_KEYS = ("run", "arm", "task", "seed", "step", "i", "text", "hidden", "hidden_lenient")
TEMPLATE = ROOT / "experiments" / "e1_rescore" / "rescore_job_template.py"
PLACEHOLDERS = ("__JUDGE__", "__REVISION__", "__S3_DATA__", "__S3_PREFIX__", "__DATA_SHA__", "__PKG_B64__")
RECORD = ROOT / "experiments" / "e1_rescore" / "rescore_launch.json"   # not runs/e1/: that folder holds the frozen E1 data
RESULTS = ROOT / "experiments" / "e1_rescore" / "results"

# ---- S3 (bucket in us-east-1 whatever region the instance runs in) ----------------------------------
BUCKET_REGION = A.BUCKET_REGION
DATA_KEY = "e1/rescore/data/e1_completions.jsonl.gz"
S3_DATA = "s3://%s/%s" % (A.BUCKET, DATA_KEY)


def prefix(short):
    return "s3://%s/e1/rescore/%s" % (A.BUCKET, short)


# ---- cost cap ---------------------------------------------------------------------------------------
TYPES = [("g6e.xlarge", 1.861), ("g6e.2xlarge", 2.242)]   # on-demand list USD/h (AWS pricing API, 2026-10-07)
REGIONS = ["us-east-2", "us-east-1"]
SUBNETS_US_EAST_1 = dict(A.SUBNETS)                        # captured before any region switch
TIMEOUT_MIN = 225      # per-job hard timeout around `uv run job.py` (3.75 h)
BACKSTOP_MIN = 240     # `shutdown -h +240` at boot; terminate-on-shutdown makes it a termination
CAP_USD = 20.0
VOLUME_GB = 120        # AMI + uv/torch cache + one 7B checkpoint (~15 GB) with room to spare
EFF_TFLOPS = 100.0     # ASSUMED effective bf16 forward throughput on one L40S (peak dense ~362)

USER_DATA = r"""#!/bin/bash
set -u
export AWS_DEFAULT_REGION=__BUCKET_REGION__   # the bucket's region, whatever region the instance is in
mkdir -p /opt/e1/data && cd /opt/e1
exec > /opt/e1/boot.log 2>&1
B="__PREFIX__"
shutdown -h +__BACKSTOP_MIN__
RC=boot-failed
finish() {
  [ -n "${LOOP:-}" ] && kill "$LOOP" 2>/dev/null
  echo "rc=$RC $(date -u +%FT%TZ)" > /opt/e1/status
  [ -f /opt/e1/job.log ] && aws s3 cp /opt/e1/job.log "$B/job.log" --only-show-errors
  [ -f /tmp/result.json ] && aws s3 cp /tmp/result.json "$B/result.json" --only-show-errors
  aws s3 cp /opt/e1/status "$B/status" --only-show-errors
  aws s3 cp /opt/e1/boot.log "$B/boot.log" --only-show-errors
  shutdown -h now
}
trap finish EXIT
trap 'RC=terminated; exit 143' TERM INT
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python3 -m pip install -q uv || { apt-get update -q && apt-get install -yq python3-pip && python3 -m pip install -q uv; }
aws s3 cp "$B/job.py" /opt/e1/job.py --only-show-errors || { RC=no-job-py; exit 1; }
echo "started $(date -u +%FT%TZ)" > /opt/e1/status && aws s3 cp /opt/e1/status "$B/status" --only-show-errors
( while sleep 300; do aws s3 cp /opt/e1/job.log "$B/job.log" --only-show-errors; done ) &
LOOP=$!
export HF_HOME=/opt/e1/hf UV_CACHE_DIR=/opt/e1/uvcache PYTHONUNBUFFERED=1
timeout --kill-after=120 __TIMEOUT_MIN__m python3 -m uv run --python 3.12 /opt/e1/job.py > /opt/e1/job.log 2>&1
RC=$?
"""


def worst_usd(n_jobs):
    return n_jobs * BACKSTOP_MIN / 60.0 * max(u for _, u in TYPES)


def committed_worst(record):
    """Worst case already committed: every launched instance runs to its backstop at its own price."""
    return sum(v.get("backstop_min", BACKSTOP_MIN) / 60.0 * float(v["usd_per_h"])
               for v in record.values() if v.get("instance_id"))


def rel(p):
    try:
        return p.relative_to(ROOT)
    except ValueError:
        return p


def load_record():
    return json.loads(RECORD.read_text()) if RECORD.exists() else {}


def render(short):
    ud = (USER_DATA.replace("__BUCKET_REGION__", BUCKET_REGION).replace("__PREFIX__", prefix(short))
          .replace("__BACKSTOP_MIN__", str(BACKSTOP_MIN)).replace("__TIMEOUT_MIN__", str(TIMEOUT_MIN)))
    left = re.findall(r"__[A-Z_]+__", ud)
    assert not left, "unsubstituted in user-data: %s" % left
    ud.encode("ascii")
    return ud


# ---- data -------------------------------------------------------------------------------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_data(path):
    """sha256 against the data builder's .sha256 file, then a structural scan. -> (sha, info) or SystemExit."""
    sha_file = path.with_name(path.name.replace(".jsonl.gz", ".sha256"))
    if not path.exists() or not sha_file.exists():
        raise FileNotFoundError("data not built yet: %s (+ %s)" % (path, sha_file.name))
    sha = sha256_file(path)
    want = sha_file.read_text(encoding="ascii").split()[0].strip().lower()
    if sha != want:
        raise SystemExit("REFUSE: %s sha256 %s != %s in %s" % (path.name, sha, want, sha_file.name))
    from flightrecorder.repro.reward_codeset import CODE_TASKS_BY_NAME
    from flightrecorder.repro.reward_judge import _JUDGE_TEMPLATE
    from flightrecorder.repro.reward_testhack import extract_code
    n, runs, arms, last, chars, bad = 0, {}, {}, None, 0, []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            n += 1
            miss = [k for k in DATA_KEYS if k not in d]
            if miss and len(bad) < 3:
                bad.append("line %d lacks %s" % (n, miss))
            try:
                key = (str(d.get("run")), int(d.get("step")), int(d.get("i")))
                if last is not None and key <= last and len(bad) < 3:
                    bad.append("line %d out of (run, step, i) order or duplicated" % n)
                last = key
            except (TypeError, ValueError):
                if len(bad) < 3:
                    bad.append("line %d has a non-integer step or i" % n)
            runs[d.get("run")] = d.get("arm")
            if d.get("task") not in CODE_TASKS_BY_NAME:
                if len(bad) < 3:
                    bad.append("line %d unknown task %r" % (n, d.get("task")))
            elif n % 16 == 1:   # sampled exact judge-prompt length, for the runtime estimate only
                chars += 16 * len(_JUDGE_TEMPLATE.format(problem=CODE_TASKS_BY_NAME[d["task"]].prompt,
                                                         code=extract_code(d.get("text", ""))))
    for a in runs.values():
        arms[a] = arms.get(a, 0) + 1
    info = {"n": n, "runs": len(runs), "arms": arms, "prompt_chars_est": chars, "problems": bad}
    return sha, info


def est_hours(prompt_chars):
    """ROUGH: ~3.2 chars/token for code-heavy prompts; forward FLOPs = 2 x params x tokens."""
    tokens = prompt_chars / 3.2
    return tokens, tokens * 2 * PARAMS_B * 1e9 / (EFF_TFLOPS * 1e12) / 3600.0


# ---- bake -------------------------------------------------------------------------------------------
def pkg_b64():
    """base64 tar.gz of the COMMITTED flightrecorder/ (git archive HEAD), built by the job builder's own
    experiments/e1_rescore/bake_rescore_job.package_b64, so the launched jobs embed exactly what the local CPU
    test ran and the job's provenance records the commit (the tar's pax comment). An uncommitted edit never
    rides along; the job also checks the three judge modules' SHA-256s."""
    sys.path.insert(0, str(ROOT / "experiments" / "e1_rescore"))
    import bake_rescore_job  # noqa: E402
    return bake_rescore_job.package_b64("HEAD")


def head_commit():
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True, capture_output=True,
                          text=True).stdout.strip()


_IMPORT_CHECK = (
    "import importlib.util, sys\n"
    "s = importlib.util.spec_from_file_location('baked_job', sys.argv[1])\n"
    "m = importlib.util.module_from_spec(s)\n"
    "s.loader.exec_module(m)\n"   # runs ONLY the module-level checks; main() is never called
    "print('module checks ok: package commit %s, judge %s@%s' % (m.PKG_COMMIT, m.JUDGE, m.REVISION[:12]))\n")


def bake(short, data_sha, b64, outdir, template=TEMPLATE):
    """-> (path, sha256 of the exact bytes to upload). Asserts every placeholder used and none left, that the
    template's own pin table carries this revision, then imports the baked job in a child process so its
    module-level checks (pin table, package unpack, judge-module and judge-template SHA-256s) fail HERE, not on
    a billed instance. main() is not called: no model, no AWS."""
    judge, rev, _ = JOBS[short]
    src = template.read_text(encoding="utf-8")
    missing = [p for p in PLACEHOLDERS if p not in src]
    assert not missing, "template %s lacks placeholders %s" % (template.name, missing)
    assert re.fullmatch(r"[0-9a-f]{40}", rev) and re.fullmatch(r"[0-9a-f]{64}", data_sha)
    assert re.search(r'"%s":\s*"%s"' % (re.escape(judge), rev), src), \
        "template KNOWN_REVISIONS does not pin %s to %s" % (judge, rev)
    for k, v in (("__JUDGE__", judge), ("__REVISION__", rev), ("__S3_DATA__", S3_DATA),
                 ("__S3_PREFIX__", prefix(short)), ("__DATA_SHA__", data_sha), ("__PKG_B64__", b64)):
        src = src.replace(k, v)
    left = re.findall(r"__[A-Z0-9_]+__", src.replace(b64, ""))
    assert not left, "unsubstituted placeholders: " + ", ".join(sorted(set(left)))
    outp = Path(outdir) / ("_rescore_%s.py" % short)
    outp.write_bytes(src.encode("utf-8"))
    py_compile.compile(str(outp), cfile=str(outp) + "c", doraise=True)   # fail locally BEFORE spending
    r = subprocess.run([sys.executable, "-c", _IMPORT_CHECK, str(outp)], capture_output=True, text=True,
                       cwd=str(outdir))
    if r.returncode != 0 or "module checks ok" not in r.stdout:
        raise SystemExit("REFUSE: baked job %s fails its own import-time checks:\n%s" % (
            short, (r.stdout + r.stderr)[-1500:]))
    return outp, hashlib.sha256(outp.read_bytes()).hexdigest()


# ---- AWS helpers (S3 always pinned to the bucket's region) ------------------------------------------
def ec2(region, *args):
    old = A.REGION
    A.REGION = region
    try:
        return A.aws(*args)
    finally:
        A.REGION = old


def s3_cp(src, dst, *extra):
    r = subprocess.run(["aws", "s3", "cp", str(src), str(dst), *extra, "--region", BUCKET_REGION,
                        "--only-show-errors"], capture_output=True, text=True)
    return r.returncode == 0


def s3_text(uri):
    r = subprocess.run(["aws", "s3", "cp", uri, "-", "--region", BUCKET_REGION, "--only-show-errors"],
                       capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def s3_head(key):
    """-> head-object dict, or None if the object does not exist."""
    try:
        return ec2(BUCKET_REGION, "s3api", "head-object", "--bucket", A.BUCKET, "--key", key)
    except RuntimeError as e:
        if "404" in str(e) or "Not Found" in str(e):
            return None
        raise


def ami_for(region):
    if region in A.REGION_CFG:
        return A.REGION_CFG[region]["ami"]
    return ec2(region, "ssm", "get-parameter", "--name", A.AMI_PARAM)["Parameter"]["Value"]


def subnets_for(region):
    return A.REGION_CFG[region]["subnets"] if region in A.REGION_CFG else SUBNETS_US_EAST_1


# ---- plan / dry run ---------------------------------------------------------------------------------
def print_plan(shorts, record, data_path, template, show_ud):
    print("E1 third-judge rescoring: %d job(s), one instance each, on-demand" % len(shorts))
    print("%-10s %-9s %-34s %-12s %-9s %s" % ("job", "role", "judge", "revision", "dtype", "s3 prefix"))
    for s in shorts:
        j, rev, role = JOBS[s]
        print("%-10s %-9s %-34s %-12s %-9s %s" % (s, role, j, rev[:12], DTYPE, prefix(s)))
    data_sha = None
    try:
        data_sha, info = check_data(data_path)
        print("data      %s\n          sha256 %s (matches .sha256), %d completions, %d runs %s%s" % (
            data_path, data_sha, info["n"], info["runs"], json.dumps(info["arms"], sort_keys=True),
            "" if info["n"] == N_EXPECTED else "   WARNING: expected %d" % N_EXPECTED))
        for p in info["problems"]:
            print("          PROBLEM: %s" % p)
        tok, h = est_hours(info["prompt_chars_est"])
        print("runtime   ROUGH: ~%.0fM judge-prompt tokens per judge -> ~%.1f h of forward passes at an ASSUMED %.0f"
              " TFLOPS on one L40S (timeout %.2f h; partial.json salvages a timeout)%s" % (
                  tok / 1e6, h, EFF_TFLOPS, TIMEOUT_MIN / 60.0,
                  "   WARNING: estimate is near/over the timeout" if h > 0.8 * TIMEOUT_MIN / 60.0 else ""))
    except FileNotFoundError as e:
        print("data      NOT READY: %s (dry run continues; --submit requires it)" % e)
    if template.exists():
        with tempfile.TemporaryDirectory(prefix="e1rescore_dry_") as td:
            b64 = pkg_b64()
            print("package   flightrecorder/ from git archive HEAD %s" % head_commit()[:12])
            for s in shorts:
                outp, jsha = bake(s, data_sha or "0" * 64, b64, td, template)
                print("bake      %-10s ok: placeholders filled, py_compile ok, import-time checks ok, %d bytes, "
                      "job sha256 %s%s" % (s, outp.stat().st_size, jsha[:16], "" if data_sha else " (dummy data sha)"))
    else:
        print("bake      NOT READY: template %s not found (dry run continues; --submit requires it)" % template)
    for s in shorts:
        render(s)
    print("user-data rendered for %s (ASCII, no placeholder left)" % ", ".join(shorts))
    if show_ud:
        print("----- user-data (%s) -----\n%s-----" % (shorts[0], render(shorts[0])))
    print("types     %s, on-demand only" % ", then ".join("%s $%.3f/h" % t for t in TYPES))
    print("regions   %s" % ", then ".join("%s (AMI %s)" % (r, A.REGION_CFG[r]["ami"] if r in A.REGION_CFG
                                                         else "from SSM " + A.AMI_PARAM) for r in REGIONS))
    print("limits    per-job timeout %d min (%.2f h), backstop shutdown +%d min, terminate on shutdown, %d GB gp3" % (
        TIMEOUT_MIN, TIMEOUT_MIN / 60.0, BACKSTOP_MIN, VOLUME_GB))
    todo = [s for s in shorts if not record.get(s, {}).get("instance_id")]
    committed = committed_worst(record)
    worst = committed + worst_usd(len(todo))
    print("worst     %d job(s) x %.2f h x $%.3f/h = $%.2f%s   cap $%.2f: %s" % (
        len(todo), BACKSTOP_MIN / 60.0, max(u for _, u in TYPES), worst_usd(len(todo)),
        " + $%.2f already launched" % committed if committed else "", CAP_USD,
        "OK" if worst <= CAP_USD else "OVER CAP, --submit will refuse"))
    return worst


# ---- read-only preflight ----------------------------------------------------------------------------
def preflight(shorts):
    print("\nPREFLIGHT (read-only: Hub GET, ssm get, ec2 describe, service-quotas get, s3api head, s3 ls)")
    for s in shorts:
        j, rev, _ = JOBS[s]
        try:
            with urllib.request.urlopen("https://huggingface.co/api/models/%s" % j, timeout=20) as r:
                d = json.loads(r.read())
            print("hub       %-34s gated=%s main=%s pinned=%s %s" % (j, d.get("gated"), d.get("sha", "?")[:12], rev[:12],
                                                                   "same" if d.get("sha") == rev else "MAIN MOVED (pin still used)"))
        except Exception as e:   # noqa: BLE001
            print("hub       %-34s check failed: %s" % (j, str(e)[:120]))
    want = {t for t, _ in TYPES}
    for region in REGIONS:
        try:
            ami = ami_for(region)
            img = ec2(region, "ec2", "describe-images", "--image-ids", ami)["Images"]
            offers = ec2(region, "ec2", "describe-instance-type-offerings", "--location-type", "availability-zone",
                         "--filters", "Name=instance-type,Values=" + ",".join(sorted(want)))["InstanceTypeOfferings"]
            azs = sorted({o["Location"] for o in offers} & set(subnets_for(region)))
            quota = ec2(region, "service-quotas", "get-service-quota", "--service-code", "ec2",
                        "--quota-code", "L-DB2E81BA")["Quota"]["Value"]
            live = ec2(region, "ec2", "describe-instances", "--filters",
                       "Name=instance-state-name,Values=pending,running")["Reservations"]
            used = sum(i["CpuOptions"]["CoreCount"] * i["CpuOptions"]["ThreadsPerCore"] for res in live
                       for i in res["Instances"] if i["InstanceType"].split(".")[0][:1] in ("g", "v"))
            print("region    %-9s AMI %s %s; g6e offered in %s; G/VT on-demand vCPU %d used of %d (need <= 8 per job)" % (
                region, ami, img[0]["State"] if img else "NOT FOUND", ",".join(azs) or "NO SUBNET AZ", used, quota))
        except RuntimeError as e:
            print("region    %-9s check failed: %s" % (region, str(e)[:200]))
    try:
        h = s3_head(DATA_KEY)
        print("s3 data   %s: %s" % (S3_DATA, "absent (uploaded on --submit)" if h is None else
                                    "present, metadata sha256=%s" % h.get("Metadata", {}).get("sha256", "?")))
        r = subprocess.run(["aws", "s3", "ls", "s3://%s/e1/rescore/" % A.BUCKET, "--recursive",
                            "--region", BUCKET_REGION], capture_output=True, text=True)
        keys = [ln.split()[-1] for ln in r.stdout.splitlines() if ln.strip()]
        print("s3 prefix e1/rescore/: %d object(s)%s" % (len(keys), (": " + ", ".join(keys[:8])) if keys else ""))
    except RuntimeError as e:
        print("s3        check failed: %s" % str(e)[:200])


# ---- submit -----------------------------------------------------------------------------------------
def upload_data(path, sha, record):
    h = s3_head(DATA_KEY)
    if h is not None:
        have = h.get("Metadata", {}).get("sha256")
        if have == sha:
            print("data already in S3 with sha256 %s: not re-uploaded" % sha[:16])
            return
        if any(v.get("instance_id") for v in record.values()):
            raise SystemExit("REFUSE: S3 data has sha256 %s but launched jobs exist; not overwriting their input" % have)
    if not s3_cp(path, S3_DATA, "--metadata", "sha256=" + sha):
        raise SystemExit("data upload failed: %s -> %s" % (path, S3_DATA))
    h = s3_head(DATA_KEY)
    if not h or h.get("Metadata", {}).get("sha256") != sha:
        raise SystemExit("data upload not confirmed by head-object")
    print("uploaded data %s (sha256 %s)" % (S3_DATA, sha[:16]))


def submit(shorts, record, data_path, template):
    todo = []
    for s in shorts:
        if record.get(s, {}).get("instance_id"):
            print("skip (already launched): %s %s" % (s, record[s]["instance_id"]))
        else:
            todo.append(s)
    if not todo:
        print("NOTHING TO LAUNCH")
        return 0
    worst = committed_worst(record) + worst_usd(len(todo))
    if worst > CAP_USD:
        raise SystemExit("REFUSE: worst case $%.2f exceeds the $%.2f cap" % (worst, CAP_USD))
    if not template.exists():
        raise SystemExit("REFUSE: template not found: %s" % template)
    try:
        data_sha, info = check_data(data_path)
    except FileNotFoundError as e:
        raise SystemExit("REFUSE: %s" % e)
    if info["n"] != N_EXPECTED or info["problems"]:
        raise SystemExit("REFUSE: data has %d completions (expected %d); problems: %s" % (
            info["n"], N_EXPECTED, info["problems"]))
    tmp = Path(tempfile.mkdtemp(prefix="e1rescore_"))     # no spaces: the aws exe takes file:// paths
    b64, pkg_commit = pkg_b64(), head_commit()
    baked = {s: bake(s, data_sha, b64, tmp, template) for s in todo}
    upload_data(data_path, data_sha, record)
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    saved = (A.REGION, A.SUBNETS, A.FALLBACK)
    try:
        for s in todo:
            judge, rev, role = JOBS[s]
            outp, job_sha = baked[s]
            if not s3_cp(outp, prefix(s) + "/job.py"):
                raise SystemExit("job.py upload failed for %s" % s)
            ud = tmp / ("%s.sh" % s)
            ud.write_text(render(s), encoding="ascii", newline="\n")
            inst = itype = usd = ami = None
            errs = []
            for region in REGIONS:
                A.REGION, A.SUBNETS, A.FALLBACK = region, subnets_for(region), list(TYPES)
                ami = ami_for(region)
                spec = {
                    "ImageId": ami, "MinCount": 1, "MaxCount": 1, "IamInstanceProfile": {"Name": A.PROFILE},
                    "InstanceInitiatedShutdownBehavior": "terminate",
                    "BlockDeviceMappings": [{"DeviceName": "/dev/sda1", "Ebs": {
                        "VolumeSize": VOLUME_GB, "VolumeType": "gp3", "DeleteOnTermination": True}}],
                    "MetadataOptions": {"HttpTokens": "required"},
                    "TagSpecifications": [{"ResourceType": "instance", "Tags": [
                        {"Key": "Name", "Value": "fr-e1-rescore-" + s}, {"Key": "Project", "Value": "flight-recorder-e1"},
                        {"Key": "Stage", "Value": "rescore"}]}],
                }   # no InstanceMarketOptions: on-demand only
                inst, itype, usd, err = BATCH.launch_with_fallback(spec, ud, tmp, 1, float("inf"))
                errs += ["%s/%s:%s" % (region, t, c) for t, c in err]
                if inst is not None:
                    break
            if inst is None:
                print("NO CAPACITY for %s: %s" % (s, ", ".join(sorted(set(errs)))))
                continue
            record[s] = {"judge_short": s, "judge": judge, "revision": rev, "role": role, "dtype": DTYPE,
                         "s3": prefix(s), "s3_data": S3_DATA, "data_sha256": data_sha, "job_sha256": job_sha,
                         "package_commit": pkg_commit,
                         "instance_id": inst["InstanceId"], "instance_type": itype, "region": A.REGION,
                         "az": inst["Placement"]["AvailabilityZone"], "ami": ami, "usd_per_h": usd,
                         "market": "ondemand", "launched": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                         "timeout_min": TIMEOUT_MIN, "backstop_min": BACKSTOP_MIN,
                         "worst_usd": round(BACKSTOP_MIN / 60.0 * usd, 2), "capacity_misses": sorted(set(errs))}
            RECORD.write_text(json.dumps(record, indent=1))
            print("LAUNCHED %-10s %s %s %s %s $%.3f/h (worst $%.2f)" % (
                s, inst["InstanceId"], A.REGION, inst["Placement"]["AvailabilityZone"], itype, usd,
                BACKSTOP_MIN / 60.0 * usd))
    finally:
        A.REGION, A.SUBNETS, A.FALLBACK = saved
    print("launch record: %s" % rel(RECORD))
    return 0


# ---- status / collect (counts only, never score values) ---------------------------------------------
def count_scores(doc):
    """Scored completions: non-null entries. The job allocates a step's 8 slots (null) when the first of them is
    scored, and length-sorted batches fill a step piecemeal, so counting slots would overstate progress."""
    return sum(1 for steps in (doc.get("scores") or {}).values() for v in steps.values() for x in (v or [])
               if x is not None)


def progress(doc, name):
    n = count_scores(doc)
    extra = "" if doc.get("n_scored") in (None, n) else " (n_scored field says %s)" % doc.get("n_scored")
    return "%s: %d / %d scores (%.1f%%)%s" % (name, n, N_EXPECTED, 100.0 * n / N_EXPECTED, extra)


def status(record):
    if not record:
        print("no launch record at %s" % rel(RECORD))
        return 0
    state = {}
    for region in sorted({v["region"] for v in record.values()}):
        ids = [v["instance_id"] for v in record.values() if v["region"] == region]
        for res in ec2(region, "ec2", "describe-instances", "--instance-ids", *ids)["Reservations"]:
            for i in res["Instances"]:
                state[i["InstanceId"]] = (i["State"]["Name"], i.get("StateTransitionReason", ""))
    now = datetime.now(timezone.utc)
    for s, v in record.items():
        st, why = state.get(v["instance_id"], ("?", ""))
        line = (s3_text(v["s3"] + "/status") or "-").strip()
        prog = "no partial.json yet"
        for name in ("result.json", "partial.json"):
            txt = s3_text(v["s3"] + "/" + name)
            if txt:
                try:
                    prog = progress(json.loads(txt), name)
                except ValueError:
                    prog = "%s unreadable" % name
                break
        up_h = (now - datetime.fromisoformat(v["launched"])).total_seconds() / 3600.0
        m = re.search(r"\((\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) GMT\)", why or "")
        if st in ("shutting-down", "terminated", "stopped") and m:
            end = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            up_h = (end - datetime.fromisoformat(v["launched"])).total_seconds() / 3600.0
        up_h = min(up_h, v.get("backstop_min", BACKSTOP_MIN) / 60.0)
        print("%-10s %s %-9s %-11s %-12s up %.2f h (~$%.2f)  status: %s\n           %s" % (
            s, v["instance_id"], v["region"], v["instance_type"], st, up_h, up_h * v["usd_per_h"], line[:40], prog))
    return 0


def collect(record):
    if not record:
        print("no launch record at %s" % rel(RECORD))
        return 0
    (RESULTS / "logs").mkdir(parents=True, exist_ok=True)
    for s, v in record.items():
        dest = RESULTS / ("%s.json" % s)
        got = next((name for name in ("result.json", "partial.json") if s3_cp(v["s3"] + "/" + name, dest)), None)
        for name in ("job.log", "boot.log", "status"):
            s3_cp(v["s3"] + "/" + name, RESULTS / "logs" / ("%s.%s" % (s, name.replace(".log", "") + ".log")))
        if got is None:
            print("%-10s nothing to collect yet (no result.json or partial.json)" % s)
            continue
        doc = json.loads(dest.read_text())
        checks = [k for k, want in (("judge", v["judge"]), ("revision", v["revision"]), ("data_sha256", v["data_sha256"]))
                  if doc.get(k) != want]
        label = ("PARTIAL" if got == "partial.json" else
                 "ABORTED" if doc.get("aborted") else
                 "COMPLETE" if doc.get("complete") and count_scores(doc) == N_EXPECTED else "RESULT-INCOMPLETE")
        print("%-10s %-17s -> %s  %s%s" % (s, label, rel(dest),
                                         progress(doc, got), "  MISMATCH: %s" % checks if checks else "  (judge, revision, data sha ok)"))
    return 0


def main(argv):
    opt = {a.split("=", 1)[0]: (a.split("=", 1)[1] if "=" in a else True) for a in argv}
    unknown = set(opt) - {"--submit", "--status", "--collect", "--preflight", "--only", "--data", "--template",
                          "--show-userdata"}
    if unknown:
        raise SystemExit("unknown option(s): %s" % ", ".join(sorted(unknown)))
    shorts = [s.strip() for s in opt["--only"].split(",")] if isinstance(opt.get("--only"), str) else list(JOBS)
    bad = [s for s in shorts if s not in JOBS]
    if bad:
        raise SystemExit("unknown job(s) %s; jobs: %s" % (bad, ", ".join(JOBS)))
    data_path = Path(opt["--data"]) if isinstance(opt.get("--data"), str) else DATA_GZ
    template = Path(opt["--template"]) if isinstance(opt.get("--template"), str) else TEMPLATE
    record = load_record()
    if "--status" in opt:
        return status({s: v for s, v in record.items() if s in shorts})
    if "--collect" in opt:
        return collect({s: v for s, v in record.items() if s in shorts})
    worst = print_plan(shorts, record, data_path, template, "--show-userdata" in opt)
    if "--preflight" in opt:
        preflight(shorts)
    if "--submit" not in opt:
        print("\nDRY RUN: nothing uploaded or launched, no launch record written. Add --submit to spend.")
        return 0
    if worst > CAP_USD:
        raise SystemExit("REFUSE: worst case $%.2f exceeds the $%.2f cap" % (worst, CAP_USD))
    return submit(shorts, record, data_path, template)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
