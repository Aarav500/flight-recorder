"""E1 on SageMaker training jobs: one job per run, the same baked job as HF Jobs and EC2. ASCII only.

WHY SAGEMAKER
  On 2026-10-07 EC2 had no capacity in any us-east-1 AZ for g6e (all sizes), g7e or p5.4xlarge,
  spot or on-demand. SageMaker training draws on separate capacity, and this account has a quota of
  1 for each of ml.g6e.xlarge / ml.g6e.8xlarge / ml.g6e.16xlarge (1x L40S 48 GB each). So a stage of
  up to 3 runs goes out as 3 concurrent jobs, one per instance type. Only the CPU count and the price
  differ between them.

WHAT RUNS
  The stage rows come from launch_e1_aws.STAGES, so the config is the one already committed. Each job
  gets a code channel holding job.py (the bake from launch_slow_onset.bake) and run.sh (below). It runs
  the job under uv with Python 3.12, tees the log to CloudWatch, and writes job.log, result.json and a
  status line to /opt/ml/output/data. That directory is uploaded to S3 as output.tar.gz. The role
  fr-e1-sagemaker can only read and write s3://BUCKET/e1/*, pull the image and write logs. No secrets
  are passed; the models are public.

    python scripts/launch_e1_sagemaker.py cap512             # dry run: price and print
    python scripts/launch_e1_sagemaker.py cap512 --submit    # spend (AWS credits)
    python scripts/launch_e1_sagemaker.py cap512 --status
    python scripts/launch_e1_sagemaker.py cap512 --collect   # -> runs/e1/<stage>/<run_id>.json
"""
from __future__ import annotations

import base64
import gzip
import io
import json
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import launch_e1_aws as AWS  # noqa: E402
import launch_slow_onset as LSO  # noqa: E402

ROLE = "arn:aws:iam::325527186655:role/fr-e1-sagemaker"
IMAGE = "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.7.1-gpu-py312-cu128-ubuntu22.04-sagemaker"
# (instance type, USD/h for training, from the AWS pricing API 2026-10-07). Quota is 1 of each.
TYPES = [("ml.g6e.xlarge", 2.61), ("ml.g6e.8xlarge", 5.66), ("ml.g6e.16xlarge", 9.47)]
STEP_S = 20.0   # ESTIMATE for one L40S at 512 tokens; the CloudWatch log gives the measured value

RUN_SH = r"""#!/bin/bash
set -u
OUT=/opt/ml/output/data
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -m pip install -q uv
cd /tmp
python -m uv run --python 3.12 /opt/ml/input/data/code/job.py 2>&1 | tee "$OUT/job.log"
rc=${PIPESTATUS[0]}
cp /tmp/result.json "$OUT/result.json" 2>/dev/null
echo "rc=$rc $(date -u +%FT%TZ)" > "$OUT/status"
exit 0
"""


def aws(*args):
    r = subprocess.run(["aws", *args, "--region", "us-east-1", "--output", "json"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("aws %s failed: %s" % (" ".join(args[:2]), (r.stderr or "").strip()[-600:]))
    return json.loads(r.stdout) if r.stdout.strip() else None


def record_path(stage):
    return ROOT / "runs" / "e1" / ("%s_sagemaker.json" % stage)


def job_name(stage, r, stamp):
    return ("fr-e1-%s-%s-s%d-%s" % (stage, r["task"].replace("_", "-"), r["seed"], stamp))[:63]


def est(r):
    h = 0.3 + r["steps"] * STEP_S / 3600.0
    return h, int(h * 2.0) + 2


def submit(stage, rows):
    rec = json.loads(record_path(stage).read_text()) if record_path(stage).exists() else {}
    stamp = datetime.now(timezone.utc).strftime("%m%d%H%M")
    tmp = Path(tempfile.mkdtemp(prefix="e1sm_"))
    (tmp / "run.sh").write_text(RUN_SH, encoding="ascii", newline="\n")
    free = [t for t in TYPES if t[0] not in {v["instance_type"] for v in rec.values()}]
    for r in rows:
        if r["run_id"] in rec:
            print("skip (already submitted): %s %s" % (r["run_id"], rec[r["run_id"]]["job"]))
            continue
        if not free:
            print("NO FREE INSTANCE TYPE for %s (quota is 1 per type); submit it after one finishes" % r["run_id"])
            continue
        itype, usd = free.pop(0)
        _, outp = LSO.bake(r["arm"], r["task"], r["seed"], r["steps"], r["lr"], AWS.POLICY, r["judge"],
                           r["probe_n"], r["probe_every"], r["maxtok"])
        code = r["s3"] + "/code"
        subprocess.run(["aws", "s3", "cp", str(outp), code + "/job.py", "--only-show-errors"], check=True)
        subprocess.run(["aws", "s3", "cp", str(tmp / "run.sh"), code + "/run.sh", "--only-show-errors"], check=True)
        h, max_h = est(r)
        name = job_name(stage, r, stamp)
        spec = {
            "TrainingJobName": name,
            "AlgorithmSpecification": {"TrainingImage": IMAGE, "TrainingInputMode": "File",
                                       "ContainerEntrypoint": ["bash", "/opt/ml/input/data/code/run.sh"]},
            "RoleArn": ROLE,
            "InputDataConfig": [{"ChannelName": "code", "DataSource": {"S3DataSource": {
                "S3DataType": "S3Prefix", "S3Uri": code + "/", "S3DataDistributionType": "FullyReplicated"}}}],
            "OutputDataConfig": {"S3OutputPath": r["s3"] + "/sm-output/"},
            "ResourceConfig": {"InstanceType": itype, "InstanceCount": 1, "VolumeSizeInGB": 150},
            "StoppingCondition": {"MaxRuntimeInSeconds": max_h * 3600},
            "Tags": [{"Key": "Project", "Value": "flight-recorder-e1"}, {"Key": "Stage", "Value": stage}],
        }
        sp = tmp / (name + ".json")
        sp.write_text(json.dumps(spec), encoding="ascii")
        aws("sagemaker", "create-training-job", "--cli-input-json", "file://" + sp.as_posix())
        rec[r["run_id"]] = dict(r, job=name, instance_type=itype, usd_per_h=usd, est_h=round(h, 2), max_h=max_h,
                                submitted=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        record_path(stage).write_text(json.dumps(rec, indent=1))
        print("SUBMITTED %-46s %s on %s ($%.2f/h, est %.1f h, max %d h)" % (r["run_id"], name, itype, usd, h, max_h))


def _last_onset(name):
    try:
        streams = aws("logs", "describe-log-streams", "--log-group-name", "/aws/sagemaker/TrainingJobs",
                      "--log-stream-name-prefix", name)["logStreams"]
        if not streams:
            return "(no log stream yet)"
        ev = aws("logs", "get-log-events", "--log-group-name", "/aws/sagemaker/TrainingJobs",
                 "--log-stream-name", streams[0]["logStreamName"], "--no-start-from-head", "--limit", "60")["events"]
        lines = [e["message"] for e in ev]
        onset = [ln for ln in lines if ln.startswith("ONSET step")]
        err = [ln for ln in lines if "Traceback" in ln or "Error" in ln]
        return (onset[-1][:110] if onset else (err[-1][:110] if err else (lines[-1][:110] if lines else "(empty)")))
    except RuntimeError as e:
        return "(log read failed: %s)" % str(e)[-80:]


def status(stage):
    for rid, v in json.loads(record_path(stage).read_text()).items():
        d = aws("sagemaker", "describe-training-job", "--training-job-name", v["job"])
        print("%-46s %-10s %-22s %s" % (rid, d["TrainingJobStatus"], d.get("SecondaryStatus", ""), _last_onset(v["job"])))
        if d.get("FailureReason"):
            print("    failure: %s" % d["FailureReason"][:300])


def collect(stage):
    out = ROOT / "runs" / "e1" / stage
    (out / "logs").mkdir(parents=True, exist_ok=True)
    for rid, v in json.loads(record_path(stage).read_text()).items():
        d = aws("sagemaker", "describe-training-job", "--training-job-name", v["job"])
        if d["TrainingJobStatus"] not in ("Completed", "Failed", "Stopped"):
            print("%-46s still %s" % (rid, d["TrainingJobStatus"]))
            continue
        uri = d.get("ModelArtifacts", {}).get("S3ModelArtifacts", "").replace("model.tar.gz", "output.tar.gz")
        if not uri:
            uri = "%s/sm-output/%s/output/output.tar.gz" % (v["s3"], v["job"])
        r = subprocess.run(["aws", "s3", "cp", uri, "-"], capture_output=True)
        if r.returncode != 0:
            print("%-46s no output.tar.gz (%s)" % (rid, d["TrainingJobStatus"]))
            continue
        files = {}
        with tarfile.open(fileobj=io.BytesIO(r.stdout), mode="r:gz") as t:
            for m in t.getmembers():
                if m.isfile():
                    files[Path(m.name).name] = t.extractfile(m).read()
        if "job.log" in files:
            (out / "logs" / (rid + ".log")).write_bytes(files["job.log"])
        res = None
        if "result.json" in files:
            res = json.loads(files["result.json"])
        elif "job.log" in files:                      # reassemble from the RESULT_B64 chunks
            chunks = {}
            for ln in files["job.log"].decode("utf-8", "replace").splitlines():
                if ln.startswith("RESULT_B64 "):
                    _, idx, data = ln.split(" ", 2)
                    i, k = (int(x) for x in idx.split("/"))
                    chunks[i] = data.strip()
            if chunks and len(chunks) == max(chunks):
                res = json.loads(gzip.decompress(base64.b64decode("".join(chunks[i] for i in sorted(chunks)))))
        if res is not None:
            (out / (rid + ".json")).write_text(json.dumps(res))
        print("%-46s %-9s status=%s result=%s billable=%ss" % (
            rid, d["TrainingJobStatus"], files.get("status", b"?").decode().strip(), "ok" if res else "MISSING",
            d.get("BillableTimeInSeconds")))


def main(argv):
    stage = argv[0] if argv and not argv[0].startswith("--") else "cap512"
    if "--status" in argv:
        return status(stage)
    if "--collect" in argv:
        return collect(stage)
    rows = AWS.plan(stage)
    tot = 0.0
    for r, (itype, usd) in zip(rows, TYPES):
        h, max_h = est(r)
        tot += h * usd
        print("%-46s %s  est %.1f h  max %d h  ~$%.2f" % (r["run_id"], itype, h, max_h, h * usd))
    print("TOTAL estimate ~$%.0f (AWS credits). Step time is an ESTIMATE." % tot)
    if "--submit" not in argv:
        print("DRY RUN: nothing uploaded or submitted. Add --submit to spend.")
        return 0
    submit(stage, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
