"""E1 batch mode on EC2: ONE instance runs several queued jobs back to back. ASCII only.

WHY
  On 2026-10-07 GPU capacity in us-east-1 was scarce: every g6e size, g7e and p5.4xlarge returned
  InsufficientInstanceCapacity for hours, then a single g6e.4xlarge slot opened in us-east-1a. A
  slot, once won, is therefore used for every queued run of the stage, one after another.

HOW
  Reuses launch_e1_aws: the same stages, bake, IAM role, S3 layout and type/AZ fallback. Each run keeps
  its own S3 prefix and status line, so `launch_e1_aws.py <stage> --status/--collect` work unchanged.
  Two safety rails:
    * A shutdown is scheduled at boot (per-run timeout x runs + 30 min). Even if a job hangs past its
      own timeout, the instance terminates. InstanceInitiatedShutdownBehavior=terminate.
    * SageMaker twins (launch_e1_sagemaker.py) are checked first. A run whose SageMaker job has
      already started is skipped. Once the EC2 batch launches, the twins still waiting for capacity
      are stopped, so nothing runs twice.

    python scripts/launch_e1_batch.py cap512                       # dry run: what would launch
    python scripts/launch_e1_batch.py cap512 --submit --ondemand   # spend (AWS credits)
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import launch_e1_aws as A  # noqa: E402

STEP_S_MEASURED = 11.0   # g6e.4xlarge, 512-token cap, 2026-10-07 (is_prime run: 134 steps in 27:36)

USER_DATA_BATCH = r"""#!/bin/bash
set -u
mkdir -p /opt/e1 && cd /opt/e1
exec > /opt/e1/boot.log 2>&1
BOOT="__BOOT__"
shutdown -h +__BACKSTOP_MIN__
finish_all() {
  aws s3 cp /opt/e1/boot.log "$BOOT/boot.log" --only-show-errors
  shutdown -h now
}
trap finish_all EXIT
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python3 -m pip install -q uv || { apt-get update -q && apt-get install -yq python3-pip && python3 -m pip install -q uv; }
export HF_HOME=/opt/e1/hf UV_CACHE_DIR=/opt/e1/uvcache PYTHONUNBUFFERED=1
for B in __PREFIXES__; do
  rm -f /tmp/result.json /opt/e1/job.log /opt/e1/job.py
  aws s3 cp "$B/job.py" /opt/e1/job.py --only-show-errors || { echo "no job.py at $B"; continue; }
  echo "started $(date -u +%FT%TZ)" > /opt/e1/status && aws s3 cp /opt/e1/status "$B/status" --only-show-errors
  ( while sleep 300; do aws s3 cp /opt/e1/job.log "$B/job.log" --only-show-errors; done ) &
  LOOP=$!
  timeout __PER_RUN_H__h python3 -m uv run --python 3.12 /opt/e1/job.py > /opt/e1/job.log 2>&1
  RC=$?
  kill $LOOP 2>/dev/null
  aws s3 cp /opt/e1/job.log "$B/job.log" --only-show-errors
  [ -f /tmp/result.json ] && aws s3 cp /tmp/result.json "$B/result.json" --only-show-errors
  echo "rc=$RC $(date -u +%FT%TZ)" > /opt/e1/status && aws s3 cp /opt/e1/status "$B/status" --only-show-errors
done
"""


def render(keep, per_run_h, backstop_min):
    ud = (USER_DATA_BATCH.replace("__BOOT__", keep[0]["s3"]).replace("__BACKSTOP_MIN__", str(backstop_min))
          .replace("__PER_RUN_H__", str(per_run_h)).replace("__PREFIXES__", " ".join(r["s3"] for r in keep)))
    left = re.findall(r"__[A-Z_]+__", ud)
    assert not left, "unsubstituted: %s" % left
    return ud


def launch_with_fallback(spec, ud_path, tmp):
    err = []
    for itype, usd in A.FALLBACK:
        spec["InstanceType"] = itype
        for az, subnet in A.SUBNETS.items():
            spec["SubnetId"] = subnet
            sp = tmp / "batch.json"
            sp.write_text(json.dumps(spec), encoding="ascii")
            try:
                inst = A.aws("ec2", "run-instances", "--cli-input-json", "file://" + sp.as_posix(),
                             "--user-data", "file://" + ud_path.as_posix())["Instances"][0]
                return inst, itype, usd, err
            except RuntimeError as e:
                msg = str(e)
                code = msg.split("An error occurred (")[-1].split(")")[0] if "An error occurred (" in msg else msg[-80:]
                if code not in ("InsufficientInstanceCapacity", "SpotMaxPriceTooLow", "Unsupported",
                                "VcpuLimitExceeded", "MaxSpotInstanceCountExceeded"):
                    raise
                err.append((itype, code))
                if code == "VcpuLimitExceeded":
                    break
    return None, None, None, err


def sagemaker_state(stage):
    """(run_id -> pending job name, run_id -> started status, record path)."""
    path = ROOT / "runs" / "e1" / ("%s_sagemaker.json" % stage)
    if not path.exists():
        return {}, {}, path
    pend, started = {}, {}
    for rid, v in json.loads(path.read_text()).items():
        if "stopped" in v:
            continue
        d = A.aws("sagemaker", "describe-training-job", "--training-job-name", v["job"])
        if d["TrainingJobStatus"] == "InProgress" and d.get("SecondaryStatus") == "Pending":
            pend[rid] = v["job"]
        else:
            started[rid] = "%s/%s" % (d["TrainingJobStatus"], d.get("SecondaryStatus"))
    return pend, started, path


def main(argv):
    stage = argv[0] if argv and not argv[0].startswith("--") else "cap512"
    market = "ondemand" if "--ondemand" in argv else "spot"
    rows = A.plan(stage)
    record_path = ROOT / "runs" / "e1" / ("%s_launch.json" % stage)
    record = json.loads(record_path.read_text()) if record_path.exists() else {}
    pend, started, sm_path = sagemaker_state(stage)
    keep = []
    for r in rows:
        if record.get(r["run_id"], {}).get("instance_id"):
            print("skip (already ran on EC2): %s" % r["run_id"])
        elif r["run_id"] in started:
            print("skip (SageMaker job already %s): %s" % (started[r["run_id"]], r["run_id"]))
        else:
            keep.append(r)
    if not keep:
        print("NOTHING TO LAUNCH")
        return 0
    per_run_h = max(2, int(math.ceil(max(r["steps"] for r in keep) * STEP_S_MEASURED * 2 / 3600.0)) + 1)
    backstop_min = per_run_h * len(keep) * 60 + 30
    ud = render(keep, per_run_h, backstop_min)
    worst = backstop_min / 60.0 * max(u for _, u in A.FALLBACK)
    print("batch of %d: %s\n  per-run timeout %d h, backstop shutdown at +%d min (worst case <= $%.0f), market %s" % (
        len(keep), ", ".join(r["run_id"] for r in keep), per_run_h, backstop_min, worst, market))
    if "--submit" not in argv:
        print("DRY RUN: nothing baked, uploaded or launched.")
        return 0
    ami = A.aws("ssm", "get-parameter", "--name", A.AMI_PARAM)["Parameter"]["Value"]
    tmp = Path(tempfile.mkdtemp(prefix="e1batch_"))
    for r in keep:
        _, outp = A.LSO.bake(r["arm"], r["task"], r["seed"], r["steps"], r["lr"], A.POLICY, r["judge"],
                             r["probe_n"], r["probe_every"], r["maxtok"])
        subprocess.run(["aws", "s3", "cp", str(outp), r["s3"] + "/job.py", "--only-show-errors"], check=True)
    ud_path = tmp / "batch.sh"
    ud_path.write_text(ud, encoding="ascii", newline="\n")
    spec = {
        "ImageId": ami, "MinCount": 1, "MaxCount": 1, "IamInstanceProfile": {"Name": A.PROFILE},
        "InstanceInitiatedShutdownBehavior": "terminate",
        "BlockDeviceMappings": [{"DeviceName": "/dev/sda1", "Ebs": {"VolumeSize": 100, "VolumeType": "gp3",
                                                                    "DeleteOnTermination": True}}],
        "MetadataOptions": {"HttpTokens": "required"},
        "TagSpecifications": [{"ResourceType": "instance", "Tags": [
            {"Key": "Name", "Value": "fr-e1-%s-batch" % stage}, {"Key": "Project", "Value": "flight-recorder-e1"},
            {"Key": "Stage", "Value": stage}]}],
    }
    if market == "spot":
        spec["InstanceMarketOptions"] = {"MarketType": "spot", "SpotOptions": {
            "SpotInstanceType": "one-time", "InstanceInterruptionBehavior": "terminate"}}
    inst, itype, usd, err = launch_with_fallback(spec, ud_path, tmp)
    if inst is None:
        print("NO CAPACITY for the batch: %s" % sorted(set(err)))
        return 0
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for r in keep:
        record[r["run_id"]] = dict(r, instance_id=inst["InstanceId"], az=inst["Placement"]["AvailabilityZone"],
                                   instance_type=itype, usd_per_h=usd, market=market, ami=ami, launched=now,
                                   batch=[k["run_id"] for k in keep], per_run_timeout_h=per_run_h,
                                   backstop_min=backstop_min)
    record_path.write_text(json.dumps(record, indent=1))
    print("LAUNCHED batch on %s %s %s $%.2f/h: %s" % (inst["InstanceId"], inst["Placement"]["AvailabilityZone"],
                                                       itype, usd, ", ".join(r["run_id"] for r in keep)))
    if pend:                      # the EC2 batch now covers these runs: stop their queued SageMaker twins
        sm = json.loads(sm_path.read_text())
        for r in keep:
            if r["run_id"] in pend:
                A.aws("sagemaker", "stop-training-job", "--training-job-name", pend[r["run_id"]])
                sm[r["run_id"]]["stopped"] = "%s: superseded by EC2 batch %s before it started" % (now, inst["InstanceId"])
                print("stopped SageMaker twin %s (still Pending, never billed)" % pend[r["run_id"]])
        sm_path.write_text(json.dumps(sm, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
