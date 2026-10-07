"""E1 on AWS EC2 spot: one g6e.xlarge (1x L40S 48 GB) per run, self-terminating. ASCII only.

Bakes each job with launch_slow_onset.bake (the same job HF Jobs would run, byte for byte), uploads
it to s3://BUCKET/e1/<stage>/<run_id>/job.py, and starts one one-time spot instance per run. The
instance installs uv from PyPI, runs the job under a hard timeout, uploads job.log, result.json and
a status line, then shuts down; InstanceInitiatedShutdownBehavior=terminate turns that into a
termination, so an idle GPU cannot keep billing. A live copy of job.log is pushed every 5 minutes.

No secrets reach the instance. The models are public on the Hub, and the instance role
(fr-e1-runner) can only read and write s3://BUCKET/e1/*, plus the SSM core policy for a shell.

Uses the aws CLI through subprocess, because boto3 cannot read an `aws login` session.

    python scripts/launch_e1_aws.py pilot             # dry run: bake, price, print. Spends nothing.
    python scripts/launch_e1_aws.py pilot --submit    # spend (spot)
    python scripts/launch_e1_aws.py cap512 --submit --ondemand   # spend, on-demand (G/VT on-demand quota 32)
    python scripts/launch_e1_aws.py pilot --status    # instance state + last logged step per run
    python scripts/launch_e1_aws.py pilot --collect   # -> runs/e1/pilot/<run_id>.json, logs/<run_id>.log

STAGES are fixed here, not on the command line, so what ran is what the commit says ran.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import launch_slow_onset as LSO  # noqa: E402  (has a __main__ guard; importing runs nothing)

BUCKET = "fr-deploy-325527186655"
PROFILE = "fr-e1-runner"
ITYPE = "g6e.xlarge"
# Capacity is per type and per AZ. On 2026-10-07 g6e.xlarge had none, spot or on-demand, in any
# us-east-1 AZ, so each run falls back through these in order: cheapest first, all with >= 48 GB of
# GPU memory. On-demand USD/h is from the AWS pricing API, 2026-10-07. A10G and L4 (24 GB) are
# excluded: 1.5B GRPO with fp32 Adam and a reference model OOMs at 24 GB.
FALLBACK = [("g6e.xlarge", 1.861), ("g6e.2xlarge", 2.242), ("g6e.4xlarge", 3.004), ("g7e.2xlarge", 3.363)]
AMI_PARAM = "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"
SUBNETS = {"us-east-1a": "subnet-0386677a3484341be", "us-east-1b": "subnet-050b222e93f526cd8",
           "us-east-1c": "subnet-042747a96276b6f96", "us-east-1d": "subnet-028805185f69156e2"}
USD_PER_HR = 1.84          # g6e.xlarge spot, us-east-1, 2026-10-07 (spot ~= on-demand for g6e today)
STEP_S_L40S = 26.0         # ESTIMATE, not measured: 1.6 x the A100's measured 16 s (LSO.STEP_S_1P5B).
                           # The pilot's job.log measures it; update this before sizing the main runs.
SETUP_H = 0.3              # uv + torch install, model download
POLICY = "Qwen/Qwen2.5-1.5B"
SMALL_JUDGE = "Qwen/Qwen2.5-0.5B-Instruct"     # reduced-capacity: the published hacking arm
TASKS = ["count_vowels", "is_prime", "second_largest"]   # the published grid's hacking tasks

# stage -> list of (arm, judge, task, seed, lr, steps, probe_n, probe_every, maxtok)
# Pilot: plan E1, "learning rates 1e-6 and 5e-7, 3 runs each" = one run per grid task.
# Step budgets hold lr x steps at the grid's 2e-6 x 200; probe cadence keeps ~81 probes per run.
# Seed 100 is new: seeds 0-2 are the frozen grid, 3-7 were baked for the heuristic arm.
STAGES = {
    "pilot": [("judge", SMALL_JUDGE, t, 100, lr, steps, 16, every, 256)
              for lr, steps, every in (("1e-6", 400, 5), ("5e-7", 800, 10)) for t in TASKS],
    # Token-cap test (experiments/e1_cap512_analysis.py): the published GRID config with
    # max_completion_length 512 instead of 256. Same seed as the 256-token GRID diagnostic runs.
    "cap512": [("judge", SMALL_JUDGE, t, 0, "2e-6", 200, 0, 5, 512) for t in TASKS],
}

USER_DATA = r"""#!/bin/bash
set -u
B="__S3__"
mkdir -p /opt/e1 && cd /opt/e1
exec > /opt/e1/boot.log 2>&1
RC=boot-failed
finish() {
  echo "rc=$RC $(date -u +%FT%TZ)" > /opt/e1/status
  aws s3 cp /opt/e1/boot.log "$B/boot.log" --only-show-errors
  [ -f /opt/e1/job.log ] && aws s3 cp /opt/e1/job.log "$B/job.log" --only-show-errors
  [ -f /tmp/result.json ] && aws s3 cp /tmp/result.json "$B/result.json" --only-show-errors
  aws s3 cp /opt/e1/status "$B/status" --only-show-errors
  shutdown -h now
}
trap finish EXIT
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python3 -m pip install -q uv || { apt-get update -q && apt-get install -yq python3-pip && python3 -m pip install -q uv; }
aws s3 cp "$B/job.py" /opt/e1/job.py --only-show-errors
echo "started $(date -u +%FT%TZ)" > /opt/e1/status && aws s3 cp /opt/e1/status "$B/status" --only-show-errors
( while sleep 300; do aws s3 cp /opt/e1/job.log "$B/job.log" --only-show-errors; done ) &
export HF_HOME=/opt/e1/hf UV_CACHE_DIR=/opt/e1/uvcache PYTHONUNBUFFERED=1
timeout __TIMEOUT_H__h python3 -m uv run --python 3.12 /opt/e1/job.py > /opt/e1/job.log 2>&1
RC=$?
"""


def aws(*args, capture=True):
    r = subprocess.run(["aws", *args, "--region", "us-east-1", "--output", "json"],
                       capture_output=capture, text=True)
    if r.returncode != 0:
        raise RuntimeError("aws %s failed: %s" % (" ".join(args[:2]), (r.stderr or "").strip()[-600:]))
    return json.loads(r.stdout) if capture and r.stdout.strip() else None


def run_id(arm, judge, task, seed, lr, maxtok=256):
    j = "judge05" if judge == SMALL_JUDGE else "judge15"
    return "%s_%s_%s_s%d_lr%s%s" % (arm, j, task, seed, lr, "" if maxtok == 256 else "_tok%d" % maxtok)


def est_hours(steps, probe_n, probe_every, maxtok=256):
    train = steps * STEP_S_L40S * (1.6 if maxtok > 256 else 1.0) / 3600.0   # 512 tokens: ESTIMATE
    probes = (steps // probe_every + 1) * probe_n * 1.6 * (LSO.GEN_S_256TOK + LSO.SANDBOX_S) / 3600.0
    return SETUP_H + train + probes


def plan(stage):
    rows = []
    for arm, judge, task, seed, lr, steps, pn, pe, mt in STAGES[stage]:
        h = est_hours(steps, pn, pe, mt)
        rows.append({"run_id": run_id(arm, judge, task, seed, lr, mt), "arm": arm, "judge": judge, "task": task,
                     "seed": seed, "lr": lr, "steps": steps, "probe_n": pn, "probe_every": pe, "maxtok": mt,
                     "est_h": round(h, 2), "timeout_h": int(math.ceil(h * 1.8)) + 1,
                     "s3": "s3://%s/e1/%s/%s" % (BUCKET, stage, run_id(arm, judge, task, seed, lr, mt))})
    return rows


def print_plan(rows):
    print("%-48s %6s %6s %6s %8s" % ("run", "steps", "est_h", "max_h", "est_usd"))
    for r in rows:
        print("%-48s %6d %6.2f %6d %8.2f" % (r["run_id"], r["steps"], r["est_h"], r["timeout_h"], r["est_h"] * USD_PER_HR))
    est = sum(r["est_h"] for r in rows) * USD_PER_HR
    worst = sum(r["timeout_h"] for r in rows) * USD_PER_HR
    print("TOTAL  %d runs   estimate $%.0f   worst case (every run hits its timeout) $%.0f   at $%.2f/h"
          % (len(rows), est, worst, USD_PER_HR))
    print("step time is an ESTIMATE (%.0f s/step on L40S); the first job.log gives the measured value" % STEP_S_L40S)


def submit(stage, rows, market="spot"):
    ami = aws("ssm", "get-parameter", "--name", AMI_PARAM)["Parameter"]["Value"]
    tmp = Path(tempfile.mkdtemp(prefix="e1aws_"))         # no spaces: the aws exe takes file:// paths
    record_path = ROOT / "runs" / "e1" / ("%s_launch.json" % stage)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record = json.loads(record_path.read_text()) if record_path.exists() else {}
    for r in rows:
        if r["run_id"] in record and record[r["run_id"]].get("instance_id"):
            print("skip (already launched): %s %s" % (r["run_id"], record[r["run_id"]]["instance_id"]))
            continue
        _, outp = LSO.bake(r["arm"], r["task"], r["seed"], r["steps"], r["lr"], POLICY, r["judge"],
                           r["probe_n"], r["probe_every"], r["maxtok"])
        subprocess.run(["aws", "s3", "cp", str(outp), r["s3"] + "/job.py", "--only-show-errors"], check=True)
        ud = tmp / ("%s.sh" % r["run_id"])
        ud.write_text(USER_DATA.replace("__S3__", r["s3"]).replace("__TIMEOUT_H__", str(r["timeout_h"])),
                      encoding="ascii", newline="\n")
        spec = {
            "ImageId": ami, "InstanceType": ITYPE, "MinCount": 1, "MaxCount": 1,
            "IamInstanceProfile": {"Name": PROFILE},
            "InstanceInitiatedShutdownBehavior": "terminate",
            "BlockDeviceMappings": [{"DeviceName": "/dev/sda1", "Ebs": {
                "VolumeSize": 100, "VolumeType": "gp3", "DeleteOnTermination": True}}],
            "MetadataOptions": {"HttpTokens": "required"},
            "TagSpecifications": [{"ResourceType": "instance", "Tags": [
                {"Key": "Name", "Value": "fr-e1-" + r["run_id"]}, {"Key": "Project", "Value": "flight-recorder-e1"},
                {"Key": "Stage", "Value": stage}]}],
        }
        if market == "spot":
            spec["InstanceMarketOptions"] = {"MarketType": "spot", "SpotOptions": {
                "SpotInstanceType": "one-time", "InstanceInterruptionBehavior": "terminate"}}
        inst, itype_used, usd_used, err = None, None, None, []
        for itype, usd in FALLBACK:
            spec["InstanceType"] = itype
            for az, subnet in SUBNETS.items():
                spec["SubnetId"] = subnet
                sp = tmp / ("%s.json" % r["run_id"])
                sp.write_text(json.dumps(spec), encoding="ascii")
                try:
                    inst = aws("ec2", "run-instances", "--cli-input-json", "file://" + sp.as_posix(),
                               "--user-data", "file://" + ud.as_posix())["Instances"][0]
                    itype_used, usd_used = itype, usd
                    break
                except RuntimeError as e:
                    msg = str(e)
                    code = msg.split("An error occurred (")[-1].split(")")[0] if "An error occurred (" in msg else msg[-80:]
                    if code not in ("InsufficientInstanceCapacity", "SpotMaxPriceTooLow", "Unsupported",
                                    "VcpuLimitExceeded", "MaxSpotInstanceCountExceeded"):
                        raise
                    err.append("%s/%s:%s" % (itype, az, code))
                    if code == "VcpuLimitExceeded":   # quota, not capacity: no other AZ will help
                        break
            if inst is not None:
                break
        if inst is None:
            print("NO CAPACITY in any AZ for %s: %s" % (r["run_id"], ", ".join(err)))
            continue
        record[r["run_id"]] = dict(r, instance_id=inst["InstanceId"], az=inst["Placement"]["AvailabilityZone"],
                                   instance_type=itype_used, usd_per_h=usd_used, market=market,
                                   ami=ami, launched=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        record_path.write_text(json.dumps(record, indent=1))
        print("LAUNCHED %-48s %s %s %s $%.2f/h" % (r["run_id"], inst["InstanceId"], inst["Placement"]["AvailabilityZone"],
                                                  itype_used, usd_used))
    print("launch record: %s" % record_path.relative_to(ROOT))


def _s3_text(uri):
    r = subprocess.run(["aws", "s3", "cp", uri, "-", "--only-show-errors"], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def status(stage):
    record = json.loads((ROOT / "runs" / "e1" / ("%s_launch.json" % stage)).read_text())
    ids = [v["instance_id"] for v in record.values()]
    desc = aws("ec2", "describe-instances", "--instance-ids", *ids)
    state = {i["InstanceId"]: (i["State"]["Name"], i.get("StateReason", {}).get("Message", ""))
             for res in desc["Reservations"] for i in res["Instances"]}
    for rid, v in record.items():
        st = (_s3_text(v["s3"] + "/status") or "-").strip()
        log = _s3_text(v["s3"] + "/job.log") or ""
        last = [ln for ln in log.splitlines() if ln.startswith("ONSET step")][-1:] or ["(no step logged yet)"]
        s, why = state.get(v["instance_id"], ("?", ""))
        print("%-48s %-11s %-30s %s" % (rid, s, st[:30], last[0][:90]))
        if why and s != "running":
            print("    reason: %s" % why)


def collect(stage):
    record = json.loads((ROOT / "runs" / "e1" / ("%s_launch.json" % stage)).read_text())
    out = ROOT / "runs" / "e1" / stage
    (out / "logs").mkdir(parents=True, exist_ok=True)
    for rid, v in record.items():
        for name, dest in (("result.json", out / (rid + ".json")), ("job.log", out / "logs" / (rid + ".log")),
                           ("boot.log", out / "logs" / (rid + ".boot.log"))):
            r = subprocess.run(["aws", "s3", "cp", v["s3"] + "/" + name, str(dest), "--only-show-errors"],
                               capture_output=True, text=True)
            print("%-48s %-11s %s" % (rid, name, "ok" if r.returncode == 0 else "missing"))


def main(argv):
    stage = argv[0] if argv and not argv[0].startswith("--") else "pilot"
    if stage not in STAGES:
        raise SystemExit("unknown stage %r; stages: %s" % (stage, ", ".join(STAGES)))
    if "--status" in argv:
        return status(stage)
    if "--collect" in argv:
        return collect(stage)
    rows = plan(stage)
    print_plan(rows)
    if "--submit" not in argv:
        print("\nDRY RUN: nothing baked, uploaded or launched. Add --submit to spend.")
        return 0
    market = "ondemand" if "--ondemand" in argv else "spot"   # g6e spot had no capacity 2026-10-07
    print("market: %s" % market)
    submit(stage, rows, market)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
