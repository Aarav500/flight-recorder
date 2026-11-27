"""MARL throwaway spikes on EC2. One self-terminating on-demand instance per stage. ASCII only.

    python marl/aws/launch_spike.py jax              # dry run: prints the plan and the price. Spends nothing.
    python marl/aws/launch_spike.py jax --submit     # spend
    python marl/aws/launch_spike.py jax --status
    python marl/aws/launch_spike.py jax --collect    # -> marl/runs/<stage>/

Patterned on scripts/launch_e1_aws.py and imports nothing from it, so E1b's pinned files are untouched.
Storage is under e1/marl/ because the instance role fr-e1-runner can only reach e1/*.

Each instance schedules its own shutdown at boot, so it can't outlive the timeout whatever the job does,
and InstanceInitiatedShutdownBehavior=terminate turns that into a termination. A stage launches once: a
second --submit refuses while a launch record exists. There are no retry loops here on purpose.
"""
import json
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "marl" / "runs"
BUCKET = "fr-deploy-325527186655"
PROFILE = "fr-e1-runner"
REGION = "us-east-1"
AMI_PARAM = "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"
SUBNETS = {"us-east-1a": "subnet-0386677a3484341be", "us-east-1b": "subnet-050b222e93f526cd8",
           "us-east-1c": "subnet-042747a96276b6f96", "us-east-1d": "subnet-028805185f69156e2"}

HEAD = r"""#!/bin/bash
set -u
B="__S3__"
mkdir -p /opt/marl && cd /opt/marl
exec > /opt/marl/boot.log 2>&1
shutdown -h +__TIMEOUT_MIN__
export HOME=/root
RC=boot-failed
finish() {
  echo "rc=$RC $(date -u +%FT%TZ)" > /opt/marl/status
  aws s3 cp /opt/marl/boot.log "$B/boot.log" --only-show-errors
  aws s3 cp /opt/marl/status "$B/status" --only-show-errors
  shutdown -h now
}
trap finish EXIT
python3 -m pip install -q uv || { apt-get update -q && apt-get install -yq python3-pip && python3 -m pip install -q uv; }
aws s3 cp "$B/bundle.tar.gz" /opt/marl/bundle.tar.gz --only-show-errors
tar xzf bundle.tar.gz
echo "started $(date -u +%FT%TZ)" > /opt/marl/status && aws s3 cp /opt/marl/status "$B/status" --only-show-errors
export UV_CACHE_DIR=/opt/marl/uvcache PYTHONUNBUFFERED=1
"""

JAX_BODY = r"""
python3 -m uv python install 3.11.9
python3 -m uv venv --python 3.11.9 /opt/marl/venv
python3 -m uv pip install --python /opt/marl/venv/bin/python --no-deps -r marl/spike_jax/requirements_frozen.txt
PY=/opt/marl/venv/bin/python
cd marl/spike_jax
mkdir -p results
export JAX_PLATFORMS=cpu
$PY test_patched_reward.py && $PY train_ippo_phi.py --n 3 --cond patched --seed 999 --steps 45056 --out /tmp/smoke
RC=$?
[ "$RC" = 0 ] || { RC=smoke-failed-$RC; exit 1; }
# cp --recursive, not sync: the role has PutObject under e1/ but no ListBucket, and sync needs it
UP='aws s3 cp results "$B/results" --recursive --only-show-errors --exclude "*.lock" --exclude "*.tmp"'
( while sleep 60; do eval "$UP"; aws s3 cp /opt/marl/boot.log "$B/boot.log" --only-show-errors; done ) &
$PY run_jobs.py __JOBS__ --script train_ippo_phi.py --workers 30 --cores-per-job 2 --logdir results/logs > results/runner.log 2>&1
$PY run_jobs.py __JOBS__ --script train_ippo_phi.py --workers 30 --cores-per-job 2 --logdir results/logs > results/runner_retry.log 2>&1
RC=$?
eval "$UP" || RC=upload-failed
"""

LLM_BODY = r"""
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
export HF_HOME=/opt/marl/hf
cd marl/spike_llm
( while sleep 180; do aws s3 cp /opt/marl/job.log "$B/job.log" --only-show-errors; done ) &
timeout __JOB_MIN__m python3 -m uv run --python 3.12 step0_llm.py __ARGS__ --out /opt/marl/result.json > /opt/marl/job.log 2>&1
RC=$?
aws s3 cp /opt/marl/job.log "$B/job.log" --only-show-errors
aws s3 cp /opt/marl/result.json "$B/result.json" --only-show-errors
"""

TRAIN_BODY = r"""
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
export HF_HOME=/opt/marl/hf
cd marl/spike_llm
mkdir -p /opt/marl/out
touch /opt/marl/job.log
UP='aws s3 cp /opt/marl/out "$B/out" --recursive --only-show-errors'
( while sleep 300; do eval "$UP"; aws s3 cp /opt/marl/job.log "$B/job.log" --only-show-errors; done ) &
__RUNS__
eval "$UP" || RC=upload-failed
aws s3 cp /opt/marl/job.log "$B/job.log" --only-show-errors
"""
TRAIN_FILES = ["marl/spike_llm/train_two_agent.py", "marl/spike_llm/step0_llm.py", "flightrecorder/__init__.py",
               "flightrecorder/repro/__init__.py", "flightrecorder/repro/reward_testhack.py",
               "flightrecorder/repro/reward_hardset.py"]
TRAIN_TYPES = ["g6e.xlarge", "g6e.2xlarge", "g6e.4xlarge"]
UV_RUN = "python3 -m uv run --python 3.12 train_two_agent.py"
# From pick_lr.py on the trcal runs: KL 0.051 against a target of 0.074, chosen without reading reward.
TRAIN_LR = "3e-5"
TRAIN_TIMEOUT_MIN = 300

_SCALE = "python3 -m uv run --python 3.12 train_scale.py --task csv_narrow --lr 3e-5 --steps 30 --sched-steps 400 --team mean"
STAGES = {
    # Two noise fixes, 30 steps each on spare seeds: 8 samples + step-averaged gradients, then 64 samples per step.
    # Rule in marl/spike_llm/gate_scale.py.
    "scalegate": dict(types=["g6e.2xlarge", "g6e.xlarge", "g6e.4xlarge"], est_h=1.4, timeout_min=170, disk=100,
                      files=TRAIN_FILES + ["marl/spike_llm/train_sym.py", "marl/spike_llm/train_scale.py"],
                      body=TRAIN_BODY.replace("__RUNS__",
                                              "timeout 35m " + _SCALE + " --seed 902 --m 8 --selfcheck"
                                              " --out /opt/marl/out/m8_ema_s902 >> /opt/marl/job.log 2>&1\n"
                                              "timeout 115m " + _SCALE + " --seed 903 --m 64 --chunk 4"
                                              " --out /opt/marl/out/m64_s903 >> /opt/marl/job.log 2>&1\n"
                                              "RC=$?")),
    # Symmetric agents, 30 steps on a spare seed: can phi see anything? Rule in marl/spike_llm/gate_sym.py.
    "symgate": dict(types=TRAIN_TYPES, est_h=0.45, timeout_min=90, disk=100,
                    files=TRAIN_FILES + ["marl/spike_llm/train_sym.py"],
                    body=TRAIN_BODY.replace("__RUNS__",
                                            "timeout 70m python3 -m uv run --python 3.12 train_sym.py --task csv_narrow"
                                            " --seed 901 --lr 3e-5 --steps 30 --sched-steps 400 --team mean --selfcheck"
                                            " --out /opt/marl/out/sym_mean_s901 > /opt/marl/job.log 2>&1\n"
                                            "RC=$?")),
    # Three learning rates, 30 steps each, on a seed never used for a result. Picks the rate by KL only.
    "trcal": dict(types=TRAIN_TYPES, est_h=0.9, timeout_min=130, disk=100, files=TRAIN_FILES,
                  body=TRAIN_BODY.replace("__RUNS__",
                                          "for LR in 3e-5 1e-4 3e-4; do\n"
                                          "  timeout 40m " + UV_RUN + " --task csv_narrow --seed 900 --lr $LR --steps 30"
                                          " --sched-steps 400 --selfcheck --out /opt/marl/out/lr$LR >> /opt/marl/job.log 2>&1\n"
                                          "  RC=$?\n"
                                          "done")),
    # 96 CPU jobs. The same job mix took 14 min on 64 vCPU in study L; the readout adds about half again.
    "jax": dict(types=["c7i.16xlarge", "m7i.16xlarge", "c6i.16xlarge"], est_h=0.8, timeout_min=180, disk=100,
                body=JAX_BODY.replace("__JOBS__", "jobs_phi.txt"), files=["marl/spike_jax"]),
    # 30 N=4 jobs with an agent id, one wave on 60 cores.
    "jax2": dict(types=["c7i.16xlarge", "m7i.16xlarge", "c6i.16xlarge"], est_h=0.35, timeout_min=90, disk=100,
                 body=JAX_BODY.replace("__JOBS__", "jobs_id.txt"), files=["marl/spike_jax"]),
    # The other 7 hard-set tasks x 3 designs. 8 vCPU, because scoring dominated the first screen.
    "llm2": dict(types=["g6e.2xlarge", "g6e.xlarge", "g6e.4xlarge", "g7e.2xlarge"], est_h=1.2, timeout_min=180,
                 disk=100, body=LLM_BODY.replace("__JOB_MIN__", "165").replace(
                     "__ARGS__", "--designs D0,D2,D3 --tasks chunk_exact,dedupe_typed,word_freq_narrow,"
                                 "round_half_up_narrow,expr_eval,merge_touching,precedence_fizz"),
                 files=["marl/spike_llm/step0_llm.py", "flightrecorder/__init__.py",
                        "flightrecorder/repro/__init__.py", "flightrecorder/repro/reward_testhack.py",
                        "flightrecorder/repro/reward_hardset.py"]),
    # 5 tasks x 4 designs. Generation plus about 10k sandbox scorings on 4 vCPU.
    "llm": dict(types=["g6e.xlarge", "g6e.2xlarge", "g6e.4xlarge", "g7e.2xlarge"], est_h=1.0, timeout_min=150,
                disk=100, body=LLM_BODY.replace("__JOB_MIN__", "120").replace("__ARGS__", ""),
                files=["marl/spike_llm/step0_llm.py", "flightrecorder/__init__.py",
                       "flightrecorder/repro/__init__.py", "flightrecorder/repro/reward_testhack.py",
                       "flightrecorder/repro/reward_hardset.py"]),
}


for _task in ("csv_narrow", "csv_exchangeable"):
    for _seed in (300, 301, 302):
        _name = "%s_s%d" % (_task, _seed)
        STAGES["tr_" + _name] = dict(
            types=TRAIN_TYPES, est_h=3.0, timeout_min=TRAIN_TIMEOUT_MIN, disk=100, files=TRAIN_FILES, train=True,
            body=TRAIN_BODY.replace("__RUNS__",
                                    "timeout %dm %s --task %s --seed %d --lr __LR__ --out /opt/marl/out/%s > /opt/marl/job.log 2>&1\n"
                                    "RC=$?" % (TRAIN_TIMEOUT_MIN - 10, UV_RUN, _task, _seed, _name)))
        # One agent on the same pair, same settings. Rule in marl/spike_llm/verdict_single.py.
        STAGES["tr1_" + _name] = dict(
            types=TRAIN_TYPES, est_h=2.0, timeout_min=240, disk=100, train=True,
            files=TRAIN_FILES + ["marl/spike_llm/train_single.py"],
            body=TRAIN_BODY.replace("__RUNS__",
                                    "timeout 230m python3 -m uv run --python 3.12 train_single.py --task %s --seed %d "
                                    "--lr __LR__ --selfcheck --out /opt/marl/out/single_%s > /opt/marl/job.log 2>&1\n"
                                    "RC=$?" % (_task, _seed, _name)))


def aws(*args):
    r = subprocess.run(["aws", *args, "--region", REGION, "--output", "json"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("aws %s failed: %s" % (" ".join(args[:2]), (r.stderr or "").strip()[-600:]))
    return json.loads(r.stdout) if r.stdout.strip() else None


def price(itype):
    """On-demand USD/h from the pricing API, or None if it can't be read."""
    try:
        out = aws("pricing", "get-products", "--service-code", "AmazonEC2", "--filters",
                  "Type=TERM_MATCH,Field=instanceType,Value=" + itype,
                  "Type=TERM_MATCH,Field=location,Value=US East (N. Virginia)",
                  "Type=TERM_MATCH,Field=operatingSystem,Value=Linux",
                  "Type=TERM_MATCH,Field=tenancy,Value=Shared",
                  "Type=TERM_MATCH,Field=preInstalledSw,Value=NA",
                  "Type=TERM_MATCH,Field=capacitystatus,Value=Used")
        item = json.loads(out["PriceList"][0])
        term = next(iter(item["terms"]["OnDemand"].values()))
        return float(next(iter(term["priceDimensions"].values()))["pricePerUnit"]["USD"])
    except Exception as e:  # the plan still prints; the caller says the price is unknown
        print("price lookup failed for %s: %s" % (itype, str(e)[:120]))
        return None


def s3_root(stage):
    return "s3://%s/e1/marl/%s" % (BUCKET, stage)


def bundle(stage, tmp):
    path = tmp / "bundle.tar.gz"
    with tarfile.open(path, "w:gz") as t:
        for f in STAGES[stage]["files"]:
            t.add(ROOT / f, arcname=f, filter=lambda ti: None if any(
                p in ti.name.split("/") for p in ("results", "__pycache__")) else ti)
    return path


def plan(stage):
    s = STAGES[stage]
    print("stage %s: first available of %s, on-demand, us-east-1" % (stage, ", ".join(s["types"])))
    for t in s["types"]:
        p = price(t)
        if p is None:
            print("  %-14s price unknown; estimate %.1f h, hard stop %.1f h" % (t, s["est_h"], s["timeout_min"] / 60))
        else:
            print("  %-14s $%.3f/h  estimate $%.2f  worst case (runs to the hard stop) $%.2f"
                  % (t, p, p * s["est_h"], p * s["timeout_min"] / 60))
    print("storage: %s" % s3_root(stage))


def submit(stage):
    record_path = RUNS / ("%s_launch.json" % stage)
    if record_path.exists():
        raise SystemExit("already launched, see %s. Delete it only if you mean to spend again." % record_path)
    s = STAGES[stage]
    tmp = Path(tempfile.mkdtemp(prefix="marlaws_"))   # no spaces: the aws exe takes file:// paths
    b = bundle(stage, tmp)
    print("bundle %.1f KB" % (b.stat().st_size / 1024))
    subprocess.run(["aws", "s3", "cp", str(b), s3_root(stage) + "/bundle.tar.gz", "--only-show-errors"], check=True)
    ud = tmp / "ud.sh"
    ud.write_text((HEAD + s["body"]).replace("__S3__", s3_root(stage)).replace("__TIMEOUT_MIN__", str(s["timeout_min"])),
                  encoding="ascii", newline="\n")
    ami = aws("ssm", "get-parameter", "--name", AMI_PARAM)["Parameter"]["Value"]
    spec = {"ImageId": ami, "MinCount": 1, "MaxCount": 1, "IamInstanceProfile": {"Name": PROFILE},
            "InstanceInitiatedShutdownBehavior": "terminate",
            "BlockDeviceMappings": [{"DeviceName": "/dev/sda1", "Ebs": {
                "VolumeSize": s["disk"], "VolumeType": "gp3", "DeleteOnTermination": True}}],
            "MetadataOptions": {"HttpTokens": "required"},
            "TagSpecifications": [{"ResourceType": "instance", "Tags": [
                {"Key": "Name", "Value": "fr-marl-" + stage}, {"Key": "Project", "Value": "flight-recorder-marl"},
                {"Key": "Stage", "Value": stage}]}]}
    inst, errs = None, []
    for itype in s["types"]:
        spec["InstanceType"] = itype
        for az, subnet in SUBNETS.items():
            spec["SubnetId"] = subnet
            sp = tmp / "spec.json"
            sp.write_text(json.dumps(spec), encoding="ascii")
            try:
                inst = aws("ec2", "run-instances", "--cli-input-json", "file://" + sp.as_posix(),
                           "--user-data", "file://" + ud.as_posix())["Instances"][0]
                break
            except RuntimeError as e:
                msg = str(e)
                code = msg.split("An error occurred (")[-1].split(")")[0] if "An error occurred (" in msg else msg[-80:]
                if code not in ("InsufficientInstanceCapacity", "Unsupported", "VcpuLimitExceeded"):
                    raise
                errs.append("%s/%s:%s" % (itype, az, code))
                if code == "VcpuLimitExceeded":
                    break
        if inst is not None:
            break
    if inst is None:
        raise SystemExit("NO CAPACITY: " + ", ".join(errs))
    RUNS.mkdir(parents=True, exist_ok=True)
    rec = dict(stage=stage, instance_id=inst["InstanceId"], instance_type=inst["InstanceType"],
               az=inst["Placement"]["AvailabilityZone"], ami=ami, s3=s3_root(stage),
               launched=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    record_path.write_text(json.dumps(rec, indent=1))
    print("LAUNCHED %s %s %s" % (rec["instance_id"], rec["instance_type"], rec["az"]))


def status(stage):
    rec = json.loads((RUNS / ("%s_launch.json" % stage)).read_text())
    d = aws("ec2", "describe-instances", "--instance-ids", rec["instance_id"])
    i = d["Reservations"][0]["Instances"][0]
    r = subprocess.run(["aws", "s3", "cp", rec["s3"] + "/status", "-", "--only-show-errors"], capture_output=True, text=True)
    print(rec["instance_id"], i["State"]["Name"], "| status:", (r.stdout or "-").strip())


def collect(stage):
    rec = json.loads((RUNS / ("%s_launch.json" % stage)).read_text())
    out = RUNS / stage
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(["aws", "s3", "sync", rec["s3"], str(out), "--only-show-errors", "--exclude", "bundle.tar.gz"], check=True)
    print("collected to", out)


def main(argv):
    if not argv or argv[0] not in STAGES:
        raise SystemExit("stage: " + ", ".join(STAGES))
    stage = argv[0]
    if "--status" in argv:
        return status(stage)
    if "--collect" in argv:
        return collect(stage)
    plan(stage)
    if STAGES[stage].get("train"):
        if TRAIN_LR is None:
            raise SystemExit("TRAIN_LR is not set. Run trcal and pick_lr.py first.")
        STAGES[stage]["body"] = STAGES[stage]["body"].replace("__LR__", TRAIN_LR)
        print("learning rate", TRAIN_LR)
    if "--submit" not in argv:
        print("\nDRY RUN: nothing uploaded or launched. Add --submit to spend.")
        return
    submit(stage)


if __name__ == "__main__":
    main(sys.argv[1:])
