"""Fake aws CLI for the J3 launcher tests (experiments/e1b_design/j3_tests/run_tests.py). ASCII only.

Never touches AWS. Every call is appended to <state>/calls.jsonl. S3 is a local mirror under <state>/s3/<bucket>/<key>;
EC2 and IAM answers come from <state>/config.json:
    {"capacity_fail": [[region, instance_type], ...],   # run-instances -> InsufficientInstanceCapacity in every AZ
     "states": {instance_id: state}, "gone": [instance_id, ...],      # describe-instances (default "running")
     "launch_times": {instance_id: iso}, "ended": {instance_id: "YYYY-MM-DD HH:MM:SS"},
     "policy_resources": [arn, ...]}                    # the role's inline S3 policy resources
The state directory is $FAKE_AWS_STATE. Launched instances go to <state>/instances.json, their user data to
<state>/userdata/<instance_id>.sh.
"""
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

STATE = Path(os.environ["FAKE_AWS_STATE"])
S3DIR = STATE / "s3"
BUCKET = "fr-deploy-325527186655"
VALUED = ("--region", "--output", "--instance-ids", "--instance-profile-name", "--role-name", "--policy-name",
          "--policy-arn", "--version-id", "--cli-input-json", "--user-data", "--metadata")


def cfg():
    p = STATE / "config.json"
    return json.loads(p.read_text()) if p.exists() else {}


def fail(code, msg="", rc=254):
    sys.stderr.write("\nAn error occurred (%s) when calling the operation: %s\n" % (code, msg))
    sys.exit(rc)


def out(obj):
    sys.stdout.write(json.dumps(obj, indent=1) + "\n")


def s3path(uri):
    assert uri.startswith("s3://"), uri
    return S3DIR / uri[5:]


def local(p):
    return Path(p[7:] if p.startswith("file://") else p)


def main(argv):
    with open(STATE / "calls.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(argv) + "\n")
    args, opt = [], {}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in VALUED:
            opt[a] = argv[i + 1]
            i += 2
            continue
        if a.startswith("--"):
            opt[a] = True
        else:
            args.append(a)
        i += 1
    svc, cmd = (args + ["", ""])[:2]
    c = cfg()
    if svc == "s3" and cmd == "cp":
        src, dst = args[2], args[3]
        if src.startswith("s3://"):
            p = s3path(src)
            if not p.is_file():
                sys.stderr.write("fatal error: An error occurred (404) when calling the HeadObject operation: Key "
                                 "\"%s\" does not exist\n" % src[5:])
                sys.exit(1)
            if dst == "-":
                sys.stdout.buffer.write(p.read_bytes())
            else:
                local(dst).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, local(dst))
            return 0
        if not local(src).is_file():
            sys.stderr.write("The user-provided path %s does not exist.\n" % src)
            sys.exit(255)
        p = s3path(dst)
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local(src), p)
        return 0
    if svc == "s3" and cmd == "ls":
        base = s3path(args[2])
        root = S3DIR / BUCKET
        keys = sorted(q for q in base.rglob("*") if q.is_file()) if base.is_dir() else []
        if not keys:
            sys.exit(1)
        for q in keys:
            sys.stdout.write("2026-10-25 12:00:00 %10d %s\n" % (q.stat().st_size, q.relative_to(root).as_posix()))
        return 0
    if svc == "ec2" and cmd == "run-instances":
        spec = json.loads(local(opt["--cli-input-json"]).read_text())
        region, itype = opt.get("--region"), spec["InstanceType"]
        if [region, itype] in c.get("capacity_fail", []):
            fail("InsufficientInstanceCapacity", "We currently do not have sufficient %s capacity" % itype)
        p = STATE / "instances.json"
        insts = json.loads(p.read_text()) if p.exists() else {}
        iid = "i-fakej3%010d" % (len(insts) + 1)
        (STATE / "userdata").mkdir(exist_ok=True)
        shutil.copyfile(local(opt["--user-data"]), STATE / "userdata" / (iid + ".sh"))
        insts[iid] = {"region": region, "spec": spec,
                      "launch": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")}
        p.write_text(json.dumps(insts, indent=1))
        out({"Instances": [{"InstanceId": iid, "InstanceType": itype, "Placement": {"AvailabilityZone": region + "a"}}]})
        return 0
    if svc == "ec2" and cmd == "describe-instances":
        iid = opt["--instance-ids"]
        if iid in c.get("gone", []):
            fail("InvalidInstanceID.NotFound", "The instance ID '%s' does not exist" % iid)
        p = STATE / "instances.json"
        insts = json.loads(p.read_text()) if p.exists() else {}
        launch = c.get("launch_times", {}).get(iid) or (insts.get(iid) or {}).get("launch") or \
            "2026-10-25T10:00:00+00:00"
        state = c.get("states", {}).get(iid, "running")
        end = c.get("ended", {}).get(iid)
        reason = "User initiated (%s GMT)" % end if end else ""
        out({"Reservations": [{"Instances": [{"InstanceId": iid, "LaunchTime": launch, "State": {"Name": state},
                                              "StateTransitionReason": reason}]}]})
        return 0
    if svc == "iam":
        if cmd == "get-instance-profile":
            out({"InstanceProfile": {"InstanceProfileName": opt["--instance-profile-name"],
                                     "Roles": [{"RoleName": "fr-e1-runner"}]}})
        elif cmd == "list-role-policies":
            out({"PolicyNames": ["e1-s3-prefix-only"]})
        elif cmd == "get-role-policy":
            res = c.get("policy_resources", ["arn:aws:s3:::%s/e1/*" % BUCKET])
            out({"PolicyDocument": {"Version": "2012-10-17", "Statement": [
                {"Sid": "Prefix", "Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject"], "Resource": res}]}})
        elif cmd == "list-attached-role-policies":
            out({"AttachedPolicies": []})
        else:
            fail("Usage", "unsupported iam %s" % cmd, 2)
        return 0
    fail("Usage", "fake aws does not support: %s" % " ".join(argv), 2)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]) or 0)
