"""Fake aws CLI for the E1b launcher tests (experiments/e1b_design/launcher_tests/run_tests.py). ASCII only.

Never touches AWS. Every call is appended to <state>/calls.jsonl. S3 is a local mirror under <state>/s3/<bucket>/<key>;
EC2, IAM and CloudTrail answers come from <state>/config.json:
    {"capacity_fail": [[region, instance_type], ...],   # run-instances -> InsufficientInstanceCapacity in every AZ
     "vcpu_fail": [region, ...],                       # run-instances -> VcpuLimitExceeded
     "states": {instance_id: state}, "gone": [instance_id, ...],   # describe-instances (default "running")
     "launch_times": {instance_id: iso}, "transitions": {instance_id: text},  # LaunchTime, StateTransitionReason
     "run_delay_s": seconds,                            # run-instances waits this long first (concurrency tests)
     "policy_resources": [arn, ...]}                    # the role's inline S3 policy resources (default: e1/*)
The state directory is $FAKE_AWS_STATE. Launched instances are recorded in <state>/instances.json and their user
data copied to <state>/userdata/<instance_id>.sh. LaunchTime defaults to the run-instances time for a launched
instance and to 2026-10-23T12:00:00+00:00 otherwise. CloudTrail events carry fake identity fields (Username,
AccessKeyId, a CloudTrailEvent with userIdentity and sourceIPAddress) so redaction can be tested.
"""
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

STATE = Path(os.environ["FAKE_AWS_STATE"])
S3DIR = STATE / "s3"
BUCKET = "fr-deploy-325527186655"


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
        if a in ("--region", "--output", "--query", "--instance-ids", "--instance-profile-name", "--role-name",
                 "--policy-name", "--policy-arn", "--version-id", "--lookup-attributes", "--cli-input-json",
                 "--user-data"):
            opt[a] = argv[i + 1]
            i += 2
            continue
        if a.startswith("--"):
            opt[a] = True
        else:
            args.append(a)
        i += 1
    if not args:
        fail("Usage", "no command", 2)
    svc, cmd = args[0], args[1] if len(args) > 1 else ""
    c = cfg()
    if svc == "s3" and cmd == "cp":
        src, dst = args[2], args[3]
        if src.startswith("s3://"):
            p = s3path(src)
            if not p.is_file():
                sys.stderr.write("fatal error: An error occurred (404) when calling the HeadObject operation: "
                                 "Key \"%s\" does not exist\n" % src[5:])
                sys.exit(1)
            if dst == "-":
                sys.stdout.buffer.write(p.read_bytes())
            else:
                local(dst).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, local(dst))
            return 0
        p = s3path(dst)
        p.parent.mkdir(parents=True, exist_ok=True)
        if not local(src).is_file():
            sys.stderr.write("The user-provided path %s does not exist.\n" % src)
            sys.exit(255)
        shutil.copyfile(local(src), p)
        return 0
    if svc == "s3" and cmd == "ls":
        base = s3path(args[2])
        root = S3DIR / BUCKET
        if base.is_file():
            keys = [base]
        elif base.is_dir() and "--recursive" in opt:
            keys = sorted(q for q in base.rglob("*") if q.is_file())
        else:
            keys = []
        if not keys:
            sys.exit(1)
        for q in keys:
            sys.stdout.write("2026-10-23 12:00:00 %10d %s\n" % (q.stat().st_size, q.relative_to(root).as_posix()))
        return 0
    if svc == "ec2" and cmd == "run-instances":
        time.sleep(float(c.get("run_delay_s", 0)))
        spec = json.loads(local(opt["--cli-input-json"]).read_text())
        region, itype = opt.get("--region"), spec["InstanceType"]
        if region in c.get("vcpu_fail", []):
            fail("VcpuLimitExceeded", "You have requested more vCPU capacity than your current vCPU limit")
        if [region, itype] in c.get("capacity_fail", []):
            fail("InsufficientInstanceCapacity", "We currently do not have sufficient %s capacity" % itype)
        insts = json.loads((STATE / "instances.json").read_text()) if (STATE / "instances.json").exists() else {}
        iid = "i-fake%012d" % (len(insts) + 1)
        (STATE / "userdata").mkdir(exist_ok=True)
        shutil.copyfile(local(opt["--user-data"]), STATE / "userdata" / (iid + ".sh"))
        insts[iid] = {"region": region, "spec": spec,
                      "launch_time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")}
        (STATE / "instances.json").write_text(json.dumps(insts, indent=1))
        out({"Instances": [{"InstanceId": iid, "InstanceType": itype,
                            "Placement": {"AvailabilityZone": region + "a"}}]})
        return 0
    if svc == "ec2" and cmd == "describe-instances":
        iid = opt["--instance-ids"]
        if iid in c.get("gone", []):
            fail("InvalidInstanceID.NotFound", "The instance ID '%s' does not exist" % iid)
        insts = json.loads((STATE / "instances.json").read_text()) if (STATE / "instances.json").exists() else {}
        launch = c.get("launch_times", {}).get(iid) or insts.get(iid, {}).get("launch_time") or             "2026-10-23T12:00:00+00:00"
        out({"Reservations": [{"Instances": [{"InstanceId": iid,
                                              "State": {"Name": c.get("states", {}).get(iid, "running")},
                                              "LaunchTime": launch,
                                              "StateTransitionReason": c.get("transitions", {}).get(iid, "")}]}]})
        return 0
    if svc == "iam":
        if cmd == "get-instance-profile":
            out({"InstanceProfile": {"InstanceProfileName": opt["--instance-profile-name"],
                                     "Roles": [{"RoleName": "fr-e1-runner"}]}})
        elif cmd == "list-role-policies":
            out({"PolicyNames": ["e1-s3-prefix-only"]})
        elif cmd == "get-role-policy":
            res = c.get("policy_resources", ["arn:aws:s3:::%s/e1/*" % BUCKET])
            out({"RoleName": "fr-e1-runner", "PolicyName": "e1-s3-prefix-only", "PolicyDocument": {
                "Version": "2012-10-17", "Statement": [{"Sid": "Prefix", "Effect": "Allow",
                                                        "Action": ["s3:GetObject", "s3:PutObject"],
                                                        "Resource": res}]}})
        elif cmd == "list-attached-role-policies":
            out({"AttachedPolicies": [{"PolicyName": "AmazonSSMManagedInstanceCore",
                                       "PolicyArn": "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"}]})
        elif cmd == "get-policy":
            out({"Policy": {"DefaultVersionId": "v2"}})
        elif cmd == "get-policy-version":
            out({"PolicyVersion": {"Document": {"Version": "2012-10-17", "Statement": [
                {"Effect": "Allow", "Action": ["ssm:UpdateInstanceInformation", "ssmmessages:*"], "Resource": "*"}]}}})
        else:
            fail("Usage", "unsupported iam %s" % cmd, 2)
        return 0
    if svc == "cloudtrail" and cmd == "lookup-events":
        iid = opt["--lookup-attributes"].split("AttributeValue=", 1)[1]
        who = {"type": "Root", "arn": "arn:aws:iam::000000000000:root", "accessKeyId": "AKIAFAKEFAKEFAKE0000"}
        out({"Events": [
            {"EventId": "e1-" + iid, "EventName": "RunInstances", "EventTime": "2026-10-23T12:00:05+00:00",
             "EventSource": "ec2.amazonaws.com", "Username": "FAKE_USER_NAME", "AccessKeyId": who["accessKeyId"],
             "ReadOnly": "false",
             "Resources": [{"ResourceType": "AWS::EC2::Instance", "ResourceName": iid}],
             "CloudTrailEvent": json.dumps({"eventName": "RunInstances", "instanceId": iid, "userIdentity": who,
                                            "sourceIPAddress": "198.51.100.7"})},
            {"EventId": "e0-" + iid, "EventName": "RunInstances", "EventTime": "2026-10-23T08:00:00-05:00",
             "EventSource": "ec2.amazonaws.com", "Username": "FAKE_USER_NAME", "AccessKeyId": who["accessKeyId"],
             "ReadOnly": "false",
             "Resources": [{"ResourceType": "AWS::EC2::Instance", "ResourceName": iid}],
             "CloudTrailEvent": json.dumps({"eventName": "RunInstances", "instanceId": iid, "userIdentity": who,
                                            "sourceIPAddress": "198.51.100.7"})},
            {"EventId": "e2-" + iid, "EventName": "TerminateInstances", "EventTime": "2026-10-23T20:00:00+00:00",
             "EventSource": "ec2.amazonaws.com", "Username": "FAKE_USER_NAME", "AccessKeyId": who["accessKeyId"],
             "Resources": [{"ResourceType": "AWS::EC2::Instance", "ResourceName": iid}],
             "CloudTrailEvent": json.dumps({"eventName": "TerminateInstances", "instanceId": iid, "userIdentity": who,
                                            "sourceIPAddress": "198.51.100.7"})}]})
        return 0
    fail("Usage", "fake aws does not support: %s" % " ".join(argv), 2)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]) or 0)
