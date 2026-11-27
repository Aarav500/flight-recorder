"""Job-health monitor for the E1 main runs. Prints NO detector, judge or oracle value (pre-registration section 8).

For each launched run: its instance state, its status line (started / rc=N), and the latest logged STEP
NUMBER only. The step number is pulled from the ONSET lines with a regex; nothing else on the line is
read or printed.

    python scripts/e1_main_status.py
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORDS = [ROOT / "runs" / "e1" / "main_launch.json", ROOT / "runs" / "e1" / "main_replace_launch.json"]
STEP = re.compile(r"ONSET step (\d+) ")


def sh(*a):
    r = subprocess.run(list(a), capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.stdout if r.returncode == 0 else ""


def main():
    rec = {}
    for p in RECORDS:
        if p.exists():
            rec.update(json.loads(p.read_text()))
    if not rec:
        print("no main runs launched yet")
        return 0
    states = {}
    by_region = {}
    for v in rec.values():
        if v.get("instance_id"):
            by_region.setdefault(v.get("region", "us-east-1"), set()).add(v["instance_id"])
    for region, ids in by_region.items():
        d = json.loads(sh("aws", "ec2", "describe-instances", "--region", region, "--instance-ids", *sorted(ids),
                          "--output", "json") or "{}")
        for res in d.get("Reservations", []):
            for i in res["Instances"]:
                states[i["InstanceId"]] = i["State"]["Name"]
    done = 0
    for rid, v in rec.items():
        st = sh("aws", "s3", "cp", v["s3"] + "/status", "-").strip() or "-"
        steps = STEP.findall(sh("aws", "s3", "cp", v["s3"] + "/job.log", "-"))
        last = steps[-1] if steps else "-"
        done += st.startswith("rc=0")
        print("%-46s %-12s %-10s %-34s step %s/%s" % (rid, v.get("instance_type", ""), states.get(v.get("instance_id"), "?"),
                                                       st[:34], last, v.get("steps", 400)))
    print("finished OK: %d of %d launched (48 planned)" % (done, len(rec)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
