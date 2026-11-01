"""Stands in for `python3 -m uv run ... job.py` when run_tests.py runs an instance's user data under bash. ASCII only.

It checks what the instance gives a job (the arguments, job.py and job.py.lock in the working directory,
E1B_STATUS_PATH), then acts like the job on the monitoring side only: status.json {run_id, step, rc: null, ts} per
step, one "STEP <n>" stdout line per step, and a result file at $FAKE_RESULT_PATH. $FAKE_JOB_PLAN maps a run id to
"ok" (3 steps), "fail" (exit 1 after step 0), "noresult" (3 steps, exit 0, no result file) or "slowok" (30 steps,
one per second). A "hang" run never reaches this script: the python3 wrapper replaces it with `sleep`.
"""
import json
import os
import re
import sys
import time
from datetime import datetime, timezone


def status(path, rid, step):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"run_id": rid, "step": step, "rc": None,
                   "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}, f)
    os.replace(tmp, path)


def main(argv):
    state = os.environ["FAKE_AWS_STATE"]
    src = open("job.py", encoding="utf-8").read()
    m = re.search(r'^ARM, TASK, SEED, STAGE, RUN_ID = .*"([^"]+)"\s*$', src, re.M)
    rid = m.group(1) if m else None
    with open(os.path.join(state, "jobcalls.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"run_id": rid, "argv": argv, "files": sorted(os.listdir(".")),
                            "status_path": os.environ.get("E1B_STATUS_PATH"),
                            "unbuffered": os.environ.get("PYTHONUNBUFFERED")}) + "\n")
    path = os.environ["E1B_STATUS_PATH"]
    plan = json.loads(os.environ.get("FAKE_JOB_PLAN", "{}")).get(rid, "ok")
    sys.stderr.write("fake job %s: plan %s\n" % (rid, plan))
    if plan == "fail":
        status(path, rid, 0)
        print("STEP 0", flush=True)
        sys.stderr.write("E1B STOP before training: fake failure\n")
        return 1
    for k in range(30 if plan == "slowok" else 3):
        if plan == "slowok":
            time.sleep(1)
        status(path, rid, k)
        print("STEP %d" % k, flush=True)
    if plan != "noresult":
        with open(os.environ["FAKE_RESULT_PATH"], "w", encoding="utf-8", newline="\n") as f:
            json.dump({"run_id": rid, "fake": True}, f)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
