"""Stands in for `uv run job.py` when run_tests.py runs the J3 instance's user data under bash. ASCII only.

Run in the job's directory with E1B_J3_STATUS_PATH set, like the real job: it writes status.json at a few phases,
partial.json, then result.json, and exits with the code FAKE_J3_MODE asks for:
    ok      rc 0, result.json complete        fail3   rc 3, result.json incomplete
    hang    writes status once, then sleeps (the watchdog must stop it)
The run id is read from job.py's RUN_ID line, as the real job has it baked in.
"""
import json
import os
import re
import sys
import time

rid = re.search(r'^RUN_ID = "([^"]+)"', open("job.py", encoding="ascii").read(), re.M).group(1)
status = os.environ["E1B_J3_STATUS_PATH"]
mode = os.environ.get("FAKE_J3_MODE", "ok")
for need in ("job.py.lock", "data.jsonl.gz"):
    if not os.path.isfile(need):
        sys.exit("fake job: %s missing" % need)


def st(phase, n):
    with open(status + ".tmp", "w") as f:
        json.dump({"run_id": rid, "phase": phase, "n_scored": n, "n_total": 16, "rc": None, "ts": "x"}, f)
    os.replace(status + ".tmp", status)
    print("J3 %s %d/16" % (phase, n), flush=True)


st("start", 0)
if mode == "hang":
    time.sleep(100000)
st("scoring", 8)
with open("partial.json", "w") as f:
    json.dump({"run_id": rid, "complete": False}, f)
st("done", 16)
with open("result.json", "w") as f:
    json.dump({"run_id": rid, "complete": mode == "ok"}, f)
sys.exit(0 if mode == "ok" else 3)
