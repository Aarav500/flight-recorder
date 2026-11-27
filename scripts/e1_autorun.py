# E1 auto-run (Aarav, chat 2026-10-08: "set the check, run the analysis automatically").
# 1. Wait until every main run has settled: its S3 status says rc=N, or its instance is no longer running.
#    Job health only while waiting: status files and instance states. No job.log is read (pre-reg section 8).
# 2. Collect into runs/e1/main with the registered collector.
# 3. Exclusion check with e1_primary.load_main, which uses only launch records, config and strict steps 0-4
#    (section 7). If anything is excluded, stop BEFORE the analysis: section 7 requires replacements first.
# 4. Otherwise run the registered analysis, python experiments/e1_primary.py, and save its report.
# Stop early by creating e1_autorun.STOP next to this file (TaskStop does not kill this on Windows).
# Exit codes: 0 analysis ran, 2 stopped, 3 replacements needed, 4 deadline passed, 5 collect or analysis error.
import json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(r"C:/Users/aarav/OneDrive/Desktop/Fligh recorder")
REC = ROOT / "runs" / "e1" / "main_launch.json"
HERE = Path(__file__).resolve().parent
STOP = HERE / "e1_autorun.STOP"
REPORT = ROOT / "experiments" / "e1_primary_report.txt"
DEADLINE = time.time() + 10 * 3600
POLL = 300


def sh(*a, cwd=None):
    r = subprocess.run(list(a), capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=cwd)
    return r.returncode, r.stdout, r.stderr


def say(*a):
    print(time.strftime("%H:%M"), *a, flush=True)


(HERE / "e1_autorun.pid").write_text(str(os.getpid()))
say("e1_autorun pid %d; poll every %d s; stop with %s" % (os.getpid(), POLL, STOP))
rec = json.loads(REC.read_text())
by_region = {}
for rid, v in rec.items():
    by_region.setdefault(v.get("region", "us-east-1"), set()).add(v["instance_id"])

last = None
while True:
    if STOP.exists():
        say("STOP file found; exiting before collection")
        sys.exit(2)
    if time.time() > DEADLINE:
        say("10 h deadline passed without every run settling; nothing collected")
        sys.exit(4)
    # One call per instance: a terminated instance eventually vanishes, and a multi-id call then fails whole.
    states = {}
    for region, ids in by_region.items():
        for iid in ids:
            rc, out, err = sh("aws", "ec2", "describe-instances", "--region", region, "--instance-ids", iid,
                              "--query", "Reservations[0].Instances[0].State.Name", "--output", "text")
            states[iid] = (out.strip() if rc == 0 and out.strip() not in ("", "None") else
                           "gone" if "InvalidInstanceID.NotFound" in err else "unknown")
    status = {}
    for rid, v in rec.items():
        rc, out, _ = sh("aws", "s3", "cp", v["s3"] + "/status", "-")
        status[rid] = out.strip() if rc == 0 else ""
    ok = [r for r, s in status.items() if s.startswith("rc=0")]
    bad = [r for r, s in status.items() if s.startswith("rc=") and not s.startswith("rc=0")]
    # A run with no rc= counts as dead only when its instance is CONFIRMED down; "unknown" keeps waiting.
    dead = [r for r, s in status.items() if not s.startswith("rc=")
            and states.get(rec[r]["instance_id"]) in ("stopping", "stopped", "shutting-down", "terminated", "gone")]
    key = (len(ok), len(bad), len(dead), tuple(sorted(states.values())))
    if key != last:
        say("finished rc=0: %d/%d; failed rc!=0: %d; instance down before finishing: %d; instances: %s" % (
            len(ok), len(rec), len(bad), len(dead), sorted(states.values())))
        last = key
    if len(ok) + len(bad) + len(dead) == len(rec):
        break
    time.sleep(POLL)

for r in bad:
    say("FAILED %s (%s)" % (r, status[r]))
for r in dead:
    say("NO RESULT %s (instance %s)" % (r, states.get(rec[r]["instance_id"])))

say("all %d runs settled; collecting into runs/e1/main" % len(rec))
rc, out, err = sh(sys.executable, "scripts/launch_e1_aws.py", "main", "--collect", cwd=str(ROOT))
missing = [ln for ln in out.splitlines() if "result.json" in ln and ln.rstrip().endswith("missing")]
say("collect exit %d; result.json missing for %d runs" % (rc, len(missing)))
for ln in missing:
    print("   ", ln.split()[0], flush=True)
if rc != 0:
    print(err[-2000:], flush=True)
    sys.exit(5)

sys.path.insert(0, str(ROOT / "experiments"))
import e1_primary as P
used, excluded, unregistered, launched, surplus = P.load_main()
say("exclusion check: used hacking %d, benign %d; excluded %d; unregistered %d; surplus %d; launched %s" % (
    len(used["hacking"]), len(used["benign"]), len(excluded), len(unregistered), len(surplus), launched))
for tag, why in excluded:
    print("    excluded %s: %s" % (tag, why), flush=True)
if excluded or bad or dead or len(used["hacking"]) != 24 or len(used["benign"]) != 24:
    say("REPLACEMENTS NEEDED before the registered analysis (section 7). Analysis NOT run.")
    sys.exit(3)

say("48/48 usable; running the registered analysis: python experiments/e1_primary.py")
t0 = time.time()
rc, out, err = sh(sys.executable, "experiments/e1_primary.py", cwd=str(ROOT))
REPORT.write_text(out + ("\n[stderr]\n" + err if err.strip() else ""), encoding="utf-8")
say("analysis exit %d after %.0f s; report saved to %s; results json experiments/e1_primary_results.json" % (
    rc, time.time() - t0, REPORT))
print(out, flush=True)
if rc != 0:
    print(err[-4000:], flush=True)
    sys.exit(5)
sys.exit(0)
