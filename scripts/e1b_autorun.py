"""E1b auto-run (pre-registration section 8; analysis spec section 5). ASCII only.

Committed before the first launch and run from the repo (E1's ran from the session scratch directory). It follows
scripts/e1_autorun.py, which stays frozen, and launches nothing.
  0. Refuse unless the collector and this script are the committed ones: scripts/launch_e1b_aws.py (the
     collector), scripts/e1b_autorun.py, the collector's own imports scripts/e1b_job.py and
     experiments/e1b_registry.py, and experiments/e1b_primary.py must each equal its blob at HEAD (a CRLF checkout
     of an LF blob counts as equal). Checked here before anything else, again just before collection, and before
     the launcher is imported, so a changed launcher cannot vouch for itself. e1b_primary --exclusions-only then
     verifies its pins (spec 7.6) before it selects anything.
  1. Poll the launcher's status function (scripts/launch_e1b_aws.status_rows) every --interval seconds until every
     launched run of the stage has a final rc, or its instance is down without one. That function reads only the
     status.json files (run_id, step, rc, ts) and the instance states, never job.log or result.json. For
     main_replace the main runs are watched too, because both feed runs/e1b/main.
  2. Collect with the launcher's collector: results, logs and status files to the build_contract paths, and
     runs/e1b/<stage>/MANIFEST.sha256, before any analysis.
  3. python experiments/e1b_primary.py --exclusions-only: load_e1b applies section 7 and prints reasons only, no
     value. Its exit code decides (build_contract section 6):
       0  nothing to replace and every main run launched: go on to step 4;
       3  a replacement is needed and allowed: the list is printed, nothing is written or launched here, and this
          script exits 3. The replacements are written to runs/e1b/main_replace.json by hand and launched with
          launch_e1b_aws.py main_replace;
       4  main runs were never launched (a partial main launch): the operator launches the remaining main runs
          with launch_e1b_aws.py main, and this script exits 4. No replacement is decided before that;
       1  a provenance failure (or an error): stop, exit 1;
       other: exit 5.
  4. Otherwise python experiments/e1b_primary.py, unmodified from the repo, with its stdout saved.
Both outputs are saved under runs/e1b/autorun/ (LF), never overwriting an earlier file; the exclusions output holds
reasons only. Stop early by creating the STOP file (default scripts/e1b_autorun.STOP, checked every second; on
Windows TaskStop does not kill this process).
Exit codes: 0 the analysis ran and exited 0; 1 --exclusions-only reported a provenance failure; 2 stopped; 3 a
replacement is needed; 4 main runs not launched; 5 nothing launched, a collect refusal or error, a file of step 0
that differs from HEAD, or another non-zero exit of e1b_primary.py.

    python scripts/e1b_autorun.py main                      # poll every 300 s
    python scripts/e1b_autorun.py main_replace --interval=600 --stop-file=PATH
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True       # leave no new __pycache__ next to the pinned files
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

HERE = Path(__file__).resolve().parent
PRIMARY = "experiments/e1b_primary.py"
# The collector, this script, the collector's imports, and the analysis: each must equal its blob at HEAD.
COMMITTED = ("scripts/launch_e1b_aws.py", "scripts/e1b_autorun.py", "scripts/e1b_job.py",
             "experiments/e1b_registry.py", PRIMARY)
OUT_DIR = ROOT / "runs" / "e1b" / "autorun"
POLL = 300
L = None                             # scripts/launch_e1b_aws, imported once COMMITTED has been checked


def say(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def not_committed():
    """[(path, why)] for each file of COMMITTED that is not its blob at HEAD. Uses git directly, not the launcher."""
    bad = []
    for p in COMMITTED:
        r = subprocess.run(["git", "-C", str(ROOT), "cat-file", "blob", "HEAD:%s" % p], capture_output=True)
        f = ROOT / p
        if r.returncode != 0:
            bad.append((p, "is not committed at HEAD"))
        elif not f.is_file():
            bad.append((p, "is missing from the working tree"))
        else:
            wt = f.read_bytes()
            if wt != r.stdout and wt.replace(b"\r\n", b"\n") != r.stdout:
                bad.append((p, "differs from its blob at HEAD"))
    return bad


def refuse_uncommitted(when):
    bad = not_committed()
    for p, why in bad:
        say("REFUSE: %s %s; the collector, the autorun and the registered analysis run unmodified from the repo "
            "(pre-registration section 8). Checked %s." % (p, why, when))
    return bool(bad)


def run_primary(*args):
    r = subprocess.run([sys.executable, PRIMARY, *args], cwd=str(ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return r.returncode, r.stdout, r.stderr


def save(name, text):
    """runs/e1b/autorun/<name>, LF; an existing file is never overwritten (a suffix is added)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p, k = OUT_DIR / name, 1
    while p.exists():
        k += 1
        p = OUT_DIR / ("%s_%d%s" % (Path(name).stem, k, Path(name).suffix))
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return p.relative_to(ROOT).as_posix()


def main(argv):
    global L
    opts = {a.split("=", 1)[0]: a.split("=", 1)[1] if "=" in a else True for a in argv if a.startswith("--")}
    pos = [a for a in argv if not a.startswith("--")]
    if len(pos) != 1 or pos[0] not in ("main", "main_replace") or set(opts) - {"--interval", "--stop-file"}:
        raise SystemExit("usage: python scripts/e1b_autorun.py {main|main_replace} [--interval=SECONDS] "
                         "[--stop-file=PATH]")
    stage = pos[0]
    watch = ["main"] if stage == "main" else ["main", "main_replace"]
    try:
        interval = int(opts.get("--interval", POLL))
    except ValueError:
        interval = 0
    if interval < 1:
        raise SystemExit("--interval must be a positive number of seconds")
    stop = Path(opts["--stop-file"]) if isinstance(opts.get("--stop-file"), str) else HERE / "e1b_autorun.STOP"
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    # 0. the committed collector, autorun and analysis, before the launcher is even imported
    if refuse_uncommitted("at start"):
        return 5
    L = importlib.import_module("launch_e1b_aws")
    (HERE / "e1b_autorun.pid").write_text(str(os.getpid()), encoding="ascii")
    say("e1b_autorun pid %d, stage %s (watching %s); poll every %d s; stop with %s"
        % (os.getpid(), stage, ", ".join(watch), interval, stop))

    # 1. wait: status.json files and instance states only
    last = None
    while True:
        if stop.exists():
            say("STOP file found; exiting before collection")
            return 2
        rows = {}
        try:
            for s in watch:
                rows[s] = L.status_rows(s)
        except (RuntimeError, OSError, ValueError) as e:      # a transient AWS or network failure: keep polling
            say("status poll failed (%s); retrying" % str(e)[:200])
            rows = None
        if rows is not None:
            if not rows[stage]:
                say("no launched run in stage %s (%s); nothing to wait for" % (stage, L.LAUNCHES.name))
                return 5
            key = tuple((s, L.summary(r)) for s, r in rows.items())
            if key != last:
                for s, r in rows.items():
                    say("%s: %s" % (s, L.summary(r)))
                last = key
            if all(x["final"] or x["lost"] for r in rows.values() for x in r):
                break
        t_end = time.time() + interval
        while time.time() < t_end:
            if stop.exists():
                break
            time.sleep(1)
    for s, r in rows.items():
        for x in r:
            if x["lost"]:
                say("NO FINAL RC %s (instance %s %s)" % (x["run_id"], x["instance_id"], x["state"]))
            elif x["rc"] != 0:
                say("FAILED %s (rc %s)" % (x["run_id"], x["rc"]))

    # 2. collect, before any analysis, with the committed collector
    if refuse_uncommitted("before collection"):
        return 5
    for s in watch:
        say("every launched run of %s has settled; collecting" % s)
        try:
            L.collect(s)
        except (SystemExit, RuntimeError, OSError) as e:
            say("collect %s refused or failed: %s" % (s, e))
            return 5

    # 3. exclusions: reasons only
    say("python %s --exclusions-only" % PRIMARY)
    rc, out, err = run_primary("--exclusions-only")
    path = save("%s_%s_exclusions.txt" % (stage, stamp), out + ("\n[stderr]\n" + err if err.strip() else ""))
    print(out, end="" if out.endswith("\n") or not out else "\n", flush=True)
    say("exclusions exit %d; output saved to %s" % (rc, path))
    n_repl = sum("REPLACEMENT NEEDED" in ln for ln in out.splitlines())
    n_main = sum("MAIN RUN NOT LAUNCHED" in ln for ln in out.splitlines())
    if rc == 3:
        say("REPLACEMENTS NEEDED before the registered analysis (section 7): %d listed above. Analysis NOT run. "
            "Nothing is written or launched from here: list them in runs/e1b/main_replace.json and run "
            "python scripts/launch_e1b_aws.py main_replace --commit=<registered commit> --submit." % n_repl)
        return 3
    if rc == 4:
        say("MAIN RUNS NOT LAUNCHED: %d listed above. The main stage is incomplete, so no replacement is decided "
            "yet. Launch the remaining main runs: python scripts/launch_e1b_aws.py main --commit=<registered commit> "
            "--submit, then run this script again. Analysis NOT run." % n_main)
        return 4
    if rc == 1:
        print(err[-4000:], flush=True)
        say("PROVENANCE FAILURE (e1b_primary --exclusions-only exit 1): stopping. Analysis NOT run; nothing is "
            "written or launched from here. See the output above.")
        return 1
    if rc != 0:
        print(err[-4000:], flush=True)
        say("e1b_primary --exclusions-only exited %d: stopping. Analysis NOT run." % rc)
        return 5

    # 4. the registered analysis
    say("no replacement needed; running the registered analysis: python %s" % PRIMARY)
    t0 = time.time()
    rc, out, err = run_primary()
    path = save("%s_%s_primary_stdout.txt" % (stage, stamp), out)
    if err.strip():
        save("%s_%s_primary_stderr.txt" % (stage, stamp), err)
    say("analysis exit %d after %.0f s; stdout saved to %s" % (rc, time.time() - t0, path))
    print(out, end="" if out.endswith("\n") or not out else "\n", flush=True)
    if rc != 0:
        print(err[-4000:], flush=True)
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
