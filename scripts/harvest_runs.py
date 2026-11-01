#!/usr/bin/env python
"""Fetch finished HF Jobs logs into the repo and collect them into run artifacts.

WHY THIS EXISTS. On 2026-08-23 a six-trap grid ran at a cost of $17.23 and produced the result
the paper's headline rests on. A month later every byte of it was gone: the job logs had been
fetched to a session scratchpad, `collect()` was never run on them, and the scratchpad was
cleaned. `runs/prime/` is tracked by git and was empty the whole time. The data was never lost
to a failure -- there was simply no step that moved it from ephemeral storage to durable storage,
and so nobody noticed it had not happened.

This is that step. It is deliberately boring and idempotent: run it after any batch of jobs and
it will fetch what it does not already have, write the raw log under runs/prime/logs/, run the
established collector over it, and leave both in a tracked directory ready to commit.

    python scripts/harvest_runs.py --jobs 6a8aaadc...,6a8aaade... --arm hack --steps 400
    python scripts/harvest_runs.py --manifest runs/prime/jobs.json

The raw log is kept alongside the parsed artifact on purpose. The parser is ours and will change;
the log is the evidence, and a re-parse under a corrected estimator is only possible if the
evidence survives. That is exactly what was missing in August.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGDIR = ROOT / "runs" / "prime" / "logs"
OUTDIR = ROOT / "runs" / "prime"
INDEX = OUTDIR / "harvest_index.json"


def _api():
    from huggingface_hub import HfApi
    return HfApi()


def fetch_log(job_id: str, force: bool = False) -> Path | None:
    """Pull one job's stdout to runs/prime/logs/<job_id>.log. Returns None if the job is not
    finished -- an unfinished job's log is a partial trajectory and parsing it would silently
    produce a short series rather than an error."""
    LOGDIR.mkdir(parents=True, exist_ok=True)
    dest = LOGDIR / ("%s.log" % job_id)
    if dest.exists() and not force:
        print("  have   %s  (%d bytes)" % (job_id[:16], dest.stat().st_size))
        return dest
    api = _api()
    try:
        stage = getattr(getattr(api.inspect_job(job_id=job_id), "status", None), "stage", None)
    except Exception as e:
        print("  ERROR  %s  inspect failed: %r" % (job_id[:16], e))
        return None
    if stage not in ("COMPLETED", "ERROR", "CANCELLED"):
        print("  skip   %s  stage=%s (not finished)" % (job_id[:16], stage))
        return None
    n = 0
    with io.open(dest, "w", encoding="utf-8", errors="replace", newline="\n") as fh:
        try:
            for line in api.fetch_job_logs(job_id=job_id):
                fh.write(str(line) + "\n")
                n += 1
        except Exception as e:
            fh.write("[log stream ended: %s]\n" % type(e).__name__)
    print("  FETCH  %s  stage=%s  %d lines -> %s" % (job_id[:16], stage, n, dest.name))
    return dest


def collect_one(logpath: Path, arm: str, task: str | None, seed: int, steps: int) -> Path | None:
    """Run the established collector over a fetched log."""
    # launch_prime_probe.py is a SCRIPT, not a module: its argv parsing and dry-run printing run
    # at import time, with no `if __name__ == "__main__"` guard anywhere in the file. Importing it
    # therefore re-invokes the launcher and parses OUR argv as a run spec. Shell out instead.
    import subprocess
    if task is None:
        task = _task_from_log(logpath)
        if task is None:
            print("  ERROR  %s  cannot determine task from log; pass --task" % logpath.name)
            return None
    # The launcher maps the arm to a LABEL for the filename ("hack" -> "hacking"), so do not
    # assume arm == label. Glob on (task, seed), which is unambiguous within a run directory.
    pat = "prime_*_%s_seed%d.json" % (task, seed)
    cmd = [sys.executable, str(ROOT / "scripts" / "launch_prime_probe.py"),
           arm, str(seed), str(steps), task, "5e-6", "--collect=%s" % logpath]
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if r.returncode != 0:
        print("  ERROR  collect failed rc=%d: %s" % (r.returncode, (r.stderr or "")[-300:]))
        return None
    # Do NOT parse the path out of stdout: the repo path contains a space ("Fligh recorder"), so
    # splitting the line on whitespace reconstructs a broken path. The collector's output name is
    # deterministic -- prime_<arm>_<task>_seed<seed>.json under runs/prime -- so build it instead.
    hits = sorted(OUTDIR.glob(pat), key=lambda p: p.stat().st_mtime, reverse=True)
    if not hits:
        print("  ERROR  collector rc=0 but nothing matching %s" % pat)
        print("         stdout tail: %s" % (r.stdout or "").strip().splitlines()[-1:])
        return None
    out = hits[0]
    print("  WROTE  %s  (%d bytes)" % (out.relative_to(ROOT), out.stat().st_size))
    return out


def _task_from_log(logpath: Path) -> str | None:
    """The task name is on every ONSET line; read the first one rather than trusting a filename."""
    import re
    rx = re.compile(r"^ONSET step \d+ task (\S+)")
    with io.open(logpath, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = rx.match(line)
            if m:
                return m.group(1)
    return None


def _load_index() -> dict:
    if INDEX.exists():
        try:
            return json.loads(INDEX.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", help="comma-separated HF job ids")
    ap.add_argument("--manifest", help="json file: [{job_id, arm, task, seed, steps}, ...]")
    ap.add_argument("--arm", default="hack")
    ap.add_argument("--task", default=None, help="default: read from the log's ONSET lines")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--force", action="store_true", help="re-fetch logs already on disk")
    a = ap.parse_args(argv)

    if a.manifest:
        spec = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    elif a.jobs:
        spec = [{"job_id": j.strip(), "arm": a.arm, "task": a.task, "seed": a.seed,
                 "steps": a.steps} for j in a.jobs.split(",") if j.strip()]
    else:
        ap.error("pass --jobs or --manifest")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    index = _load_index()
    ok = 0
    for s in spec:
        jid = s["job_id"]
        print("job %s" % jid)
        log = fetch_log(jid, force=a.force)
        if log is None:
            continue
        art = collect_one(log, s.get("arm", a.arm), s.get("task", a.task),
                          int(s.get("seed", a.seed)), int(s.get("steps", a.steps)))
        index[jid] = {"log": str(log.relative_to(ROOT)),
                      "artifact": str(art.relative_to(ROOT)) if art else None,
                      "arm": s.get("arm", a.arm), "seed": int(s.get("seed", a.seed)),
                      "task": s.get("task") or _task_from_log(log)}
        if art:
            ok += 1

    INDEX.write_text(json.dumps(index, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print()
    print("harvested %d/%d  ->  %s" % (ok, len(spec), OUTDIR.relative_to(ROOT)))
    print("index: %s" % INDEX.relative_to(ROOT))
    print()
    print("COMMIT THESE. runs/prime/ is tracked and was empty for a month while the only copy of"
          " a $17 grid sat in a scratchpad that then got cleaned.")
    return 0 if ok == len(spec) else 1


if __name__ == "__main__":
    raise SystemExit(main())
