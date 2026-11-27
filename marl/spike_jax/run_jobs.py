"""
run_jobs.py -- run jobs from a job file, at most --workers at a time.
Each job line: arguments for the script, e.g. "--n 4 --cond exploit --seed 0 --params indep".
Per-job stdout goes to <logdir>/<job>.log. Scripts skip jobs whose output already exists, so
the runner can be restarted safely. On Linux, --cores-per-job pins each job to its own block of
cores with taskset (worker slot k gets cores offset + k*c .. offset + (k+1)*c - 1).

    python run_jobs.py jobs_L.txt --workers 26 --cores-per-job 2
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobfile")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--script", default="train_ippo.py")
    ap.add_argument("--logdir", default=os.path.join("results", "L", "logs"))
    ap.add_argument("--cores-per-job", type=int, default=0)
    ap.add_argument("--core-offset", type=int, default=0)
    a = ap.parse_args()
    jobs = [l.strip() for l in open(a.jobfile) if l.strip() and not l.startswith("#")]
    logdir = os.path.join(HERE, a.logdir)
    os.makedirs(logdir, exist_ok=True)
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    pin = a.cores_per_job > 0 and sys.platform.startswith("linux")
    free_slots = list(range(a.workers))
    running, t0 = [], time.time()
    queue = list(jobs)
    while queue or running:
        while queue and free_slots:
            args = queue.pop(0)
            slot = free_slots.pop(0)
            name = args.replace("--", "").replace(" ", "_")
            f = open(os.path.join(logdir, name + ".log"), "w")
            cmd = [sys.executable, os.path.join(HERE, a.script)] + args.split()
            if pin:
                lo = a.core_offset + slot * a.cores_per_job
                cmd = ["taskset", "-c", f"{lo}-{lo + a.cores_per_job - 1}"] + cmd
            p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=HERE)
            running.append((p, f, args, time.time(), slot))
            print(f"[{time.time() - t0:6.0f}s] start {args} (slot {slot})", flush=True)
        for item in list(running):
            p, f, args, ts, slot = item
            if p.poll() is not None:
                f.close()
                running.remove(item)
                free_slots.append(slot)
                print(f"[{time.time() - t0:6.0f}s] done  {args} rc={p.returncode} ({time.time() - ts:.0f}s)",
                      flush=True)
        time.sleep(5)
    print(f"all jobs finished in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
