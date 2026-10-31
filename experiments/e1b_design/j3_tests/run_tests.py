"""Tests of scripts/e1b_j3_job.py and scripts/launch_e1b_j3.py without AWS or a GPU (a design record). ASCII only.

WHAT IT DOES (every file it writes goes under --scratch, which must lie outside the repository)
  data     the real experiments/e1b_primary.py (this checkout) on synthetic runs from its own self-test builders
           (_st_run, _st_record, _st_manifest): build_data keeps exactly load_e1b's used runs (an excluded run is
           left out), every stored text and judge_small, sorted rows, the same bytes twice; a missing data manifest
           and a tampered data file are refused. R10: the launch-record readers name runs/e1b/launches.json exactly,
           no E1b reader walks runs/e1b, and a J3 file name is not a result-file name.
  bake     a scratch clone holding this checkout's E1b files, committed there: bake() from that commit; its
           provenance, placeholders, header (scripts/e1b_job.py's) and lock; refusals for a bad run id, an edited
           template, a changed judge module at the commit, a tampered data file. With --uv (uv 0.12.23):
           `uv export --script --locked` accepts experiments/e1b_uv.lock as the baked job's lock.
  stops    (--venv: CPython 3.12.15 with the locked packages and CPU torch, e.g. the job smoke harness's venv) the
           baked job stops with exit 1 and no score line when the package, the lock or the data is not the baked one.
  cpu      (--venv) the baked job, patched only where a CPU run needs it (a tiny public model in place of both
           judges, cpu, float32, a small token budget), scores the synthetic data end to end: stdout is only
           "J3 <phase> <n>/<total>" lines, no score value on stdout or stderr, the result has the E1 rescoring's
           format keyed by the E1b run ids, complete; experiments/e1b_primary.load_j3 reads it (coverage, and the
           judge, revision and dtype flags; a relabelled copy has none; a hole makes it provisional). Then the
           validation-gate failure (rc 2, target never loaded) and an abort mid-scoring (result marked aborted). R8:
           a judge whose Hub cache shows another snapshot stops the job (exit 1) before anything is scored.
  launcher a scratch clone with a freeze record (manifest of the freeze commit's blobs, a fake pending
           OpenTimestamps receipt, a filled reader and J3 approval), a stand-in e1b_primary.py
           (stub_e1b_primary.py: the real one's pins name local-only files) and a fake aws CLI (fake_aws.py) first on
           PATH: dry runs; every --submit refusal; a submit and its uploads, spec, user data and launch record; a
           capacity fallback with a smaller backstop; the ceiling room after a launch; --status (counts only, measured
           hours); --collect before and after the final rc, its checks and its never-overwrite rule. R7: the reader
           and approval cells by launch_e1b_aws's rule (a real date, curly quotes accepted). R8: collect refuses a
           result whose judge, revision, dtype or measured snapshots differ from the registered values. R10: the J3
           record stays out of the main launcher's records. R9: a main and a J3 launcher writing
           runs/e1b/instance_hours.json at once lose no entry.
  userdata the instance's user data run under bash with fake aws, python3 (fake_job.py for `uv run`), nvidia-smi and
           shutdown: a normal run (uploads, final status last, backstop first), a failing job, a missing job and
           the watchdog.

    python experiments/e1b_design/j3_tests/run_tests.py --scratch DIR [--venv PY] [--uv UV] [--hf-home DIR]
Exit 0 when every check passes and no section was skipped (--allow-skip lets the venv sections be skipped).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PY = sys.executable
BUCKET = "fr-deploy-325527186655"
TINY = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
OTS_MAGIC = b"\x00OpenTimestamps\x00\x00Proof\x00\xbf\x89\xe2\xe8\x84\xe8\x92\x94"
COPY = ("scripts/e1b_j3_job.py", "scripts/launch_e1b_j3.py", "scripts/e1b_job.py", "scripts/launch_e1b_aws.py",
        "scripts/e1b_autorun.py", "experiments/e1b_registry.py", "experiments/e1b_uv.lock",
        "flightrecorder/repro/reward_judge.py", "flightrecorder/repro/twin_gate.py", "experiments/e1b_power.py",
        "experiments/e1b_pilot_gate.py", "experiments/e1b_preregistration.md", "experiments/e1b_analysis_spec.md",
        "experiments/e1b_pins.json", "experiments/e1b_selftest.txt")
SCORE_RE = re.compile(r"(?<![\d.])0\.\d{3,}|(?<![\d.])1\.0{3,}")      # a score-like number
J3_LINE = re.compile(r"^J3 [a-z0-9_]+ \d+/\d+$")
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    detail = str(detail)[-2500:].encode("ascii", "backslashreplace").decode("ascii")    # printable on any console
    print("%s  %s%s" % ("PASS" if cond else "FAIL", name, "" if cond else "\n      " + detail), flush=True)
    return bool(cond)


def run(cmd, cwd, env=None, timeout=1800):
    r = subprocess.run([str(c) for c in cmd], cwd=str(cwd), env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout)
    return r.returncode, r.stdout, r.stderr


def lf(p):
    return Path(p).read_bytes().replace(b"\r\n", b"\n")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def short(p):
    """The 8.3 form of an existing path on Windows (MAX_PATH breaks transformers' imports and the Hub cache)."""
    p = str(p)
    if os.name != "nt":
        return p
    import ctypes
    buf = ctypes.create_unicode_buffer(1024)
    n = ctypes.windll.kernel32.GetShortPathNameW(p, buf, 1024)
    return buf.value if 0 < n < 1024 else p


def rmtree(p):
    def onexc(func, path, exc):
        os.chmod(path, stat.S_IWRITE)
        func(path)
    if Path(p).exists():
        shutil.rmtree(p, onexc=onexc) if sys.version_info >= (3, 12) else shutil.rmtree(
            p, onerror=lambda f, q, e: onexc(f, q, e))


def load(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def git_bash():
    if os.name != "nt":
        return shutil.which("bash")
    g = shutil.which("git")
    for p in (Path(g).resolve().parents if g else []):
        for c in (p / "usr" / "bin" / "bash.exe", p / "bin" / "bash.exe"):
            if c.is_file():
                return str(c)
    raise SystemExit("Git bash not found")


def posix(p):
    """A path as Git bash sees it (C:/x -> /c/x)."""
    s = Path(p).as_posix()
    return "/" + s[0].lower() + s[2:] if os.name == "nt" and len(s) > 1 and s[1] == ":" else s


class Ctx:
    def __init__(self, a):
        self.root = Path(a.scratch).resolve()
        if self.root == REPO or REPO in self.root.parents:
            raise SystemExit("--scratch must lie outside the repository")
        self.root.mkdir(parents=True, exist_ok=True)
        self.venv = a.venv
        self.uv = a.uv
        self.hf_home = a.hf_home
        self.E = None            # the real experiments/e1b_primary.py
        self.data = None         # {"path", "sha256", "runs", "header", "main", "records", "prereg"}
        self.clone = None


def git(clone, *a):
    rc, out, err = run(["git", *a], clone)
    if rc != 0:
        raise RuntimeError("git %s: %s" % (" ".join(a), out + err))
    return out.strip()


def make_clone(ctx, name):
    """A scratch clone with this checkout's E1b files (LF) and the stand-in e1b_primary.py, committed."""
    clone = ctx.root / name
    rmtree(clone)
    rc, out, err = run(["git", "-c", "core.autocrlf=false", "clone", "-q", "--no-hardlinks", str(REPO), str(clone)],
                       ctx.root)
    assert rc == 0, out + err
    for k, v in (("core.autocrlf", "false"), ("user.name", "e1b-j3-test"), ("user.email", "e1b-j3-test@example.invalid")):
        git(clone, "config", k, v)
    files = re.search(r'FILES="([^"]*)"', (REPO / "experiments/e1b_preregistration.md").read_text(
        encoding="utf-8")).group(1).split()                  # section 13's freeze list, as this checkout has it
    for rel in sorted(set(COPY) | set(files)):
        if (REPO / rel).is_file():
            (clone / rel).parent.mkdir(parents=True, exist_ok=True)
            (clone / rel).write_bytes(lf(REPO / rel))
    (clone / "experiments/e1b_primary.py").write_bytes(lf(HERE / "stub_e1b_primary.py"))
    # F holds the draft's record cells whatever the real file holds: the tests write their own, and the refusal tests
    # need the draft values
    P = clone / "experiments/e1b_preregistration.md"
    b = P.read_bytes()
    for pat, rep in ((rb"(?m)^(\| Commit hash \| )[^|\n]*( \|)$", rb"\1none yet\2"),
                     (rb"(?m)^(\| Human reader before the freeze[^\n]*\| )[^|\n]*( \|)$", rb"\1none yet\2"),
                     (rb"(?m)^(\| Pilot \| 6 runs[^\n]*\| )[^|\n]*( \|)$", rb"\1pending\2"),
                     (rb"(?m)^(\| Primary \| 96 runs[^\n]*\| )[^|\n]*( \|)$", rb"\1pending\2"),
                     (rb"(?m)^(\| J3 scoring [^\n]*\| )[^|\n]*( \|)$", rb"\1pending approval\2")):
        b = re.sub(pat, rep, b)
    P.write_bytes(b)
    git(clone, "add", "-A")
    git(clone, "commit", "-q", "-m", "scratch: E1b files with the J3 job (freeze candidate F)")
    return clone, git(clone, "rev-parse", "HEAD")


# ==========================================================================================
# data: the real load_e1b on synthetic runs
def synth_set(ctx, E, name, spec):
    """spec: [(arm, task, seed, kwargs)] -> (main dir with result files and manifests, load_e1b's keyword arguments
    for the synthetic registry: launch records, registered SHA-256s, pre-registration, git repository, no measured
    hours; {run id: result})."""
    root = ctx.root / "data" / name
    rmtree(root)
    main = root / "main"
    main.mkdir(parents=True)
    git_dir = E._st_git(root)              # fills E._ST_COMMIT with the synthetic registered commits
    runs = [E._st_run(arm, task, seed, "j3", **kw) for arm, task, seed, kw in spec]
    for d in runs:
        E._st_dump(main / (d["run_id"] + ".json"), d)
    E._st_manifest(main)
    recs = [E._st_record(d, i) for i, d in enumerate(runs)]
    prereg = E._st_prereg(root / "prereg.md")
    kw = {"launch_records": recs, "registered": E._ST_REG, "prereg": prereg, "git_dir": git_dir, "instance_hours": {}}
    return main, kw, {d["run_id"]: d for d in runs}


def _moved(paths):
    """Rename the existing ones aside; return a function that puts them back."""
    done = [(q, q.with_name(q.name + ".moved")) for q in paths if q.is_file()]
    for a, b in done:
        a.rename(b)
    return lambda: [b.rename(a) for a, b in done]


def test_data(ctx):
    sys.path[:0] = [str(REPO / "experiments"), str(REPO / "scripts")]
    E = ctx.E = load("e1b_primary", REPO / "experiments" / "e1b_primary.py")
    JJ = load("e1b_j3_job_repo", REPO / "scripts" / "e1b_j3_job.py")
    spec = [("judge", "count_vowels", 300, {}), ("judgegate", "count_vowels", 300, {}),
            ("judge", "is_prime", 300, {"zero_first5": True})]          # excluded by rule (b)
    main, kw, by_id = synth_set(ctx, E, "two", spec)
    out = ctx.root / "data" / "two" / "data.jsonl.gz"
    info = JJ.build_data(out, main, **kw)
    sel = E.load_e1b(main, **kw)
    used = sorted(E._run_id(r) for r in sel["used"]["hacking"] + sel["used"]["twin"])
    check("data: the runs are exactly load_e1b's used runs (the rule-(b) run is left out)",
          sorted(info["header"]["runs"]) == used == ["e1b_judge_count_vowels_s300", "e1b_judgegate_count_vowels_s300"],
          (sorted(info["header"]["runs"]), used, sel["excluded"], sel["unregistered"]))
    check("data: 2 runs x 400 steps x 8 rows; header counts and used tags as e1b_primary records them",
          info["n_rows"] == 6400 and info["n_runs"] == 2 and info["header"]["used"] ==
          {a: [E._tag(r) for r in v] for a, v in sel["used"].items()} and
          info["header"]["data_manifest"]["sha256"] == E.data_manifest(main)["sha256"], info["header"]["used"])
    with gzip.open(out, "rt", encoding="ascii") as f:
        f.readline()
        rows = [json.loads(x) for x in f]
    ok = all(r["text"] == by_id[r["run"]]["completions"][r["step"]]["texts"][r["i"]] and
             r["judge_small"] == by_id[r["run"]]["completions"][r["step"]]["judge_small"][r["i"]] for r in rows)
    check("data: every row is the stored full text and judge_small; rows sorted by (run, step, i)",
          ok and [(r["run"], r["step"], r["i"]) for r in rows] == sorted((r["run"], r["step"], r["i"]) for r in rows))
    out2 = ctx.root / "data" / "two" / "again.jsonl.gz"
    info2 = JJ.build_data(out2, main, **kw)
    check("data: the same runs give the same bytes (deterministic gzip)", info2["sha256"] == info["sha256"] ==
          JJ.check_data(out)[0])
    back = _moved([main / "MANIFEST.sha256", main.parent / "main_replace" / "MANIFEST.sha256"])
    try:
        JJ.build_data(ctx.root / "data" / "two" / "x.jsonl.gz", main, **kw)
        refused = None
    except SystemExit as e:
        refused = str(e)
    finally:
        back()
    check("data: refused without the collector's data manifest", refused and "data manifest" in refused, refused)
    bad = ctx.root / "data" / "two" / "bad.jsonl.gz"
    with gzip.open(out, "rt", encoding="ascii") as f:
        lines = f.read().splitlines(True)
    with gzip.open(bad, "wt", encoding="ascii") as f:
        f.write("".join(lines[:-1]))
    try:
        JJ.check_data(bad)
        why = None
    except ValueError as e:
        why = str(e)
    check("data: check_data refuses a file with a row missing", why is not None, why)
    ctx.data = {"path": out, "sha256": info["sha256"], "header": info["header"], "main": main, "kw": kw,
                "used": sel["used"]}
    test_records_apart_static(ctx, E)


# E1b files that read launch records or result files (finding R10: none may take a J3 file for a main record)
READERS = ("experiments/e1b_primary.py", "experiments/e1b_pilot_gate.py", "scripts/e1b_autorun.py",
           "scripts/launch_e1b_aws.py", "experiments/e1b_power.py", "experiments/e1b_registry.py",
           "experiments/e1b_offline_gate.py")


def test_records_apart_static(ctx, E):
    """Finding R10: J3 launch records live at runs/e1b/j3/launches.json, and nothing else reads them as main or
    pilot records: the launch-record readers name runs/e1b/launches.json exactly, and no E1b reader walks runs/e1b
    (no rglob, os.walk or '**' glob); the result readers glob only e1b_*_s<seed>.json."""
    sys.path[:0] = [str(REPO / "scripts")]
    L = load("launch_e1b_aws_repo", REPO / "scripts" / "launch_e1b_aws.py")
    J3m = load("launch_e1b_j3_repo", REPO / "scripts" / "launch_e1b_j3.py")
    runs = REPO / "runs" / "e1b"
    check("records apart: main and pilot launch records are read from runs/e1b/launches.json by its exact path (the "
          "launcher, e1b_primary); J3's go to runs/e1b/j3/launches.json",
          L.LAUNCHES == runs / "launches.json" == E.LAUNCHES_B and J3m.LAUNCHES == runs / "j3" / "launches.json",
          (L.LAUNCHES, E.LAUNCHES_B, J3m.LAUNCHES))
    walks = {}
    for rel in READERS:
        src = (REPO / rel).read_text(encoding="utf-8")
        hits = re.findall(r"\.rglob\(|os\.walk\(|glob\(\s*[\"'][^\"']*\*\*", src)
        if hits:
            walks[rel] = hits
    globs = {rel: re.findall(r"(\S+)\.glob\(([^)]*)\)", (REPO / rel).read_text(encoding="utf-8")) for rel in READERS}
    globs = {k: v for k, v in globs.items() if v}
    check("records apart: no E1b reader walks runs/e1b (no rglob, os.walk or '**' glob); the only globs are the pilot "
          "gate's e1b_*_s<seed>.json and report names in runs/e1b/pilot and e1b_primary's E1 data in runs/e1/main",
          not walks and globs == {"experiments/e1b_pilot_gate.py": [("pilot_dir", '"e1b_*_s%d.json" % seed'),
                                                                  ("pilot", '"pilot_gate*"')],
                                  "experiments/e1b_primary.py": [("sorted(Path(P.MAIN)", '"*.json"')]} and
          Path(E.P.MAIN).resolve() == (REPO / "runs" / "e1" / "main").resolve(), (walks, globs, E.P.MAIN))
    check("records apart: a J3 run id or its status file name is not a result-file name (e1b_*_s<seed>.json)",
          not E.RESULT_FILE.fullmatch("e1b_j3_qwen25_7b_a1.json") and not E.RESULT_FILE.fullmatch("launches.json")
          and not E.RESULT_FILE.fullmatch("qwen25_7b.json") and E.RESULT_FILE.fullmatch("e1b_judge_is_prime_s300.json"))


# ==========================================================================================
# bake
def bake_cli(clone, data, rid, out, commit="HEAD"):
    """scripts/e1b_j3_job.py --bake in the clone (its own modules, its own git) -> (rc, provenance or None, text)."""
    rc, so, se = run([PY, "-B", "scripts/e1b_j3_job.py", "--bake", commit, str(data), rid, str(out)], clone)
    try:
        prov = json.loads(so)
    except ValueError:
        prov = None
    return rc, prov, so + se


def header_of(clone):
    rc, so, se = run([PY, "-B", "-c", "import sys; sys.path.insert(0, 'scripts'); import e1b_job as J, e1b_j3_job as JJ;"
                      " sys.stdout.write(repr((J.header() == JJ.header(), J.header())))"], clone)
    import ast
    return ast.literal_eval(so) if rc == 0 else (False, se)


def test_bake(ctx):
    clone, F = make_clone(ctx, "clone_bake")
    ctx.clone, ctx.F = clone, F
    out = ctx.root / "baked"
    rmtree(out)
    rid = "e1b_j3_qwen25_7b_a1"
    rc, prov, text = bake_cli(clone, ctx.data["path"], rid, out)
    check("bake: --bake from the clone's commit succeeds", rc == 0 and prov, text[-1500:])
    if not prov:
        raise SystemExit("cannot continue without a baked job")
    job = Path(prov["job_path"])
    body = job.read_bytes()
    check("bake: provenance = the commit, the blobs' SHA-256 and the data's",
          prov["git_commit"] == F and prov["template_sha256"] == sha(lf(clone / "scripts/e1b_j3_job.py")) and
          prov["lock_sha256"] == sha(lf(clone / "experiments/e1b_uv.lock")) and
          prov["data_sha256"] == ctx.data["sha256"] and prov["job_sha256"] == sha(body) and
          prov["n_completions"] == 6400 and prov["n_runs"] == 2 and job.parent == out / rid, prov)
    check("bake: job.py is ASCII with LF only, no placeholder left; job.py.lock is the lock; the data sits beside it",
          body.decode("ascii") and b"\r" not in body and not re.findall(rb"__[A-Z][A-Z0-9_]*__", body) and
          (job.parent / "job.py.lock").read_bytes() == lf(clone / "experiments/e1b_uv.lock") and
          sha((job.parent / "data.jsonl.gz").read_bytes()) == ctx.data["sha256"])
    same, hdr = header_of(clone)
    check("bake: the PEP 723 header is scripts/e1b_job.py's, byte for byte", same is True and
          body.decode("ascii").startswith(hdr), hdr)
    check("bake: the baked run id, commit, data SHA-256 and row count are the baked constants",
          ('RUN_ID = "%s"' % rid).encode() in body and F.encode() in body and
          ('DATA_SHA256, N_ROWS = "%s", 6400' % ctx.data["sha256"]).encode() in body)
    for bad_id in ("e1b_j3_qwen25_7b_a0", "e1b_judge_count_vowels_s300", "x"):
        rc, _, text = bake_cli(clone, ctx.data["path"], bad_id, ctx.root / "baked_bad")
        check("bake: refuses run id %r" % bad_id, rc != 0 and "is not a J3 run id" in text, text[-600:])
    p = clone / "scripts/e1b_j3_job.py"
    keep = p.read_bytes()
    p.write_bytes(keep + b"# edited\n")
    rc, _, text = bake_cli(clone, ctx.data["path"], rid, ctx.root / "baked_bad")
    p.write_bytes(keep)
    check("bake: refuses a working-tree template that differs from the commit", rc != 0 and
          "differs from its committed version" in text, text[-600:])
    git(clone, "checkout", "-q", "-b", "testhack_changed")
    th = clone / "flightrecorder/repro/reward_testhack.py"
    th.write_bytes(lf(th) + b"# changed\n")
    git(clone, "commit", "-q", "-am", "scratch: reward_testhack changed")
    rc, _, text = bake_cli(clone, ctx.data["path"], rid, ctx.root / "baked_bad")
    git(clone, "checkout", "-q", "-")
    check("bake: refuses a commit whose extract_code module differs from the E1 rescoring's", rc != 0 and
          "reward_testhack.py" in text and "differs from the E1 rescoring's version" in text, text[-600:])
    rc, _, text = bake_cli(clone, ctx.root / "data" / "two" / "bad.jsonl.gz", rid, ctx.root / "baked_bad")
    check("bake: refuses a data file that check_data refuses", rc != 0 and "ValueError" in text, text[-600:])
    ctx.job, ctx.prov = job, prov
    if ctx.uv:
        (ctx.root / "uvcache").mkdir(exist_ok=True)
        env = dict(os.environ, UV_NO_CONFIG="1", UV_CACHE_DIR=short(ctx.root / "uvcache"))
        rc, so, se = run([ctx.uv, "export", "--no-config", "--script", job, "--format", "requirements.txt", "--locked",
                          "-o", ctx.root / "export.txt"], ctx.root, env=env, timeout=900)
        check("bake: uv %s export --script --locked accepts experiments/e1b_uv.lock for the baked job"
              % run([ctx.uv, "--version"], ctx.root)[1].split()[1], rc == 0 and "torch==2.11.0+cu128" in
              (ctx.root / "export.txt").read_text(), so + se)
    else:
        print("SKIP  bake: uv lock check (no --uv)")


# ==========================================================================================
# the baked job under the job's Python (no model for the stops; a tiny model for the CPU run)
def job_env(ctx, case_dir):
    e = {k: v for k, v in os.environ.items() if k not in (
        "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HUB_OFFLINE", "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")}
    hf = Path(ctx.hf_home) if ctx.hf_home else ctx.root / "hf"
    hf.mkdir(parents=True, exist_ok=True)
    tmp = ctx.root / "tmp"
    tmp.mkdir(exist_ok=True)
    e.update(HF_HOME=(r"\\?" + "\\" if os.name == "nt" else "") + str(hf.resolve()), TEMP=short(tmp), TMP=short(tmp),
             TMPDIR=short(tmp), PYTHONUNBUFFERED="1", E1B_J3_STATUS_PATH=str(Path(case_dir) / "status.json"))
    return e


def case(ctx, name, src, lock=True, data=True):
    d = ctx.root / "cases" / name
    rmtree(d)
    d.mkdir(parents=True)
    (d / "job.py").write_text(src, encoding="ascii", newline="\n")
    if lock:
        shutil.copyfile(ctx.job.parent / "job.py.lock", d / "job.py.lock")
    if data:
        shutil.copyfile(ctx.job.parent / "data.jsonl.gz", d / "data.jsonl.gz")
    return d


def run_case(ctx, d, timeout=7200):
    t = time.time()
    rc, so, se = run([short(ctx.venv), short(d / "job.py")], short(d), env=job_env(ctx, short(d)), timeout=timeout)
    rd = lambda n: json.loads((d / n).read_text()) if (d / n).is_file() else None  # noqa: E731
    return {"rc": rc, "stdout": so, "stderr": se, "result": rd("result.json"), "partial": rd("partial.json"),
            "status": rd("status.json"), "seconds": time.time() - t}


def sub1(src, old, new):
    n = src.count(old)
    if n != 1:
        raise RuntimeError("patch target occurs %d times: %r" % (n, old[:80]))
    return src.replace(old, new)


def test_stops(ctx):
    src = ctx.job.read_text(encoding="ascii")
    m = re.search(r'_TAR = gzip\.decompress\(base64\.b64decode\("([A-Za-z0-9+/=]*)"\)\)', src)
    import base64
    tar = bytearray(gzip.decompress(base64.b64decode(m.group(1))))
    tar[700] ^= 0x20
    bad_pkg = src[:m.start(1)] + base64.b64encode(gzip.compress(bytes(tar), mtime=0)).decode() + src[m.end(1):]
    cpu = sub1(src, 'DEVICE = "cuda"', 'DEVICE = "cpu"')      # the data check comes after the device check
    cases = [("package", bad_pkg, {}, "embedded package"), ("lock", src, {"lock": b"x"}, "job.py.lock's SHA-256"),
             ("nolock", src, {"nolock": True}, "no job.py.lock"), ("nocuda", src, {}, "no CUDA device"),
             ("data", cpu, {"data": b"x"}, "data SHA-256")]
    for name, s, how, msg in cases:
        d = case(ctx, "stop_" + name, s, lock=not how.get("nolock"))
        if "lock" in how:
            (d / "job.py.lock").write_bytes((d / "job.py.lock").read_bytes() + how["lock"])
        if "data" in how:
            with open(d / "data.jsonl.gz", "ab") as f:
                f.write(how["data"])
        r = run_case(ctx, d, timeout=900)
        lines = [ln for ln in r["stdout"].splitlines() if ln.strip()]
        check("stops: %s -> exit 1, 'E1B J3 STOP: ...%s', no score line, no result" % (name, msg),
              r["rc"] == 1 and ("E1B J3 STOP" in r["stderr"] and msg in r["stderr"]) and r["result"] is None and
              all(J3_LINE.match(x) and x.split()[1] == "start" for x in lines), (r["rc"], r["stdout"], r["stderr"][-800:]))


def tiny_revision(ctx):
    hf = Path(ctx.hf_home) if ctx.hf_home else None
    snaps = hf / "hub" / ("models--" + TINY.replace("/", "--")) / "snapshots" if hf else None
    if snaps and snaps.is_dir() and len(list(snaps.iterdir())) == 1:
        return next(snaps.iterdir()).name
    with urllib.request.urlopen("https://huggingface.co/api/models/" + TINY, timeout=60) as r:
        return json.load(r)["sha"]


def cpu_patch(src, rev, budget=2048):
    tiny = '{"judge": "%s", "revision": "%s", "dtype": "float32"}' % (TINY, rev)
    src = sub1(src, 'TARGET = {"judge": "Qwen/Qwen2.5-7B-Instruct", "revision": "a09a35458c702b33eeacc393d103063234e8bc28", '
                    '"dtype": "bfloat16"}', "TARGET = " + tiny)
    src = sub1(src, 'VALIDATION = {"judge": "Qwen/Qwen2.5-0.5B-Instruct", "revision": '
                    '"7ae557604adf67be50417f59c2c2f167def9a775", "dtype": "float16"}', "VALIDATION = " + tiny)
    src = sub1(src, 'DEVICE = "cuda"', 'DEVICE = "cpu"')
    src = sub1(src, "CPU_TOKEN_BUDGET = 8192", "CPU_TOKEN_BUDGET = %d" % budget)
    return src


def no_scores(text):
    return [m.group(0) for m in SCORE_RE.finditer(text)]


def test_cpu(ctx):
    E = ctx.E
    rev = tiny_revision(ctx)
    # random tiny weights cannot reproduce the synthetic runs' judge_small, so the per-step tolerance is opened for
    # the main run (the batched-path tolerance stays); the gate's arithmetic and its failure path are checked below
    src = cpu_patch(ctx.job.read_text(encoding="ascii"), rev)
    opened = sub1(src, "VAL_TOL_STEP = 0.02", "VAL_TOL_STEP = 1.0")
    d = case(ctx, "cpu_main", opened)
    r = run_case(ctx, d)
    res = r["result"] or {}
    lines = r["stdout"].splitlines()
    print("      cpu run: rc %s in %.0f s" % (r["rc"], r["seconds"]))
    check("cpu: exit 0, complete, every completion scored", r["rc"] == 0 and res.get("complete") is True and
          res.get("n_scored") == res.get("n_total") == 6400 and res.get("n_nonfinite") == 0,
          (r["rc"], r["stderr"][-2000:]))
    check("cpu: stdout is only 'J3 <phase> <n>/<total>' lines, in phase order",
          lines and all(J3_LINE.match(x) for x in lines) and
          [x.split()[1] for x in lines if not x.split()[1] == "scoring"] ==
          ["start", "data", "validated", "target_loaded", "chunk_edges", "chunk_every5", "chunk_rest", "done"],
          lines)
    check("cpu: no score value on stdout or stderr", not no_scores(r["stdout"]) and not no_scores(r["stderr"]),
          (no_scores(r["stdout"])[:5], no_scores(r["stderr"])[:5], r["stderr"][-1500:]))
    e1 = json.loads((REPO / "experiments/e1_rescore/results/qwen25_7b.json").read_text())
    check("cpu: the result has every top-level key of the E1 rescoring's qwen25_7b.json", set(e1) <= set(res),
          sorted(set(e1) - set(res)))
    runs = ctx.data["header"]["runs"]
    check("cpu: scores are keyed by the E1b run ids, str(step), 8 per step; runs table as E1's",
          sorted(res.get("scores", {})) == sorted(runs) and
          all(sorted(v, key=int) == [str(s) for s in range(400)] and all(len(x) == 8 for x in v.values())
              for v in res["scores"].values()) and
          res["runs"] == {k: {"arm": v["arm"], "task": v["task"], "seed": v["seed"]} for k, v in runs.items()})
    check("cpu: measured provenance: job_file_sha256 of the file run, the commit, the module SHA-256s, the data",
          res.get("job_file_sha256") == sha((d / "job.py").read_bytes()) and
          res["provenance"]["git_commit"] == ctx.F and res["data_sha256"] == ctx.data["sha256"] and
          set(res["provenance"]["module_sha256"]) == {"flightrecorder/repro/reward_judge.py",
                                                      "flightrecorder/repro/reward_testhack.py",
                                                      "flightrecorder/repro/reward_codeset.py"} and
          res["used"] == ctx.data["header"]["used"], res.get("provenance"))
    snap = (res.get("snapshots") or {}).get("target") or {}
    check("cpu: the snapshot loaded is the pinned revision, and no refs/main was written (no default-branch read)",
          snap.get("snapshots") == [rev] and "main" not in snap.get("refs", {}), snap)
    check("cpu: validation gate ran on the first hacking and the first twin run and passed; mode 'test'",
          res["validation"]["passed"] and res["validation"]["runs"] == sorted(runs) and res["mode"] == "test" and
          res["judge_facts"]["padding_side"] == "right" and res["judge_facts"]["max_length"] == 1024,
          res.get("validation", {}).get("max_abs_diff"))
    v = res.get("validation") or {}
    dat = {}
    with gzip.open(ctx.data["path"], "rt", encoding="ascii") as f:
        f.readline()
        for x in f:
            row = json.loads(x)
            dat.setdefault((row["run"], row["step"]), []).append(row["judge_small"])
    ok = v.get("steps") == list(range(10)) and v.get("n_completions") == 160 and all(
        abs(e["stored"] - sum(dat[(run, int(s_))]) / 8) < 1e-12 and abs(e["rescored"] - sum(e["scores"]) / 8) < 1e-9 and
        abs(e["diff"] - (e["rescored"] - e["stored"])) < 1e-12 and len(e["batched"]) == 8
        for run, st_ in v.get("per_step", {}).items() for s_, e in st_.items()) and \
        abs(v["max_abs_diff"] - max(abs(e["diff"]) for st_ in v["per_step"].values() for e in st_.values())) < 1e-12 \
        and v["batched_path_max_abs_diff"] <= v["tolerance_path"] == 0.01
    check("cpu: validation arithmetic: stored = mean of the data's judge_small, diff = rescored - stored, steps 0-9 of "
          "2 runs; the batched path matches the per-step path within 0.01", ok, {k: v.get(k) for k in (
              "steps", "n_completions", "max_abs_diff", "batched_path_max_abs_diff")})
    check("cpu: status.json ends at phase done with all scored and rc null; partial.json equals the result",
          r["status"] == {"run_id": "e1b_j3_qwen25_7b_a1", "phase": "done", "n_scored": 6400, "n_total": 6400,
                          "rc": None, "ts": r["status"]["ts"]} and r["partial"] == res, r["status"])
    # experiments/e1b_primary.load_j3 on the file
    used = ctx.data["used"]["hacking"] + ctx.data["used"]["twin"]
    path = d / "result.json"
    out = []
    j = E.load_j3(path, used, out=out)
    check("load_j3: reads the job's file: complete, 6400 of 6400, 800 of 800 steps; judge, revision, dtype flagged "
          "(tiny model)", j and j["complete"] and
          j["coverage"]["completions_scored"] == 6400 and j["coverage"]["steps_fully_scored"] == 800 and
          not any(j["coverage"]["issues"].values()) and len(j["flags"]) == 3, out)
    rel = dict(res, judge=E.J3["judge"], revision=E.J3["revision"], dtype=E.J3["dtype"])
    p2 = d / "relabelled.json"
    p2.write_text(json.dumps(rel), encoding="ascii")
    j2 = E.load_j3(p2, used, out=[])
    check("load_j3: the same file labelled with the registered judge, revision and dtype: no flag, complete",
          j2["flags"] == [] and j2["status"] == "complete" and j2["judge"] == E.J3["judge"])
    holed = json.loads(json.dumps(rel))
    first = sorted(holed["scores"])[0]
    holed["scores"][first]["7"][3] = None
    p3 = d / "holed.json"
    p3.write_text(json.dumps(holed), encoding="ascii")
    j3 = E.load_j3(p3, used, out=[])
    check("load_j3: one unscored completion makes the C6 rows provisional", j3["status"] == "provisional" and
          j3["coverage"]["steps_fully_scored"] == 799)
    ctx.cpu_result = res
    # validation-gate failure: rc 2, the target never loads
    d2 = case(ctx, "cpu_valfail", sub1(src, "VAL_TOL_STEP = 0.02", "VAL_TOL_STEP = -1.0"))
    r2 = run_case(ctx, d2)
    res2 = r2["result"] or {}
    check("cpu: a failed validation gate -> exit 2, result 'validation gate failed', target never loaded, nothing "
          "scored, last stdout line 'J3 validation_failed 0/6400'",
          r2["rc"] == 2 and res2.get("aborted") == "validation gate failed" and res2.get("judge_facts") is None and
          res2.get("n_scored") == 0 and r2["stdout"].splitlines()[-1] == "J3 validation_failed 0/6400" and
          not no_scores(r2["stdout"] + r2["stderr"]), (r2["rc"], r2["stdout"], r2["stderr"][-800:]))
    # an abort mid-scoring (as the instance timeout's SIGTERM would cause): what is scored is written, marked aborted
    d3 = case(ctx, "cpu_abort", sub1(opened, "            if tick:\n                tick()\n",
                                     "            if tick:\n                tick()\n                if state[\"batches\"] >= 3:"
                                     "\n                    raise SystemExit(143)\n"))
    r3 = run_case(ctx, d3)
    res3 = r3["result"] or {}
    check("cpu: an abort in the target stage -> exit 143, result.json marked aborted with the scores so far, "
          "complete false, last stdout line 'J3 aborted <n>/6400'",
          r3["rc"] == 143 and "target stage stopped early" in str(res3.get("aborted")) and
          0 < res3.get("n_scored", 0) < 6400 and res3.get("complete") is False and
          re.fullmatch(r"J3 aborted \d+/6400", r3["stdout"].splitlines()[-1]) and
          not no_scores(r3["stdout"] + r3["stderr"]), (r3["rc"], r3["stdout"][-300:], r3["stderr"][-800:]))
    # an abort during the validation gate: the result says so and nothing is scored
    d4 = case(ctx, "cpu_abort_val", sub1(opened, "    free(sc)\n    del sc\n",
                                         "    raise SystemExit(143)\n    free(sc)\n    del sc\n"))
    r4 = run_case(ctx, d4)
    res4 = r4["result"] or {}
    check("cpu: an abort in the validation stage -> exit 143, result.json 'validation stage stopped early', nothing "
          "scored, last stdout line 'J3 aborted 0/6400'",
          r4["rc"] == 143 and "validation stage stopped early" in str(res4.get("aborted")) and
          res4.get("n_scored") == 0 and r4["stdout"].splitlines()[-1] == "J3 aborted 0/6400" and
          not no_scores(r4["stdout"] + r4["stderr"]), (r4["rc"], r4["stdout"][-300:], r4["stderr"][-800:]))
    test_cpu_snapshot_stops(ctx, opened)


def test_cpu_snapshot_stops(ctx, opened):
    """Finding R8, the job half: as the training job does, the J3 job stops before scoring with a judge whose Hub
    cache holds anything but exactly its pinned revision. The cache is left alone: hub_cache's answer is patched to
    show a second snapshot directory, as a default-branch or other-revision read would leave, for the validation
    judge (its first call) or for the target judge only (its second call)."""
    ret = '    return {"snapshots": snaps, "refs": refs}\n'
    other = "f" * 40
    both = sub1(opened, ret, '    return {"snapshots": sorted(snaps + ["%s"]), "refs": refs}\n' % other)
    second = sub1(sub1(opened, "def hub_cache(model_id):\n", "_HC = []\n\n\ndef hub_cache(model_id):\n"
                       "    _HC.append(model_id)\n"),
                  ret, '    return {"snapshots": sorted(snaps + (["%s"] if len(_HC) > 1 else [])), "refs": refs}\n'
                  % other)
    for name, src, stage, phases in (
            ("validation", both, "validation", ["start", "data", "aborted"]),
            ("target", second, "target", ["start", "data", "validated", "aborted"])):
        d = case(ctx, "cpu_snap_" + name, src)
        r = run_case(ctx, d)
        res = r["result"] or {}
        lines = r["stdout"].splitlines()
        check("cpu: a %s judge whose Hub cache holds a second snapshot -> exit 1 'E1B J3 STOP: the Hub snapshot(s) "
              "the %s judge ...', nothing scored, result.json marked aborted in the %s stage, no chunk line" % (
                  name, name, stage),
              r["rc"] == 1 and ("E1B J3 STOP: the Hub snapshot(s) the %s judge" % name) in r["stderr"] and
              "not exactly its pinned revision" in r["stderr"] and res.get("n_scored") == 0 and
              ("%s stage stopped early" % stage) in str(res.get("aborted")) and res.get("complete") is False and
              [x.split()[1] for x in lines] == phases and lines[-1] == "J3 aborted 0/6400" and
              not any(v for st_ in (res.get("scores") or {}).values() for v in st_.values()) and
              not no_scores(r["stdout"] + r["stderr"]), (r["rc"], lines, r["stderr"][-1200:]))


# ==========================================================================================
# launcher (scratch clone, stand-in analysis, fake aws)
class L3:
    def __init__(self, ctx):
        self.ctx = ctx
        self.clone, self.F = make_clone(ctx, "clone_launcher")
        self.state = ctx.root / "aws_state"
        rmtree(self.state)
        for d in (self.state, self.state / "bin_cli", self.state / "bin_sh"):
            d.mkdir(parents=True)
        fa = HERE / "fake_aws.py"
        if os.name == "nt":
            (self.state / "bin_cli" / "aws.cmd").write_text('@"%s" "%s" %%*\r\n' % (PY, fa))
        else:
            self.sh(self.state / "bin_cli" / "aws", 'exec "%s" "%s" "$@"\n' % (PY, fa))
        self.sh(self.state / "bin_sh" / "aws", 'exec "%s" "%s" "$@"\n' % (posix(PY), posix(fa)))
        self.sh(self.state / "bin_sh" / "nvidia-smi", 'echo "NVIDIA L40S (fake), 000.00, 46068 MiB"\n')
        self.sh(self.state / "bin_sh" / "shutdown", 'echo "$*" >> "$FAKE_AWS_STATE/shutdown.log"\n')
        self.sh(self.state / "bin_sh" / "python3", (
            'if [ "$1" = "-m" ] && [ "$2" = "pip" ]; then echo "fake pip: $*"; exit 0; fi\n'
            'if [ "$1" = "-m" ] && [ "$2" = "uv" ]; then\n'
            '  if [ "$3" = "--version" ]; then echo "uv 0.12.23 (fake)"; exit 0; fi\n'
            '  echo "fake uv: $*" >&2\n'
            '  exec "%s" "%s"\n'
            'fi\n'
            'exec "%s" "$@"\n') % (posix(PY), posix(HERE / "fake_job.py"), posix(PY)))
        self.config()
        self.runs_main()
        self.freeze("good")

    @staticmethod
    def sh(path, body):
        with open(path, "w", encoding="ascii", newline="\n") as f:
            f.write("#!/bin/sh\n" + body)
        os.chmod(path, 0o755)

    def config(self, **kw):
        base = {"capacity_fail": [], "states": {}, "gone": [], "launch_times": {}, "ended": {},
                "policy_resources": ["arn:aws:s3:::%s/e1/*" % BUCKET]}
        base.update(kw)
        (self.state / "config.json").write_text(json.dumps(base, indent=1))

    def env(self):
        e = dict(os.environ)
        e["PATH"] = str(self.state / "bin_cli") + os.pathsep + e.get("PATH", "")
        e["FAKE_AWS_STATE"] = str(self.state)
        return e

    def calls(self):
        p = self.state / "calls.jsonl"
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []

    def clear(self):
        (self.state / "calls.jsonl").unlink(missing_ok=True)

    def s3p(self, key):
        return self.state / "s3" / BUCKET / key

    def launcher(self, *a):
        rc, so, se = run([PY, "scripts/launch_e1b_j3.py", *a], self.clone, self.env())
        return rc, so + se

    def runs_main(self):
        """Two synthetic result files the stand-in load_e1b uses, and the collector's manifest."""
        main = self.clone / "runs" / "e1b" / "main"
        main.mkdir(parents=True, exist_ok=True)
        for arm in ("judge", "judgegate"):
            rid = "e1b_%s_count_vowels_s300" % arm
            d = {"run_id": rid, "arm": arm, "task": "count_vowels", "seed": 300,
                 "completions": [{"step": s, "texts": ["def count_vowels(s):\n    return %d" % i for i in range(8)],
                                  "judge_small": [0.5] * 8} for s in range(400)]}
            (main / (rid + ".json")).write_text(json.dumps(d), encoding="ascii")
        lines = "".join("%s  %s\n" % (sha(f.read_bytes()), f.name) for f in sorted(main.glob("e1b_*.json")))
        (main / "MANIFEST.sha256").write_text(lines, encoding="ascii", newline="\n")

    def freeze(self, name, commit_hash=True, reader=True, approval=True, ceiling=None, manifest="good",
               receipt="good"):
        """A branch on F holding a freeze record. reader and approval: True for a filled cell, False to leave it as
        the draft has it, or the cell's text."""
        c = self.clone
        git(c, "checkout", "-q", "-B", name, self.F)
        p = c / "experiments/e1b_preregistration.md"
        t = p.read_text(encoding="utf-8")
        subs = []
        if commit_hash:
            subs.append((r"^(\| Commit hash \| )none yet( \|)$", self.F))
        if reader:
            subs.append((r"^(\| Human reader before the freeze[^\n]*\| )none yet( \|)$",
                         reader if isinstance(reader, str) else "Test Reader (scratch clone), 2026-10-25, sections 1-12"))
        if approval:
            subs.append((r"^(\| J3 scoring [^\n]*\| )pending approval( \|)$",
                         approval if isinstance(approval, str) else '"go" (test approval, 2026-10-25)'))
        if ceiling:
            subs.append((r"^(\| J3 scoring [^|]*\|[^|]*\| )8( \|)", ceiling))
        for pat, cell in subs:
            t, n = re.subn(pat, lambda m, cell=cell: m.group(1) + cell + m.group(2), t, flags=re.M)
            assert n == 1, pat
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(t)
        files = re.search(r'FILES="([^"]*)"', t).group(1).split()
        if manifest:
            lines = ["%s  %s\n" % (sha(subprocess.run(["git", "cat-file", "blob", "%s:%s" % (self.F, f)], cwd=str(c),
                                                     capture_output=True).stdout), f) for f in files]
            if manifest == "tampered":
                lines[0] = ("0" if lines[0][0] != "0" else "1") + lines[0][1:]
            man = "".join(lines).encode("ascii")
            (c / "experiments/e1b_freeze_manifest.txt").write_bytes(man)
            if receipt:
                dg = hashlib.sha256(man if receipt == "good" else b"another file").digest()
                url = b"https://a.pool.opentimestamps.org"          # a pending calendar attestation
                payload = bytes([len(url)]) + url
                (c / "experiments/e1b_freeze_manifest.txt.ots").write_bytes(
                    OTS_MAGIC + b"\x01\x08" + dg + b"\xf0\x10" + os.urandom(16) + b"\x08\x00" +
                    bytes.fromhex("83dfe30d2ef90c8e") + bytes([len(payload)]) + payload)
        git(c, "add", "-A", "experiments")
        git(c, "commit", "-q", "-m", "scratch: freeze record (%s)" % name)

    def use(self, name):
        git(self.clone, "checkout", "-q", name)

    def j3dir(self):
        return self.clone / "runs" / "e1b" / "j3"

    def reset_local(self):
        rmtree(self.j3dir())
        for p in (self.clone / "runs/e1b/instance_hours.json", self.clone / "runs/e1b/stub_selection.json",
                  self.clone / "experiments/e1b_primary_results.json", self.clone / "runs/e1b/stub_calls.txt"):
            p.unlink(missing_ok=True)


def no_spend(calls):
    for c in calls:
        if c[:2] == ["ec2", "run-instances"] or (c[:2] == ["s3", "cp"] and len(c) > 3 and c[3].startswith("s3://")):
            return False
    return True


def refusal(L, name, setup, want, *args):
    L.use("good")
    L.reset_local()
    L.config()
    L.clear()
    undo = setup() if setup else None
    rc, out = L.launcher("--commit=" + L.F, "--submit", *args)
    ok = rc != 0 and want in out and no_spend(L.calls())
    check("launcher refuses: %s" % name, ok, out[-1500:])
    if undo:
        undo()


def test_launcher(ctx):
    L = L3(ctx)
    F = L.F
    L.use("good")
    L.clear()
    rc, out = L.launcher()
    check("dry run at HEAD: exit 0, DRY RUN, data built, baked; HEAD (the freeze-record commit) is not a registered "
          "launch commit; no AWS call", rc == 0 and "DRY RUN" in out and "data: 6400 completions of 2 used runs" in out
          and "baked from" in out and "--commit is required with --submit" in out and
          "is neither section 13's freeze commit nor a section 14 commit" in out and not L.calls(), out[-2000:])
    (L.clone / "runs/e1b/stub_calls.txt").unlink(missing_ok=True)
    rc, out = L.launcher("--commit=" + F, "--show-userdata")
    calls = (L.clone / "runs/e1b/stub_calls.txt").read_text() if (L.clone / "runs/e1b/stub_calls.txt").exists() else ""
    check("dry run at F: no refusal; user data shown with the 480-minute backstop; the pins verified once",
          rc == 0 and "would REFUSE" not in out and "shutdown -h +480" in out and "baked from %s" % F[:12] in out and
          calls == "verify_pins_b\n", ([x for x in out.splitlines() if "REFUSE" in x], calls))
    check("dry run at F: the job's import closure is the 5 package files",
          "flightrecorder/__init__.py, flightrecorder/repro/__init__.py, flightrecorder/repro/reward_codeset.py, "
          "flightrecorder/repro/reward_judge.py, flightrecorder/repro/reward_testhack.py" in out, out)
    check("dry run prints no score-like number", not no_scores(out), no_scores(out))

    # refusals
    L.use("good")
    L.clear()
    rc, out = L.launcher("--submit")
    check("launcher refuses: --submit without --commit", rc != 0 and "--commit is required" in out and
          no_spend(L.calls()), out[-800:])
    L.use("good")
    git(L.clone, "commit", "-q", "--allow-empty", "-m", "scratch: an unregistered commit")
    G = git(L.clone, "rev-parse", "HEAD")
    rc, out = L.launcher("--commit=" + G, "--submit")
    check("launcher refuses: a commit that is neither the freeze commit nor in section 14",
          rc != 0 and "is neither section 13's freeze commit nor a section 14 commit" in out and no_spend(L.calls()),
          out[-800:])
    for name, kw, want in (
            ("section 13 commit hash empty", {"commit_hash": False}, "names no commit"),
            ("human reader missing", {"reader": False}, "'Human reader before the freeze' says 'none yet' at HEAD"),
            ("J3 approval pending", {"approval": False}, "'Approved by Aarav' says 'pending' at HEAD"),
            ("J3 ceiling differs from the launcher's", {"ceiling": "9"}, "ceilings (9.0 GPU-h"),
            ("no freeze manifest", {"manifest": None}, "e1b_freeze_manifest.txt is not committed at HEAD"),
            ("tampered manifest line", {"manifest": "tampered"}, "differs from its manifest line at the freeze commit"),
            ("no receipt", {"receipt": None}, "no timestamp receipt"),
            ("a receipt of another file", {"receipt": "bad"}, "does not stamp the manifest")):
        L.freeze("b_" + name.replace(" ", "_"), **kw)
        L.reset_local()
        L.clear()
        rc, out = L.launcher("--commit=" + F, "--submit")
        check("launcher refuses: %s" % name, rc != 0 and want in out and no_spend(L.calls()), out[-1200:])
    test_cells(L)

    def edit_template():
        p = L.clone / "scripts/e1b_j3_job.py"
        keep = p.read_bytes()
        p.write_bytes(keep + b"# edited\n")
        return lambda: p.write_bytes(keep)
    refusal(L, "the J3 template differs from the commit", edit_template, "scripts/e1b_j3_job.py differs from its blob")

    def edit_judge():
        p = L.clone / "flightrecorder/repro/reward_judge.py"
        keep = p.read_bytes()
        p.write_bytes(keep + b"# edited\n")
        return lambda: p.write_bytes(keep)
    refusal(L, "a job import-closure file differs from the commit", edit_judge,
            "flightrecorder/repro/reward_judge.py differs from its blob")

    def need_repl():
        (L.clone / "runs/e1b/stub_selection.json").write_text('{"replacements_needed": 1}')
    refusal(L, "a replacement is still needed (no registered results yet)", need_repl, "the used runs are not final")

    def results_mismatch():
        (L.clone / "experiments/e1b_primary_results.json").write_text(json.dumps(
            {"used": {"hacking": ["hacking/count_vowels/s300"], "twin": []}, "data_manifest": {"sha256": "0" * 64}}))
    refusal(L, "the used runs differ from the registered results file's", results_mismatch,
            "the used runs differ from experiments/e1b_primary_results.json")

    def no_manifest():
        p = L.clone / "runs/e1b/main/MANIFEST.sha256"
        keep = p.read_bytes()
        p.unlink()
        return lambda: p.write_bytes(keep)
    refusal(L, "no collector data manifest", no_manifest, "data build refused")

    def collected():
        L.j3dir().mkdir(parents=True, exist_ok=True)
        (L.j3dir() / "qwen25_7b.json").write_text("{}")
    refusal(L, "a J3 file already collected", collected, "is already collected")

    def lock():
        L.j3dir().mkdir(parents=True, exist_ok=True)
        (L.j3dir() / "launch.lock").write_text("pid 1\n")
    L.use("good")
    L.reset_local()
    lock()
    L.clear()
    rc, out = L.launcher("--commit=" + F, "--submit")
    check("launcher refuses: another launcher holds runs/e1b/j3/launch.lock (nothing launched)",
          rc != 0 and "another J3 launcher is running" in out and not any(c[:2] == ["ec2", "run-instances"]
                                                                          for c in L.calls()), out[-800:])
    L.reset_local()
    L.config(policy_resources=["arn:aws:s3:::%s/e1b/*" % BUCKET])
    L.clear()
    rc, out = L.launcher("--commit=" + F, "--submit")
    check("launcher refuses: the instance role cannot reach e1/e1b/j3/", rc != 0 and "does not allow s3:GetObject" in out
          and no_spend(L.calls()), out[-800:])
    L.config()
    L.s3p("e1/e1b/j3/e1b_j3_qwen25_7b_a1/status.json").parent.mkdir(parents=True, exist_ok=True)
    L.s3p("e1/e1b/j3/e1b_j3_qwen25_7b_a1/status.json").write_text("{}")
    L.clear()
    rc, out = L.launcher("--commit=" + F, "--submit")
    check("launcher refuses: something is already stored under the run's S3 prefix", rc != 0 and
          "already holds objects" in out and no_spend(L.calls()), out[-800:])
    rmtree(L.state / "s3")

    # a successful submit
    L.reset_local()
    L.config()
    L.clear()
    rc, out = L.launcher("--commit=" + F, "--submit")
    recs = json.loads((L.j3dir() / "launches.json").read_text()) if (L.j3dir() / "launches.json").exists() else []
    check("submit: launched one instance, one launch record", rc == 0 and "LAUNCHED e1b_j3_qwen25_7b_a1" in out and
          len(recs) == 1, out[-2000:])
    if not recs:
        return
    r = recs[0]
    J3m = load("launch_e1b_j3_clone", L.clone / "scripts/launch_e1b_j3.py")
    check("submit: the record has exactly the launcher's keys; commit F; judge, revision, dtype registered; data sha",
          tuple(r) == J3m.RECORD_KEYS and r["git_commit"] == F and r["judge"] == "Qwen/Qwen2.5-7B-Instruct" and
          r["revision"] == "a09a35458c702b33eeacc393d103063234e8bc28" and r["dtype"] == "bfloat16" and
          r["n_completions"] == 6400 and r["run_ids"] == ["e1b_judge_count_vowels_s300",
                                                          "e1b_judgegate_count_vowels_s300"] and
          r["s3"] == "s3://%s/e1/e1b/j3/e1b_j3_qwen25_7b_a1" % BUCKET, r)
    up = sorted(c[3] for c in L.calls() if c[:2] == ["s3", "cp"] and c[3].startswith("s3://"))
    check("submit: job.py, job.py.lock and data.jsonl.gz uploaded under e1/e1b/j3/<run_id>/ and nothing else",
          up == ["s3://%s/e1/e1b/j3/e1b_j3_qwen25_7b_a1/%s" % (BUCKET, n) for n in ("data.jsonl.gz", "job.py",
                                                                                   "job.py.lock")], up)
    job_s3 = L.s3p("e1/e1b/j3/e1b_j3_qwen25_7b_a1/job.py").read_bytes()
    check("submit: the uploaded job is the recorded job_sha256", sha(job_s3) == r["job_sha256"])
    insts = json.loads((L.state / "instances.json").read_text())
    spec = insts[r["instance_id"]]["spec"]
    check("submit: on-demand g6e.xlarge, terminate on shutdown, IMDSv2, 120 GB gp3, role fr-e1-runner",
          spec["InstanceType"] == "g6e.xlarge" and "InstanceMarketOptions" not in spec and
          spec["InstanceInitiatedShutdownBehavior"] == "terminate" and spec["MetadataOptions"]["HttpTokens"] ==
          "required" and spec["BlockDeviceMappings"][0]["Ebs"]["VolumeSize"] == 120 and
          spec["IamInstanceProfile"]["Name"] == "fr-e1-runner", spec)
    ud = (L.state / "userdata" / (r["instance_id"] + ".sh")).read_text()
    first = [ln for ln in ud.splitlines() if ln and not ln.startswith("#")][:2]
    check("submit: user data schedules the 480-minute backstop before anything else; job timeout 460 min; worst "
          "case USD 14.89 <= 15", first == ["set -u", "shutdown -h +480      # backstop first: whatever fails below, "
                                                       "the instance powers off (and terminates)"] and
          "timeout --kill-after=120 460m python3 -m uv run --no-config --locked --python 3.12.15 --managed-python "
          "job.py" in ud and r["backstop_min"] == 480 and r["worst_usd"] == 14.89, (first, r["worst_usd"]))
    check("submit: the lock file is released", not (L.j3dir() / "launch.lock").exists())
    check("submit prints no score-like number", not no_scores(out), no_scores(out))
    test_records_apart_launcher(L, r)
    # a second launch now: the first instance is up, and the room left is under 4 hours
    L.clear()
    rc, out = L.launcher("--commit=" + F, "--submit")
    check("launcher refuses: an earlier J3 instance is still up", rc != 0 and "still up" in out and
          no_spend(L.calls()), out[-800:])
    L.config(states={r["instance_id"]: "terminated"}, ended={r["instance_id"]: "2026-10-25 17:00:00"},
             launch_times={r["instance_id"]: "2026-10-25T10:00:00+00:00"})
    L.clear()
    rc, out = L.launcher("--commit=" + F, "--submit")
    hrs = json.loads((L.clone / "runs/e1b/instance_hours.json").read_text())
    check("launcher refuses: after a 7-hour first instance the J3 ceiling has no 4-hour room (measured hours "
          "recorded, final)", rc != 0 and "no g6e type has 240 minutes of room" in out and
          hrs[r["instance_id"]] == {"hours": 7.0, "final": True} and no_spend(L.calls()), (out[-800:], hrs))

    # status and collect
    L.config()
    L.clear()
    rc, out = L.launcher("--status")
    reads = [c for c in L.calls() if any(x in " ".join(c) for x in ("result.json", "partial.json", "job.log"))]
    check("status: counts only, reads nothing but status.json and the instance state", rc == 0 and
          "e1b_j3_qwen25_7b_a1" in out and not reads and not no_scores(out), (out, reads))
    rc, out = L.launcher("--collect")
    check("collect refuses before a final rc while the instance runs", rc != 0 and "no final rc yet" in out and
          not (L.j3dir() / "qwen25_7b.json").exists(), out[-600:])
    pre = "e1/e1b/j3/e1b_j3_qwen25_7b_a1/"
    L.s3p(pre + "status.json").parent.mkdir(parents=True, exist_ok=True)
    L.s3p(pre + "status.json").write_text(json.dumps({"run_id": r["run_id"], "phase": "done", "n_scored": 6400,
                                                     "n_total": 6400, "rc": 0, "ts": "x"}))
    doc = synthetic_result(r)
    L.s3p(pre + "result.json").write_text(json.dumps(doc))
    L.s3p(pre + "job.log").write_text("J3 done 6400/6400\n")
    L.clear()
    rc, out = L.launcher("--status")
    check("status: shows the phase and n_scored/n_total and the rc", rc == 0 and "done" in out and "6400/6400" in out
          and not no_scores(out), out)
    rc, out = L.launcher("--collect")
    dest = L.j3dir() / "qwen25_7b.json"
    check("collect: writes runs/e1b/j3/qwen25_7b.json (COMPLETE), the logs, the status, MANIFEST.sha256; counts only",
          rc == 0 and dest.is_file() and "COMPLETE; 6400 of 6400 completions scored, 800 of 800 steps" in out and
          (L.j3dir() / "MANIFEST.sha256").read_text() == "%s  qwen25_7b.json\n" % sha(dest.read_bytes()) and
          (L.j3dir() / "logs" / "e1b_j3_qwen25_7b_a1.log").is_file() and not no_scores(out), out[-1500:])
    rc, out = L.launcher("--collect")
    check("collect: a second collect of the same bytes is fine", rc == 0, out[-600:])
    L.s3p(pre + "result.json").write_text(json.dumps(dict(doc, n_nonfinite=1)))
    rc, out = L.launcher("--collect")
    check("collect refuses to replace a collected file with different bytes", rc != 0 and "different bytes" in out,
          out[-600:])
    dest.unlink()
    L.s3p(pre + "result.json").write_text(json.dumps(dict(doc, revision="0" * 40)))
    rc, out = L.launcher("--collect")
    check("collect refuses a file whose revision is not the registered J3", rc != 0 and "revision" in out and
          not dest.exists(), out[-600:])
    L.s3p(pre + "result.json").write_text(json.dumps(dict(doc, job_file_sha256="0" * 64)))
    rc, out = L.launcher("--collect")
    check("collect refuses a file whose job_file_sha256 is not the launched job's", rc != 0 and "job_file_sha256" in out
          and not dest.exists(), out[-600:])
    test_collect_snapshots(L, doc, pre, dest)
    L.s3p(pre + "result.json").unlink()
    L.s3p(pre + "partial.json").write_text(json.dumps(dict(doc, complete=False)))
    rc, out = L.launcher("--collect")
    check("collect refuses a partial.json without --accept-partial", rc != 0 and "--accept-partial" in out and
          not dest.exists(), out[-600:])
    rc, out = L.launcher("--collect", "--accept-partial")
    check("collect --accept-partial writes it, labelled PARTIAL", rc == 0 and "PARTIAL" in out and dest.is_file(),
          out[-600:])
    if hasattr(ctx, "E") and ctx.E is not None:
        stub_runs = []
        for f in sorted((L.clone / "runs/e1b/main").glob("e1b_*.json")):
            d = json.loads(f.read_text())
            d["log"] = [{}] * 400
            stub_runs.append(d)
        j = ctx.E.load_j3(dest, stub_runs, out=[])
        check("collect + load_j3 (real e1b_primary): the collected file reads complete, with no flag",
              j["status"] == "complete" and j["flags"] == [], j and (j["status"], j["flags"]))
    # capacity fallback in a fresh launcher state: g6e.xlarge refused everywhere -> 2xlarge, smaller backstop
    L.reset_local()
    rmtree(L.state / "s3")
    (L.state / "instances.json").unlink(missing_ok=True)
    L.config(capacity_fail=[[rg, "g6e.xlarge"] for rg in ("us-east-1", "us-east-2")])
    L.clear()
    rc, out = L.launcher("--commit=" + F, "--submit")
    recs = json.loads((L.j3dir() / "launches.json").read_text()) if (L.j3dir() / "launches.json").exists() else [{}]
    check("submit: capacity fallback to g6e.2xlarge with the backstop the ceiling allows (401 min, worst USD 14.98)",
          rc == 0 and recs[0].get("instance_type") == "g6e.2xlarge" and recs[0].get("backstop_min") == 401 and
          recs[0].get("worst_usd") <= 15.0 and "capacity misses before it" in out, (out[-1200:], recs))
    # no capacity anywhere: the uploads have used the run id's prefix, so a record without an instance spends the id
    L.reset_local()
    rmtree(L.state / "s3")
    (L.state / "instances.json").unlink(missing_ok=True)
    L.config(capacity_fail=[[rg, t] for rg in ("us-east-1", "us-east-2") for t in ("g6e.xlarge", "g6e.2xlarge",
                                                                                   "g6e.4xlarge")])
    rc, out = L.launcher("--commit=" + F, "--submit")
    recs = json.loads((L.j3dir() / "launches.json").read_text()) if (L.j3dir() / "launches.json").exists() else []
    check("submit with no capacity anywhere: NOT LAUNCHED (rc 1); a record without an instance spends run id a1",
          rc == 1 and "NOT LAUNCHED e1b_j3_qwen25_7b_a1" in out and len(recs) == 1 and recs[0]["instance_id"] is None
          and "the next launch uses e1b_j3_qwen25_7b_a2" in out, (out[-1200:], recs))
    L.config()
    rc, out = L.launcher("--commit=" + F)
    check("after it: the next launch is a2 on a fresh prefix, nothing committed; --status and --collect skip it",
          rc == 0 and "E1b J3 scoring: e1b_j3_qwen25_7b_a2" in out and "committed USD 0.00" in out and
          "would REFUSE" not in out and "not launched" in L.launcher("--status")[1] and
          "no launched J3 instance" in L.launcher("--collect")[1], out[-1500:])
    test_hours_concurrent(L)
    ctx.launcher_state = L


def test_records_apart_launcher(L, r):
    """Finding R10, in the clone after a J3 launch: the J3 record is in runs/e1b/j3/launches.json only, and the main
    launcher's --status and dry run of every stage see no J3 run and write no runs/e1b/launches.json."""
    main_recs = L.clone / "runs" / "e1b" / "launches.json"
    check("records apart: the J3 launch record went to runs/e1b/j3/launches.json and nowhere else",
          json.loads((L.j3dir() / "launches.json").read_text())[-1]["run_id"] == r["run_id"] and
          not main_recs.exists())
    L.clear()
    outs = []
    for stage in ("pilot", "main", "main_replace"):
        rc, so, se = run([PY, "-B", "scripts/launch_e1b_aws.py", stage, "--status"], L.clone, L.env())
        outs.append((rc, so + se))
    check("records apart: launch_e1b_aws --status of pilot, main and main_replace lists no run (the J3 record is not "
          "a main or pilot record), makes no AWS call, writes no runs/e1b/launches.json",
          all(rc == 0 and "no launched run in stage" in o and r["run_id"] not in o and r["instance_id"] not in o
              for rc, o in outs) and not L.calls() and not main_recs.exists(), [o[-300:] for _rc, o in outs])
    rc, so, se = run([PY, "-B", "scripts/launch_e1b_aws.py", "pilot", "--commit=" + L.F], L.clone, L.env())
    check("records apart: the main launcher's pilot dry run counts no J3 instance in the Pilot ceiling",
          rc == 0 and "committed USD 0.00 / 0.0 GPU-h (0 instance(s)" in so and r["instance_id"] not in so and
          not main_recs.exists(), (so + se)[-1500:])


def test_hours_concurrent(L):
    """Finding R9: runs/e1b/instance_hours.json written by both launchers at once. A main-launcher process and a J3
    launcher process each record the measured hours of 120 instances of their own, one at a time (each call a
    read-merge-write of the shared file; the AWS reads stubbed), started together, while a third process records
    a shared instance through both. No entry may be lost, every final entry keeps its value, and no lock or
    temporary file is left behind."""
    hours = L.clone / "runs" / "e1b" / "instance_hours.json"
    hours.unlink(missing_ok=True)
    n = 120
    head = ("import sys; from datetime import datetime, timedelta, timezone; sys.path.insert(0, 'scripts')\n"
            "t0 = datetime(2026, 10, 25, tzinfo=timezone.utc)\n")
    main_w = head + ("import launch_e1b_aws as L\n"
                     "for i in range(%d):\n"
                     "    iid = 'i-main%%04d' %% i\n"
                     "    L.record_hours([{'instance_id': iid, 'submitted_utc': '2026-10-25T00:00:00Z'}],\n"
                     "                   {iid: {'state': 'terminated', 'launch': t0, 'end': t0 + timedelta(hours=1)}})\n"
                     % n)
    j3_w = head + ("import launch_e1b_j3 as J3\n"
                   "J3.describe = lambda rec: ('terminated', t0, t0 + timedelta(hours=2))\n"
                   "for i in range(%d):\n"
                   "    J3.record_hours([{'instance_id': 'i-j3%%04d' %% i, 'submitted_utc': '2026-10-25T00:00:00Z',\n"
                   "                      'backstop_min': 480}])\n" % n)
    shared_w = head + ("import launch_e1b_aws as L, launch_e1b_j3 as J3\n"
                       "J3.describe = lambda rec: ('running', t0, None)\n"
                       "for i in range(%d):\n"
                       "    L.record_hours([{'instance_id': 'i-shared', 'submitted_utc': '2026-10-25T00:00:00Z'}],\n"
                       "                   {'i-shared': {'state': 'running', 'launch': t0, 'end': None}})\n"
                       "    J3.record_hours([{'instance_id': 'i-shared', 'submitted_utc': '2026-10-25T00:00:00Z'}])\n"
                       % (n // 4))
    procs = [subprocess.Popen([PY, "-B", "-c", code], cwd=str(L.clone), env=L.env(), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT) for code in (main_w, j3_w, shared_w)]
    outs = [p.communicate(timeout=900)[0].decode("utf-8", "replace") for p in procs]
    got = json.loads(hours.read_text()) if hours.is_file() else {}
    want = dict({"i-main%04d" % i: {"hours": 1.0, "final": True} for i in range(n)},
                **{"i-j3%04d" % i: {"hours": 2.0, "final": True} for i in range(n)})
    lost = sorted(set(want) - set(got))
    check("instance hours: a main launcher, a J3 launcher and a third writer at once lose no entry (%d of %d kept), "
          "each final value as written, the shared running instance recorded, no lock or temp file left" % (
              len(set(want) & set(got)), len(want)),
          all(p.returncode == 0 for p in procs) and not lost and all(got[k] == v for k, v in want.items()) and
          set(got) == set(want) | {"i-shared"} and got["i-shared"]["final"] is False and
          sorted(x.name for x in hours.parent.iterdir() if x.name.startswith("instance_hours")) ==
          ["instance_hours.json"], ([p.returncode for p in procs], lost[:6], [o[-600:] for o in outs]))
    hours.unlink(missing_ok=True)


def refused_fields(out):
    """The field names of collect's 'differs ... in: a, b; nothing written' refusal, or None."""
    m = re.search(r"differs from the launch record or the registered J3 in: ([^;]+); nothing written", out)
    return m.group(1).split(", ") if m else None


def test_collect_snapshots(L, doc, pre, dest):
    """Finding R8, the collect half: a result whose judge, revision, dtype or measured Hub snapshots differ from the
    registered values is refused, and nothing is written; a failed validation gate's file (nothing scored, the
    target never loaded) is still collected."""
    other = "f" * 40
    snaps, val, facts = doc["snapshots"], doc["validation"], doc["judge_facts"]
    for name, bad, field in (
            ("a target loaded from another Hub snapshot",
             {"snapshots": dict(snaps, target={"snapshots": [other], "refs": {}})}, "snapshots.target"),
            ("a target cache holding a second snapshot",
             {"snapshots": dict(snaps, target={"snapshots": sorted([doc["revision"], other]), "refs": {}})},
             "snapshots.target"),
            ("scores with no snapshot recorded", {"snapshots": None}, "snapshots.target"),
            ("a validation judge loaded from another snapshot",
             {"validation": dict(val, hub_cache={"snapshots": [other], "refs": {}})}, "validation.hub_cache"),
            ("snapshots.validation of another revision",
             {"snapshots": dict(snaps, validation={"snapshots": [other], "refs": {}})}, "snapshots.validation"),
            ("a validation judge of another revision", {"validation": dict(val, revision=other)}, "validation.revision"),
            ("a measured dtype that is not bfloat16", {"judge_facts": dict(facts, param_dtype="float16")},
             "judge_facts.param_dtype"),
            ("a scorer run at another revision", {"judge_facts": dict(facts, revision=other)}, "judge_facts.revision"),
            ("another dtype", {"dtype": "float16"}, "dtype"),
            ("another judge", {"judge": "Qwen/Qwen2.5-1.5B-Instruct"}, "judge"),
            ("scores without a passed validation gate", {"validation": dict(val, passed=False)}, "validation.passed")):
        L.s3p(pre + "result.json").write_text(json.dumps(dict(doc, **bad)))
        rc, out = L.launcher("--collect")
        got = refused_fields(out)
        check("collect refuses %s: names %s, writes nothing" % (name, field), rc != 0 and got and field in got and
              not dest.exists() and not no_scores(out), (got, out[-600:]))
    failed = dict(doc, scores={}, n_scored=0, complete=False, aborted="validation gate failed", judge_facts=None,
                  snapshots=None, validation=dict(val, passed=False))
    L.s3p(pre + "result.json").write_text(json.dumps(failed))
    rc, out = L.launcher("--collect")
    check("collect takes a failed validation gate's file (nothing scored, target never loaded, the validation judge's "
          "snapshot checked) as ABORTED", rc == 0 and "ABORTED" in out and dest.is_file(), out[-600:])
    dest.unlink(missing_ok=True)
    L.s3p(pre + "result.json").write_text(json.dumps(dict(failed, validation=dict(
        val, passed=False, hub_cache={"snapshots": [other], "refs": {}}))))
    rc, out = L.launcher("--collect")
    got = refused_fields(out)
    check("collect refuses a failed gate's file whose validation judge came from another snapshot",
          rc != 0 and got == ["validation.hub_cache"] and not dest.exists(), (got, out[-600:]))


def test_cells(L):
    """Finding R7: the J3 launcher reads section 13's reader cell and section 12's J3 approval cell by the main
    launcher's rule (launch_e1b_aws.reader_problem and approval_problem), not by a copy of its own: a real date is
    required and curly quotes count. Each cell is a freeze branch on F and a dry run at F, whose only refusal must be
    that cell's (or none)."""
    q1, q2 = chr(0x201c), chr(0x201d)
    src = (L.clone / "scripts/launch_e1b_j3.py").read_text(encoding="ascii")
    check("cells: the J3 launcher keeps no rule of its own (no BAD_WORDS, DATE or _cell_ok) and calls "
          "launch_e1b_aws.reader_problem and approval_problem", not re.search(r"\b(BAD_WORDS|_cell_ok|DATE)\b", src) and
          "L.reader_problem(" in src and "L.approval_problem(" in src)
    for name, kw, want in (
            ("a reader cell with an impossible date", {"reader": "Test Reader, 2026-13-45, sections 1-12"},
             "'Human reader before the freeze' has no YYYY-MM-DD date"),
            ("a reader cell that names no section", {"reader": "Test Reader, 2026-10-25"},
             "'Human reader before the freeze' does not say which sections were read"),
            ("an approval with an impossible date", {"approval": '"go" (test approval, 2026-13-45)'},
             "'Approved by Aarav' has no YYYY-MM-DD date"),
            ("an approval whose quotes hold only a space", {"approval": '" " (test approval, 2026-10-25)'},
             "'Approved by Aarav' has no quoted words of Aarav's"),
            ("an approval that says TBD", {"approval": '"go" (2026-10-25, TBD)'}, "'Approved by Aarav' says 'TBD'"),
            ("an approval in curly quotes (accepted, as by the main launcher)",
             {"approval": q1 + "go" + q2 + " (test approval, 2026-10-25)"}, None)):
        L.freeze("cell_%d" % len(RESULTS), **kw)
        L.reset_local()
        L.clear()
        rc, out = L.launcher("--commit=" + L.F)
        refused = [ln for ln in out.splitlines() if "--submit would REFUSE" in ln]
        ok = rc == 0 and not L.calls() and (refused == [] if want is None else
                                            len(refused) == 1 and want in refused[0])
        check("cells: dry run at F, %s -> %s" % (name, "no refusal" if want is None else "refused: " + want), ok,
              (refused, out[-1200:]))
    L.use("good")


def synthetic_result(rec, **kw):
    """A J3 result as the job writes it for the launch record rec, with the registered judges' measured snapshots."""
    scores = {rid: {str(s): [0.25] * 8 for s in range(400)} for rid in rec["run_ids"]}
    val_rev = "7ae557604adf67be50417f59c2c2f167def9a775"
    doc = {"judge": rec["judge"], "revision": rec["revision"], "dtype": rec["dtype"], "data_sha256": rec["data_sha256"],
           "scores": scores, "n_scored": 6400, "n_total": 6400, "complete": True, "run_id": rec["run_id"],
           "stage": "j3", "job_file_sha256": rec["job_sha256"], "n_nonfinite": 0,
           "validation": {"model": "Qwen/Qwen2.5-0.5B-Instruct", "revision": val_rev, "dtype": "float16",
                          "passed": True, "hub_cache": {"snapshots": [val_rev], "refs": {}}},
           "judge_facts": {"param_dtype": "bfloat16", "revision": rec["revision"]},
           "snapshots": {"target": {"snapshots": [rec["revision"]], "refs": {}},
                         "validation": {"snapshots": [val_rev], "refs": {}}}}
    doc.update(kw)
    return doc


# ==========================================================================================
# the instance's user data under bash
def test_userdata(ctx):
    L = getattr(ctx, "launcher_state", None) or L3(ctx)
    J3m = load("launch_e1b_j3_ud", L.clone / "scripts/launch_e1b_j3.py")
    bash = git_bash()
    for mode, want_rc, extra in (("ok", 0, {}), ("fail3", 3, {}), ("hang", "stalled", {"stall_s": 3}),
                                 ("nojob", "no-job", {}), ("signal", "interrupted", {})):
        rid = "e1b_j3_qwen25_7b_a9"
        work = ctx.root / "ud" / mode
        rmtree(work)
        work.mkdir(parents=True)
        rmtree(L.state / "s3")
        (L.state / "shutdown.log").unlink(missing_ok=True)
        pre = "e1/e1b/j3/%s/" % rid
        if mode != "nojob":
            L.s3p(pre + "job.py").parent.mkdir(parents=True, exist_ok=True)
            L.s3p(pre + "job.py").write_text('RUN_ID = "%s"\n' % rid)
            L.s3p(pre + "job.py.lock").write_text("lock\n")
            L.s3p(pre + "data.jsonl.gz").write_text("data\n")
        ud = J3m.render(rid, 480, workdir=posix(work), tick_s=1, push_s=2, stall_s=extra.get("stall_s", 600))
        script = work / "ud.sh"
        script.write_text(ud, encoding="ascii", newline="\n")
        env = dict(os.environ, FAKE_AWS_STATE=str(L.state), FAKE_J3_MODE="hang" if mode == "signal" else mode,
                   PATH=posix(L.state / "bin_sh") + ":" + "/usr/bin:/bin")
        if os.name == "nt":
            env["PATH"] = str(L.state / "bin_sh") + os.pathsep + str(Path(bash).parent) + os.pathsep + os.environ["PATH"]
        t0 = time.time()
        if mode == "signal":         # the instance gets SIGTERM while the job runs (a stop, or the backstop's shutdown)
            outer = work / "outer.sh"
            outer.write_text('bash "%s" & P=$!\nsleep 4\nkill -TERM $P\nwait $P\n' % posix(script), encoding="ascii",
                             newline="\n")
            rc, so, se = run([bash, posix(outer)], work, env=env, timeout=300)
        else:
            rc, so, se = run([bash, posix(script)], work, env=env, timeout=300)
        st = json.loads(L.s3p(pre + "status.json").read_text()) if L.s3p(pre + "status.json").exists() else {}
        boot = (work / "boot.log").read_text() if (work / "boot.log").exists() else ""
        sd = (L.state / "shutdown.log").read_text().split("\n") if (L.state / "shutdown.log").exists() else []
        puts = [c[3].split("/")[-1] for c in L.calls() if c[:2] == ["s3", "cp"] and len(c) > 3 and
                c[3].startswith("s3://%s/%s" % (BUCKET, pre))]
        status_last = [p for p in puts if p != "boot.log"][-1:] == ["status.json"]
        ok = st.get("rc") == want_rc and sd[:1] == ["-h +480"] and "-h now" in sd and status_last
        if mode in ("ok", "fail3"):
            ok = ok and {"result.json", "partial.json", "job.log", "boot.log"} <= set(puts) and \
                st.get("phase") == "done" and st.get("n_scored") == 16
        if mode in ("hang", "signal"):
            ok = ok and "job.log" in puts and st.get("phase") == "start" and st.get("n_total") == 16
        check("user data (%s): final rc %r last, backstop scheduled first, powered off; %d s" % (
            mode, want_rc, time.time() - t0), ok, (st, sd, puts, boot[-1500:]))
        L.clear()


# ==========================================================================================
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--venv", help="python of a venv with CPython 3.12.15, the locked packages and CPU torch")
    ap.add_argument("--uv", help="uv 0.12.23, for the lock check")
    ap.add_argument("--hf-home", help="an HF_HOME whose cache may hold the tiny model")
    ap.add_argument("--only", help="comma-separated sections: data,bake,stops,cpu,launcher,userdata")
    ap.add_argument("--allow-skip", action="store_true")
    a = ap.parse_args(argv)
    ctx = Ctx(a)
    only = set(a.only.split(",")) if a.only else {"data", "bake", "stops", "cpu", "launcher", "userdata"}
    skipped = []
    t0 = time.time()
    test_data(ctx)                     # every later section uses its data
    if only & {"bake", "stops", "cpu"}:
        test_bake(ctx)
    for name, fn in (("stops", test_stops), ("cpu", test_cpu)):
        if name in only:
            if ctx.venv:
                fn(ctx)
            else:
                skipped.append(name)
                print("SKIP  %s (no --venv)" % name)
    if "launcher" in only:
        test_launcher(ctx)
    if "userdata" in only:
        test_userdata(ctx)
    n_ok = sum(ok for _, ok in RESULTS)
    print("\n%d of %d checks passed%s in %.0f s" % (n_ok, len(RESULTS), "; skipped: %s" % ", ".join(skipped)
                                                    if skipped else "", time.time() - t0))
    return 0 if n_ok == len(RESULTS) and (a.allow_skip or not skipped) else 1


if __name__ == "__main__":
    raise SystemExit(main())
