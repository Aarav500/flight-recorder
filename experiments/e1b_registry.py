"""E1b registry: the registered launch commits and the spend rule, shared by the launcher and the analysis.

Both scripts/launch_e1b_aws.py (before a launch) and experiments/e1b_primary.py (rule (e), the replacement
allowance) need the same answers to two questions, so both import this module instead of keeping two copies:

1. Which commits may jobs be built from? Spec 7.1: "the freeze commit or a later commit logged in section 14
   before launch". registered_commits(prereg_text) returns the hex commit ids in the pre-registration's section 13
   "Commit hash" cell and in the "Commit" column of section 14's deviation log, and nothing else.
2. How much has a budget group committed? Pre-registration section 12 (stage ceilings) and section 11 (Budget).
   The groups are 'pilot', 'primary' (main and main_replace) and 'j3' (section 12's "J3 scoring" row, launched by
   scripts/launch_e1b_j3.py). committed(records, group, hours) prices each launched instance at its type's
   on-demand list price for the larger of its planning hours (PLAN_H_PER_RECORD[group] per launch record: runs x
   PLAN_H, the figure the ceilings were built from, or J3_PLAN_H for the one J3 instance) and its measured hours
   (launch to termination, or to now while it runs, as the launchers record them in runs/e1b/instance_hours.json).
   room(...) is the ceiling less that. Neither reads an outcome value.
3. How are measured hours written? Both launchers write runs/e1b/instance_hours.json, and several of their commands
   (--status, --collect, --submit, the autorun's polls) can run at once. update_instance_hours() is the one writer:
   an exclusive lock file around the read-merge-write, the file read again under the lock, and an atomic replace.

Standard library only: the launcher runs from a public clone, where paper/ does not exist.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "experiments" / "e1b_preregistration.md"
INSTANCE_HOURS = REPO / "runs" / "e1b" / "instance_hours.json"

# g6e only (section 12: "AWS g6e"), every size with one L40S 48 GB, cheapest first. On-demand list USD/h from the AWS
# pricing API, 2026-10-07 (E1's FALLBACK). A spot instance is priced at these too: its price is capped by them.
TYPES = (("g6e.xlarge", 1.861), ("g6e.2xlarge", 2.242), ("g6e.4xlarge", 3.004))
PRICE = dict(TYPES)
# Planning GPU-hours of one run, the figure the ceilings were built from (section 4: "About 1.4 GPU-hours per run";
# experiments/e1b_power.py GPU_H_PER_RUN = 125/90). At g6e.xlarge's price it is section 4's USD 2.58 per run.
PLAN_H = 125 / 90
# Planning GPU-hours of the J3 instance, which scores every used run in one job (the figure scripts/launch_e1b_j3.py
# used before it moved here): section 12's "J3 scoring" row, "estimate USD 4.4 at the E1 rescoring's measured USD 2.2
# per 153,600 completions", at g6e.xlarge's list price (USD 1.861/h), so about 2.36 hours.
J3_PLAN_H = 4.4 / 1.861
# section 12 rows: (the row's first cell, or for J3 its start; ceiling GPU-hours; ceiling USD)
CEILINGS = {"pilot": ("Pilot", 15.0, 30.0), "primary": ("Primary", 153.0, 285.0), "j3": ("J3 scoring", 8.0, 15.0)}
BUDGET_GROUP = {"pilot": "pilot", "main": "primary", "main_replace": "primary", "j3": "j3"}     # stage -> group
# Planning hours per launch record: a pilot or main record is one run (PLAN_H); a J3 record is one instance.
PLAN_H_PER_RECORD = {"pilot": PLAN_H, "primary": PLAN_H, "j3": J3_PLAN_H}
CAP_USD = 330.0                                                                    # section 12 total
CAP_H = 176.0                                                                      # section 12 total
HOURS_LOCK_WAIT_S = 120.0     # how long update_instance_hours waits for another writer's lock file

_HEX = re.compile(r"\b[0-9a-f]{7,40}\b")


def section(text, num):
    """Lines of '## <num>.' up to the next '## ' heading, or None."""
    lines = text.split("\n")
    start = next((i for i, ln in enumerate(lines) if ln.startswith("## %d." % num)), None)
    if start is None:
        return None
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    return lines[start + 1:end]


def table_rows(lines):
    """Markdown table rows as lists of stripped cells, separator rows dropped."""
    out = []
    for ln in lines or []:
        if ln.strip().startswith("|"):
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if not all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                out.append(cells)
    return out


def freeze_commit(text):
    """The commit id in section 13's 'Commit hash' cell, or None while it reads 'none yet'."""
    for r in table_rows(section(text, 13)):
        if r and r[0].startswith("Commit hash") and len(r) > 1:
            m = _HEX.search(r[1])
            return m.group(0) if m else None
    return None


def deviation_commits(text):
    """Commit ids in the 'Commit' column of section 14's deviation log."""
    rows = table_rows(section(text, 14))
    if not rows:
        return []
    head = rows[0]
    col = next((i for i, c in enumerate(head) if c == "Commit"), None)
    if col is None:
        return []
    return [m.group(0) for r in rows[1:] if col < len(r) for m in _HEX.finditer(r[col])]


def registered_commits(text):
    """The freeze commit and every section 14 commit, as written (7 to 40 hex characters)."""
    fc = freeze_commit(text)
    return ([fc] if fc else []) + deviation_commits(text)


def commit_registered(commit, tokens):
    """True when the full commit id starts with one of the registered ids (an abbreviated id counts)."""
    return isinstance(commit, str) and bool(re.fullmatch(r"[0-9a-f]{40}", commit)) and \
        any(commit.startswith(t) for t in tokens)


def load_instance_hours(path=INSTANCE_HOURS):
    """{instance_id: {"hours": float, "final": bool}} as the launchers record it; {} when absent. A Windows sharing
    error (a writer replacing the file at that moment) is retried for up to a second."""
    for _ in range(20):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except PermissionError:
            time.sleep(0.05)
        except (OSError, ValueError):
            return {}
    return {}


def _read_hours_strict(path):
    """The hours file as the writer reads it under the lock: {} when absent. A file that cannot be read, or is not a
    JSON object, stops the writer (SystemExit) instead of being replaced by a file without its entries."""
    for _ in range(40):
        try:
            raw = Path(path).read_bytes()
        except FileNotFoundError:
            return {}
        except PermissionError:          # Windows: a reader outside the lock has it open for a moment
            time.sleep(0.05)
            continue
        try:
            d = json.loads(raw.decode("utf-8"))
        except ValueError:
            d = None
        if not isinstance(d, dict):
            raise SystemExit("REFUSE: %s is not a JSON object of instance hours; nothing written (restore it by hand)"
                             % Path(path).name)
        return d
    raise SystemExit("REFUSE: could not read %s; nothing written" % Path(path).name)


def _merge_hours(prev, new):
    """prev with new merged in: no entry of prev is dropped, a final entry is never changed, a final measurement
    replaces a non-final one, and a non-final one never lowers the hours already recorded (a running instance's
    measured hours only grow, so an older measurement written later cannot undo a newer one)."""
    out = dict(prev)
    for iid, v in new.items():
        if v == prev.get(iid):           # unchanged (a writer passes an entry back as it read it)
            continue
        h = v.get("hours") if isinstance(v, dict) else None
        if isinstance(h, bool) or not isinstance(h, (int, float)):
            raise ValueError("measured hours of %s are not {'hours': number, 'final': bool}: %r" % (iid, v))
        v = {"hours": float(h), "final": bool(v.get("final"))}
        old = prev.get(iid)
        if isinstance(old, dict) and old.get("final"):
            continue
        if isinstance(old, dict) and not v["final"]:
            v["hours"] = max(v["hours"], float(old.get("hours") or 0.0))
        out[iid] = v
    return out


def update_instance_hours(update, path=INSTANCE_HOURS, wait_s=HOURS_LOCK_WAIT_S):
    """Merge measured hours into runs/e1b/instance_hours.json, safely when several launcher commands write at once.
    update is {instance_id: {"hours", "final"}}, or a function of the file's current content that returns it (called
    under the lock, so a measurement that depends on the last one sees the latest file). Under an exclusive lock file
    <path>.lock (created with O_EXCL, holding the writer's pid and time, removed at the end) the file is read again,
    merged by _merge_hours and, when that changes it, written to a temporary file in the same folder that os.replace
    moves over the old one, so a reader sees the old file or the new one, never part of one. A lock file still there
    after wait_s stops the writer (a writer was killed while it held it: delete it by hand). -> the whole file."""
    path = Path(path)
    lock = path.with_name(path.name + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    while True:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except (FileExistsError, PermissionError):   # PermissionError: Windows, while its holder deletes it
            if time.monotonic() - t0 > wait_s:
                try:
                    held = lock.read_text(encoding="ascii", errors="replace").strip()
                except OSError:
                    held = "unreadable"
                raise SystemExit("REFUSE: %s exists (%s) and was not released within %d s: another launcher is "
                                 "writing the instance hours, or one was killed while it did. If no launcher is "
                                 "running, delete the file and run this again; nothing written"
                                 % (lock.name, held or "empty", wait_s))
            time.sleep(0.02)
    try:
        os.write(fd, ("pid %d, since %s\n" % (os.getpid(), datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"))).encode("ascii"))
        os.close(fd)
        fd = None
        prev = _read_hours_strict(path)
        out = _merge_hours(prev, update(dict(prev)) if callable(update) else update)
        if out != prev:
            tfd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
            with os.fdopen(tfd, "w", encoding="ascii", newline="\n") as f:
                f.write(json.dumps({k: out[k] for k in sorted(out)}, indent=1) + "\n")
            for i in range(100):
                try:
                    os.replace(tmp, path)
                    break
                except PermissionError:          # Windows: a reader outside the lock has the file open
                    if i == 99:
                        os.unlink(tmp)
                        raise
                    time.sleep(0.05)
        return out
    finally:
        if fd is not None:
            os.close(fd)
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def committed(records, group, hours=None):
    """(USD, GPU-hours) committed in a budget group ('pilot', 'primary' or 'j3'). One entry per launched instance:
    the larger of its records x PLAN_H_PER_RECORD[group] (runs x PLAN_H; J3_PLAN_H for a J3 instance) and its
    measured hours, at its type's list price (an unknown type at the dearest)."""
    if group not in CEILINGS:
        raise ValueError("unknown budget group %r (one of %s)" % (group, ", ".join(CEILINGS)))
    hours = load_instance_hours() if hours is None else hours
    per = {}
    for r in records:
        if r.get("instance_id") and BUDGET_GROUP.get(r.get("stage")) == group:
            per.setdefault(r["instance_id"], []).append(r)
    usd = gpu_h = 0.0
    for iid, v in per.items():
        h = max(len(v) * PLAN_H_PER_RECORD[group], float((hours.get(iid) or {}).get("hours") or 0.0))
        usd += h * PRICE.get(v[0].get("instance_type"), max(PRICE.values()))
        gpu_h += h
    return usd, gpu_h


def room(records, group, hours=None):
    """(USD, GPU-hours) left under the group's section 12 ceiling."""
    usd, gpu_h = committed(records, group, hours)
    _, ceil_h, ceil_usd = CEILINGS[group]
    return ceil_usd - usd, ceil_h - gpu_h


def _selftest():
    text = "\n".join([
        "## 13. Freeze record", "", "| Field | Value |", "|---|---|",
        "| Commit hash | 0123456789abcdef0123456789abcdef01234567 |", "| Commit time | none yet |",
        "## 14. Deviation log", "",
        "| Date | Section changed | What changed | Why | Before or after a look at the data? | Effect on the claims | Commit |",
        "|---|---|---|---|---|---|---|",
        "| 2026-11-01 | 9 | logging fix 89abcdef | P0 | before | none | fedcba9 |",
    ])
    assert freeze_commit(text) == "0123456789abcdef0123456789abcdef01234567"
    assert deviation_commits(text) == ["fedcba9"], deviation_commits(text)   # the hex in 'What changed' is ignored
    toks = registered_commits(text)
    assert commit_registered("0123456789abcdef0123456789abcdef01234567", toks)
    assert commit_registered("fedcba9" + "0" * 33, toks)
    assert not commit_registered("89abcdef" + "0" * 32, toks)
    assert not commit_registered("fedcba9", toks)                             # a full id is required
    assert freeze_commit(text.replace("0123456789abcdef0123456789abcdef01234567", "none yet")) is None
    recs = [{"stage": "main", "instance_id": "i-1", "instance_type": "g6e.xlarge"}] * 12 + \
           [{"stage": "pilot", "instance_id": "i-2", "instance_type": "g6e.2xlarge"}] * 6
    usd, gh = committed(recs, "primary", hours={})
    assert abs(usd - 12 * PLAN_H * 1.861) < 1e-9 and abs(gh - 12 * PLAN_H) < 1e-9
    usd2, _ = committed(recs, "primary", hours={"i-1": {"hours": 20.0, "final": True}})
    assert abs(usd2 - 20.0 * 1.861) < 1e-9                                    # measured hours above plan count
    usd3, _ = committed(recs, "primary", hours={"i-1": {"hours": 1.0, "final": True}})
    assert abs(usd3 - usd) < 1e-9                                             # never below the plan
    assert abs(committed(recs, "pilot", hours={})[0] - 6 * PLAN_H * 2.242) < 1e-9
    _selftest_j3_group(recs)
    _selftest_hours_writer()
    print("SELFTEST OK")


def _selftest_j3_group(recs):
    """The 'j3' group: section 12's row, J3_PLAN_H per instance, kept apart from the other groups."""
    assert CEILINGS["j3"] == ("J3 scoring", 8.0, 15.0) and BUDGET_GROUP["j3"] == "j3"
    assert set(BUDGET_GROUP.values()) == set(CEILINGS) == set(PLAN_H_PER_RECORD)
    assert abs(sum(c[2] for c in CEILINGS.values()) - CAP_USD) < 1e-9          # 30 + 285 + 15 = 330
    assert abs(sum(c[1] for c in CEILINGS.values()) - CAP_H) < 1e-9            # 15 + 153 + 8 = 176
    assert abs(J3_PLAN_H * PRICE["g6e.xlarge"] - 4.4) < 1e-9                   # section 12's USD 4.4 estimate
    j3 = [{"stage": "j3", "instance_id": "i-3", "instance_type": "g6e.xlarge"},
          {"stage": "j3", "instance_id": None, "instance_type": None}]          # a launch that found no capacity
    both = recs + j3
    usd, gh = committed(both, "j3", hours={})
    assert abs(usd - 4.4) < 1e-9 and abs(gh - J3_PLAN_H) < 1e-9, (usd, gh)
    usd, gh = committed(both, "j3", hours={"i-3": {"hours": 7.0, "final": True}})
    assert abs(usd - 7.0 * 1.861) < 1e-9 and abs(gh - 7.0) < 1e-9
    assert abs(committed(both, "j3", hours={"i-3": {"hours": 1.0, "final": True}})[0] - 4.4) < 1e-9
    assert committed(both, "primary", hours={}) == committed(recs, "primary", hours={})   # J3 is its own group
    assert committed(both, "pilot", hours={}) == committed(recs, "pilot", hours={})
    assert committed(recs, "j3", hours={}) == (0.0, 0.0)
    r_usd, r_h = room(both, "j3", hours={})
    assert abs(r_usd - (15.0 - 4.4)) < 1e-9 and abs(r_h - (8.0 - J3_PLAN_H)) < 1e-9
    try:
        committed(both, "J3", hours={})
        raise AssertionError("an unknown group was accepted")
    except ValueError:
        pass


_HAMMER = r"""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
import e1b_registry as R
path, tag, n = sys.argv[2], sys.argv[3], int(sys.argv[4])
for i in range(n):
    R.update_instance_hours({"%s-%03d" % (tag, i): {"hours": float(i), "final": i % 2 == 0},
                             "shared": {"hours": float(i), "final": False}}, path)
"""


def _selftest_hours_writer():
    """update_instance_hours: the merge rule, the lock, a refused unreadable file, and two concurrent writers."""
    import subprocess
    import sys
    prev = {"a": {"hours": 5.0, "final": True}, "b": {"hours": 3.0, "final": False},
            "c": {"hours": 2.0, "final": False}, "e": {"hours": 1.0, "final": True}}
    got = _merge_hours(prev, {"a": {"hours": 9.0, "final": False}, "b": {"hours": 1.0, "final": False},
                              "c": {"hours": 1.5, "final": True}, "d": {"hours": 0.5, "final": False}})
    assert got == {"a": {"hours": 5.0, "final": True}, "b": {"hours": 3.0, "final": False},
                   "c": {"hours": 1.5, "final": True}, "d": {"hours": 0.5, "final": False},
                   "e": {"hours": 1.0, "final": True}}, got
    with tempfile.TemporaryDirectory(prefix="e1b_registry_selftest_") as td:
        p = Path(td) / "instance_hours.json"
        assert update_instance_hours({"i-1": {"hours": 1.0, "final": False}}, p) == {"i-1": {"hours": 1.0,
                                                                                            "final": False}}
        seen = []
        out = update_instance_hours(lambda cur: seen.append(dict(cur)) or {"i-2": {"hours": 2.0, "final": True}}, p)
        assert seen == [{"i-1": {"hours": 1.0, "final": False}}] and out == load_instance_hours(p) and len(out) == 2
        raw = p.read_bytes()
        assert raw.endswith(b"\n") and b"\r" not in raw
        lock = Path(td) / "instance_hours.json.lock"
        lock.write_text("pid 1, since 2026-10-25T00:00:00Z\n")                  # a writer killed while it held it
        try:
            update_instance_hours({"i-3": {"hours": 3.0, "final": False}}, p, wait_s=0.3)
            raise AssertionError("a held lock was ignored")
        except SystemExit as e:
            assert "instance_hours.json.lock exists" in str(e), e
        assert p.read_bytes() == raw and lock.exists()                          # nothing written, the lock left
        lock.unlink()
        p.write_text("{not json")
        try:
            update_instance_hours({"i-3": {"hours": 3.0, "final": False}}, p)
            raise AssertionError("an unreadable file was replaced")
        except SystemExit as e:
            assert "not a JSON object" in str(e), e
        assert p.read_text() == "{not json" and not lock.exists()
        p.write_bytes(raw)
        # two writers at once, each merging 150 entries of its own and one shared running instance, while this
        # process reads: every read is a whole file holding at least what the previous read held, and no entry is lost
        n = 150
        procs = [subprocess.Popen([sys.executable, "-B", "-c", _HAMMER, str(Path(__file__).resolve().parent), str(p),
                                   tag, str(n)]) for tag in ("w1", "w2")]
        last, reads = {}, 0
        while any(q.poll() is None for q in procs):
            cur = load_instance_hours(p)
            assert set(last) <= set(cur), sorted(set(last) - set(cur))[:5]
            last, reads = cur, reads + 1
            time.sleep(0.002)
        assert all(q.wait() == 0 for q in procs), [q.returncode for q in procs]
        final = load_instance_hours(p)
        want = {"%s-%03d" % (t, i) for t in ("w1", "w2") for i in range(n)} | {"i-1", "i-2", "shared"}
        assert set(final) == want, (len(final), sorted(want - set(final))[:5])
        assert all(final["%s-%03d" % (t, i)] == {"hours": float(i), "final": i % 2 == 0}
                   for t in ("w1", "w2") for i in range(n))
        assert final["shared"] == {"hours": float(n - 1), "final": False}       # the largest measurement stays
        assert sorted(x.name for x in Path(td).iterdir()) == ["instance_hours.json"], list(Path(td).iterdir())


if __name__ == "__main__":
    _selftest()
