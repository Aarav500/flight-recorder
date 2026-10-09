"""E1 third-judge rescoring: build the completions file the GPU job scores.

NOT pinned and NOT registered. Written 2026-10-09. Changes no frozen file: it imports experiments/e1_primary.py
(load_main, _registered, _exclusion) and reads runs/e1/main, nothing else.

Output, one JSON object per stored completion, sorted by (run, step, i):
    {"run": file stem, "arm": "hacking"|"benign", "task", "seed", "step", "i" (0-7),
     "text", "hidden", "hidden_lenient"}
"text" is completions[step].texts[i] exactly as the job logged it. The job stored x[:1500] (scripts/
launch_slow_onset.py, COMPS.append), so a text of exactly 1500 characters may be a truncated completion;
those are counted and printed because the training-time judge saw the untruncated text.

The gzip stream is deterministic (mtime 0, no file name), so re-running gives the same bytes and SHA-256.

    python experiments/e1_rescore/prep_data.py [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True       # leave no __pycache__ next to the pinned files
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import e1_primary as P  # noqa: E402

DEFAULT_OUT = Path(__import__("os").environ.get(
    "E1_RESCORE_SCRATCH", str(Path(__file__).resolve().parents[2] / "experiments" / "e1_rescore" / "_work")))
N_RUNS, N_STEPS, N_GEN = 48, 400, 8
TEXT_CAP = 1500                       # the job's COMPS truncation


def stem_map(main_dir=P.MAIN):
    """(arm, task, seed) -> file stem, by the same rule load_main uses (sorted names, registered,
    not excluded, first wins). load_main does not return stems, so this re-derives them."""
    out = {}
    for f in sorted(Path(main_dir).glob("*.json")):
        d = json.loads(f.read_text())
        if not P._registered(d) or P._exclusion(d):
            continue
        key = ("hacking" if d["judge"] == P.SMALL else "benign", d["task"], int(d["seed"]))
        out.setdefault(key, f.stem)
    return out


def build_rows():
    used, excluded, unregistered, _launched, surplus = P.load_main()
    runs = used["hacking"] + used["benign"]
    assert len(used["hacking"]) == 24 and len(used["benign"]) == 24, {a: len(v) for a, v in used.items()}
    assert len(runs) == N_RUNS
    stems = stem_map()
    rows, stats = [], Counter()
    seen = set()
    for r in runs:
        arm, task, seed = r["_arm"], r["task"], int(r["seed"])
        stem = stems[(arm, task, seed)]
        assert stem not in seen, stem
        seen.add(stem)
        comps, log = r["completions"], r["log"]
        assert len(comps) == N_STEPS, (stem, len(comps))
        assert len(log) == N_STEPS, (stem, len(log))
        for s, c in enumerate(comps):
            assert int(c["step"]) == s, (stem, s, c["step"])
            texts, hid, hlen = c["texts"], c["hidden"], c["hidden_lenient"]
            assert len(texts) == N_GEN and len(hid) == N_GEN and len(hlen) == N_GEN, (stem, s)
            # the stored per-completion oracle must average to the logged step value
            if abs(sum(hid) / N_GEN - float(log[s]["hidden"])) > 1e-9:
                stats["hidden_mean_mismatch"] += 1
            for i in range(N_GEN):
                t = texts[i]
                assert isinstance(t, str), (stem, s, i)
                stats["capped_1500"] += len(t) >= TEXT_CAP
                stats["empty"] += (t == "")
                rows.append({"run": stem, "arm": arm, "task": task, "seed": seed, "step": s, "i": i,
                             "text": t, "hidden": float(hid[i]), "hidden_lenient": float(hlen[i])})
    rows.sort(key=lambda x: (x["run"], x["step"], x["i"]))
    assert len(rows) == N_RUNS * N_STEPS * N_GEN == 153_600, len(rows)
    meta = {"excluded": excluded, "unregistered": unregistered, "surplus": surplus,
            "arms": Counter(r["_arm"] for r in runs), "tasks": Counter((r["_arm"], r["task"]) for r in runs)}
    return rows, stats, meta


def write(rows, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "e1_completions.jsonl.gz"
    with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0,
                                                compresslevel=9) as gz:
        for row in rows:
            gz.write((json.dumps(row) + "\n").encode("ascii"))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    (out_dir / "e1_completions.sha256").write_text(sha + "\n", encoding="ascii", newline="\n")
    return path, sha


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    a = ap.parse_args(argv)
    rows, stats, meta = build_rows()
    path, sha = write(rows, Path(a.out_dir))
    print("runs: %s; per (arm, task): %s" % (dict(meta["arms"]), dict(meta["tasks"])))
    print("load_main: %d excluded, %d unregistered, %d surplus" % (
        len(meta["excluded"]), len(meta["unregistered"]), len(meta["surplus"])))
    for tag, why in meta["excluded"]:
        print("  excluded %s: %s" % (tag, why))
    print("rows: %d (%d runs x %d steps x %d)" % (len(rows), N_RUNS, N_STEPS, N_GEN))
    print("texts at the %d-char storage cap (possibly truncated): %d; empty texts: %d; "
          "steps whose stored hidden mean != logged hidden: %d" % (
              TEXT_CAP, stats["capped_1500"], stats["empty"], stats["hidden_mean_mismatch"]))
    print("wrote %s  (%d bytes)" % (path, path.stat().st_size))
    print("sha256 %s" % sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
