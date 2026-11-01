"""Stand-in for experiments/e1b_primary.py in the J3 launcher tests' scratch clone (run_tests.py copies it there and
commits it). ASCII only.

The real file's pins name local-only files (paper/), so its verify_pins_b cannot pass in a clone. This stub keeps
the names scripts/e1b_j3_job.build_data and its prep child call, with the real file's shapes:
  verify_pins_b()         appends "verify_pins_b" to runs/e1b/stub_calls.txt (the test checks it ran first)
  data_manifest(main_dir), _manifest_failure(man)
  load_e1b(main_dir, ...) every runs/e1b/main/e1b_*_s<seed>.json as used, by arm; runs/e1b/stub_selection.json
                          may set counts of replacements_needed and main_not_launched
  _run_id, _tag, MAIN_B, J3
The registered analysis is tested with the real file (run_tests.py, the data and load_j3 sections).
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN_B = ROOT / "runs" / "e1b" / "main"
J3 = {"judge": "Qwen/Qwen2.5-7B-Instruct", "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
      "dtype": "bfloat16"}


def verify_pins_b(path=None):
    p = ROOT / "runs" / "e1b" / "stub_calls.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="ascii", newline="\n") as f:
        f.write("verify_pins_b\n")
    return {}


def _tag(run):
    return run.get("_tag") or "%s/%s/s%s" % (run.get("arm", "run"), run.get("task"), run.get("seed"))


def _run_id(run):
    return run.get("run_id") or "e1b_%s_%s_s%d" % (run.get("arm"), run.get("task"), int(run.get("seed")))


def data_manifest(main_dir=MAIN_B):
    files = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(Path(main_dir).glob("e1b_*.json"))}
    text = "".join("%s  %s\n" % (h, n) for n, h in sorted(files.items()))
    m = Path(main_dir) / "MANIFEST.sha256"
    return {"files": files, "n_files": len(files), "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "collector": {"main/MANIFEST.sha256": hashlib.sha256(m.read_bytes()).hexdigest() if m.is_file() else None}}


def _manifest_failure(man):
    if not any(man["collector"].values()):
        return "no data manifest (stub)"
    return None


def load_e1b(main_dir=MAIN_B, launch_records=None, registered=None, manifest=None, prereg=None):
    used = {"hacking": [], "twin": []}
    for f in sorted(Path(main_dir).glob("e1b_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        lab = "hacking" if d["arm"] == "judge" else "twin"
        d["_tag"] = "%s/%s/s%d" % (lab, d["task"], int(d["seed"]))
        used[lab].append(d)
    sp = ROOT / "runs" / "e1b" / "stub_selection.json"
    s = json.loads(sp.read_text()) if sp.is_file() else {}
    return {"used": used, "excluded": [], "unregistered": [], "surplus": [],
            "replacements_needed": [("judge", "count_vowels", 316)] * int(s.get("replacements_needed", 0)),
            "replacements_blocked": [], "main_not_launched": [("judge", "is_prime", 315)] * int(
                s.get("main_not_launched", 0)), "lines": []}
