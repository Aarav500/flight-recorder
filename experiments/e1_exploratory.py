"""E1 exploratory report: analyses X1-X7 of the pre-registration, section 10 (experiments/e1_preregistration.md).

EXPLORATORY. NOT the registered analysis (that is experiments/e1_primary.py) and NOT pinned. Written 2026-10-08,
AFTER the registered result (experiments/e1_primary_report.txt: CLEAN NULL, non-superiority shown). Nothing here can
change the registered verdict. It changes no pinned file and no registered rule.

It checks the 14 pinned files, then imports and runs the seven modules in experiments/e1_exploratory_parts/ in order
X1..X7. Each module's run() returns {"id", "title", "lines", "data", "figures"}. The lines are printed under
"== Xn: <title> ==". Each module imports the registered code (e1_primary, e1_event, e1_detectors) and calls it
unchanged; every detail the pre-registration leaves open is printed by the module as a line beginning
"CHOICE (made after the data):". If a module raises, its error is printed and the next one runs.
All modules' data goes to experiments/e1_exploratory_results.json.

    python experiments/e1_exploratory.py > experiments/e1_exploratory_report.txt
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
import sys
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True       # read-only: leave no new __pycache__ next to the pinned files
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
import e1_primary as P  # noqa: E402

MODULES = [("X1", "x1_probe_event"), ("X2", "x2_published_events"), ("X3", "x3_stratified_test"),
           ("X4", "x4_lead_vs_window"), ("X5", "x5_pilot"), ("X6", "x6_judge_trajectories"),
           ("X7", "x7_length_mechanism")]
PARTS = EXP / "e1_exploratory_parts"
OUT = EXP / "e1_exploratory_results.json"
X = "[EXPLORATORY]"


def provenance():
    """As e1_supplementary.provenance: every pinned file must match experiments/e1_pins.json."""
    pins = json.loads(P.PINS.read_text())
    pins.pop("_env", None)
    bad = [rel for rel, h in pins.items() if P._sha(P.ROOT / rel) != h]
    if bad:
        raise SystemExit("PROVENANCE FAILURE: %s differ from the frozen versions" % bad)
    return len(pins)


def rel(p):
    p = Path(p).resolve()
    return p.relative_to(P.ROOT).as_posix() if p.is_relative_to(P.ROOT) else str(p)


# An absolute path under the repo becomes repo-relative (the repo is public; no local user path in the outputs).
ABS = re.compile(r"""(?:%s|%s)[\\/][^\s,;'")]*""" % (re.escape(str(P.ROOT)), re.escape(P.ROOT.as_posix())))


def tidy(s):
    return ABS.sub(lambda m: rel(m.group(0)), s)


def clean(o):
    """Strict JSON: numpy scalars to Python numbers, non-finite floats to the strings 'nan', '+inf', '-inf'."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if hasattr(o, "tolist"):                      # numpy scalar or array
        return clean(o.tolist())
    if isinstance(o, str):
        return tidy(o)
    if isinstance(o, float) and not math.isfinite(o):
        return "nan" if math.isnan(o) else ("+inf" if o > 0 else "-inf")
    return o


def main():
    sys.stdout.reconfigure(errors="backslashreplace")   # a stray non-cp1252 character must not abort the report
    say = print
    say("E1 EXPLORATORY REPORT: analyses X1-X7, pre-registration section 10. Unpinned. NOT the registered analysis "
        "(that is experiments/e1_primary.py).")
    say("%s Written 2026-10-08, after the registered result (experiments/e1_primary_report.txt). Every number below "
        "is exploratory and cannot change the registered verdict." % X)
    say("%s provenance: %d pinned files match experiments/e1_pins.json" % (X, provenance()))
    shas = {xid: hashlib.sha256((PARTS / (m + ".py")).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for xid, m in MODULES}
    say("%s module files (unpinned; SHA-256, LF), experiments/e1_exploratory_parts/:" % X)
    for xid, m in MODULES:
        say("  %s %-26s %s" % (xid, m + ".py", shas[xid]))
    results, figures, failed = {}, [], []
    for xid, m in MODULES:
        t0 = time.time()
        try:
            r = importlib.import_module("e1_exploratory_parts." + m).run()
            if r.get("id") != xid:
                raise ValueError("module returned id %r, expected %r" % (r.get("id"), xid))
            say("\n== %s: %s ==" % (xid, r["title"]))
            for line in r["lines"]:
                say(tidy(line))
            figs = [rel(f) for f in r.get("figures", [])]
            figures += [(xid, f) for f in figs]
            results[xid] = {"title": r["title"], "module": m + ".py", "module_sha256": shas[xid],
                            "data": clean(r["data"]), "figures": figs}
        except (Exception, SystemExit) as e:      # report it and continue with the next X
            failed.append(xid)
            say("\n== %s: ERROR ==" % xid)
            err = tidy("%s: %s" % (type(e).__name__, e))
            say("%s %s (%s.py) raised %s. No %s result is reported." % (X, xid, m, err, xid))
            results[xid] = {"module": m + ".py", "module_sha256": shas[xid], "error": err,
                            "traceback": tidy(traceback.format_exc())}
        print("%s done in %.1f s" % (xid, time.time() - t0), file=sys.stderr)
    OUT.write_text(json.dumps({"_note": "EXPLORATORY (pre-registration section 10). Not the registered analysis. "
                                        "Written 2026-10-08, after the registered result.",
                               **results}, indent=1, allow_nan=False) + "\n")
    say("\n== Outputs ==")
    say("%s data for all modules: %s" % (X, rel(OUT)))
    say("%s figures (%d):" % (X, len(figures)))
    for xid, f in figures:
        say("  %s %s%s" % (xid, f, "" if (P.ROOT / f).exists() else "  (MISSING)"))
    say("%s modules run: %d of %d%s" % (X, len(MODULES) - len(failed), len(MODULES),
                                         "; failed: %s" % ", ".join(failed) if failed else ""))
    reg = json.loads((EXP / "e1_primary_results.json").read_text())
    say("\nREGISTERED (restated from experiments/e1_primary_results.json, not exploratory): %s. Primary one-sided "
        "Fisher p %.4f. Nothing in X1-X7 changes it." % (reg["verdict"], reg["primary"]["fisher_p_one_sided"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
