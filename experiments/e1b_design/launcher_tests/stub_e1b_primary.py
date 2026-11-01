"""TEST STUB of experiments/e1b_primary.py, copied over it in the scratch clone only (run_tests.py). ASCII only.

--exclusions-only exits $STUB_EXCLUSIONS_RC (default 0) and prints the reason lines the real one prints for that code
(build_contract section 6): 3 a REPLACEMENT NEEDED line, 4 MAIN RUN NOT LAUNCHED lines, 1 a provenance failure on
stderr. No argument prints a stub report and exits 0. Each call leaves a marker file in $STUB_MARKER_DIR.
"""
import os
import sys
from pathlib import Path

marker = Path(os.environ.get("STUB_MARKER_DIR", "."))
if sys.argv[1:] == ["--exclusions-only"]:
    rc = int(os.environ.get("STUB_EXCLUSIONS_RC", "0"))
    (marker / "exclusions.called").write_text("x", encoding="ascii")
    if rc == 3:
        print("  REPLACEMENT NEEDED: judge is_prime seed 316 (run id e1b_judge_is_prime_s316, stage main_replace)")
    elif rc == 4:
        for s in (304, 305):
            print("  MAIN RUN NOT LAUNCHED: judge count_vowels seed %d (run id e1b_judge_count_vowels_s%d, stage main)"
                  % (s, s))
    elif rc == 1:
        sys.stderr.write("PROVENANCE: pins do not match (stub)\n")
    else:
        print("exclusions: none")
    sys.exit(rc)
if not sys.argv[1:]:
    (marker / "analysis.called").write_text("x", encoding="ascii")
    print("STUB REPORT: the registered analysis ran")
    sys.exit(0)
sys.exit(2)
