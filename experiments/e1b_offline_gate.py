"""Recompute the twin gate g offline from stored completion texts, under the job's exact Python version.

ast.parse and compile change between Python versions, so g recomputed under the analysis Python could differ
from the g the job computed (spec section 7.6). `offline_gate(items, python_version)` runs
flightrecorder/repro/twin_gate.gate in this process when this interpreter's full version equals python_version
(e.g. 3.12.15, the run's logged versions.python), and otherwise in a subprocess under
`uv run --no-project --python <full version> --managed-python --no-python-downloads` (a uv-managed interpreter of
exactly that version, installed beforehand with `uv python install <full version>`;
stdlib only: twin_gate imports nothing outside the standard library and reward_testhack). The version the
subprocess reports must equal python_version exactly; a different patch release is refused, not used.

Used by experiments/e1b_primary.py (C5 and exclusion rule (d); its pins record the interpreter version through
interpreter_version) and experiments/e1b_pilot_gate.py (P0).

CLI (what the subprocess runs): python experiments/e1b_offline_gate.py IN.json OUT.json
    IN:  {"items": [[text, fn_name], ...]}
    OUT: {"python": "3.12.x", "G": [0.0 or 1.0, ...]}
python experiments/e1b_offline_gate.py --selftest   # the checks of _selftest (no uv call, no network)
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from flightrecorder.repro.twin_gate import gate  # noqa: E402


def uv_python(version: str):
    """The command prefix that runs a uv-managed CPython of exactly `version` (-I isolated, -B no bytecode). uv
    never downloads an interpreter here (--no-python-downloads): install it once with `uv python install <version>`,
    so that no analysis step reaches the network on its own."""
    return ["uv", "run", "--no-project", "--python", str(version), "--managed-python", "--no-python-downloads",
            "python", "-I", "-B"]


def interpreter_version(version: str):
    """The full version uv's managed interpreter for `version` reports, or None when uv cannot run one (no uv, or
    no managed interpreter of that version and none downloadable)."""
    try:
        r = subprocess.run(uv_python(version) + ["-c", "import platform; print(platform.python_version())"],
                           capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError):
        return None
    return (r.stdout.strip() or None) if r.returncode == 0 else None


def offline_gate(items, python_version: str):
    """items: [(text, fn_name)]. Returns (python version that computed g, list of g). The version returned always
    equals python_version: in this process when this interpreter is exactly that version, else under uv."""
    items = [[t, f] for t, f in items]
    want = str(python_version)
    if want == platform.python_version():
        return want, [gate(t, f) for t, f in items]
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d) / "in.json", Path(d) / "out.json"
        src.write_text(json.dumps({"items": items}), encoding="utf-8")
        cmd = uv_python(want) + [str(Path(__file__).resolve()), str(src), str(dst)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("offline gate under Python %s failed: %s" % (want, r.stderr[-2000:]))
        out = json.loads(dst.read_text(encoding="utf-8"))
    if out.get("python") != want:
        raise RuntimeError("offline gate ran under Python %s, wanted exactly %s" % (out.get("python"), want))
    if len(out["G"]) != len(items):
        raise RuntimeError("offline gate returned %d values for %d items" % (len(out["G"]), len(items)))
    return out["python"], out["G"]


def _selftest():
    """The version rule without running uv: subprocess.run is replaced by a fake that plays the uv child."""
    here = platform.python_version()
    other = "3.12.15" if here != "3.12.15" else "3.13.14"
    items = [("Here:\n```python\ndef f(x):\n    return x\n```", "f"), ("Sure, in words.", "f")]
    used, g = offline_gate(items, here)
    assert used == here and g == [gate(t, f) for t, f in items] == [1.0, 0.0], (used, g)
    calls, real = [], subprocess.run

    def fake(report):
        def run(cmd, **kw):
            calls.append(cmd)
            if cmd[-2:] and cmd[-1].endswith("out.json"):
                ins = json.loads(Path(cmd[-2]).read_text(encoding="utf-8"))["items"]
                Path(cmd[-1]).write_text(json.dumps({"python": report, "G": [gate(t, f) for t, f in ins]}),
                                         encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 0, "", "")
            return subprocess.CompletedProcess(cmd, 0, report + "\n", "")
        return run
    try:
        subprocess.run = fake(other)
        used, g = offline_gate(items, other)
        assert used == other and g == [1.0, 0.0], (used, g)
        assert calls[-1][:9] == ["uv", "run", "--no-project", "--python", other, "--managed-python",
                                 "--no-python-downloads", "python", "-I"], calls[-1]
        assert interpreter_version(other) == other
        patch = other.rsplit(".", 1)[0] + ".0"                 # same minor, another patch release
        subprocess.run = fake(patch)
        try:
            offline_gate(items, other)
            raise AssertionError("a different patch release was accepted")
        except RuntimeError as exc:
            assert "wanted exactly %s" % other in str(exc), exc
    finally:
        subprocess.run = real
    print("SELFTEST OK: in-process only at the exact version %s; otherwise uv run --python <full version> "
          "--managed-python --no-python-downloads, and a reply from %s refused for %s" % (here, patch, other))
    return 0


def _cli(argv):
    if argv == ["--selftest"]:
        return _selftest()
    src, dst = Path(argv[0]), Path(argv[1])
    items = json.loads(src.read_text(encoding="utf-8"))["items"]
    out = {"python": platform.python_version(), "G": [gate(t, f) for t, f in items]}
    dst.write_text(json.dumps(out), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv[1:]))
