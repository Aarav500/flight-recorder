"""Tests of scripts/launch_e1b_aws.py and scripts/e1b_autorun.py without AWS (a design record). ASCII only.

WHAT IT DOES
  Clones this repo into a scratch folder, copies in the working-tree E1b files (template, lock, gate, the edited
  reward_judge.py, launcher, autorun, registry, power script, pilot gate) and a stub e1b_primary.py, commits them there
  as the freeze candidate F, and builds branches on F that hold a fake freeze record: section 13's commit hash (F), a
  manifest of F's blobs, an OpenTimestamps receipt of it (a well-formed pending proof), a filled human-reader field
  and approved spend rows, or one of them missing or wrong. Nothing is committed, written or launched in this repo.
  A fake aws CLI (fake_aws.py) goes first on PATH: it records every call and serves S3 from a local mirror, and EC2,
  IAM and CloudTrail from a small state file. The instance's user data is run for real under bash, with fake
  python3 (fake_job.py stands in for `uv run job.py`), aws, nvidia-smi and shutdown on PATH.

COVERED
  dry runs (pilot at HEAD and at F, main at F); every --submit refusal (no --commit, a changed closure file, template,
  lock, gate, launcher or autorun, no manifest, an uncommitted manifest, a tampered or short manifest, no receipt, a
  wrong receipt (OpenTimestamps proof of another file, a bare OTS header, a malformed OTS proof, a timestamp request
  .tsq committed as the .tsr, and a real RFC 3161 response of another file from a throwaway openssl TSA when openssl
  is on PATH), no freeze commit, an unregistered --commit (a branch off F, or a later commit), a section 14 commit
  whose row does not name every changed file, a reader or approval cell without a date, its words or with "TBD" /
  "not", the pilot ceiling, LAUNCH_CAP, the 14-replacement allowance, a malformed replacement list, the role policy
  (e1/* passes, a policy without it refuses), an already-used S3 prefix, a held launch lock, measured instance hours
  over the ceiling); a pilot submit with every guard satisfied (with a capacity fallback), its launch records
  (job_sha256 included) and uploads; status and the instance hours it records; collect refused before the final rc
  and done after; the manifest; a lost run; the audit trail (raw to gitignored paper/freeze/, redacted copy to
  runs/e1b/audit/, refused when paper/ is not ignored); a main submit of two instances; two concurrent submits; a run
  launched meanwhile by another process; the user data's runs, watchdog, missing job and signal paths; the
  autorun's outcomes (STOP, replacement needed, main runs not launched, provenance failure, analysis ran) plus a
  changed analysis script, launcher or autorun; and runs/e1b/instance_hours.json written by three processes at once
  while a fourth reads it (finding R9: no entry lost, no partial file read).

    python experiments/e1b_design/launcher_tests/run_tests.py --scratch <empty or new folder>
Exit 0 when every check passes. Output goes to stdout and the scratch folder only.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PY = sys.executable
BUCKET = "fr-deploy-325527186655"
S3R = "e1/e1b"                       # the E1b root inside the bucket (build_contract section 8)
RECORD_KEYS = ["stage", "run_id", "arm", "task", "seed", "git_commit", "package_sha256", "template_sha256",
               "lock_sha256", "gate_sha256", "instance_id", "region", "ami", "instance_type", "market",
               "submitted_utc", "s3", "job_sha256"]
COPY = ("scripts/e1b_job.py", "experiments/e1b_uv.lock", "flightrecorder/repro/twin_gate.py",
        "flightrecorder/repro/reward_judge.py", "scripts/launch_e1b_aws.py", "scripts/e1b_autorun.py",
        "experiments/e1b_registry.py", "experiments/e1b_power.py", "experiments/e1b_pilot_gate.py",
        "experiments/e1b_preregistration.md", "experiments/e1b_analysis_spec.md")
OTS_MAGIC = bytes.fromhex("004f70656e54696d657374616d7073000050726f6f6600bf89e2e884e89294")
OTS_PENDING = bytes.fromhex("83dfe30d2ef90c8e")
# opentimestamps 0.4.5's own serialization (DetachedTimestampFile of sha256(b"manifest"): append, sha256, then a
# Bitcoin and a pending attestation), from the body after the 65-byte header
OTS_LIB_BODY = bytes.fromhex(
    "f0102222222222222222222222222222222208ff000588960d73d7190103f7ef150083dfe30d2ef90c8e2e2d68747470733a2f2f61"
    "6c6963652e6274632e63616c656e6461722e6f70656e74696d657374616d70732e6f7267")
TSA_CNF = """[ req ]
distinguished_name = dn
prompt = no
[ dn ]
CN = scratch test TSA
[ v3_tsa ]
basicConstraints = CA:FALSE
keyUsage = critical, digitalSignature
extendedKeyUsage = critical, timeStamping
[ tsa ]
default_tsa = tsa_config1
[ tsa_config1 ]
dir = .
serial = ./tsaserial
crypto_device = builtin
signer_cert = ./tsa.crt
signer_key = ./tsa.key
signer_digest = sha256
default_policy = 1.2.3.4.1
digests = sha256
accuracy = secs:1
ordering = no
tsa_name = no
ess_cert_id_chain = no
ess_cert_id_alg = sha256
"""
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    detail = str(detail)[-2500:].encode("ascii", "backslashreplace").decode("ascii")    # printable on any console
    print("%s  %s%s" % ("PASS" if cond else "FAIL", name, "" if cond else "\n      " + detail), flush=True)
    return bool(cond)


def run(cmd, cwd, env=None, timeout=900):
    r = subprocess.run([str(c) for c in cmd], cwd=str(cwd), env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout)
    return r.returncode, r.stdout + r.stderr


def lf_bytes(p):
    return Path(p).read_bytes().replace(b"\r\n", b"\n")


def git_bash():
    if os.name != "nt":
        return shutil.which("bash")
    g = shutil.which("git")
    for p in (Path(g).resolve().parents if g else []):
        for c in (p / "usr" / "bin" / "bash.exe", p / "bin" / "bash.exe"):
            if c.is_file():
                return str(c)
    raise SystemExit("Git bash not found")


def ots_receipt(digest):
    """A well-formed pending OpenTimestamps proof of digest, as `ots stamp` writes one (append a nonce, sha256, then a
    fork to two calendars' pending attestations). opentimestamps 0.4.5 reads these bytes."""
    def vb(b):
        return bytes([len(b)]) + b

    def pending(uri):
        return b"\x00" + OTS_PENDING + vb(vb(uri))
    return (OTS_MAGIC + b"\x01\x08" + digest + b"\xf0" + vb(b"\x11" * 16) + b"\x08" + b"\xff"
            + pending(b"https://a.pool.opentimestamps.org") + pending(b"https://b.pool.opentimestamps.org"))


def tsq_bytes(digest):
    """An RFC 3161 TimeStampReq (what `openssl ts -query -sha256 -cert` writes): version 1, the SHA-256 imprint,
    certReq TRUE. A request, not a timestamp."""
    return (bytes.fromhex("3039020101" "3031300d060960864801650304020105000420") + digest + bytes.fromhex("0101ff"))


def short(p):
    """The 8.3 form of an existing path on Windows. The fake S3 mirror nests keys like
    <state>/s3/<bucket>/e1/e1b/main_replace/<run_id>/ deep enough that a long scratch folder passes MAX_PATH."""
    p = str(p)
    if os.name != "nt":
        return p
    import ctypes
    buf = ctypes.create_unicode_buffer(1024)
    n = ctypes.windll.kernel32.GetShortPathNameW(p, buf, 1024)
    return buf.value if 0 < n < 1024 else p


class Ctx:
    def __init__(self, scratch):
        root = Path(scratch).resolve()
        root.mkdir(parents=True, exist_ok=True)
        self.root = Path(short(root))
        self.clone, self.state = self.root / "clone", self.root / "state"
        self.bin_cli, self.bin_sh = self.state / "bin_cli", self.state / "bin_sh"
        self.bash = git_bash()
        self.openssl = shutil.which("openssl")

    # ---- environment
    def env(self, extra=None, path_first=None):
        e = dict(os.environ)
        e["PATH"] = str(path_first or self.bin_cli) + os.pathsep + e.get("PATH", "")
        e["FAKE_AWS_STATE"] = str(self.state)
        e.update(extra or {})
        return e

    def config(self, **kw):
        base = {"capacity_fail": [], "vcpu_fail": [], "states": {}, "gone": [],
                "policy_resources": ["arn:aws:s3:::%s/e1/*" % BUCKET]}       # the real policy (read 2026-10-23)
        base.update(kw)
        (self.state / "config.json").write_text(json.dumps(base, indent=1))

    def calls(self):
        p = self.state / "calls.jsonl"
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []

    def clear_calls(self):
        (self.state / "calls.jsonl").unlink(missing_ok=True)

    def s3p(self, key):
        return self.state / "s3" / BUCKET / key

    def s3put(self, key, data):
        p = self.s3p(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data if isinstance(data, bytes) else data.encode())

    def s3json(self, key):
        p = self.s3p(key)
        return json.loads(p.read_text()) if p.exists() else None

    def git(self, *a):
        rc, out = run(["git", *a], self.clone)
        if rc != 0:
            raise RuntimeError("git %s: %s" % (" ".join(a), out))
        return out.strip()

    def launcher(self, *a, extra=None):
        return run([PY, "-B", "scripts/launch_e1b_aws.py", *a], self.clone, self.env(extra))

    def pyc(self, code, *args):
        """Python code run in the clone (fake aws on PATH), -> (rc, output)."""
        return run([PY, "-B", "-c", code, *args], self.clone, self.env())

    def runs(self):
        return self.clone / "runs" / "e1b"

    def lock(self):
        return self.runs() / "launch.lock"

    def launches(self):
        p = self.runs() / "launches.json"
        return json.loads(p.read_text()) if p.exists() else None

    def hours(self):
        p = self.runs() / "instance_hours.json"
        return json.loads(p.read_text()) if p.exists() else None

    # ---- setup
    def setup(self):
        for d in (self.clone, self.state):
            if d.exists():
                shutil.rmtree(d)
        self.root.mkdir(parents=True, exist_ok=True)
        rc, out = run(["git", "-c", "core.autocrlf=false", "clone", "-q", str(REPO), str(self.clone)], self.root)
        assert rc == 0, out
        self.git("config", "core.autocrlf", "false")
        self.git("config", "user.name", "e1b-launcher-test")
        self.git("config", "user.email", "e1b-launcher-test@example.invalid")
        # COPY, plus every file section 13's FILES names in the working-tree pre-registration (the list grows as the
        # freeze is prepared); a named file that does not exist yet gets a placeholder, so the scratch freeze commit
        # holds every manifest file
        listed = re.search(r'FILES="([^"]*)"', lf_bytes(REPO / "experiments/e1b_preregistration.md").decode("utf-8"))
        for rel in dict.fromkeys(list(COPY) + (listed.group(1).split() if listed else [])):
            (self.clone / rel).parent.mkdir(parents=True, exist_ok=True)
            src = REPO / rel
            (self.clone / rel).write_bytes(lf_bytes(src) if src.is_file() else b"placeholder (launcher tests)\n")
        (self.clone / "experiments/e1b_primary.py").write_bytes(lf_bytes(HERE / "stub_e1b_primary.py"))
        for rel, text in (("experiments/e1b_pins.json", "{}\n"), ("experiments/e1b_selftest.txt", "placeholder\n")):
            src = REPO / rel
            (self.clone / rel).write_bytes(lf_bytes(src) if src.exists() else text.encode())
        # F holds the draft's record cells whatever the real file holds: the tests write their own, and the refusal tests
        # need the draft values
        P = self.clone / "experiments/e1b_preregistration.md"
        b = P.read_bytes()
        for pat, rep in ((rb"(?m)^(\| Commit hash \| )[^|\n]*( \|)$", rb"\1none yet\2"),
                         (rb"(?m)^(\| Human reader before the freeze[^\n]*\| )[^|\n]*( \|)$", rb"\1none yet\2"),
                         (rb"(?m)^(\| Pilot \| 6 runs[^\n]*\| )[^|\n]*( \|)$", rb"\1pending\2"),
                         (rb"(?m)^(\| Primary \| 96 runs[^\n]*\| )[^|\n]*( \|)$", rb"\1pending\2"),
                         (rb"(?m)^(\| J3 scoring [^\n]*\| )[^|\n]*( \|)$", rb"\1pending approval\2")):
            b = re.sub(pat, rep, b)
        P.write_bytes(b)
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "scratch: E1b freeze candidate F")
        self.F = self.git("rev-parse", "HEAD")
        pre = self.git("show", "%s:experiments/e1b_preregistration.md" % self.F)
        self.files = re.search(r'FILES="([^"]*)"', pre).group(1).split()
        for d in (self.state, self.bin_cli, self.bin_sh, self.state / "markers"):
            d.mkdir(parents=True, exist_ok=True)
        py, fa, fj = Path(PY).as_posix(), (HERE / "fake_aws.py").as_posix(), (HERE / "fake_job.py").as_posix()
        if os.name == "nt":
            (self.bin_cli / "aws.cmd").write_text('@"%s" "%s" %%*\r\n' % (PY, HERE / "fake_aws.py"))
        else:
            self._sh(self.bin_cli / "aws", 'exec "%s" "%s" "$@"\n' % (py, fa))
        self._sh(self.bin_sh / "aws", 'exec "%s" "%s" "$@"\n' % (py, fa))
        self._sh(self.bin_sh / "nvidia-smi", 'echo "name, driver_version, memory.total [MiB]"\n'
                                              'echo "NVIDIA L40S (fake), 000.00, 46068 MiB"\n')
        self._sh(self.bin_sh / "shutdown", 'echo "$*" >> "$FAKE_AWS_STATE/shutdown.log"\n')
        self._sh(self.bin_sh / "python3", (
            'if [ "$1" = "-m" ] && [ "$2" = "pip" ]; then echo "fake pip: $*"; exit 0; fi\n'
            'if [ "$1" = "-m" ] && [ "$2" = "uv" ]; then\n'
            '  if [ "$3" = "--version" ]; then echo "uv 0.12.23 (fake)"; exit 0; fi\n'
            '  RID=$(sed -n \'s/^ARM, TASK, SEED, STAGE, RUN_ID = .*"\\([^"]*\\)"[[:space:]]*$/\\1/p\' job.py)\n'
            '  case " ${FAKE_HANG:-} " in *" $RID "*) echo "fake job $RID: hang" >&2; exec sleep 100000;; esac\n'
            '  exec "%s" "%s" "$@"\n'
            'fi\n'
            'exec "%s" "$@"\n') % (py, fj, py))
        self.config()

    @staticmethod
    def _sh(path, body):
        with open(path, "w", encoding="ascii", newline="\n") as f:
            f.write("#!/bin/sh\n" + body)
        os.chmod(path, 0o755)

    def manifest(self, mode="good"):
        lines = []
        for p in self.files:
            b = subprocess.run(["git", "cat-file", "blob", "%s:%s" % (self.F, p)], cwd=str(self.clone),
                               capture_output=True).stdout
            lines.append("%s  %s\n" % (hashlib.sha256(b).hexdigest(), p))
        if mode == "tampered":
            lines[0] = ("0" if lines[0][0] != "0" else "1") + lines[0][1:]
        if mode == "short":
            lines = lines[:-1]
        return "".join(lines).encode("ascii")

    def prereg(self):
        return self.clone / "experiments/e1b_preregistration.md"

    def edit_prereg(self, subs):
        p = self.prereg()
        t = p.read_text(encoding="utf-8")
        for pat, rep in subs:
            t, n = re.subn(pat, rep, t, flags=re.M)
            assert n == 1, pat
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(t)

    def add_row14(self, commit, what):
        """Append a section 14 deviation row naming commit, in the working tree."""
        p = self.prereg()
        t = p.read_text(encoding="utf-8")
        head = "| Date | Section changed | What changed | Why | Before or after a look at the data? | Effect on the claims " \
               "| Commit |\n|---|---|---|---|---|---|---|\n"
        assert head in t
        row = "| 2026-10-25 | 9 | %s | P0 (test) | before | none | %s |\n" % (what, commit)
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(t.replace(head, head + row))

    def branch(self, name, manifest="good", receipt="good", reader=True, pilot=True, primary=True, freeze=True,
               reader_text=None, pilot_text=None):
        """A branch on F holding a freeze record: section 13 commit hash and reader, section 12 approvals, manifest and
        receipt."""
        self.git("checkout", "-q", "-B", name, self.F)
        subs = []
        if freeze:
            subs.append((r"^(\| Commit hash \| )none yet( \|)$", r"\g<1>%s\2" % self.F))
        if reader or reader_text:
            subs.append((r"^(\| Human reader before the freeze[^\n]*\| )none yet( \|)$",
                         r"\g<1>%s\2" % (reader_text or "Test Reader (scratch clone), 2026-10-23, sections 1-12")))
        if pilot or pilot_text:
            subs.append((r"^(\| Pilot \|[^\n]*\| )pending( \|)$",
                         r"\g<1>%s\2" % (pilot_text or '"go" (test approval, 2026-10-23)')))
        if primary:
            subs.append((r"^(\| Primary \|[^\n]*\| )pending( \|)$", r'\g<1>"go" (test approval, 2026-10-23)\2'))
        self.edit_prereg(subs)
        if manifest:
            man = self.manifest(manifest)
            (self.clone / "experiments/e1b_freeze_manifest.txt").write_bytes(man)
            d = hashlib.sha256(man).digest()
            ots = self.clone / "experiments/e1b_freeze_manifest.txt.ots"
            if receipt == "good":
                ots.write_bytes(ots_receipt(d))
            elif receipt == "bad":
                ots.write_bytes(ots_receipt(hashlib.sha256(b"another file").digest()))
            elif receipt == "bare":                       # the reviewer's 65-byte header: no timestamp at all
                ots.write_bytes(OTS_MAGIC + b"\x01\x08" + d)
            elif receipt == "garbage":                    # header, then bytes that are not an OTS timestamp tree
                ots.write_bytes(OTS_MAGIC + b"\x01\x08" + d + b"\xf0\x10fake-pending-attestation")
            elif receipt == "tsq":                        # the reviewer's case: the query committed as the .tsr
                (self.clone / "experiments/e1b_freeze_manifest.tsr").write_bytes(self.tsq(man))
            elif receipt in ("tsr", "badtsr"):
                (self.clone / "experiments/e1b_freeze_manifest.txt.tsr").write_bytes(
                    self.tsr(man if receipt == "tsr" else b"another file"))
        self.git("add", "-A", "experiments")
        self.git("commit", "-q", "-m", "scratch: freeze record (%s)" % name)

    def use(self, name):
        self.git("checkout", "-q", name)

    def tsa_dir(self):
        d = self.state / "tsa"
        if not (d / "tsa.crt").exists():
            d.mkdir(parents=True, exist_ok=True)
            (d / "tsa.cnf").write_text(TSA_CNF)
            (d / "tsaserial").write_text("01\n")
            rc, out = run([self.openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", "tsa.key", "-out",
                           "tsa.crt", "-days", "2", "-config", "tsa.cnf", "-extensions", "v3_tsa"], d)
            assert rc == 0, out
        return d

    def tsq(self, data):
        """The RFC 3161 query of data: from openssl when it is on PATH, else built by hand (same layout)."""
        if not self.openssl:
            return tsq_bytes(hashlib.sha256(data).digest())
        d = self.tsa_dir()
        (d / "data.bin").write_bytes(data)
        (d / "q.tsq").unlink(missing_ok=True)
        rc, out = run([self.openssl, "ts", "-query", "-data", "data.bin", "-sha256", "-cert", "-out", "q.tsq"], d)
        assert rc == 0, out
        return (d / "q.tsq").read_bytes()

    def tsr(self, data):
        """A real RFC 3161 response (status Granted) over data, from a throwaway TSA made with openssl here."""
        d = self.tsa_dir()
        (d / "data.bin").write_bytes(data)
        for f in ("q.tsq", "r.tsr"):
            (d / f).unlink(missing_ok=True)
        rc, out = run([self.openssl, "ts", "-query", "-data", "data.bin", "-sha256", "-cert", "-out", "q.tsq"], d)
        rc2, out2 = run([self.openssl, "ts", "-reply", "-config", "tsa.cnf", "-queryfile", "q.tsq", "-out", "r.tsr"], d)
        assert rc == 0 and (d / "r.tsr").exists(), out + out2
        return (d / "r.tsr").read_bytes()


def no_spend(calls):
    """No instance started and nothing uploaded."""
    for c in calls:
        if c[:2] == ["ec2", "run-instances"]:
            return False
        if c[:2] == ["s3", "cp"] and len(c) > 3 and c[3].startswith("s3://"):
            return False
    return True


def reads_outputs(calls):
    return [c for c in calls if any(("job.log" in a or "result.json" in a) for a in c)]


def refusals(out):
    return [ln for ln in out.splitlines() if "--submit would REFUSE" in ln]


def rec(stage, arm, task, seed, iid, itype="g6e.xlarge", region="us-east-1"):
    rid = "e1b_%s_%s_s%d" % (arm, task, seed)
    return {"stage": stage, "run_id": rid, "arm": arm, "task": task, "seed": seed, "git_commit": "0" * 40,
            "package_sha256": "0" * 64, "template_sha256": "0" * 64, "lock_sha256": "0" * 64, "gate_sha256": "0" * 64,
            "instance_id": iid, "region": region, "ami": "ami-x", "instance_type": itype, "market": "ondemand",
            "submitted_utc": "2026-10-23T00:00:00Z", "s3": "s3://%s/%s/%s/%s" % (BUCKET, S3R, stage, rid),
            "job_sha256": "0" * 64}


TASKS = ("count_vowels", "is_prime", "second_largest")
ARMS = ("judge", "judgegate")


def main_records():
    rows = [(a, t, s) for s in range(300, 316) for a in ARMS for t in TASKS]
    return [rec("main", a, t, s, "i-oldmain%04d" % (i // 12)) for i, (a, t, s) in enumerate(rows)]


# ------------------------------------------------------------------------------------------
def test_dry_runs(c):
    c.use("good")
    c.clear_calls()
    rc, out = c.launcher("pilot")
    ref = refusals(out)
    check("dry run pilot at HEAD: exit 0, DRY RUN, 6 runs baked", rc == 0 and "DRY RUN" in out
          and "baked 6 job(s)" in out, out)
    check("dry run pilot at HEAD: the only refusal is commit identity (HEAD is not the freeze commit F, nor a "
          "section 14 commit)", len(ref) == 1 and "is neither the freeze commit" in ref[0], "\n".join(ref))
    check("dry run pilot at HEAD: --commit missing is noted", "--commit is required with --submit" in out, out)
    rc, out = c.launcher("pilot", "--commit=" + c.F, "--show-userdata")
    check("dry run pilot at F: exit 0, no refusal, user data shown", rc == 0 and "would REFUSE" not in out and
          "shutdown -h +1110" in out and "DRY RUN" in out and "freeze commit (section 13): %s" % c.F in out, out)
    check("dry run pilot at F: import closure is the 7 package files",
          "flightrecorder/__init__.py, flightrecorder/repro/__init__.py, flightrecorder/repro/oracle_lenient.py, "
          "flightrecorder/repro/reward_codeset.py, flightrecorder/repro/reward_judge.py, "
          "flightrecorder/repro/reward_testhack.py, flightrecorder/repro/twin_gate.py" in out, out)
    check("dry run pilot at F: estimate USD 15.51 at g6e.xlarge, receipt and approvals read",
          "USD 15.51 at g6e.xlarge" in out and "receipt experiments/e1b_freeze_manifest.txt.ots stamps the "
          "manifest's sha256" in out and "section 12 Pilot approval" in out and "human reader: Test Reader" in out,
          out)
    check("dry run pilot at F: the S3 prefixes are under e1/e1b/pilot/",
          "s3://%s/e1/e1b/pilot/e1b_judge_count_vowels_s299" % BUCKET in out and "/e1b/pilot/" in out and
          "s3://%s/e1b/" % BUCKET not in out, out)
    t0 = time.time()
    rc, out = c.launcher("main", "--commit=" + c.F)
    plan = re.findall(r"^(\d+)\s+(e1b_\S+)\s+s3://", out, re.M)
    check("dry run main at F: 96 runs in 8 instances of 12, seed-major, no refusal (%.0f s)" % (time.time() - t0),
          rc == 0 and len(plan) == 96 and sorted({b for b, _ in plan}) == [str(i) for i in range(8)]
          and [r for _, r in plan[:6]] == ["e1b_%s_%s_s300" % (a, t) for a in ARMS for t in TASKS]
          and plan[11][1] == "e1b_judgegate_second_largest_s301" and plan[12][1] == "e1b_judge_count_vowels_s302"
          and "would REFUSE" not in out and "section 12 Primary approval" in out, out[-3000:])
    check("dry run main: the worst case is still printed", "worst case (every instance to its backstop" in out, out)
    check("dry runs make no AWS call and write no launch record", c.calls() == [] and c.launches() is None,
          c.calls()[:5])


def refuse(c, name, args, expect, branch="good", local=True, extra=None):
    c.use(branch)
    c.clear_calls()
    rc, out = c.launcher(*args, extra=extra)
    calls = c.calls()
    ok = rc != 0 and expect in out and no_spend(calls) and (not local or calls == [])
    check("refuse: %s" % name, ok, "rc %d calls %s\n%s" % (rc, calls[:4], out[-2500:]))
    return out


def modified(c, rel, name):
    c.use("good")
    p = c.clone / rel
    keep = p.read_bytes()
    p.write_bytes(keep + b"\n# scratch modification\n")
    try:
        refuse(c, name, ("pilot", "--submit", "--commit=" + c.F), rel + " differs from its blob")
    finally:
        p.write_bytes(keep)


def test_receipts(c):
    """F3: a receipt must be a timestamp. The reviewer's .tsq-as-.tsr and 65-byte OTS header cases, end to end."""
    sub = ("pilot", "--submit", "--commit=" + c.F)
    refuse(c, "a timestamp request (.tsq) committed as experiments/e1b_freeze_manifest.tsr", sub,
           "receipt experiments/e1b_freeze_manifest.tsr is not a timestamp response", "tsq")
    refuse(c, "a bare 65-byte OpenTimestamps header (no operation, no attestation)", sub,
           "receipt experiments/e1b_freeze_manifest.txt.ots is a bare header", "bare")
    refuse(c, "an OpenTimestamps header followed by bytes that are not a timestamp tree", sub,
           "is not a well-formed OpenTimestamps proof", "garbage")
    refuse(c, "an OpenTimestamps proof of another file", sub, "stamps another file digest, not the manifest's sha256",
           "badreceipt")
    # unit cases on the receipt functions
    code = ("import hashlib, json, sys; sys.path.insert(0, 'scripts'); import launch_e1b_aws as L\n"
            "d = hashlib.sha256(b'manifest').digest(); h = L.OTS_MAGIC + bytes([1, 8]) + d\n"
            "body = bytes.fromhex(sys.argv[1]); tsq = bytes.fromhex(sys.argv[2])\n"
            "rej = bytes.fromhex('3005300302010" "2')\n"
            "print(json.dumps({'lib': L.ots_problem(h + body, d), 'lib_trailing': L.ots_problem(h + body + b'0', d),\n"
            "  'lib_cut': L.ots_problem(h + body[:-3], d), 'no_attestation': L.ots_problem(h + bytes.fromhex('f00122')"
            " + bytes([8]) + bytes([0]) + bytes.fromhex('1122334455667788') + bytes([0]), d),\n"
            "  'tsq': L.tsr_problem(tsq, d), 'rejection': L.tsr_problem(rej, d), 'empty': L.tsr_problem(b'', d)}))\n")
    rc, out = c.pyc(code, OTS_LIB_BODY.hex(), tsq_bytes(hashlib.sha256(b"manifest").digest()).hex())
    try:
        got = json.loads(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        got = {}
    check("receipts: the reference library's proof (pending + Bitcoin) is accepted", rc == 0 and got.get("lib") is None
          and "lib" in got, out)
    check("receipts: trailing bytes, a cut proof, and an unknown attestation only are refused",
          "after the timestamp" in (got.get("lib_trailing") or "") and "well-formed" in (got.get("lib_cut") or "")
          and "no pending or Bitcoin attestation" in (got.get("no_attestation") or ""), got)
    check("receipts: a hand-built .tsq, a rejection response and an empty file are refused as .tsr",
          "not a timestamp response" in (got.get("tsq") or "") and "not granted" in (got.get("rejection") or "")
          and "not one DER SEQUENCE" in (got.get("empty") or ""), got)
    if c.openssl:
        c.use("tsr")
        rc, out = c.launcher("pilot", "--commit=" + c.F)
        check("an RFC 3161 receipt (.tsr, status Granted, from a scratch TSA) of the manifest is accepted",
              rc == 0 and "would REFUSE" not in out and
              "receipt experiments/e1b_freeze_manifest.txt.tsr stamps the manifest's sha256" in out, out)
        refuse(c, "an RFC 3161 receipt of another file", sub,
               "receipt experiments/e1b_freeze_manifest.txt.tsr has a TSTInfo whose message imprint is not the "
               "manifest's sha256", "badtsr")
    else:
        print("SKIP  the openssl .tsr receipt checks: no openssl on PATH", flush=True)


def test_commit_identity(c):
    """F1: only the freeze commit or a section 14 commit; a section 14 commit changes only the files its row names; the
    manifest lines hold at the freeze commit."""
    # A1 (reviewer): a branch off F changes reward_judge.py; the working tree holds that file
    c.git("checkout", "-q", "-B", "pkgfix", c.F)
    rj = c.clone / "flightrecorder/repro/reward_judge.py"
    rj.write_bytes(rj.read_bytes() + b"\n# post-freeze change to the reward module (test)\nPOST_FREEZE = True\n")
    c.git("commit", "-q", "-am", "post-freeze change to reward_judge.py")
    f2 = c.git("rev-parse", "HEAD")
    c.use("good")
    c.git("checkout", f2, "--", "flightrecorder/repro/reward_judge.py")
    try:
        rc, out = c.launcher("pilot", "--commit=" + f2)
        ref = refusals(out)
        check("commit identity A1: a commit off the freeze commit that changes reward_judge.py is refused, and that is "
              "the only refusal", rc == 0 and len(ref) == 1 and "commit %s is neither the freeze commit" % f2[:12]
              in ref[0], "\n".join(ref) or out[-2000:])
        refuse(c, "A1 submit at the unregistered commit", ("pilot", "--submit", "--commit=" + f2),
               "commit %s is neither the freeze commit" % f2[:12])
        check("A1: no launch record", c.launches() is None)
    finally:
        c.git("checkout", "good", "--", "flightrecorder/repro/reward_judge.py")
    refuse(c, "no freeze commit in section 13 ('Commit hash' none yet)", ("pilot", "--submit", "--commit=" + c.F),
           "section 13 'Commit hash' is not filled at HEAD", "nofreeze")
    # A2 (reviewer): linear history; a later commit P changes reward_judge.py and the template
    c.git("checkout", "-q", "-B", "linear", "good")
    tpl = c.clone / "scripts/e1b_job.py"
    for p in (rj, tpl):
        p.write_bytes(p.read_bytes() + b"\n# post-freeze fix (test)\n")
    c.git("commit", "-q", "-am", "post-freeze fix in reward_judge.py and e1b_job.py")
    P = c.git("rev-parse", "HEAD")
    rc, out = c.launcher("pilot")
    ref = refusals(out)
    check("commit identity A2: a later commit (HEAD) that section 14 does not log is refused, the only refusal",
          rc == 0 and len(ref) == 1 and "commit %s is neither the freeze commit" % P[:12] in ref[0],
          "\n".join(ref) or out[-2000:])
    rows = [("fix (test), no file named", None),
            ("fix in flightrecorder/repro/reward_judge.py (test)", "scripts/e1b_job.py"),
            ("fix in flightrecorder/repro/reward_judge.py and `scripts/e1b_job.py` (test)", "")]
    keep = c.prereg().read_bytes()
    for i, (what, unnamed) in enumerate(rows):
        c.prereg().write_bytes(keep)
        c.add_row14(P[:12], what)
        c.git("commit", "-q", "-am", "section 14 row %d for %s" % (i, P[:12]))
        rc, out = c.launcher("pilot", "--commit=" + P)
        ref = refusals(out)
        if unnamed is None:
            check("section 14 commit whose row names no file: refused for both changed files",
                  len(ref) == 1 and "differs from the freeze commit %s in flightrecorder/repro/reward_judge.py, "
                  "scripts/e1b_job.py, which its section 14 row does not name" % c.F[:12] in ref[0],
                  "\n".join(ref) or out[-2000:])
        elif unnamed:
            check("section 14 commit whose row names reward_judge.py only: refused for scripts/e1b_job.py only",
                  len(ref) == 1 and "in scripts/e1b_job.py, which its section 14 row" in ref[0]
                  and "reward_judge" not in ref[0], "\n".join(ref) or out[-2000:])
        else:
            check("section 14 commit whose row names both files: accepted; the manifest lines are checked at the "
                  "freeze commit (the template's line differs at P)",
                  rc == 0 and not ref and "is a section 14 commit; against the freeze commit %s it changes: "
                  "flightrecorder/repro/reward_judge.py, scripts/e1b_job.py" % c.F[:12] in out
                  and "baked 6 job(s) from %s" % P[:12] in out, "\n".join(ref) or out[-2500:])
    c.use("good")


def test_refusals(c):
    sub = ("pilot", "--submit", "--commit=" + c.F)
    refuse(c, "no --commit", ("pilot", "--submit"), "REFUSE: --commit is required with --submit")
    modified(c, "flightrecorder/repro/reward_testhack.py", "a changed file of the job's import closure")
    modified(c, "flightrecorder/repro/reward_judge.py", "a changed reward_judge.py")
    modified(c, "scripts/e1b_job.py", "a changed template")
    modified(c, "experiments/e1b_uv.lock", "a changed lock")
    modified(c, "flightrecorder/repro/twin_gate.py", "a changed gate")
    modified(c, "scripts/launch_e1b_aws.py", "a changed launcher")
    modified(c, "scripts/e1b_autorun.py", "a changed autorun")
    modified(c, "experiments/e1b_registry.py", "a changed registry (the spend rule)")
    refuse(c, "no manifest", sub, "experiments/e1b_freeze_manifest.txt is not committed at HEAD", "nomanifest")
    (c.clone / "experiments/e1b_freeze_manifest.txt").write_bytes(c.manifest())
    refuse(c, "a manifest in the working tree only", sub, "experiments/e1b_freeze_manifest.txt is not committed at HEAD",
           "nomanifest")
    (c.clone / "experiments/e1b_freeze_manifest.txt").unlink()
    refuse(c, "a manifest line that does not match the blob at the freeze commit", sub,
           "differs from its manifest line at the freeze commit", "tampered")
    refuse(c, "a manifest without every section 13 file", sub, "manifest lacks section 13 file(s)", "short")
    refuse(c, "no timestamp receipt", sub, "no timestamp receipt committed at HEAD", "noreceipt")
    refuse(c, "human reader 'none yet'", sub, "section 13 'Human reader before the freeze' says 'none yet'", "noreader")
    refuse(c, "pilot approval pending", sub, "section 12 Pilot row: 'Approved by Aarav' says 'pending'", "pending")
    refuse(c, "primary approval pending (main)", ("main", "--submit", "--commit=" + c.F, "--max-runs=2"),
           "section 12 Primary row: 'Approved by Aarav' says 'pending'", "pilotonly")
    out = refuse(c, "reader 'TBD' and pilot approval 'not yet approved' (the reviewer's case D)", sub,
                 "section 13 'Human reader before the freeze' says 'TBD'", "words")
    check("case D: the approval 'not yet approved' is refused too",
          "section 12 Pilot row: 'Approved by Aarav' says 'not'" in out, out[-2000:])
    # a held lock: another --submit is running (or was killed)
    c.use("good")
    c.lock().parent.mkdir(parents=True, exist_ok=True)
    c.lock().write_text("pid 1, since 2026-10-25T00:00:00Z\n")
    refuse(c, "runs/e1b/launch.lock exists", sub, "REFUSE: runs/e1b/launch.lock exists (pid 1")
    check("a lock this launcher did not create is left in place", c.lock().exists())
    c.lock().unlink()
    # pilot rerun before seed 299 was launched; then the pilot ceiling
    refuse(c, "pilot rerun before seed 299 ran", ("pilot", "--rerun=1"), "before every pilot run at seed 299")
    launches = c.runs() / "launches.json"
    launches.parent.mkdir(parents=True, exist_ok=True)
    launches.write_text(json.dumps([rec("pilot", a, t, 299, "i-oldpilot", "g6e.4xlarge") for a in ARMS for t in TASKS]))
    refuse(c, "pilot ceiling (seed 299 on g6e.4xlarge, then a 6-run rerun)", ("pilot", "--rerun=1", "--submit",
                                                                              "--commit=" + c.F), "Pilot ceiling")
    # LAUNCH_CAP: 60 launches of arm judge with this one
    reps = [("judge", t, s) for t in TASKS for s in range(316, 320)]
    launches.write_text(json.dumps(main_records() + [rec("main_replace", a, t, s, "i-oldrep") for a, t, s in reps]))
    (c.runs() / "main_replace.json").write_text(json.dumps([list(r) for r in reps] + [["judge", "second_largest", 320]]))
    refuse(c, "LAUNCH_CAP (61 launches of arm judge)", ("main_replace", "--submit", "--commit=" + c.F),
           "LAUNCH_CAP: 61 launches in arm judge")
    # 14 replacements in all
    reps = [(a, "count_vowels", s) for a in ARMS for s in range(316, 323)]
    launches.write_text(json.dumps(main_records() + [rec("main_replace", a, t, s, "i-oldrep") for a, t, s in reps]))
    (c.runs() / "main_replace.json").write_text(json.dumps([["judge", "is_prime", 316]]))
    refuse(c, "a 15th replacement", ("main_replace", "--submit", "--commit=" + c.F),
           "15 replacements with this launch; at most 14 in all")
    launches.unlink()
    (c.runs() / "main_replace.json").write_text(json.dumps([["judge", "count_vowels", s] for s in range(316, 328)] +
                                                           [{"arm": "judge", "task": "is_prime", "seed": s}
                                                            for s in (316, 317, 318)]))
    refuse(c, "a replacement list of 15", ("main_replace",), "lists 15 replacements; at most 14 in all")
    (c.runs() / "main_replace.json").write_text(json.dumps([["judge", "is_prime", 317]]))
    refuse(c, "a replacement seed that skips 316", ("main_replace",), "the next unused seed for that arm and task is 316")
    (c.runs() / "main_replace.json").unlink()
    # read-only AWS checks. The real policy allows e1/* only: E1b's e1/e1b/ root lies inside it.
    c.config(policy_resources=["arn:aws:s3:::%s/e1b/*" % BUCKET, "arn:aws:s3:::%s/other/*" % BUCKET])
    refuse(c, "an instance role whose policy does not cover e1/*", sub,
           "does not allow s3:GetObject on arn:aws:s3:::%s/e1/e1b/pilot/" % BUCKET, local=False)
    c.config()
    c.s3put("%s/pilot/e1b_judge_count_vowels_s299/status.json" % S3R, b'{"run_id": "e1b_judge_count_vowels_s299"}')
    refuse(c, "a run whose S3 prefix already has a status.json", sub,
           "e1/e1b/pilot/e1b_judge_count_vowels_s299/status.json already exists", local=False)
    shutil.rmtree(c.state / "s3")
    check("refusals leave no launch record and no lock", c.launches() is None and not c.lock().exists())


def test_cell_units(c):
    """F9 on the cell functions directly."""
    q1, q2 = chr(0x201c), chr(0x201d)
    reader = ["Aarav, 2026-10-24, sections 1-12", "Aarav, sections 1-12", "Aarav, 2026-10-24", "TBD",
              "awaiting Aarav, 2026-10-24, sections 1-12", "none yet", "Aarav 2026-02-30 sections 1-12", "",
              "Aarav, 2026-10-24, Section 12 not read"]
    approval = ['"go" (2026-10-24)', "go 2026-10-24", '"go"', '"go" 2026-10-24, TBD', q1 + "go" + q2 + " 2026-10-24",
                "pending", "not yet approved", '"" 2026-10-24', "awaiting 2026-10-24 \"go\""]
    code = ("import json, sys; sys.path.insert(0, 'scripts'); import launch_e1b_aws as L\n"
            "a = json.loads(open(sys.argv[1], encoding='utf-8').read())\n"
            "print(json.dumps([[L.reader_problem(x) for x in a[0]], [L.approval_problem(x) for x in a[1]]]))\n")
    f = c.state / "cells.json"
    f.write_text(json.dumps([reader, approval]), encoding="utf-8")
    rc, out = c.pyc(code, str(f))
    try:
        r, a = json.loads(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        r, a = [None] * len(reader), [None] * len(approval)
    want_r = [None, "no YYYY-MM-DD date", "no word 'section'", "says 'TBD'", "says 'awaiting'", "says 'none yet'",
              "no YYYY-MM-DD date", "is empty", "says 'not'"]
    want_a = [None, "no quoted words", "no YYYY-MM-DD date", "says 'TBD'", None, "says 'pending'", "says 'not'",
              "no quoted words", "says 'awaiting'"]
    ok = all((w is None and g is None) or (w is not None and g is not None and w in g) for w, g in zip(want_r, r)) \
        and all((w is None and g is None) or (w is not None and g is not None and w in g) for w, g in zip(want_a, a))
    check("cells: the reader needs a date and 'section', the approval a date and quoted words; 'not', 'TBD', "
          "'awaiting', 'pending', 'none yet' refuse; curly quotes count", rc == 0 and ok,
          json.dumps(list(zip(reader, r))) + "\n" + json.dumps(list(zip(approval, a))))


def test_measured_spend(c):
    """F7: committed spend per instance is the larger of plan and measured hours; a submit measures first."""
    c.use("good")
    launches = c.runs() / "launches.json"
    launches.parent.mkdir(parents=True, exist_ok=True)
    recs = main_records()
    launches.write_text(json.dumps(recs))
    (c.runs() / "main_replace.json").write_text(json.dumps([["judge", "count_vowels", 316]]))
    iids = sorted({r["instance_id"] for r in recs})
    c.config(states={i: "terminated" for i in iids}, launch_times={i: "2026-10-20T00:00:00+00:00" for i in iids},
             transitions={i: "User initiated (2026-10-21 00:00:00 GMT)" for i in iids})
    rc, out = c.launcher("main_replace", "--commit=" + c.F)
    check("measured spend: with nothing measured, the dry run prices plan hours (USD 248.13 committed) and allows "
          "one replacement", rc == 0 and "would REFUSE" not in out and "committed USD 248.13" in out, out[-2500:])
    c.clear_calls()
    rc, out = c.launcher("main_replace", "--submit", "--commit=" + c.F)
    hours = c.hours() or {}
    check("measured spend: 8 main instances measured at 24 h each (USD 357.31) refuse the replacement the plan alone "
          "would allow", rc != 0 and "Primary ceiling: committed USD 357.31" in out and no_spend(c.calls()) and
          hours == {i: {"hours": 24.0, "final": True} for i in iids} and not c.lock().exists(), (hours, out[-2500:]))
    rc, out = c.launcher("main_replace", "--commit=" + c.F)
    check("measured spend: the dry run then reads runs/e1b/instance_hours.json and would refuse",
          rc == 0 and "--submit would REFUSE: Primary ceiling: committed USD 357.31" in out, out[-2500:])
    code = ("import json, sys; from datetime import datetime, timezone as tz; sys.path.insert(0, 'scripts')\n"
            "import launch_e1b_aws as L\n"
            "t = lambda *a: datetime(*a, tzinfo=tz.utc)\n"
            "recs = [{'instance_id': i, 'submitted_utc': '2026-10-20T00:00:00Z'} for i, n in\n"
            "        (('i-run', 12), ('i-gone', 12), ('i-term', 6), ('i-noend', 1), ('i-unk', 1), ('i-fin', 1))\n"
            "        for _ in range(n)]\n"
            "info = {'i-run': {'state': 'running', 'launch': t(2026, 10, 25, 10), 'end': None},\n"
            "        'i-gone': {'state': 'gone', 'launch': None, 'end': None},\n"
            "        'i-term': {'state': 'terminated', 'launch': t(2026, 10, 24, 0), 'end': t(2026, 10, 24, 9, 30)},\n"
            "        'i-noend': {'state': 'terminated', 'launch': t(2026, 10, 25, 11), 'end': None},\n"
            "        'i-unk': {'state': 'unknown', 'launch': None, 'end': None},\n"
            "        'i-fin': {'state': 'running', 'launch': t(2026, 10, 25, 0), 'end': None}}\n"
            "prev = {'i-unk': {'hours': 3.0, 'final': False}, 'i-fin': {'hours': 5.0, 'final': True}}\n"
            "print(json.dumps(L.measure_hours(recs, info, prev, t(2026, 10, 25, 12)), sort_keys=True))\n")
    rc, out = c.pyc(code)
    try:
        got = json.loads(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        got = {}
    want = {"i-run": {"hours": 2.0, "final": False}, "i-gone": {"hours": 36.75, "final": True},
            "i-term": {"hours": 9.5, "final": True}, "i-noend": {"hours": 1.0, "final": True},
            "i-unk": {"hours": 3.0, "final": False}, "i-fin": {"hours": 5.0, "final": True}}
    check("measured hours: running to now; terminated to its time; down without a time bounded by the backstop "
          "(12 x 180 + 30 + 15 min = 36.75 h); unknown and final kept", got == want, (got, out[-1500:]))
    launches.unlink()
    (c.runs() / "main_replace.json").unlink()
    (c.runs() / "instance_hours.json").unlink()
    c.config()


def test_submit_pilot(c):
    c.use("good")
    c.config(capacity_fail=[["us-east-1", "g6e.xlarge"]])
    c.clear_calls()
    rc, out = c.launcher("pilot", "--submit", "--commit=" + c.F)
    recs = c.launches() or []
    check("submit pilot (every guard satisfied, the role allowing e1/* only): exit 0, one batch launched on the "
          "fallback type", rc == 0 and "LAUNCHED batch 0 on i-fake000000000001 us-east-1" in out and "g6e.2xlarge" in out,
          out)
    check("launch records: 6, exactly build_contract section 3's keys (job_sha256 last)",
          len(recs) == 6 and all(list(r) == RECORD_KEYS for r in recs), recs[:1])
    check("the launch lock is removed after the submit", not c.lock().exists())
    rc2, bo = run([PY, "-B", "scripts/e1b_job.py", "--bake", "judgegate", "is_prime", "299", "pilot", c.F,
                   str(c.root / "bake_check")], c.clone)
    prov = json.loads(bo[:bo.rindex("}") + 1]) if rc2 == 0 else {}
    r = next((x for x in recs if x["run_id"] == "e1b_judgegate_is_prime_s299"), {})
    indep = c.root / "bake_check" / "e1b_judgegate_is_prime_s299" / "job.py"
    indep_sha = hashlib.sha256(indep.read_bytes()).hexdigest() if indep.exists() else None
    check("launch record provenance equals an independent bake at F (job_sha256 included)",
          prov and all(r.get(k) == prov[k] for k in ("git_commit", "package_sha256", "template_sha256",
                                                     "lock_sha256", "gate_sha256")) and r.get("git_commit") == c.F
          and r.get("job_sha256") == indep_sha and prov.get("job_sha256", indep_sha) == indep_sha, (prov, r, indep_sha))
    check("launch record job_sha256 is the SHA-256 of the job.py uploaded to its prefix",
          len(recs) == 6 and all(x["job_sha256"] == hashlib.sha256(c.s3p("%s/pilot/%s/job.py" % (S3R, x["run_id"]))
                                                                   .read_bytes()).hexdigest() for x in recs), recs[:1])
    check("launch record fields: stage, seed, region, AMI, type, market, s3 under e1/e1b",
          len(recs) == 6 and all(x["stage"] == "pilot" and x["seed"] == 299 and x["region"] == "us-east-1" and
              x["ami"] == "ami-00803022f231f100e" and x["instance_type"] == "g6e.2xlarge" and
              x["market"] == "ondemand" and x["s3"] == "s3://%s/e1/e1b/pilot/%s" % (BUCKET, x["run_id"]) for x in recs),
          recs[:1])
    lock = subprocess.run(["git", "cat-file", "blob", "%s:experiments/e1b_uv.lock" % c.F], cwd=str(c.clone),
                          capture_output=True).stdout
    ok = len(recs) == 6 and all(c.s3p("%s/pilot/%s/job.py.lock" % (S3R, x["run_id"])).read_bytes() == lock and
                                '"pilot", "%s"\n' % x["run_id"] in c.s3p("%s/pilot/%s/job.py" % (S3R, x["run_id"]))
                                .read_text() for x in recs)
    check("uploads: job.py and the unmodified lock as job.py.lock under every run's e1/e1b prefix", ok)
    calls = c.calls()
    ri = [x for x in calls if x[:2] == ["ec2", "run-instances"]]
    check("capacity fallback: 6 us-east-1 g6e.xlarge misses (one per AZ), then g6e.2xlarge",
          len(ri) == 7, len(ri))
    first_spend = next(i for i, x in enumerate(calls) if x[:2] == ["s3", "cp"])
    check("the IAM and S3 checks run before any upload",
          {x[1] for x in calls[:first_spend] if x[0] == "iam"} >= {"get-instance-profile", "get-role-policy"} and
          any(x[:2] == ["s3", "ls"] and x[2] == "s3://%s/e1/e1b/pilot/" % BUCKET for x in calls[:first_spend]),
          calls[:first_spend])
    iid = "i-fake000000000001"
    inst = json.loads((c.state / "instances.json").read_text())[iid]["spec"]
    check("instance spec: pinned AMI, role, terminate on shutdown, IMDSv2, 100 GB gp3, no spot",
          inst["ImageId"] == "ami-00803022f231f100e" and inst["IamInstanceProfile"] == {"Name": "fr-e1-runner"}
          and inst["InstanceInitiatedShutdownBehavior"] == "terminate" and
          inst["MetadataOptions"] == {"HttpTokens": "required"} and "InstanceMarketOptions" not in inst and
          inst["BlockDeviceMappings"][0]["Ebs"]["VolumeSize"] == 100, inst)
    ud = (c.state / "userdata" / (iid + ".sh")).read_bytes()
    rids = " ".join(x["run_id"] for x in recs)
    want = ["shutdown -h +1110", 'W="/opt/e1b"', 'RES="/tmp/result.json"',
            'S3="s3://%s/e1/e1b/pilot"' % BUCKET, 'RUNS="%s"' % rids,
            "uv==0.12.23 --hash=sha256:565c6e2874dbeae86c02f3dea97255e878fec672659a73d4930c6b93fcab2fff",
            "--only-binary :all: --require-hashes",
            'timeout --kill-after=120 180m python3 -m uv run --no-config --locked --python 3.12.15 --managed-python '
            'job.py > "$W/run/job.log" 2>&1 &', "-gt 2700 ]", "sleep 60 & wait", "-ge 300 ]",
            "export HF_HOME=\"$W/hf\" UV_CACHE_DIR=\"$W/uvcache\" PYTHONUNBUFFERED=1"]
    missing = [w for w in want if w.encode() not in ud]
    check("user data: backstop, e1/e1b prefix, pinned uv with hash, locked uv run, per-run timeout, watchdog, LF, "
          "ASCII", not missing and b"\r" not in ud and max(ud) < 128, missing)
    rc, bo = run([c.bash, "-n", (c.state / "userdata" / (iid + ".sh")).as_posix()], c.root)
    check("user data: bash -n syntax check", rc == 0, bo)
    code = ("import sys; sys.path.insert(0, 'scripts'); import launch_e1b_aws as L; "
            "sys.stdout.write(L.render('pilot', %r))" % rids.split())
    rc, rendered = run([PY, "-B", "-c", code], c.clone)
    check("the uploaded user data is exactly render() with the production values", rendered.encode() == ud)
    c.clear_calls()
    rc, out = c.launcher("pilot", "--submit", "--commit=" + c.F)
    check("submit again: every run skipped as already launched, nothing launched",
          rc == 0 and out.count("skip (already launched)") == 6 and "NOTHING TO LAUNCH" in out and c.calls() == [],
          out)
    c.config()
    return recs


def test_status_and_refused_collect(c, recs):
    rid = [x["run_id"] for x in recs]
    c.s3put("%s/pilot/%s/status.json" % (S3R, rid[0]), json.dumps(
        {"run_id": rid[0], "step": 41, "rc": None, "ts": "2026-10-23T12:30:00Z", "extra": "VALUE_NOT_TO_PRINT"}))
    c.s3put("%s/pilot/%s/status.json" % (S3R, rid[1]), json.dumps(
        {"run_id": rid[1], "step": 399, "rc": 0, "ts": "2026-10-23T13:00:00Z"}))
    c.s3put("%s/pilot/%s/status.json" % (S3R, rid[2]), json.dumps({"run_id": "someone_else", "step": 3, "rc": None}))
    c.s3put("%s/pilot/%s/result.json" % (S3R, rid[1]), b'{"proxy": 0.987654}')
    c.s3put("%s/pilot/%s/job.log" % (S3R, rid[1]), b"STEP 0\nproxy 0.987654\n")
    c.clear_calls()
    rc, out = c.launcher("pilot", "--status")
    calls = c.calls()
    check("status: one row per run with step and rc, malformed file flagged, extra fields not printed",
          rc == 0 and re.search(r"%s\s+i-fake000000000001\s+running\s+42/400\s+-" % rid[0], out) and
          re.search(r"%s\s+\S+\s+running\s+400/400\s+0 " % rid[1], out) and "malformed" in out and
          "VALUE_NOT_TO_PRINT" not in out and "finished rc=0 1" in out, out)
    check("status: reads only status.json files and instance state",
          calls and all(x[:2] == ["ec2", "describe-instances"] or (x[:2] == ["s3", "cp"] and
                                                                  x[2].endswith("/status.json") and x[3] == "-")
                        for x in calls) and not reads_outputs(calls), calls)
    h = (c.hours() or {}).get("i-fake000000000001")
    check("status: records the running instance's hours (launch to now, not final) in runs/e1b/instance_hours.json",
          isinstance(h, dict) and h.get("final") is False and 0.0 <= h.get("hours", -1) < 1.0 and
          re.search(r"instance i-fake000000000001\s+running\s+\d+\.\d\d h so far", out), (h, out[-800:]))
    c.clear_calls()
    rc, out = c.launcher("pilot", "--collect")
    check("collect before every final rc: refused, nothing read or written",
          rc != 0 and "REFUSE: 5 of 6 launched runs of pilot have no final rc yet" in out and
          not reads_outputs(c.calls()) and not (c.runs() / "pilot").exists(), out)
    for x in rid[:3]:
        for name in ("status.json", "result.json", "job.log"):
            c.s3p("%s/pilot/%s/%s" % (S3R, x, name)).unlink(missing_ok=True)


def run_userdata(c, stage, rids, tag, plan, hang=(), signal_at=None):
    """Render the user data with short test timings, run it under bash with the fake commands first on PATH."""
    wd, res = (c.state / ("inst_" + tag)).as_posix(), (c.state / ("result_" + tag + ".json")).as_posix()
    code = ("import sys; sys.path.insert(0, 'scripts'); import launch_e1b_aws as L; "
            "sys.stdout.write(L.render(%r, %r, workdir=%r, result_path=%r, per_run_min=1, kill_after_s=5, "
            "stall_s=4, tick_s=1, push_s=2, backstop=999))" % (stage, rids, wd, res))
    rc, ud = run([PY, "-B", "-c", code], c.clone)
    path = c.state / ("ud_%s.sh" % tag)
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write(ud)
    env = c.env({"FAKE_JOB_PLAN": json.dumps(plan), "FAKE_RESULT_PATH": res, "FAKE_HANG": " ".join(hang)},
                path_first=str(c.bin_sh) + os.pathsep + str(Path(c.bash).parent))
    t0 = time.time()
    if signal_at is None:
        rc, out = run([c.bash, path.as_posix()], c.root, env, timeout=600)
    else:
        pidf = (c.state / ("pid_" + tag)).as_posix()
        p = subprocess.Popen([c.bash, "-c", 'bash "$0" & echo $! > "$1"; wait', path.as_posix(), pidf],
                             cwd=str(c.root), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        key, step = signal_at
        deadline = time.time() + 120
        while time.time() < deadline:
            st = c.s3json("%s/%s/%s/status.json" % (S3R, stage, key))
            if st and isinstance(st.get("step"), int) and st["step"] >= step:
                break
            time.sleep(0.5)
        subprocess.run([c.bash, "-c", 'kill -TERM "$(cat "$0")"', pidf], env=env)
        out = p.communicate(timeout=300)[0].decode("utf-8", "replace")
        rc = p.returncode
    boot = Path(wd, "boot.log").read_text(encoding="utf-8", errors="replace") if Path(wd, "boot.log").exists() else ""
    return rc, out, boot, time.time() - t0


def test_instance_and_collect(c, recs):
    rid = [x["run_id"] for x in recs]
    key = lambda x, name: "%s/pilot/%s/%s" % (S3R, x, name)  # noqa: E731
    (c.state / "shutdown.log").unlink(missing_ok=True)
    (c.state / "jobcalls.jsonl").unlink(missing_ok=True)
    c.s3p(key(rid[5], "job.py")).unlink()          # run 6: no job.py -> "no-job"
    plan = {rid[0]: "ok", rid[1]: "fail", rid[3]: "ok", rid[4]: "noresult"}
    c.clear_calls()
    rc, out, boot, secs = run_userdata(c, "pilot", rid, "pilot", plan, hang=[rid[2]])
    st = {x: c.s3json(key(x, "status.json")) for x in rid}
    check("user data run (%.0f s): final status of each run: rc 0, 1, stalled, 0, 0, no-job" % secs,
          [st[x] and st[x]["rc"] for x in rid] == [0, 1, "stalled", 0, 0, "no-job"] and
          all(list(st[x]) == ["run_id", "step", "rc", "ts"] for x in rid), (st, boot[-3000:]))
    check("user data run: the job's last step is kept in the final status (2, 0, none)",
          [st[x]["step"] for x in rid[:3]] == [2, 0, None], st)
    check("user data run: result.json for the runs that wrote one; job.log for every run that started; "
          "boot.log under every prefix",
          [c.s3p(key(x, "result.json")).exists() for x in rid] == [True, False, False, True, False, False]
          and [c.s3p(key(x, "job.log")).exists() for x in rid] == [True] * 5 + [False]
          and all(c.s3p(key(x, "boot.log")).exists() for x in rid), boot[-2000:])
    calls = c.calls()
    idx = [i for i, x in enumerate(calls) if x[:2] == ["s3", "cp"] and len(x) > 3 and x[3].startswith("s3://")
           and "/%s/" % rid[0] in x[3]]
    last_status = max(i for i in idx if calls[i][3].endswith("status.json"))
    res_up = [i for i in idx if calls[i][3].endswith("result.json")]
    check("user data run: result.json and job.log are uploaded before the final status.json, under e1/e1b",
          res_up and res_up[0] < last_status and all("/e1/e1b/pilot/" in calls[i][3] for i in idx) and
          max(i for i in idx if calls[i][3].endswith("job.log")) < last_status, [calls[i][3] for i in idx])
    jc = [json.loads(x) for x in (c.state / "jobcalls.jsonl").read_text().splitlines()]
    check("user data run: the job runs as `-m uv run --no-config --locked --python 3.12.15 --managed-python job.py` "
          "with job.py.lock beside it and E1B_STATUS_PATH set",
          len(jc) == 4 and all(j["argv"] == ["-m", "uv", "run", "--no-config", "--locked", "--python", "3.12.15",
                                              "--managed-python", "job.py"] and "job.py" in j["files"] and
                               "job.py.lock" in j["files"] and j["status_path"].endswith("/run/status.json")
                               and j["unbuffered"] == "1" for j in jc), jc)
    shut = (c.state / "shutdown.log").read_text().split("\n") if (c.state / "shutdown.log").exists() else []
    check("user data run: backstop shutdown at boot, shutdown now at the end", shut[:2] == ["-h +999", "-h now"], shut)
    check("user data run: watchdog stopped the hung run; boot.log holds no job output",
          "watchdog: status.json unchanged" in boot and "STEP" not in boot, boot[-2000:])
    jl = c.s3p(key(rid[0], "job.log")).read_text()
    check("user data run: job.log holds the job's STEP lines", "STEP 0\nSTEP 1\nSTEP 2\n" in jl.replace("\r", ""), jl)
    # collect
    iid = "i-fake000000000001"
    c.config(states={iid: "terminated"}, launch_times={iid: "2026-10-23T10:00:00+00:00"},
             transitions={iid: "User initiated (2026-10-23 12:30:00 GMT)"})
    c.clear_calls()
    rc, out = c.launcher("pilot", "--collect")
    pdir = c.runs() / "pilot"
    man = (pdir / "MANIFEST.sha256").read_bytes() if (pdir / "MANIFEST.sha256").exists() else b""
    want = "".join("%s  %s\n" % (hashlib.sha256((pdir / n).read_bytes()).hexdigest(), n)
                   for n in sorted([rid[0] + ".json", rid[3] + ".json"]) if (pdir / n).exists())
    check("collect after the final rc: exit 0, results, logs, boot logs and status files at the contract paths",
          rc == 0 and sorted(p.name for p in pdir.glob("*.json")) == sorted([rid[0] + ".json", rid[3] + ".json"])
          and sorted(p.name for p in (pdir / "logs").glob("*.log")) ==
          sorted([x + ".log" for x in rid[:5]] + [x + ".boot.log" for x in rid])
          and sorted(p.name for p in (pdir / "status").glob("*.json")) == sorted(x + ".json" for x in rid), out)
    check("collect: MANIFEST.sha256 = '<sha256>  <file name>' per result file, sorted, LF",
          man.decode() == want and want.count("\n") == 2 and b"\r" not in man, (man, want))
    check("collect: prints no value from a result or log", '"fake"' not in out and "true" not in out.lower()
          and "STEP" not in out, out)
    check("collect: the terminated instance's hours are final, launch to its termination time (2.5 h)",
          (c.hours() or {}).get(iid) == {"hours": 2.5, "final": True}, c.hours())
    rc, out2 = c.launcher("pilot", "--collect")
    check("collect again: exit 0, manifest unchanged", rc == 0 and (pdir / "MANIFEST.sha256").read_bytes() == man, out2)
    f = pdir / (rid[0] + ".json")
    keep = f.read_bytes()
    f.write_bytes(keep + b" ")
    rc, out3 = c.launcher("pilot", "--collect")
    check("collect never replaces a collected result with different bytes",
          rc != 0 and "differ from S3, left unchanged" in out3 and f.read_bytes() == keep + b" " and
          (pdir / "MANIFEST.sha256").read_bytes() == man, out3)
    f.write_bytes(keep)
    rc, out = c.launcher("pilot", "--status")
    check("status after the run: terminated, final rc per run", rc == 0 and "terminated" in out and "stalled" in out
          and "no-job" in out and "finished rc=0 3, finished rc!=0 3" in out and "2.50 h (final)" in out, out)
    # audit trail: raw events only to gitignored paper/freeze/, a redacted copy to runs/e1b/audit/
    gi = c.clone / ".gitignore"
    keep_gi = gi.read_bytes()
    gi.write_bytes(b"\n".join(ln for ln in keep_gi.split(b"\n") if ln.strip() != b"paper/") + b"\n")
    try:
        c.clear_calls()
        rc, out = c.launcher("pilot", "--audit-trail")
        check("audit trail: refused when git does not ignore paper/freeze/e1b_pilot/cloudtrail.json; nothing written",
              rc != 0 and "is not ignored by git" in out and c.calls() == [] and not (c.clone / "paper").exists()
              and not (c.runs() / "audit").exists(), out)
    finally:
        gi.write_bytes(keep_gi)
    c.clear_calls()
    rc, out = c.launcher("pilot", "--audit-trail")
    raw_p = c.clone / "paper" / "freeze" / "e1b_pilot" / "cloudtrail.json"
    red_p = c.runs() / "audit" / "pilot_cloudtrail.json"
    raw = json.loads(raw_p.read_text()) if raw_p.exists() else {}
    red = json.loads(red_p.read_text()) if red_p.exists() else {}
    red_bytes = red_p.read_bytes() if red_p.exists() else b""
    ev = red.get("instances", {}).get(iid, {}).get("events", [])
    check("audit trail: raw events of the recorded instance in paper/freeze/e1b_pilot/cloudtrail.json (gitignored)",
          rc == 0 and list(raw.get("instances", {})) == [iid] and len(raw["instances"][iid]["events"]) == 3 and
          b"FAKE_USER_NAME" in raw_p.read_bytes() and
          run(["git", "check-ignore", "-q", "paper/freeze/e1b_pilot/cloudtrail.json"], c.clone)[0] == 0 and
          all(x[:2] == ["cloudtrail", "lookup-events"] for x in c.calls()), out)
    check("audit trail: the redacted copy in runs/e1b/audit/ keeps only EventId, EventName, EventTime, EventSource, "
          "Resources", len(ev) == 3 and all(sorted(e) == sorted(["EventId", "EventName", "EventTime", "EventSource",
                                                                  "Resources"]) for e in ev) and
          not any(s in red_bytes for s in (b"FAKE_USER_NAME", b"AKIAFAKE", b"198.51.100.7", b"userIdentity",
                                           b"CloudTrailEvent", b"arn:aws:iam")), red)
    check("audit trail: the earliest RunInstances time (compared as times, not strings) printed and kept",
          "earliest RunInstances event: 2026-10-23T12:00:05+00:00" in out and
          red.get("earliest_run_instances") == "2026-10-23T12:00:05+00:00", (out, red.get("earliest_run_instances")))
    check("audit trail: nothing but result files in runs/e1b/pilot or runs/e1b/main",
          not list(c.runs().glob("*/cloudtrail.json")) and
          sorted(p.name for p in pdir.glob("*.json")) == sorted([rid[0] + ".json", rid[3] + ".json"]), out)


def test_lost(c):
    launches = c.runs() / "launches.json"
    keep = launches.read_bytes()
    launches.write_text(json.dumps([rec("main", "judge", "count_vowels", 300, "i-gone0001")]))
    c.config(gone=["i-gone0001"])
    rc, out = c.launcher("main", "--status")
    check("lost run: an instance AWS no longer knows, with no final rc, shows LOST", rc == 0 and
          re.search(r"e1b_judge_count_vowels_s300\s+i-gone0001\s+gone\s+-\s+LOST", out), out)
    check("lost run: its hours are bounded by the backstop (1 x 180 + 30 + 15 min = 3.75 h), final",
          (c.hours() or {}).get("i-gone0001") == {"hours": 3.75, "final": True}, c.hours())
    rc, out = c.launcher("main", "--collect")
    check("lost run: collect proceeds, reports it, and writes an empty manifest", rc == 0 and "LOST: instance gone" in out
          and (c.runs() / "main" / "MANIFEST.sha256").read_bytes() == b"", out)
    launches.write_bytes(keep)
    shutil.rmtree(c.runs() / "main")
    c.config()


def test_submit_main(c):
    c.clear_calls()
    rc, out = c.launcher("main", "--submit", "--commit=" + c.F, "--max-runs=24")
    recs = [x for x in (c.launches() or []) if x["stage"] == "main"]
    insts = json.loads((c.state / "instances.json").read_text())
    new = [k for k in insts if k != "i-fake000000000001"]
    uds = [(c.state / "userdata" / (k + ".sh")).read_text() for k in new]
    check("submit main --max-runs=24: two instances of 12 runs, seeds 300-303, backstop 2190 min",
          rc == 0 and out.count("LAUNCHED batch") == 2 and len(recs) == 24 and len(new) == 2 and
          sorted({x["seed"] for x in recs}) == [300, 301, 302, 303] and
          all("shutdown -h +2190" in u and len(re.search(r'RUNS="([^"]*)"', u).group(1).split()) == 12 for u in uds)
          and [insts[k]["spec"]["TagSpecifications"][0]["Tags"][0]["Value"] for k in new] ==
          ["fr-e1b-main-0", "fr-e1b-main-1"], out)
    check("submit main: committed spend priced per instance (the plan notes it)", "Primary ceiling USD 285" in out, out)
    return recs


def test_signal_path(c):
    rids = ["e1b_judge_count_vowels_s%d" % s for s in (316, 317, 318)]
    for i, r in enumerate(rids):
        c.s3put("%s/main_replace/%s/job.py" % (S3R, r),
                'ARM, TASK, SEED, STAGE, RUN_ID = "judge", "count_vowels", %d, "main_replace", "%s"\n' % (316 + i, r))
        c.s3put("%s/main_replace/%s/job.py.lock" % (S3R, r), b"lock\n")
    (c.state / "shutdown.log").unlink(missing_ok=True)
    rc, out, boot, secs = run_userdata(c, "main_replace", rids, "signal", {rids[0]: "ok", rids[1]: "slowok"},
                                       signal_at=(rids[1], 2))
    st = [c.s3json("%s/main_replace/%s/status.json" % (S3R, r)) for r in rids]
    check("signal during a run (%.0f s): rc 0, 'interrupted' (its step kept), 'not-run'; boot.log everywhere" % secs,
          [s and s["rc"] for s in st] == [0, "interrupted", "not-run"] and isinstance(st[1]["step"], int)
          and st[1]["step"] >= 2 and st[2]["step"] is None and
          all(c.s3p("%s/main_replace/%s/boot.log" % (S3R, r)).exists() for r in rids) and
          c.s3p("%s/main_replace/%s/job.log" % (S3R, rids[1])).exists(), (st, boot[-2500:]))
    shut = (c.state / "shutdown.log").read_text().split("\n") if (c.state / "shutdown.log").exists() else []
    check("signal: the instance still powers off", "-h now" in shut and "signal" in boot, (shut, boot[-1500:]))
    shutil.rmtree(c.state / "s3" / BUCKET / "e1" / "e1b" / "main_replace")


def autorun(c, *args, extra=None, popen=False):
    env = c.env(dict({"STUB_MARKER_DIR": str(c.state / "markers")}, **(extra or {})))
    cmd = [PY, "-B", "scripts/e1b_autorun.py", *args]
    if popen:
        return subprocess.Popen(cmd, cwd=str(c.clone), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return run(cmd, c.clone, env)


def test_autorun(c, recs):
    stop = c.state / "autorun.STOP"
    marks = c.state / "markers"

    def clear_marks():
        for m in ("exclusions.called", "analysis.called"):
            (marks / m).unlink(missing_ok=True)
    stop.write_text("stop")
    rc, out = autorun(c, "main", "--interval=1", "--stop-file=" + str(stop))
    check("autorun STOP file: exit 2 before any collection", rc == 2 and "STOP file found" in out and
          not (c.runs() / "main").exists() and not (marks / "exclusions.called").exists(), out)
    stop.unlink()
    iids = sorted({x["instance_id"] for x in recs})
    for x in recs:
        c.s3put("%s/main/%s/status.json" % (S3R, x["run_id"]),
                json.dumps({"run_id": x["run_id"], "step": 10, "rc": None, "ts": "2026-10-23T12:00:00Z"}))
    c.clear_calls()
    p = autorun(c, "main", "--interval=1", "--stop-file=" + str(stop), extra={"STUB_EXCLUSIONS_RC": "3"}, popen=True)
    time.sleep(6)
    before = c.calls()
    still = p.poll() is None
    for x in recs:
        c.s3put("%s/main/%s/result.json" % (S3R, x["run_id"]), json.dumps({"run_id": x["run_id"], "fake": True}))
        c.s3put("%s/main/%s/job.log" % (S3R, x["run_id"]), b"STEP 0\n")
        c.s3put("%s/main/%s/status.json" % (S3R, x["run_id"]),
                json.dumps({"run_id": x["run_id"], "step": 399, "rc": 0, "ts": "2026-10-23T13:00:00Z"}))
    c.config(states={i: "terminated" for i in iids})
    out = p.communicate(timeout=300)[0].decode("utf-8", "replace")
    saved = sorted((c.runs() / "autorun").glob("main_*_exclusions.txt"))
    check("autorun polls while runs are unfinished, reading only status.json and instance state",
          still and before and not reads_outputs(before) and
          all(x[:2] == ["ec2", "describe-instances"] or x[2].endswith("/status.json") for x in before), before[:5])
    check("autorun replacement needed (exit 3): collects, prints the list, exits 3, analysis not run, nothing "
          "written for the launcher", p.returncode == 3 and "REPLACEMENT NEEDED: judge is_prime seed 316" in out and
          "REPLACEMENTS NEEDED before the registered analysis (section 7): 1 listed above" in out and
          (marks / "exclusions.called").exists() and not (marks / "analysis.called").exists() and
          not (c.runs() / "main_replace.json").exists() and
          len(list((c.runs() / "main").glob("*.json"))) == 24 and
          (c.runs() / "main" / "MANIFEST.sha256").read_text().count("\n") == 24 and len(saved) == 1, out)
    clear_marks()
    rc, out = autorun(c, "main", "--interval=1", "--stop-file=" + str(stop), extra={"STUB_EXCLUSIONS_RC": "4"})
    check("autorun main runs not launched (exit 4): tells the operator to launch the remaining main runs, exits 4, "
          "analysis not run", rc == 4 and "MAIN RUNS NOT LAUNCHED: 2 listed above" in out and
          "Launch the remaining main runs: python scripts/launch_e1b_aws.py main" in out and
          "main_replace.json" not in out.split("MAIN RUNS NOT LAUNCHED")[1] and
          (marks / "exclusions.called").exists() and not (marks / "analysis.called").exists(), out)
    clear_marks()
    rc, out = autorun(c, "main", "--interval=1", "--stop-file=" + str(stop), extra={"STUB_EXCLUSIONS_RC": "1"})
    check("autorun provenance failure (exit 1): stops, exits 1, analysis not run",
          rc == 1 and "PROVENANCE FAILURE" in out and "pins do not match (stub)" in out and
          not (marks / "analysis.called").exists(), out)
    clear_marks()
    rc, out = autorun(c, "main", "--interval=1", "--stop-file=" + str(stop), extra={"STUB_EXCLUSIONS_RC": "0"})
    saved = sorted((c.runs() / "autorun").glob("main_*_primary_stdout.txt"))
    check("autorun no replacement (exit 0): runs the registered analysis, saves its stdout, exits 0",
          rc == 0 and (marks / "analysis.called").exists() and len(saved) == 1 and
          "STUB REPORT: the registered analysis ran" in saved[0].read_text() and b"\r" not in saved[0].read_bytes(),
          out)
    for rel in ("experiments/e1b_primary.py", "scripts/launch_e1b_aws.py", "scripts/e1b_autorun.py"):
        f = c.clone / rel
        keep = f.read_bytes()
        f.write_bytes(keep + b"\n# scratch modification\n")
        clear_marks()
        c.clear_calls()
        try:
            rc, out = autorun(c, "main", "--interval=1", "--stop-file=" + str(stop))
        finally:
            f.write_bytes(keep)
        check("autorun refuses a %s that differs from HEAD before polling or collecting (exit 5)" % rel,
              rc == 5 and "REFUSE: %s differs from its blob at HEAD" % rel in out and c.calls() == [] and
              not (marks / "exclusions.called").exists(), out)
    rc, out = autorun(c, "main_replace", "--interval=1", "--stop-file=" + str(stop))
    check("autorun main_replace with nothing launched there: exit 5", rc == 5 and "no launched run in stage "
          "main_replace" in out, out)


def test_concurrent_submits(c):
    """F5: two launchers started together (one per region, as the docstring's --region example invites)."""
    before = c.launches() or []
    c.config(run_delay_s=2)
    c.clear_calls()
    cmd = [PY, "-B", "scripts/launch_e1b_aws.py", "main", "--submit", "--commit=" + c.F, "--max-runs=12"]
    procs = [subprocess.Popen(cmd + [reg], cwd=str(c.clone), env=c.env(), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT) for reg in ("--region=us-east-1", "--region=us-east-2")]
    outs = [p.communicate(timeout=900)[0].decode("utf-8", "replace") for p in procs]
    rcs = [p.returncode for p in procs]
    recs = c.launches() or []
    cnt = collections.Counter(r["run_id"] for r in recs)
    won = [o for o, r in zip(outs, rcs) if r == 0 and "LAUNCHED batch 0" in o]
    lost = [o for o, r in zip(outs, rcs) if r != 0 and "REFUSE: runs/e1b/launch.lock exists" in o]
    check("two concurrent submits: one launches, the other stops on runs/e1b/launch.lock; no run launched twice",
          len(won) == 1 and len(lost) == 1 and len(recs) == len(before) + 12 and max(cnt.values()) == 1 and
          not c.lock().exists() and sum(1 for x in c.calls() if x[:2] == ["ec2", "run-instances"]) == 1,
          (rcs, [o[-700:] for o in outs], len(recs)))
    c.config()


def test_hours_concurrent(c):
    """Finding R9: runs/e1b/instance_hours.json is a read-modify-write file that several commands write (--status,
    --collect, --submit, the autorun's polls, the J3 launcher). Two launcher processes each record 120 instances of
    their own, one call at a time (describe-instances stubbed by passing its answer), started together, plus a third
    that re-measures one running instance through them; a fourth reads the file throughout. No entry may be lost, a
    final value never changes, every read is a whole JSON object holding at least the previous read's entries, and
    no lock or temporary file is left."""
    hours = c.runs() / "instance_hours.json"
    keep = hours.read_bytes() if hours.exists() else None
    hours.unlink(missing_ok=True)
    n = 120
    head = ("import json, sys, time; from datetime import datetime, timedelta, timezone; sys.path.insert(0, 'scripts')\n"
            "import launch_e1b_aws as L\nt0 = datetime(2026, 10, 25, tzinfo=timezone.utc)\n")
    writer = head + ("tag, h = sys.argv[1], float(sys.argv[2])\n"
                     "for i in range(%d):\n"
                     "    iid = 'i-%%s%%04d' %% (tag, i)\n"
                     "    L.record_hours([{'instance_id': iid, 'submitted_utc': '2026-10-25T00:00:00Z'}],\n"
                     "                   {iid: {'state': 'terminated', 'launch': t0, 'end': t0 + timedelta(hours=h)}})\n"
                     % n)
    shared = head + ("for i in range(%d):\n"
                     "    L.record_hours([{'instance_id': 'i-shared', 'submitted_utc': '2026-10-25T00:00:00Z'}],\n"
                     "                   {'i-shared': {'state': 'running', 'launch': t0, 'end': None}})\n" % (n // 2))
    reader = head + ("p, last, bad, reads = L.INSTANCE_HOURS, {}, [], 0\n"
                     "while not (p.parent / 'stop_reader').exists():\n"
                     "    try:\n"
                     "        raw = p.read_text(encoding='utf-8')\n"
                     "    except FileNotFoundError:\n"
                     "        continue\n"
                     "    except PermissionError:\n"
                     "        continue\n"
                     "    try:\n"
                     "        cur = json.loads(raw)\n"
                     "    except ValueError:\n"
                     "        bad.append('partial file'); continue\n"
                     "    if not set(last) <= set(cur):\n"
                     "        bad.append('entries vanished: %s' % sorted(set(last) - set(cur))[:3])\n"
                     "    last, reads = cur, reads + 1\n"
                     "    time.sleep(0.005)\n"
                     "print(json.dumps({'bad': bad[:5], 'reads': reads}))\n")
    stop = c.runs() / "stop_reader"
    stop.unlink(missing_ok=True)
    rd = subprocess.Popen([PY, "-B", "-c", reader], cwd=str(c.clone), env=c.env(), stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
    procs = [subprocess.Popen([PY, "-B", "-c", code, *a], cwd=str(c.clone), env=c.env(), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT)
             for code, a in ((writer, ("a", "1")), (writer, ("b", "2")), (shared, ()))]
    outs = [p.communicate(timeout=900)[0].decode("utf-8", "replace") for p in procs]
    stop.write_text("")
    rout = rd.communicate(timeout=120)[0].decode("utf-8", "replace")
    stop.unlink(missing_ok=True)
    try:
        rres = json.loads(rout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        rres = {"bad": ["reader output: %s" % rout[-300:]], "reads": 0}
    got = c.hours() or {}
    want = dict({"i-a%04d" % i: {"hours": 1.0, "final": True} for i in range(n)},
                **{"i-b%04d" % i: {"hours": 2.0, "final": True} for i in range(n)})
    lost = sorted(set(want) - set(got))
    left = sorted(x.name for x in c.runs().iterdir() if x.name.startswith("instance_hours"))
    check("instance hours: two launcher processes and a third writer at once lose no entry (%d of %d kept), keep "
          "every final value, and a reader sees only whole files that never lose an entry (%d reads); no lock or temp "
          "file left" % (len(set(want) & set(got)), len(want), rres.get("reads", 0)),
          all(p.returncode == 0 for p in procs) and not lost and all(got[k] == v for k, v in want.items()) and
          set(got) == set(want) | {"i-shared"} and got["i-shared"]["final"] is False and not rres["bad"] and
          left == ["instance_hours.json"], ([p.returncode for p in procs], lost[:6], rres, left,
                                             [o[-600:] for o in outs]))
    if keep is None:
        hours.unlink(missing_ok=True)
    else:
        hours.write_bytes(keep)


def test_reread_before_batch(c):
    """F5: launches.json is read again before each instance; runs another process launched meanwhile are dropped."""
    code = ("import json, sys; sys.path.insert(0, 'scripts'); import launch_e1b_aws as L\n"
            "done = L.launched_ids(L.load_launches())\n"
            "second = [r for r in L.plan('main') if L.J.run_id(*r) not in done][12:24]\n"
            "orig = L.launch_batch\n"
            "def wrapped(stage, k, *a, **kw):\n"
            "    out = orig(stage, k, *a, **kw)\n"
            "    if k == 0:\n"
            "        recs = L.load_launches()\n"
            "        for arm, task, seed in second:\n"
            "            recs.append({'stage': 'main', 'run_id': L.J.run_id(arm, task, seed), 'arm': arm, 'task': task,\n"
            "                         'seed': seed, 'instance_id': 'i-otherlauncher', 'region': 'us-east-2',\n"
            "                         'instance_type': 'g6e.xlarge', 'submitted_utc': '2026-10-25T00:00:00Z'})\n"
            "        L._write_json(L.LAUNCHES, recs)\n"
            "    return out\n"
            "L.launch_batch = wrapped\n"
            "sys.exit(L.main(['main', '--submit', '--commit=' + sys.argv[1], '--max-runs=24']))\n")
    n_inst = len(json.loads((c.state / "instances.json").read_text()))
    c.clear_calls()
    rc, out = c.pyc(code, c.F)
    recs = c.launches() or []
    cnt = collections.Counter(r["run_id"] for r in recs)
    check("a batch whose runs another process launched meanwhile is dropped before upload and launch",
          rc == 0 and out.count("LAUNCHED batch") == 1 and "skip (launched meanwhile" in out and
          max(cnt.values()) == 1 and len(json.loads((c.state / "instances.json").read_text())) == n_inst + 1 and
          sum(1 for r in recs if r.get("instance_id") == "i-otherlauncher") == 12, out[-2500:])


def main(argv):
    scratch = argv[argv.index("--scratch") + 1] if "--scratch" in argv else tempfile.mkdtemp(prefix="e1b_lt_")
    c = Ctx(scratch)
    t0 = time.time()
    c.setup()
    print("scratch %s; clone at F = %s; bash %s; openssl %s" % (c.root, c.F, c.bash, c.openssl), flush=True)
    c.branch("good")
    c.branch("nomanifest", manifest=None)
    c.branch("tampered", manifest="tampered")
    c.branch("short", manifest="short")
    c.branch("noreceipt", receipt=None)
    c.branch("badreceipt", receipt="bad")
    c.branch("bare", receipt="bare")
    c.branch("garbage", receipt="garbage")
    c.branch("tsq", receipt="tsq")
    c.branch("noreader", reader=False)
    c.branch("pending", pilot=False)
    c.branch("pilotonly", primary=False)
    c.branch("nofreeze", freeze=False)
    c.branch("words", reader=False, pilot=False, reader_text="TBD", pilot_text="not yet approved")
    if c.openssl:
        c.branch("tsr", receipt="tsr")
        c.branch("badtsr", receipt="badtsr")
    test_dry_runs(c)
    test_receipts(c)
    test_commit_identity(c)
    test_refusals(c)
    test_cell_units(c)
    test_measured_spend(c)
    recs = test_submit_pilot(c)
    test_status_and_refused_collect(c, recs)
    test_instance_and_collect(c, recs)
    test_lost(c)
    main_recs = test_submit_main(c)
    test_signal_path(c)
    test_autorun(c, main_recs)
    test_concurrent_submits(c)
    test_reread_before_batch(c)
    test_hours_concurrent(c)
    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print("\n%d checks, %d failed, %.0f s" % (len(RESULTS), n_fail, time.time() - t0))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
