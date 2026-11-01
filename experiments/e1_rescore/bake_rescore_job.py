"""Bake experiments/e1_rescore/rescore_job_template.py into a runnable job. BUILD ONLY: no AWS call, no spend.

Substitutes the six placeholders, embeds the COMMITTED flightrecorder package (git archive HEAD, so an
uncommitted edit can never ride along; the job also checks the judge modules' SHA-256s), compiles the
result and refuses any leftover placeholder.

    python experiments/e1_rescore/bake_rescore_job.py olmo2_7b      # -> <out-dir>/rescore_olmo2_7b.py
    python experiments/e1_rescore/bake_rescore_job.py qwen25_7b
    python experiments/e1_rescore/bake_rescore_job.py local_test    # Qwen 0.5B target, for the CPU test only
"""
from __future__ import annotations

import argparse
import base64
import gzip
import py_compile
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TEMPLATE = HERE / "rescore_job_template.py"
SCRATCH = Path(__import__("os").environ.get("E1_RESCORE_SCRATCH", str(Path(__file__).resolve().parents[2] / "experiments" / "e1_rescore" / "_work")))
BUCKET = "fr-deploy-325527186655"                       # us-east-1
S3_DATA = "s3://%s/e1/rescore/data/e1_completions.jsonl.gz" % BUCKET
# main's commit sha, https://huggingface.co/api/models/<id>, fetched 2026-10-09; both ungated
JUDGES = {
    "olmo2_7b": ("allenai/OLMo-2-1124-7B-Instruct", "470b1fba1ae01581f270116362ee4aa1b97f4c84"),   # PRIMARY
    "qwen25_7b": ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"),         # SECONDARY
    "local_test": ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"),      # CPU test only
}


def s3_prefix(short):
    return "s3://%s/e1/rescore/%s" % (BUCKET, short)


def package_b64(ref="HEAD"):
    """tar.gz of flightrecorder/ at `ref`, deterministic gzip header; the tar's pax comment is the commit id."""
    tar = subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar", ref, "flightrecorder"],
                         check=True, capture_output=True).stdout
    return base64.b64encode(gzip.compress(tar, compresslevel=9, mtime=0)).decode("ascii")


def bake(short, data_sha=None, out_dir=SCRATCH / "job"):
    judge, rev = JUDGES[short]
    if data_sha is None:
        data_sha = (SCRATCH / "e1_completions.sha256").read_text().split()[0]
    assert re.fullmatch(r"[0-9a-f]{64}", data_sha), data_sha
    b64 = package_b64()
    src = TEMPLATE.read_text(encoding="utf-8")
    for k, v in (("__JUDGE__", judge), ("__REVISION__", rev), ("__S3_DATA__", S3_DATA),
                 ("__S3_PREFIX__", s3_prefix(short)), ("__DATA_SHA__", data_sha), ("__PKG_B64__", b64)):
        assert src.count('"%s"' % k) == 1, "placeholder %s must appear exactly once" % k
        src = src.replace('"%s"' % k, '"%s"' % v)
    left = re.findall(r"__[A-Z0-9_]+__", src.replace(b64, ""))
    assert not left, "unsubstituted placeholders: %s" % sorted(set(left))
    out_dir.mkdir(parents=True, exist_ok=True)
    outp = out_dir / ("rescore_%s.py" % short)
    outp.write_text(src, encoding="utf-8", newline="\n")
    py_compile.compile(str(outp), cfile=str(out_dir / ("rescore_%s.pyc" % short)), doraise=True)
    (out_dir / ("rescore_%s.pyc" % short)).unlink()
    return outp, len(b64)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("judges", nargs="+", choices=sorted(JUDGES))
    ap.add_argument("--data-sha", default=None)
    ap.add_argument("--out-dir", default=str(SCRATCH / "job"))
    a = ap.parse_args(argv)
    for short in a.judges:
        outp, n = bake(short, a.data_sha, Path(a.out_dir))
        print("baked+compiles OK  %s %s@%s  prefix %s  (%d b64 chars) -> %s" % (
            short, JUDGES[short][0], JUDGES[short][1][:12], s3_prefix(short), n, outp))
    print("BUILD ONLY: nothing uploaded, nothing launched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
