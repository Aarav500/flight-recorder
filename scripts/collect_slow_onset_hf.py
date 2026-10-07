"""Status and collection for slow-onset jobs submitted to HF Jobs with `launch_slow_onset.py --record`.

The job prints its result into the log as RESULT_B64 chunks (gzip + base64), because an HF Job's
filesystem is discarded at exit. This reassembles them into the same JSON the job wrote to
/tmp/result.json, which is what experiments/e1_pilot_gate.py reads.

    python scripts/collect_slow_onset_hf.py runs/e1/pilot_hf_jobs.json            # status
    python scripts/collect_slow_onset_hf.py runs/e1/pilot_hf_jobs.json --collect  # -> runs/e1/pilot/
"""
from __future__ import annotations

import base64
import gzip
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]


def main(argv):
    rec_path = Path(argv[0])
    out = ROOT / "runs" / "e1" / rec_path.stem.replace("_hf_jobs", "")
    collect = "--collect" in argv
    api = HfApi()
    for name, v in json.loads(rec_path.read_text()).items():
        info = api.inspect_job(job_id=v["job_id"])
        stage = getattr(getattr(info, "status", None), "stage", "?")
        if not collect:
            tail = list(api.fetch_job_logs(job_id=v["job_id"], tail=200))
            onset = [ln for ln in tail if ln.startswith("ONSET step")][-1:] or ["(no step logged yet)"]
            print("%-52s %-10s %s" % (name, stage, onset[0][:100]))
            continue
        lines = list(api.fetch_job_logs(job_id=v["job_id"]))
        (out / "logs").mkdir(parents=True, exist_ok=True)
        (out / "logs" / (name + ".log")).write_text("\n".join(lines), encoding="utf-8")
        chunks = {}
        for ln in lines:
            if ln.startswith("RESULT_B64 "):
                _, idx, data = ln.split(" ", 2)
                i, k = (int(x) for x in idx.split("/"))
                chunks[i] = data.strip()
        if chunks and len(chunks) == k:
            res = json.loads(gzip.decompress(base64.b64decode("".join(chunks[i] for i in range(1, k + 1)))))
            (out / (name.replace("_slow_onset_", "") + ".json")).write_text(json.dumps(res))
            print("%-52s %-10s collected (%d steps)" % (name, stage, len(res["log"])))
        else:
            print("%-52s %-10s NO RESULT in log (%d/%s chunks)" % (name, stage, len(chunks), "?"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
