"""Replay persisted benign-run artifacts through the pre-registered ContractionTube
detector and report, per run, the onset step and peak warmup-normalized CUSUM S. Used to
place the multi-task benign batch on the same CUSUM axis as the strength-swept onsets
(the crossover) for the overlay figure. Pure offline replay; no GPU, no oracle."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flightrecorder.types import RolloutFrame  # noqa: E402
from flightrecorder.config import load_thresholds  # noqa: E402
from flightrecorder.detector.contraction_tube import ContractionTubeDetector  # noqa: E402
FRAME_FIELDS = set(RolloutFrame.__dataclass_fields__)


def replay(path, thr):
    det = ContractionTubeDetector(warmup=thr.warmup, rho=thr.rho, tau=thr.tube_tau,
                                  k=thr.cusum_k, h=thr.cusum_h, cusum_signal=thr.cusum_signal,
                                  persistence=thr.persistence)
    onset, peak_S, n = None, 0.0, 0
    for line in Path(path).read_text().splitlines():
        e = json.loads(line)
        if e.get("kind") != "frame":
            continue
        p = e["payload"]; n += 1
        fr = RolloutFrame(**{k: p[k] for k in FRAME_FIELDS if k in p})
        st = det.update(fr)
        peak_S = max(peak_S, st.cusum)
        if st.onset and onset is None:
            onset = st.onset_step
    return onset, peak_S, n


if __name__ == "__main__":
    thr = load_thresholds(str(ROOT / "configs" / "thresholds.yaml"))
    paths = sys.argv[1:] or [str(ROOT / "runs" / "gentle_seed0.jsonl")]
    for path in paths:
        o, s, n = replay(path, thr)
        fired = "FIRES" if o is not None else "quiet"
        print(f"{Path(path).name:28s} frames={n:3d}  onset={str(o):>4s}  peak_CUSUM_S={s:7.2f}  [{fired}]")
