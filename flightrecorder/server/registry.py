"""Reads JSONL run artifacts and computes onset/lead summaries for the API."""
from __future__ import annotations

import json
from pathlib import Path

from ..eval.evaluator import oracle_gap_turn, lead_time


class RunRegistry:
    def __init__(self, runs_dir):
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str) -> Path:
        return self.runs_dir / f"{run_id}.jsonl"

    def read_events(self, run_id: str):
        p = self._path(run_id)
        if not p.exists():
            return None
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

    def list_runs(self):
        return [{"id": p.stem, **self._summary_from_events(self._load(p))}
                for p in sorted(self.runs_dir.glob("*.jsonl"))]

    def summary(self, run_id: str):
        events = self.read_events(run_id)
        if events is None:
            return None
        return {"id": run_id, **self._summary_from_events(events)}

    @staticmethod
    def _load(p: Path):
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

    @staticmethod
    def _summary_from_events(events) -> dict:
        frame_steps = [e["step"] for e in events if e["kind"] == "frame"]
        onset = next((e["step"] for e in events if e["kind"] == "onset"), None)
        if onset is None:
            onset = next((e["payload"].get("onset_step") for e in events
                          if e["kind"] == "detector" and e["payload"].get("onset_step") is not None), None)
        oracle_series = [e["payload"]["oracle_reward"] for e in events if e["kind"] == "oracle"]
        warmup = min(20, max(2, len(oracle_series) // 5)) if oracle_series else 2
        turn = oracle_gap_turn(oracle_series, warmup=warmup) if oracle_series else None
        return {"steps": (max(frame_steps) + 1 if frame_steps else 0),
                "onset_step": onset, "oracle_turn": turn,
                "lead": lead_time(onset, turn),
                "status": "onset" if onset is not None else "clean"}
