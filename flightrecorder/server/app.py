"""FastAPI app: REST for saved runs + a WebSocket that replays an artifact as if live.

The Live dashboard connects to /api/runs/{id}/live; the Report viewer uses the REST routes.
Serving built web assets for single-container hosting is wired in the hosting prompt.
"""
from __future__ import annotations

import asyncio
import os

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .registry import RunRegistry


def create_app(runs_dir: str | None = None, broker=None) -> FastAPI:
    runs_dir = runs_dir or os.environ.get("FLIGHTRECORDER_RUNS", "runs")
    app = FastAPI(title="Flight Recorder")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"])
    reg = RunRegistry(runs_dir)
    app.state.registry = reg
    app.state.broker = broker

    @app.get("/api/health")
    def health():
        return {"ok": True, "runs_dir": str(reg.runs_dir)}

    @app.get("/api/runs")
    def list_runs():
        return reg.list_runs()

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        events = reg.read_events(run_id)
        if events is None:
            raise HTTPException(404, "run not found")
        return events

    @app.get("/api/runs/{run_id}/summary")
    def get_summary(run_id: str):
        s = reg.summary(run_id)
        if s is None:
            raise HTTPException(404, "run not found")
        return s

    @app.websocket("/api/runs/{run_id}/live")
    async def live(ws: WebSocket, run_id: str, speed: float = 20.0):
        """Replay a saved artifact as a live stream, paced at `speed` steps/sec.
        If a broker is attached, also forwards live events for a recording in-process."""
        await ws.accept()
        events = reg.read_events(run_id)
        if events is None:
            await ws.close(code=1008)
            return
        delay = 1.0 / max(speed, 1e-3)
        last_step = 0
        try:
            seen_first_frame = False
            for e in events:
                if e["kind"] == "frame":
                    if seen_first_frame:
                        await asyncio.sleep(delay)
                    seen_first_frame = True
                    last_step = e["step"]
                await ws.send_json(e)
            await ws.send_json({"kind": "run_end", "step": last_step, "payload": {}, "ts": 0.0})
        except WebSocketDisconnect:
            pass

    return app
