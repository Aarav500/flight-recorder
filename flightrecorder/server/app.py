"""FastAPI app: REST for saved runs + a WebSocket that replays an artifact as if live,
and (in production) serving the built React app from a static directory as one container.
"""
from __future__ import annotations

import asyncio
import os

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .registry import RunRegistry


def create_app(runs_dir: str | None = None, broker=None, static_dir: str | None = None) -> FastAPI:
    runs_dir = runs_dir or os.environ.get("FLIGHTRECORDER_RUNS", "runs")
    static_dir = static_dir or os.environ.get("FLIGHTRECORDER_STATIC")
    app = FastAPI(title="Flight Recorder")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"])
    reg = RunRegistry(runs_dir)
    app.state.registry = reg
    app.state.broker = broker

    # Plain liveness probe used by the deploy to verify the container came up.
    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/health")
    def api_health():
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

    _mount_static(app, static_dir)
    return app


def _mount_static(app: FastAPI, static_dir: str | None) -> None:
    """Serve the built React app (web/dist) with SPA fallback, if present."""
    if not static_dir or not os.path.isdir(static_dir):
        return
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    assets = os.path.join(static_dir, "assets")
    if os.path.isdir(assets):
        app.mount("/assets", StaticFiles(directory=assets), name="assets")
    index = os.path.join(static_dir, "index.html")

    @app.get("/{full_path:path}")
    def spa(full_path: str = ""):
        # /api/* and /health are matched by their explicit routes above; this only
        # catches client-side routes (/, /report/x, /live/x) and returns the SPA shell.
        if full_path.startswith("api/") or full_path == "health":
            raise HTTPException(404, "not found")
        return FileResponse(index)
