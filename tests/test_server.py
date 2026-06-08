from fastapi.testclient import TestClient
from flightrecorder.server.app import create_app
from flightrecorder.sinks.websocket import Broker, WebSocketSink
from flightrecorder.types import Event
from flightrecorder.repro.synthetic import generate_run
from flightrecorder.recorder import Recorder
from flightrecorder.sinks.jsonl import JSONLSink
from flightrecorder.detector.contraction_tube import ContractionTubeDetector


def _make_run(runs_dir, name="r1", tstar=60, n=120):
    run = generate_run(seed=1, n_steps=n, tstar=tstar)
    rec = Recorder(ContractionTubeDetector(warmup=20), [JSONLSink(runs_dir / f"{name}.jsonl")])
    for rf, of in zip(run["rollout_frames"], run["oracle_frames"]):
        rec.record_frames(rf, of)
    rec.close()


def test_list_get_summary(tmp_path):
    _make_run(tmp_path)
    c = TestClient(create_app(str(tmp_path)))
    runs = c.get("/api/runs").json()
    assert any(r["id"] == "r1" for r in runs)
    r1 = next(r for r in runs if r["id"] == "r1")
    assert r1["status"] == "onset" and r1["onset_step"] is not None
    events = c.get("/api/runs/r1").json()
    assert any(e["kind"] == "frame" for e in events)
    s = c.get("/api/runs/r1/summary").json()
    assert s["onset_step"] is not None and s["oracle_turn"] is not None
    assert s["lead"] == s["oracle_turn"] - s["onset_step"]


def test_missing_run_404(tmp_path):
    c = TestClient(create_app(str(tmp_path)))
    assert c.get("/api/runs/nope/summary").status_code == 404
    assert c.get("/api/runs/nope").status_code == 404


def test_ws_replay_streams_events(tmp_path):
    _make_run(tmp_path)
    c = TestClient(create_app(str(tmp_path)))
    kinds = set()
    with c.websocket_connect("/api/runs/r1/live?speed=5000") as ws:
        for _ in range(5000):
            m = ws.receive_json()
            kinds.add(m["kind"])
            if m["kind"] == "run_end":
                break
    assert {"frame", "oracle", "detector"} <= kinds and "run_end" in kinds


def test_health_endpoint(tmp_path):
    c = TestClient(create_app(str(tmp_path)))
    r = c.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_static_spa_fallback_without_shadowing_api(tmp_path):
    runs = tmp_path / "runs"
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html><title>app shell</title>", encoding="utf-8")
    _make_run(runs)
    c = TestClient(create_app(str(runs), static_dir=str(static)))
    # client-side route -> SPA shell
    assert "app shell" in c.get("/report/r1").text
    # API still resolves (not shadowed by the catch-all)
    assert c.get("/api/runs/r1/summary").json()["onset_step"] is not None
    # health still resolves
    assert c.get("/health").json() == {"status": "ok"}


def test_websocket_sink_publishes_to_broker():
    class FakeBroker:
        def __init__(self): self.sent = []
        def publish(self, run_id, payload): self.sent.append((run_id, payload))
    fb = FakeBroker()
    WebSocketSink(fb, "run1").emit(Event(kind="frame", step=0, payload={"x": 1}, ts=0.0))
    assert fb.sent[0][0] == "run1" and fb.sent[0][1]["kind"] == "frame"
