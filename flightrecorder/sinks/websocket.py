"""In-process pub/sub bridging the sync Recorder to async WebSocket clients (true-live).

For the V1 demo the server primarily replays saved artifacts; this Broker/sink is the path
for streaming a live recorder running in the same process as the server."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
from ..types import Event


class Broker:
    def __init__(self):
        self._subs: dict[str, set[asyncio.Queue]] = {}

    def subscribe(self, run_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subs.setdefault(run_id, set()).add(q)
        return q

    def unsubscribe(self, run_id: str, q: asyncio.Queue) -> None:
        self._subs.get(run_id, set()).discard(q)

    def publish(self, run_id: str, payload: dict) -> None:
        for q in list(self._subs.get(run_id, set())):
            q.put_nowait(payload)


class WebSocketSink:
    """Recorder sink that publishes each event to a Broker keyed by run_id."""

    def __init__(self, broker: Broker, run_id: str):
        self.broker = broker
        self.run_id = run_id

    def emit(self, event: Event) -> None:
        self.broker.publish(self.run_id, asdict(event))

    def close(self) -> None:
        self.broker.publish(self.run_id, {"kind": "run_end", "step": -1, "payload": {}, "ts": 0.0})
