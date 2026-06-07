from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path
from ..types import Event


class JSONLSink:
    def __init__(self, path):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8")

    def emit(self, event: Event) -> None:
        self._fh.write(json.dumps(asdict(event)) + "\n"); self._fh.flush()

    def close(self) -> None:
        self._fh.close()
