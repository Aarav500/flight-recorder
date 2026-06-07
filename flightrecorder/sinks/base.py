from __future__ import annotations
from typing import Protocol
from ..types import Event


class EventSink(Protocol):
    def emit(self, event: Event) -> None: ...
    def close(self) -> None: ...
