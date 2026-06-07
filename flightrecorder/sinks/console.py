from __future__ import annotations
from ..types import Event


class ConsoleSink:
    def emit(self, event: Event) -> None:
        if event.kind == "onset":
            print(f"[ONSET] step={event.step} {event.payload.get('explanation','')}")

    def close(self) -> None:
        pass
