"""E1's primary collapse event: the stable import path. ADOPTED by Aarav on 2026-10-07.

    from e1_event import event      # event(steps, series) -> int | None

It wraps experiments/event_candidates/sustained_drop_v2.py and refuses to load if that file's
SHA-256 differs from the one recorded at adoption. The primary analysis cannot silently run a
different event; a change to the event must be a new, logged adoption.

Provenance: experiments/event_selection.md.
  - The fixed rule (event_harness.py, b7474cf + 6b481f9) selected cusum_page_down_v2.
  - Aarav adopted sustained_drop_v2 instead, logged as a deviation in e1_preregistration.md section 14.
    The rule's first criterion, the battery score, was in-sample, and on real runs cusum_v2 dated
    4 of 8 declined probe runs at its earliest possible step.

Sensitivity analyses (reported alongside, never primary): cusum_page_down_v2 and the old
baseline_drop(0.6, persist=2).
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

_FILE = Path(__file__).resolve().parent / "event_candidates" / "sustained_drop_v2.py"
ADOPTED_SHA256 = "12a6d147c918f4d56f1cd17c5d10ad754188c215f0667a64b5f3cdbb58257dd0"


def _load():
    # hash with Windows line endings normalized, so a checkout with core.autocrlf does not trip the pin
    digest = hashlib.sha256(_FILE.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if digest != ADOPTED_SHA256:
        raise RuntimeError("E1 event file changed since adoption (sha256 %s != %s). Log a new adoption "
                           "before using it." % (digest, ADOPTED_SHA256))
    spec = importlib.util.spec_from_file_location("e1_event_impl", _FILE)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_impl = _load()
NAME = _impl.NAME
PARAMS = _impl.PARAMS
event = _impl.event
