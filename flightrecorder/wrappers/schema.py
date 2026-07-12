"""Record schema for the Gym reward-audit wrapper.

This mirrors the shared core worked out across the Inspect (``MonitorRecord``) and
Gymnasium proposal threads: ``SignalValue``/``SignalDerivation``/``ProducerRef``,
append-only frozen records, and an oracle-isolation rule. Everything else (identity
envelope, record granularity) is domain-specific to a Gym wrapper rather than shared,
per the conclusion reached in that discussion -- each system keeps its own identity
envelope instead of forcing one universal record type.

Oracle isolation: this module has no concept of a "true"/ground-truth reward at all.
If a caller wants to compare an audited episode against a held-out oracle signal, that
comparison must happen in code the caller owns, using data the caller supplies
out-of-band -- never by adding a field to ``EpisodeRecord`` for it. This is the same
discipline Flight Recorder's own ``RolloutFrame``/``OracleFrame`` split enforces
elsewhere in this project: oracle leakage is a type error, not a discipline problem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Mapping


@dataclass(frozen=True)
class SignalDerivation:
    """How a :class:`SignalValue` was computed."""

    method: Literal["instant", "ema", "rolling", "custom"]
    window_size: int | None = None
    alpha: float | None = None
    parameters: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SignalValue:
    """One audited quantity, with an explicit availability state.

    ``state`` exists so a rolling/EMA signal's cold-start prefix can be reported
    honestly (``state="unavailable"``) instead of standing in a filler ``0.0`` that is
    indistinguishable downstream from a genuine zero reading.
    """

    value: float | None
    state: Literal["observed", "unavailable", "error"]
    derivation: SignalDerivation
    reason: str | None = None


@dataclass(frozen=True)
class ProducerRef:
    """Identifies what produced a record: the wrapper itself, plus its audited
    reward function's version/digest (never the reward function's source)."""

    name: str
    version: str
    config: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeRecord:
    """One completed episode's audited reward signals.

    Append-only: once emitted, a record is never mutated. ``record_kind`` is fixed at
    ``"episode"`` for this module (no step-level or summary-level record kind is
    needed -- see the design discussion for why episode granularity was found
    sufficient for every producer considered so far).
    """

    schema_version: Literal["1"]
    record_id: str
    record_kind: Literal["episode"]
    emitted_at: datetime

    run_id: str
    env_id: str
    episode_uuid: str
    episode_index: int
    seed: int | None

    producer: ProducerRef
    audit_status: Literal["complete", "warming_up", "errored"]

    signals: Mapping[str, SignalValue]
