"""Inspect AI Hooks-based producer for Flight Recorder.

Deliberately scoped to only the two signals genuinely computable at Inspect
eval time -- rolling entropy/surprisal (from per-token logprobs, when a task
is run with ``logprobs=True``) and scorer disagreement (spread across scorer
values when a task uses more than one scorer). KL-vs-reference and
advantage-variance/drift are RL training-loop quantities with no eval-context
equivalent, so this producer does not attempt them.

Built as a plain ``inspect_ai.hooks.Hooks`` subclass against existing hook
points (``on_sample_event``, ``on_sample_end``) -- no new core Inspect API.
This design, and the reasoning for not attempting a wider signal set, is the
resolution of https://github.com/UKGovernmentBEIS/inspect_ai/pull/4474.

inspect_ai is imported lazily/optionally so this module imports and
unit-tests with no inspect_ai installed, matching ``adapters/trl.py``'s
established pattern in this repo.
"""
from __future__ import annotations

import time
from typing import Any

from ..core.entropy import EntropyExtractor
from ..core.rolling import RunningMoments
from ..sinks.base import EventSink
from ..sinks.jsonl import JSONLSink
from ..types import Event as FREvent

try:  # real base when available; stubs otherwise so the module imports anywhere
    from inspect_ai.event import ModelEvent, ScoreEvent
    from inspect_ai.hooks import Hooks as _Hooks
    from inspect_ai.hooks import hooks as _hooks_decorator
except Exception:  # pragma: no cover - exercised only without inspect_ai

    class _Hooks:  # noqa: D401
        def enabled(self) -> bool:
            return True

    def _hooks_decorator(name: str, description: str):
        def _wrap(cls):
            return cls

        return _wrap

    class ModelEvent:  # noqa: D401 - marker stub, NOT `object` (isinstance must still discriminate)
        pass

    class ScoreEvent:  # noqa: D401 - marker stub, NOT `object` (isinstance must still discriminate)
        pass

Hooks = _Hooks
hooks = _hooks_decorator


def _token_logprobs(model_event: Any) -> list[float] | None:
    """Extract per-token chosen-token logprobs from a ModelEvent's first choice.

    Returns None if the task wasn't run with logprobs enabled (no
    ``choices[0].logprobs``) -- entropy is simply skipped for that call, not
    faked with a filler value.
    """
    output = getattr(model_event, "output", None)
    choices = getattr(output, "choices", None) if output is not None else None
    if not choices:
        return None
    logprobs = getattr(choices[0], "logprobs", None)
    content = getattr(logprobs, "content", None) if logprobs is not None else None
    if not content:
        return None
    return [float(lp.logprob) for lp in content]


class _SampleState:
    """Per-sample accumulator, keyed by sample_id in FlightRecorderInspectHooks."""

    __slots__ = ("entropy", "last_entropy_summary", "score_values")

    def __init__(self) -> None:
        self.entropy = EntropyExtractor()
        self.last_entropy_summary: dict[str, float] = {"entropy_mean": float("nan"), "entropy_trend": 0.0}
        self.score_values: list[float] = []


class FlightRecorderInspectHooks(Hooks):
    """Emits one Flight Recorder Event per completed Inspect sample.

    Usage (register with Inspect's own decorator, applied by the caller so this
    class stays a plain, directly-testable object)::

        from inspect_ai.hooks import hooks
        from flightrecorder.adapters.inspect_hooks import FlightRecorderInspectHooks

        @hooks(name="flight-recorder", description="Entropy + scorer-disagreement audit")
        class _Registered(FlightRecorderInspectHooks):
            def __init__(self):
                super().__init__(jsonl_path="flight_recorder_inspect.jsonl")
    """

    def __init__(self, sink: EventSink | None = None, jsonl_path: str | None = None):
        if sink is not None and jsonl_path is not None:
            raise ValueError("pass at most one of sink, jsonl_path")
        self._sink: EventSink | None = sink or (JSONLSink(jsonl_path) if jsonl_path else None)
        self._samples: dict[str, _SampleState] = {}

    def _state_for(self, sample_id: str) -> _SampleState:
        state = self._samples.get(sample_id)
        if state is None:
            state = _SampleState()
            self._samples[sample_id] = state
        return state

    async def on_sample_event(self, data: Any) -> None:
        event = data.event
        state = self._state_for(data.sample_id)
        if isinstance(event, ModelEvent):
            token_lps = _token_logprobs(event)
            if token_lps:
                state.last_entropy_summary = state.entropy.update(entropy=None, logprobs=token_lps)
        elif isinstance(event, ScoreEvent) and not event.intermediate:
            value = event.score.value
            if isinstance(value, (int, float)) and value == value:  # numeric, non-NaN
                state.score_values.append(float(value))

    async def on_sample_end(self, data: Any) -> None:
        state = self._samples.pop(data.sample_id, None)
        if state is None or self._sink is None:
            return
        disagreement = None
        if len(state.score_values) >= 2:
            moments = RunningMoments()
            for v in state.score_values:
                moments.update(v)
            disagreement = moments.std
        payload = {
            "sample_id": data.sample_id,
            "eval_id": data.eval_id,
            "run_id": data.run_id,
            "entropy_mean": state.last_entropy_summary["entropy_mean"],
            "entropy_trend": state.last_entropy_summary["entropy_trend"],
            "scorer_disagreement": disagreement,
            "n_scorers": len(state.score_values),
        }
        self._sink.emit(FREvent(kind="frame", step=0, payload=payload, ts=time.time()))

    def close(self) -> None:
        if self._sink is not None:
            self._sink.close()
