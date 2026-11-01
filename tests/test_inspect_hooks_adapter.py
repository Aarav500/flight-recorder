import asyncio
from types import SimpleNamespace

from flightrecorder.adapters.inspect_hooks import (
    FlightRecorderInspectHooks,
    ModelEvent,
    ScoreEvent,
    _token_logprobs,
)


class CapturingSink:
    def __init__(self):
        self.events = []
        self.closed = False

    def emit(self, e):
        self.events.append(e)

    def close(self):
        self.closed = True


class _FakeModelEvent(ModelEvent):
    """Minimal stand-in matching the attribute path this adapter reads:
    output.choices[0].logprobs.content[i].logprob"""

    def __init__(self, token_logprobs):
        content = [SimpleNamespace(logprob=lp) for lp in token_logprobs] if token_logprobs is not None else None
        logprobs = SimpleNamespace(content=content) if content is not None else None
        choice = SimpleNamespace(logprobs=logprobs)
        self.output = SimpleNamespace(choices=[choice])


class _FakeModelEventNoLogprobs(ModelEvent):
    def __init__(self):
        self.output = SimpleNamespace(choices=[SimpleNamespace(logprobs=None)])


class _FakeScoreEvent(ScoreEvent):
    def __init__(self, value, intermediate=False):
        self.score = SimpleNamespace(value=value)
        self.intermediate = intermediate


def _sample_event(sample_id, event, eval_id="eval-1", run_id="run-1"):
    return SimpleNamespace(sample_id=sample_id, eval_id=eval_id, run_id=run_id, event=event)


def _sample_end(sample_id, eval_id="eval-1", run_id="run-1"):
    return SimpleNamespace(sample_id=sample_id, eval_id=eval_id, run_id=run_id)


async def _run(hooks_obj, sample_id, model_events=(), score_events=()):
    for lps in model_events:
        await hooks_obj.on_sample_event(_sample_event(sample_id, _FakeModelEvent(lps)))
    for value, intermediate in score_events:
        await hooks_obj.on_sample_event(_sample_event(sample_id, _FakeScoreEvent(value, intermediate)))
    await hooks_obj.on_sample_end(_sample_end(sample_id))


def test_token_logprobs_extracts_chosen_token_values():
    ev = _FakeModelEvent([-0.1, -0.5, -1.2])
    assert _token_logprobs(ev) == [-0.1, -0.5, -1.2]


def test_token_logprobs_returns_none_when_no_logprobs_configured():
    assert _token_logprobs(_FakeModelEventNoLogprobs()) is None


def test_emits_entropy_and_no_disagreement_for_single_scorer():
    sink = CapturingSink()
    h = FlightRecorderInspectHooks(sink=sink)
    asyncio.run(_run(h, "s1", model_events=[[-0.2, -0.3], [-0.1]], score_events=[(1.0, False)]))
    assert len(sink.events) == 1
    payload = sink.events[0].payload
    assert payload["sample_id"] == "s1"
    assert payload["entropy_mean"] == payload["entropy_mean"]  # not NaN
    assert payload["scorer_disagreement"] is None  # only 1 scorer value -> no spread computable
    assert payload["n_scorers"] == 1


def test_computes_scorer_disagreement_across_multiple_scorers():
    sink = CapturingSink()
    h = FlightRecorderInspectHooks(sink=sink)
    asyncio.run(_run(h, "s2", score_events=[(1.0, False), (0.0, False), (0.5, False)]))
    payload = sink.events[0].payload
    assert payload["n_scorers"] == 3
    assert payload["scorer_disagreement"] is not None
    assert payload["scorer_disagreement"] > 0.0


def test_intermediate_scores_are_excluded_from_disagreement():
    sink = CapturingSink()
    h = FlightRecorderInspectHooks(sink=sink)
    asyncio.run(_run(h, "s3", score_events=[(1.0, True), (1.0, False)]))  # 1 intermediate + 1 final
    payload = sink.events[0].payload
    assert payload["n_scorers"] == 1  # intermediate score not counted
    assert payload["scorer_disagreement"] is None


def test_sample_state_is_isolated_and_cleared_after_end():
    sink = CapturingSink()
    h = FlightRecorderInspectHooks(sink=sink)
    asyncio.run(_run(h, "a", score_events=[(1.0, False)]))
    asyncio.run(_run(h, "b", score_events=[(0.0, False)]))
    assert len(sink.events) == 2
    assert h._samples == {}  # both samples popped on their own on_sample_end


def test_no_sink_configured_is_a_safe_noop():
    h = FlightRecorderInspectHooks()  # no sink, no jsonl_path
    asyncio.run(_run(h, "s1", score_events=[(1.0, False)]))  # must not raise
    assert h._samples == {}


def test_sink_and_jsonl_path_together_is_an_error(tmp_path):
    try:
        FlightRecorderInspectHooks(sink=CapturingSink(), jsonl_path=str(tmp_path / "out.jsonl"))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_close_closes_the_underlying_sink():
    sink = CapturingSink()
    h = FlightRecorderInspectHooks(sink=sink)
    h.close()
    assert sink.closed is True
