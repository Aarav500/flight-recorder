"""Orchestrates extractors -> RolloutFrame (+ OracleFrame), feeds ONLY RolloutFrame
to the detector, emits events to sinks."""
from __future__ import annotations

import time
from dataclasses import asdict
import numpy as np
from .types import RolloutBatch, RolloutFrame, OracleFrame, Event
from .core.kl import KLExtractor
from .core.entropy import EntropyExtractor
from .core.advantage import AdvantageExtractor
from .core.genstats import GenStatsExtractor
from .core.oracle import OracleExtractor
from .core.rolling import Smoother


class Recorder:
    def __init__(self, detector, sinks):
        self.detector = detector; self.sinks = list(sinks)
        self._kl = KLExtractor(); self._ent = EntropyExtractor()
        self._adv = AdvantageExtractor(); self._gen = GenStatsExtractor()
        self._oracle = OracleExtractor(); self._train = Smoother()

    def _emit(self, kind, step, payload):
        ev = Event(kind=kind, step=step, payload=payload, ts=time.time())
        for s in self.sinks:
            s.emit(ev)

    def record(self, batch: RolloutBatch):
        train_r = float(np.mean(batch.train_rewards))
        self._train.update(train_r)
        rf = RolloutFrame(
            step=batch.step, train_reward=train_r, train_reward_slope=self._train.slope,
            **self._kl.update(batch.logprobs, batch.ref_logprobs),
            **self._ent.update(batch.entropy, batch.logprobs),
            **self._adv.update(batch.advantages),
            **self._gen.update(batch.completions, batch.logprobs))
        of = None
        if batch.oracle_rewards is not None:
            oracle_r = float(np.mean(batch.oracle_rewards))
            od = self._oracle.update(train_r, oracle_r)
            of = OracleFrame(step=batch.step, **od)
        self._dispatch(rf, of)
        return rf, of

    def record_frames(self, rf: RolloutFrame, of: OracleFrame | None = None):
        self._dispatch(rf, of)
        return rf, of

    def _dispatch(self, rf: RolloutFrame, of: OracleFrame | None):
        self._emit("frame", rf.step, asdict(rf))
        if of is not None:
            self._emit("oracle", of.step, asdict(of))
        state = self.detector.update(rf)          # RolloutFrame ONLY
        self._emit("detector", rf.step, asdict(state))
        if state.onset:
            self._emit("onset", rf.step, asdict(state))

    def close(self):
        for s in self.sinks:
            s.close()
