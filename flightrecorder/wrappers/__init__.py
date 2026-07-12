"""Opt-in reward-audit wrappers for Gymnasium environments.

See ``flightrecorder/wrappers/gym_audit.py`` for the scope and boundary this module
commits to: it records only what the environment or user explicitly exposes
(episode-level reward statistics, a reward-function version/digest, and
readiness/derivation metadata for rolling signals). It does not, and cannot, infer
reward *composition* from inside a closed environment.
"""

from flightrecorder.wrappers.gym_audit import RewardAuditWrapper
from flightrecorder.wrappers.schema import EpisodeRecord, ProducerRef, SignalDerivation, SignalValue

__all__ = [
    "RewardAuditWrapper",
    "EpisodeRecord",
    "ProducerRef",
    "SignalDerivation",
    "SignalValue",
]
