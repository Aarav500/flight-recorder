"""Trainer adapters convert trainer-native per-step data into a RolloutBatch and call
recorder.record(batch).

V1 ships the TRL adapter (exercised). verl and OpenRLHF adapters are intentionally NOT
built: they follow the exact same contract (map their per-step rollout dict to
RolloutBatch, call recorder.record), and their callback surfaces churn too often to carry
speculative stubs. The `Adapter` protocol in base.py is the single extension point.
"""
