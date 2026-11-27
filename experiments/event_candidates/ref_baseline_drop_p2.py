"""REFERENCE: the current E1 gate event, kept unchanged so the harness can show why it is replaced."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark"))
import score_prime as SP  # noqa: E402

NAME = "ref_baseline_drop(0.6,persist=2)"
PARAMS = {"frac": 0.6, "persist": 2, "baseline_steps": 5}


def event(steps, series):
    c = SP.collapse_step(list(steps), list(series), "baseline_drop", 0.6, 2)
    return None if c is None else int(c)
