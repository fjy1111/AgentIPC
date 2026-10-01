from __future__ import annotations
from collections.abc import Callable
from agentipc.evaluation.runner import run_abcd_suite
from .aggregation import flatten_record, aggregate_records

def run_e1(*, tasks: list[str], repeat: int, run_factory: Callable):
    if type(repeat) is not int or repeat < 1: raise ValueError("repeat must be >= 1")
    raw = run_abcd_suite(tasks=tasks, seeds=list(range(repeat)), run_factory=run_factory)
    rows = [flatten_record(r, i % repeat) for i, r in enumerate(raw)]
    return rows, aggregate_records(rows)
