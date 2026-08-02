"""Bounded concurrency for model calls, via a thread pool.

Model calls are I/O bound (network in production, mock latency in tests),
so a thread pool with a configurable worker limit is sufficient; no async
runtime is required.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, Iterator, TypeVar

T = TypeVar("T")
R = TypeVar("R")

__all__ = ["run_concurrent"]


def run_concurrent(tasks: Iterable[T], fn: Callable[[T], R], *, limit: int) -> Iterator[tuple[T, R]]:
    """Run ``fn`` over ``tasks`` with at most ``limit`` in flight.

    Yields (task, result) pairs as they complete, in completion order (not
    input order), so a caller can write each result to disk as soon as
    it's ready rather than waiting for the whole batch.
    """
    if limit < 1:
        raise ValueError("limit must be >= 1")
    tasks = list(tasks)
    if not tasks:
        return
    with ThreadPoolExecutor(max_workers=limit) as pool:
        future_to_task = {pool.submit(fn, task): task for task in tasks}
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            yield task, future.result()
