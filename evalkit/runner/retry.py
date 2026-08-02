"""Retry with backoff on rate limits.

No global random state: every call takes an explicit ``seed`` that
determines its jitter sequence, so retry timing is reproducible and one
task's jitter never depends on call order relative to any other task.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

from .client import RateLimitError

__all__ = ["RetryPolicy", "call_with_retry"]

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with jitter, capped at ``max_delay``."""

    max_attempts: int = 5
    base_delay: float = 1.0
    max_delay: float = 30.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_delay <= 0 or self.max_delay <= 0:
            raise ValueError("base_delay and max_delay must be positive")


def call_with_retry(fn: Callable[[], T], *, policy: RetryPolicy, seed: int,
                     sleep: Callable[[float], None] = time.sleep) -> T:
    """Call ``fn``, retrying on RateLimitError with exponential backoff + jitter.

    ``sleep`` is injectable so tests can assert on retry behavior without
    waiting in real time. Any exception other than RateLimitError
    propagates immediately without retrying.
    """
    rng = random.Random(seed)
    last_error: RateLimitError | None = None
    for attempt in range(policy.max_attempts):
        try:
            return fn()
        except RateLimitError as exc:
            last_error = exc
            if attempt == policy.max_attempts - 1:
                break
            delay = min(policy.base_delay * (2 ** attempt), policy.max_delay)
            delay *= 0.5 + rng.random()  # jitter in [0.5x, 1.5x)
            sleep(delay)
    assert last_error is not None
    raise last_error
