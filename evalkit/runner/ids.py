"""Deterministic id and seed derivation.

Ids are hashes of their inputs, never random or counter-based, so
re-running the same configuration (same items, variants, seeds) reproduces
the exact same ids. That's what lets the cache and the incremental store
recognize repeated work across separate run invocations.
"""

from __future__ import annotations

import hashlib

__all__ = ["derive_id", "derive_seed"]


def _digest(parts: tuple[object, ...]) -> str:
    payload = "\x1f".join(repr(p) for p in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def derive_id(*parts: object) -> str:
    """A short, stable, hex id derived from ``parts``."""
    return _digest(parts)[:24]


def derive_seed(*parts: object) -> int:
    """A stable non-negative int seed derived from ``parts``.

    For feeding a ``random.Random`` (e.g. retry jitter) deterministically
    from task identity, without any shared/global random state.
    """
    return int(_digest(parts)[:16], 16)
