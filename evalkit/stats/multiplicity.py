"""Multiple comparison correction (docs/statistics-spec.md §6, DECISION D6).

Holm step-down (FWER) is normative for confirmatory comparisons;
Benjamini-Hochberg (FDR) only for declared screening runs. Both return
adjusted p-values in the input order.
"""

from __future__ import annotations

import numpy as np

__all__ = ["benjamini_hochberg", "holm"]


def _validated(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("p must be a non-empty 1-d array of p-values")
    if np.any(~np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
        raise ValueError("p-values must lie in [0, 1]")
    return p


def holm(p) -> np.ndarray:
    """Holm step-down adjusted p-values (§6):
    ``p~_(j) = min(1, max_{k<=j} (m-k+1) * p_(k))`` with 1-based ranks."""
    p = _validated(p)
    m = p.size
    order = np.argsort(p, kind="stable")
    mult = m - np.arange(m)  # m, m-1, ..., 1
    adj_sorted = np.minimum(1.0, np.maximum.accumulate(mult * p[order]))
    out = np.empty(m, dtype=float)
    out[order] = adj_sorted
    return out


def benjamini_hochberg(p) -> np.ndarray:
    """BH adjusted p-values (§6): ``p~_(j) = min(1, min_{k>=j} (m/k) * p_(k))``.

    Controls FDR, not FWER; the report must say so and any shortlist must be
    confirmed on fresh data (§6).
    """
    p = _validated(p)
    m = p.size
    order = np.argsort(p, kind="stable")
    ranks = np.arange(1, m + 1)
    q = p[order] * m / ranks
    adj_sorted = np.minimum(1.0, np.minimum.accumulate(q[::-1])[::-1])
    out = np.empty(m, dtype=float)
    out[order] = adj_sorted
    return out
