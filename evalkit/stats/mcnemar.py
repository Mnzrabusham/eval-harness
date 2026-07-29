"""Exact McNemar test (docs/statistics-spec.md §5).

Classical cross-check for scenario C with no clustering and single
judgments. Conditional on the discordant count ``n_d = n_pos + n_neg``,
under H0 ``n_pos ~ Binomial(n_d, 1/2)``. Mid-p is deliberately NOT
implemented (FLAGGED F5: better calibrated on average but not guaranteed
conservative).
"""

from __future__ import annotations

import numpy as np
from scipy import stats as _sps

__all__ = ["discordant_counts", "mcnemar_exact"]


def mcnemar_exact(n_pos: int, n_neg: int) -> float:
    """Two-sided exact McNemar p-value: ``min(1, 2*P(X <= min(n_pos, n_neg)))``.

    Returns 1.0 when ``n_pos == n_neg`` (including zero discordant items).
    """
    if n_pos < 0 or n_neg < 0 or n_pos != int(n_pos) or n_neg != int(n_neg):
        raise ValueError("n_pos and n_neg must be non-negative integers")
    n_pos, n_neg = int(n_pos), int(n_neg)
    n_d = n_pos + n_neg
    if n_d == 0 or n_pos == n_neg:
        return 1.0
    k = min(n_pos, n_neg)
    return float(min(1.0, 2.0 * _sps.binom.cdf(k, n_d, 0.5)))


def discordant_counts(d) -> tuple[int, int, int]:
    """(n_pos, n_neg, n_zero) from binary-singleton differences d_i in {-1, 0, +1}."""
    d = np.asarray(d, dtype=float)
    if not np.all(np.isin(d, (-1.0, 0.0, 1.0))):
        raise ValueError("d must contain only -1, 0, +1 (binary, single response/judgment)")
    n_pos = int(np.count_nonzero(d == 1.0))
    n_neg = int(np.count_nonzero(d == -1.0))
    return n_pos, n_neg, int(d.size - n_pos - n_neg)
