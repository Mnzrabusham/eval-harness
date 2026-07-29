"""Judge nondeterminism as a variance component (docs/statistics-spec.md §8).

Method-of-moments estimators (FLAGGED F9: REML/MixedLM is the defensible
heavier alternative; method of moments is normative because it is
transparent and sufficient).

None of this is needed for §2's tests to be valid — item-level analysis
absorbs judge noise automatically. It exists so reports can say how much of
the observed variance is the judge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "JudgeVariance",
    "VarianceComponents",
    "sigma2_judge",
    "sigma2_within_item",
    "variance_components",
]


@dataclass(frozen=True)
class JudgeVariance:
    """Pooled within-response judge-call variance (§8.2)."""

    sigma2_j: float
    n_responses_used: int
    df: int


def sigma2_judge(replicate_sets) -> JudgeVariance | None:
    """``sigma^2_J = sum_r sum_j (y_rj - mean_r)^2 / sum_r (m_r - 1)`` over
    responses with >= 2 judge calls. Unbiased under any imbalance.

    Returns None when no response has replicate calls — sigma^2_J is then
    unavailable and the report must say so (§8.2, DECISION D4).
    """
    num = 0.0
    den = 0
    used = 0
    for vals in replicate_sets:
        v = np.asarray(vals, dtype=float)
        if v.size >= 2:
            num += float(((v - v.mean()) ** 2).sum())
            den += v.size - 1
            used += 1
    if den == 0:
        return None
    return JudgeVariance(num / den, used, den)


def sigma2_within_item(*, r: int, m: float, sigma2_j: float, sigma2_g: float = 0.0, pairwise: bool = False) -> float:
    """§8.1: ``sigma^2_W = 2*(sigma^2_G/r + sigma^2_J/(r*m))`` for
    scalar/binary; without the factor 2 for pairwise (one call judges the
    pair).

    For designs where m varies across responses within an item, compute the
    exact per-item variance externally and pass it to
    ``variance_components`` directly; this helper covers the balanced case.
    """
    if r < 1 or m < 1:
        raise ValueError("r >= 1 and m >= 1 required")
    w = sigma2_g / r + sigma2_j / (r * m)
    return w if pairwise else 2.0 * w


@dataclass(frozen=True)
class VarianceComponents:
    """§8.2 method-of-moments decomposition of Var(d_i)."""

    sigma2_b: float
    mean_sigma2_w: float
    s2_d: float
    icc_judge: float
    truncated: bool  # truncation at zero triggered; must be reported


def variance_components(d, sigma2_w_items) -> VarianceComponents:
    """``sigma^2_B = max(0, s^2_d - mean_i sigma^2_W,i)`` and
    ``ICC_judge = sigma^2_B / (sigma^2_B + sigma^2_W)`` (§8.2).

    ``sigma2_w_items`` is a scalar or per-item array of within-item variances
    from ``sigma2_within_item`` (or computed exactly for unbalanced designs).
    """
    d = np.asarray(d, dtype=float)
    if d.size < 2:
        raise ValueError("need at least 2 items")
    w = float(np.mean(np.asarray(sigma2_w_items, dtype=float)))
    if w < 0:
        raise ValueError("within-item variances must be non-negative")
    s2 = float(np.var(d, ddof=1))
    raw = s2 - w
    b = max(0.0, raw)
    icc = b / (b + w) if (b + w) > 0 else float("nan")
    return VarianceComponents(b, w, s2, icc, raw < 0)
