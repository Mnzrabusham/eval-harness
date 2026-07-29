"""Shared inference engine (docs/statistics-spec.md §2).

One engine for every scenario: cluster percentile bootstrap CI (§2.1),
cluster sign-flip permutation test (§2.2), analytic cluster-robust
cross-check (§2.3), small-sample warnings (§2.4).

RNG discipline (§0): every stochastic function takes an explicit integer
seed and builds ``numpy.random.Generator(PCG64(seed))``. Within
``run_engine`` the stream is consumed bootstrap-first, then permutation;
that order is part of the reproducibility contract.

Deterministic ordering (§0): items are sorted lexicographically by
(cluster id, item id) as strings before any resampling. When ids are not
supplied, zero-padded input positions are used so lexicographic order
equals input order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as _sps

__all__ = ["AnalyticResult", "EngineResult", "make_rng", "run_engine"]

_ALTERNATIVES = ("two-sided", "greater", "less")


def make_rng(seed: int) -> np.random.Generator:
    """The pinned RNG construction (§0): ``Generator(PCG64(seed))``."""
    return np.random.Generator(np.random.PCG64(seed))


@dataclass(frozen=True)
class AnalyticResult:
    """Cluster-robust t interval (§2.3). All fields NaN when C < 2.

    The CI is always two-sided at level 1 - alpha; only the p-value follows
    the requested alternative.
    """

    se: float
    t_stat: float
    df: int
    ci_low: float
    ci_high: float
    p_value: float


@dataclass(frozen=True)
class EngineResult:
    estimate: float
    ci_low: float
    ci_high: float
    boot_se: float
    p_value: float
    test_method: str  # "sign-flip-exact" | "sign-flip-mc"
    alternative: str  # "two-sided" | "greater" | "less"
    n_items: int
    n_clusters: int
    cluster_sizes: dict  # size -> number of clusters of that size (§1.3)
    analytic: AnalyticResult
    degenerate: bool
    warnings: tuple
    alpha: float
    n_boot: int
    n_perm: int
    seed: int


def _sorted_cluster_stats(d, clusters, item_ids):
    """Cluster sums and sizes in the pinned lexicographic order (§0)."""
    n = len(d)
    if item_ids is None:
        item_ids = [f"{i:012d}" for i in range(n)]
    if clusters is None:
        # §1.4: an item with no source doc is its own singleton cluster.
        clusters = [f"__singleton__:{item_ids[i]}" for i in range(n)]
    order = sorted(range(n), key=lambda i: (str(clusters[i]), str(item_ids[i])))
    sums: list[float] = []
    counts: list[int] = []
    prev = object()
    for i in order:
        lab = str(clusters[i])
        if lab != prev:
            sums.append(0.0)
            counts.append(0)
            prev = lab
        sums[-1] += float(d[i])
        counts[-1] += 1
    return np.asarray(sums, dtype=float), np.asarray(counts, dtype=np.int64)


def _analytic(S, m_c, n, C, estimate, alpha, alternative):
    if C < 2:
        nan = float("nan")
        return AnalyticResult(nan, nan, C - 1, nan, nan, nan)
    resid = S - m_c * estimate
    se = float(np.sqrt(C / (C - 1) * np.sum(resid**2)) / n)
    df = C - 1
    tcrit = float(_sps.t.ppf(1 - alpha / 2, df))
    if se > 0:
        t_stat = estimate / se
        if alternative == "two-sided":
            p = float(2 * _sps.t.sf(abs(t_stat), df))
        elif alternative == "greater":
            p = float(_sps.t.sf(t_stat, df))
        else:
            p = float(_sps.t.cdf(t_stat, df))
    else:
        t_stat = 0.0 if estimate == 0 else float(np.sign(estimate)) * float("inf")
        if alternative == "two-sided":
            p = 1.0 if estimate == 0 else 0.0
        elif alternative == "greater":
            p = 1.0 if estimate <= 0 else 0.0
        else:
            p = 1.0 if estimate >= 0 else 0.0
    return AnalyticResult(se, t_stat, df, estimate - tcrit * se, estimate + tcrit * se, p)


def _perm_pvalue(t_star, t_obs, alternative, mc: bool, n_perm: int) -> float:
    """§2.2: exact float comparisons; the identity assignment is counted."""
    if alternative == "two-sided":
        hits = np.abs(t_star) >= abs(t_obs)
    elif alternative == "greater":
        hits = t_star >= t_obs
    else:
        hits = t_star <= t_obs
    if mc:
        return (1 + int(np.count_nonzero(hits))) / (n_perm + 1)
    return float(np.mean(hits))


def run_engine(
    d,
    clusters=None,
    item_ids=None,
    *,
    seed: int,
    alpha: float = 0.05,
    n_boot: int = 10_000,
    n_perm: int = 10_000,
    enumerate_threshold: int = 13,
    alternative: str = "two-sided",
) -> EngineResult:
    """Run the §2 inference engine on item-level differences ``d``.

    Parameters
    ----------
    d : array-like of item-level differences ``d_i`` (§1.2).
    clusters : per-item cluster ids (``source_doc_id``); None means every
        item is its own singleton cluster (§1.4).
    item_ids : per-item ids, used only for the deterministic sort (§0).
    seed : explicit RNG seed; required, no global state (§0).
    alternative : "two-sided" (default), or a pre-declared one-sided test
        (DECISION D1). One-sided p-values carry O(skew/sqrt(C)) level error
        under a mean-zero-but-asymmetric null (FLAGGED F12, §2.2); the CI is
        always two-sided.
    """
    d = np.asarray(d, dtype=float)
    if d.ndim != 1 or d.size == 0:
        raise ValueError("d must be a non-empty 1-d array of item differences")
    if not np.all(np.isfinite(d)):
        raise ValueError("d contains non-finite values")
    if alternative not in _ALTERNATIVES:
        raise ValueError(f"alternative must be one of {_ALTERNATIVES}")
    n = int(d.size)
    if clusters is not None and len(clusters) != n:
        raise ValueError("clusters must align with d")
    if item_ids is not None and len(item_ids) != n:
        raise ValueError("item_ids must align with d")
    if n_boot < 2 or n_perm < 1:
        raise ValueError("n_boot >= 2 and n_perm >= 1 required")

    S, m_c = _sorted_cluster_stats(d, clusters, item_ids)
    C = int(S.size)
    estimate = float(np.sum(S) / n)

    warnings: list[str] = []
    if C < 20:
        warnings.append(
            f"C={C} clusters (< 20): bootstrap and cluster-robust intervals are "
            f"unreliable; permutation p-value granularity >= 2^-C (§2.4, F4)"
        )
    degenerate = bool(np.all(d == d[0]))
    if degenerate:
        warnings.append("all item-level differences identical: CI collapses to a point (§2.1)")
    if C < 2:
        warnings.append("C=1: analytic cluster-robust interval undefined (§2.3)")

    rng = make_rng(seed)

    # --- §2.1 cluster percentile bootstrap (consumes the stream first) ---
    idx = rng.integers(0, C, size=(n_boot, C))
    boot = S[idx].sum(axis=1) / m_c[idx].sum(axis=1)
    q = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0], method="linear")
    ci_low, ci_high = float(q[0]), float(q[1])
    boot_se = float(np.std(boot, ddof=1))

    # --- §2.2 cluster sign-flip permutation test ---
    # T* is always computed via the same (signs @ S) / n path so that the
    # identity assignment reproduces the observed statistic bit-for-bit and
    # the required exact float comparison counts it (§2.2).
    if C <= enumerate_threshold:
        bits = (np.arange(2**C, dtype=np.int64)[:, None] >> np.arange(C)) & 1
        signs = 2.0 * bits - 1.0
        t_all = signs @ S / n
        t_obs = float(t_all[-1])  # all-ones row == identity assignment
        p_value = _perm_pvalue(t_all, t_obs, alternative, mc=False, n_perm=n_perm)
        test_method = "sign-flip-exact"
    else:
        signs = 2.0 * rng.integers(0, 2, size=(n_perm, C)) - 1.0
        t_obs = float(np.ones(C) @ S / n)
        t_star = signs @ S / n
        p_value = _perm_pvalue(t_star, t_obs, alternative, mc=True, n_perm=n_perm)
        test_method = "sign-flip-mc"

    analytic = _analytic(S, m_c, n, C, estimate, alpha, alternative)

    sizes, size_counts = np.unique(m_c, return_counts=True)
    cluster_sizes = {int(s): int(c) for s, c in zip(sizes, size_counts)}

    return EngineResult(
        estimate=estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        boot_se=boot_se,
        p_value=float(p_value),
        test_method=test_method,
        alternative=alternative,
        n_items=n,
        n_clusters=C,
        cluster_sizes=cluster_sizes,
        analytic=analytic,
        degenerate=degenerate,
        warnings=tuple(warnings),
        alpha=alpha,
        n_boot=n_boot,
        n_perm=n_perm,
        seed=seed,
    )
