"""Scenario front-ends (docs/statistics-spec.md §3, §4, §5) emitting the §10
report block: reduction -> engine -> scenario-specific diagnostics, achieved
MDE, and mandatory caveats."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from math import sqrt

import numpy as np
from scipy import stats as _sps

from .engine import AnalyticResult, EngineResult, run_engine
from .mcnemar import discordant_counts, mcnemar_exact
from .reduction import MATERIAL_SKEW_THRESHOLD, reduce_pairwise, reduce_scores, sample_skewness

__all__ = ["JUDGE_RELATIVITY_CAVEAT", "ComparisonReport", "compare_binary",
           "compare_pairwise", "compare_scalar"]

# Scope limitation, required verbatim-or-equivalent on every report (spec preamble).
JUDGE_RELATIVITY_CAVEAT = (
    "All results are relative to the judge configuration used: read as "
    "'A beats B as scored by judge J', never 'A is better than B'. "
    "Judge-level systematic error is not reduced by more items or replicate "
    "judgments; it is measured in bias/ and calibrated in agreement/."
)


@dataclass(frozen=True)
class ComparisonReport:
    """§10 machine-readable report block. Estimand-scale fields (estimate,
    CI, MDE) are on the scenario's reporting scale (theta for scenario A);
    ``engine`` and ``analytic`` stay on the d scale."""

    scenario: str
    estimand: str
    estimate: float
    ci_low: float
    ci_high: float
    ci_level: float
    ci_method: str
    p_value: float
    test_method: str
    alternative: str
    n_items: int
    n_clusters: int
    cluster_sizes: dict
    n_excluded_items: int
    boot_se: float
    analytic: AnalyticResult
    mde_80: float
    sd_d: float
    extras: dict
    warnings: tuple
    caveats: tuple
    alpha: float
    n_boot: int
    n_perm: int
    seed: int
    engine: EngineResult = field(repr=False)

    def to_dict(self) -> dict:
        out = asdict(self)
        del out["engine"]
        return out


def _achieved_mde(analytic: AnalyticResult, alpha: float, power: float = 0.8) -> float:
    """Achieved MDE at 80% power from the cluster-robust SE (§9): the number
    that makes a null result honest. NaN when the SE is unavailable."""
    if not np.isfinite(analytic.se) or analytic.df < 1:
        return float("nan")
    return float((_sps.t.ppf(1 - alpha / 2, analytic.df) + _sps.t.ppf(power, analytic.df)) * analytic.se)


def _one_sided_skew_warning(alternative: str, d) -> list[str]:
    g = sample_skewness(d)
    if alternative != "two-sided" and abs(g) > MATERIAL_SKEW_THRESHOLD:
        return [
            f"one-sided test with materially skewed differences (sample skewness "
            f"{g:.2f}): one-sided sign-flip p-values carry O(skew/sqrt(C)) level "
            f"error under a mean-zero-but-asymmetric null (F12, §2.2)"
        ]
    return []


def compare_scalar(records, variant_a: str, variant_b: str, *, seed: int,
                   alpha: float = 0.05, n_boot: int = 10_000, n_perm: int = 10_000,
                   alternative: str = "two-sided",
                   scale: tuple[float, float] | None = None) -> ComparisonReport:
    """Scenario B (§4): scalar rubric scores, Delta in raw rubric units."""
    records = list(records)
    red = reduce_scores(records, variant_a, variant_b)
    if red.n_items == 0:
        raise ValueError("no items with responses for both variants")
    eng = run_engine(red.d, red.clusters, red.item_ids, seed=seed, alpha=alpha,
                     n_boot=n_boot, n_perm=n_perm, alternative=alternative)
    warnings = list(red.warnings) + list(eng.warnings) + _one_sided_skew_warning(alternative, red.d)

    # §4 / F3: per-variant score distribution so scale degeneracy is visible.
    dist = {}
    for v in (variant_a, variant_b):
        vals = Counter(float(r["judgment"] if isinstance(r, dict) else r.judgment)
                       for r in records
                       if (r["variant_id"] if isinstance(r, dict) else r.variant_id) == v)
        dist[v] = dict(sorted(vals.items()))
        total = sum(vals.values())
        if total and max(vals.values()) / total > 0.5:
            top = max(vals, key=vals.get)
            warnings.append(
                f"variant {v!r}: >50% of judgments on a single scale point "
                f"({top}); scale is effectively binary (F3, §4)"
            )
    extras = {
        "mean_a": float(np.mean(red.x_a)),
        "mean_b": float(np.mean(red.x_b)),
        "score_distribution": dist,
        "d_skewness": red.d_skewness,
        "side_unbalanced_items": red.side_unbalanced_items,
    }
    if scale is not None:
        lo, hi = scale
        extras["ceiling_fraction"] = {
            v: (sum(c for val, c in dist[v].items() if val == hi) / max(1, sum(dist[v].values())))
            for v in (variant_a, variant_b)
        }
    return _report("B-scalar", "Delta = E[x_iA - x_iB] in raw rubric units (§4)",
                   eng.estimate, eng, red, extras, warnings, alpha, seed,
                   mde=_achieved_mde(eng.analytic, alpha),
                   sd_d=float(np.std(red.d, ddof=1)) if red.n_items > 1 else float("nan"))


def compare_binary(records, variant_a: str, variant_b: str, *, seed: int,
                   alpha: float = 0.05, n_boot: int = 10_000, n_perm: int = 10_000,
                   alternative: str = "two-sided") -> ComparisonReport:
    """Scenario C (§5): binary pass/fail, Delta = pass-rate difference."""
    records = list(records)
    for r in records:
        j = float(r["judgment"] if isinstance(r, dict) else r.judgment)
        if j not in (0.0, 1.0):
            raise ValueError(f"binary judgment must be 0 or 1, got {j}")
    red = reduce_scores(records, variant_a, variant_b)
    if red.n_items == 0:
        raise ValueError("no items with responses for both variants")
    eng = run_engine(red.d, red.clusters, red.item_ids, seed=seed, alpha=alpha,
                     n_boot=n_boot, n_perm=n_perm, alternative=alternative)
    warnings = list(red.warnings) + list(eng.warnings) + _one_sided_skew_warning(alternative, red.d)
    extras = {
        "pass_rate_a": float(np.mean(red.x_a)),
        "pass_rate_b": float(np.mean(red.x_b)),
        "d_skewness": red.d_skewness,
        "side_unbalanced_items": red.side_unbalanced_items,
    }
    single = bool(np.all(np.isin(red.x_a, (0.0, 1.0))) and np.all(np.isin(red.x_b, (0.0, 1.0))))
    if single:
        n_pos, n_neg, _ = discordant_counts(red.d)
        n_d = n_pos + n_neg
        xa, xb = red.x_a, red.x_b
        extras["concordance"] = {
            "n_pass_pass": int(np.sum((xa == 1) & (xb == 1))),
            "n_pass_fail": n_pos,
            "n_fail_pass": n_neg,
            "n_fail_fail": int(np.sum((xa == 0) & (xb == 0))),
        }
        extras["n_discordant"] = n_d
        if n_d < 10:
            warnings.append(
                f"only {n_d} discordant items: CI unstable, run underpowered; "
                f"read the discordant counts directly (§2.4, §5)"
            )
        if eng.n_clusters == eng.n_items:
            # Required classical cross-check (§5): singletons + single judgments.
            extras["mcnemar_p"] = mcnemar_exact(n_pos, n_neg)
    return _report("C-binary", "Delta = p_A - p_B, paired pass-rate difference (§5)",
                   eng.estimate, eng, red, extras, warnings, alpha, seed,
                   mde=_achieved_mde(eng.analytic, alpha),
                   sd_d=float(np.std(red.d, ddof=1)) if red.n_items > 1 else float("nan"))


def compare_pairwise(records, variant_a: str, variant_b: str, *, seed: int,
                     alpha: float = 0.05, n_boot: int = 10_000, n_perm: int = 10_000,
                     alternative: str = "two-sided") -> ComparisonReport:
    """Scenario A (§3): pairwise preference. Reported on the win-rate scale
    theta = E[s_i] (H0: theta = 0.5, ties 0.5 per DECISION D2); the engine
    runs on d = 2s - 1 and the report maps back."""
    records = list(records)
    red = reduce_pairwise(records, variant_a, variant_b)
    if red.n_items == 0:
        raise ValueError("no items with usable pairs for these variants")
    eng = run_engine(red.d, red.clusters, red.item_ids, seed=seed, alpha=alpha,
                     n_boot=n_boot, n_perm=n_perm, alternative=alternative)
    warnings = list(red.warnings) + list(eng.warnings) + _one_sided_skew_warning(alternative, red.d)
    if np.isfinite(red.tie_rate) and red.tie_rate > 0.3:
        warnings.append(
            f"tie rate {red.tie_rate:.0%} (> 30%): win rate compresses toward 0.5; "
            f"the judge rarely discriminates — a finding about the judge, not "
            f"evidence of variant equivalence (§3, D2)"
        )
    extras = {
        "theta": float(np.mean(red.s)),
        "net_preference": eng.estimate,
        "tie_rate": red.tie_rate,
        "counterbalanced": red.counterbalanced,
        "n_pairs": red.n_pairs,
        "n_pairs_single_order": red.n_pairs_single_order,
        "scale_note": "engine/analytic fields are on the d = 2s - 1 scale",
    }
    rep = _report("A-pairwise", "win rate theta = E[s_i], ties 0.5 (§3, D2); H0: theta = 0.5",
                  (1.0 + eng.estimate) / 2.0, eng, red, extras, warnings, alpha, seed,
                  mde=_achieved_mde(eng.analytic, alpha) / 2.0,
                  sd_d=float(np.std(red.s, ddof=1)) if red.n_items > 1 else float("nan"),
                  ci=((1.0 + eng.ci_low) / 2.0, (1.0 + eng.ci_high) / 2.0),
                  boot_se=eng.boot_se / 2.0)
    return rep


def _report(scenario, estimand, estimate, eng: EngineResult, red, extras,
            warnings, alpha, seed, *, mde, sd_d, ci=None, boot_se=None) -> ComparisonReport:
    ci_low, ci_high = ci if ci is not None else (eng.ci_low, eng.ci_high)
    n_excluded = getattr(red, "n_excluded", 0)
    return ComparisonReport(
        scenario=scenario,
        estimand=estimand,
        estimate=float(estimate),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        ci_level=1.0 - alpha,
        ci_method="cluster-percentile-bootstrap (§2.1)",
        p_value=eng.p_value,
        test_method=eng.test_method,
        alternative=eng.alternative,
        n_items=eng.n_items,
        n_clusters=eng.n_clusters,
        cluster_sizes=eng.cluster_sizes,
        n_excluded_items=n_excluded,
        boot_se=float(boot_se if boot_se is not None else eng.boot_se),
        analytic=eng.analytic,
        mde_80=float(mde),
        sd_d=float(sd_d),
        extras=extras,
        warnings=tuple(warnings),
        caveats=(JUDGE_RELATIVITY_CAVEAT,),
        alpha=alpha,
        n_boot=eng.n_boot,
        n_perm=eng.n_perm,
        seed=seed,
        engine=eng,
    )
