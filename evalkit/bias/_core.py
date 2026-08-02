"""Shared machinery for bias estimators (docs/statistics-spec.md §14).

Every estimator here reduces judgment records to an item-level vector and
hands it to the §2 engine (``evalkit.stats.engine.run_engine``) unchanged:
bias/ contains no resampling or permutation code of its own. Point
estimates and intervals are linear transforms of the engine's d-scale
outputs.

§14.1 reduction: recode each call to z in {0, 1/2, 1}, average within
(pair, order), then across the two presentation orders, then across pairs,
then items. Both presentation orders are REQUIRED here (DECISION D10,
unlike §1.2/F1): the across-order average is what cancels the nuisance
term, so single-order pairs are excluded from the estimand and counted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from ..stats.coverage_evidence import coverage_disclosure, primary_ci_method
from ..stats.engine import EngineResult, run_engine
from ..stats.reduction import _cluster_key, _get

__all__ = [
    "BiasReport",
    "OrderBalancedReduction",
    "build_bias_report",
    "order_balanced_reduce",
]


@dataclass(frozen=True)
class OrderBalancedReduction:
    """Output of the §14.1 order-balanced recode reduction."""

    g: np.ndarray  # item-level order-balanced scores in [0, 1]
    item_ids: tuple
    clusters: tuple
    n_items: int
    n_items_dropped: int  # items whose every pair lacked one order
    n_pairs_used: int
    n_pairs_single_order: int  # excluded from the estimand (§14.1)
    single_order_pair_ids: tuple  # the pair_ids behind n_pairs_single_order
    n_calls_used: int
    n_calls_skipped: int  # recode returned None (e.g. equal lengths)
    n_calls_same_variant: int  # a variant paired with itself: no order key
    tie_rate: float
    judge_models: tuple
    warnings: tuple


def order_balanced_reduce(records, code) -> OrderBalancedReduction:
    """Reduce pairwise judgment records with the call-level recode ``code``.

    ``code(rec, judgment)`` returns the call's score z in [0, 1], or None
    to exclude the call (excluded calls are counted). Averaging follows
    §1.2/§14.1: within (pair, order) -> across orders -> across pairs ->
    item. Pairs observed in only one presentation order are excluded from
    the estimand (§14.1, D10) and counted.
    """
    per: dict[str, dict[str, dict]] = {}
    item_cluster: dict[str, str] = {}
    judges: set[str] = set()
    used = skipped = same_variant = ties = 0
    for rec in records:
        judgment = _get(rec, "judgment")
        if judgment not in ("first", "second", "tie"):
            raise ValueError(f"pairwise judgment must be first|second|tie, got {judgment!r}")
        vf = str(_get(rec, "variant_first"))
        vs = str(_get(rec, "variant_second"))
        if vf == vs:
            same_variant += 1
            continue
        z = code(rec, judgment)
        if z is None:
            skipped += 1
            continue
        item = str(_get(rec, "item_id"))
        ckey = _cluster_key(rec, item)
        if item_cluster.setdefault(item, ckey) != ckey:
            raise ValueError(f"item {item!r} appears with conflicting source_doc_id")
        pair = str(_get(rec, "pair_id"))
        slot = per.setdefault(item, {}).setdefault(
            pair, {"vset": frozenset((vf, vs)), "orders": {}})
        if slot["vset"] != frozenset((vf, vs)):
            raise ValueError(f"pair {pair!r} on item {item!r} appears with conflicting variant pairs")
        slot["orders"].setdefault(vf, []).append(float(z))
        judges.add(str(_get(rec, "judge_model", default="<unrecorded>")))
        if judgment == "tie":
            ties += 1
        used += 1

    item_ids: list[str] = []
    clusters: list[str] = []
    g_list: list[float] = []
    single_order_pairs: list[str] = []
    pairs_used = items_dropped = 0
    for item in sorted(per):
        pair_scores = []
        for pair in sorted(per[item]):
            orders = per[item][pair]["orders"]
            if len(orders) < 2:
                single_order_pairs.append(pair)
                continue
            order_means = [float(np.mean(v)) for _, v in sorted(orders.items())]
            pair_scores.append(float(np.mean(order_means)))
            pairs_used += 1
        if not pair_scores:
            items_dropped += 1
            continue
        item_ids.append(item)
        clusters.append(item_cluster[item])
        g_list.append(float(np.mean(pair_scores)))

    single_order = len(single_order_pairs)
    warnings: list[str] = []
    if single_order:
        # PROMINENT: this is data silently leaving the estimand, not a soft
        # signal like F11's skew-conditional escalation -- a one-sided pair
        # included as-is would reintroduce the position bias averaging
        # across orders exists to cancel, so every affected pair_id is named
        # (capped, as in reduction.py's F11 warning) alongside the count.
        head = ", ".join(single_order_pairs[:8]) + (", ..." if single_order > 8 else "")
        warnings.append(
            f"PROMINENT: {single_order} pairs excluded from the bias estimand "
            f"[{head}] for having only one presentation order judged "
            f"({items_dropped} items dropped entirely because every one of "
            f"their pairs was excluded); both orders are required for "
            f"identification (§14.1, D10)"
        )
    if len(judges) > 1:
        warnings.append(
            f"records pool {len(judges)} judge configurations "
            f"({', '.join(sorted(judges))}): the estimate is a mixture across "
            f"judges, not any single judge's bias (§14)"
        )
    return OrderBalancedReduction(
        g=np.asarray(g_list, dtype=float),
        item_ids=tuple(item_ids),
        clusters=tuple(clusters),
        n_items=len(item_ids),
        n_items_dropped=items_dropped,
        n_pairs_used=pairs_used,
        n_pairs_single_order=single_order,
        single_order_pair_ids=tuple(single_order_pairs),
        n_calls_used=used,
        n_calls_skipped=skipped,
        n_calls_same_variant=same_variant,
        tie_rate=ties / used if used else float("nan"),
        judge_models=tuple(sorted(judges)),
        warnings=tuple(warnings),
    )


@dataclass(frozen=True)
class BiasReport:
    """§14.6 report block for one bias/association estimate.

    ``quantity`` follows the F14 naming rule: descriptive associations are
    named "...-association" and carry ``identified=False``. Estimate, CIs,
    and boot_se are on the quantity's reporting scale; ``engine`` stays on
    the d scale.
    """

    quantity: str
    estimand: str
    identified: bool
    estimate: float
    ci_low: float
    ci_high: float
    ci_level: float
    ci_method: str
    bootstrap_ci: tuple
    analytic_ci: tuple
    boot_se: float
    p_value: float
    test_method: str
    null_hypothesis: str
    n_items: int
    n_clusters: int
    cluster_sizes: dict
    diagnostics: dict
    warnings: tuple
    disclosures: tuple
    alpha: float
    n_boot: int
    n_perm: int
    seed: int
    engine: EngineResult = field(repr=False)

    def to_dict(self) -> dict:
        out = asdict(self)
        del out["engine"]
        return out


def build_bias_report(d, clusters, item_ids, *, scale: float, offset: float,
                      quantity: str, estimand: str, identified: bool,
                      null_hypothesis: str, diagnostics: dict, warnings,
                      disclosures, seed: int, alpha: float = 0.05,
                      n_boot: int = 10_000, n_perm: int = 10_000) -> BiasReport:
    """Run the §2 engine on ``d`` and map its outputs by x -> scale*x + offset.

    The engine's H0 (E[d] = 0) must correspond to the quantity's null under
    the mapping, i.e. the null value is ``offset`` on the reporting scale;
    the p-value carries over unchanged. Primary-CI selection follows
    DECISION D9 exactly as in stats/scenarios.py.
    """
    if scale <= 0:
        raise ValueError("scale must be positive so interval order is preserved")
    eng = run_engine(d, list(clusters), list(item_ids), seed=seed, alpha=alpha,
                     n_boot=n_boot, n_perm=n_perm)

    def f(x: float) -> float:
        return scale * x + offset

    boot_ci = (f(eng.ci_low), f(eng.ci_high))
    analytic_ci = (f(eng.analytic.ci_low), f(eng.analytic.ci_high))
    warnings = list(warnings) + list(eng.warnings)
    primary = primary_ci_method(eng.n_clusters)
    if primary == "analytic" and not (np.isfinite(analytic_ci[0]) and np.isfinite(analytic_ci[1])):
        primary = "bootstrap"
        warnings.append("analytic interval unavailable (C < 2); falling back to "
                        "bootstrap despite C below the D9 analytic-primary threshold")
    if primary == "analytic":
        ci_low, ci_high = analytic_ci
        ci_method = "analytic-cluster-robust-t (§2.3); primary below C=50 per DECISION D9"
    else:
        ci_low, ci_high = boot_ci
        ci_method = "cluster-percentile-bootstrap (§2.1)"
    diagnostics = dict(diagnostics)
    diagnostics["ci_report"] = {
        "primary_method": primary,
        "bootstrap_ci": boot_ci,
        "analytic_ci": analytic_ci,
        "coverage_disclosure": coverage_disclosure(eng.n_clusters),
    }
    return BiasReport(
        quantity=quantity,
        estimand=estimand,
        identified=identified,
        estimate=f(eng.estimate),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        ci_level=1.0 - alpha,
        ci_method=ci_method,
        bootstrap_ci=boot_ci,
        analytic_ci=analytic_ci,
        boot_se=scale * eng.boot_se,
        p_value=eng.p_value,
        test_method=eng.test_method,
        null_hypothesis=null_hypothesis,
        n_items=eng.n_items,
        n_clusters=eng.n_clusters,
        cluster_sizes=eng.cluster_sizes,
        diagnostics=diagnostics,
        warnings=tuple(warnings),
        disclosures=tuple(disclosures),
        alpha=alpha,
        n_boot=n_boot,
        n_perm=n_perm,
        seed=seed,
        engine=eng,
    )
