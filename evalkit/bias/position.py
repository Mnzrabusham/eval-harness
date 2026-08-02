"""Position bias (docs/statistics-spec.md §14.2).

``beta_pos`` is the additive marginal position effect: the average
increment, in probability points, to a response's chance of being
preferred from being displayed first. For a pair judged in both orders the
order-balanced preference-for-first score has expectation 1/2 + beta_p
regardless of content preference and tie propensity, so
``beta_hat = mean_i(g_i) - 1/2``. Uncertainty comes from the §2 engine on
``d = 2g - 1`` (E[d] = 2*beta_pos), mapped back by halving.
"""

from __future__ import annotations

from ._core import BiasReport, build_bias_report, order_balanced_reduce

__all__ = ["F13_DISCLOSURE", "position_bias"]

F13_DISCLOSURE = (
    "beta_pos is the additive marginal position effect (F13, §14.2): a "
    "position x variant interaction and any non-additive position structure "
    "are not separately identified from counterbalanced preference data; "
    "they widen the interval without moving the point estimate."
)


def _pref_first(rec, judgment: str) -> float:
    if judgment == "first":
        return 1.0
    if judgment == "second":
        return 0.0
    return 0.5


def position_bias(records, *, seed: int, alpha: float = 0.05,
                  n_boot: int = 10_000, n_perm: int = 10_000) -> BiasReport:
    """Estimate beta_pos from counterbalanced pairwise judgment records."""
    red = order_balanced_reduce(list(records), _pref_first)
    if red.n_items == 0:
        raise ValueError(
            "no pairs judged in both presentation orders: position bias is "
            "unidentified without counterbalancing (§14.1)"
        )
    diagnostics = {
        "tie_rate": red.tie_rate,
        "judge_models": red.judge_models,
        "n_pairs_used": red.n_pairs_used,
        "n_pairs_single_order_excluded": red.n_pairs_single_order,
        "single_order_pair_ids_excluded": red.single_order_pair_ids,
        "n_items_dropped": red.n_items_dropped,
        "n_calls_used": red.n_calls_used,
        "n_calls_same_variant_skipped": red.n_calls_same_variant,
    }
    return build_bias_report(
        2.0 * red.g - 1.0, red.clusters, red.item_ids,
        scale=0.5, offset=0.0,
        quantity="position-bias",
        estimand=("beta_pos = E_i[g_i] - 1/2: additive marginal increment to "
                  "the probability of preferring the first-shown response, in "
                  "probability points (§14.2)"),
        identified=True,
        null_hypothesis="beta_pos = 0 (no additive position effect on any pair)",
        diagnostics=diagnostics,
        warnings=red.warnings,
        disclosures=(F13_DISCLOSURE,),
        seed=seed, alpha=alpha, n_boot=n_boot, n_perm=n_perm,
    )
