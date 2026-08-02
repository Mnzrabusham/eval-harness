"""Self-preference bias (docs/statistics-spec.md §14.4, F15).

Identified by a cross-judge contrast on identical pairs:
``sigma_self = theta_self^(J) - mean_K theta_self^(K)``, where each theta
is the order-balanced win rate of the self variant (§1.2 reduction, ties
1/2) as scored by one judge on the same pairs. Genuine quality of the self
model's outputs appears in every judge's score and cancels in the paired
per-item difference; the naive ``theta_self^(J) - 1/2`` does not have that
property and is exposed only as a labeled diagnostic.
"""

from __future__ import annotations

import numpy as np

from ..stats.reduction import _get, reduce_pairwise
from ._core import BiasReport, build_bias_report

__all__ = ["F15_DISCLOSURES", "self_preference_bias"]

F15_DISCLOSURES = (
    "Assumes panel neutrality (§14.4): the other judges are on average "
    "unbiased toward or against the self model on these pairs. Any "
    "panel-shared style preference correlated with the self model's style "
    "survives the contrast and biases the estimate one-for-one; a "
    "same-family panel is the worst case.",
    "Judge-idiosyncratic taste that aligns with the self model's style is "
    "operationally indistinguishable from self-recognition; the contrast "
    "attributes it to self-preference (F15, §14.4).",
)

_NAIVE_LABEL = ("theta_self^(J) - 1/2 confounds self-preference with the self "
                "model's genuine quality; diagnostic only, NOT a bias estimate "
                "(§14.4)")


def self_preference_bias(records, *, self_judge: str, self_variant: str,
                         other_variant: str, seed: int, alpha: float = 0.05,
                         n_boot: int = 10_000, n_perm: int = 10_000) -> BiasReport:
    """Estimate sigma_self from multi-judge pairwise records on shared pairs.

    ``self_judge`` is the judge_model whose bias is measured;
    ``self_variant`` is the variant produced by that judge's own model.
    Records from all other judge_models form the panel. Items lacking
    either the self judge's score or at least one panel score are excluded
    and counted. Position effects cancel provided every judge saw the same
    counterbalanced orders (§14.4).
    """
    by_judge: dict[str, list] = {}
    for rec in records:
        by_judge.setdefault(str(_get(rec, "judge_model")), []).append(rec)
    if self_judge not in by_judge:
        raise ValueError(f"no records from self_judge {self_judge!r}")
    others = sorted(k for k in by_judge if k != self_judge)
    if not others:
        raise ValueError(
            "cross-judge contrast requires judgments from at least one judge "
            "other than self_judge: the naive self win rate confounds bias "
            "with genuine quality and is not a bias estimate (§14.4)"
        )

    red_self = reduce_pairwise(by_judge[self_judge], self_variant, other_variant)
    warnings = [f"judge {self_judge!r}: {w}" for w in red_self.warnings]
    s_self = dict(zip(red_self.item_ids, red_self.s))
    cluster_of = dict(zip(red_self.item_ids, red_self.clusters))

    panel_scores: dict[str, list[float]] = {}
    for judge in others:
        red_k = reduce_pairwise(by_judge[judge], self_variant, other_variant)
        warnings += [f"judge {judge!r}: {w}" for w in red_k.warnings]
        for item, s, cluster in zip(red_k.item_ids, red_k.s, red_k.clusters):
            if cluster_of.setdefault(item, cluster) != cluster:
                raise ValueError(f"item {item!r} has conflicting source_doc_id across judges")
            panel_scores.setdefault(item, []).append(float(s))

    common = sorted(set(s_self) & set(panel_scores))
    if not common:
        raise ValueError("no items judged by both self_judge and the panel")
    d = np.asarray([s_self[i] - float(np.mean(panel_scores[i])) for i in common],
                   dtype=float)
    judges_per_item = [len(panel_scores[i]) for i in common]

    if len(others) == 1:
        warnings.append(
            f"single other judge ({others[0]!r}): panel neutrality rests "
            f"entirely on that judge (F15, §14.4)"
        )
    diagnostics = {
        "self_judge": self_judge,
        "self_variant": self_variant,
        "other_judges": tuple(others),
        "n_other_judges": len(others),
        "panel_judges_per_item_min": int(min(judges_per_item)),
        "panel_judges_per_item_max": int(max(judges_per_item)),
        "n_items_self_judge_only_excluded": len(s_self) - len(common),
        "n_items_panel_only_excluded": len(panel_scores) - len(common),
        "naive_self_theta_minus_half": float(np.mean(red_self.s)) - 0.5,
        "naive_label": _NAIVE_LABEL,
        "tie_rate_self_judge": red_self.tie_rate,
    }
    return build_bias_report(
        d, [cluster_of[i] for i in common], common,
        scale=1.0, offset=0.0,
        quantity="self-preference-bias",
        estimand=("sigma_self = theta_self^(J) - mean_K theta_self^(K) on "
                  "identical pairs, in win-rate points (§14.4)"),
        identified=True,
        null_hypothesis=("sigma_self = 0 (self judge exchangeable with the "
                         "panel on every item)"),
        diagnostics=diagnostics,
        warnings=warnings,
        disclosures=F15_DISCLOSURES,
        seed=seed, alpha=alpha, n_boot=n_boot, n_perm=n_perm,
    )
