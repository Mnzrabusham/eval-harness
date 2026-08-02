"""Verbosity: length-preference association vs verbosity bias (§14.3, F14).

``gamma_verb`` — the increment in preference probability attributable to
greater length with quality held fixed — is NOT identified from
observational judgment data: the observable association satisfies
``A_len - 1/2 = gamma_verb + (quality-length component)`` with the
decomposition (including the sign of gamma_verb) unknown. This module
therefore exposes exactly two things:

- ``verbosity_association``: the descriptive order-balanced win rate of
  the longer response. Per the F14 naming rule it is named an association
  everywhere and carries ``identified=False``; no public symbol containing
  "verbosity_bias" returns it.
- ``verbosity_bias_controlled``: gamma from a designed length-manipulation
  study (content-matched original-vs-padded and original-vs-condensed
  rewrites), reporting both arm estimates and their paired difference as
  the manipulation check on assumption (ii).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..stats.reduction import _get
from ._core import BiasReport, build_bias_report, order_balanced_reduce

__all__ = [
    "ASSOCIATION_DISCLOSURES",
    "CONTROLLED_DISCLOSURES",
    "VerbosityControlledReport",
    "verbosity_association",
    "verbosity_bias_controlled",
]

ASSOCIATION_DISCLOSURES = (
    "This is a length-preference ASSOCIATION, not a verbosity-bias estimate "
    "(F14, §14.3): A_len - 1/2 = gamma_verb + (quality-length component), and "
    "the decomposition — including the sign of gamma_verb — is unidentified "
    "from observational judgment data.",
    "A verbosity-bias estimate requires a controlled length-manipulation "
    "study; see verbosity_bias_controlled (§14.3).",
)

CONTROLLED_DISCLOSURES = (
    "Valid under: (i) the rewrites are quality-preserving; (ii) the judge "
    "responds to length, not to detectable rewrite artifacts. Assumption "
    "(ii) is checked by the padded-vs-condensed manipulation check (§14.3).",
    "Transportability is assumed, not identified (F14, §14.3): gamma "
    "measured at the manipulated length gaps on this item set need not "
    "equal the judge's length effect on natural pairs, and it says nothing "
    "about verbosity bias in scalar scoring.",
)


def _pref_longer(rec, judgment: str):
    """Recode to preference-for-longer; None when lengths are equal."""
    tf = int(_get(rec, "tokens_first"))
    ts = int(_get(rec, "tokens_second"))
    if tf == ts:
        return None
    if judgment == "tie":
        return 0.5
    first_longer = tf > ts
    if judgment == "first":
        return 1.0 if first_longer else 0.0
    return 0.0 if first_longer else 1.0


def _length_diagnostics(records) -> dict:
    """Per-variant token-count summaries (call-weighted: a response judged
    in both orders contributes twice) plus the length-gap summary."""
    per_variant: dict[str, list[int]] = {}
    gaps: list[int] = []
    for rec in records:
        tf = int(_get(rec, "tokens_first"))
        ts = int(_get(rec, "tokens_second"))
        per_variant.setdefault(str(_get(rec, "variant_first")), []).append(tf)
        per_variant.setdefault(str(_get(rec, "variant_second")), []).append(ts)
        gaps.append(abs(tf - ts))
    summary = {
        v: {"n_calls": len(xs), "mean": float(np.mean(xs)),
            "min": int(min(xs)), "max": int(max(xs))}
        for v, xs in sorted(per_variant.items())
    }
    return {
        "token_counts_per_variant": summary,
        "abs_token_gap_mean": float(np.mean(gaps)) if gaps else float("nan"),
    }


def _reduction_diagnostics(red) -> dict:
    return {
        "tie_rate": red.tie_rate,
        "judge_models": red.judge_models,
        "n_pairs_used": red.n_pairs_used,
        "n_pairs_single_order_excluded": red.n_pairs_single_order,
        "n_items_dropped": red.n_items_dropped,
        "n_calls_used": red.n_calls_used,
        "n_calls_equal_length_excluded": red.n_calls_skipped,
        "n_calls_same_variant_skipped": red.n_calls_same_variant,
    }


def verbosity_association(records, *, seed: int, alpha: float = 0.05,
                          n_boot: int = 10_000, n_perm: int = 10_000) -> BiasReport:
    """Descriptive length-preference association A_len (NOT a bias, F14).

    A_len = order-balanced win rate of the longer response, ties 1/2,
    equal-length pairs excluded and counted. H0: A_len = 1/2.
    """
    records = list(records)
    red = order_balanced_reduce(records, _pref_longer)
    if red.n_items == 0:
        raise ValueError(
            "no length-discordant pairs judged in both presentation orders: "
            "the length-preference association is not estimable (§14.3)"
        )
    diagnostics = _reduction_diagnostics(red)
    diagnostics.update(_length_diagnostics(records))
    return build_bias_report(
        2.0 * red.g - 1.0, red.clusters, red.item_ids,
        scale=0.5, offset=0.5,
        quantity="verbosity-association",
        estimand=("A_len = E_i[g_i]: order-balanced win rate of the longer "
                  "response — a descriptive association, not a bias (§14.3, F14)"),
        identified=False,
        null_hypothesis="A_len = 1/2 (no net length preference on any pair)",
        diagnostics=diagnostics,
        warnings=red.warnings,
        disclosures=ASSOCIATION_DISCLOSURES,
        seed=seed, alpha=alpha, n_boot=n_boot, n_perm=n_perm,
    )


@dataclass(frozen=True)
class VerbosityControlledReport:
    """§14.3 controlled-study report: arm estimates plus manipulation check.

    ``manipulation_check`` is the paired per-item difference
    gamma_padded - gamma_condensed on items with both arms; an interval
    away from 0 means the judge is detecting the manipulation (or a rewrite
    changed quality) and the pooled gamma is not a clean length effect.
    Arms absent from the design are None.
    """

    gamma_pooled: BiasReport
    gamma_padded: BiasReport | None
    gamma_condensed: BiasReport | None
    manipulation_check: BiasReport | None
    warnings: tuple


def _arm_records(records, original_variant: str, manipulated_variant: str) -> list:
    want = frozenset((str(original_variant), str(manipulated_variant)))
    out = [r for r in records
           if frozenset((str(_get(r, "variant_first")), str(_get(r, "variant_second")))) == want]
    if not out:
        raise ValueError(f"no records for arm {original_variant!r} vs {manipulated_variant!r}")
    return out


def _gamma_report(records, label: str, seed: int, alpha, n_boot, n_perm):
    red = order_balanced_reduce(records, _pref_longer)
    if red.n_items == 0:
        raise ValueError(
            f"arm {label!r}: no length-discordant pairs judged in both orders"
        )
    report = build_bias_report(
        2.0 * red.g - 1.0, red.clusters, red.item_ids,
        scale=0.5, offset=0.0,
        quantity=f"verbosity-bias-controlled-{label}",
        estimand=(f"gamma ({label} arm) = A_len - 1/2 on content-matched, "
                  f"length-manipulated pairs (§14.3)"),
        identified=True,
        null_hypothesis="gamma = 0 (no length effect on any pair)",
        diagnostics=_reduction_diagnostics(red),
        warnings=red.warnings,
        disclosures=CONTROLLED_DISCLOSURES,
        seed=seed, alpha=alpha, n_boot=n_boot, n_perm=n_perm,
    )
    return red, report


def verbosity_bias_controlled(records, *, original_variant: str,
                              padded_variant: str | None = None,
                              condensed_variant: str | None = None,
                              seed: int, alpha: float = 0.05,
                              n_boot: int = 10_000, n_perm: int = 10_000) -> VerbosityControlledReport:
    """gamma from a controlled length-manipulation study (§14.3).

    Records must come from a designed study whose arms are encoded as
    variants: ``original_variant`` vs a padded (longer) rewrite and/or vs a
    condensed (shorter) rewrite of the same response. Engine seeds are
    derived deterministically: seed (pooled), seed+1 (padded), seed+2
    (condensed), seed+3 (manipulation check).
    """
    if padded_variant is None and condensed_variant is None:
        raise ValueError("at least one manipulation arm (padded_variant or "
                         "condensed_variant) is required")
    records = list(records)
    warnings: list[str] = []
    kw = dict(alpha=alpha, n_boot=n_boot, n_perm=n_perm)

    red_pad = rep_pad = red_cond = rep_cond = None
    pooled_records: list = []
    if padded_variant is not None:
        arm = _arm_records(records, original_variant, padded_variant)
        pooled_records += arm
        red_pad, rep_pad = _gamma_report(arm, "padded", seed + 1, **kw)
    if condensed_variant is not None:
        arm = _arm_records(records, original_variant, condensed_variant)
        pooled_records += arm
        red_cond, rep_cond = _gamma_report(arm, "condensed", seed + 2, **kw)
    _, rep_pooled = _gamma_report(pooled_records, "pooled", seed, **kw)

    check = None
    if red_pad is not None and red_cond is not None:
        g_pad = dict(zip(red_pad.item_ids, red_pad.g))
        g_cond = dict(zip(red_cond.item_ids, red_cond.g))
        cl_pad = dict(zip(red_pad.item_ids, red_pad.clusters))
        cl_cond = dict(zip(red_cond.item_ids, red_cond.clusters))
        common = sorted(set(g_pad) & set(g_cond))
        if not common:
            warnings.append("manipulation check unavailable: no items carry "
                            "both arms (§14.3); assumption (ii) is UNCHECKED")
        else:
            for item in common:
                if cl_pad[item] != cl_cond[item]:
                    raise ValueError(f"item {item!r} has conflicting source_doc_id across arms")
            d_check = np.asarray([g_pad[i] - g_cond[i] for i in common], dtype=float)
            check = build_bias_report(
                d_check, [cl_pad[i] for i in common], common,
                scale=1.0, offset=0.0,
                quantity="verbosity-manipulation-check",
                estimand=("gamma_padded - gamma_condensed, paired per item on "
                          "items carrying both arms (§14.3)"),
                identified=True,
                null_hypothesis=("gamma_padded = gamma_condensed (pure length "
                                 "response; no artifact detection)"),
                diagnostics={
                    "n_items_both_arms": len(common),
                    "n_items_padded_only": len(g_pad) - len(common),
                    "n_items_condensed_only": len(g_cond) - len(common),
                },
                warnings=(),
                disclosures=(
                    "A check interval away from 0 means the judge is detecting "
                    "the manipulation (or a rewrite changed quality); the pooled "
                    "gamma is then not a clean length effect (§14.3).",
                ),
                seed=seed + 3, **kw,
            )
            if check.ci_low > 0.0 or check.ci_high < 0.0:
                warnings.append(
                    f"manipulation check rejects agreement "
                    f"(gamma_padded - gamma_condensed = {check.estimate:+.3f}, "
                    f"CI [{check.ci_low:+.3f}, {check.ci_high:+.3f}] excludes 0): "
                    f"the judge is detecting the manipulation or a rewrite "
                    f"changed quality; the pooled gamma is not a clean length "
                    f"effect (§14.3)"
                )
    else:
        warnings.append(
            "only one manipulation arm present: assumption (ii) — the judge "
            "responds to length, not rewrite artifacts — is UNCHECKED "
            "(F14, §14.3); run both padded and condensed arms"
        )
    return VerbosityControlledReport(
        gamma_pooled=rep_pooled,
        gamma_padded=rep_pad,
        gamma_condensed=rep_cond,
        manipulation_check=check,
        warnings=tuple(warnings),
    )
