"""Item-level reduction (docs/statistics-spec.md §1.2, §1.4).

Raw judgment rows reduce to one number per item, ``d_i in [-1, 1]`` (or raw
rubric units for scalar), with ``E[d_i] = 0`` under "no difference". The
item — or its source-document cluster — is the unit of all downstream
inference.

Records are mappings (or attribute objects) using the field names of
docs/data-model.md: ResponseJudgment for scalar/binary, PairwiseJudgment
for preference.

F11 side-balance check (§2.2): the sign-flip test's symmetry-by-construction
argument requires the same (r, m) replicate structure on both sides of every
item. ``reduce_scores`` detects per-item structure mismatches and warns,
prominently when the observed differences are also materially skewed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy import stats as _sps

__all__ = [
    "MATERIAL_SKEW_THRESHOLD",
    "PairwiseReduction",
    "ScoreReduction",
    "reduce_pairwise",
    "reduce_scores",
]

# §2.2 / F11: |sample skewness| above this makes a side-balance warning
# "prominent" (the mean-zero null is then materially outside exact validity).
MATERIAL_SKEW_THRESHOLD = 1.0

_SINGLETON = "__singleton__:"


def _get(rec, key, default=KeyError):
    if isinstance(rec, Mapping):
        if key in rec:
            return rec[key]
    elif hasattr(rec, key):
        return getattr(rec, key)
    if default is KeyError:
        raise KeyError(f"record missing required field {key!r}")
    return default


def _cluster_key(rec, item: str) -> str:
    doc = _get(rec, "source_doc_id", default=None)
    return str(doc) if doc not in (None, "") else _SINGLETON + item


def sample_skewness(d) -> float:
    """Bias-corrected sample skewness; 0.0 when undefined (n < 3 or no spread)."""
    d = np.asarray(d, dtype=float)
    if d.size < 3 or np.all(d == d[0]):
        return 0.0
    g = float(_sps.skew(d, bias=False))
    return g if np.isfinite(g) else 0.0


@dataclass(frozen=True)
class ScoreReduction:
    """Output of §1.2 for scalar/binary outcomes."""

    d: np.ndarray
    x_a: np.ndarray
    x_b: np.ndarray
    item_ids: tuple
    clusters: tuple
    n_items: int
    n_excluded: int
    excluded_item_ids: tuple
    side_unbalanced_items: tuple
    d_skewness: float
    warnings: tuple
    n_records_used: int
    n_records_ignored: int


def reduce_scores(records, variant_a: str, variant_b: str) -> ScoreReduction:
    """§1.2 scalar/binary reduction: judge calls -> response means -> item
    means -> ``d_i = x_iA - x_iB``. Responses get equal weight regardless of
    how many judge calls each received."""
    if variant_a == variant_b:
        raise ValueError("variant_a and variant_b must differ")
    per: dict[str, dict[str, dict[str, list[float]]]] = {}
    item_cluster: dict[str, str] = {}
    used = ignored = 0
    for rec in records:
        v = _get(rec, "variant_id")
        if v not in (variant_a, variant_b):
            ignored += 1
            continue
        item = str(_get(rec, "item_id"))
        ckey = _cluster_key(rec, item)
        if item_cluster.setdefault(item, ckey) != ckey:
            raise ValueError(f"item {item!r} appears with conflicting source_doc_id")
        y = float(_get(rec, "judgment"))
        resp = str(_get(rec, "response_id"))
        per.setdefault(item, {}).setdefault(v, {}).setdefault(resp, []).append(y)
        used += 1

    item_ids: list[str] = []
    clusters: list[str] = []
    d: list[float] = []
    xa: list[float] = []
    xb: list[float] = []
    excluded: list[str] = []
    unbalanced: list[str] = []
    for item in sorted(per):
        sides = per[item]
        if variant_a not in sides or variant_b not in sides:
            excluded.append(item)
            continue
        x = {}
        struct = {}
        for v in (variant_a, variant_b):
            responses = sides[v]
            resp_means = [float(np.mean(ys)) for _, ys in sorted(responses.items())]
            x[v] = float(np.mean(resp_means))
            # (r, m) structure: multiset of judge-call counts across responses
            struct[v] = tuple(sorted(len(ys) for ys in responses.values()))
        if struct[variant_a] != struct[variant_b]:
            unbalanced.append(item)
        item_ids.append(item)
        clusters.append(item_cluster[item])
        xa.append(x[variant_a])
        xb.append(x[variant_b])
        d.append(x[variant_a] - x[variant_b])

    d_arr = np.asarray(d, dtype=float)
    skew = sample_skewness(d_arr)
    warnings: list[str] = []
    if excluded:
        warnings.append(
            f"{len(excluded)} items excluded for missing one variant's responses "
            f"(§1.2); systematic missingness biases the estimand"
        )
    if unbalanced:
        head = ", ".join(unbalanced[:8]) + (", ..." if len(unbalanced) > 8 else "")
        base = (
            f"side-unbalanced replicate structure (r, m) between variants on "
            f"{len(unbalanced)} of {len(item_ids)} items [{head}] (F11, §2.2)"
        )
        if abs(skew) > MATERIAL_SKEW_THRESHOLD:
            warnings.append(
                "PROMINENT: " + base + f": item-level differences are materially "
                f"skewed (sample skewness {skew:.2f}); sign-flip exactness for the "
                f"mean-zero null is compromised"
            )
        else:
            warnings.append(base + ": d_i symmetry under H0 is not guaranteed if scores are skewed")

    return ScoreReduction(
        d=d_arr,
        x_a=np.asarray(xa, dtype=float),
        x_b=np.asarray(xb, dtype=float),
        item_ids=tuple(item_ids),
        clusters=tuple(clusters),
        n_items=len(item_ids),
        n_excluded=len(excluded),
        excluded_item_ids=tuple(excluded),
        side_unbalanced_items=tuple(unbalanced),
        d_skewness=skew,
        warnings=tuple(warnings),
        n_records_used=used,
        n_records_ignored=ignored,
    )


@dataclass(frozen=True)
class PairwiseReduction:
    """Output of §1.2 for pairwise preference (scenario A)."""

    d: np.ndarray  # 2*s_i - 1
    s: np.ndarray  # order-balanced item scores
    item_ids: tuple
    clusters: tuple
    n_items: int
    n_pairs: int
    n_pairs_single_order: int
    counterbalanced: bool
    tie_rate: float
    warnings: tuple
    n_records_used: int
    n_records_ignored: int


def reduce_pairwise(records, variant_a: str, variant_b: str) -> PairwiseReduction:
    """§1.2 pairwise reduction with counterbalanced presentation: recode to
    preference-for-A (ties 0.5, DECISION D2), average within order first,
    then across orders (this removes any additive position effect even under
    unequal call counts), then across pairs."""
    if variant_a == variant_b:
        raise ValueError("variant_a and variant_b must differ")
    per: dict[str, dict[str, dict[str, list[float]]]] = {}
    item_cluster: dict[str, str] = {}
    used = ignored = ties = 0
    for rec in records:
        vf = _get(rec, "variant_first")
        vs = _get(rec, "variant_second")
        if {vf, vs} != {variant_a, variant_b}:
            ignored += 1
            continue
        j = _get(rec, "judgment")
        if j not in ("first", "second", "tie"):
            raise ValueError(f"pairwise judgment must be first|second|tie, got {j!r}")
        a_first = vf == variant_a
        if j == "tie":
            y = 0.5
            ties += 1
        elif j == "first":
            y = 1.0 if a_first else 0.0
        else:
            y = 0.0 if a_first else 1.0
        item = str(_get(rec, "item_id"))
        ckey = _cluster_key(rec, item)
        if item_cluster.setdefault(item, ckey) != ckey:
            raise ValueError(f"item {item!r} appears with conflicting source_doc_id")
        pair = str(_get(rec, "pair_id"))
        order = "A-first" if a_first else "B-first"
        per.setdefault(item, {}).setdefault(pair, {}).setdefault(order, []).append(y)
        used += 1

    item_ids: list[str] = []
    clusters: list[str] = []
    s_list: list[float] = []
    n_pairs = single_order = 0
    for item in sorted(per):
        pair_scores = []
        for pair in sorted(per[item]):
            orders = per[item][pair]
            n_pairs += 1
            order_means = [float(np.mean(v)) for _, v in sorted(orders.items())]
            if len(order_means) == 1:
                single_order += 1
            pair_scores.append(float(np.mean(order_means)))
        item_ids.append(item)
        clusters.append(item_cluster[item])
        s_list.append(float(np.mean(pair_scores)))

    s = np.asarray(s_list, dtype=float)
    counterbalanced = n_pairs > 0 and single_order == 0
    tie_rate = ties / used if used else float("nan")
    warnings: list[str] = []
    if not counterbalanced and n_pairs > 0:
        warnings.append(
            f"{single_order} of {n_pairs} pairs judged in only one presentation "
            f"order: not counterbalanced; position bias uncorrected (F1, §1.2)"
        )
    return PairwiseReduction(
        d=2.0 * s - 1.0,
        s=s,
        item_ids=tuple(item_ids),
        clusters=tuple(clusters),
        n_items=len(item_ids),
        n_pairs=n_pairs,
        n_pairs_single_order=single_order,
        counterbalanced=counterbalanced,
        tie_rate=tie_rate,
        warnings=tuple(warnings),
        n_records_used=used,
        n_records_ignored=ignored,
    )
