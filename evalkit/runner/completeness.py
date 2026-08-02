"""Pairwise-pair completeness reporting for a resumable run.

A pairwise run with some pairs judged in only one presentation order is a
normal, resumable intermediate state -- a run still in progress, or one
that hit a transient failure on the missing order -- not an error.
Nothing here raises; it only reports.

The place incompleteness actually matters is the reduction boundary
(``evalkit.bias._core.order_balanced_reduce``), which already excludes
single-order pairs from any bias estimand (DECISION D10, §14.1): averaging
a one-sided pair in would silently reintroduce the position bias the
estimand exists to cancel. This module is for operational visibility into
*why* a bias estimand's excluded count is what it is, before that point,
not a gate on the run itself.
"""

from __future__ import annotations

from typing import Iterable, Mapping

__all__ = ["incomplete_pairs"]


def _field(rec, name: str):
    if isinstance(rec, Mapping):
        return rec[name]
    return getattr(rec, name)


def incomplete_pairs(records: Iterable) -> dict[str, str]:
    """``pair_id -> the single presentation order seen`` for incomplete pairs.

    ``records`` are ``evalkit.judge.PairwiseJudgment`` instances or their
    dict form (e.g. from ``JsonlStore.read_raw()``). The presentation
    order is identified by ``variant_first``: a pair_id absent from the
    returned mapping has been judged with both variants appearing first at
    least once and is complete; a pair_id present has only ever been
    judged with the one named variant first.
    """
    orders_seen: dict[str, set[str]] = {}
    for rec in records:
        pair_id = str(_field(rec, "pair_id"))
        orders_seen.setdefault(pair_id, set()).add(str(_field(rec, "variant_first")))
    return {pid: next(iter(orders)) for pid, orders in orders_seen.items() if len(orders) < 2}
