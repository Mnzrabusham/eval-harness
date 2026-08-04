"""The pairwise request plan: D4 replicate draw, PlannedJudgeCall, and the
gap-9 gate (docs/data-model.md; spec §8.2 DECISION D4, §12 gap 9, DECISION
D11).

Three responsibilities, in the order a run actually uses them:

1. ``draw_replicate_item_subset`` -- DECISION D4's replicate-judging subset
   is drawn per *item*, never per response: the same drawn items get the
   extra replicate call mirrored onto every side, so every item keeps an
   identical per-side (r, m) structure. A per-response draw was the
   original wording and is rejected by the spec because it manufactures
   the F11 side-unbalance condition on exactly the items it touches.
2. ``build_pairwise_request_plan`` -- turns a set of already-built
   counterbalanced task pairs into the full call list (``PairwiseJudgeCall``,
   what the runner will actually execute) and the full plan (
   ``PlannedJudgeCall``, what gets written to disk *before* any of those
   calls are made). Both share one deterministic ``judge_call_id`` per
   call, computed by ``pairwise_call_id``.
3. ``gate_pairwise_realized`` -- after judging runs, joins the realized
   ``PairwiseJudgment`` records back to the plan and classifies every
   (pair, order) into the four gap-9 cases.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from evalkit.judge import PairwiseTask, PlannedJudgeCall

from .judging import PairwiseJudgeCall, pairwise_call_id
from .store import JsonlStore

__all__ = [
    "PairwiseCallGate",
    "PairwiseRequestPlan",
    "build_pairwise_request_plan",
    "draw_replicate_item_subset",
    "gate_pairwise_realized",
    "replicate_subset_size",
    "write_planned_calls",
]


# --- D4 item-level replicate draw ------------------------------------------


def replicate_subset_size(n_items: int, *, min_responses: int = 10, max_responses: int = 30,
                          fraction: float = 0.2, responses_per_item: int = 2) -> int:
    """Item count for DECISION D4's replicate subset.

    Spec §8.2: sized so at least ``min_responses`` (and at most
    ``min(max_responses, fraction * total_responses)``) responses receive
    replicate judgments. ``responses_per_item`` converts that
    response-count target into an item count -- for a single counterbalanced
    pair per item, replicating an item means judging both presentation
    orders again, i.e. 2 responses per item.
    """
    if n_items <= 0:
        return 0
    total_responses = n_items * responses_per_item
    target_responses = max(min_responses, min(max_responses, round(fraction * total_responses)))
    target_responses = min(target_responses, total_responses)
    return -(-target_responses // responses_per_item)  # ceil division


def draw_replicate_item_subset(item_ids: Iterable[str], *, seed: int,
                               min_responses: int = 10, max_responses: int = 30,
                               fraction: float = 0.2,
                               responses_per_item: int = 2) -> frozenset[str]:
    """The seed-drawn item subset that gets replicate-judged (DECISION D4).

    Drawing at the item level -- and, in ``build_pairwise_request_plan``,
    mirroring the replicate call onto both presentation orders of every
    drawn item -- is what keeps the design side-balanced by construction.
    Sorting before sampling makes the draw a pure function of ``item_ids``'
    contents, not the order they happen to arrive in.
    """
    unique_items = sorted(set(str(i) for i in item_ids))
    n_items = len(unique_items)
    target_items = replicate_subset_size(
        n_items, min_responses=min_responses, max_responses=max_responses,
        fraction=fraction, responses_per_item=responses_per_item,
    )
    if target_items <= 0:
        return frozenset()
    rng = random.Random(seed)
    return frozenset(rng.sample(unique_items, k=target_items))


# --- plan construction -------------------------------------------------------


@dataclass(frozen=True)
class PairwiseRequestPlan:
    """The output of planning a pairwise run: what to call, and what was planned.

    ``calls`` is what the runner executes; ``planned`` is what gets written
    to disk first (``write_planned_calls``), before any of those calls are
    made, so the gate has something immutable to join realized records
    against. ``replicated_items`` is the D4 draw that produced both.
    """

    calls: tuple[PairwiseJudgeCall, ...]
    planned: tuple[PlannedJudgeCall, ...]
    replicated_items: frozenset[str]


def build_pairwise_request_plan(item_tasks: Mapping[str, tuple[PairwiseTask, PairwiseTask]], *,
                                run_id: str, judge_model: str, judge_config_id: str,
                                judge_seed: int, judge_sampling_params: Mapping[str, Any],
                                created_at: str, replicate_seed: int,
                                min_replicate_responses: int = 10,
                                max_replicate_responses: int = 30,
                                replicate_fraction: float = 0.2) -> PairwiseRequestPlan:
    """Build the full call list and plan for one item's worth of pairwise judging.

    ``item_tasks`` maps ``item_id`` to the ``(first-order, second-order)``
    task pair produced by ``evalkit.judge.counterbalanced_pairwise_tasks`` --
    this function does not build tasks itself, matching ``judging.py``'s
    existing convention. Every item gets a primary call on both orders;
    items drawn by ``draw_replicate_item_subset`` additionally get a
    replicate call on *both* orders, never just one -- that mirroring is
    the side-balance guarantee DECISION D4 requires.
    """
    item_tasks = dict(item_tasks)
    replicated_items = draw_replicate_item_subset(
        item_tasks.keys(), seed=replicate_seed,
        min_responses=min_replicate_responses, max_responses=max_replicate_responses,
        fraction=replicate_fraction, responses_per_item=2,
    )

    calls: list[PairwiseJudgeCall] = []
    planned: list[PlannedJudgeCall] = []
    for item_id in sorted(item_tasks):
        t1, t2 = item_tasks[item_id]
        roles = ["primary", "replicate"] if item_id in replicated_items else ["primary"]
        for task in (t1, t2):
            for replicate, call_role in enumerate(roles):
                call = PairwiseJudgeCall(
                    task=task, replicate=replicate, call_role=call_role,
                    judge_model=judge_model, judge_config_id=judge_config_id,
                    judge_seed=judge_seed, judge_sampling_params=judge_sampling_params,
                    created_at=created_at,
                )
                calls.append(call)
                planned.append(PlannedJudgeCall(
                    run_id=run_id, item_id=task.item_id, source_doc_id=task.source_doc_id,
                    pair_id=task.pair_id, variant_first=task.first.variant_id,
                    variant_second=task.second.variant_id, judge_model=judge_model,
                    judge_config_id=judge_config_id, call_role=call_role,
                    judge_call_id=pairwise_call_id(call), created_at=created_at,
                ))

    return PairwiseRequestPlan(calls=tuple(calls), planned=tuple(planned),
                               replicated_items=replicated_items)


def write_planned_calls(planned: Iterable[PlannedJudgeCall], store: JsonlStore) -> None:
    """Persist the plan before any judge call is made (spec §12 gap 9)."""
    for call in planned:
        store.append(call)


# --- the gap-9 gate -----------------------------------------------------------


def _field(rec, name: str):
    if isinstance(rec, Mapping):
        return rec[name]
    return getattr(rec, name)


@dataclass(frozen=True)
class PairwiseCallGate:
    """Per-(pair, order) classification of realized records against the plan.

    Every ``(pair_id, variant_first)`` key seen in either the plan or the
    realized records falls into exactly one bucket (spec §12 gap 9,
    DECISION D11):

    - ``normal`` -- the planned primary call succeeded.
    - ``promotions`` -- the planned primary exists but has no successful
      record; a planned replicate succeeded instead. Counted, does not
      block the run.
    - ``plumbing_errors`` -- a realized record with no matching planned
      call, or a (pair, order) with records but no planned primary.
      Counted separately, blocks the run.
    - ``exclusions`` -- neither the planned primary nor any planned
      replicate has a successful record; falls to the D10 exclusion path.
    """

    normal: tuple[tuple[str, str], ...]
    promotions: tuple[tuple[str, str], ...]
    plumbing_errors: tuple[tuple[str, str], ...]
    exclusions: tuple[tuple[str, str], ...]

    @property
    def promotion_count(self) -> int:
        return len(self.promotions)

    @property
    def plumbing_error_count(self) -> int:
        return len(self.plumbing_errors)

    @property
    def exclusion_count(self) -> int:
        return len(self.exclusions)

    @property
    def blocks_run(self) -> bool:
        return self.plumbing_error_count > 0


def gate_pairwise_realized(planned: Iterable, realized: Iterable) -> PairwiseCallGate:
    """Join realized PairwiseJudgment records to the PlannedJudgeCall plan.

    ``planned`` and ``realized`` accept either the dataclass instances or
    their dict form (e.g. from ``JsonlStore.read_raw()``). Promotion is
    never written back onto any record -- it is derived here, from records
    plus plan, every time the gate runs (data-model.md Constraints).
    """
    planned = list(planned)
    realized = list(realized)
    planned_ids = {_field(p, "judge_call_id") for p in planned}
    realized_ids = {_field(r, "judge_call_id") for r in realized}

    planned_groups: dict[tuple[str, str], list] = {}
    for p in planned:
        key = (str(_field(p, "pair_id")), str(_field(p, "variant_first")))
        planned_groups.setdefault(key, []).append(p)

    realized_groups: dict[tuple[str, str], list] = {}
    for r in realized:
        key = (str(_field(r, "pair_id")), str(_field(r, "variant_first")))
        realized_groups.setdefault(key, []).append(r)

    normal: list[tuple[str, str]] = []
    promotions: list[tuple[str, str]] = []
    plumbing_errors: list[tuple[str, str]] = []
    exclusions: list[tuple[str, str]] = []

    for key in sorted(set(planned_groups) | set(realized_groups)):
        p_group = planned_groups.get(key, [])
        r_group = realized_groups.get(key, [])

        unmatched = [r for r in r_group if _field(r, "judge_call_id") not in planned_ids]
        primaries = [p for p in p_group if _field(p, "call_role") == "primary"]

        if unmatched or (r_group and not primaries):
            plumbing_errors.append(key)
            continue
        if not primaries:
            # Nothing planned and nothing realized under this key: can't
            # actually happen given how the groups are built (a key only
            # exists because something landed in one side or the other),
            # but this is not a case any bucket above claims.
            continue

        primary_id = _field(primaries[0], "judge_call_id")
        if primary_id in realized_ids:
            normal.append(key)
            continue

        replicate_ids = {_field(p, "judge_call_id") for p in p_group
                         if _field(p, "call_role") == "replicate"}
        if replicate_ids & realized_ids:
            promotions.append(key)
        else:
            exclusions.append(key)

    return PairwiseCallGate(
        normal=tuple(normal), promotions=tuple(promotions),
        plumbing_errors=tuple(plumbing_errors), exclusions=tuple(exclusions),
    )
