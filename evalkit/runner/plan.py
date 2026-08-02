"""Dry-run planning: report call counts before spending anything.

A dry run reconstructs the same task/call lists a real run would build
(generation tasks, pairwise judge calls, response judge calls) and
classifies each one by what would happen: skipped because the store
already has the record (resumed from a prior run), skipped because the
cache already has the completion (paid for in a prior run but not yet
turned into a record), or an actual call against the model. Only
read-only cache/store lookups happen -- no model client is touched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, TypeVar

from .cache import ResponseCache
from .generate import GenerationTask, generation_response_id
from .judging import PairwiseJudgeCall, ResponseJudgeCall, pairwise_call_id, response_call_id, validate_judge_model
from .store import JsonlStore

__all__ = ["DryRunReport", "dry_run"]

T = TypeVar("T")


@dataclass(frozen=True)
class DryRunReport:
    generation_total: int
    generation_already_recorded: int
    generation_cache_hits: int
    generation_model_calls: int
    pairwise_total: int
    pairwise_already_recorded: int
    pairwise_cache_hits: int
    pairwise_model_calls: int
    response_total: int
    response_already_recorded: int
    response_cache_hits: int
    response_model_calls: int

    @property
    def total_model_calls(self) -> int:
        return self.generation_model_calls + self.pairwise_model_calls + self.response_model_calls


@dataclass(frozen=True)
class _Tally:
    total: int
    already_recorded: int
    cache_hits: int
    model_calls: int


def _classify(tasks: Iterable[T], *, natural_key_fn: Callable[[T], str],
              store: JsonlStore | None, store_key_field: str,
              cache_lookup_fn: Callable[[T], object] | None) -> _Tally:
    tasks = list(tasks)
    done_ids = store.written_keys(lambda row: row[store_key_field]) if store is not None else set()
    already_recorded = 0
    cache_hits = 0
    model_calls = 0
    for task in tasks:
        if natural_key_fn(task) in done_ids:
            already_recorded += 1
            continue
        if cache_lookup_fn is not None and cache_lookup_fn(task) is not None:
            cache_hits += 1
            continue
        model_calls += 1
    return _Tally(len(tasks), already_recorded, cache_hits, model_calls)


def dry_run(*, generation_tasks: Iterable[GenerationTask] = (),
            pairwise_calls: Iterable[PairwiseJudgeCall] = (),
            response_calls: Iterable[ResponseJudgeCall] = (),
            generation_cache: ResponseCache | None = None,
            generation_store: JsonlStore | None = None,
            judge_cache: ResponseCache | None = None,
            pairwise_store: JsonlStore | None = None,
            response_store: JsonlStore | None = None) -> DryRunReport:
    generation_tasks = list(generation_tasks)
    pairwise_calls = list(pairwise_calls)
    response_calls = list(response_calls)

    for call in pairwise_calls:
        validate_judge_model(call.judge_model)
    for call in response_calls:
        validate_judge_model(call.judge_model)

    gen = _classify(
        generation_tasks, natural_key_fn=generation_response_id,
        store=generation_store, store_key_field="response_id",
        cache_lookup_fn=(
            (lambda t: generation_cache.get(t.prompt, model=t.model, seed=t.seed,
                                            sampling_params=t.sampling_params))
            if generation_cache is not None else None
        ),
    )
    pw = _classify(
        pairwise_calls, natural_key_fn=pairwise_call_id,
        store=pairwise_store, store_key_field="judge_call_id",
        cache_lookup_fn=(
            (lambda c: judge_cache.get(c.task.prompt, model=c.judge_model, seed=c.judge_seed,
                                       sampling_params=c.judge_sampling_params))
            if judge_cache is not None else None
        ),
    )
    resp = _classify(
        response_calls, natural_key_fn=response_call_id,
        store=response_store, store_key_field="judge_call_id",
        cache_lookup_fn=(
            (lambda c: judge_cache.get(c.prompt, model=c.judge_model, seed=c.judge_seed,
                                       sampling_params=c.judge_sampling_params))
            if judge_cache is not None else None
        ),
    )

    return DryRunReport(
        generation_total=gen.total, generation_already_recorded=gen.already_recorded,
        generation_cache_hits=gen.cache_hits, generation_model_calls=gen.model_calls,
        pairwise_total=pw.total, pairwise_already_recorded=pw.already_recorded,
        pairwise_cache_hits=pw.cache_hits, pairwise_model_calls=pw.model_calls,
        response_total=resp.total, response_already_recorded=resp.already_recorded,
        response_cache_hits=resp.cache_hits, response_model_calls=resp.model_calls,
    )
