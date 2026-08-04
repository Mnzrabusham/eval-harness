"""Judge execution: turn judge module tasks into recorded judgments.

Pairwise judge calls always wrap a ``evalkit.judge.PairwiseTask`` built by
``counterbalanced_pairwise_tasks`` -- this module does not construct
pairwise tasks itself, so both presentation orders of a pair are always
in play under one pair_id, never one in isolation.

``judge_call_id`` (and the ``response_id``/``pair_id`` its derived from)
is deterministic, not assigned by the runner arbitrarily: a replicate is
identified by an explicit ``replicate`` index rather than a counter or
random id, so re-running the same configuration reproduces the same
judge_call_id and is recognized as already-recorded (data-model.md:
replicate judge calls are separate rows, each independently resumable).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from evalkit.judge import (
    PairwiseJudgment,
    PairwiseTask,
    ResponseJudgment,
    pairwise_judgment_from_task,
    response_judgment_from_verdict,
)

from .cache import ResponseCache
from .client import ModelClient
from .concurrency import run_concurrent
from .execute import get_or_generate
from .ids import derive_id, derive_seed
from .retry import RetryPolicy
from .store import JsonlStore

__all__ = [
    "PairwiseJudgeCall",
    "ResponseJudgeCall",
    "pairwise_call_id",
    "response_call_id",
    "run_pairwise_judging",
    "run_response_judging",
    "validate_judge_model",
]


def validate_judge_model(judge_model: str) -> None:
    """Pre-flight check for the data-model.md judge_model constraint.

    ``PairwiseJudgment``/``ResponseJudgment`` already enforce this at
    construction (evalkit.judge.records); this lets a dry run (or any
    other pre-flight check) reject a floating alias before a single model
    call is made, not just before the record would be written.
    """
    if str(judge_model).endswith("-latest"):
        raise ValueError(
            f"judge_model must be an immutable snapshot id; floating alias "
            f"{judge_model!r} rejected (docs/data-model.md)"
        )


@dataclass(frozen=True)
class PairwiseJudgeCall:
    """One judge invocation to make for a PairwiseTask.

    ``replicate`` distinguishes independent judge calls on the same task
    (data-model.md: replicates are separate rows sharing everything but
    judge_call_id). ``call_role`` is the request-creation-time label (spec
    §12 gap 9, DECISION D11) carried onto the realized PairwiseJudgment
    unchanged -- it is not derived here or anywhere downstream.
    """

    task: PairwiseTask
    replicate: int
    call_role: str
    judge_model: str
    judge_config_id: str
    judge_seed: int
    judge_sampling_params: Mapping[str, Any]
    created_at: str
    annotator_id: str | None = None


@dataclass(frozen=True)
class ResponseJudgeCall:
    """One judge invocation to make for a single response (scalar/binary)."""

    item_id: str
    source_doc_id: str | None
    variant_id: str
    response_id: str
    gen_seed: object
    response_tokens: int
    prompt: str
    judgment_type: str  # "scalar" | "binary"
    replicate: int
    judge_model: str
    judge_config_id: str
    judge_seed: int
    judge_sampling_params: Mapping[str, Any]
    created_at: str
    scale_min: float | None = None
    scale_max: float | None = None
    annotator_id: str | None = None


def pairwise_call_id(call: PairwiseJudgeCall) -> str:
    return derive_id(call.task.pair_id, call.task.first.response_id,
                     call.task.second.response_id, call.judge_config_id, call.replicate)


def response_call_id(call: ResponseJudgeCall) -> str:
    return derive_id(call.response_id, call.judge_config_id, call.replicate)


def _run_one_pairwise(call: PairwiseJudgeCall, *, client: ModelClient, cache: ResponseCache,
                       retry_policy: RetryPolicy, run_id: str) -> PairwiseJudgment:
    validate_judge_model(call.judge_model)
    judge_call_id = pairwise_call_id(call)
    completion, _cached = get_or_generate(
        client, cache, call.task.prompt, model=call.judge_model, seed=call.judge_seed,
        sampling_params=call.judge_sampling_params, retry_policy=retry_policy,
        retry_seed=derive_seed(judge_call_id, "retry"),
    )
    return pairwise_judgment_from_task(
        call.task, completion.text, run_id=run_id, judge_model=call.judge_model,
        judge_config_id=call.judge_config_id, judge_call_id=judge_call_id,
        call_role=call.call_role, created_at=call.created_at, annotator_id=call.annotator_id,
    )


def run_pairwise_judging(calls: Iterable[PairwiseJudgeCall], *, run_id: str, client: ModelClient,
                          cache: ResponseCache, store: JsonlStore, retry_policy: RetryPolicy,
                          concurrency: int) -> list[PairwiseJudgment]:
    """Judge every call's pair, skipping ones already in ``store``.

    Returns the full set of judgments for ``calls`` in order -- some read
    back from ``store`` (already done, possibly in an earlier, interrupted
    run), some freshly judged and appended to ``store`` in this call. A
    partial list here would silently understate a resumed run's evidence
    to any downstream reduction; see ``run_generation`` for the same
    contract on the generation side.
    """
    calls = list(calls)
    existing = {row["judge_call_id"]: PairwiseJudgment(**row) for row in store.read_raw()}

    ids_in_order = [pairwise_call_id(c) for c in calls]
    pending = [c for c, jid in zip(calls, ids_in_order) if jid not in existing]

    def _run(call: PairwiseJudgeCall) -> PairwiseJudgment:
        return _run_one_pairwise(call, client=client, cache=cache, retry_policy=retry_policy, run_id=run_id)

    for _call, judgment in run_concurrent(pending, _run, limit=concurrency):
        store.append(judgment)
        existing[judgment.judge_call_id] = judgment

    return [existing[jid] for jid in ids_in_order]


def _run_one_response(call: ResponseJudgeCall, *, client: ModelClient, cache: ResponseCache,
                       retry_policy: RetryPolicy, run_id: str) -> ResponseJudgment:
    validate_judge_model(call.judge_model)
    judge_call_id = response_call_id(call)
    completion, _cached = get_or_generate(
        client, cache, call.prompt, model=call.judge_model, seed=call.judge_seed,
        sampling_params=call.judge_sampling_params, retry_policy=retry_policy,
        retry_seed=derive_seed(judge_call_id, "retry"),
    )
    return response_judgment_from_verdict(
        completion.text, judgment_type=call.judgment_type, run_id=run_id,
        item_id=call.item_id, source_doc_id=call.source_doc_id, variant_id=call.variant_id,
        response_id=call.response_id, gen_seed=call.gen_seed, response_tokens=call.response_tokens,
        judge_model=call.judge_model, judge_config_id=call.judge_config_id,
        judge_call_id=judge_call_id, created_at=call.created_at,
        scale_min=call.scale_min, scale_max=call.scale_max, annotator_id=call.annotator_id,
    )


def run_response_judging(calls: Iterable[ResponseJudgeCall], *, run_id: str, client: ModelClient,
                          cache: ResponseCache, store: JsonlStore, retry_policy: RetryPolicy,
                          concurrency: int) -> list[ResponseJudgment]:
    """Judge every call's response, skipping ones already in ``store``.

    Returns the full set of judgments for ``calls`` in order; see
    ``run_pairwise_judging`` for why this must include already-recorded
    matches, not just ones written in this call.
    """
    calls = list(calls)
    existing = {row["judge_call_id"]: ResponseJudgment(**row) for row in store.read_raw()}

    ids_in_order = [response_call_id(c) for c in calls]
    pending = [c for c, jid in zip(calls, ids_in_order) if jid not in existing]

    def _run(call: ResponseJudgeCall) -> ResponseJudgment:
        return _run_one_response(call, client=client, cache=cache, retry_policy=retry_policy, run_id=run_id)

    for _call, judgment in run_concurrent(pending, _run, limit=concurrency):
        store.append(judgment)
        existing[judgment.judge_call_id] = judgment

    return [existing[jid] for jid in ids_in_order]
