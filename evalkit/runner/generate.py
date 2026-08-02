"""Response generation: variant x item -> a cached, id-stable response.

``response_id`` is derived deterministically from the task's identity
(item, variant, model, seed, sampling params) rather than assigned
randomly or by counter, so re-running the same configuration reproduces
the same response_id -- and therefore hits the same cache entry and is
recognized as already-recorded in the store, which is what makes
generation resumable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .cache import ResponseCache
from .client import ModelClient
from .concurrency import run_concurrent
from .execute import get_or_generate
from .ids import derive_id, derive_seed
from .retry import RetryPolicy
from .store import JsonlStore

__all__ = [
    "GenerationTask",
    "GeneratedResponse",
    "generation_response_id",
    "generate_response",
    "run_generation",
]


@dataclass(frozen=True)
class GenerationTask:
    """One response to generate: a variant's rendered prompt for one item."""

    item_id: str
    source_doc_id: str | None
    variant_id: str
    prompt: str
    model: str
    seed: int
    sampling_params: Mapping[str, Any]


@dataclass(frozen=True)
class GeneratedResponse:
    """The generation stage's output: the raw material for judge tasks.

    Not one of docs/data-model.md's three record types -- the data model
    records only judgments, keyed by response_id. This is what a
    judgment's response_id refers to, and what becomes a
    ``evalkit.judge.ResponseSide`` when building judge tasks.
    """

    item_id: str
    source_doc_id: str | None
    variant_id: str
    response_id: str
    text: str
    tokens: int
    gen_seed: object


def generation_response_id(task: GenerationTask) -> str:
    return derive_id(task.item_id, task.variant_id, task.model, task.seed,
                     sorted(task.sampling_params.items()))


def generate_response(task: GenerationTask, *, client: ModelClient, cache: ResponseCache,
                       retry_policy: RetryPolicy) -> GeneratedResponse:
    response_id = generation_response_id(task)
    gen_seed = {"seed": task.seed, **task.sampling_params}
    completion, _cached = get_or_generate(
        client, cache, task.prompt, model=task.model, seed=task.seed,
        sampling_params=task.sampling_params, retry_policy=retry_policy,
        retry_seed=derive_seed(response_id, "retry"),
    )
    return GeneratedResponse(
        item_id=task.item_id, source_doc_id=task.source_doc_id,
        variant_id=task.variant_id, response_id=response_id,
        text=completion.text, tokens=completion.tokens, gen_seed=gen_seed,
    )


def run_generation(tasks: Iterable[GenerationTask], *, client: ModelClient, cache: ResponseCache,
                    store: JsonlStore, retry_policy: RetryPolicy,
                    concurrency: int) -> list[GeneratedResponse]:
    """Generate every task's response, skipping ones already in ``store``.

    Returns the full set of responses for ``tasks`` in order -- some read
    back from ``store`` (already done, possibly in an earlier, interrupted
    run), some freshly generated and appended to ``store`` in this call.
    """
    tasks = list(tasks)
    existing = {row["response_id"]: GeneratedResponse(**row) for row in store.read_raw()}

    ids_in_order = [generation_response_id(task) for task in tasks]
    pending = [task for task, rid in zip(tasks, ids_in_order) if rid not in existing]

    def _run(task: GenerationTask) -> GeneratedResponse:
        return generate_response(task, client=client, cache=cache, retry_policy=retry_policy)

    for _task, response in run_concurrent(pending, _run, limit=concurrency):
        store.append(response)
        existing[response.response_id] = response

    return [existing[rid] for rid in ids_in_order]
