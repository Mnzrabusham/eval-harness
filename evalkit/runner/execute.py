"""A single model call: cache-checked and retried, in that order.

Shared by response generation and judge invocation -- both are just "get
the completion for this prompt against this model," differing only in
what's done with the resulting text.
"""

from __future__ import annotations

from typing import Any, Mapping

from .cache import ResponseCache
from .client import Completion, ModelClient
from .retry import RetryPolicy, call_with_retry

__all__ = ["get_or_generate"]


def get_or_generate(client: ModelClient, cache: ResponseCache, prompt: str, *,
                     model: str, seed: int, sampling_params: Mapping[str, Any],
                     retry_policy: RetryPolicy, retry_seed: int) -> tuple[Completion, bool]:
    """Return ``(completion, was_cached)``.

    A cache hit skips the client entirely: no retry, no cost. A miss calls
    the client under the retry policy, then writes the result to cache
    before returning, so a later call with the same key -- this run or a
    resumed one -- hits. The whole check-then-call-then-store sequence is
    serialized per key (``cache.lock``), so concurrent tasks that resolve
    to the same key never both pay for the call: the loser blocks briefly
    and then reads what the winner wrote.
    """
    key = cache.key(prompt, model=model, seed=seed, sampling_params=sampling_params)
    with cache.lock(key):
        cached = cache.get_by_key(key)
        if cached is not None:
            return cached, True
        completion = call_with_retry(
            lambda: client.complete(prompt, model=model, seed=seed, sampling_params=sampling_params),
            policy=retry_policy, seed=retry_seed,
        )
        cache.set_by_key(key, completion)
    return completion, False
