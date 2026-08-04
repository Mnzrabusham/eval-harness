"""Unit tests for runner/: caching, retry/backoff, concurrency limits,
resumable incremental writes, dry-run planning, and the judge module's
counterbalancing/write-time constraints carried through end to end.
Everything here uses a mock client; no network calls.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from evalkit.judge import (
    PairwiseJudgment,
    PlannedJudgeCall,
    ResponseSide,
    VerdictParseError,
    counterbalanced_pairwise_tasks,
)
from evalkit.runner import (
    Completion,
    GenerationTask,
    PairwiseJudgeCall,
    RateLimitError,
    ResponseCache,
    ResponseJudgeCall,
    RetryPolicy,
    build_pairwise_request_plan,
    call_with_retry,
    draw_replicate_item_subset,
    dry_run,
    gate_pairwise_realized,
    generation_response_id,
    get_or_generate,
    incomplete_pairs,
    pairwise_call_id,
    replicate_subset_size,
    run_concurrent,
    run_generation,
    run_pairwise_judging,
    run_response_judging,
    validate_judge_model,
    write_planned_calls,
)
from evalkit.runner.store import JsonlStore

REPO_ROOT = Path(__file__).resolve().parent.parent


class MockClient:
    """Scripted client: canned responses, optional forced rate-limit
    failures, optional per-prompt delay (to control completion ordering
    under concurrency), and optional forced arbitrary exceptions."""

    def __init__(self, responses: dict[str, str] | None = None,
                 fail_times: dict[str, int] | None = None,
                 delays: dict[str, float] | None = None,
                 raise_errors: dict[str, Exception] | None = None):
        self.responses = responses or {}
        self.fail_times = dict(fail_times or {})
        self.delays = delays or {}
        self.raise_errors = raise_errors or {}
        self.calls: list[tuple[str, str, int]] = []
        self._lock = threading.Lock()

    def complete(self, prompt, *, model, seed, sampling_params):
        delay = self.delays.get(prompt)
        if delay:
            time.sleep(delay)
        with self._lock:
            self.calls.append((prompt, model, seed))
            if self.fail_times.get(prompt, 0) > 0:
                self.fail_times[prompt] -= 1
                raise RateLimitError("rate limited")
        if prompt in self.raise_errors:
            raise self.raise_errors[prompt]
        text = self.responses.get(prompt, "VERDICT: FIRST")
        return Completion(text=text, tokens=len(text.split()))


class ExplodingClient:
    """A client that fails the test if it is ever called."""

    def complete(self, prompt, *, model, seed, sampling_params):
        raise AssertionError("model client should not have been called")


def _policy(**kw):
    return RetryPolicy(max_attempts=kw.get("max_attempts", 5), base_delay=0.01, max_delay=0.05)


# --- caching ----------------------------------------------------------


def test_cache_hit_skips_the_client(tmp_path):
    client = MockClient()
    cache = ResponseCache(tmp_path / "cache")
    for _ in range(3):
        completion, cached = get_or_generate(
            client, cache, "hello", model="gen-2026-01-01", seed=1,
            sampling_params={"temperature": 0.0}, retry_policy=_policy(), retry_seed=0,
        )
    assert len(client.calls) == 1
    assert cached is True
    assert completion.text == "VERDICT: FIRST"


def test_cache_key_includes_seed(tmp_path):
    client = MockClient()
    cache = ResponseCache(tmp_path / "cache")
    get_or_generate(client, cache, "hello", model="m", seed=1,
                     sampling_params={}, retry_policy=_policy(), retry_seed=0)
    get_or_generate(client, cache, "hello", model="m", seed=2,
                     sampling_params={}, retry_policy=_policy(), retry_seed=0)
    assert len(client.calls) == 2  # distinct seeds -> distinct cache entries


def test_cache_key_is_invariant_to_sampling_params_order(tmp_path):
    cache = ResponseCache(tmp_path / "cache")
    key_a = cache.key("hello", model="m", seed=1,
                      sampling_params={"temperature": 0.0, "top_p": 1.0})
    key_b = cache.key("hello", model="m", seed=1,
                      sampling_params={"top_p": 1.0, "temperature": 0.0})
    assert key_a == key_b

    # And the same, end to end: a second call with reordered params must
    # be a cache hit, not a second model call.
    client = MockClient()
    get_or_generate(client, cache, "hello", model="m", seed=1,
                    sampling_params={"temperature": 0.0, "top_p": 1.0},
                    retry_policy=_policy(), retry_seed=0)
    get_or_generate(client, cache, "hello", model="m", seed=1,
                    sampling_params={"top_p": 1.0, "temperature": 0.0},
                    retry_policy=_policy(), retry_seed=0)
    assert len(client.calls) == 1


def test_cache_is_shared_across_distinct_tasks_with_the_same_prompt(tmp_path):
    # Two different generation tasks (different variant/item) that happen to
    # render the same prompt/model/seed/params should only cost one call.
    client = MockClient()
    cache = ResponseCache(tmp_path / "cache")
    store = JsonlStore(tmp_path / "responses.jsonl")
    tasks = [
        GenerationTask(item_id="item-1", source_doc_id=None, variant_id="v1",
                       prompt="shared prompt", model="gen-2026-01-01", seed=7,
                       sampling_params={"temperature": 0.0}),
        GenerationTask(item_id="item-1", source_doc_id=None, variant_id="v2",
                       prompt="shared prompt", model="gen-2026-01-01", seed=7,
                       sampling_params={"temperature": 0.0}),
    ]
    responses = run_generation(tasks, client=client, cache=cache, store=store,
                               retry_policy=_policy(), concurrency=2)
    assert len(responses) == 2
    assert responses[0].response_id != responses[1].response_id
    assert len(client.calls) == 1  # cache hit on the second, distinct record


# --- retry / backoff ----------------------------------------------------


def test_retries_then_succeeds(tmp_path):
    client = MockClient(fail_times={"p": 2})
    sleeps = []
    completion = call_with_retry(
        lambda: client.complete("p", model="m", seed=1, sampling_params={}),
        policy=_policy(max_attempts=5), seed=42, sleep=sleeps.append,
    )
    assert completion.text == "VERDICT: FIRST"
    assert len(client.calls) == 3
    assert len(sleeps) == 2


def test_retry_exhaustion_raises(tmp_path):
    client = MockClient(fail_times={"p": 10})
    with pytest.raises(RateLimitError):
        call_with_retry(
            lambda: client.complete("p", model="m", seed=1, sampling_params={}),
            policy=_policy(max_attempts=3), seed=42, sleep=lambda d: None,
        )
    assert len(client.calls) == 3


def test_retry_seed_is_deterministic():
    delays_a: list[float] = []
    delays_b: list[float] = []
    client_a = MockClient(fail_times={"p": 3})
    client_b = MockClient(fail_times={"p": 3})
    call_with_retry(lambda: client_a.complete("p", model="m", seed=1, sampling_params={}),
                    policy=_policy(max_attempts=5), seed=123, sleep=delays_a.append)
    call_with_retry(lambda: client_b.complete("p", model="m", seed=1, sampling_params={}),
                    policy=_policy(max_attempts=5), seed=123, sleep=delays_b.append)
    assert delays_a == delays_b


def test_non_rate_limit_errors_are_not_retried():
    def _boom():
        raise ValueError("not a rate limit")
    with pytest.raises(ValueError):
        call_with_retry(_boom, policy=_policy(), seed=0, sleep=lambda d: None)


# --- concurrency --------------------------------------------------------


def test_concurrency_limit_is_respected():
    limit = 3
    in_flight = 0
    peak = 0
    lock = threading.Lock()

    def fn(x):
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        time.sleep(0.02)
        with lock:
            in_flight -= 1
        return x * 2

    results = dict(run_concurrent(range(10), fn, limit=limit))
    assert results == {x: x * 2 for x in range(10)}
    assert peak <= limit


# --- store integrity ------------------------------------------------------


def test_truncated_trailing_line_raises_a_clear_error(tmp_path):
    # Simulate a crash mid-append: two complete records, then a partial
    # line with no closing brace and no trailing newline.
    store_path = tmp_path / "responses.jsonl"
    store_path.write_text(
        '{"item_id": "a", "response_id": "r1"}\n'
        '{"item_id": "b", "response_id": "r2"}\n'
        '{"item_id": "c", "resp',
        encoding="utf-8",
    )
    store = JsonlStore(store_path)
    with pytest.raises(ValueError, match=r"truncated") as excinfo:
        list(store.read_raw())
    # The error names the offending file, not just "some JSON is broken".
    assert str(store_path) in str(excinfo.value)
    assert "line 3" in str(excinfo.value)


# --- generation resumability --------------------------------------------


def test_run_generation_is_resumable(tmp_path):
    cache = ResponseCache(tmp_path / "cache")
    store_path = tmp_path / "responses.jsonl"
    tasks = [
        GenerationTask(item_id=f"item-{i}", source_doc_id=None, variant_id="v1",
                       prompt=f"prompt {i}", model="gen-2026-01-01", seed=1,
                       sampling_params={"temperature": 0.0})
        for i in range(3)
    ]

    client = MockClient()
    store = JsonlStore(store_path)
    first = run_generation(tasks, client=client, cache=cache, store=store,
                           retry_policy=_policy(), concurrency=2)
    assert len(client.calls) == 3

    # Simulate a fresh process resuming against the same on-disk store: a
    # client that explodes if called proves no work is redone.
    resumed_store = JsonlStore(store_path)
    second = run_generation(tasks, client=ExplodingClient(), cache=cache,
                            store=resumed_store, retry_policy=_policy(), concurrency=2)
    assert [r.response_id for r in second] == [r.response_id for r in first]
    assert [r.text for r in second] == [r.text for r in first]


def test_run_generation_resumes_from_store_alone_with_a_cold_cache(tmp_path):
    # The previous test reuses the same warm cache on resume, so it can't
    # tell whether resumability comes from the store or from the cache.
    # Here the cache directory is fresh and empty: only the store's
    # bookkeeping can make this a no-op, proving resume doesn't secretly
    # depend on the cache being warm too.
    store_path = tmp_path / "responses.jsonl"
    tasks = [
        GenerationTask(item_id=f"item-{i}", source_doc_id=None, variant_id="v1",
                       prompt=f"prompt {i}", model="gen-2026-01-01", seed=1,
                       sampling_params={"temperature": 0.0})
        for i in range(3)
    ]
    first_cache = ResponseCache(tmp_path / "cache-run-1")
    store = JsonlStore(store_path)
    first = run_generation(tasks, client=MockClient(), cache=first_cache, store=store,
                           retry_policy=_policy(), concurrency=2)

    cold_cache = ResponseCache(tmp_path / "cache-run-2-cold")
    resumed_store = JsonlStore(store_path)
    second = run_generation(tasks, client=ExplodingClient(), cache=cold_cache,
                            store=resumed_store, retry_policy=_policy(), concurrency=2)
    assert [r.response_id for r in second] == [r.response_id for r in first]
    assert [r.text for r in second] == [r.text for r in first]
    # The cold cache was never touched -- no dangling cache files from a
    # resume that (correctly) never called the client.
    assert list(cold_cache._dir.glob("*.json")) == []


def test_partial_failure_preserves_completed_work(tmp_path):
    # One task fails with a non-rate-limit error; two others succeed. The
    # bad task is delayed so it's the last to finish, guaranteeing the two
    # successes are written to the store before the exception propagates.
    good_a = GenerationTask(item_id="item-good-a", source_doc_id=None, variant_id="v1",
                            prompt="good a", model="m", seed=1, sampling_params={})
    good_b = GenerationTask(item_id="item-good-b", source_doc_id=None, variant_id="v1",
                            prompt="good b", model="m", seed=1, sampling_params={})
    bad = GenerationTask(item_id="item-bad", source_doc_id=None, variant_id="v1",
                         prompt="bad", model="m", seed=1, sampling_params={})
    client = MockClient(delays={"bad": 0.1}, raise_errors={"bad": ValueError("boom")})
    store_path = tmp_path / "responses.jsonl"
    store = JsonlStore(store_path)
    cache = ResponseCache(tmp_path / "cache")

    with pytest.raises(ValueError, match="boom"):
        run_generation([good_a, good_b, bad], client=client, cache=cache, store=store,
                       retry_policy=_policy(), concurrency=3)

    recorded = list(JsonlStore(store_path).read_raw())
    recorded_items = {row["item_id"] for row in recorded}
    assert recorded_items == {"item-good-a", "item-good-b"}


def test_retry_exhaustion_preserves_store_and_stays_resumable(tmp_path):
    good = GenerationTask(item_id="item-good", source_doc_id=None, variant_id="v1",
                          prompt="good prompt", model="m", seed=1, sampling_params={})
    bad = GenerationTask(item_id="item-bad", source_doc_id=None, variant_id="v1",
                        prompt="bad prompt", model="m", seed=1, sampling_params={})
    client = MockClient(fail_times={"bad prompt": 999})  # rate-limited forever
    store_path = tmp_path / "responses.jsonl"
    store = JsonlStore(store_path)
    cache = ResponseCache(tmp_path / "cache")
    policy = RetryPolicy(max_attempts=2, base_delay=0.05, max_delay=0.05)

    with pytest.raises(RateLimitError):
        run_generation([good, bad], client=client, cache=cache, store=store,
                       retry_policy=policy, concurrency=2)

    # The store isn't corrupted by the failure: the good task's record is a
    # complete, parseable line, and the bad one simply never got written.
    recorded_ids = JsonlStore(store_path).written_keys(lambda row: row["response_id"])
    assert generation_response_id(good) in recorded_ids
    assert generation_response_id(bad) not in recorded_ids

    # Once the rate limit clears, a rerun redoes only the bad task.
    class SelectiveExplodingClient:
        def complete(self, prompt, *, model, seed, sampling_params):
            if prompt == "good prompt":
                raise AssertionError("good task should not be redone after resume")
            return Completion(text="recovered", tokens=1)

    resumed_store = JsonlStore(store_path)
    final = run_generation([good, bad], client=SelectiveExplodingClient(), cache=cache,
                           store=resumed_store, retry_policy=policy, concurrency=2)
    assert {r.item_id: r.text for r in final} == {
        "item-good": "VERDICT: FIRST", "item-bad": "recovered",
    }


# --- pairwise judging: counterbalancing + resumability + write-time guard --


def _pairwise_calls(*, judge_model="judge-2026-01-01", replicate=0, call_role="primary"):
    side_a = ResponseSide(variant_id="variant-alpha", response_id="resp-a1",
                          text="the alpha answer", tokens=12)
    side_b = ResponseSide(variant_id="variant-beta", response_id="resp-b1",
                          text="the beta answer", tokens=20)
    t1, t2 = counterbalanced_pairwise_tasks(item_id="item-1", task_text="What is X?",
                                           side_a=side_a, side_b=side_b)
    common = dict(replicate=replicate, call_role=call_role, judge_model=judge_model,
                  judge_config_id="cfg-1", judge_seed=1,
                  judge_sampling_params={"temperature": 0.0}, created_at="2026-01-01")
    return [
        PairwiseJudgeCall(task=t1, **common),
        PairwiseJudgeCall(task=t2, **common),
    ]


def test_pairwise_judging_preserves_structural_counterbalancing(tmp_path):
    calls = _pairwise_calls()
    client = MockClient(responses={calls[0].task.prompt: "VERDICT: FIRST",
                                   calls[1].task.prompt: "VERDICT: SECOND"})
    cache = ResponseCache(tmp_path / "cache")
    store = JsonlStore(tmp_path / "pairwise.jsonl")
    judgments = run_pairwise_judging(calls, run_id="run-1", client=client, cache=cache,
                                     store=store, retry_policy=_policy(), concurrency=2)
    assert len(judgments) == 2
    assert judgments[0].pair_id == judgments[1].pair_id
    # Both presentation orders present, each with a presentation-terms verdict.
    verdicts = {j.judgment for j in judgments}
    assert verdicts == {"first", "second"}


def test_pairwise_judging_is_resumable(tmp_path):
    calls = _pairwise_calls()
    cache = ResponseCache(tmp_path / "cache")
    store_path = tmp_path / "pairwise.jsonl"
    client = MockClient()
    store = JsonlStore(store_path)
    first = run_pairwise_judging(calls, run_id="run-1", client=client, cache=cache,
                                 store=store, retry_policy=_policy(), concurrency=2)
    assert len(client.calls) == 2

    # Full-set contract (mirrors run_generation): a resumed call must return
    # every judgment for `calls`, not just the ones (zero, here) written in
    # this call. A partial list would silently understate the evidence fed
    # to a downstream reduction -- a valid-looking estimate on a subset.
    resumed_store = JsonlStore(store_path)
    second = run_pairwise_judging(calls, run_id="run-1", client=ExplodingClient(),
                                  cache=cache, store=resumed_store, retry_policy=_policy(),
                                  concurrency=2)
    assert [j.judge_call_id for j in second] == [j.judge_call_id for j in first]
    assert [j.judgment for j in second] == [j.judgment for j in first]


def test_response_judging_full_set_contract_on_resume(tmp_path):
    call = ResponseJudgeCall(
        item_id="item-1", source_doc_id=None, variant_id="v1", response_id="resp-1",
        gen_seed={"seed": 1}, response_tokens=10, prompt="judge this",
        judgment_type="binary", replicate=0, judge_model="judge-2026-01-01",
        judge_config_id="cfg-1", judge_seed=1, judge_sampling_params={}, created_at="2026-01-01",
    )
    client = MockClient(responses={"judge this": "VERDICT: PASS"})
    cache = ResponseCache(tmp_path / "cache")
    store_path = tmp_path / "response.jsonl"
    store = JsonlStore(store_path)
    first = run_response_judging([call], run_id="run-1", client=client, cache=cache,
                                 store=store, retry_policy=_policy(), concurrency=1)
    assert len(first) == 1

    resumed_store = JsonlStore(store_path)
    second = run_response_judging([call], run_id="run-1", client=ExplodingClient(),
                                  cache=cache, store=resumed_store, retry_policy=_policy(),
                                  concurrency=1)
    assert [j.judge_call_id for j in second] == [j.judge_call_id for j in first]
    assert [j.judgment for j in second] == [j.judgment for j in first]


def test_pairwise_replicates_are_separate_rows(tmp_path):
    rep0 = _pairwise_calls(replicate=0, call_role="primary")
    rep1 = _pairwise_calls(replicate=1, call_role="replicate")
    ids_0 = {pairwise_call_id(c) for c in rep0}
    ids_1 = {pairwise_call_id(c) for c in rep1}
    assert ids_0.isdisjoint(ids_1)


def test_floating_judge_alias_rejected_before_any_call(tmp_path):
    calls = _pairwise_calls(judge_model="judge-latest")
    client = MockClient()
    cache = ResponseCache(tmp_path / "cache")
    store = JsonlStore(tmp_path / "pairwise.jsonl")
    with pytest.raises(ValueError):
        run_pairwise_judging(calls, run_id="run-1", client=client, cache=cache,
                             store=store, retry_policy=_policy(), concurrency=2)


def test_validate_judge_model_rejects_latest_alias():
    with pytest.raises(ValueError):
        validate_judge_model("judge-model-latest")
    validate_judge_model("judge-model-2026-01-01")  # does not raise


def test_raw_completion_is_cached_before_parsing(tmp_path):
    # A malformed judge response fails to parse into a record, but the raw
    # completion must already be on disk by the time that failure happens:
    # otherwise, fixing the parser and re-running would still re-pay for
    # the model call just to get back the same text.
    calls = _pairwise_calls()
    malformed = "no verdict anywhere in this text"
    client = MockClient(responses={calls[0].task.prompt: malformed,
                                   calls[1].task.prompt: malformed})
    cache = ResponseCache(tmp_path / "cache")
    store = JsonlStore(tmp_path / "pairwise.jsonl")
    with pytest.raises(VerdictParseError):
        run_pairwise_judging(calls, run_id="run-1", client=client, cache=cache,
                             store=store, retry_policy=_policy(), concurrency=1)
    assert len(client.calls) >= 1

    # Re-running against an exploding client must still fail to parse (the
    # text is unchanged) but must NOT touch the client -- if it did,
    # ExplodingClient raises AssertionError instead, and this assertion
    # catches that.
    with pytest.raises(VerdictParseError):
        run_pairwise_judging(calls, run_id="run-1", client=ExplodingClient(), cache=cache,
                             store=JsonlStore(tmp_path / "pairwise.jsonl"),
                             retry_policy=_policy(), concurrency=1)


def test_incomplete_pairs_flags_one_sided_pairs_only(tmp_path):
    calls = _pairwise_calls()  # two calls: both presentation orders of one pair
    complete_pair_id = calls[0].task.pair_id

    side_c = ResponseSide(variant_id="variant-gamma", response_id="resp-c1",
                          text="the gamma answer", tokens=15)
    side_d = ResponseSide(variant_id="variant-delta", response_id="resp-d1",
                          text="the delta answer", tokens=18)
    t_only_one_order, _t_never_run = counterbalanced_pairwise_tasks(
        item_id="item-2", task_text="What is Y?", side_a=side_c, side_b=side_d)
    incomplete_call = PairwiseJudgeCall(
        task=t_only_one_order, replicate=0, call_role="primary", judge_model="judge-2026-01-01",
        judge_config_id="cfg-1", judge_seed=1, judge_sampling_params={}, created_at="2026-01-01",
    )
    incomplete_pair_id = t_only_one_order.pair_id

    client = MockClient()
    cache = ResponseCache(tmp_path / "cache")
    store = JsonlStore(tmp_path / "pairwise.jsonl")
    run_pairwise_judging(calls + [incomplete_call], run_id="run-1", client=client, cache=cache,
                        store=store, retry_policy=_policy(), concurrency=2)

    flagged = incomplete_pairs(store.read_raw())
    assert incomplete_pair_id in flagged
    assert complete_pair_id not in flagged


def test_incomplete_pairs_does_not_raise_on_a_partial_run(tmp_path):
    # A partially-complete pairwise run is a legitimate, resumable
    # intermediate state, not an error: the checker must only report.
    calls = _pairwise_calls()
    only_one_order = [calls[0]]
    client = MockClient()
    cache = ResponseCache(tmp_path / "cache")
    store = JsonlStore(tmp_path / "pairwise.jsonl")
    run_pairwise_judging(only_one_order, run_id="run-1", client=client, cache=cache,
                        store=store, retry_policy=_policy(), concurrency=1)
    flagged = incomplete_pairs(store.read_raw())  # must not raise
    assert calls[0].task.pair_id in flagged


# --- response (scalar/binary) judging -----------------------------------


def test_run_response_judging_binary(tmp_path):
    client = MockClient(responses={"judge this": "VERDICT: PASS"})
    cache = ResponseCache(tmp_path / "cache")
    store = JsonlStore(tmp_path / "response.jsonl")
    call = ResponseJudgeCall(
        item_id="item-1", source_doc_id=None, variant_id="v1", response_id="resp-1",
        gen_seed={"seed": 1}, response_tokens=10, prompt="judge this",
        judgment_type="binary", replicate=0, judge_model="judge-2026-01-01",
        judge_config_id="cfg-1", judge_seed=1, judge_sampling_params={}, created_at="2026-01-01",
    )
    judgments = run_response_judging([call], run_id="run-1", client=client, cache=cache,
                                     store=store, retry_policy=_policy(), concurrency=1)
    assert len(judgments) == 1
    assert judgments[0].judgment == 1.0


# --- dry run --------------------------------------------------------------


def test_dry_run_makes_no_calls_and_classifies_correctly(tmp_path):
    cache = ResponseCache(tmp_path / "cache")
    gen_store = JsonlStore(tmp_path / "responses.jsonl")

    gen_tasks = [
        GenerationTask(item_id=f"item-{i}", source_doc_id=None, variant_id="v1",
                       prompt=f"prompt {i}", model="gen-2026-01-01", seed=1,
                       sampling_params={}) for i in range(3)
    ]
    # Pre-record one task's response, as if from an earlier run.
    already = run_generation([gen_tasks[0]], client=MockClient(), cache=cache,
                             store=gen_store, retry_policy=_policy(), concurrency=1)
    assert len(already) == 1

    # Pre-warm the cache (but not the store) for a second task.
    get_or_generate(MockClient(), cache, "prompt 1", model="gen-2026-01-01", seed=1,
                    sampling_params={}, retry_policy=_policy(), retry_seed=0)

    report = dry_run(generation_tasks=gen_tasks, generation_cache=cache,
                     generation_store=gen_store)
    assert report.generation_total == 3
    assert report.generation_already_recorded == 1
    assert report.generation_cache_hits == 1
    assert report.generation_model_calls == 1
    assert report.total_model_calls == 1


def test_dry_run_rejects_floating_judge_alias_without_calling_client():
    calls = _pairwise_calls(judge_model="judge-latest")
    with pytest.raises(ValueError):
        dry_run(pairwise_calls=calls)


def test_dry_run_pairwise_and_response_counts(tmp_path):
    pw_calls = _pairwise_calls()
    resp_call = ResponseJudgeCall(
        item_id="item-1", source_doc_id=None, variant_id="v1", response_id="resp-1",
        gen_seed={"seed": 1}, response_tokens=10, prompt="judge this",
        judgment_type="binary", replicate=0, judge_model="judge-2026-01-01",
        judge_config_id="cfg-1", judge_seed=1, judge_sampling_params={}, created_at="2026-01-01",
    )
    report = dry_run(pairwise_calls=pw_calls, response_calls=[resp_call])
    assert report.pairwise_total == 2
    assert report.pairwise_model_calls == 2
    assert report.response_total == 1
    assert report.response_model_calls == 1
    assert report.total_model_calls == 3


def test_dry_run_prediction_matches_the_actual_run(tmp_path):
    # Internal arithmetic (total = recorded + cache_hits + model_calls) is
    # not enough: this checks dry_run's predicted model_calls against what
    # an actual run really does to a client, for both generation and
    # pairwise judging.
    cache = ResponseCache(tmp_path / "cache")
    gen_store = JsonlStore(tmp_path / "responses.jsonl")
    gen_tasks = [
        GenerationTask(item_id=f"item-{i}", source_doc_id=None, variant_id="v1",
                       prompt=f"prompt {i}", model="gen-2026-01-01", seed=1,
                       sampling_params={}) for i in range(4)
    ]
    # One already recorded (prior run), one pre-cached but not recorded,
    # two entirely fresh.
    run_generation([gen_tasks[0]], client=MockClient(), cache=cache, store=gen_store,
                   retry_policy=_policy(), concurrency=1)
    get_or_generate(MockClient(), cache, "prompt 1", model="gen-2026-01-01", seed=1,
                    sampling_params={}, retry_policy=_policy(), retry_seed=0)

    predicted = dry_run(generation_tasks=gen_tasks, generation_cache=cache,
                        generation_store=gen_store).generation_model_calls

    actual_client = MockClient()
    run_generation(gen_tasks, client=actual_client, cache=cache, store=gen_store,
                   retry_policy=_policy(), concurrency=2)
    assert len(actual_client.calls) == predicted

    # Same cross-check for pairwise judging: one call pre-recorded, one fresh.
    pw_calls = _pairwise_calls()
    pw_store = JsonlStore(tmp_path / "pairwise.jsonl")
    run_pairwise_judging([pw_calls[0]], run_id="run-1", client=MockClient(), cache=cache,
                        store=pw_store, retry_policy=_policy(), concurrency=1)

    predicted_pw = dry_run(pairwise_calls=pw_calls, judge_cache=cache,
                           pairwise_store=pw_store).pairwise_model_calls
    actual_pw_client = MockClient()
    run_pairwise_judging(pw_calls, run_id="run-1", client=actual_pw_client, cache=cache,
                        store=pw_store, retry_policy=_policy(), concurrency=2)
    assert len(actual_pw_client.calls) == predicted_pw


# --- generation_response_id stability -------------------------------------


def test_generation_response_id_is_stable_across_dict_ordering():
    t1 = GenerationTask(item_id="i", source_doc_id=None, variant_id="v", prompt="p",
                        model="m", seed=1, sampling_params={"a": 1, "b": 2})
    t2 = GenerationTask(item_id="i", source_doc_id=None, variant_id="v", prompt="p",
                        model="m", seed=1, sampling_params={"b": 2, "a": 1})
    assert generation_response_id(t1) == generation_response_id(t2)


# --- determinism across processes -----------------------------------------


def test_ids_are_stable_across_different_hash_seeds():
    # ids.py must derive ids from hashlib, not Python's built-in hash():
    # hash() of str/bytes is salted per-process by PYTHONHASHSEED, so an id
    # built from it would differ across processes and silently break resume
    # (a rerun in a fresh process would never recognize its own prior work)
    # while appearing to work perfectly within a single run/test process.
    script = (
        "from evalkit.runner.ids import derive_id, derive_seed\n"
        "print(derive_id('item-1', 'variant-a', ('gen-2026-01-01', 1), "
        "[('temperature', 0.0), ('top_p', 1.0)]))\n"
        "print(derive_seed('resp-1', 'retry'))\n"
    )

    def _run(hash_seed: str) -> str:
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = hash_seed
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=str(REPO_ROOT),
            capture_output=True, text=True, env=env, check=True,
        )
        return result.stdout

    out_seed_1 = _run("1")
    out_seed_2 = _run("2")
    assert out_seed_1 == out_seed_2
    assert out_seed_1.strip() != ""


# --- D4 item-level replicate draw (docs/data-model.md; spec §8.2, DECISION D4) --


def _item_task_pairs(n: int) -> dict:
    pairs = {}
    for i in range(n):
        side_a = ResponseSide(variant_id="variant-alpha", response_id=f"resp-a{i}",
                              text=f"alpha answer {i}", tokens=10)
        side_b = ResponseSide(variant_id="variant-beta", response_id=f"resp-b{i}",
                              text=f"beta answer {i}", tokens=10)
        pairs[f"item-{i}"] = counterbalanced_pairwise_tasks(
            item_id=f"item-{i}", task_text=f"task {i}", side_a=side_a, side_b=side_b)
    return pairs


def test_d4_replicate_draw_is_item_level_and_side_balanced():
    item_tasks = _item_task_pairs(20)
    plan = build_pairwise_request_plan(
        item_tasks, run_id="run-1", judge_model="judge-2026-01-01", judge_config_id="cfg-1",
        judge_seed=1, judge_sampling_params={}, created_at="2026-01-01", replicate_seed=42,
    )
    # A genuine, non-trivial subset was drawn -- not all-or-nothing.
    assert 0 < len(plan.replicated_items) < 20

    for item_id, (t1, t2) in item_tasks.items():
        reps_first = {c.replicate for c in plan.calls if c.task == t1}
        reps_second = {c.replicate for c in plan.calls if c.task == t2}
        # Side-balanced by construction: whichever replicate indices exist
        # on one presentation order exist on the other. A per-response
        # reimplementation (drawing each side independently) would break
        # this for some item with near certainty -- see the next test for
        # confirmation that independent per-side draws actually do diverge.
        assert reps_first == reps_second, f"{item_id} is side-unbalanced"
        if item_id in plan.replicated_items:
            assert reps_first == {0, 1}
        else:
            assert reps_first == {0}


def test_independent_per_side_draws_diverge_so_the_balance_check_has_teeth():
    # Guards the guard: a per-response reimplementation is, in effect, an
    # independent draw per side. Two independently-seeded draws over the
    # same items disagreeing shows the assertion above would actually catch
    # that mistake instead of passing vacuously.
    item_ids = [f"item-{i}" for i in range(20)]
    drawn_side_a = draw_replicate_item_subset(item_ids, seed=1)
    drawn_side_b = draw_replicate_item_subset(item_ids, seed=2)
    assert drawn_side_a != drawn_side_b


def test_d4_draw_is_deterministic_given_the_same_seed():
    item_ids = [f"item-{i}" for i in range(15)]
    assert draw_replicate_item_subset(item_ids, seed=7) == draw_replicate_item_subset(item_ids, seed=7)


def test_replicate_subset_size_respects_bounds():
    # Never fewer than 10 responses' worth of items...
    assert replicate_subset_size(1000, min_responses=10, max_responses=30, fraction=0.0,
                                 responses_per_item=2) == 5  # ceil(10 / 2)
    # ...capped at max_responses even for a huge item bank...
    assert replicate_subset_size(1000, min_responses=10, max_responses=30, fraction=1.0,
                                 responses_per_item=2) == 15  # ceil(30 / 2)
    # ...and never more items than exist.
    assert replicate_subset_size(3, min_responses=10, max_responses=30, fraction=1.0,
                                 responses_per_item=2) == 3


# --- gap-9 gate (docs/data-model.md; spec §12 gap 9, DECISION D11) ------------


def _planned(pair_id, variant_first, variant_second, call_role, judge_call_id):
    return PlannedJudgeCall(
        run_id="run-1", item_id="item-1", source_doc_id=None, pair_id=pair_id,
        variant_first=variant_first, variant_second=variant_second,
        judge_model="judge-2026-01-01", judge_config_id="cfg-1", call_role=call_role,
        judge_call_id=judge_call_id, created_at="2026-01-01",
    )


def _realized(pair_id, variant_first, variant_second, call_role, judge_call_id, judgment="first"):
    return PairwiseJudgment(
        run_id="run-1", item_id="item-1", source_doc_id=None, pair_id=pair_id,
        variant_first=variant_first, variant_second=variant_second,
        response_id_first="r1", response_id_second="r2", gen_seed_first=0, gen_seed_second=0,
        tokens_first=1, tokens_second=1, judge_model="judge-2026-01-01", judge_config_id="cfg-1",
        judge_call_id=judge_call_id, call_role=call_role, judgment=judgment, created_at="2026-01-01",
    )


def test_gate_classifies_the_four_gap_9_cases():
    planned = [
        # pair-normal: primary planned and realized.
        _planned("pair-normal", "v1", "v2", "primary", "call-normal-primary"),
        # pair-promoted: primary planned but never realized; its replicate was.
        _planned("pair-promoted", "v1", "v2", "primary", "call-promoted-primary"),
        _planned("pair-promoted", "v1", "v2", "replicate", "call-promoted-replicate"),
        # pair-excluded: primary and replicate planned, neither realized.
        _planned("pair-excluded", "v1", "v2", "primary", "call-excluded-primary"),
        _planned("pair-excluded", "v1", "v2", "replicate", "call-excluded-replicate"),
        # pair-no-primary-planned: a plan-authoring bug -- only a replicate
        # was ever planned. It succeeded, but that's still a plumbing error:
        # the primary was never requested for this (pair, order).
        _planned("pair-no-primary-planned", "v1", "v2", "replicate", "call-np-replicate"),
    ]
    realized = [
        _realized("pair-normal", "v1", "v2", "primary", "call-normal-primary"),
        _realized("pair-promoted", "v1", "v2", "replicate", "call-promoted-replicate"),
        _realized("pair-no-primary-planned", "v1", "v2", "replicate", "call-np-replicate"),
        # A stray record: a judge_call_id with no plan row anywhere.
        _realized("orphan-pair", "v1", "v2", "primary", "unplanned-call-id"),
    ]

    gate = gate_pairwise_realized(planned, realized)

    assert gate.normal == (("pair-normal", "v1"),)
    assert gate.promotions == (("pair-promoted", "v1"),)
    assert gate.exclusions == (("pair-excluded", "v1"),)
    assert set(gate.plumbing_errors) == {("pair-no-primary-planned", "v1"), ("orphan-pair", "v1")}
    assert gate.promotion_count == 1
    assert gate.plumbing_error_count == 2
    assert gate.exclusion_count == 1
    assert gate.blocks_run is True


def test_gate_does_not_block_on_promotions_and_exclusions_alone():
    planned = [
        _planned("pair-promoted", "v1", "v2", "primary", "call-promoted-primary"),
        _planned("pair-promoted", "v1", "v2", "replicate", "call-promoted-replicate"),
        _planned("pair-excluded", "v1", "v2", "primary", "call-excluded-primary"),
    ]
    realized = [
        _realized("pair-promoted", "v1", "v2", "replicate", "call-promoted-replicate"),
    ]
    gate = gate_pairwise_realized(planned, realized)
    assert gate.blocks_run is False
    assert gate.promotion_count == 1
    assert gate.exclusion_count == 1
    assert gate.plumbing_error_count == 0


def test_promotion_is_never_written_back_only_derived():
    # docs/data-model.md Constraints: promotions are derived at analysis
    # time by joining records to PlannedJudgeCall, never written back. The
    # realized record itself must still say call_role="replicate" even
    # though the gate reports this (pair, order) as a promotion.
    planned = [
        _planned("pair-promoted", "v1", "v2", "primary", "call-promoted-primary"),
        _planned("pair-promoted", "v1", "v2", "replicate", "call-promoted-replicate"),
    ]
    realized = [_realized("pair-promoted", "v1", "v2", "replicate", "call-promoted-replicate")]
    assert realized[0].call_role == "replicate"
    gate = gate_pairwise_realized(planned, realized)
    assert gate.promotions == (("pair-promoted", "v1"),)


# --- plan -> execute -> gate, end to end --------------------------------------


def test_plan_is_on_disk_before_any_call_and_a_clean_run_gates_all_normal(tmp_path):
    item_tasks = _item_task_pairs(3)
    plan = build_pairwise_request_plan(
        item_tasks, run_id="run-1", judge_model="judge-2026-01-01", judge_config_id="cfg-1",
        judge_seed=1, judge_sampling_params={"temperature": 0.0}, created_at="2026-01-01",
        replicate_seed=1,
    )
    plan_store = JsonlStore(tmp_path / "planned.jsonl")
    write_planned_calls(plan.planned, plan_store)
    # The plan is fully persisted before a single judge call has been made.
    assert len(list(plan_store.read_raw())) == len(plan.planned)

    client = MockClient()
    cache = ResponseCache(tmp_path / "cache")
    judgment_store = JsonlStore(tmp_path / "pairwise.jsonl")
    run_pairwise_judging(plan.calls, run_id="run-1", client=client, cache=cache,
                        store=judgment_store, retry_policy=_policy(), concurrency=4)

    gate = gate_pairwise_realized(plan_store.read_raw(), judgment_store.read_raw())
    # MockClient never fails, so every planned primary succeeded: no
    # promotions, no plumbing errors, no exclusions, even though replicate
    # calls were also planned, made, and recorded for the drawn items.
    assert gate.plumbing_error_count == 0
    assert gate.promotion_count == 0
    assert gate.exclusion_count == 0
    expected = {(t1.pair_id, t1.first.variant_id) for t1, _t2 in item_tasks.values()}
    expected |= {(t2.pair_id, t2.first.variant_id) for _t1, t2 in item_tasks.values()}
    assert set(gate.normal) == expected


def test_gate_flags_a_record_the_plan_never_committed_to(tmp_path):
    item_tasks = _item_task_pairs(2)
    plan = build_pairwise_request_plan(
        item_tasks, run_id="run-1", judge_model="judge-2026-01-01", judge_config_id="cfg-1",
        judge_seed=1, judge_sampling_params={}, created_at="2026-01-01", replicate_seed=1,
    )
    client = MockClient()
    cache = ResponseCache(tmp_path / "cache")
    store = JsonlStore(tmp_path / "pairwise.jsonl")
    run_pairwise_judging(plan.calls, run_id="run-1", client=client, cache=cache,
                        store=store, retry_policy=_policy(), concurrency=2)

    gate_clean = gate_pairwise_realized(plan.planned, store.read_raw())
    assert gate_clean.blocks_run is False

    # Simulate a plumbing bug: a realized record whose judge_call_id was
    # never in the plan the run committed to before judging started.
    stray = _realized("orphan-pair", "v1", "v2", "primary", "unplanned-call-id")
    gate_dirty = gate_pairwise_realized(plan.planned, list(store.read_raw()) + [stray])
    assert gate_dirty.blocks_run is True
    assert gate_dirty.plumbing_error_count == 1
