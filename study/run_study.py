"""Position-bias study driver (docs/position-bias-study.md sections 2, 3, 5,
6.5, 11).

A subcommand CLI, one phase per invocation. No subcommand triggers another --
each is a separate ``python study/run_study.py <phase>`` call, so every phase
can be run, inspected, and (if something looks wrong) stopped before the next
one starts. The intended order, matching section 11's blocking prerequisites:

    generate -> check-arms -> plan -> dry-run -> judge -> gate

(``dry-run`` has no hard prerequisite beyond ``generate`` and can also be run
standalone at any point to preview cost.)

Every phase that writes records does so incrementally (``JsonlStore.append``
per completed unit, via the same ``evalkit.runner`` machinery used
elsewhere), so an interrupted run loses at most the one record in flight and
resumes without re-paying for completed work. Every phase that can spend API
budget (``generate``, ``check-arms`` when it needs to regenerate, ``judge``)
prints the predicted model-call count and estimated cost first and refuses to
proceed without an explicit ``--confirm`` flag.

Artifacts (under ``study/artifacts/``, all resumable JSONL/JSON, none of it
committed by this script -- see the top-level "stage, do not commit" rule):

    responses.jsonl       GeneratedResponse records (generate)
    gen_cache/             ResponseCache for generation
    arm_check.json         section 11 item 5 gate result (check-arms)
    planned.jsonl          PlannedJudgeCall records (plan)
    judge_cache/            ResponseCache for judging
    judgments.jsonl        PairwiseJudgment records (judge)

Design constants below (model IDs, generation prompt, judge config, D4
replicate sizing) are transcribed from the sections this task named; see the
inline section references for where each one comes from.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from evalkit.judge import ResponseSide, counterbalanced_pairwise_tasks
from evalkit.judge.prompts import build_generation_prompt
from evalkit.runner import (
    Completion,
    GeneratedResponse,
    GenerationTask,
    PairwiseRequestPlan,
    RateLimitError,
    ResponseCache,
    RetryPolicy,
    build_pairwise_request_plan,
    derive_seed,
    dry_run,
    gate_pairwise_realized,
    run_generation,
    run_pairwise_judging,
    write_planned_calls,
)
from evalkit.runner.store import JsonlStore

# --- paths -------------------------------------------------------------------

STUDY_DIR = Path("study")
ARTIFACTS_DIR = STUDY_DIR / "artifacts"
PROMPT_SET_PATH = STUDY_DIR / "prompt_set.jsonl"

RESPONSES_PATH = ARTIFACTS_DIR / "responses.jsonl"
GEN_CACHE_DIR = ARTIFACTS_DIR / "gen_cache"
ARM_CHECK_PATH = ARTIFACTS_DIR / "arm_check.json"
PLANNED_PATH = ARTIFACTS_DIR / "planned.jsonl"
JUDGE_CACHE_DIR = ARTIFACTS_DIR / "judge_cache"
JUDGMENTS_PATH = ARTIFACTS_DIR / "judgments.jsonl"

# --- design constants ---------------------------------------------------------

RUN_ID = "posbias-2026-08"
RUN_SEED = 20260804  # study/build_prompt_set.py's seed, reused for continuity

# section 2: three judge models.
MODEL_OPUS = "claude-opus-5"
MODEL_SONNET = "claude-sonnet-5"
MODEL_HAIKU = "claude-haiku-4-5-20251001"  # section 2's model-pinning caveat: use the dated snapshot
JUDGES = (MODEL_OPUS, MODEL_SONNET, MODEL_HAIKU)

# section 2: $ per million tokens (in, out).
PRICING = {
    MODEL_OPUS: (5.0, 25.0),
    MODEL_SONNET: (3.0, 15.0),
    MODEL_HAIKU: (1.0, 5.0),
}

# section 6.5's own planning assumptions, used as-is so dry-run's estimate is
# a direct comparison against the table, not a re-estimate.
GEN_TOKENS = (120, 350)     # (input, output)
JUDGE_TOKENS = (1100, 64)   # (input, output)

# section 3.2: max_tokens ceilings sized for the instruction's word range.
# 350 words is roughly 500-600 tokens even before the newer, heavier
# tokenizer Sonnet 5/Opus 5 use (see docs/position-bias-study.md section
# 6.5's count_tokens.py note); 700 leaves headroom without being wasteful.
GEN_MAX_TOKENS = 700
# section 6.5: judge output is a single free-text VERDICT line, <= 64 tokens.
JUDGE_MAX_TOKENS = 64

# section 3.2: distinct seeds per generation task so the response cache
# (keyed on prompt+model+seed+params, not on item_id/variant_id) never
# collides two different arms that happen to render the same prompt text
# under the same model -- not just the near-tie pair's own two arms, which
# is the one section 3.3 calls out explicitly, but every same-model arm.
SEED_SONNET_CLEARGAP = 0
SEED_HAIKU_CLEARGAP = 0  # different model; no collision risk with the above
SEED_SONNET_NEARTIE_0 = 1
SEED_SONNET_NEARTIE_1 = 2
SEED_SONNET_NEARTIE_1_REGEN = 3  # section 11 item 5: "regenerated once with a fresh gen_seed"

# section 5 / DECISION D4: 60 pairs per judge (10% of 600), both orders
# replicated -- override the library defaults (10-30 responses, 20%) to hit
# that exact figure: 10% of 1200 responses = 120 responses = 60 items.
REPLICATE_MIN_RESPONSES = 10
REPLICATE_MAX_RESPONSES = 120
REPLICATE_FRACTION = 0.10

NEAR_TIE_EXCLUSION_THRESHOLD = 0.05  # section 11 item 5

DEFAULT_CONCURRENCY = 5
DEFAULT_RETRY_POLICY = RetryPolicy()


# --- shared helpers ------------------------------------------------------------


def _load_prompts() -> list[dict]:
    if not PROMPT_SET_PATH.exists():
        raise SystemExit(f"{PROMPT_SET_PATH} not found; run study/build_prompt_set.py first")
    rows = []
    with PROMPT_SET_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_responses() -> dict[tuple[str, str], GeneratedResponse]:
    """(prompt_id, variant_id) -> GeneratedResponse, from the generation store."""
    if not RESPONSES_PATH.exists():
        return {}
    out: dict[tuple[str, str], GeneratedResponse] = {}
    for row in JsonlStore(RESPONSES_PATH).read_raw():
        resp = GeneratedResponse(**row)
        out[(resp.item_id, resp.variant_id)] = resp
    return out


def _load_arm_check() -> dict | None:
    if not ARM_CHECK_PATH.exists():
        return None
    return json.loads(ARM_CHECK_PATH.read_text(encoding="utf-8"))


def _cost(model: str, calls: int, tokens: tuple[int, int]) -> float:
    price_in, price_out = PRICING[model]
    tok_in, tok_out = tokens
    return calls * (tok_in * price_in + tok_out * price_out) / 1_000_000.0


def _judge_config_id(judge_model: str) -> str:
    return f"{judge_model}-nothink-v1"


def _judge_sampling_params(judge_model: str) -> dict:
    """section 2: thinking disabled where the API permits it; Haiku 4.5 has
    no thinking by default, so its config simply omits the key. No sampling
    parameters (temperature/top_p) for any judge."""
    params: dict = {"max_tokens": JUDGE_MAX_TOKENS}
    if judge_model != MODEL_HAIKU:
        params["thinking"] = {"type": "disabled"}
    return params


def _side(resp: GeneratedResponse, variant_label: str) -> ResponseSide:
    return ResponseSide(variant_id=variant_label, response_id=resp.response_id,
                        text=resp.text, tokens=resp.tokens, gen_seed=resp.gen_seed)


def _generation_tasks(prompts: list[dict]) -> list[GenerationTask]:
    """section 3.2: per prompt, one Sonnet + one Haiku response for the
    clear-gap pair, and two Sonnet samples (distinct gen_seed) for the
    near-tie pair -- 4 tasks/prompt, 1200 total, 900 Sonnet + 300 Haiku."""
    tasks = []
    for row in prompts:
        pid = row["prompt_id"]
        rendered = build_generation_prompt(row["text"])
        tasks.append(GenerationTask(
            item_id=pid, source_doc_id=pid, variant_id="sonnet-cleargap",
            prompt=rendered, model=MODEL_SONNET, seed=SEED_SONNET_CLEARGAP,
            sampling_params={"thinking": {"type": "disabled"}, "max_tokens": GEN_MAX_TOKENS},
        ))
        tasks.append(GenerationTask(
            item_id=pid, source_doc_id=pid, variant_id="haiku-cleargap",
            prompt=rendered, model=MODEL_HAIKU, seed=SEED_HAIKU_CLEARGAP,
            sampling_params={"max_tokens": GEN_MAX_TOKENS},
        ))
        tasks.append(GenerationTask(
            item_id=pid, source_doc_id=pid, variant_id="sonnet-neartie-0",
            prompt=rendered, model=MODEL_SONNET, seed=SEED_SONNET_NEARTIE_0,
            sampling_params={"thinking": {"type": "disabled"}, "max_tokens": GEN_MAX_TOKENS},
        ))
        tasks.append(GenerationTask(
            item_id=pid, source_doc_id=pid, variant_id="sonnet-neartie-1",
            prompt=rendered, model=MODEL_SONNET, seed=SEED_SONNET_NEARTIE_1,
            sampling_params={"thinking": {"type": "disabled"}, "max_tokens": GEN_MAX_TOKENS},
        ))
    return tasks


def build_item_tasks(prompts: list[dict], responses: dict[tuple[str, str], GeneratedResponse],
                     excluded_prompt_ids: set[str], arm1_variant: dict[str, str]):
    """item_id -> (first-order PairwiseTask, second-order PairwiseTask) for
    every pair to be judged: every prompt's clear-gap pair, and every
    prompt's near-tie pair except those excluded by check-arms (section 11
    item 5). ``arm1_variant`` names which generated response is the near-tie
    pair's second arm -- the original ``sonnet-neartie-1``, or its
    regenerated replacement if check-arms swapped it in.
    """
    item_tasks: dict[str, tuple] = {}
    for row in prompts:
        pid = row["prompt_id"]
        task_text = row["text"]

        strong = responses.get((pid, "sonnet-cleargap"))
        weak = responses.get((pid, "haiku-cleargap"))
        if strong is None or weak is None:
            raise ValueError(f"missing clear-gap responses for prompt {pid!r}; run `generate` first")
        cg_item = f"{pid}-cleargap"
        item_tasks[cg_item] = counterbalanced_pairwise_tasks(
            item_id=cg_item, task_text=task_text, source_doc_id=pid,
            side_a=_side(strong, "sonnet-cleargap"), side_b=_side(weak, "haiku-cleargap"),
        )

        if pid in excluded_prompt_ids:
            continue
        arm0 = responses.get((pid, "sonnet-neartie-0"))
        arm1_id = arm1_variant.get(pid, "sonnet-neartie-1")
        arm1 = responses.get((pid, arm1_id))
        if arm0 is None or arm1 is None:
            raise ValueError(
                f"missing near-tie responses for prompt {pid!r} (arm1={arm1_id!r}); "
                f"run `generate` (and `check-arms` if a regeneration is expected) first"
            )
        nt_item = f"{pid}-neartie"
        item_tasks[nt_item] = counterbalanced_pairwise_tasks(
            item_id=nt_item, task_text=task_text, source_doc_id=pid,
            side_a=_side(arm0, "sonnet-neartie-0"), side_b=_side(arm1, arm1_id),
        )
    return item_tasks


def build_all_judge_plans(item_tasks, *, created_at: str) -> dict[str, PairwiseRequestPlan]:
    """One PairwiseRequestPlan per judge (section 2), sharing the same item
    set, each with its own D4 replicate draw (section 5: 60 pairs/judge)."""
    return {
        judge_model: build_pairwise_request_plan(
            item_tasks, run_id=RUN_ID, judge_model=judge_model,
            judge_config_id=_judge_config_id(judge_model),
            judge_seed=derive_seed(RUN_SEED, "judge-seed", judge_model),
            judge_sampling_params=_judge_sampling_params(judge_model),
            created_at=created_at,
            replicate_seed=derive_seed(RUN_SEED, "replicate-seed", judge_model),
            min_replicate_responses=REPLICATE_MIN_RESPONSES,
            max_replicate_responses=REPLICATE_MAX_RESPONSES,
            replicate_fraction=REPLICATE_FRACTION,
        )
        for judge_model in JUDGES
    }


def _build_anthropic_client():
    """The real ModelClient (evalkit.runner.client.ModelClient protocol),
    backed by the Anthropic API. Imported lazily so every phase that makes
    no model calls works without the ``anthropic`` package installed --
    matching study/count_tokens.py's precedent of not adding it to
    pyproject.toml's core dependency set.
    """
    import anthropic

    class AnthropicModelClient:
        def __init__(self) -> None:
            self._client = anthropic.Anthropic()

        def complete(self, prompt: str, *, model: str, seed: int, sampling_params) -> Completion:
            # ``seed`` is accepted for protocol compliance but never sent to
            # the API: section 3.3 -- the API offers no generation seed;
            # gen_seed/seed is a cache discriminator only, and by the time
            # complete() runs, ResponseCache has already used it to decide
            # this is a miss.
            kwargs = dict(sampling_params)
            max_tokens = kwargs.pop("max_tokens")
            thinking = kwargs.pop("thinking", None)
            if kwargs:
                raise ValueError(f"unrecognized sampling_params keys: {sorted(kwargs)}")
            request = dict(model=model, max_tokens=max_tokens,
                          messages=[{"role": "user", "content": prompt}])
            if thinking is not None:
                request["thinking"] = thinking
            try:
                response = self._client.messages.create(**request)
            except anthropic.RateLimitError as exc:
                raise RateLimitError(str(exc)) from exc
            if response.stop_reason == "refusal":
                raise ValueError(f"call refused (stop_details={response.stop_details})")
            text = "".join(b.text for b in response.content if b.type == "text")
            return Completion(text=text, tokens=response.usage.output_tokens)

    return AnthropicModelClient()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- generate ------------------------------------------------------------------


def cmd_generate(args) -> int:
    prompts = _load_prompts()
    tasks = _generation_tasks(prompts)
    cache = ResponseCache(GEN_CACHE_DIR)
    store = JsonlStore(RESPONSES_PATH)

    print(f"generation tasks: {len(tasks)} "
         f"({sum(1 for t in tasks if t.model == MODEL_SONNET)} {MODEL_SONNET}, "
         f"{sum(1 for t in tasks if t.model == MODEL_HAIKU)} {MODEL_HAIKU})")
    total_calls = 0
    total_cost = 0.0
    for model in (MODEL_SONNET, MODEL_HAIKU):
        subset = [t for t in tasks if t.model == model]
        report = dry_run(generation_tasks=subset, generation_cache=cache, generation_store=store)
        cost = _cost(model, report.generation_model_calls, GEN_TOKENS)
        total_calls += report.generation_model_calls
        total_cost += cost
        print(f"  {model}: total={report.generation_total} "
             f"already_recorded={report.generation_already_recorded} "
             f"cache_hits={report.generation_cache_hits} "
             f"model_calls={report.generation_model_calls} est=${cost:.2f}")
    print(f"predicted new model calls: {total_calls}, est ${total_cost:.2f} "
         f"(section 6.5 table: 1200 calls, ~$5.70)")

    if not args.confirm:
        print("Pass --confirm to proceed.")
        return 1

    client = _build_anthropic_client()
    responses = run_generation(tasks, client=client, cache=cache, store=store,
                               retry_policy=DEFAULT_RETRY_POLICY, concurrency=args.concurrency)
    print(f"generated {len(responses)} responses (of {len(tasks)} tasks); store: {RESPONSES_PATH}")
    return 0


# --- check-arms ------------------------------------------------------------------


def cmd_check_arms(args) -> int:
    prompts = _load_prompts()
    responses = _load_responses()
    if not responses:
        print("no generated responses found; run `generate` first.")
        return 1

    to_regen: list[str] = []
    arm1_variant: dict[str, str] = {}
    checked = 0
    for row in prompts:
        pid = row["prompt_id"]
        a = responses.get((pid, "sonnet-neartie-0"))
        b = responses.get((pid, "sonnet-neartie-1"))
        if a is None or b is None:
            print(f"missing near-tie responses for prompt {pid!r}; run `generate` first.")
            return 1
        checked += 1
        arm1_variant[pid] = "sonnet-neartie-1"
        if a.text == b.text:
            to_regen.append(pid)

    print(f"checked {checked} near-tie pairs; {len(to_regen)} identical, need regeneration")

    still_identical: list[str] = []
    if to_regen:
        prompts_by_id = {row["prompt_id"]: row for row in prompts}
        regen_tasks = [
            GenerationTask(
                item_id=pid, source_doc_id=pid, variant_id="sonnet-neartie-1-regen",
                prompt=build_generation_prompt(prompts_by_id[pid]["text"]),
                model=MODEL_SONNET, seed=SEED_SONNET_NEARTIE_1_REGEN,
                sampling_params={"thinking": {"type": "disabled"}, "max_tokens": GEN_MAX_TOKENS},
            )
            for pid in to_regen
        ]
        cache = ResponseCache(GEN_CACHE_DIR)
        store = JsonlStore(RESPONSES_PATH)
        report = dry_run(generation_tasks=regen_tasks, generation_cache=cache, generation_store=store)
        cost = _cost(MODEL_SONNET, report.generation_model_calls, GEN_TOKENS)
        print(f"regeneration: {report.generation_model_calls} model calls, est ${cost:.2f} "
             f"({MODEL_SONNET})")
        if report.generation_model_calls > 0 and not args.confirm:
            print("Pass --confirm to regenerate.")
            return 1

        client = _build_anthropic_client()
        regenerated = run_generation(regen_tasks, client=client, cache=cache, store=store,
                                     retry_policy=DEFAULT_RETRY_POLICY, concurrency=args.concurrency)
        regen_by_pid = {r.item_id: r for r in regenerated}

        for pid in to_regen:
            a = responses[(pid, "sonnet-neartie-0")]
            b2 = regen_by_pid[pid]
            if a.text == b2.text:
                still_identical.append(pid)
            else:
                arm1_variant[pid] = "sonnet-neartie-1-regen"

    excluded = still_identical
    excluded_fraction = (len(excluded) / checked) if checked else 0.0
    passed = excluded_fraction <= NEAR_TIE_EXCLUSION_THRESHOLD

    result = {
        "checked": checked,
        "regenerated_prompt_ids": to_regen,
        "excluded_prompt_ids": excluded,
        "excluded_fraction": excluded_fraction,
        "passed": passed,
        "arm1_variant_by_prompt": arm1_variant,
    }
    ARM_CHECK_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"excluded {len(excluded)}/{checked} ({excluded_fraction:.1%}) near-tie pairs "
         f"(still identical after regeneration)")
    if passed:
        print(f"PASS (<= {NEAR_TIE_EXCLUSION_THRESHOLD:.0%}). Wrote {ARM_CHECK_PATH}.")
        return 0
    print(f"FAIL: {excluded_fraction:.1%} > {NEAR_TIE_EXCLUSION_THRESHOLD:.0%}. Per section 11 "
         f"item 5, the near-tie stratum is degenerating -- decoding is too deterministic under "
         f"the pinned generation config. Stop and revise the generation config (e.g. re-enable "
         f"thinking for generation, re-budget section 6.5). Do not weaken this gate.")
    return 1


# --- plan ------------------------------------------------------------------------


def _require_arm_check_passed() -> dict | None:
    arm_check = _load_arm_check()
    if arm_check is None:
        print("check-arms has not been run; run `check-arms` first (section 11 item 5 "
             "must pass before any judge call).")
        return None
    if not arm_check["passed"]:
        print("check-arms did not pass; the near-tie stratum is degenerating. Fix the "
             "generation config and rerun `generate` + `check-arms` before planning.")
        return None
    return arm_check


def cmd_plan(args) -> int:
    if not RESPONSES_PATH.exists():
        print("no generated responses found; run `generate` first.")
        return 1
    arm_check = _require_arm_check_passed()
    if arm_check is None:
        return 1

    prompts = _load_prompts()
    responses = _load_responses()
    excluded = set(arm_check["excluded_prompt_ids"])
    arm1_variant = arm_check["arm1_variant_by_prompt"]
    item_tasks = build_item_tasks(prompts, responses, excluded, arm1_variant)
    print(f"built {len(item_tasks)} items to judge "
         f"({len(prompts)} clear-gap + {len(prompts) - len(excluded)} near-tie)")

    plans = build_all_judge_plans(item_tasks, created_at=_now())

    all_planned = []
    for judge_model, plan_result in plans.items():
        all_planned.extend(plan_result.planned)
        print(f"  {judge_model}: {len(plan_result.calls)} calls "
             f"({len(plan_result.replicated_items)} items replicated x2 orders, section 5)")

    store = JsonlStore(PLANNED_PATH)
    write_planned_calls(all_planned, store)
    print(f"wrote {len(all_planned)} planned calls to {PLANNED_PATH} "
         f"(spec section 12 gap 9: written before any judge call is made)")
    return 0


# --- dry-run -----------------------------------------------------------------------


def cmd_dry_run(args) -> int:
    prompts = _load_prompts()
    gen_tasks = _generation_tasks(prompts)
    gen_cache = ResponseCache(GEN_CACHE_DIR)
    gen_store = JsonlStore(RESPONSES_PATH)

    print("=== generation ===")
    gen_total_calls = 0
    gen_total_cost = 0.0
    for model in (MODEL_SONNET, MODEL_HAIKU):
        subset = [t for t in gen_tasks if t.model == model]
        report = dry_run(generation_tasks=subset, generation_cache=gen_cache, generation_store=gen_store)
        cost = _cost(model, report.generation_model_calls, GEN_TOKENS)
        gen_total_calls += report.generation_model_calls
        gen_total_cost += cost
        print(f"  {model}: total={report.generation_total} "
             f"already_recorded={report.generation_already_recorded} "
             f"cache_hits={report.generation_cache_hits} "
             f"model_calls={report.generation_model_calls} est=${cost:.2f}")
    print(f"  generation: {gen_total_calls} model calls, est ${gen_total_cost:.2f} "
         f"(section 6.5 table: 1200 calls, ~$5.70)")

    print("=== judging ===")
    if not RESPONSES_PATH.exists():
        judge_total_calls = 1320 * len(JUDGES)
        judge_total_cost = sum(_cost(m, 1320, JUDGE_TOKENS) for m in JUDGES)
        print("  no generated responses yet; showing the full theoretical plan "
             "(section 6.5), not a live cache/store check")
        for judge_model in JUDGES:
            print(f"  {judge_model}: 1320 calls (theoretical) "
                 f"est=${_cost(judge_model, 1320, JUDGE_TOKENS):.2f}")
    else:
        responses = _load_responses()
        arm_check = _load_arm_check()
        if arm_check is None:
            excluded, arm1_variant = set(), {}
            print("  check-arms has not been run; near-tie exclusions not yet known, "
                 "reporting all 600 pairs")
        else:
            excluded = set(arm_check["excluded_prompt_ids"])
            arm1_variant = arm_check["arm1_variant_by_prompt"]
        item_tasks = build_item_tasks(prompts, responses, excluded, arm1_variant)
        plans = build_all_judge_plans(item_tasks, created_at=_now())

        judge_cache = ResponseCache(JUDGE_CACHE_DIR)
        judge_store = JsonlStore(JUDGMENTS_PATH)
        judge_total_calls = 0
        judge_total_cost = 0.0
        for judge_model, plan_result in plans.items():
            report = dry_run(pairwise_calls=plan_result.calls, judge_cache=judge_cache,
                             pairwise_store=judge_store)
            cost = _cost(judge_model, report.pairwise_model_calls, JUDGE_TOKENS)
            judge_total_calls += report.pairwise_model_calls
            judge_total_cost += cost
            print(f"  {judge_model}: total={report.pairwise_total} "
                 f"already_recorded={report.pairwise_already_recorded} "
                 f"cache_hits={report.pairwise_cache_hits} "
                 f"model_calls={report.pairwise_model_calls} est=${cost:.2f}")
    print(f"  judging: {judge_total_calls} model calls, est ${judge_total_cost:.2f} "
         f"(section 6.5 table: 3,960 calls, ~$17.00)")

    print(f"=== combined: {gen_total_calls + judge_total_calls} model calls, "
         f"est ${gen_total_cost + judge_total_cost:.2f} (section 6.5 table: ~$23) ===")
    return 0


# --- judge -------------------------------------------------------------------------


def cmd_judge(args) -> int:
    if not PLANNED_PATH.exists():
        print("no plan found; run `plan` first (spec section 12 gap 9 / D11: the plan "
             "must be written before any judge call is made).")
        return 1
    arm_check = _require_arm_check_passed()
    if arm_check is None:
        return 1

    prompts = _load_prompts()
    responses = _load_responses()
    excluded = set(arm_check["excluded_prompt_ids"])
    arm1_variant = arm_check["arm1_variant_by_prompt"]
    item_tasks = build_item_tasks(prompts, responses, excluded, arm1_variant)
    plans = build_all_judge_plans(item_tasks, created_at=_now())

    judge_cache = ResponseCache(JUDGE_CACHE_DIR)
    judge_store = JsonlStore(JUDGMENTS_PATH)

    total_calls = 0
    total_cost = 0.0
    print("predicted judge calls:")
    for judge_model, plan_result in plans.items():
        report = dry_run(pairwise_calls=plan_result.calls, judge_cache=judge_cache,
                         pairwise_store=judge_store)
        cost = _cost(judge_model, report.pairwise_model_calls, JUDGE_TOKENS)
        total_calls += report.pairwise_model_calls
        total_cost += cost
        print(f"  {judge_model}: {report.pairwise_model_calls} model calls, est ${cost:.2f}")
    print(f"total: {total_calls} model calls, est ${total_cost:.2f}")

    if not args.confirm:
        print("Pass --confirm to proceed.")
        return 1

    client = _build_anthropic_client()
    all_calls = [c for plan_result in plans.values() for c in plan_result.calls]
    judgments = run_pairwise_judging(all_calls, run_id=RUN_ID, client=client, cache=judge_cache,
                                     store=judge_store, retry_policy=DEFAULT_RETRY_POLICY,
                                     concurrency=args.concurrency)
    print(f"judged {len(judgments)} calls (of {len(all_calls)}); store: {JUDGMENTS_PATH}")
    return 0


# --- gate --------------------------------------------------------------------------


def cmd_gate(args) -> int:
    if not PLANNED_PATH.exists():
        print("no plan found; run `plan` first.")
        return 1
    if not JUDGMENTS_PATH.exists():
        print("no judgments found; run `judge` first.")
        return 1

    planned = list(JsonlStore(PLANNED_PATH).read_raw())
    realized = list(JsonlStore(JUDGMENTS_PATH).read_raw())
    result = gate_pairwise_realized(planned, realized)

    print(f"normal: {len(result.normal)}")
    print(f"promotions: {result.promotion_count} (does not block, spec section 12 gap 9)")
    print(f"plumbing errors: {result.plumbing_error_count} (blocks the run if > 0)")
    print(f"exclusions: {result.exclusion_count} (D10 exclusion path)")
    print(f"blocks_run: {result.blocks_run}")
    if result.blocks_run:
        print("BLOCKED: plumbing errors present -- records the plan cannot explain, or a "
             "side never planned. Do not proceed to analysis until explained.")
        return 1
    print("not blocked.")
    return 0


# --- CLI ------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Sub:
    name: str
    help: str
    func: object
    needs_confirm: bool


_SUBCOMMANDS = (
    _Sub("generate", "Build and (with --confirm) execute the 1,200 generation calls.",
        cmd_generate, True),
    _Sub("check-arms", "The section 11 item 5 near-tie distinctness gate.", cmd_check_arms, True),
    _Sub("plan", "Build the pairwise judge plan (PlannedJudgeCall records). No model calls.",
        cmd_plan, False),
    _Sub("dry-run", "Report predicted calls and cost against section 6.5. No model calls.",
        cmd_dry_run, False),
    _Sub("judge", "Execute the planned judge calls.", cmd_judge, True),
    _Sub("gate", "Run the plan-join gate (spec section 12 gap 9). No model calls.",
        cmd_gate, False),
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_study.py",
        description="Position-bias study driver -- one phase per invocation. "
                    "See the module docstring for the full pipeline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for spec in _SUBCOMMANDS:
        p = sub.add_parser(spec.name, help=spec.help)
        if spec.needs_confirm:
            p.add_argument("--confirm", action="store_true",
                           help="required to actually spend money; omit to preview only")
            p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
        p.set_defaults(func=spec.func)
    return parser


def main() -> int:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    parser = build_arg_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
