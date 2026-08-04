"""Verify position-bias-study.md §6.5's token-planning assumptions.

Renders every prompt in study/prompt_set.jsonl through the *actual*
generation and judge prompt templates (evalkit.judge.build_generation_prompt,
evalkit.judge.build_pairwise_prompt -- the same functions the runner calls,
not a re-description of them) and counts input tokens with the Anthropic
token-counting endpoint, per model. Prints realized mean and p95 input
tokens for both call types against §6.5's planning values (120 for
generation, 1,100 for judging) and writes the same numbers to
study/token_planning_check.csv.

Token counting is free (no model credit) and draws from a separate RPM pool
from the Messages API -- confirmed against the live docs 2026-08-04, not
assumed. This script is safe to run repeatedly without touching the ~$23
budget in §6.5.

Not run as part of any session that produced this file. Requires
ANTHROPIC_API_KEY (or an `ant auth login` profile the default client
picks up) in the environment. Run manually:

    python study/count_tokens.py

Caveat on the judge measurement: §2 specifies the real judge calls use
structured outputs (output_config.format with a verdict JSON schema), which
may add a small amount of schema overhead beyond what's measured here --
this script measures the prompt-template contribution only, via the same
free-text VERDICT-line template evalkit.judge.build_pairwise_prompt renders
for every judge call in this codebase.

Caveat on the generation measurement: real per-response text doesn't exist
yet (generation hasn't run -- position-bias-study.md §11 item 4 is still
open), so the judge-call measurement uses placeholder response text sized
at 250 words, the midpoint of the 150-350 word target §3.2's generation
instruction asks for. That isolates the effect of the *prompt's* realized
length -- the thing this check exists to verify -- rather than adding
response-length variance the original planning number already assumed
away by fixing a word count.
"""

from __future__ import annotations

import csv
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

from evalkit.judge import build_generation_prompt, build_pairwise_prompt

PROMPT_SET = Path("study/prompt_set.jsonl")
OUT_CSV = Path("study/token_planning_check.csv")

# §3.2: claude-sonnet-5 generates for both strata (900 of 1,200 generation
# calls per §6.5's budget table), claude-haiku-4-5 generates the clear-gap
# weak arm only (300 calls). Both use the identical generation prompt.
GENERATION_MODELS = ["claude-sonnet-5", "claude-haiku-4-5"]
GENERATION_CALL_WEIGHTS = {"claude-sonnet-5": 900, "claude-haiku-4-5": 300}

# §2: all three judges share one fixed prompt template.
JUDGE_MODELS = ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"]

PLANNED_GENERATION_INPUT = 120
PLANNED_JUDGE_INPUT = 1100

_PLACEHOLDER_WORDS = 250
_FILLER_SENTENCE = (
    "This placeholder response stands in for a real generated answer of "
    "roughly the target length, so the judge-prompt token count reflects "
    "the realized prompt length rather than response-length variance."
)
_filler_words = _FILLER_SENTENCE.split()
_repeated = _filler_words * (_PLACEHOLDER_WORDS // len(_filler_words) + 1)
_PLACEHOLDER_RESPONSE = " ".join(_repeated[:_PLACEHOLDER_WORDS])

CONCURRENCY = 8


def load_prompts() -> list[dict]:
    rows = []
    with PROMPT_SET.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mean_p95(values: list[int]) -> tuple[float, float]:
    mean = statistics.mean(values)
    p95 = statistics.quantiles(values, n=100, method="inclusive")[94]
    return mean, p95


def _count(client: anthropic.Anthropic, model: str, content: str) -> int:
    response = client.messages.count_tokens(
        model=model, messages=[{"role": "user", "content": content}]
    )
    return response.input_tokens


def count_generation_tokens(client: anthropic.Anthropic, model: str,
                            prompts: list[dict]) -> list[int]:
    rendered = [build_generation_prompt(row["text"]) for row in prompts]
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        return list(pool.map(lambda c: _count(client, model, c), rendered))


def count_judge_tokens(client: anthropic.Anthropic, model: str,
                       prompts: list[dict]) -> list[int]:
    rendered = [
        build_pairwise_prompt(row["text"], _PLACEHOLDER_RESPONSE, _PLACEHOLDER_RESPONSE,
                              ties_allowed=True)
        for row in prompts
    ]
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        return list(pool.map(lambda c: _count(client, model, c), rendered))


def main() -> None:
    client = anthropic.Anthropic()
    prompts = load_prompts()
    print(f"loaded {len(prompts)} prompts from {PROMPT_SET}")

    csv_rows = []

    print("\n--- generation calls ---")
    print(f"planning value: {PLANNED_GENERATION_INPUT} input tokens")
    weighted_sum = 0.0
    for model in GENERATION_MODELS:
        counts = count_generation_tokens(client, model, prompts)
        mean, p95 = mean_p95(counts)
        delta_pct = (mean - PLANNED_GENERATION_INPUT) / PLANNED_GENERATION_INPUT * 100
        print(f"{model}: mean={mean:.1f}  p95={p95:.1f}  "
              f"delta_vs_planning={delta_pct:+.1f}%")
        csv_rows.append({
            "call_type": "generation", "model": model, "n": len(counts),
            "mean_input_tokens": f"{mean:.1f}", "p95_input_tokens": f"{p95:.1f}",
            "planning_value": PLANNED_GENERATION_INPUT, "delta_pct": f"{delta_pct:+.1f}",
        })
        weighted_sum += mean * GENERATION_CALL_WEIGHTS[model]
    total_calls = sum(GENERATION_CALL_WEIGHTS.values())
    weighted_mean = weighted_sum / total_calls
    print(f"call-weighted mean across both generation models "
          f"({GENERATION_CALL_WEIGHTS}): {weighted_mean:.1f}  "
          f"delta_vs_planning={(weighted_mean - PLANNED_GENERATION_INPUT) / PLANNED_GENERATION_INPUT * 100:+.1f}%")

    print(f"\n--- judge calls (placeholder responses, {_PLACEHOLDER_WORDS} words each side) ---")
    print(f"planning value: {PLANNED_JUDGE_INPUT} input tokens")
    for model in JUDGE_MODELS:
        counts = count_judge_tokens(client, model, prompts)
        mean, p95 = mean_p95(counts)
        delta_pct = (mean - PLANNED_JUDGE_INPUT) / PLANNED_JUDGE_INPUT * 100
        print(f"{model}: mean={mean:.1f}  p95={p95:.1f}  "
              f"delta_vs_planning={delta_pct:+.1f}%")
        csv_rows.append({
            "call_type": "judge", "model": model, "n": len(counts),
            "mean_input_tokens": f"{mean:.1f}", "p95_input_tokens": f"{p95:.1f}",
            "planning_value": PLANNED_JUDGE_INPUT, "delta_pct": f"{delta_pct:+.1f}",
        })

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "call_type", "model", "n", "mean_input_tokens", "p95_input_tokens",
            "planning_value", "delta_pct",
        ])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
