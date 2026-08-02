"""Judgment prompt construction (docs/statistics-spec.md §3–§5).

Pairwise prompts refer to the two responses only by presentation position
(FIRST / SECOND); variant identity never appears in a prompt, so a parsed
verdict cannot be expressed in variant terms. Counterbalancing is
structural (§1.2, §14.1): ``counterbalanced_pairwise_tasks`` is the only
way to build pairwise tasks and it always returns both presentation orders
of the pair under the same pair_id.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "PairwiseTask",
    "ResponseSide",
    "build_binary_prompt",
    "build_pairwise_prompt",
    "build_scalar_prompt",
    "counterbalanced_pairwise_tasks",
]


@dataclass(frozen=True)
class ResponseSide:
    """One response entering a pairwise comparison.

    ``tokens`` is the response's token count (data model:
    ``tokens_first`` / ``tokens_second``, needed for verbosity analysis).
    Supply the generating tokenizer's count; this module does not
    approximate it.
    """

    variant_id: str
    response_id: str
    text: str
    tokens: int
    gen_seed: object = None


@dataclass(frozen=True)
class PairwiseTask:
    """A single judge call to make: one presentation order of one pair."""

    item_id: str
    source_doc_id: str | None
    pair_id: str
    first: ResponseSide
    second: ResponseSide
    ties_allowed: bool
    prompt: str


_PAIRWISE_TEMPLATE = """\
You are judging two responses to the same task. Judge only what is
written; do not speculate about how either response was produced.

# Task

{task}

# Response FIRST

{first}

# Response SECOND

{second}

Which response accomplishes the task better?
Reply with exactly one line:
{verdict_line}
"""

_SCALAR_TEMPLATE = """\
You are scoring one response to a task against a rubric. Judge only what
is written; do not speculate about how the response was produced.

# Task

{task}

# Response

{response}

# Rubric (score {lo} to {hi})

{rubric}

Score the response on the scale from {lo} (worst) to {hi} (best).
Reply with exactly one line:
SCORE: <number between {lo} and {hi}>
"""

_BINARY_TEMPLATE = """\
You are judging whether one response to a task meets the pass criteria.
Judge only what is written; do not speculate about how the response was
produced.

# Task

{task}

# Response

{response}

# Pass criteria

{criteria}

Does the response meet the pass criteria?
Reply with exactly one line:
VERDICT: PASS | VERDICT: FAIL
"""


def build_pairwise_prompt(task_text: str, first_text: str, second_text: str,
                          *, ties_allowed: bool = True) -> str:
    """Pairwise prompt in presentation terms only (no variant identity)."""
    verdict_line = (
        "VERDICT: FIRST | VERDICT: SECOND | VERDICT: TIE"
        if ties_allowed
        else "VERDICT: FIRST | VERDICT: SECOND"
    )
    return _PAIRWISE_TEMPLATE.format(task=task_text, first=first_text,
                                     second=second_text, verdict_line=verdict_line)


def build_scalar_prompt(task_text: str, response_text: str, *, rubric: str,
                        scale_min: float, scale_max: float) -> str:
    if not scale_min < scale_max:
        raise ValueError("scale_min must be < scale_max")
    return _SCALAR_TEMPLATE.format(task=task_text, response=response_text,
                                   rubric=rubric, lo=scale_min, hi=scale_max)


def build_binary_prompt(task_text: str, response_text: str, *, criteria: str) -> str:
    return _BINARY_TEMPLATE.format(task=task_text, response=response_text,
                                   criteria=criteria)


def counterbalanced_pairwise_tasks(*, item_id: str, task_text: str,
                                   side_a: ResponseSide, side_b: ResponseSide,
                                   source_doc_id: str | None = None,
                                   ties_allowed: bool = True) -> tuple[PairwiseTask, PairwiseTask]:
    """Both presentation orders of (side_a, side_b) under one pair_id.

    pair_id is derived from the item and the *unordered* response pair, so
    both orders — and replicate calls on either — share it. Keying on
    response ids (not just variant ids) keeps multiple response pairs per
    item distinct, matching §1.2's pair-and-order reduction.
    """
    if side_a.response_id == side_b.response_id:
        raise ValueError("a pair requires two distinct responses")
    lo, hi = sorted((side_a.response_id, side_b.response_id))
    pair_id = f"{item_id}::{lo}::{hi}"

    def _task(first: ResponseSide, second: ResponseSide) -> PairwiseTask:
        return PairwiseTask(
            item_id=item_id,
            source_doc_id=source_doc_id,
            pair_id=pair_id,
            first=first,
            second=second,
            ties_allowed=ties_allowed,
            prompt=build_pairwise_prompt(task_text, first.text, second.text,
                                         ties_allowed=ties_allowed),
        )

    return _task(side_a, side_b), _task(side_b, side_a)
