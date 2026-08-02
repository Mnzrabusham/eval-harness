"""judge/: judgment prompt construction and verdict parsing.

Emits the docs/data-model.md record types. Pairwise judgments are emitted
in presentation terms (first/second/tie), never variant terms, and
counterbalancing is structural: ``counterbalanced_pairwise_tasks`` always
produces both presentation orders of a pair under the same pair_id.
"""

from .parse import (
    VerdictParseError,
    pairwise_judgment_from_task,
    parse_binary_verdict,
    parse_pairwise_verdict,
    parse_scalar_verdict,
    response_judgment_from_verdict,
)
from .prompts import (
    PairwiseTask,
    ResponseSide,
    build_binary_prompt,
    build_pairwise_prompt,
    build_scalar_prompt,
    counterbalanced_pairwise_tasks,
)
from .records import PairwiseJudgment, ResponseJudgment, RunConfig

__all__ = [
    "PairwiseJudgment",
    "PairwiseTask",
    "ResponseJudgment",
    "ResponseSide",
    "RunConfig",
    "VerdictParseError",
    "build_binary_prompt",
    "build_pairwise_prompt",
    "build_scalar_prompt",
    "counterbalanced_pairwise_tasks",
    "pairwise_judgment_from_task",
    "parse_binary_verdict",
    "parse_pairwise_verdict",
    "parse_scalar_verdict",
    "response_judgment_from_verdict",
]
