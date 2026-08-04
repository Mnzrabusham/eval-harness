"""runner/: execute generation and judgment calls against models.

Ties evalkit.judge (prompt construction, verdict parsing, and the
docs/data-model.md record types) to a disk-backed response cache, bounded
concurrency, retry-with-backoff, and an incremental on-disk store, so an
interrupted run resumes rather than restarts. See ``dry_run`` for counting
the model calls a configuration would make before running it for real.
"""

from .cache import ResponseCache
from .client import Completion, ModelClient, RateLimitError
from .completeness import incomplete_pairs
from .concurrency import run_concurrent
from .execute import get_or_generate
from .generate import (
    GeneratedResponse,
    GenerationTask,
    generate_response,
    generation_response_id,
    run_generation,
)
from .ids import derive_id, derive_seed
from .judging import (
    PairwiseJudgeCall,
    ResponseJudgeCall,
    pairwise_call_id,
    response_call_id,
    run_pairwise_judging,
    run_response_judging,
    validate_judge_model,
)
from .pairwise_plan import (
    PairwiseCallGate,
    PairwiseRequestPlan,
    build_pairwise_request_plan,
    draw_replicate_item_subset,
    gate_pairwise_realized,
    replicate_subset_size,
    write_planned_calls,
)
from .plan import DryRunReport, dry_run
from .retry import RetryPolicy, call_with_retry
from .store import JsonlStore

__all__ = [
    "Completion",
    "DryRunReport",
    "GeneratedResponse",
    "GenerationTask",
    "JsonlStore",
    "ModelClient",
    "PairwiseCallGate",
    "PairwiseJudgeCall",
    "PairwiseRequestPlan",
    "RateLimitError",
    "ResponseCache",
    "ResponseJudgeCall",
    "RetryPolicy",
    "build_pairwise_request_plan",
    "call_with_retry",
    "derive_id",
    "derive_seed",
    "draw_replicate_item_subset",
    "dry_run",
    "gate_pairwise_realized",
    "generate_response",
    "generation_response_id",
    "get_or_generate",
    "incomplete_pairs",
    "pairwise_call_id",
    "replicate_subset_size",
    "response_call_id",
    "run_concurrent",
    "run_generation",
    "run_pairwise_judging",
    "run_response_judging",
    "validate_judge_model",
    "write_planned_calls",
]
