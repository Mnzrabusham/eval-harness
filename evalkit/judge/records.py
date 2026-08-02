"""Judgment records (docs/data-model.md).

Dataclass forms of the data model's three structures, with the data
model's write-time constraints enforced at construction:

- ``judge_model`` must be an immutable snapshot id; floating aliases
  ending in "-latest" are rejected at write time.
- ``PairwiseJudgment.judgment`` is recorded in presentation terms
  (first/second/tie), never variant terms; which variant won is derived at
  analysis time (evalkit.stats.reduction).
- RunConfig is immutable per run_id (frozen dataclass; persistence-level
  immutability is the store's job).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["PairwiseJudgment", "ResponseJudgment", "RunConfig"]

_JUDGMENT_TYPES = ("pairwise", "scalar", "binary")
_PAIRWISE_VERDICTS = ("first", "second", "tie")


def _check_judge_model(judge_model: str) -> None:
    if str(judge_model).endswith("-latest"):
        raise ValueError(
            f"judge_model must be an immutable snapshot id; floating alias "
            f"{judge_model!r} rejected at write time (docs/data-model.md)"
        )


@dataclass(frozen=True)
class RunConfig:
    """One per run_id, immutable once written."""

    run_id: str
    judgment_type: str
    created_at: str
    scale_min: float | None = None
    scale_max: float | None = None
    ties_allowed: bool = True
    baseline_variant_id: str | None = None

    def __post_init__(self) -> None:
        if self.judgment_type not in _JUDGMENT_TYPES:
            raise ValueError(f"judgment_type must be one of {_JUDGMENT_TYPES}, got {self.judgment_type!r}")
        if self.judgment_type == "scalar":
            if self.scale_min is None or self.scale_max is None:
                raise ValueError("scalar runs require scale_min and scale_max")
            if not self.scale_min < self.scale_max:
                raise ValueError("scale_min must be < scale_max")


@dataclass(frozen=True)
class ResponseJudgment:
    """Scalar and binary judgments: one judge call on one response.

    Replicate judge calls are separate rows with distinct judge_call_id.
    """

    run_id: str
    item_id: str
    source_doc_id: str | None
    variant_id: str
    response_id: str
    gen_seed: object
    response_tokens: int
    judge_model: str
    judge_config_id: str
    judge_call_id: str
    judgment: float
    created_at: str
    annotator_id: str | None = None

    def __post_init__(self) -> None:
        _check_judge_model(self.judge_model)


@dataclass(frozen=True)
class PairwiseJudgment:
    """Preference judgments. ``judgment`` is in presentation terms only."""

    run_id: str
    item_id: str
    source_doc_id: str | None
    pair_id: str
    variant_first: str
    variant_second: str
    response_id_first: str
    response_id_second: str
    gen_seed_first: object
    gen_seed_second: object
    tokens_first: int
    tokens_second: int
    judge_model: str
    judge_config_id: str
    judge_call_id: str
    judgment: str
    created_at: str
    annotator_id: str | None = None

    def __post_init__(self) -> None:
        _check_judge_model(self.judge_model)
        if self.judgment not in _PAIRWISE_VERDICTS:
            raise ValueError(
                f"pairwise judgment must be one of {_PAIRWISE_VERDICTS} "
                f"(presentation terms, never variant terms), got {self.judgment!r}"
            )
