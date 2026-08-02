"""Strict verdict parsing and record construction.

Parsers refuse to guess: a missing verdict, conflicting verdicts, a
disallowed tie, or an out-of-scale score raises ``VerdictParseError``
instead of being coerced to something plausible. Pairwise verdicts are
returned — and recorded — in presentation terms (first/second/tie), never
variant terms (data-model constraint); mapping to variants happens at
analysis time in ``evalkit.stats.reduction``.
"""

from __future__ import annotations

import re

from .prompts import PairwiseTask
from .records import PairwiseJudgment, ResponseJudgment

__all__ = [
    "VerdictParseError",
    "pairwise_judgment_from_task",
    "parse_binary_verdict",
    "parse_pairwise_verdict",
    "parse_scalar_verdict",
    "response_judgment_from_verdict",
]


class VerdictParseError(ValueError):
    """The judge's output did not contain exactly one well-formed verdict."""


_PAIRWISE_RE = re.compile(r"VERDICT\s*:\s*(FIRST|SECOND|TIE)\b", re.IGNORECASE)
_BINARY_RE = re.compile(r"VERDICT\s*:\s*(PASS|FAIL)\b", re.IGNORECASE)
_SCORE_RE = re.compile(r"SCORE\s*:\s*([+-]?\d+(?:\.\d+)?)")


def _unique(values: set, text: str, what: str):
    if not values:
        raise VerdictParseError(f"no {what} found in judge output: {text[:200]!r}")
    if len(values) > 1:
        raise VerdictParseError(f"conflicting {what}s {sorted(values)} in judge output")
    return values.pop()


def parse_pairwise_verdict(text: str, *, ties_allowed: bool = True) -> str:
    """Parse to 'first' | 'second' | 'tie' — presentation terms only."""
    verdict = _unique({m.group(1).lower() for m in _PAIRWISE_RE.finditer(text)},
                      text, "pairwise VERDICT")
    if verdict == "tie" and not ties_allowed:
        raise VerdictParseError("judge returned a tie but ties are not allowed for this run")
    return verdict


def parse_binary_verdict(text: str) -> int:
    """Parse to 1 (pass) or 0 (fail)."""
    verdict = _unique({m.group(1).lower() for m in _BINARY_RE.finditer(text)},
                      text, "binary VERDICT")
    return 1 if verdict == "pass" else 0


def parse_scalar_verdict(text: str, *, scale_min: float, scale_max: float) -> float:
    """Parse a SCORE line; out-of-scale values are an error, not clamped."""
    values = {float(m.group(1)) for m in _SCORE_RE.finditer(text)}
    value = _unique(values, text, "SCORE")
    if not scale_min <= value <= scale_max:
        raise VerdictParseError(f"score {value} outside scale [{scale_min}, {scale_max}]")
    return value


def pairwise_judgment_from_task(task: PairwiseTask, judge_output: str, *,
                                run_id: str, judge_model: str, judge_config_id: str,
                                judge_call_id: str, created_at: str,
                                annotator_id: str | None = None) -> PairwiseJudgment:
    """Parse the judge output for ``task`` into a PairwiseJudgment record.

    The record's judgment is the parsed presentation-term verdict; variant
    identity flows only into the structural variant_first/variant_second
    fields, so a verdict can never be emitted in variant terms.
    """
    verdict = parse_pairwise_verdict(judge_output, ties_allowed=task.ties_allowed)
    return PairwiseJudgment(
        run_id=run_id,
        item_id=task.item_id,
        source_doc_id=task.source_doc_id,
        pair_id=task.pair_id,
        variant_first=task.first.variant_id,
        variant_second=task.second.variant_id,
        response_id_first=task.first.response_id,
        response_id_second=task.second.response_id,
        gen_seed_first=task.first.gen_seed,
        gen_seed_second=task.second.gen_seed,
        tokens_first=task.first.tokens,
        tokens_second=task.second.tokens,
        judge_model=judge_model,
        judge_config_id=judge_config_id,
        judge_call_id=judge_call_id,
        annotator_id=annotator_id,
        judgment=verdict,
        created_at=created_at,
    )


def response_judgment_from_verdict(judge_output: str, *, judgment_type: str,
                                   run_id: str, item_id: str,
                                   source_doc_id: str | None, variant_id: str,
                                   response_id: str, gen_seed: object,
                                   response_tokens: int, judge_model: str,
                                   judge_config_id: str, judge_call_id: str,
                                   created_at: str,
                                   scale_min: float | None = None,
                                   scale_max: float | None = None,
                                   annotator_id: str | None = None) -> ResponseJudgment:
    """Parse a scalar or binary judge output into a ResponseJudgment record."""
    if judgment_type == "scalar":
        if scale_min is None or scale_max is None:
            raise ValueError("scalar judgments require scale_min and scale_max")
        judgment = parse_scalar_verdict(judge_output, scale_min=scale_min, scale_max=scale_max)
    elif judgment_type == "binary":
        judgment = float(parse_binary_verdict(judge_output))
    else:
        raise ValueError(f"judgment_type must be 'scalar' or 'binary', got {judgment_type!r}")
    return ResponseJudgment(
        run_id=run_id,
        item_id=item_id,
        source_doc_id=source_doc_id,
        variant_id=variant_id,
        response_id=response_id,
        gen_seed=gen_seed,
        response_tokens=response_tokens,
        judge_model=judge_model,
        judge_config_id=judge_config_id,
        judge_call_id=judge_call_id,
        annotator_id=annotator_id,
        judgment=judgment,
        created_at=created_at,
    )
