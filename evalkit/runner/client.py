"""Model client interface used by the runner.

The runner is client-agnostic: it depends only on this protocol, so
production code plugs in a real HTTP-backed client and tests plug in a
mock, with no network calls either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

__all__ = ["Completion", "ModelClient", "RateLimitError"]


class RateLimitError(Exception):
    """Raised by a ModelClient when the provider rate-limits a call.

    The runner's retry policy (evalkit.runner.retry) catches exactly this
    exception and backs off; any other exception propagates immediately
    and aborts the call.
    """


@dataclass(frozen=True)
class Completion:
    """One model response: the generated text and its token count."""

    text: str
    tokens: int


class ModelClient(Protocol):
    """A model endpoint: prompt and sampling params in, Completion out."""

    def complete(self, prompt: str, *, model: str, seed: int,
                 sampling_params: Mapping[str, Any]) -> Completion:
        ...
