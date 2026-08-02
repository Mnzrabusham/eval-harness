"""Disk-backed response cache keyed on (prompt, model, seed, sampling params).

The seed is part of the key: two calls that differ only by seed are
distinct cache entries, so replicate judge calls (data-model.md:
``judge_call_id`` unique per invocation, replicates are separate rows)
are never collapsed into one cached completion. A cache hit means a
rerun -- interrupted, resumed, or simply repeated -- pays nothing to
reproduce a response already generated.

``lock`` gives callers (evalkit.runner.execute) a per-key mutex so that
concurrent tasks which happen to resolve to the same key serialize around
the check-then-call-then-store sequence: only the first pays for the
model call, the rest block briefly and then read what it wrote, rather
than a bounded-concurrency batch racing the same call several times over.
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from .client import Completion

__all__ = ["ResponseCache"]


class ResponseCache:
    def __init__(self, cache_dir: str | Path) -> None:
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    @staticmethod
    def key(prompt: str, *, model: str, seed: int, sampling_params: Mapping[str, Any]) -> str:
        payload = json.dumps(
            {"prompt": prompt, "model": model, "seed": seed,
             "sampling_params": sampling_params},
            sort_keys=True, default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def lock(self, key: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(key, threading.Lock())

    def _path(self, key: str) -> Path:
        return self._dir / f"{key}.json"

    def get_by_key(self, key: str) -> Completion | None:
        path = self._path(key)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return Completion(text=data["text"], tokens=data["tokens"])

    def set_by_key(self, key: str, completion: Completion) -> None:
        path = self._path(key)
        # Unique per call (not just per key): concurrent writers for the same
        # key must not share a tmp file, or one's replace() can lose a race
        # against another's write/delete of that same path.
        tmp = self._dir / f"{key}.{uuid.uuid4().hex}.tmp"
        tmp.write_text(json.dumps(asdict(completion)), encoding="utf-8")
        tmp.replace(path)

    def get(self, prompt: str, *, model: str, seed: int,
             sampling_params: Mapping[str, Any]) -> Completion | None:
        return self.get_by_key(self.key(prompt, model=model, seed=seed,
                                        sampling_params=sampling_params))

    def set(self, prompt: str, *, model: str, seed: int, sampling_params: Mapping[str, Any],
            completion: Completion) -> None:
        self.set_by_key(self.key(prompt, model=model, seed=seed,
                                 sampling_params=sampling_params), completion)
