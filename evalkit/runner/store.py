"""Append-only JSONL record store, so an interrupted run is resumable.

Records are dataclasses (evalkit.judge's RunConfig, ResponseJudgment,
PairwiseJudgment, or this package's GeneratedResponse). Each write is
flushed and fsynced before returning, under a lock: a crash mid-run loses
at most the one record in flight, never one already written, and
concurrent writers from a thread pool never interleave partial lines.

On restart, ``written_keys`` replays the file to find which natural keys
(response_id, judge_call_id, ...) are already recorded, so the runner
skips redoing that work instead of losing it or duplicating it.
"""

from __future__ import annotations

import dataclasses
import json
import os
import threading
from pathlib import Path
from typing import Callable, Iterator, TypeVar

T = TypeVar("T")

__all__ = ["JsonlStore"]


class JsonlStore:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, record: object) -> None:
        line = json.dumps(dataclasses.asdict(record), sort_keys=True)
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())

    def read_raw(self) -> Iterator[dict]:
        if not self._path.exists():
            return
        with open(self._path, "r", encoding="utf-8") as fh:
            for lineno, raw_line in enumerate(fh, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{self._path}: unparseable record on line {lineno} -- "
                        f"likely a truncated write from a crash mid-append. "
                        f"Remove or repair that line before resuming this run."
                    ) from exc

    def written_keys(self, key_fn: Callable[[dict], T]) -> set[T]:
        return {key_fn(row) for row in self.read_raw()}
