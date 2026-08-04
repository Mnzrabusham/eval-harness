"""Build the position-bias study prompt set from Arena-Hard-Auto v0.1.

Deterministic: same seed and same source revision give the same 300 prompts.
Run: python study/build_prompt_set.py
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

SOURCE = "lmarena-ai/arena-hard-auto-v0.1"
REVISION = "2a69efe86cff85c593cfd4c2b4491c128e52c6f1" 
SEED = 20260804        # record this in the design doc
N_PROMPTS = 300
MIN_CHARS, MAX_CHARS = 40, 2500
OUT = Path("study/prompt_set.jsonl")

# Excluded: a response naming its own model or maker would leak arm
# identity to the judge.
SELF_ID = re.compile(
    r"(what model are you|who (made|created|trained) you"
    r"|are you (chatgpt|gpt|claude|gemini|llama)"
    r"|your training data|as an ai language model|which llm)",
    re.IGNORECASE,
)


def main() -> None:
    ds = load_dataset(SOURCE, split="train", revision=REVISION)

    pool = []
    dropped = {"self_id": 0, "too_short": 0, "too_long": 0}
    for row in ds:
        text = row["turns"][0]["content"]
        if SELF_ID.search(text):
            dropped["self_id"] += 1
            continue
        if len(text) < MIN_CHARS:
            dropped["too_short"] += 1
            continue
        if len(text) > MAX_CHARS:
            dropped["too_long"] += 1
            continue
        pool.append({
            "prompt_id": row["question_id"],
            "text": text,
            "category": row["category"],
            "cluster": row["cluster"],
            "source": SOURCE,
        })

    # Deterministic order before sampling, so the seed alone fixes the draw.
    pool.sort(key=lambda r: r["prompt_id"])

    if len(pool) < N_PROMPTS:
        raise SystemExit(f"pool has {len(pool)}, need {N_PROMPTS}")

    selected = random.Random(SEED).sample(pool, N_PROMPTS)
    selected.sort(key=lambda r: r["prompt_id"])

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    lengths = sorted(len(r["text"]) for r in selected)

    mean_chars = sum(lengths) / len(lengths)

    print(f"source        {SOURCE} revision={REVISION}")
    print(f"seed          {SEED}")
    print(f"pool          {len(ds)} -> {len(pool)} after filters {dropped}")
    print(f"selected      {len(selected)}")
    print(f"chars         min {lengths[0]}  mean {mean_chars:.0f}"
      f"  median {lengths[len(lengths)//2]}  max {lengths[-1]}")
    print(f"sha256        {digest}")



if __name__ == "__main__":
    main()