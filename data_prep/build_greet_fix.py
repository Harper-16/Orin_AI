#!/usr/bin/env python3
"""Build a tiny, focused greet/small-talk fix set.

human3 failed because 20 smalltalk rows drowned in ~15k DailyDialog rows:
800 batch-1 steps barely showed each greeting once, while the noise shifted
the fragile 'hi' peak away. This mix flips the ratio: smalltalk x40 plus
identity core x10, shuffled — ~1.4k rows total, every step teaches greetings.
"""
import random
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
SEED = 7


def read_rows(name):
    path = BASE / name
    if not path.is_file():
        print(f"skip missing: {name}")
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row.get("messages"), list):
                rows.append(row)
    return rows


def main():
    rng = random.Random(SEED)
    out = read_rows("examples/smalltalk_fix.jsonl") * 40
    out += read_rows("examples/conversation_core.jsonl") * 10
    rng.shuffle(out)
    dest = BASE / "data/greet_fix.jsonl"
    with dest.open("w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"WROTE {len(out):,} rows -> {dest}")


if __name__ == "__main__":
    main()
