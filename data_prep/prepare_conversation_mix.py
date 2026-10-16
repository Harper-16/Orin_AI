#!/usr/bin/env python3
"""Combine real multi-turn dialogue with a small, repeated conversation starter set.

Also accepts SmolTalk-style row files (pass --dialogues multiple times).
Dedupes by message text so mixing overlapping sets does not overweight repeats.
"""
import argparse
import hashlib
import json
from pathlib import Path


def read_rows(path: Path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row.get("messages"), list):
                yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dialogues", action="append", default=["data/conversation_focused.jsonl"],
                        help="Dialogue JSONL file(s). Repeatable; SmolTalk files work too.")
    parser.add_argument("--starter", default="examples/conversation_core.jsonl")
    parser.add_argument("--output", default="data/conversation_mix.jsonl")
    parser.add_argument("--starter-repeats", type=int, default=5)
    args = parser.parse_args()
    if args.starter_repeats < 1:
        parser.error("--starter-repeats must be positive")
    dialogues = []
    seen = set()
    for dlg in (args.dialogues or []):
        path = Path(dlg)
        if not path.is_file():
            print(f"Skipping missing --dialogues file: {path}")
            continue
        for row in read_rows(path):
            key = hashlib.sha256(json.dumps(row.get("messages"), ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            dialogues.append(row)
    starter = list(read_rows(Path(args.starter)))
    if not dialogues or not starter:
        parser.error("Both the dialogue and starter files must contain valid conversations")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as target:
        for row in dialogues:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
        for _ in range(args.starter_repeats):
            for row in starter:
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(dialogues):,} dialogue rows plus {len(starter) * args.starter_repeats:,} starter rows to {output}")


if __name__ == "__main__":
    main()
