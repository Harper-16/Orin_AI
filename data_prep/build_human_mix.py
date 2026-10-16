#!/usr/bin/env python3
"""Build a warm, human-style chat mix from the project's conversation sets.

Human != more UltraChat. It means:
  * short multi-turn everyday talk FIRST (DailyDialog mix, conversation_core)
  * SmolTalk everyday second (natural small-talk, not assistant-y lectures)
  * UltraChat LAST and down-weighted (it teaches long formal answers that
    make a 75M sound robotic and ramble into salad)
  * identity/name/memory rows repeated so 'who are you / my name is' sticks
  * NO grounded_qa refusal rows (they teach 'The passage does not...')

Writes data/human_chat_mix.jsonl by default.
"""
import argparse
import hashlib
import json
from pathlib import Path


def read_rows(path: Path):
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


def assistant_text(row) -> str:
    return " ".join(m.get("content", "") for m in row.get("messages", [])
                    if m.get("role") == "assistant")


def is_human_shaped(row, max_assistant_chars: int) -> bool:
    text = assistant_text(row)
    if not text or len(text) > max_assistant_chars:
        return False
    # Drop code/math dumps and bullet-lecture answers: they make small models ramble.
    low = text.casefold()
    if any(marker in low for marker in ("```", "def ", "import ", "theorem", "q.e.d")):
        return False
    if text.count("\n- ") >= 3 or text.count("\n* ") >= 3:
        return False
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="data/human_chat_mix.jsonl")
    p.add_argument("--ultrachat", default="data/ultrachat_10000.jsonl")
    p.add_argument("--ultrachat-keep", type=float, default=0.25,
                   help="Fraction of UltraChat rows to keep (default 0.25).")
    p.add_argument("--max-assistant-chars", type=int, default=600,
                   help="Drop assistant replies longer than this (default 600).")
    p.add_argument("--identity-repeats", type=int, default=25)
    p.add_argument("--seed", type=int, default=4242)
    args = p.parse_args()

    import random
    rng = random.Random(args.seed)
    base = Path(__file__).resolve().parents[1]

    ordered_sources = [
        (base / "data/conversation_mix.jsonl", 1.0),      # DailyDialog multi-turn first
        (base / "data/smoltalk_everyday.jsonl", 1.0),     # natural small-talk second
        (base / "data/chat_sft_greet.jsonl", 1.0),
        (base / "data/chat_polish.jsonl", 1.0),
        (base / args.ultrachat, args.ultrachat_keep),     # formal UltraChat last, cut down
    ]

    seen, out = set(), []
    for path, keep in ordered_sources:
        if not path.is_file():
            print(f"skip missing: {path}")
            continue
        rows = read_rows(path)
        rng.shuffle(rows)
        kept = skipped = 0
        for row in rows:
            if rng.random() > keep:
                continue
            if not is_human_shaped(row, args.max_assistant_chars):
                skipped += 1
                continue
            key = hashlib.sha256(json.dumps(row["messages"], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
            kept += 1
        print(f"{path.name}: kept {kept:,}, dropped-long/dup {skipped:,}")

    # Identity/memory always survive: repeat them.
    for name in ("examples/conversation_core.jsonl", "examples/basic_qa.jsonl"):
        path = base / name
        if not path.is_file():
            continue
        core = read_rows(path)
        out.extend(core * (args.identity_repeats if "core" in name else 3))
        print(f"{name}: repeated x{args.identity_repeats if 'core' in name else 3}")

    # Never teach refusal in a human-chat mix.
    out = [r for r in out if "does not give enough information" not in assistant_text(r)]
    rng.shuffle(out)

    dest = base / args.output if not Path(args.output).is_absolute() else Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"WROTE {len(out):,} rows -> {dest}")


if __name__ == "__main__":
    main()
