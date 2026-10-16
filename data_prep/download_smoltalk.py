#!/usr/bin/env python3
"""Download a SmolTalk config (default: everyday-conversations) as chat JSONL.

SmolTalk is multi-config (all, everyday-conversations, smol-magpie-ultra, ...),
so data_prep/download_data.py cannot load it (it has no --config option).
This script fills that gap and writes the same {"messages": [...]} JSONL
format the fine-tuners expect.
"""
import argparse
import hashlib
import json
from pathlib import Path

DATASET = "HuggingFaceTB/smoltalk"
ALLOWED_ROLES = {"system", "user", "assistant"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="everyday-conversations")
    p.add_argument("--split", default="train")
    p.add_argument("--output", default="data/smoltalk_everyday.jsonl")
    p.add_argument("--max-examples", type=int, default=20000)
    p.add_argument("--max-total-chars", type=int, default=20_000_000)
    p.add_argument("--max-message-chars", type=int, default=4000)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def normalize(messages, max_message_chars):
    if not isinstance(messages, list):
        return None
    out = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "")).strip().lower()
        content = m.get("content", "")
        if not isinstance(content, str):
            continue
        content = content.strip()
        if role not in ALLOWED_ROLES or not content or len(content) > max_message_chars:
            continue
        out.append({"role": role, "content": content})
    if len(out) < 2:
        return None
    if not any(m["role"] == "user" for m in out):
        return None
    if not any(m["role"] == "assistant" for m in out):
        return None
    return out


def main():
    args = parse_args()
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Install deps first: pip install -r requirements.txt")
    rows = load_dataset(DATASET, args.config, split=args.split, streaming=True)
    rows = rows.shuffle(seed=args.seed, buffer_size=1024)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    seen, kept, skipped, total_chars = set(), 0, 0, 0
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            if kept >= args.max_examples or total_chars >= args.max_total_chars:
                break
            if not isinstance(row, dict):
                skipped += 1
                continue
            messages = normalize(row.get("messages"), args.max_message_chars)
            if messages is None:
                skipped += 1
                continue
            key = hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            if key in seen:
                skipped += 1
                continue
            chars = sum(len(m["content"]) for m in messages)
            if total_chars + chars > args.max_total_chars:
                break
            seen.add(key)
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            kept += 1
            total_chars += chars
    print(f"Source: {DATASET}:{args.config}/{args.split}")
    print(f"Saved examples: {kept:,}  skipped: {skipped:,}  chars: {total_chars:,}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
