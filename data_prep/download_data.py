#!/usr/bin/env python3
"""Download a bounded, cleaned sample of public chat data for SFT."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_DATASET = "HuggingFaceH4/ultrachat_200k"
DEFAULT_SPLIT = "train_sft"
ALLOWED_ROLES = {"system", "user", "assistant"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save a size-limited conversational dataset as JSONL."
    )
    parser.add_argument("--output", default="data/ultrachat.jsonl")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument(
        "--input",
        help="Use an existing JSONL file instead of downloading from Hugging Face.",
    )
    parser.add_argument("--max-examples", type=int, default=2000)
    parser.add_argument("--max-total-chars", type=int, default=20_000_000)
    parser.add_argument("--max-message-chars", type=int, default=8000)
    parser.add_argument("--shuffle-buffer", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def normalize_messages(value: Any, max_message_chars: int) -> list[dict[str, str]] | None:
    if not isinstance(value, list):
        return None

    messages: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue

        role = str(item.get("role", "")).strip().lower()
        content = item.get("content", item.get("value", ""))
        if not isinstance(content, str):
            continue
        content = content.strip()

        if role not in ALLOWED_ROLES or not content or len(content) > max_message_chars:
            continue
        messages.append({"role": role, "content": content})

    if len(messages) < 2:
        return None
    if not any(m["role"] == "user" for m in messages):
        return None
    if not any(m["role"] == "assistant" for m in messages):
        return None
    return messages


def rows_from_file(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"[skip] malformed JSON at {path}:{line_number}")


def rows_from_hub(dataset_name: str, split: str, shuffle_buffer: int, seed: int):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install dependencies first: pip install -r requirements.txt") from exc

    rows = load_dataset(dataset_name, split=split, streaming=True)
    if shuffle_buffer > 1:
        rows = rows.shuffle(seed=seed, buffer_size=shuffle_buffer)
    return rows


def main() -> None:
    args = parse_args()
    if args.max_examples < 1 or args.max_total_chars < 1 or args.max_message_chars < 1:
        raise SystemExit("All size limits must be positive.")

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.input:
        source_path = Path(args.input).expanduser()
        rows = rows_from_file(source_path)
        source_name = str(source_path)
    else:
        rows = rows_from_hub(args.dataset, args.split, args.shuffle_buffer, args.seed)
        source_name = f"{args.dataset}:{args.split}"

    seen: set[str] = set()
    kept = skipped = total_chars = 0

    with output.open("w", encoding="utf-8") as out:
        for row in rows:
            if kept >= args.max_examples or total_chars >= args.max_total_chars:
                break

            if not isinstance(row, dict):
                skipped += 1
                continue

            messages = normalize_messages(row.get("messages"), args.max_message_chars)
            if messages is None:
                skipped += 1
                continue

            serialized = json.dumps(messages, ensure_ascii=False, sort_keys=True)
            key = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
            if key in seen:
                skipped += 1
                continue

            example_chars = sum(len(message["content"]) for message in messages)
            if total_chars + example_chars > args.max_total_chars:
                break

            seen.add(key)
            out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            kept += 1
            total_chars += example_chars

    print(f"Source: {source_name}")
    print(f"Saved examples: {kept:,}")
    print(f"Skipped rows: {skipped:,}")
    print(f"Text characters: {total_chars:,} (limit {args.max_total_chars:,})")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
