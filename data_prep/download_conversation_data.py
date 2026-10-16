#!/usr/bin/env python3
"""Prepare short, multi-turn DailyDialog conversations for scratch-model tuning."""

import argparse
import hashlib
import json
import re
from pathlib import Path


DATASET = "elricwan/dailydialog"
DATASET_URL = "https://huggingface.co/datasets/elricwan/dailydialog"
ORIGINAL_URL = "https://huggingface.co/datasets/roskoN/dailydialog"
LICENSE = "CC-BY-NC-SA-4.0 (original DailyDialog distribution)"
SYSTEM_PROMPT = "You are a helpful assistant. Be clear and honest when uncertain. You have no live internet access."
SPEAKER_PREFIX = re.compile(r"^Person\s+[AB]\s*:\s*", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/conversation_focused.jsonl")
    parser.add_argument("--max-dialogues", type=int, default=13000)
    parser.add_argument("--max-total-chars", type=int, default=5_000_000)
    parser.add_argument("--max-utterance-chars", type=int, default=600)
    parser.add_argument("--min-turns", type=int, default=4)
    parser.add_argument("--shuffle-buffer", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.max_dialogues, args.max_total_chars, args.max_utterance_chars, args.min_turns) < 1:
        raise SystemExit("All size limits must be positive.")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install dependencies first: pip install -r requirements.txt") from exc

    rows = load_dataset(DATASET, split="train", streaming=True)
    rows = rows.shuffle(seed=args.seed, buffer_size=max(1, args.shuffle_buffer))
    iterator = iter(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    kept = skipped = total_chars = 0
    seen = set()

    with output.open("w", encoding="utf-8") as target:
        for row in iterator:
            raw_dialogue = row.get("conversation")
            if not isinstance(raw_dialogue, list):
                skipped += 1
                continue
            utterances = []
            for raw in raw_dialogue:
                if not isinstance(raw, str):
                    continue
                utterance = SPEAKER_PREFIX.sub("", raw)
                utterance = re.sub(r"\s+", " ", utterance).strip()
                if not utterance or len(utterance) > args.max_utterance_chars:
                    utterances = []
                    break
                utterances.append(utterance)

            # Conversations start with Person A. Keep complete user/assistant pairs.
            if len(utterances) % 2:
                utterances.pop()
            if len(utterances) < args.min_turns:
                skipped += 1
                continue

            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
                {"role": "user" if i % 2 == 0 else "assistant", "content": utterance}
                for i, utterance in enumerate(utterances)
            ]
            serialized_messages = json.dumps(messages, ensure_ascii=False)
            digest = hashlib.sha256(serialized_messages.encode("utf-8")).digest()
            if digest in seen:
                skipped += 1
                continue
            chars = sum(len(message["content"]) for message in messages)
            if total_chars + chars > args.max_total_chars:
                break

            seen.add(digest)
            target.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            kept += 1
            total_chars += chars
            if kept >= args.max_dialogues:
                break

    close_iterator = getattr(iterator, "close", None)
    if callable(close_iterator):
        close_iterator()
    del iterator, rows

    if kept == 0:
        raise SystemExit("No usable multi-turn dialogue rows were found.")

    manifest = {
        "dataset": DATASET,
        "dataset_url": DATASET_URL,
        "original_dataset_url": ORIGINAL_URL,
        "license": LICENSE,
        "split": "train",
        "system_prompt": SYSTEM_PROMPT,
        "limits": {
            "max_dialogues": args.max_dialogues,
            "max_total_chars": args.max_total_chars,
            "max_utterance_chars": args.max_utterance_chars,
            "min_utterances": args.min_turns,
        },
        "dialogues": kept,
        "characters": total_chars,
        "skipped_or_duplicate_rows": skipped,
        "seed": args.seed,
    }
    manifest_path = output.with_suffix(".source.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {kept:,} multi-turn conversations ({total_chars:,} characters).")
    print(f"Skipped rows: {skipped:,}")
    print(f"Output: {output}")
    print(f"Source manifest: {manifest_path}")


if __name__ == "__main__":
    main()
