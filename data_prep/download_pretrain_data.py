#!/usr/bin/env python3
"""Stream a small, deduplicated FineWeb-Edu sample for from-scratch pretraining."""

import argparse
import gc
import hashlib
import json
import random
import re
from pathlib import Path


DATASET = "HuggingFaceFW/fineweb-edu"
CONFIG = "sample-10BT"
LICENSE = "ODC-By-1.0"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/pretrain")
    parser.add_argument("--max-documents", type=int, default=5000)
    parser.add_argument("--max-total-chars", type=int, default=10_000_000)
    parser.add_argument("--max-document-chars", type=int, default=20_000)
    parser.add_argument("--min-document-chars", type=int, default=200)
    parser.add_argument("--shuffle-buffer", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.max_documents, args.max_total_chars, args.max_document_chars) < 1:
        raise SystemExit("Size limits must be positive.")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install dependencies first: pip install -r requirements.txt") from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_dataset(DATASET, CONFIG, split="train", streaming=True)
    rows = rows.shuffle(seed=args.seed, buffer_size=max(1, args.shuffle_buffer))
    iterator = iter(rows)
    documents = []
    seen = set()
    total_chars = skipped = 0
    for row in iterator:
        raw = row.get("text", "")
        if not isinstance(raw, str):
            skipped += 1
            continue
        text = re.sub(r"\s+", " ", raw).strip()
        if len(text) > args.max_document_chars:
            text = text[:args.max_document_chars].rsplit(" ", 1)[0]
        if len(text) < args.min_document_chars:
            skipped += 1
            continue
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        if digest in seen:
            skipped += 1
            continue
        if total_chars + len(text) > args.max_total_chars:
            break
        seen.add(digest)
        documents.append(text)
        total_chars += len(text)
        if len(documents) >= args.max_documents:
            break

    close_iterator = getattr(iterator, "close", None)
    if callable(close_iterator):
        close_iterator()
    del iterator, rows
    gc.collect()

    if len(documents) < 20:
        raise SystemExit(f"Only collected {len(documents)} documents; check the internet connection.")

    random.Random(args.seed).shuffle(documents)
    split_at = max(1, int(len(documents) * 0.95))
    paths = {
        "train": output_dir / "train.txt",
        "validation": output_dir / "validation.txt",
    }
    counts = {}
    for name, selected in (("train", documents[:split_at]), ("validation", documents[split_at:])):
        paths[name].write_text("\n".join(selected) + "\n", encoding="utf-8")
        counts[name] = len(selected)

    manifest = {
        "dataset": DATASET,
        "configuration": CONFIG,
        "split": "train (streamed)",
        "license": LICENSE,
        "dataset_card": f"https://huggingface.co/datasets/{DATASET}",
        "limits": {
            "max_documents": args.max_documents,
            "max_total_chars": args.max_total_chars,
            "max_document_chars": args.max_document_chars,
        },
        "documents": counts,
        "characters": total_chars,
        "skipped_or_duplicate_rows": skipped,
        "seed": args.seed,
    }
    (output_dir / "source.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(documents):,} documents ({total_chars:,} chars).")
    print(f"Train: {counts['train']:,}; validation: {counts['validation']:,}")
    print(f"Source manifest: {output_dir / 'source.json'}")


if __name__ == "__main__":
    main()
