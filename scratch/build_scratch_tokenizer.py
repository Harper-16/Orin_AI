#!/usr/bin/env python3
"""Train a fresh byte-level BPE tokenizer from the downloaded text and chat data."""

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


SPECIAL_TOKENS = [
    "<|pad|>",
    "<|unk|>",
    "<|bos|>",
    "<|eos|>",
    "<|end|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="data/pretrain/train.txt")
    parser.add_argument("--chat-data", default="data/ultrachat.jsonl")
    parser.add_argument("--output", default="data/scratch_tokenizer.json")
    parser.add_argument("--vocab-size", type=int, default=8192)
    return parser.parse_args()


def training_lines(text_path: Path, chat_path: Path):
    with text_path.open("r", encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if line:
                yield line
    if chat_path.exists():
        with chat_path.open("r", encoding="utf-8") as source:
            for line in source:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for message in row.get("messages", []):
                    content = message.get("content")
                    if isinstance(content, str) and content.strip():
                        yield content.strip()


def main():
    args = parse_args()
    text_path = Path(args.text)
    if not text_path.is_file():
        raise SystemExit(f"Pretraining text not found: {text_path}. Run data_prep/download_pretrain_data.py first.")
    if args.vocab_size <= len(SPECIAL_TOKENS) + 256:
        raise SystemExit("Choose a vocabulary size greater than the byte alphabet and special tokens.")

    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(
        training_lines(text_path, Path(args.chat_data)),
        trainer=trainer,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output))
    print(f"Saved {tokenizer.get_vocab_size():,}-token tokenizer to {output}")


if __name__ == "__main__":
    main()
