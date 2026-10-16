#!/usr/bin/env python3
"""Pretrain the custom Transformer from random weights on local text."""

import argparse
import math
import os
import random
from array import array
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

# Allow running this file directly from anywhere inside the project.
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scratch.scratch_model import ModelConfig, create_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-text", default="data/pretrain/train.txt")
    parser.add_argument("--validation-text", default="data/pretrain/validation.txt")
    parser.add_argument("--tokenizer", default="data/scratch_tokenizer.json")
    parser.add_argument("--output", default="checkpoints/scratch/scratch-17m")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--resume", help="Checkpoint file to continue.")
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def tokenize_text(text_path: Path, token_path: Path, tokenizer: Tokenizer) -> int:
    eos_id = tokenizer.token_to_id("<|eos|>")
    if eos_id is None:
        raise SystemExit("Tokenizer must contain <|eos|>.")
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_count = 0
    buffer = array("H")
    with text_path.open("r", encoding="utf-8") as source, token_path.open("wb") as output:
        for line in source:
            text = line.strip()
            if not text:
                continue
            encoded = tokenizer.encode(text).ids
            buffer.extend(encoded)
            buffer.append(eos_id)
            token_count += len(encoded) + 1
            if len(buffer) >= 250_000:
                buffer.tofile(output)
                buffer = array("H")
        if buffer:
            buffer.tofile(output)
    return token_count


def load_tokens(text_path: Path, cache_path: Path, tokenizer: Tokenizer) -> np.memmap:
    if not text_path.is_file():
        raise SystemExit(f"Missing text corpus: {text_path}. Run data_prep/download_pretrain_data.py first.")
    if not cache_path.is_file() or cache_path.stat().st_mtime < text_path.stat().st_mtime:
        count = tokenize_text(text_path, cache_path, tokenizer)
        print(f"Tokenized {text_path.name}: {count:,} tokens.")
    tokens = np.memmap(cache_path, dtype=np.uint16, mode="r")
    return tokens


def get_batch(tokens: np.memmap, batch_size: int, seq_len: int, device: torch.device):
    starts = np.random.randint(0, len(tokens) - seq_len + 1, size=batch_size)
    samples = np.stack(
        [np.asarray(tokens[start : start + seq_len], dtype=np.int64) for start in starts]
    )
    batch = torch.from_numpy(samples.copy()).to(device)
    return batch, batch


def save_checkpoint(path: Path, model, optimizer, config, step: int, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config.to_dict(),
            "step": step,
            "seed": args.seed,
        },
        temporary,
    )
    temporary.replace(path)


def main():
    args = parse_args()
    if min(args.steps, args.seq_len, args.batch_size, args.grad_accum) < 1:
        raise SystemExit("Steps, sequence length, batch size, and accumulation must be positive.")
    if args.seq_len > 512:
        raise SystemExit("This model supports up to 512 training tokens per sequence.")
    torch.set_num_threads(max(1, min(args.threads, os.cpu_count() or 1)))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.is_file():
        raise SystemExit(f"Missing tokenizer: {tokenizer_path}. Run scratch/build_scratch_tokenizer.py first.")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    output = Path(args.output)
    train_tokens = load_tokens(
        Path(args.train_text), output / "train_tokens.u16", tokenizer
    )
    validation_tokens = load_tokens(
        Path(args.validation_text), output / "validation_tokens.u16", tokenizer
    )
    for name, tokens in (("training", train_tokens), ("validation", validation_tokens)):
        if len(tokens) < args.seq_len:
            raise SystemExit(f"The {name} corpus is too small for sequence length {args.seq_len}.")

    config = ModelConfig(vocab_size=tokenizer.get_vocab_size())
    model = create_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        config = ModelConfig.from_dict(checkpoint["config"])
        model = create_model(config).to(device)
        model.load_state_dict(checkpoint["model"])
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])

    initialization = "resumed from an existing scratch checkpoint" if args.resume else "randomly initialized"
    print(f"Architecture: {model.parameter_count():,} parameters, {initialization}.")
    print(f"Device: {device}; train tokens: {len(train_tokens):,}; validation tokens: {len(validation_tokens):,}")
    print(f"Steps: {args.steps:,}; sequence length: {args.seq_len}; effective batch: {args.batch_size * args.grad_accum}")

    def evaluate():
        model.eval()
        losses = []
        with torch.no_grad():
            for _ in range(12):
                x, y = get_batch(validation_tokens, args.batch_size, args.seq_len, device)
                losses.append(model(x, y)["loss"].item())
        model.train()
        return sum(losses) / len(losses)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step in range(start_step + 1, args.steps + 1):
        progress = (step - start_step) / max(1, args.steps - start_step)
        warmup = min(1.0, step / max(1, min(100, args.steps // 20)))
        lr = args.learning_rate * warmup * 0.5 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = max(lr, args.learning_rate * 0.02)
        running_loss = 0.0
        for _ in range(args.grad_accum):
            x, y = get_batch(train_tokens, args.batch_size, args.seq_len, device)
            loss = model(x, y)["loss"] / args.grad_accum
            loss.backward()
            running_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step == 1 or step % 20 == 0:
            print(f"step {step:>5}/{args.steps} | train loss {running_loss:.4f} | lr {lr:.2e}")
        if step % args.eval_interval == 0 or step == args.steps:
            validation_loss = evaluate()
            print(f"  validation loss: {validation_loss:.4f}")
        if step % args.save_interval == 0 or step == args.steps:
            save_checkpoint(output / "latest.pt", model, optimizer, config, step, args)

    final_path = output / "final.pt"
    save_checkpoint(final_path, model, optimizer, config, args.steps, args)
    print(f"Saved scratch-trained model to {final_path.resolve()}")


if __name__ == "__main__":
    main()
