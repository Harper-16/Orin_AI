#!/usr/bin/env python3
"""Teach the from-scratch Transformer to respond in chat format."""

import argparse
import json
import os
import random
from pathlib import Path

import torch
from tokenizers import Tokenizer

# Allow running this file directly from anywhere inside the project.
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scratch.scratch_model import ModelConfig, create_model


ROLE_TOKENS = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="JSONL conversation data to learn from.")
    parser.add_argument("--tokenizer", default="data/scratch_tokenizer.json")
    parser.add_argument("--pretrained", required=True, help="Starting scratch checkpoint; it is left unchanged.")
    parser.add_argument("--output", default="checkpoints/scratch/scratch-finetune-experiment")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--history-turns", type=int, default=2, help="Keep this many recent user/assistant exchanges in each training prompt.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--dropout", type=float, help="Dropout probability during fine-tuning; defaults to the checkpoint setting (0.1 for older checkpoints).")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=4242)
    return parser.parse_args()


def load_examples(path: Path, tokenizer: Tokenizer, seq_len: int, history_turns: int = 2):
    if not path.is_file():
        raise SystemExit(f"Chat data not found: {path}.")
    token_ids = {name: tokenizer.token_to_id(token) for name, token in ROLE_TOKENS.items()}
    token_ids["end"] = tokenizer.token_to_id("<|end|>")
    token_ids["bos"] = tokenizer.token_to_id("<|bos|>")
    if any(value is None for value in token_ids.values()):
        raise SystemExit("Tokenizer is missing chat special tokens; rebuild it with scratch/build_scratch_tokenizer.py.")
    if history_turns < 1:
        raise SystemExit("--history-turns must be at least 1.")

    examples = []
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            messages = row.get("messages")
            if not isinstance(messages, list):
                continue

            # Each assistant answer is its own example. Retain whole message
            # boundaries and the same recent-turn window used by chat inference.
            history = []
            for message in messages:
                role = message.get("role")
                content = message.get("content")
                if role not in ROLE_TOKENS or not isinstance(content, str):
                    continue
                block = [token_ids[role], *tokenizer.encode(content).ids, token_ids["end"]]
                if role == "assistant" and len(block) > 2:
                    systems = [item for item in history if item[0] == token_ids["system"]]
                    turns = [item for item in history if item[0] != token_ids["system"]]
                    turns = turns[-(history_turns * 2):]
                    response = block[1:]
                    # Leave space for BOS, a minimal user message, and the
                    # assistant marker even when an answer is very long.
                    response = response[:max(2, seq_len - 4)]
                    response[-1] = token_ids["end"]

                    def make_prefix():
                        ids = [token_ids["bos"]]
                        for item in systems + turns:
                            ids.extend(item)
                        ids.append(token_ids["assistant"])
                        return ids

                    prefix = make_prefix()
                    while len(prefix) + len(response) > seq_len and len(turns) > 1:
                        turns.pop(0)
                        prefix = make_prefix()
                    if len(prefix) + len(response) > seq_len and systems:
                        systems = []
                        prefix = make_prefix()
                    if len(prefix) + len(response) > seq_len and turns:
                        latest = turns[-1]
                        max_block_tokens = max(2, seq_len - len(response) - len(prefix) + len(latest))
                        turns[-1] = latest[:max_block_tokens - 1] + [token_ids["end"]]
                        prefix = make_prefix()
                    if len(prefix) + len(response) > seq_len:
                        response = response[:max(2, seq_len - len(prefix))]
                        response[-1] = token_ids["end"]
                    if len(prefix) + len(response) <= seq_len and len(response) >= 2:
                        ids = prefix + response
                        labels = [-100] * len(prefix) + response
                        examples.append((ids, labels))
                history.append(block)

    if not examples:
        raise SystemExit("No usable assistant replies were found in the conversation data.")
    return examples

def make_batch(examples, batch_size, seq_len, pad_id, device):
    chosen = random.choices(examples, k=batch_size)
    inputs, labels = [], []
    for ids, targets in chosen:
        padding = seq_len - len(ids)
        inputs.append(ids + [pad_id] * padding)
        labels.append(targets + [-100] * padding)
    return torch.tensor(inputs, dtype=torch.long, device=device), torch.tensor(
        labels, dtype=torch.long, device=device
    )


def save_checkpoint(path, model, optimizer, config, step):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
         "config": config.to_dict(), "step": step},
        temporary,
    )
    temporary.replace(path)


def main():
    args = parse_args()
    if min(args.steps, args.batch_size) < 1 or not 6 <= args.seq_len <= 512:
        raise SystemExit("Steps and batch size must be positive; sequence length must be 6–512 tokens.")
    torch.set_num_threads(max(1, min(args.threads, os.cpu_count() or 1)))
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    examples = load_examples(Path(args.data), tokenizer, args.seq_len, args.history_turns)
    pad_id = tokenizer.token_to_id("<|pad|>")
    checkpoint = torch.load(args.pretrained, map_location="cpu", weights_only=True)
    config = ModelConfig.from_dict(checkpoint["config"])
    if args.dropout is not None:
        if not 0.0 <= args.dropout < 1.0:
            raise SystemExit("--dropout must be in the range [0, 1).")
        config.dropout = args.dropout
    if args.seq_len > config.max_seq_len:
        raise SystemExit(f"Sequence length exceeds model context ({config.max_seq_len}).")
    model = create_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    output = Path(args.output)
    print(f"Loaded {model.parameter_count():,} pretrained-from-scratch parameters.")
    print(f"Assistant-turn examples: {len(examples):,}; history window: {args.history_turns} turns; dropout: {config.dropout:g}; device: {device}; steps: {args.steps:,}")

    model.train()
    for step in range(1, args.steps + 1):
        x, y = make_batch(examples, args.batch_size, args.seq_len, pad_id, device)
        loss = model(x, y)["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 20 == 0:
            print(f"step {step:>5}/{args.steps} | chat loss {loss.item():.4f}")
        if step % args.save_interval == 0 or step == args.steps:
            save_checkpoint(output / "latest.pt", model, optimizer, config, step)

    final_path = output / "final.pt"
    save_checkpoint(final_path, model, optimizer, config, args.steps)
    print(f"Saved chat-trained model to {final_path.resolve()}")


if __name__ == "__main__":
    main()
