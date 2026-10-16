#!/usr/bin/env python3
"""Teach the hybrid model to chat on its own — no preset replies.

After pretraining, the hybrid only knows how to continue raw text; it has not
learned the chat format or turns like "hi" -> a greeting. This script does a
full-weight supervised fine-tune on conversation data so the model itself
learns greetings, thanks, goodbyes, small talk and simple Q&A. Once trained,
hybrid/chat_hybrid.py produces those replies with the model alone (run it WITHOUT
--conversation-helpers).

The example/label building and tokenizer plumbing are shared with
scratch/finetune_from_scratch.py, so the two models are trained identically.
"""

import argparse
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

from hybrid.hybrid_model import HybridConfig, create_hybrid
from scratch.finetune_from_scratch import load_examples, make_batch, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", action="append", required=True,
        help="Conversation JSONL file(s) to learn from. Repeat for several files.",
    )
    parser.add_argument("--tokenizer", default="data/scratch_tokenizer.json")
    parser.add_argument("--init-from", required=True, help="Pretrained hybrid checkpoint to start from.")
    parser.add_argument("--output", default="checkpoints/hybrid/hybrid-chat")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--history-turns", type=int, default=2, help="Recent user/assistant exchanges kept per example.")
    parser.add_argument("--n-loops", type=int, help="Recurrence depth used while fine-tuning. Defaults to the checkpoint maximum.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Accumulate gradients over this many batches before stepping. "
                             "Effective batch = batch-size x grad-accum; smooths batch-1 noise without extra RAM.")
    parser.add_argument("--max-examples", type=int, help="Cap the number of assistant-turn examples used (keeps big files fast).")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the combined examples before capping.")
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--dropout", type=float, help="Dropout during fine-tuning; defaults to the checkpoint setting.")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--resume", help="Resume tuning from a previous --output/latest.pt (keeps optimizer + step).")
    parser.add_argument("--seed", type=int, default=4242)
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.steps, args.batch_size) < 1 or not 6 <= args.seq_len <= 512:
        raise SystemExit("Steps and batch size must be positive; sequence length must be 6-512 tokens.")
    torch.set_num_threads(max(1, min(args.threads, os.cpu_count() or 1)))
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    pad_id = tokenizer.token_to_id("<|pad|>")

    checkpoint = torch.load(args.init_from, map_location="cpu", weights_only=True)
    config = HybridConfig.from_dict(checkpoint["config"])
    if args.dropout is not None:
        if not 0.0 <= args.dropout < 1.0:
            raise SystemExit("--dropout must be in the range [0, 1).")
        config.dropout = args.dropout
    if args.seq_len > config.max_seq_len:
        raise SystemExit(f"Sequence length exceeds model context ({config.max_seq_len}).")
    n_loops = config.max_loop_iters if not args.n_loops else max(1, min(args.n_loops, config.max_loop_iters))

    examples: list = []
    for data_path in args.data:
        examples.extend(load_examples(Path(data_path), tokenizer, args.seq_len, args.history_turns))
    if args.shuffle:
        random.shuffle(examples)
    if args.max_examples is not None and args.max_examples < len(examples):
        print(f"Capping {len(examples):,} examples to {args.max_examples:,}.")
        examples = examples[: args.max_examples]
    if not examples:
        raise SystemExit("No usable assistant replies were found in the conversation data.")

    model = create_hybrid(config).to(device)
    model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    start_step = 0
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise SystemExit(f"Resume checkpoint not found: {resume_path}")
        resumed = torch.load(resume_path, map_location="cpu", weights_only=True)
        model.load_state_dict(resumed["model"])
        try:
            optimizer.load_state_dict(resumed["optimizer"])
        except (KeyError, ValueError, TypeError):
            print("Could not restore optimizer state; continuing with a fresh optimizer.")
        start_step = int(resumed.get("step", 0))
        print(f"Resuming tuning at step {start_step:,} from {resume_path}.")
    output = Path(args.output)
    print(f"Loaded {model.parameter_count():,} hybrid parameters from {args.init_from}.")
    print(f"Assistant-turn examples: {len(examples):,}; history window: {args.history_turns} turns; loops: {n_loops}; dropout: {config.dropout:g}; device: {device}; steps: {args.steps:,}")
    print("Training the model to greet and chat on its own (no preset helpers).")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step in range(start_step + 1, args.steps + 1):
        total_loss = 0.0
        for _ in range(max(1, args.grad_accum)):
            x, y = make_batch(examples, args.batch_size, args.seq_len, pad_id, device)
            loss = model(x, y, n_loops=n_loops)["loss"] / max(1, args.grad_accum)
            loss.backward()
            total_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % 20 == 0:
            print(f"step {step:>5}/{args.steps} | chat loss {total_loss:.4f}")
        if step % args.save_interval == 0 or step == args.steps:
            save_checkpoint(output / "latest.pt", model, optimizer, config, step)

    final_path = output / "final.pt"
    save_checkpoint(final_path, model, optimizer, config, args.steps)
    print(f"Saved chat-trained hybrid to {final_path.resolve()}")
    print(f"Chat with it (no presets): python hybrid/chat_hybrid.py --model {final_path}")


if __name__ == "__main__":
    main()

