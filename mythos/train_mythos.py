#!/usr/bin/env python3
"""Pretrain the Mythos recurrent-depth Transformer from random weights.

Mirrors scratch/train_from_scratch.py, but builds a Mythos model (Prelude + looped
Recurrent Block + Coda) and ramps the number of loop iterations over training
so the shared block learns to work at every depth. It reuses the tokenizer,
token cache, batching and checkpoint helpers of the original trainer.
"""

import argparse
import math
import os
import random
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

from mythos.mythos_model import MythosConfig, create_mythos
from scratch.train_from_scratch import load_tokens, get_batch, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-text", default="data/pretrain/train.txt")
    parser.add_argument("--validation-text", default="data/pretrain/validation.txt")
    parser.add_argument("--tokenizer", default="data/scratch_tokenizer.json")
    parser.add_argument("--output", default="checkpoints/mythos/mythos-17m")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--prelude-layers", type=int, default=3)
    parser.add_argument("--coda-layers", type=int, default=3)
    parser.add_argument("--max-loop-iters", type=int, default=6)
    parser.add_argument("--depth-lora-rank", type=int, default=8)
    parser.add_argument("--ponder-weight", type=float, default=0.01)
    parser.add_argument("--no-halting", action="store_true", help="Disable adaptive halting and the ponder cost.")
    parser.add_argument("--no-loop-curriculum", action="store_true", help="Always use the full loop count.")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--resume", help="Checkpoint file to continue.")
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def build_config(args) -> MythosConfig:
    return MythosConfig(
        prelude_layers=args.prelude_layers,
        coda_layers=args.coda_layers,
        max_loop_iters=args.max_loop_iters,
        depth_lora_rank=args.depth_lora_rank,
        use_halting=not args.no_halting,
        ponder_weight=args.ponder_weight,
    )


def loops_for_step(step: int, total: int, max_iters: int, curriculum: bool) -> int:
    """Ramp the loop count from 1 up to max_iters across training."""
    if not curriculum or max_iters <= 1:
        return max_iters
    progress = step / max(1, total)
    return max(1, min(max_iters, 1 + round((max_iters - 1) * progress)))


def main():
    args = parse_args()
    if min(args.steps, args.seq_len, args.batch_size, args.grad_accum) < 1:
        raise SystemExit("Steps, sequence length, batch size, and accumulation must be positive.")
    if args.seq_len > 512:
        raise SystemExit("This model supports up to 512 training tokens per sequence.")
    if args.max_loop_iters < 1:
        raise SystemExit("--max-loop-iters must be at least 1.")
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
    output.mkdir(parents=True, exist_ok=True)
    train_tokens = load_tokens(Path(args.train_text), output / "train_tokens.u16", tokenizer)
    validation_tokens = load_tokens(
        Path(args.validation_text), output / "validation_tokens.u16", tokenizer
    )
    for name, tokens in (("training", train_tokens), ("validation", validation_tokens)):
        if len(tokens) < args.seq_len:
            raise SystemExit(f"The {name} corpus is too small for sequence length {args.seq_len}.")

    config = build_config(args)
    config.vocab_size = tokenizer.get_vocab_size()
    model = create_mythos(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        config = MythosConfig.from_dict(checkpoint["config"])
        model = create_mythos(config).to(device)
        model.load_state_dict(checkpoint["model"])
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])

    initialization = "resumed from an existing Mythos checkpoint" if args.resume else "randomly initialized"
    print(f"Architecture: Mythos RDT, {model.parameter_count():,} parameters, {initialization}.")
    print(f"Prelude {config.prelude_layers} | Recurrent 1 (looped <= {config.max_loop_iters}) | Coda {config.coda_layers}")
    print(f"Depth-LoRA rank: {config.depth_lora_rank}; halting: {config.use_halting}; spectral radius A: {model.spectral_radius():.4f}")
    print(f"Device: {device}; train tokens: {len(train_tokens):,}; validation tokens: {len(validation_tokens):,}")
    print(f"Steps: {args.steps:,}; sequence length: {args.seq_len}; effective batch: {args.batch_size * args.grad_accum}")

    def evaluate():
        model.eval()
        losses = []
        with torch.no_grad():
            for _ in range(12):
                x, y = get_batch(validation_tokens, args.batch_size, args.seq_len, device)
                losses.append(model(x, y, n_loops=config.max_loop_iters)["loss"].item())
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
        n_loops = loops_for_step(step, args.steps, config.max_loop_iters, not args.no_loop_curriculum)
        running_lm = 0.0
        running_ponder = 0.0
        for _ in range(args.grad_accum):
            x, y = get_batch(train_tokens, args.batch_size, args.seq_len, device)
            result = model(x, y, n_loops=n_loops)
            loss = result["loss"] / args.grad_accum
            loss.backward()
            running_lm += (result["loss"] - result["ponder"]).item() / args.grad_accum
            running_ponder += result["ponder"].item() / args.grad_accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step == 1 or step % 20 == 0:
            print(f"step {step:>5}/{args.steps} | loops {n_loops} | lm {running_lm:.4f} | ponder {running_ponder:.4f} | lr {lr:.2e}")
        if step % args.eval_interval == 0 or step == args.steps:
            print(f"  validation loss: {evaluate():.4f}")
        if step % args.save_interval == 0 or step == args.steps:
            save_checkpoint(output / "latest.pt", model, optimizer, config, step, args)

    final_path = output / "final.pt"
    save_checkpoint(final_path, model, optimizer, config, args.steps, args)
    print(f"Saved Mythos model to {final_path.resolve()}")


if __name__ == "__main__":
    main()

