#!/usr/bin/env python3
"""CPU-first LoRA supervised fine-tuning for a small instruct model."""

import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer


DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a local assistant adapter.")
    parser.add_argument("--data", default="data/ultrachat.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", default="adapters/qwen3-0.6b-laptop")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--resume", help="Resume from a saved Trainer checkpoint directory.")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data).expanduser()
    if not data_path.is_file():
        raise SystemExit(f"Training data not found: {data_path}. Run data_prep/download_data.py first.")
    if args.max_steps < 1 or args.max_length < 64 or args.grad_accum < 1:
        raise SystemExit("Steps and accumulation must be positive; max length must be at least 64.")

    has_cuda = torch.cuda.is_available()
    if not has_cuda:
        torch.set_num_threads(max(1, min(args.threads, os.cpu_count() or 1)))
    use_bf16 = has_cuda and torch.cuda.is_bf16_supported()
    model_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if has_cuda else torch.float32)

    dataset = load_dataset("json", data_files=str(data_path), split="train")
    if len(dataset) < 1 or "messages" not in dataset.column_names:
        raise SystemExit("The JSONL file must contain at least one row with a 'messages' list.")
    extra_columns = [column for column in dataset.column_names if column != "messages"]
    if extra_columns:
        dataset = dataset.remove_columns(extra_columns)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    save_steps = max(1, min(50, args.max_steps // 2 or 1))
    training_args = SFTConfig(
        output_dir=args.output,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, args.max_steps * 3 // 100),
        logging_steps=5,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        max_length=args.max_length,
        packing=False,
        assistant_only_loss=True,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        optim="adamw_torch",
        use_cpu=not has_cuda,
        bf16=use_bf16,
        fp16=has_cuda and not use_bf16,
        report_to="none",
        model_init_kwargs={
            "dtype": model_dtype,
            "low_cpu_mem_usage": True,
        },
    )

    adapter_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    print(f"Model: {args.model}")
    print(f"Device: {'CUDA' if has_cuda else 'CPU'}")
    print(f"Training examples: {len(dataset):,}")
    print(f"Sequence limit: {args.max_length}")
    print(f"Steps: {args.max_steps:,}; effective batch: {args.grad_accum}")
    print("Only LoRA adapter weights will be trained.")

    trainer = SFTTrainer(
        model=args.model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=adapter_config,
    )
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Saved adapter to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
