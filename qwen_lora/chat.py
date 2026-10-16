#!/usr/bin/env python3
"""Run the base model or a locally fine-tuned adapter in a terminal chat."""

import argparse
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
SYSTEM_PROMPT = (
    "You are a helpful assistant. Be clear and honest when uncertain. "
    "You do not have live internet access; do not invent current facts or sources."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with the local assistant.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", help="Path to a LoRA adapter saved by qwen_lora/train.py.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but PyTorch cannot see a CUDA GPU.")

    device = "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    if device == "cpu":
        torch.set_num_threads(max(1, min(args.threads, os.cpu_count() or 1)))
        dtype = torch.float32
    else:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    if args.adapter:
        adapter_path = Path(args.adapter).expanduser()
        if not adapter_path.exists():
            raise SystemExit(f"Adapter not found: {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.to(device)
    model.eval()

    print(f"Loaded {args.model} on {device.upper()}.")
    print("Commands: /clear, /exit")
    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_text.lower() in {"/exit", "/quit"}:
            break
        if user_text.lower() == "/clear":
            history = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("Conversation cleared.")
            continue
        if not user_text:
            continue

        history.append({"role": "user", "content": user_text})
        while True:
            template_kwargs = dict(
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            try:
                encoded = tokenizer.apply_chat_template(
                    history, enable_thinking=False, **template_kwargs
                )
            except TypeError:
                # Older transformers: apply_chat_template has no enable_thinking.
                encoded = tokenizer.apply_chat_template(history, **template_kwargs)
            input_ids = encoded["input_ids"]
            if input_ids.shape[-1] <= args.max_input_tokens or len(history) <= 2:
                break
            del history[1:3]

        if input_ids.shape[-1] > args.max_input_tokens:
            input_ids = input_ids[:, -args.max_input_tokens:]
        input_ids = input_ids.to(device)

        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=max(args.temperature, 1e-5),
                top_p=0.9,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        new_tokens = output[0, input_ids.shape[-1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        print(f"Assistant: {answer}\n")
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
