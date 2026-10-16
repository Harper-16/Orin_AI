#!/usr/bin/env python3
"""Chat locally with the custom Transformer trained from random initialization."""

import argparse
import os
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
from common.conversation_helpers import answer_arithmetic, handle_conversation, soften_unrelated_question_reply
from common.knowledge_store import KnowledgeStore, SYSTEM_PROMPT, format_grounded_question


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Checkpoint path. Defaults to the strongest available chat checkpoint.")
    parser.add_argument("--tokenizer", default="data/scratch_tokenizer.json")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--history-turns", type=int, default=2, help="Keep this many recent user/assistant exchanges.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--conversation-helpers", action="store_true", help="Enable optional preset replies for common greetings, feelings, names, and arithmetic.")
    parser.add_argument("--no-relevance-filter", action="store_true", help="Keep longer answers even if they share no words with an information question.")
    parser.add_argument("--knowledge-db", default="data/squad_knowledge.sqlite3", help="Local Wikipedia passage index created by data_prep/prepare_grounded_qa.py.")
    parser.add_argument("--no-knowledge", action="store_true", help="Disable local passage search and use only the Transformer.")
    return parser.parse_args()


def encode_prompt(messages, tokenizer):
    ids = [tokenizer.token_to_id("<|bos|>")]
    for role, content in messages:
        ids.append(tokenizer.token_to_id(f"<|{role}|>"))
        ids.extend(tokenizer.encode(content).ids)
        ids.append(tokenizer.token_to_id("<|end|>"))
    ids.append(tokenizer.token_to_id("<|assistant|>"))
    return ids


def main():
    args = parse_args()
    if args.history_turns < 1:
        raise SystemExit("--history-turns must be at least 1.")
    if args.model:
        checkpoint_path = Path(args.model)
    else:
        checkpoint_path = next(
            (candidate for candidate in (
                Path("checkpoints/scratch/scratch-chat-grounded/final.pt"),
                Path("checkpoints/scratch/scratch-chat-expanded/final.pt"),
                Path("checkpoints/scratch/scratch-chat/final.pt"),
            ) if candidate.is_file()),
            Path("checkpoints/scratch/scratch-chat-expanded/final.pt"),
        )
    tokenizer_path = Path(args.tokenizer)
    if not checkpoint_path.is_file():
        raise SystemExit(f"Model checkpoint not found: {checkpoint_path}. Train it first.")
    if not tokenizer_path.is_file():
        raise SystemExit(f"Tokenizer not found: {tokenizer_path}. Build it first.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but PyTorch cannot see a CUDA GPU.")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(args.threads, os.cpu_count() or 1)))

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = ModelConfig.from_dict(checkpoint["config"])
    model = create_model(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    knowledge = None
    if not args.no_knowledge and Path(args.knowledge_db).is_file():
        try:
            knowledge = KnowledgeStore(Path(args.knowledge_db))
        except Exception as error:
            print(f"Local knowledge search is unavailable: {error}")
    end_id = tokenizer.token_to_id("<|end|>")
    eos_id = tokenizer.token_to_id("<|eos|>")
    stop_ids = {value for value in (end_id, eos_id) if value is not None}

    print(
        f"Loaded your {model.parameter_count():,}-parameter scratch model on "
        f"{device.type.upper()}."
    )
    if args.conversation_helpers:
        print("Reply mode: Transformer with optional preset conversation helpers.")
    else:
        print("Reply mode: Transformer-generated.")
    if knowledge is not None:
        print(f"Local knowledge: {knowledge.passage_count():,} searchable passages.")
    else:
        print("Local knowledge: not loaded; run data_prep/prepare_grounded_qa.py to add factual sources.")
    print("Commands: /clear, /exit")
    history = [("system", SYSTEM_PROMPT)]
    remembered_name = None

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_text.lower() in {"/exit", "/quit"}:
            break
        if user_text.lower() == "/clear":
            history = [("system", SYSTEM_PROMPT)]
            remembered_name = None
            print("Conversation cleared.")
            continue
        if not user_text:
            continue

        helper_answer = None
        if args.conversation_helpers:
            helper_answer, remembered_name = handle_conversation(user_text, remembered_name)
            if helper_answer is None:
                helper_answer = answer_arithmetic(user_text)
        if helper_answer is not None:
            history.append(("user", user_text))
            answer = helper_answer
            print(f"Assistant: {answer}\n")
            history.append(("assistant", answer))
            continue
        retrieved = knowledge.search(user_text) if knowledge is not None else None
        prompt_text = (
            format_grounded_question(user_text, retrieved["passage"], retrieved["title"])
            if retrieved else user_text
        )
        history.append(("user", prompt_text))
        recent_messages = args.history_turns * 2
        if len(history) > 1 + recent_messages:
            history = [history[0]] + history[-recent_messages:]
        input_ids = encode_prompt(history, tokenizer)
        while len(input_ids) > config.max_seq_len and len(history) > 2:
            del history[1:3]
            input_ids = encode_prompt(history, tokenizer)
        if len(input_ids) > config.max_seq_len and len(history) > 1:
            user_content_ids = tokenizer.encode(history[-1][1]).ids
            fixed_length = len(input_ids) - len(user_content_ids)
            content_budget = max(0, config.max_seq_len - fixed_length)
            history[-1] = ("user", tokenizer.decode(user_content_ids[:content_budget]))
            input_ids = encode_prompt(history, tokenizer)
        if len(input_ids) > config.max_seq_len:
            raise SystemExit("The model context is too small for the current prompt and system message.")
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            output = model.generate(
                input_tensor,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                stop_token_ids=stop_ids,
            )
        generated = output[0, len(input_ids):].tolist()
        answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
        if args.conversation_helpers and not args.no_relevance_filter:
            answer = soften_unrelated_question_reply(user_text, answer)
        print(f"Assistant: {answer}\n")
        if retrieved:
            print(f"Source: Wikipedia — {retrieved['title']}\n")
        history.append(("assistant", answer))

    if knowledge is not None:
        knowledge.close()


if __name__ == "__main__":
    main()
