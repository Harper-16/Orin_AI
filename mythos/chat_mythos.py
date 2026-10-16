#!/usr/bin/env python3
"""Chat locally with the Mythos recurrent-depth Transformer.

This is scratch/chat_from_scratch.py plus a self-verifying "rethink" loop. With
--thinking on (the default), the model thinks at several recurrence depths,
samples a few candidate answers and returns the best *verified* one instead of
trusting a single greedy decode. Use --loops N to pin a fixed thinking depth,
or type /think to toggle the loop at runtime.
"""

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

from mythos.mythos_model import MythosConfig, create_mythos
from common.conversation_helpers import answer_arithmetic, handle_conversation, soften_unrelated_question_reply
from common.knowledge_store import KnowledgeStore, SYSTEM_PROMPT, format_grounded_question
from mythos.mythos_reasoning import SelfVerifyingReasoner


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Mythos checkpoint path. Defaults to the strongest available one.")
    parser.add_argument("--tokenizer", default="data/scratch_tokenizer.json")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--history-turns", type=int, default=2, help="Keep this many recent user/assistant exchanges.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--loops", type=int, help="Fixed recurrence depth for the model. Defaults to the trained maximum.")
    parser.add_argument("--max-loops", type=int, help="Maximum thinking depth for the verifier. Defaults to the trained maximum.")
    parser.add_argument("--candidates", type=int, default=4, help="Candidate answers sampled per thinking depth.")
    parser.add_argument("--patience", type=int, default=2, help="Stop deepening after this many non-improving depths.")
    parser.add_argument("--thinking", dest="thinking", action="store_true", default=True, help="Enable the self-verifying rethink loop (default).")
    parser.add_argument("--no-thinking", dest="thinking", action="store_false", help="Single greedy decode; no rethink loop.")
    parser.add_argument("--conversation-helpers", action="store_true", help="Enable preset replies for greetings, feelings, names and arithmetic.")
    parser.add_argument("--no-relevance-filter", action="store_true", help="Keep longer answers even if they share no words with an information question.")
    parser.add_argument("--knowledge-db", default="data/squad_knowledge.sqlite3", help="Local Wikipedia passage index created by data_prep/prepare_grounded_qa.py.")
    parser.add_argument("--no-knowledge", action="store_true", help="Disable local passage search and use only the model.")
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
                Path("checkpoints/mythos/mythos-grounded/final.pt"),
                Path("checkpoints/mythos/mythos-17m/final.pt"),
            ) if candidate.is_file()),
            Path("checkpoints/mythos/mythos-17m/final.pt"),
        )
    tokenizer_path = Path(args.tokenizer)
    if not checkpoint_path.is_file():
        raise SystemExit(f"Mythos checkpoint not found: {checkpoint_path}. Train it with mythos/train_mythos.py first.")
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
    config = MythosConfig.from_dict(checkpoint["config"])
    model = create_mythos(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    knowledge = None
    if not args.no_knowledge and Path(args.knowledge_db).is_file():
        try:
            knowledge = KnowledgeStore(Path(args.knowledge_db))
        except Exception as error:
            print(f"Local knowledge search is unavailable: {error}")
    stop_ids = {
        value for value in (tokenizer.token_to_id("<|end|>"), tokenizer.token_to_id("<|eos|>"))
        if value is not None
    }

    max_loops = args.max_loops or config.max_loop_iters
    reasoner = SelfVerifyingReasoner(
        model, tokenizer, device, stop_ids,
        temperature=args.temperature, top_k=args.top_k, max_new_tokens=args.max_new_tokens,
        min_loops=1, max_loops=max_loops, candidates=args.candidates, patience=args.patience,
        max_seq_len=config.max_seq_len,
    )
    thinking = args.thinking
    fixed_loops = args.loops

    print(f"Loaded Mythos RDT on {device.type.upper()}: {model.parameter_count():,} parameters.")
    print(f"Trained loop depth: {config.max_loop_iters}; depth-LoRA rank: {config.depth_lora_rank}; halting: {config.use_halting}.")
    print(f"Rethink loop: {'ON (self-verifying)' if thinking else 'OFF (single decode)'}; max depth {max_loops}.")
    if knowledge is not None:
        print(f"Local knowledge: {knowledge.passage_count():,} searchable passages.")
    else:
        print("Local knowledge: not loaded; run data_prep/prepare_grounded_qa.py to add factual sources.")
    print("Commands: /clear, /think, /loops <n>, /exit")

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
        if user_text.lower() == "/think":
            thinking = not thinking
            print(f"Rethink loop {'enabled' if thinking else 'disabled'}.")
            continue
        if user_text.lower().startswith("/loops"):
            parts = user_text.split()
            if len(parts) == 2 and parts[1].isdigit():
                fixed_loops = max(1, min(int(parts[1]), config.max_loop_iters))
                reasoner.max_loops = fixed_loops
                print(f"Thinking depth capped at {fixed_loops} loop(s).")
            else:
                fixed_loops = None
                reasoner.max_loops = max_loops
                print("Thinking depth cleared; using the trained maximum.")
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

        meta = None
        if thinking:
            result = reasoner.reason(input_ids, user_text, retrieved=retrieved)
            answer = result["answer"]
            meta = result
        else:
            tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
            with torch.inference_mode():
                output = model.generate(
                    tensor, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                    top_k=args.top_k, stop_token_ids=stop_ids, n_loops=fixed_loops,
                )
            generated = output[0, len(input_ids):].tolist()
            answer = tokenizer.decode(generated, skip_special_tokens=True).strip()

        if args.conversation_helpers and not args.no_relevance_filter:
            answer = soften_unrelated_question_reply(user_text, answer)
        print(f"Assistant: {answer}")
        if meta is not None:
            detail = meta["detail"]
            bits = [f"{meta['loops']} loop(s)", f"score {meta['score']}"]
            if "support" in detail:
                bits.append(f"support {detail['support']}")
            if "consistency" in detail:
                bits.append(f"agreement {detail['consistency']}")
            print(f"  [rethink: {', '.join(bits)}; {meta['sampled']} candidates]")
        if retrieved:
            print(f"Source: Wikipedia — {retrieved['title']}")
        print()
        history.append(("assistant", answer))

    if knowledge is not None:
        knowledge.close()


if __name__ == "__main__":
    main()

