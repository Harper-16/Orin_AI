#!/usr/bin/env python3
"""Chat with the hybrid Transformer + Mythos model, mixing architectures.

The hybrid itself already combines both paths. On top of that, --mix-with lets
the self-verifying rethink loop draw candidate answers from OTHER trained
checkpoints too (a plain Transformer, a Mythos model, another hybrid), so the
final answer is verified across architectures, not just sampled within one.
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

from scratch.scratch_model import ModelConfig, create_model
from mythos.mythos_model import MythosConfig, create_mythos
from hybrid.hybrid_model import HybridConfig, create_hybrid
from agent.intent_router import Intent, classify
from common.conversation_helpers import answer_arithmetic, handle_conversation, soften_unrelated_question_reply
from common.knowledge_store import KnowledgeStore, SYSTEM_PROMPT, format_grounded_question
from mythos.mythos_reasoning import SelfVerifyingReasoner


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Hybrid checkpoint path. Defaults to the strongest available one.")
    parser.add_argument("--mix-with", action="append", default=[], help="Extra checkpoint(s) whose answers also enter the rethink pool. Repeatable.")
    parser.add_argument("--tokenizer", default="data/scratch_tokenizer.json")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--history-turns", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--loops", type=int, help="Fixed recurrence depth. Defaults to the trained maximum.")
    parser.add_argument("--max-loops", type=int, help="Maximum thinking depth for the verifier.")
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0,
                        help="Sampling seed per turn (0 = deterministic per prompt hash; set >0 for fixed replies).")
    parser.add_argument("--fast", action="store_true",
                        help="Speed preset for weak CPUs: 1 loop, 1 candidate, greedy decode, helpers off.")
    parser.add_argument("--human", action="store_true",
                        help="Warm human preset: model-only, temp 0.5, top-k 40, 2 loops, 2 candidates, "
                             "history 3 turns, 96 tokens. Natural, still stable.")
    parser.add_argument("--thinking", dest="thinking", action="store_true", default=True)
    parser.add_argument("--no-thinking", dest="thinking", action="store_false")
    parser.add_argument("--conversation-helpers", action="store_true")
    parser.add_argument("--no-relevance-filter", action="store_true")
    parser.add_argument("--knowledge-db", default="data/squad_knowledge.sqlite3")
    parser.add_argument("--no-knowledge", action="store_true")
    return parser.parse_args()


def encode_prompt(messages, tokenizer):
    ids = [tokenizer.token_to_id("<|bos|>")]
    for role, content in messages:
        ids.append(tokenizer.token_to_id(f"<|{role}|>"))
        ids.extend(tokenizer.encode(content).ids)
        ids.append(tokenizer.token_to_id("<|end|>"))
    ids.append(tokenizer.token_to_id("<|assistant|>"))
    return ids


def load_checkpoint_model(path: Path, device: torch.device):
    """Rebuild any architecture in this project from its checkpoint config."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint.get("config", {})
    if "stages" in config:
        model = create_hybrid(HybridConfig.from_dict(config))
        kind = "hybrid"
    elif "max_loop_iters" in config:
        model = create_mythos(MythosConfig.from_dict(config))
        kind = "mythos"
    else:
        model = create_model(ModelConfig.from_dict(config))
        kind = "transformer"
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), kind


def make_sampler(model, tokenizer, device, stop_ids, max_new_tokens, temperature, top_k):
    """Return fn(prompt_ids, n_loops) -> text for the rethink candidate pool."""
    supports_loops = hasattr(model.config, "max_loop_iters")

    def sampler(prompt_ids, n_loops):
        prompt = prompt_ids[-model.config.max_seq_len :]
        tensor = torch.tensor([prompt], dtype=torch.long, device=device)
        kwargs = dict(
            max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k,
            stop_token_ids=stop_ids,
        )
        if supports_loops:
            kwargs["n_loops"] = n_loops
        with torch.inference_mode():
            output = model.generate(tensor, **kwargs)
        return tokenizer.decode(output[0, len(prompt):].tolist(), skip_special_tokens=True).strip()

    return sampler


def main():
    args = parse_args()
    if args.history_turns < 1:
        raise SystemExit("--history-turns must be at least 1.")
    if args.model:
        checkpoint_path = Path(args.model)
    else:
        checkpoint_path = next(
            (candidate for candidate in (
                Path("checkpoints/hybrid/hybrid-75m-chat/final.pt"),
                Path("checkpoints/hybrid/hybrid-grounded/final.pt"),
                Path("checkpoints/hybrid/hybrid-17m/final.pt"),
            ) if candidate.is_file()),
            Path("checkpoints/hybrid/hybrid-17m/final.pt"),
        )
    tokenizer_path = Path(args.tokenizer)
    if not checkpoint_path.is_file():
        raise SystemExit(f"Hybrid checkpoint not found: {checkpoint_path}. Train it with hybrid/train_hybrid.py first.")
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
    if args.fast:
        # Weak-CPU preset: greedy single decode, no rethink sampling.
        # No preset helpers: replies come from the model alone.
        args.thinking = False
        args.temperature = 0.0
        args.candidates = 1
        args.conversation_helpers = False
    if args.human:
        # Warm but bounded: a little sampling for variety, NOT greedy-robot.
        # Helpers stay OFF (model-only, as requested).
        args.conversation_helpers = False
        args.temperature = 0.5
        args.top_k = 40
        args.candidates = 2
        args.patience = 1
        args.history_turns = max(args.history_turns, 3)
        args.max_new_tokens = max(args.max_new_tokens, 96)
        if not args.loops:
            fixed_loops_want = 2
        else:
            fixed_loops_want = args.loops
        args.loops = fixed_loops_want

    if args.seed and args.seed > 0:
        torch.manual_seed(args.seed)
        base_seed = args.seed
    else:
        base_seed = None

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    model, _ = load_checkpoint_model(checkpoint_path, device)
    config = model.config
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

    # Optional cross-architecture candidates: rebuild each extra model.
    extra_samplers = []
    for extra in args.mix_with:
        extra_path = Path(extra)
        if not extra_path.is_file():
            print(f"Skipping missing mix checkpoint: {extra_path}")
            continue
        extra_model, kind = load_checkpoint_model(extra_path, device)
        if extra_model.config.vocab_size != config.vocab_size:
            print(f"Skipping {extra_path}: vocabulary does not match the tokenizer.")
            continue
        extra_samplers.append((
            f"{kind}:{extra_path.name}",
            make_sampler(extra_model, tokenizer, device, stop_ids, args.max_new_tokens, args.temperature, args.top_k),
        ))
        print(f"Mixing in {kind} checkpoint: {extra_path}")

    max_loops = args.max_loops or getattr(config, "max_loop_iters", 1)
    reasoner = SelfVerifyingReasoner(
        model, tokenizer, device, stop_ids,
        temperature=args.temperature, top_k=args.top_k, max_new_tokens=args.max_new_tokens,
        min_loops=1, max_loops=max_loops, candidates=args.candidates, patience=args.patience,
        max_seq_len=config.max_seq_len, extra_samplers=extra_samplers,
    )
    thinking = args.thinking
    fixed_loops = 1 if args.fast else args.loops

    print(f"Loaded Hybrid (Transformer + Mythos) on {device.type.upper()}: {model.parameter_count():,} parameters.")
    print(f"Loops <= {getattr(config, 'max_loop_iters', 1)}; effective depth at max loops: {model.effective_depth(max_loops) if hasattr(model, 'effective_depth') else 'n/a'}.")
    print(f"Rethink loop: {'ON (self-verifying)' if thinking else 'OFF (single decode)'}; mixed sources: {len(extra_samplers)}.")
    if args.conversation_helpers:
        print("Reply mode: model + preset helpers (greetings/arithmetic come from code).")
    else:
        print("Reply mode: model-generated ON ITS OWN (no preset helpers).")
    if knowledge is not None:
        print(f"Local knowledge: {knowledge.passage_count():,} searchable passages.")
    print("Commands: /clear, /think, /loops <n>, /sources, /exit")

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
        if user_text.lower() == "/sources":
            if extra_samplers:
                print("Mixing in: " + ", ".join(label for label, _ in extra_samplers))
            else:
                print("No extra sources. Start with --mix-with <checkpoint> to mix other architectures.")
            continue
        if user_text.lower().startswith("/loops"):
            parts = user_text.split()
            if len(parts) == 2 and parts[1].isdigit():
                fixed_loops = max(1, min(int(parts[1]), getattr(config, "max_loop_iters", 1)))
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

        retrieved = None
        intent, reason = classify(user_text, have_evidence=False)
        if intent == Intent.SEARCH:
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
            if base_seed is None:
                # Deterministic per prompt: same question -> same seed -> same answer.
                import hashlib
                turn_seed = int(hashlib.sha256(" ".join(map(str, input_ids[-64:])).encode()).hexdigest(), 16) % (2 ** 31)
                torch.manual_seed(turn_seed)
            result = reasoner.reason(input_ids, user_text, retrieved=retrieved)
            answer = result["answer"]
            meta = result
        else:
            tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
            kwargs = dict(
                max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                top_k=args.top_k, stop_token_ids=stop_ids,
            )
            if hasattr(model.config, "max_loop_iters"):
                kwargs["n_loops"] = fixed_loops
            with torch.inference_mode():
                output = model.generate(tensor, **kwargs)
            answer = tokenizer.decode(output[0, len(input_ids):].tolist(), skip_special_tokens=True).strip()

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

