#!/usr/bin/env python3
"""The hybrid model wired into the Wikipedia agent: retrieve, read, verify.

This joins the two halves of the project:

  1. **Retrieval agent** (agent/wiki_agent.py) searches Wikipedia recurrently until
     it has a passage that answers the question.
  2. **Hybrid model** (hybrid/hybrid_model.py) *reads* that passage and writes the
     answer in its own words, instead of the agent quoting a sentence.
  3. **Grounding check** — the model's answer is only shown if its content
     words actually appear in the evidence. If the model drifts, we fall back
     to the agent's extractive answer rather than show something unsupported.

Small talk ("hi", "thanks") skips retrieval entirely and is answered by the
model on its own, so it keeps its own conversational ability.

    python agent/chat_agent_hybrid.py --model checkpoints/hybrid/hybrid-grounded/final.pt
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

from hybrid.hybrid_model import HybridConfig, create_hybrid
from agent.intent_router import Intent, classify
from mythos.mythos_reasoning import SelfVerifyingReasoner, normalise
from agent.wiki_agent import RetrievalAgent
from agent.wiki_tools import LocalIndexSearch, WikipediaSearch
from common.knowledge_store import SYSTEM_PROMPT, format_grounded_question


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="checkpoints/hybrid/hybrid-75m-chat/final.pt")
    parser.add_argument("--tokenizer", default="data/scratch_tokenizer.json")
    parser.add_argument("--offline", action="store_true", help="Search the local SQuAD index instead of the web.")
    parser.add_argument("--knowledge-db", default="data/squad_knowledge.sqlite3")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--max-steps", type=int, default=3, help="Maximum searches per question.")
    parser.add_argument("--passages", type=int, default=3)
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between Wikipedia requests.")
    parser.add_argument("--candidates", type=int, default=3, help="Model samples per question.")
    parser.add_argument("--max-loops", type=int, help="Max recurrence depth for the reader (defaults to trained max).")
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--min-grounding", type=float, default=0.5, help="Share of answer words that must appear in the evidence.")
    parser.add_argument("--always-search", action="store_true", help="Skip the intent router and search every turn.")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--show-trace", action="store_true")
    return parser.parse_args()


def looks_like_question(text: str) -> bool:
    """Backwards-compatible shim: does this need the retriever at all?"""
    return classify(text, have_evidence=False)[0] in (Intent.SEARCH, Intent.FOLLOWUP)


def grounding_score(answer: str, passage: str) -> float:
    """Share of the answer's content words that the evidence actually supports."""
    answer_terms = set(normalise(answer).split())
    if not answer_terms:
        return 0.0
    passage_terms = set(normalise(passage).split())
    return len(answer_terms & passage_terms) / len(answer_terms)


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
    model_path = Path(args.model)
    if not model_path.is_file():
        raise SystemExit(f"Model not found: {model_path}. Fine-tune one with hybrid/finetune_hybrid.py.")
    torch.set_num_threads(max(1, min(args.threads, os.cpu_count() or 1)))
    tokenizer = Tokenizer.from_file(args.tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    config = HybridConfig.from_dict(checkpoint["config"])
    model = create_hybrid(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    if args.offline:
        retriever = LocalIndexSearch(Path(args.knowledge_db))
        source_label = f"local index ({args.knowledge_db})"
    else:
        retriever = WikipediaSearch(lang=args.lang, delay=args.delay)
        source_label = f"Wikipedia ({args.lang})"

    max_loops = args.max_loops or config.max_loop_iters
    reasoner = SelfVerifyingReasoner(
        model, tokenizer, device,
        {value for value in (tokenizer.token_to_id("<|end|>"), tokenizer.token_to_id("<|eos|>")) if value is not None},
        temperature=args.temperature, max_new_tokens=48,
        min_loops=1, max_loops=max_loops, candidates=args.candidates,
        patience=1, max_seq_len=config.max_seq_len,
    )
    agent = RetrievalAgent(retriever, max_steps=args.max_steps, passages_per_step=args.passages)

    print(f"Loaded hybrid reader: {model.parameter_count():,} parameters on {device.type.upper()}.")
    print(f"Searching: {source_label} | thinking up to {max_loops} loops | grounding >= {args.min_grounding}")
    print("Small talk is answered by the model; questions are looked up, read and verified.")
    print("Commands: /trace on|off, /model, /search, /why, /exit\n")

    show_trace = args.show_trace
    force_extractive = False
    always_search = args.always_search
    history = [("system", SYSTEM_PROMPT)]
    last_evidence = None  # (question, title, passage) for follow-ups

    def chat_reply(user_text: str) -> str:
        history.append(("user", user_text))
        ids = encode_prompt(history[-6:], tokenizer)[-config.max_seq_len:]
        with torch.inference_mode():
            out = model.generate(
                torch.tensor([ids], dtype=torch.long, device=device),
                max_new_tokens=48, temperature=args.temperature, top_k=40,
                stop_token_ids=reasoner.stop_token_ids, n_loops=max_loops,
            )
        answer = tokenizer.decode(out[0, len(ids):].tolist(), skip_special_tokens=True).strip()
        history.append(("assistant", answer))
        return answer

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_text.lower() in {"/exit", "/quit"}:
            break
        if user_text.lower() == "/model":
            force_extractive = not force_extractive
            print(f"Answering: {'extractive (agent only)' if force_extractive else 'model reads the evidence'}.")
            continue
        if user_text.lower() in {"/search", "/alwayssearch"}:
            always_search = not always_search
            print(f"Intent router {'bypassed — searching every turn' if always_search else 'enabled'}.")
            continue
        if user_text.lower() == "/why":
            if last_evidence:
                question, title, _ = last_evidence
                print(f"Last lookup: \"{question}\" -> {title}")
            else:
                print("Nothing looked up yet this session.")
            continue
        if user_text.lower().startswith("/trace"):
            show_trace = "off" not in user_text.lower()
            print(f"Search trace {'on' if show_trace else 'off'}.")
            continue
        if not user_text:
            continue

        # ---- 0. Decide whether this turn needs a search at all ----
        if always_search:
            intent, reason = Intent.SEARCH, "searching every turn (router bypassed)"
        else:
            intent, reason = classify(user_text, have_evidence=last_evidence is not None)
        if show_trace:
            print(f"  route: {intent} — {reason}")

        if intent == Intent.UNKNOWN:
            print("Assistant: I can't answer that — it needs live data (the time, weather, "
                  "news or today's results) and I only have Wikipedia's static articles.\n")
            continue

        # ---- 1. Conversation, or a follow-up we can answer from what we have ----
        if intent == Intent.CHAT:
            print(f"Assistant: {chat_reply(user_text)}\n")
            continue

        if intent == Intent.FOLLOWUP:
            question, title, passage = last_evidence
            prompt = format_grounded_question(
                f"{question} (follow-up: {user_text})", passage, title
            )
            ids = encode_prompt([("system", SYSTEM_PROMPT), ("user", prompt)], tokenizer)
            ids = ids[-config.max_seq_len:]
            result = reasoner.reason(ids, user_text, retrieved={"title": title, "passage": passage})
            support = grounding_score(result["answer"], passage)
            if result["answer"] and support >= args.min_grounding:
                print(f"Assistant: {result['answer']}")
                print(f"Reusing evidence from: {title}  [grounded {round(support, 2)}, no new search]\n")
            else:
                print("Assistant: I don't have enough in the passage I already found to go on.\n")
            continue

        # ---- 2. Retrieve: the agent searches until it can answer ----
        found = agent.answer(user_text)
        if show_trace:
            for step in found["steps"]:
                print(f"  search {step['step']}: \"{step['query']}\" -> {step['results']} passage(s), "
                      f"best: {step['best']} (confidence {step['confidence']})")
        if not found["answer"] or not found["grounded_passage"]:
            print("Assistant: I searched but could not find a reliable answer for that.\n")
            continue

        passage = found["grounded_passage"]
        source_title = found["source"]
        last_evidence = (user_text, source_title, passage)

        # ---- 3. Extractive answer (always available as a safe fallback) ----
        if force_extractive:
            print(f"Assistant: {found['answer']}")
            print(f"Source: {source_title}  [extractive, confidence {found['confidence']}]\n")
            continue

        # ---- 4. The model reads the evidence and writes the answer ----
        prompt = format_grounded_question(user_text, passage, source_title)
        messages = [("system", SYSTEM_PROMPT), ("user", prompt)]
        ids = encode_prompt(messages, tokenizer)
        if len(ids) > config.max_seq_len:
            # Keep the tail: the instruction and the end of the passage matter most.
            ids = ids[-config.max_seq_len:]
        result = reasoner.reason(ids, user_text, retrieved={"title": source_title, "passage": passage})

        # ---- 5. Verify the model's answer is actually supported ----
        support = grounding_score(result["answer"], passage)
        if result["answer"] and support >= args.min_grounding:
            print(f"Assistant: {result['answer']}")
            print(f"Source: {source_title}  [model, grounded {round(support, 2)}]")
        else:
            print(f"Assistant: {found['answer']}")
            print(f"Source: {source_title}  [extractive fallback — model answer was not supported]")
        print()

    if isinstance(retriever, LocalIndexSearch):
        retriever.close()


if __name__ == "__main__":
    main()

