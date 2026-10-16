#!/usr/bin/env python3
"""Ask a question; the agent searches Wikipedia repeatedly until it can answer.

    python agent/chat_agent.py

The agent plans a search, reads the passages, judges whether the evidence
actually answers the question, and if it does not it *refines the query and
searches again*. It stops when it is confident, when refinement stops
helping, or when it runs out of steps — and if it never finds solid evidence
it says so instead of guessing. Every answer is printed with its source, the
sentence it came from, its confidence, and the searches it ran.
"""

import argparse
from pathlib import Path

# Allow running this file directly from anywhere inside the project.
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent.wiki_agent import RetrievalAgent
from agent.wiki_tools import LocalIndexSearch, WikipediaSearch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline", action="store_true", help="Use the local SQuAD index instead of the web.")
    parser.add_argument("--knowledge-db", default="data/squad_knowledge.sqlite3")
    parser.add_argument("--lang", default="en", help="Wikipedia language edition.")
    parser.add_argument("--max-steps", type=int, default=4, help="Maximum searches per question.")
    parser.add_argument("--passages", type=int, default=4, help="Passages to read per search.")
    parser.add_argument("--good-enough", type=float, default=0.55, help="Confidence needed to stop searching.")
    parser.add_argument("--timeout", type=int, default=12)
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between requests (be polite).")
    parser.add_argument("--show-trace", action="store_true", help="Print every search the agent ran.")
    return parser.parse_args()


def build_agent(args):
    if args.offline:
        retriever = LocalIndexSearch(Path(args.knowledge_db))
        label = f"local index ({args.knowledge_db})"
    else:
        retriever = WikipediaSearch(lang=args.lang, timeout=args.timeout, delay=args.delay)
        label = f"Wikipedia ({args.lang})"
    agent = RetrievalAgent(
        retriever,
        max_steps=args.max_steps,
        passages_per_step=args.passages,
        good_enough=args.good_enough,
    )
    return agent, retriever, label


def show(question: str, result: dict, show_trace: bool) -> None:
    if show_trace:
        for step in result["steps"]:
            print(f"  search {step['step']}: \"{step['query']}\" "
                  f"-> {step['results']} passage(s), best: {step['best']} "
                  f"(confidence {step['confidence']})")
    if not result["answer"]:
        print("Assistant: I could not find a reliable answer for that in the sources I searched.\n")
        return
    print(f"Assistant: {result['answer']}")
    if result.get("supporting_sentence") and result["supporting_sentence"] != result["answer"]:
        print(f"  from: \"{result['supporting_sentence']}\"")
    status = "verified" if result["verified"] else "low confidence"
    print(f"Source: Wikipedia — {result['source']}  [{status}, confidence {result['confidence']}]")
    print()


def main():
    args = parse_args()
    agent, retriever, label = build_agent(args)
    print(f"Agent ready. Searching: {label}")
    print(f"Max {args.max_steps} searches per question; answers must reach confidence "
          f"{args.good_enough} to be called verified.")
    print("Commands: /trace on|off, /steps <n>, /exit\n")
    show_trace = args.show_trace

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"/exit", "/quit"}:
            break
        if question.lower().startswith("/trace"):
            show_trace = "off" not in question.lower()
            print(f"Search trace {'on' if show_trace else 'off'}.")
            continue
        if question.lower().startswith("/steps"):
            parts = question.split()
            if len(parts) == 2 and parts[1].isdigit():
                agent.max_steps = max(1, min(int(parts[1]), 12))
                print(f"Max searches per question: {agent.max_steps}")
            else:
                print(f"Max searches per question: {agent.max_steps}")
            continue
        if not question:
            continue
        result = agent.answer(question)
        show(question, result, show_trace)

    if isinstance(retriever, LocalIndexSearch):
        retriever.close()


if __name__ == "__main__":
    main()
