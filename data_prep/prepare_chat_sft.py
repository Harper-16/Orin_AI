#!/usr/bin/env python3
"""Build a chat SFT set that teaches greetings and small talk on its own.

The model should greet from its own weights, not from a hardcoded preset. This
script expands many greeting/thanks/goodbye phrasings (and a few multi-turn
openers) into supervised examples, then mixes in any extra conversation files
you pass with --extra. Feed the result to hybrid/finetune_hybrid.py.
"""

import argparse
import json
from pathlib import Path


SYSTEM = (
    "You are a helpful assistant. Be clear and honest when uncertain. "
    "You have no live internet access."
)

# user phrase -> several acceptable assistant replies
OPENERS = {
    "hi": ["Hi! How can I help you?", "Hello! What would you like to talk about?", "Hi there. What can I do for you?"],
    "hey": ["Hey! What's up?", "Hey there. How can I help?", "Hey! What would you like to talk about?"],
    "hello": ["Hello! How can I help?", "Hello. What would you like to chat about?", "Hi! Good to see you."],
    "hiya": ["Hiya! What can I do for you?", "Hey! How's it going?", "Hi there!"],
    "yo": ["Hey! What's up?", "Yo! What can I help with?", "Hey there."],
    "good morning": ["Good morning! How are you today?", "Good morning! How can I help?", "Morning! How's it going?"],
    "good afternoon": ["Good afternoon! What can I help with?", "Good afternoon. How can I help you today?"],
    "good evening": ["Good evening! What would you like to talk about?", "Good evening! How are you?"],
    "how are you": ["I'm doing well, thanks for asking. How are you?", "I'm here and ready to chat. How are you?", "I'm good, thanks!"],
    "how's it going": ["It's going well, thanks! How about you?", "Pretty good! How are you doing?"],
    "what's up": ["Not much, just ready to chat. What's up with you?", "Not much! How are you?"],
    "are you there": ["Yes, I'm here. What would you like to talk about?", "I'm here! What can I help with?"],
    "nice to meet you": ["Nice to meet you too! How can I help?", "Nice to meet you! What would you like to do?"],
    "thanks": ["You're welcome!", "You're welcome. I'm glad to help.", "No problem at all."],
    "thank you": ["You're welcome!", "Happy to help!", "You're welcome. I'm glad I could help."],
    "thanks for your help": ["You're welcome! Anytime.", "No problem, glad to help."],
    "bye": ["Goodbye! Have a good day.", "See you! Take care.", "Bye! Come back anytime."],
    "goodbye": ["Goodbye! Take care.", "Bye! Have a great day."],
    "bye for now": ["See you later! Take care.", "Bye for now! Nice talking to you."],
    "see you": ["See you! Take care.", "See you soon! Bye."],
}

# Multi-turn openers: follow-up user turns and how the assistant keeps chatting.
FOLLOWUPS = [
    (["hi", "how are you?"], ["Hi! How can I help you?", "I'm doing well, thanks! How are you?"]),
    (["hello", "what can you do?"], ["Hello! How can I help?", "I can chat and answer some simple questions. What would you like to try?"]),
    (["hey", "I'm bored."], ["Hey! What's up?", "I'm sorry to hear that. Want a joke or a small game?"]),
    (["good morning", "I'm tired."], ["Good morning! How are you today?", "That's understandable. Hope your day gets easier."]),
    (["hi", "my name is Sam."], ["Hi! How can I help you?", "Nice to meet you, Sam!"]),
    (["thanks", "that helped a lot."], ["You're welcome!", "I'm glad it helped!"]),
]


def make_row(turns):
    return {"messages": [{"role": "system", "content": SYSTEM}] + turns}


def build_rows():
    rows = []
    for phrase, replies in OPENERS.items():
        for text in {phrase, phrase.capitalize()}:
            for reply in replies:
                rows.append(make_row([
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": reply},
                ]))
    for user_turns, assistant_turns in FOLLOWUPS:
        messages = []
        for user, assistant in zip(user_turns, assistant_turns):
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": assistant})
        rows.append(make_row(messages))
    return rows


def read_rows(path: Path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row.get("messages"), list):
                yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/chat_sft.jsonl")
    parser.add_argument("--extra", action="append", default=[], help="Extra conversation JSONL to mix in. Repeatable.")
    parser.add_argument("--repeats", type=int, default=3, help="How many times to repeat the generated greeting rows.")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    greetings = build_rows()
    extras = []
    for extra in args.extra:
        extra_path = Path(extra)
        if extra_path.is_file():
            extras.extend(read_rows(extra_path))
        else:
            print(f"Skipping missing --extra file: {extra_path}")

    with output.open("w", encoding="utf-8") as target:
        for _ in range(args.repeats):
            for row in greetings:
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
        for row in extras:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"Wrote {len(greetings) * args.repeats:,} greeting rows "
        f"plus {len(extras):,} extra rows to {output}"
    )


if __name__ == "__main__":
    main()
