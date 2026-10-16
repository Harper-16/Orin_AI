#!/usr/bin/env python3
"""Download SQuAD 2.0 and prepare grounded Q&A plus a local passage index."""

import argparse
import json
import random
import re
import sqlite3
import urllib.request
from pathlib import Path

# Allow running this file directly from anywhere inside the project.
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.knowledge_store import SYSTEM_PROMPT, format_grounded_question, query_terms


SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
ABSTAIN_ANSWER = "The passage does not give enough information to answer that."


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row.get("messages"), list):
                yield row


def word_windows(context: str, width: int, overlap: int):
    words = list(re.finditer(r"\S+", context))
    if not words:
        return []
    stride = width - overlap
    chunks = []
    for first in range(0, len(words), stride):
        last = min(first + width, len(words))
        start, end = words[first].start(), words[last - 1].end()
        chunks.append((start, end, context[start:end]))
        if last == len(words):
            break
    return chunks


def choose_positive_chunk(chunks, answer_start: int, answer_text: str):
    answer_end = answer_start + len(answer_text)
    for start, end, passage in chunks:
        if start <= answer_start and answer_end <= end:
            return passage
    folded = answer_text.casefold()
    for start, end, passage in chunks:
        local = passage.casefold().find(folded)
        if local >= 0:
            return passage
    if chunks:
        return min(chunks, key=lambda chunk: abs(chunk[0] - answer_start))[2]
    return ""


def choose_question_chunk(chunks, question: str):
    terms = set(query_terms(question))
    if not chunks:
        return ""
    return max(
        enumerate(chunks),
        key=lambda item: (len(terms & set(query_terms(item[1][2]))), -item[0]),
    )[1][2]


def make_example(question: str, passage: str, title: str, answer: str):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_grounded_question(question, passage, title)},
            {"role": "assistant", "content": answer},
        ]
    }


def select_examples(rows, limit: int, rng: random.Random):
    if len(rows) <= limit:
        return rows
    return rng.sample(rows, limit)


def build_index(path: Path, passages):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE VIRTUAL TABLE knowledge USING fts5(title, passage)")
    except sqlite3.OperationalError as error:
        raise SystemExit(
            "This Python SQLite build does not include FTS5, which is needed for local search. "
            f"Details: {error}"
        ) from error
    connection.executemany(
        "INSERT INTO knowledge(rowid, title, passage) VALUES (?, ?, ?)",
        ((index, title, passage) for index, (title, passage) in enumerate(passages, start=1)),
    )
    connection.commit()
    connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--squad-json", default="data/squad/train-v2.0.json")
    parser.add_argument("--dialogues", default="data/conversation_mix.jsonl")
    parser.add_argument("--basic-qa", default="examples/basic_qa.jsonl")
    parser.add_argument("--output", default="data/grounded_qa_mix.jsonl")
    parser.add_argument("--knowledge-db", default="data/squad_knowledge.sqlite3")
    parser.add_argument("--max-qa", type=int, default=8000)
    parser.add_argument("--max-dialogues", type=int, default=8000)
    parser.add_argument("--context-words", type=int, default=110)
    parser.add_argument("--overlap-words", type=int, default=30)
    parser.add_argument("--seed", type=int, default=4242)
    args = parser.parse_args()
    if args.max_qa < 1 or args.max_dialogues < 1:
        parser.error("--max-qa and --max-dialogues must be positive")
    if args.context_words < 20 or not 0 <= args.overlap_words < args.context_words:
        parser.error("Use at least 20 context words and overlap less than the context size")

    source_path = Path(args.squad_json)
    if not source_path.is_file():
        source_path.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading the official SQuAD 2.0 training set (CC BY-SA 4.0)...")
        try:
            urllib.request.urlretrieve(SQUAD_URL, source_path)
        except Exception as error:
            source_path.unlink(missing_ok=True)
            raise SystemExit(f"Could not download SQuAD data: {error}") from error

    try:
        with source_path.open(encoding="utf-8") as source:
            squad = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Could not read SQuAD file {source_path}: {error}") from error

    rng = random.Random(args.seed)
    positive_examples, negative_examples = [], []
    all_passages = []
    seen_passages = set()
    for article in squad.get("data", []):
        title = article.get("title", "Unknown article")
        for paragraph in article.get("paragraphs", []):
            context = paragraph.get("context", "")
            chunks = word_windows(context, args.context_words, args.overlap_words)
            for _, _, passage in chunks:
                key = (title, passage)
                if key not in seen_passages:
                    seen_passages.add(key)
                    all_passages.append(key)
            for qa in paragraph.get("qas", []):
                question = qa.get("question", "").strip()
                if not question or not chunks:
                    continue
                answers = qa.get("answers") or []
                if answers and not qa.get("is_impossible", False):
                    answer = answers[0]
                    answer_text = answer.get("text", "").strip()
                    answer_start = int(answer.get("answer_start", 0))
                    passage = choose_positive_chunk(chunks, answer_start, answer_text)
                    if answer_text and passage:
                        positive_examples.append(
                            make_example(question, passage, title, answer_text)
                        )
                else:
                    passage = choose_question_chunk(chunks, question)
                    negative_examples.append(
                        make_example(question, passage, title, ABSTAIN_ANSWER)
                    )

    positive_limit = round(args.max_qa * 0.85)
    negative_limit = args.max_qa - positive_limit
    selected_positive = select_examples(positive_examples, positive_limit, rng)
    selected_negative = select_examples(negative_examples, negative_limit, rng)
    remaining = args.max_qa - len(selected_positive) - len(selected_negative)
    if remaining > 0:
        unused = [
            row for row in positive_examples if row not in selected_positive
        ] + [row for row in negative_examples if row not in selected_negative]
        selected_positive.extend(select_examples(unused, remaining, rng))
    qa_examples = selected_positive + selected_negative
    rng.shuffle(qa_examples)
    if not qa_examples:
        raise SystemExit("No usable answerable or unanswerable questions were found in SQuAD.")

    dialogue_path = Path(args.dialogues)
    if not dialogue_path.is_file():
        raise SystemExit(
            f"Conversation data not found: {dialogue_path}. Run data_prep/prepare_conversation_mix.py first."
        )
    dialogues = select_examples(list(read_jsonl(dialogue_path)), args.max_dialogues, rng)
    basic_path = Path(args.basic_qa)
    basic = list(read_jsonl(basic_path)) if basic_path.is_file() else []
    if not dialogues:
        raise SystemExit("No usable multi-turn or chat examples were found.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as target:
        for row in qa_examples + dialogues + basic:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")

    build_index(Path(args.knowledge_db), all_passages)
    attribution = Path(args.knowledge_db).with_suffix(".source.json")
    attribution.write_text(
        json.dumps(
            {
                "name": "Stanford Question Answering Dataset 2.0 (SQuAD)",
                "url": "https://rajpurkar.github.io/SQuAD-explorer/",
                "license": "CC BY-SA 4.0",
                "source_data": str(source_path),
                "passages": len(all_passages),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(qa_examples):,} grounded QA rows, {len(dialogues):,} dialogue rows, "
        f"and {len(basic):,} starter QA rows to {output}."
    )
    print(f"Indexed {len(all_passages):,} local Wikipedia passages in {args.knowledge_db}.")


if __name__ == "__main__":
    main()
