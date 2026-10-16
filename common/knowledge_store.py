#!/usr/bin/env python3
"""Small, local full-text index used to ground factual chat replies."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a helpful assistant. Keep conversations natural. When a reference "
    "passage is provided, use it for factual answers. If it does not support "
    "an answer, say you are unsure instead of inventing facts."
)

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
    "can", "could", "did", "do", "does", "for", "from", "had", "has",
    "have", "how", "i", "in", "is", "it", "its", "of", "on", "or", "the",
    "their", "them", "there", "these", "this", "those", "to", "was", "we",
    "were", "what", "when", "where", "which", "who", "whom", "why", "will",
    "with", "would", "you", "your",
}


def format_grounded_question(question: str, passage: str, title: str) -> str:
    return f"Question: {question}\n\nReference passage (Wikipedia: {title}):\n{passage}"


def query_terms(text: str) -> list[str]:
    return [
        term for term in re.findall(r"[\w]+", text.casefold())
        if len(term) >= 2 and term not in STOP_WORDS
    ]


class KnowledgeStore:
    def __init__(self, path: Path):
        self.path = path
        self.connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)

    def close(self) -> None:
        self.connection.close()

    def passage_count(self) -> int:
        row = self.connection.execute("SELECT count(*) FROM knowledge").fetchone()
        return int(row[0])

    def search(self, question: str) -> dict[str, str] | None:
        terms = list(dict.fromkeys(query_terms(question)))
        if not terms or len(re.findall(r"\w+", question)) < 3:
            return None
        # Quote each word so user punctuation cannot change the FTS query.
        expression = " OR ".join(f'"{term}"*' for term in terms)
        rows = self.connection.execute(
            "SELECT title, passage FROM knowledge "
            "WHERE knowledge MATCH ? ORDER BY bm25(knowledge) LIMIT 8",
            (expression,),
        ).fetchall()
        minimum_overlap = min(2, len(terms))
        best = None
        best_score = -1
        for title, passage in rows:
            passage_terms = set(query_terms(f"{title} {passage}"))
            overlap = sum(
                1 for term in terms
                if term in passage_terms or any(word.startswith(term) for word in passage_terms)
            )
            if overlap > best_score:
                best = {"title": title, "passage": passage}
                best_score = overlap
        if best_score < minimum_overlap:
            return None
        best["passage"] = best["passage"][:900].strip()
        return best
