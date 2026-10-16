#!/usr/bin/env python3
"""A retrieval agent that searches Wikipedia recurrently until it can answer.

The loop is deliberately simple and honest — it plans a query, searches, reads
the passages, extracts a candidate answer sentence, *judges* how well it fits,
and if it is not good enough it refines the query and searches again. It stops
when it is confident, when refinement stops helping, or when it hits the step
budget. Every answer carries its sources and a step-by-step trace, and answers
that cannot be grounded in a passage are reported as uncertain instead of
invented.

It works with any backend exposing ``search_passages(query, limit)`` (live
Wikipedia or the offline SQuAD index) and, optionally, a local language model
used only to rephrase the grounded sentence.
"""

from __future__ import annotations

import re

# Allow running this file directly from anywhere inside the project.
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.knowledge_store import query_terms


QUESTION_WORDS = {
    "what", "who", "whom", "whose", "when", "where", "which", "why", "how",
    "did", "do", "does", "is", "are", "was", "were", "the", "a", "an", "of",
    "in", "on", "at", "to", "for", "and", "or", "that", "this", "it", "its",
}

YEAR = re.compile(r"\b(?:1[0-9]{3}|20[0-9]{2})\b")
NUMBER = re.compile(r"\b\d[\d,.]*\b")
MONTH = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\b"
)

# Question verbs do not appear verbatim in answers ("wrote" -> "written").
# A small map of common irregulars so the answer sentence can be recognised.
VERB_FORMS = {
    "wrote": ("written", "wrote", "writ"),
    "write": ("written", "wrote", "writ"),
    "won": ("won", "win"),
    "sang": ("sang", "sung"),
    "founded": ("founded", "founder"),
    "built": ("built", "build"),
    "discovered": ("discovered", "discover"),
    "invented": ("invented", "invent"),
    "created": ("created", "create"),
    "died": ("died", "death", "die"),
    "born": ("born", "birth"),
    "made": ("made", "make"),
    "released": ("released", "release"),
    "published": ("published", "publish"),
    "directed": ("directed", "director"),
    "composed": ("composed", "composer"),
    "governed": ("governed", "ruled"),
    "spoke": ("spoke", "spoken"),
    "began": ("began", "begun", "begin"),
    "ended": ("ended", "end"),
    "led": ("led", "lead", "leader"),
}

DATE = re.compile(
    r"\b(?:\d{1,2}\s+)?(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{1,2}?,?\s*\d{4}\b"
    r"|\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{4}\b"
    r"|\b(?:1[0-9]{3}|20[0-9]{2})\b"
)
PROPER = re.compile(r"\b[A-Z][\w'’-]+(?:\s+[A-Z][\w'’-]+)*\b")


def keywords(text: str) -> list[str]:
    return [term for term in query_terms(text) if term not in QUESTION_WORDS]


def question_type(question: str) -> str:
    first = keywords(question)[:1]
    head = question.strip().casefold().split()[:1]
    if head and head[0] in {"who", "whom", "whose"}:
        return "who"
    if "when" in head or "what year" in question.casefold() or "what date" in question.casefold():
        return "when"
    if "where" in head:
        return "where"
    if "how many" in question.casefold() or "how much" in question.casefold():
        return "number"
    if head and head[0] == "why":
        return "why"
    return "general"


def split_sentences(text: str) -> list[str]:
    rough = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    return [sentence.strip() for sentence in rough if len(sentence.strip()) > 15]


def _capitals(sentence: str) -> int:
    words = sentence.split()
    return sum(1 for word in words[1:] if word[:1].isupper())


def property_terms(question: str, subject: list[str]) -> list[str]:
    """The words describing *what* is being asked ("capital", "largest").

    "What is the capital of France?" is about France (subject) and asks for its
    capital (property). A sentence stating "Paris is the capital of France"
    contains both, which is what identifies the answer.
    """
    return [term for term in keywords(question) if term not in subject]


def score_sentence(sentence: str, terms: list[str], qtype: str, *, is_lead: bool = False,
                   properties: list[str] | None = None) -> float:
    stripped = sentence.strip()
    if stripped.endswith("?"):  # a question is never the answer
        return -1.0
    lowered = stripped.casefold()
    overlap = sum(1 for term in terms if term in lowered)
    if overlap == 0:
        return -1.0  # an answer must share at least one word with the question
    base = overlap / max(1, len(terms))
    # A sentence carrying BOTH the property ("capital") and the subject
    # ("France") is the definition-style answer: "Paris is the capital of
    # France". Only meaningful for "what/which" questions — for "who wrote the
    # play Hamlet" the word "play" is not a property, and rewarding it would
    # pick a sentence that merely mentions it.
    if properties and qtype == "general":
        subject_set = {term for term in terms if term not in properties}
        has_property = any(item in lowered for item in properties)
        has_subject = any(item in lowered for item in subject_set)
        # The definition-style answer is the one that names the property AND
        # stays on the subject: "Its capital ... is Paris". Make that pairing
        # clearly dominant, otherwise the article's first sentence wins simply
        # because it repeats the subject.
        if has_property and has_subject:
            base += 1.0
        elif has_property:
            # Often the answer sentence names the property without repeating
            # the subject ("Its capital ... is Paris"), so this is nearly as
            # strong — the property word is the most discriminative term.
            base += 0.85
    # "Who *wrote* Hamlet?" is answered by a sentence saying "written by".
    # This is the strongest signal available without a language model, so it
    # outweighs the weaker "shares a common word" evidence.
    for term in terms:
        for form in VERB_FORMS.get(term, ()):
            if form in lowered:
                base += 0.6
                break
    bonus = 0.0
    # The lead sentence is a weak prior, not a strong one: Wikipedia lead
    # sentences describe the topic, they rarely contain the specific answer.
    if qtype == "when" and (YEAR.search(sentence) or MONTH.search(sentence)):
        bonus += 0.5
    elif qtype == "number" and NUMBER.search(sentence):
        bonus += 0.45
    elif qtype == "who" and _capitals(sentence) >= 1:
        bonus += 0.12
    elif qtype == "where" and _capitals(sentence) >= 1:
        bonus += 0.12
    # A lead sentence that states the fact is still a mild plus.
    if is_lead:
        bonus += 0.05
    # Very short fragments rarely read as complete answers.
    if len(sentence.split()) < 5:
        bonus -= 0.15
    return base + bonus


def subject_terms(question: str) -> list[str]:
    """The entity the question is *about* — the proper nouns it mentions.

    "What is the capital of France?" is about France, "Who wrote the play
    Hamlet?" is about Hamlet. Matching the article title against these is a far
    better relevance signal than matching every question word.
    """
    proper = [
        term.casefold()
        for term in re.findall(r"\b[A-Z][\w'’-]{2,}\b", question)
        if term.casefold() not in QUESTION_WORDS
    ]
    if proper:
        return list(dict.fromkeys(proper))
    # No capitals: fall back to the trailing content words ("... the Qatari
    # capital" -> qatari, capital).
    tail = [term for term in keywords(question) if term not in ("what", "who", "when", "where")]
    return tail[-2:] if tail else []


def extract_short_answer(sentence: str, qtype: str, question: str) -> str:
    """Pull the actual fact out of the sentence, not the whole sentence.

    "is a tragedy written by William Shakespeare" -> "William Shakespeare".
    "Its capital ... is Paris" -> "Paris".
    Returns "" when nothing type-appropriate is found, so the caller can fall
    back to quoting the sentence.
    """
    if qtype == "when":
        dates = DATE.findall(sentence) or [match.group(0) for match in DATE.finditer(sentence)]
        if dates:
            # "When did X *end*?" wants the last date in the sentence;
            # "When did X *begin*?" wants the first.
            lowered_question = question.casefold()
            if any(word in lowered_question for word in ("end", "finish", "die", "conclude")):
                return dates[-1].strip()
            if any(word in lowered_question for word in ("begin", "start", "born", "found", "open")):
                return dates[0].strip()
            return dates[0].strip()
    elif qtype in {"who", "where"}:
        asked = {word.casefold() for word in question.split()}
        for match in PROPER.finditer(sentence):
            candidate = match.group(0).strip()
            words = candidate.split()
            if len(words) < 2 and qtype == "who":
                continue
            if all(word.casefold().strip(",.") not in asked for word in words):
                # Prefer the longest multi-word name available.
                return candidate
    elif qtype == "number":
        match = NUMBER.search(sentence)
        if match:
            return match.group(0)
    else:
        # "What is the capital of France?" -> the noun after "capital is".
        definition = re.search(
            r"\b(?:is|was|are|were)\s+(?:the\s+)?([A-Z][\w'’-]+(?:\s+[A-Z][\w'’-]+)*)",
            sentence,
        )
        if definition:
            return definition.group(1).strip()
    return ""


class RetrievalAgent:
    """Search -> read -> judge -> refine, repeatedly, until confident."""

    def __init__(self, retriever, *, max_steps: int = 4, good_enough: float = 0.55,
                 passages_per_step: int = 4, max_answer_words: int = 60, generator=None):
        self.retriever = retriever
        self.max_steps = max(1, max_steps)
        self.good_enough = good_enough
        self.passages_per_step = max(1, passages_per_step)
        self.max_answer_words = max_answer_words
        # Optional local LM: fn(question, passage) -> text, used only to rephrase.
        self.generator = generator

    def _candidate_from_passage(self, question, passage, terms, qtype, subject=(), properties=()):
        sentences = split_sentences(passage)
        if not sentences:
            return "", 0.0
        subject = [item for item in subject if item]
        scored = [
            (score_sentence(sentence, terms, qtype, is_lead=index < 3, properties=properties), sentence)
            for index, sentence in enumerate(sentences)
        ]
        scored = [item for item in scored if item[0] >= 0.0]
        if not scored:
            return "", 0.0
        # Mentioning the subject is a bonus, never a filter: the best answer
        # often says "Its capital ... is Paris" without repeating "France",
        # and a hard filter would throw that sentence away.
        if subject:
            for index, (score, sentence) in enumerate(scored):
                lowered = sentence.casefold()
                if any(term in lowered for term in subject):
                    scored[index] = (score + 0.15, sentence)
        best_score, best_sentence = max(scored, key=lambda item: item[0])
        words = best_sentence.split()
        if len(words) > self.max_answer_words:
            best_sentence = " ".join(words[: self.max_answer_words]).rstrip(",;:") + "..."
        return best_sentence, best_score

    def _evidence_score(self, question_terms, passage, subject, title):
        """How well a source matches the question, and whether it is *about* it.

        Returns (content_overlap, subject_match). subject_match is 1.0 when the
        article's title is the entity the question is about, which is the single
        most reliable signal that we found the right page.
        """
        passage_terms = set(query_terms(passage))
        overlap = sum(1 for term in question_terms if term in passage_terms)
        content_overlap = overlap / max(1, len(question_terms))

        title_words = set(query_terms(title.replace("_", " ")))
        subject_match = 0.0
        if subject and title_words:
            hits = sum(
                1 for term in subject
                if any(word.startswith(term) or term.startswith(word) for word in title_words)
            )
            subject_match = min(1.0, hits / max(1, len(subject)))
        return content_overlap, subject_match

    def _refine(self, question, terms, subject, tried, best, page_titles):
        """Propose a new query when the current step did not answer the question.

        It widens the net: the subject entity, the subject plus the question's
        property word ("Hamlet written by"), and a narrowed property phrase.
        """
        proposals = []
        if subject:
            proposals.append(" ".join(subject))
            property_words = [term for term in terms if term not in subject]
            if property_words:
                proposals.append(f"{' '.join(subject)} {' '.join(property_words[:2])}")
        if terms:
            proposals.append(" ".join(terms))
        if best and best.get("title"):
            proposals.append(best["title"].replace("_", " "))
        if page_titles:
            proposals.append(page_titles[0].replace("_", " "))
        for proposal in proposals:
            proposal = proposal.strip()
            if len(proposal) >= 3 and proposal not in tried:
                return proposal
        return None

    def answer(self, question: str) -> dict:
        terms = keywords(question)
        subject = subject_terms(question)
        properties = property_terms(question, subject)
        qtype = question_type(question)
        tried: list[str] = []
        trace: list[dict] = []
        best = None
        seen_titles: set[str] = set()
        stall = 0
        # Search the subject entity first ("capital of France" -> "France"),
        # because the page we want is the one *about* that entity. Then fall
        # back to keywords, the raw question, and any refinement proposed.
        queue = []
        if subject:
            queue.append(" ".join(subject))
        keyword_query = " ".join(terms)
        if keyword_query and keyword_query != question:
            queue.append(keyword_query)
        queue.append(question)

        # If the subject names a real article, read that page directly. Search
        # alone can rank a disambiguation page or a same-named novel above it.
        if subject and hasattr(self.retriever, "exact_passage"):
            direct = self.retriever.exact_passage(" ".join(subject))
            if direct:
                queue.insert(0, None)  # marker: the direct page comes first
                direct_passages = [direct]

        step = 0
        while queue and step < self.max_steps:
            query = queue.pop(0)
            if query is None:  # the exact-title page we already fetched
                query = " ".join(subject)
                passages = list(direct_passages)
            else:
                query = query.strip()
                if not query or query in tried:
                    continue
                passages = self.retriever.search_passages(query, self.passages_per_step)
            tried.append(query)
            step += 1
            step_best = None
            for rank, item in enumerate(passages):
                if item["title"] in seen_titles:
                    continue
                seen_titles.add(item["title"])
                candidate, sentence_score = self._candidate_from_passage(
                    question, item["passage"], terms, qtype, subject, properties
                )
                if not candidate:
                    continue
                content_overlap, subject_match = self._evidence_score(
                    terms, item["passage"], subject, item["title"]
                )
                # Confidence is a weighted blend, deliberately not reaching 1.0
                # unless everything lines up: it ranks candidates and gates
                # "verified", so a perfect 1.0 should be rare.
                rank_bonus = 0.05 * (1.0 - rank / max(1, self.passages_per_step))
                confidence = (
                    0.35 * subject_match
                    + 0.25 * content_overlap
                    + 0.30 * min(1.0, max(0.0, sentence_score))
                    + rank_bonus
                )
                confidence = round(min(0.99, confidence), 3)
                record = {
                    "title": item["title"], "passage": item["passage"], "answer": candidate,
                    "content_overlap": round(content_overlap, 3),
                    "subject_match": round(subject_match, 3),
                    "sentence_score": round(sentence_score, 3), "confidence": round(confidence, 3),
                }
                if step_best is None or confidence > step_best["confidence"]:
                    step_best = record
            trace.append({"step": step, "query": query, "results": len(passages),
                          "best": step_best["title"] if step_best else None,
                          "confidence": step_best["confidence"] if step_best else 0.0})
            improved = step_best is not None and (best is None or step_best["confidence"] > best["confidence"])
            if improved:
                best = step_best
                stall = 0
            else:
                stall += 1
            if best and best["confidence"] >= self.good_enough:
                break
            if stall >= 2:  # refinement is not helping; stop rather than guess
                break
            refined = self._refine(question, terms, subject, tried, best, [item["title"] for item in passages])
            if refined:
                queue.append(refined)

        if best is None:
            return {"answer": None, "confidence": 0.0, "source": None, "grounded_passage": None,
                    "steps": trace, "verified": False}

        answer_text = best["answer"]
        short = extract_short_answer(best["answer"], qtype, question)
        if short:
            answer_text = short
        rephrased = False
        if self.generator is not None:
            generated = self.generator(question, best["passage"])
            if generated and len(generated.split()) >= 3:
                answer_text, rephrased = generated, True

        verified = best["confidence"] >= self.good_enough
        return {
            "answer": answer_text,
            "supporting_sentence": best["answer"],
            "confidence": best["confidence"],
            "source": best["title"],
            "grounded_passage": best["passage"][:900],
            "rephrased_by_model": rephrased,
            "steps": trace,
            "verified": verified,
        }