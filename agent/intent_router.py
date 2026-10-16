#!/usr/bin/env python3
"""Decide whether a turn needs a Wikipedia search at all.

Searching on every sentence that ends in "?" is wrong in both directions: it
wastes time on small talk, and it searches for things Wikipedia can never
answer. This module classifies a turn into one of four kinds:

    CHAT      the model should just talk (greetings, opinions, meta questions)
    FOLLOWUP  a continuation of the last answer — reuse the evidence we have
    SEARCH    a real-world fact question worth looking up
    UNKNOWN   a question that needs live data we do not have (time, weather,
              news) — answer honestly instead of searching

The router is intentionally explicit and explainable: :func:`classify` returns
the reason for its decision so the chat program can show *why* it did not
search, and rules can be extended in one place.
"""

from __future__ import annotations

import re


class Intent:
    CHAT = "chat"
    FOLLOWUP = "followup"
    SEARCH = "search"
    UNKNOWN = "unknown"


def _matches(text: str, pattern: str) -> bool:
    return re.search(pattern, text, re.I) is not None


# --- small talk, feelings, and the assistant talking about itself -----------
_CHAT_PATTERNS = [
    r"^\s*(hi|hey|hello|yo|hiya|sup|howdy)\b",
    r"^\s*(good\s+(morning|afternoon|evening|night))\b",
    r"^\s*(how\s+are\s+you|how're\s+you|hows\s+it\s+going|how's\s+it\s+going|whats\s+up|what's\s+up)\b",
    r"^\s*(thanks|thank\s+you|cheers|ta|nice|cool|awesome|great|ok|okay|sure|yes|yeah|no|nope)\b[\s!.]*$",
    r"^\s*(bye|goodbye|see\s+you|later|good\s+night|goodnight|ttyl)\b",
    r"\b(who|what)\s+are\s+you\b",
    r"\bwhat\s+(can|do)\s+you\s+(do|help)\b",
    r"\bcan\s+you\s+help\b",
    r"\bhow\s+(do\s+you|does\s+this)\s+work\b",
    r"\bwhat\s+do\s+you\s+(think|feel|know|remember|mean)\b",
    r"\bdo\s+you\s+(like|feel|think|remember|know|have)\b",
    r"\bwhat.*your\s+(name|favourite|favorite|opinion)\b",
    r"^\s*(tell|write|give)\s+me\s+(a|an|me)?\s*(joke|poem|story|song|riddle|haiku)",
    r"^\s*(say|sing|act|pretend|imagine)\b",
    r"\b(my\s+name|what\s+did\s+i\s+say|do\s+you\s+know\s+me|remember\s+(me|what))\b",
    r"\b(are\s+you\s+(a\s+)?(real|alive|human|bot|robot|ai)|are\s+you\s+stupid)\b",
]

# --- questions that need live data Wikipedia does not have ------------------
_UNKNOWN_PATTERNS = [
    r"\b(what('s| is)\s+the\s+(time|date|weather|temperature|day))\b",
    r"\bwhat\s+(time|date)\b",
    r"\b(what('s| is)\s+the\s+weather)\b",
    r"\b(weather|forecast|temperature)\b",
    r"\b(today('s)?|right\s+now|currently|these\s+days|this\s+week)\b",
    r"\b(news|headlines|latest)\b",
    r"\b(who|what)\s+(won|win)\b",
    r"\b(price|stock|share\s+price|exchange\s+rate|score|standings)\b",
    r"\b(what('s| is)\s+the\s+(current|latest|president|prime\s+minister))\b",
    r"\b(are\s+you\s+online|is\s+the\s+internet\s+working)\b",
]

# --- continuations of the previous answer: reuse the evidence we have ------
_FOLLOWUP_PATTERNS = [
    r"^\s*(why|how)\s*\??$",
    r"^\s*(and|so|but|also|then|really|ok|okay)\b.*\??$",
    r"^\s*(tell\s+me\s+more|more|go\s+on|continue|elaborate|explain\s+more)\b",
    r"^\s*what\s+about\b",
    r"^\s*(who|what|when|where)\s+(else|about\s+that|about\s+it)\b",
    r"^\s*(is|was|are|were|does|do|did|can|has|have)\s+(it|that|this|he|she|they)\b",
    r"^\s*\?+\s*$",
    r"^\s*(hmm+|ah+|oh+|i\s+see)\b",
]

# --- an explicit instruction to search, overriding everything ---------------
_FORCE_SEARCH = [
    r"^\s*/search\b",
    r"\b(search|look\s+up|google|wiki|find\s+out)\b.*\b(for|about|on)\b",
    r"\baccording\s+to\s+(wikipedia|the\s+internet)\b",
    r"^\s*find\s+(me\s+)?(information|info|details)\b",
]

# --- an explicit instruction NOT to search ---------------------------------
_FORCE_CHAT = [
    r"^\s*/nosearch\b",
    r"\b(don't|do\s+not|never|no\s+need\s+to)\s+(search|look\s+up|google|check\s+online)\b",
    r"\bwithout\s+(searching|looking\s+it\s+up|going\s+online)\b",
    r"^\s*/chat\b",
]

# A turn worth looking up: an interrogative with some substance behind it.
_SUBSTANTIVE = re.compile(
    r"\b(what|who|when|where|which|why|how|name|define|list|explain)\b", re.I
)

# Noun phrases that name a fact even without a question word, so "the capital
# of Japan" is recognised as a lookup rather than a remark.
_PROPERTY = re.compile(
    r"\b(capital|population|president|king|queen|river|mountain|planet|ocean|"
    r"author|writer|composer|director|founder|inventor|language|currency|"
    r"element|scientist|actor|artist|album|novel|species|speed|height|age)\b",
    re.I,
)
_PROPER_NOUN = re.compile(r"\b[A-Z][a-z]{2,}\b")


def classify(text: str, have_evidence: bool = False) -> tuple[str, str]:
    """Return ``(intent, reason)`` for a user turn.

    ``have_evidence`` is True when a previous turn already retrieved a passage;
    that turns bare continuations ("why?", "tell me more") into FOLLOWUP instead
    of a pointless new search.
    """
    stripped = text.strip()
    if not stripped:
        return Intent.CHAT, "empty message"

    for pattern in _FORCE_CHAT:
        if _matches(stripped, pattern):
            return Intent.CHAT, "you asked me not to search"

    for pattern in _FORCE_SEARCH:
        if _matches(stripped, pattern):
            return Intent.SEARCH, "you asked me to look it up"

    for pattern in _UNKNOWN_PATTERNS:
        if _matches(stripped, pattern):
            return Intent.UNKNOWN, "this needs live data Wikipedia does not have"

    for pattern in _FOLLOWUP_PATTERNS:
        if _matches(stripped, pattern):
            if have_evidence:
                return Intent.FOLLOWUP, "a follow-up, so I reuse the evidence I already have"
            return Intent.CHAT, "too short to search on its own"

    for pattern in _CHAT_PATTERNS:
        if _matches(stripped, pattern):
            return Intent.CHAT, "small talk or a question about me, not a fact to look up"

    if not _SUBSTANTIVE.search(stripped):
        # No question word, but a fact-shaped noun phrase about a named thing
        # is still a lookup: "the capital of Japan".
        if _PROPERTY.search(stripped) and _PROPER_NOUN.search(stripped):
            return Intent.SEARCH, "a fact about a named thing, even without a question mark"
        return Intent.CHAT, "not a question, so I just talk"

    # A bare word or short phrase is not worth a search.
    if len(stripped.split()) < 3 and not stripped.endswith("?"):
        return Intent.CHAT, "too little to search meaningfully"

    if stripped.endswith("?"):
        return Intent.SEARCH, "a factual question worth looking up"

    # Something interrogative but not phrased as a question ("the capital of France").
    return Intent.SEARCH, "asking about a real-world fact"
