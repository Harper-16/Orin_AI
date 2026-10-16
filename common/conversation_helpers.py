#!/usr/bin/env python3
"""Small offline helpers for more natural everyday conversation around the scratch LM."""

import random
import re


def _clean(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[.!?]+$", "", text)
    return re.sub(r"\s+", " ", text)


_LAST_REPLY_BY_OPTIONS = {}


def _pick(options):
    options = tuple(options)
    if len(options) < 2:
        return options[0]
    available = [reply for reply in options if reply != _LAST_REPLY_BY_OPTIONS.get(options)]
    reply = random.choice(available or options)
    _LAST_REPLY_BY_OPTIONS[options] = reply
    return reply


def handle_conversation(text: str, remembered_name: str | None = None):
    """Return (reply, updated_name); return (None, name) for LM handling."""
    clean = _clean(text)

    if re.search(r"\bforget (?:my|the) name\b|\bclear my name\b", clean):
        return "Okay, I’ve forgotten your name.", None

    name_match = re.fullmatch(r"(?:my name is|call me)\s+([a-z][a-z'-]{0,29})", clean)
    if name_match:
        name = name_match.group(1).capitalize()
        return _pick((
            f"Nice to meet you, {name}.",
            f"Good to meet you, {name}!",
            f"Hey, {name}. Got it.",
        )), name

    if re.fullmatch(r"(?:what is|what's|do you (?:know|remember)|can you remember) my name", clean):
        if remembered_name:
            return _pick((
                f"You told me your name is {remembered_name}.",
                f"I remember—you said your name is {remembered_name}.",
                f"It’s {remembered_name}, right?",
            )), remembered_name
        return _pick((
            "I don’t think you’ve told me yet. What should I call you?",
            "I don’t know your name so far—what do you go by?",
        )), None

    if clean in {"what are you", "who are you", "what is this ai", "what are you exactly"}:
        return (
            "I’m the little AI you’re building—trained from scratch to chat and answer questions. "
            "I’m still learning, so I can get things wrong.", remembered_name
        )
    if clean in {"what can you do", "what do you do"}:
        return _pick((
            "I can chat, keep track of what you tell me in this conversation, and try to answer questions. What are you in the mood for?",
            "Mostly chat and help with questions. I’m still learning, though.",
            "I can talk things through with you and try to answer questions. I’m not great at everything yet.",
        )), remembered_name
    if clean in {"what is your name", "what's your name", "what should i call you"}:
        return "You can call me your AI assistant.", remembered_name

    greetings = {
        "hi": ("Hey!", "Hi there.", "Hey, how’s it going?", "Hi! Good to see you."),
        "hey": ("Hey!", "Hey, what’s up?", "Oh hey!", "Hey there."),
        "hello": ("Hello!", "Hey there.", "Hi, how’s your day going?"),
        "hiya": ("Hiya!", "Hey!", "Hi there."),
        "yo": ("Hey!", "Yo, what’s up?", "Hey there."),
        "good morning": ("Morning! How’s it going?", "Good morning!", "Hey, morning."),
        "good afternoon": ("Good afternoon!", "Hey, how’s your day going?"),
        "good evening": ("Evening! How are you?", "Good evening!", "Hey, how’s it going?"),
    }
    greeting = re.fullmatch(r"(hi|hey|hello|hiya|yo)(?: there)?", clean)
    greeting_key = greeting.group(1) if greeting else clean
    if greeting_key in greetings:
        return _pick(greetings[greeting_key]), remembered_name

    if clean in {"how are you", "how are you doing"}:
        return _pick((
            "I’m here and ready to chat. How about you?",
            "Doing fine over here—what’s up with you?",
            "All good on my end. How’s your day going?",
        )), remembered_name
    if clean in {"are you there", "are you still there"}:
        return _pick(("Yep, I’m here.", "I’m here!", "Still here. What’s up?")), remembered_name
    if clean in {"can i ask you something", "can i ask something"}:
        return _pick(("Sure, go ahead.", "Of course—what’s up?", "Yeah, ask away.")), remembered_name
    if clean in {"can you help me", "i need help"}:
        return _pick(("I’ll try. What’s going on?", "Sure—what do you need?", "Yeah, tell me what you’re stuck on.")), remembered_name
    if clean in {"thanks", "thank you", "thanks a lot"}:
        return _pick(("You’re welcome!", "Anytime.", "No problem—glad I could help.")), remembered_name
    if clean in {"bye", "goodbye", "see you", "see you later"}:
        return _pick(("See you!", "Bye—take care.", "Catch you later!")), remembered_name
    if clean in {"tell me more", "what do you mean", "i do not understand", "i don't understand"}:
        return _pick((
            "Sure. Which bit should I explain differently?",
            "No worries—what part lost you?",
            "I might’ve explained that badly. What should I clear up?",
        )), remembered_name

    if re.search(r"\b(feel|feeling|am|i'm)\s+(?:really\s+)?(?:sad|down|upset|lonely)\b", clean):
        return _pick((
            "I’m sorry. Sounds like today’s been rough.",
            "Aw, that’s hard. I’m here if you want to talk about it.",
            "Oof, I’m sorry you’re feeling like that.",
            "That sounds rough. Want to tell me what happened?",
        )), remembered_name
    if re.search(r"\b(feel|feeling|am|i'm)\s+(?:really\s+)?(?:happy|excited|great)\b", clean):
        return _pick((
            "Oh, nice! Glad to hear it.",
            "That’s great—love that for you.",
            "Aw, that’s lovely.",
            "Nice! What made your day?",
        )), remembered_name
    if re.search(r"\b(?:bored|have nothing to do)\b", clean):
        return _pick((
            "We could play a quick word game, if you’re up for it.",
            "Want a joke, a game, or just a random topic?",
            "I can think of something fun to do with you. Any preference?",
        )), remembered_name
    if re.search(r"\b(?:nervous|worried|anxious)\b", clean):
        return _pick((
            "That can be a lot to sit with. What’s going on?",
            "I’m sorry you’re dealing with that. Want to talk it through?",
            "Oof, nerves are no fun. What’s making you feel that way?",
        )), remembered_name
    if clean in {"i had a long day", "i had a bad day"}:
        return _pick((
            "Oof, sounds like a tiring one.",
            "I’m sorry. Hope you get a chance to unwind.",
            "That’s rough. I’m here if you want to vent.",
        )), remembered_name
    if clean == "i made a mistake":
        return _pick(("It happens. Try not to be too hard on yourself.", "Ah, that’s frustrating. Want to talk about it?")), remembered_name
    if clean in {"i want to talk", "i need someone to listen"}:
        return _pick(("Sure. I’m listening.", "Go ahead—I’m here.", "I’m with you. What’s on your mind?")), remembered_name
    if clean in {"i do not know what to say", "i don't know what to say"}:
        return _pick(("No pressure. We can start anywhere.", "That’s okay—no need to think of anything clever.", "We can just take it easy. How’s your day been?")), remembered_name
    if clean in {"what should we talk about", "what should we talk about?"}:
        return _pick(("Anything you like—your day, games, music, or whatever’s on your mind.", "I’m easy. What have you been into lately?", "We could talk about something you enjoy. What comes to mind?")), remembered_name
    if clean in {"that was helpful", "that helped"}:
        return _pick(("Glad it helped.", "Nice, happy that made sense.", "Good! Anything else on that?")), remembered_name

    like_match = re.fullmatch(r"i (?:like|love) (.+)", clean)
    if like_match:
        topic = like_match.group(1)
        if "tea" in topic:
            return _pick(("Tea’s nice. What kind do you like?", "Ooh, tea—good choice.", "Nice. Are you a mint tea person or something else?")), remembered_name
        if "music" in topic:
            return _pick(("Nice! What kind of music are you into?", "Good choice. What have you been listening to lately?", "Oh cool—any favorite artists?")), remembered_name
        if "draw" in topic:
            return _pick(("That sounds fun. What do you like drawing?", "Oh nice—what have you been drawing lately?", "Cool. What do you like most about drawing?")), remembered_name
        return _pick((f"Oh, nice! Tell me more about {topic}.", "Nice! What do you like most about it?", "Oh cool—that sounds like your kind of thing.")), remembered_name

    return None, remembered_name


def answer_arithmetic(question: str) -> str | None:
    """Answer small, explicit arithmetic questions without relying on the tiny LM."""
    match = re.search(
        r"(?<![\w.])(-?\d{1,4})\s*(plus|\+|minus|-|times|multiplied by|divided by|[×x/])\s*(-?\d{1,4})(?![\w.])",
        question.lower(),
    )
    if not match:
        return None
    left, operator, right = match.groups()
    a, b = int(left), int(right)
    if operator in {"plus", "+"}:
        result, symbol = a + b, "+"
    elif operator in {"minus", "-"}:
        result, symbol = a - b, "-"
    elif operator in {"times", "multiplied by", "x", "×"}:
        result, symbol = a * b, "×"
    else:
        if b == 0:
            return "I can’t divide by zero."
        result = a / b
        result = int(result) if result.is_integer() else round(result, 6)
        symbol = "÷"
    return f"{a} {symbol} {b} = {result}."


def soften_unrelated_question_reply(question: str, answer: str) -> str:
    """Replace obvious off-topic replies to information questions with uncertainty."""
    clean_answer = _clean(answer)
    clean_question = _clean(question)
    question_like = "?" in question or re.match(
        r"^(who|what|when|where|why|how|is|are|can|could|do|does|did|will|would)\b",
        clean_question,
    )
    asks_name = bool(re.search(r"\b(?:my name|name should i)\b", clean_question))
    unrelated_starts = (
        "you're welcome", "you’re welcome", "no. i'm glad", "no, i'm glad",
        "goodbye", "i'm sorry", "i’m sorry", "no. i can answer",
    )
    unrelated_name_memory = "you told me your name" in clean_answer and not asks_name

    # A long information answer with no words in common with its question is
    # usually a topic drift in this tiny model. Short direct facts stay intact.
    content_words = {
        word for word in re.findall(r"[a-z0-9]+", clean_question)
        if word not in {
            "what", "who", "when", "where", "why", "how", "is", "are", "was", "were",
            "do", "does", "did", "can", "could", "would", "will", "the", "a", "an",
            "to", "of", "in", "on", "for", "and", "or", "you", "your", "i", "me",
            "my", "it", "this", "that", "please", "tell", "explain",
        }
    }
    answer_words = set(re.findall(r"[a-z0-9]+", clean_answer))
    information_question = bool(re.match(r"^(what|who|when|where|why|how)\b", clean_question))
    unrelated_by_words = (
        information_question and len(answer_words) > 2 and content_words
        and not content_words.intersection(answer_words)
    )
    if question_like and (
        clean_answer.startswith(unrelated_starts)
        or unrelated_name_memory
        or unrelated_by_words
    ):
        return "I’m not sure yet. I’m still learning, so I may need more information to answer that well."
    return answer
