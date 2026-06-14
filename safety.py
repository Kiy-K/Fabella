"""Input sanitization and content safety for Fabella."""

import re

MAX_NAME_LEN = 30
MAX_SITUATION_LEN = 600     # a few sentences of context from the parent
MAX_MORAL_LEN = 120          # kept for any lingering legacy call sites

CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
PROFANITY = {
    "damn", "hell", "shit", "fuck", "bitch", "asshole",
    "bastard", "crap", "dick", "piss",
}


def clean_text(value: str, max_len: int) -> str:
    """Strip control chars, collapse whitespace, trim, cap length."""
    if not value:
        return ""
    value = CONTROL_CHARS.sub("", str(value))
    value = re.sub(r"\s+", " ", value).strip()
    return value[:max_len]


def sanitize_name(raw: str) -> str:
    return clean_text(raw, MAX_NAME_LEN)


def sanitize_situation(raw: str) -> str:
    """The freeform situation the parent describes."""
    return clean_text(raw, MAX_SITUATION_LEN)


def sanitize_moral(raw: str) -> str:
    """Kept for legacy call sites. No longer used in the new explainer."""
    return clean_text(raw, MAX_MORAL_LEN)


def has_profanity(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    tokens = re.findall(r"[a-z']+", lowered)
    return any(tok in PROFANITY for tok in tokens)


def age_bucket(age: int) -> str:
    if age <= 7:
        return "young"
    if age <= 9:
        return "middle"
    return "older"


def length_to_words(length: str) -> tuple[int, int]:
    """Return (min_words, max_words) for a length label.

    Kept for legacy call sites. The new explainer uses its own
    `explain_to_words()` mapping by tone.
    """
    return {
        "short": (120, 220),
        "medium": (280, 420),
        "long": (500, 800),
    }.get(length, (280, 420))


def explain_to_words(tone: str) -> tuple[int, int]:
    """Target word count (min, max) for the explanation body, by tone.

    Explanations are deliberately short — 60-160 words depending on
    tone. Parents skim these before reading aloud; long is a bug.
    """
    return {
        "gentle": (60, 110),
        "matter-of-fact": (70, 130),
        "playful": (50, 100),
    }.get(tone, (60, 110))
