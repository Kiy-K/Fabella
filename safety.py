"""Input sanitization and content safety for Fabella."""

import re

MAX_NAME_LEN = 30
MAX_THEMES = 3
MAX_THEME_LEN = 20
MAX_MORAL_LEN = 120

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


def sanitize_themes(raw) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, str):
        items = [raw]
    else:
        items = list(raw)
    out = []
    seen = set()
    for t in items:
        t = clean_text(str(t), MAX_THEME_LEN)
        if not t:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
        if len(out) >= MAX_THEMES:
            break
    return out


def sanitize_moral(raw: str) -> str:
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
    """Return (min_words, max_words) for a length label."""
    return {
        "short": (120, 220),
        "medium": (280, 420),
        "long": (500, 800),
    }.get(length, (280, 420))
