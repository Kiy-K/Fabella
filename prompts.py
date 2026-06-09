"""Prompt builders for the real model path."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from safety import age_bucket, length_to_words


def system_prompt() -> str:
    return (
        "You are Fabella, a gentle storyteller for children aged 6 to 10. "
        "You write warm, magical, age-appropriate short stories. "
        "Never include scary content for children under 8. "
        "Never mention real public figures by name. "
        "Never include violence beyond gentle peril. "
        "Never include romantic content. "
        "Always end with a clear resolution that reflects the moral lesson. "
        "Always begin your response with a line in the exact format: Title: <the story title> "
        "Then a blank line, then the story body. "
        "Do not include any other text, labels, or commentary."
    )


def user_prompt(name: str, age: int, themes: list[str], moral: str, length: str) -> str:
    bucket = age_bucket(age)
    min_w, max_w = length_to_words(length)
    themes_str = ", ".join(themes) if themes else "everyday adventures"
    moral_str = moral or "being kind to others"

    vocab = {
        "young": "Use very simple sentences. Use short paragraphs. Keep language concrete and warm.",
        "middle": "Use clear sentences with some descriptive language. Paragraphs of 3 to 5 sentences.",
        "older": "Use richer vocabulary and longer paragraphs. Keep the tone gentle and kind.",
    }[bucket]

    return (
        f"Write a personalized children's story.\n"
        f"Child's name: {name}\n"
        f"Age: {age}\n"
        f"Favorite themes: {themes_str}\n"
        f"Moral lesson: {moral_str}\n"
        f"Story length: about {min_w} to {max_w} words.\n"
        f"{vocab}\n"
        f"The child's name must appear at least once. "
        f"Each favorite theme must appear at least once. "
        f"The moral must be clearly resolved in the ending."
    )
