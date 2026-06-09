"""Dispatcher: routes to mock or real based on USE_MOCK env var."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mock import mock_story
from schema import StoryRequest


def _truthy(val: str | None, default: bool = True) -> bool:
    if val is None:
        return default
    return val.strip().lower() not in ("0", "false", "no", "off", "")


USE_MOCK = _truthy(os.environ.get("USE_MOCK"), default=True)


def generate_story(req: StoryRequest) -> tuple[str, str, str]:
    """Returns (title, body, mode_badge_text)."""
    if USE_MOCK:
        title, body = mock_story(req.name, req.age, req.themes, req.moral, req.length, req.seed)
        return title, body, "Mock mode"

    from real import generate_real

    title, body = generate_real(req)
    return title, body, "Real model"
