"""Typed inputs for Fabella story generation."""

from dataclasses import dataclass


@dataclass
class StoryRequest:
    name: str
    age: int
    themes: list[str]
    moral: str
    length: str
    seed: int = 0
