"""Typed inputs and outputs for Fabella — the small-words-for-big-questions tool."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, field_validator


@dataclass
class ExplainRequest:
    """What a parent tells Fabella to explain to their child.

    Attributes:
        situation: A 1-3 sentence freeform description of the situation the
            parent needs help explaining ("We're moving to a new house in
            3 weeks", "Why is grandma in the hospital?").
        age: The child's age in years (5-12 range supported).
        child_name: Optional name. If set, the explanation addresses the
            child directly. If empty, the parent is addressed ("your child").
        tone: "gentle" | "matter-of-fact" | "playful". Controls the
            register of the explanation.
        seed: Determinism for the drafter (the judge is temperature=0).
    """

    situation: str
    age: int
    child_name: str = ""
    tone: str = "gentle"
    seed: int = 0


# --- Judge output ---------------------------------------------------------


Verdict = Literal["approve", "revise"]


class JudgeVerdict(BaseModel):
    """Structured output of the small Nemotron judge.

    The judge receives a draft explanation and a 6-criterion rubric, and
    returns one of these. Validated by Pydantic so that any deviation
    from the schema is caught immediately.
    """

    ok: bool = Field(
        description="True iff the draft is good enough to ship as-is."
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Concrete, actionable problems with the draft. Empty if ok=true.",
    )
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="0..1 quality score. >=0.8 is generally approve-worthy.",
    )
    verdict: Verdict = Field(
        description='"approve" if the draft is ready, "revise" if the drafter should rewrite.'
    )
    reasoning: str = Field(
        default="",
        description="One short sentence explaining the verdict.",
    )

    @field_validator("issues")
    @classmethod
    def _issues_short(cls, v: list[str]) -> list[str]:
        # Strip and bound each issue to 200 chars so a runaway model can't
        # blow up the drafter's context window.
        return [str(i).strip()[:200] for i in v if str(i).strip()]

    @field_validator("reasoning")
    @classmethod
    def _reasoning_short(cls, v: str) -> str:
        return (v or "").strip()[:300]


class JudgeFailed(Exception):
    """Raised when the judge output cannot be parsed after a retry.

    The caller (the validate_explanation tool) should fall back to the
    rule-based check.
    """

    def __init__(self, message: str, last_text: str = ""):
        super().__init__(message)
        self.last_text = last_text
