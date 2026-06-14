"""Structured judge for Fabella explanations.

The judge task is bounded: one rubric, one draft, one structured verdict.
No tools, no agent loop, no state machine. So we use a single LLM call
plus Pydantic validation, with one repair retry on failure.

The drafter stays on LangGraph (ReAct, validate_explanation tool, revise
loop). The judge is its own concern.
"""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from safety import age_bucket
from schema import JudgeFailed, JudgeVerdict


SYSTEM_PROMPT = """You are a strict editor reviewing an explanation a parent will
read aloud to their child.

You will receive a draft (in the Opener / Body / Closer / optional
'If they ask more' shape) and a rubric.

You must respond with EXACTLY ONE JSON object, with this exact schema:

{
  "ok": true|false,                    // true if the draft is good enough
  "issues": ["...", ...],             // concrete problems; [] if ok=true
  "score": 0.0,                       // 0.0..1.0, >=0.8 = approve-worthy
  "verdict": "approve" | "revise",     // your recommendation
  "reasoning": "..."                  // one short sentence (max ~30 words)
}

Hard rules:
- Output ONLY that JSON object. No prose. No markdown. No code fences.
- "ok" and "verdict" must agree: if ok=true then verdict="approve";
  if ok=false then verdict="revise".
- Every issue must be a CONCRETE, ACTIONABLE change. "Make it better"
  is not an issue. "Body is too short (39 words, target 60-130)" is.
- The score is a float in [0.0, 1.0]. Use the full range. 1.0 means
  the draft is exemplary; 0.0 means it's broken or off-topic.
- Reasoning must be one short sentence, max ~30 words.
"""


REPAIR_PROMPT = """Your previous response was not parseable as the required JSON
object. The expected schema is:

{{ok, issues, score, verdict, reasoning}}

Repair instructions:
- Return ONLY one JSON object. No prose, no markdown, no code fences.
- "ok" and "verdict" must agree.
- "score" is a number in [0.0, 1.0].
- "issues" is a list of strings (empty if ok=true).
- "reasoning" is a short single sentence.

Here was your last response (unparseable):

{last}

Now respond again with ONLY the JSON object."""


def _build_rubric(req_age: int, req_tone: str, child_name: str, situation: str) -> str:
    bucket = age_bucket(req_age)
    vocab = {
        "young": "very simple sentences (under 12 words each), short paragraphs, no abstract or figurative language",
        "middle": "clear sentences, paragraphs of 3-5 sentences, concrete metaphors are fine",
        "older": "richer vocabulary and slightly longer paragraphs are fine, but keep it direct",
    }[bucket]
    name_hint = (
        f"The child's name is '{child_name}'. Use it naturally once."
        if child_name else "No name was given. Address the parent ('your child') or use 'you'."
    )
    return (
        f"The child is {req_age} years old ({bucket} reader). Target vocabulary: {vocab}.\n"
        f"The tone is {req_tone}.\n"
        f"{name_hint}\n"
        f"The parent's situation (for context, not for inclusion in the explanation):\n"
        f"  {situation}\n\n"
        f"Evaluate the draft against this rubric:\n"
        f"1. The Opener, Body, and Closer are all present and clearly labelled.\n"
        f"2. The body addresses the child's likely feeling in the first paragraph.\n"
        f"3. The body explains the situation in concrete, age-appropriate terms. No abstract or vague language.\n"
        f"4. The body is roughly 60-130 words. It does NOT lecture or moralize.\n"
        f"5. The closer feels like something a real parent would say. Not therapist-speak. Not corporate.\n"
        f"6. No scary imagery, no threats, no vivid descriptions of harm. The explanation does not invent facts."
    )


def _extract_json(text: str) -> str | None:
    """Pull the first {...} block from the model's text output.

    Tolerant of:
    - Leading prose like "We need to evaluate..."
    - Markdown code fences
    - Pretty-printed JSON with newlines

    Returns the candidate JSON string, or None if no brace block is found.
    """
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```\s*$", "", t)
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j < 0 or j <= i:
        return None
    return t[i : j + 1]


def judge_explanation(
    llm,
    draft: str,
    req_age: int,
    req_tone: str,
    child_name: str,
    situation: str,
) -> JudgeVerdict:
    """Ask the small judge model for a structured verdict on `draft`.

    Tries once, then once more with a repair prompt if the first response
    doesn't validate. Raises `JudgeFailed` if both attempts fail; the
    caller is expected to fall back to the rule-based check.
    """
    rubric = _build_rubric(req_age, req_tone, child_name, situation)
    user = (
        f"Draft to evaluate:\n{draft}\n\nRubric:\n{rubric}\n\n"
        f"Respond with ONLY the JSON object, no prose."
    )
    last_text = ""
    for attempt in (1, 2):
        if attempt == 1:
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user),
            ]
        else:
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=REPAIR_PROMPT.format(last=last_text)),
            ]
        try:
            resp = llm.invoke(messages)
        except Exception as e:
            if attempt == 2:
                raise JudgeFailed(f"judge LLM call failed: {e}", last_text=last_text) from e
            continue

        last_text = (resp.content if isinstance(resp.content, str) else str(resp.content)).strip()
        candidate = _extract_json(last_text)
        if candidate is None:
            continue
        try:
            verdict = JudgeVerdict.model_validate_json(candidate)
        except Exception:
            continue

        # Cross-field consistency: ok and verdict must agree.
        if verdict.ok and verdict.verdict != "approve":
            verdict = verdict.model_copy(update={"verdict": "approve"})
        elif (not verdict.ok) and verdict.verdict != "revise":
            verdict = verdict.model_copy(update={"verdict": "revise"})

        return verdict

    raise JudgeFailed("judge output was not parseable JSON", last_text=last_text)
