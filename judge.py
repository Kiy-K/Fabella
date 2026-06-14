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
from typing import Any

from openai import OpenAI

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
    return (
        f"The child is {req_age} years old ({bucket} reader). Target vocabulary: {vocab}.\n"
        f"The tone is {req_tone}.\n"
        f"Address the child as 'you'. Never invent or use a name.\n"
        f"The parent's situation (for context, not for inclusion in the explanation):\n"
        f"  {situation}\n\n"
        f"Evaluate the draft against this rubric:\n"
        f"1. The JSON object has 'opener', 'body', 'closer', and (optionally) 'followup', all non-empty.\n"
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


def _normalize_verdict(verdict: JudgeVerdict) -> JudgeVerdict:
    """Keep model output internally consistent after Pydantic validation."""
    if verdict.ok and verdict.verdict != "approve":
        return verdict.model_copy(update={"verdict": "approve"})
    if (not verdict.ok) and verdict.verdict != "revise":
        return verdict.model_copy(update={"verdict": "revise"})
    return verdict


def _openai_client_for(llm: Any) -> tuple[OpenAI, str] | None:
    base_url = getattr(llm, "base_url", None)
    model_name = getattr(llm, "model_name", None)
    if not base_url or not model_name:
        return None
    return OpenAI(base_url=f"{base_url}/v1", api_key="EMPTY"), model_name


def _direct_structured_verdict(llm: Any, user: str) -> JudgeVerdict | None:
    """Use vLLM's OpenAI-compatible structured output when available.

    This keeps the judge bounded and Pydantic-owned instead of turning it
    into another LangChain agent/tool loop. Some vLLM builds or models may
    reject `response_format`; callers fall back to prompt-only JSON parsing.
    """
    client_model = _openai_client_for(llm)
    if client_model is None:
        return None
    client, model_name = client_model

    schema = JudgeVerdict.model_json_schema()
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "judge_verdict",
                "schema": schema,
                "strict": True,
            },
        },
    )
    text = (response.choices[0].message.content or "").strip()
    if not text:
        return None
    return JudgeVerdict.model_validate_json(text)


def _direct_json_verdict(llm: Any, user: str, repair_from: str = "") -> tuple[JudgeVerdict | None, str]:
    """Prompt for JSON directly through OpenAI client, then validate with Pydantic."""
    client_model = _openai_client_for(llm)
    if client_model is None:
        return None, ""
    client, model_name = client_model
    prompt = user if not repair_from else REPAIR_PROMPT.format(last=repair_from)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
    )
    text = (response.choices[0].message.content or "").strip()
    candidate = _extract_json(text)
    if candidate is None:
        return None, text
    return JudgeVerdict.model_validate_json(candidate), text


def judge_explanation(
    llm,
    draft,
    req_age: int,
    req_tone: str,
    child_name: str,
    situation: str,
) -> JudgeVerdict:
    """Ask the small judge model for a structured verdict on `draft`.

    `draft` may be a JSON object, a JSON string, or a labeled draft text.
    The judge is told to verify the JSON shape (opener/body/closer/followup).
    Tries once with structured output, then once with prompt-only JSON plus
    a repair retry. Raises `JudgeFailed` if both attempts fail; the caller
    is expected to fall back to the rule-based check.
    """
    if isinstance(draft, (dict, list)):
        draft_text = json.dumps(draft, ensure_ascii=False)
    else:
        draft_text = str(draft or "")
    rubric = _build_rubric(req_age, req_tone, child_name, situation)
    user = (
        f"Draft to evaluate (JSON object with 'opener', 'body', 'closer', 'followup'):\n"
        f"{draft_text}\n\n"
        f"Rubric:\n{rubric}\n\n"
        f"Respond with ONLY the JSON object, no prose."
    )

    # First choice: vLLM/OpenAI structured output constrained by the Pydantic
    # JSON schema. This bypasses LangChain entirely for the judge path.
    if _openai_client_for(llm) is not None:
        try:
            structured = _direct_structured_verdict(llm, user)
            if structured is not None:
                return _normalize_verdict(structured)
        except Exception as e:
            print(f"[judge] structured response_format failed; falling back: {type(e).__name__}: {e}", flush=True)

        # Second choice: same direct OpenAI client, but prompt-only JSON plus
        # Pydantic validation and one repair retry. This is still not LangChain.
        last_text = ""
        for attempt in (1, 2):
            try:
                verdict, last_text = _direct_json_verdict(
                    llm,
                    user,
                    repair_from=last_text if attempt == 2 else "",
                )
            except Exception as e:
                if attempt == 2:
                    raise JudgeFailed(f"judge direct JSON call failed: {e}", last_text=last_text) from e
                continue
            if verdict is not None:
                return _normalize_verdict(verdict)

        raise JudgeFailed("judge output was not parseable JSON", last_text=last_text)

    raise JudgeFailed(
        "judge requires an OpenAI-compatible llm with base_url and model_name",
        last_text="",
    )
