"""Fabella — small words for big questions.

The agent has one tool, validate_explanation, which sends a draft to the
small Nemotron judge for a multi-criteria review. The ReAct loop:

1. Read the parent's situation and the child's age.
2. Draft a short, kind, concrete explanation. Call validate_explanation.
3. If OK, jump to "end" (we extract the draft from the last tool-call args).
4. If issues, model revises and calls validate_explanation again.
5. After at most 2 tool calls, force a final answer.
"""

import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain.tools import tool
from langgraph.runtime import Runtime

from safety import age_bucket, explain_to_words
from schema import ExplainRequest


SYSTEM_PROMPT = """You are Fabella, a kind helper for parents who need to explain
hard things to their child. A parent has just told you about a real
situation they're facing, and you've written a short, gentle
explanation the parent can read aloud.

Output shape (always exactly one JSON object, no markdown, no extra
prose, no code fences):

{
  "opener": "<one short sentence the parent can say to start the conversation>",
  "body": "<1-3 short paragraphs, written in the second person ('you' / 'your child'), concrete and warm. About 60-130 words total for the body.>",
  "closer": "<one short sentence the parent can say to land the conversation>",
  "followup": "<one optional follow-up sentence the parent can use if the child has another question>"
}

Strict rules:
- Output a single JSON object, exactly those four keys.
- "followup" may be an empty string if there is nothing useful to add.
- Never use scary imagery, threats, or vivid descriptions of harm.
- Never moralize, sermonize, or lecture. ("You should always…")
- Never promise things that aren't true ("It'll all be fine" — only if
  the parent's situation actually supports that).
- Never invent facts. If you don't know, say "I don't know" or
  acknowledge the uncertainty in kid-appropriate language.
- Use the child's age to pick vocabulary and sentence length. For
  young kids (5-7), very short sentences, no abstract words. For
  older kids (8-12), you can be a little more direct.
- Address the child as "you". Do not invent or use a name.
- The opener and closer should sound like something a real parent
  would actually say out loud. Not therapist-speak. Not corporate.

Workflow:
1. Compose the four fields as a single JSON object.
2. Call the validate_explanation tool with that JSON object.
3. If the tool says "OK", output the same JSON object as your final
   answer. If the tool reports issues, revise and call
   validate_explanation again. After 2 tool calls, output your best
   JSON object anyway.
"""


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def _strip_prefix(line: str) -> str:
    """Strip a leading 'Opener:' / 'Body:' / etc. label from a line."""
    return re.sub(r"^(Opener|Body|Closer|If they ask more)\s*:\s*", "", line, flags=re.IGNORECASE).strip()


def _parse_sections(draft) -> dict[str, str]:
    """Parse the drafter's output into opener / body / closer / followup.

    Prefers a JSON object. Falls back to a labeled "Opener: / Body: / ..."
    draft for backward compatibility, and to a whole-string body if no
    labels are present.
    """
    out = {"opener": "", "body": "", "closer": "", "followup": ""}
    if draft is None:
        return out

    if isinstance(draft, dict):
        out["opener"] = str(draft.get("opener", "") or "").strip()
        out["body"] = str(draft.get("body", "") or "").strip()
        out["closer"] = str(draft.get("closer", "") or "").strip()
        out["followup"] = str(draft.get("followup", "") or "").strip()
        return out

    if isinstance(draft, str):
        s = draft.strip()
        # Try a leading JSON object first.
        obj: dict | None = None
        if s.startswith("{"):
            try:
                parsed = json.loads(s)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                obj = parsed
        if obj is None:
            m = re.search(r"\{[\s\S]*\}", s)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except Exception:
                    parsed = None
                if isinstance(parsed, dict):
                    obj = parsed
        if obj is not None and any(
            str(obj.get(k, "") or "").strip()
            for k in ("opener", "body", "closer", "followup")
        ):
            out["opener"] = str(obj.get("opener", "") or "").strip()
            out["body"] = str(obj.get("body", "") or "").strip()
            out["closer"] = str(obj.get("closer", "") or "").strip()
            out["followup"] = str(obj.get("followup", "") or "").strip()
            return out

        # Legacy labeled draft.
        label_re = re.compile(
            r"^(Opener|Body|Closer|If they ask more)\s*:",
            re.IGNORECASE | re.MULTILINE,
        )
        matches = list(label_re.finditer(s))
        if matches:
            for i, m in enumerate(matches):
                label = m.group(1).lower()
                key = "followup" if label == "if they ask more" else label
                start = m.end()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(s)
                out[key] = s[start:end].strip()
            return out

        # No labels, no JSON: treat the whole string as the body.
        out["body"] = s
        return out

    out["body"] = str(draft).strip()
    return out


def make_validate_tool(
    req_age: int,
    req_tone: str,
    judge_llm=None,
    child_name: str = "",
    situation: str = "",
):
    """Closure over the request fields so the tool has them at call time.

    If a `judge_llm` is provided, the tool sends the draft to the judge
    (see `judge.py`) for a Pydantic-validated multi-criteria review.
    Otherwise it falls back to a deterministic rule check.
    """
    min_w, max_w = explain_to_words(req_tone)
    bucket = age_bucket(req_age)

    def _judge(draft) -> str:
        assert judge_llm is not None
        try:
            from judge import judge_explanation, JudgeFailed
            verdict = judge_explanation(
                llm=judge_llm,
                draft=draft,
                req_age=req_age,
                req_tone=req_tone,
                child_name=child_name,
                situation=situation,
            )
        except JudgeFailed as e:
            print(f"[validate_explanation] judge failed after retry: {e}; falling back", flush=True)
            return _rule_based_check(draft)
        except Exception as e:
            print(f"[validate_explanation] judge call failed: {type(e).__name__}: {e}", flush=True)
            return _rule_based_check(draft)

        if verdict.ok and verdict.verdict == "approve":
            return "OK"
        if verdict.issues:
            return "Issues: " + " ".join(verdict.issues)
        # ok=false but no concrete issues — be safe, revise
        return "Issues: " + (verdict.reasoning or "The draft does not meet the rubric.")

    def _rule_based_check(draft) -> str:
        issues = []
        sections = _parse_sections(draft)
        if not sections["opener"]:
            issues.append("Missing the 'opener' field.")
        if not sections["body"]:
            issues.append("Missing the 'body' field.")
        if not sections["closer"]:
            issues.append("Missing the 'closer' field.")
        body_words = _word_count(sections["body"])
        if body_words < min_w:
            issues.append(f"Body too short ({body_words} words; minimum {min_w}).")
        elif body_words > max_w:
            issues.append(f"Body too long ({body_words} words; maximum {max_w}).")
        # Light moralizing / lecturing detection
        bad_phrases = ["you should always", "you must always", "remember to", "it's important to", "the lesson here is"]
        body_lower = sections["body"].lower()
        for p in bad_phrases:
            if p in body_lower:
                issues.append(f"Avoid lecturing. Body contains a phrase like '{p}'.")
        if not issues:
            return "OK"
        return "Issues: " + " ".join(issues)

    @tool
    def validate_explanation(draft) -> str:
        """Validate a Fabella explanation draft against the request.

        The drafter is asked to pass a JSON object with the four fields:
        'opener', 'body', 'closer', and optional 'followup'. The validator
        tolerates a JSON string, a Python dict, or a labeled draft text
        (backward compatibility).

        When a judge model is available, the judge does a multi-criteria
        review (opener/body/closer present, length, tone, no moralizing,
        age-appropriateness). Otherwise a deterministic rule check is used.

        Args:
            draft: JSON object, JSON string, or labeled draft text.

        Returns:
            'OK' if the draft passes. Otherwise a short report listing
            the issues to fix.
        """
        if judge_llm is not None:
            return _judge(draft)
        return _rule_based_check(draft)

    return validate_explanation


def _parse_judge_json(text: str) -> dict | None:
    """Tolerate markdown fences, leading prose, and pretty-printed JSON.

    Kept for backward-compat; the new judge module uses Pydantic and
    does not use this helper. Tests and legacy callers can still use it.
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
    candidate = t[i : j + 1]
    try:
        return json.loads(candidate)
    except Exception:
        return None


class FabellaAgentMiddleware(AgentMiddleware):
    """Ends the ReAct loop once the explanation has been validated, and
    forces a final answer after a small hard ceiling of iterations.

    The @hook_config(can_jump_to=["end"]) decorator is required — without
    it, LangGraph never creates the conditional edge and the early-exit
    silently does nothing.
    """

    def __init__(self, max_tool_calls: int = 2):
        super().__init__()
        self.max_tool_calls = max_tool_calls

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: AgentState, runtime: Runtime):
        from langchain.messages import ToolMessage

        tool_calls = [m for m in state.get("messages", []) if isinstance(m, ToolMessage)]
        last_tool = tool_calls[-1] if tool_calls else None

        if last_tool is not None and last_tool.content.strip() == "OK":
            print(f"[middleware] tool OK on call {len(tool_calls)}; jumping to end", flush=True)
            return {"jump_to": "end"}

        if len(tool_calls) >= self.max_tool_calls:
            print(f"[middleware] hit max tool calls ({self.max_tool_calls}); jumping to end", flush=True)
            return {"jump_to": "end"}

        return None


def _build_user_prompt(req) -> str:
    """Build the parent's request as a user-prompt for the drafter."""
    bucket = age_bucket(req.age)
    vocab = {
        "young": "very simple sentences (under 12 words each), short paragraphs, no abstract words",
        "middle": "clear sentences, concrete metaphors are fine",
        "older": "richer vocabulary is fine, but keep it direct",
    }[bucket]
    history = list(getattr(req, "history", []) or [])
    memory_block = ""
    recent_lines: list[str] = []
    for turn in history[-6:]:
        role = (turn.get("role") or "").strip().lower()
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        if role == "memory":
            memory_block = content
            continue
        label = "Parent" if role == "parent" else "Fabella"
        recent_lines.append(f"{label}: {content}")
    history_block = ""
    if recent_lines:
        history_block = (
            "Earlier in this conversation (for context, do not repeat verbatim):\n"
            + "\n".join(recent_lines)
            + "\n\n"
        )
    durable_block = ""
    if memory_block:
        durable_block = (
            "Long-term memory for this parent (use it; do not repeat verbatim):\n"
            f"{memory_block}\n\n"
        )
    return (
        f"A parent needs help explaining a hard thing to their child.\n\n"
        f"{durable_block}"
        f"{history_block}"
        f"The latest thing the parent is asking about: {req.situation}\n\n"
        f"The child is {req.age} years old ({bucket} reader).\n"
        f"Address the child as 'you'. Do not invent or use a name.\n"
        f"Tone: {req.tone}. Vocabulary: {vocab}.\n\n"
        f"If this is a follow-up, stay consistent with the previous explanation "
        f"and answer the new question in the same warm register. "
        f"Respond with a single JSON object that has exactly these four "
        f"keys: 'opener', 'body', 'closer', 'followup'. Then call "
        f"validate_explanation with the same JSON object."
    )


def build_agent(llm, req, judge_llm=None):
    """Build a one-shot ReAct agent bound to a specific request's tools.

    If `judge_llm` is provided, the validate_explanation tool sends the
    draft to the judge for a multi-criteria review. Otherwise the tool
    falls back to a deterministic rule check.
    """
    validate = make_validate_tool(
        req_age=req.age,
        req_tone=req.tone,
        judge_llm=judge_llm,
        child_name=req.child_name,
        situation=req.situation,
    )
    agent = create_agent(
        model=llm,
        tools=[validate],
        system_prompt=SYSTEM_PROMPT,
        middleware=[FabellaAgentMiddleware(max_tool_calls=2)],
    )
    return agent, _build_user_prompt(req)


def run_agent(llm, req, judge_llm=None) -> dict:
    """Run the agent. Return a dict with the parsed explanation.

    The result dict has keys: opener, body, closer, followup, raw.
    """
    print(
        f"[agent] building for age={req.age} tone={req.tone} judge={'yes' if judge_llm else 'no'}",
        flush=True,
    )
    agent, user_prompt = build_agent(llm, req, judge_llm=judge_llm)
    print(f"[agent] invoking", flush=True)
    started = time.monotonic()
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_prompt}]},
            {"recursion_limit": 12},
        )
    except Exception as e:
        print(f"[agent] invoke error: {type(e).__name__}: {e}", flush=True)
        return {
            "opener": "Fabella (agent error)",
            "body": f"_Agent loop failed: {type(e).__name__}: {e}_",
            "closer": "",
            "followup": "",
            "raw": "",
        }
    print(f"[agent] invoke complete", flush=True)
    msgs = result.get("messages", []) if isinstance(result, dict) else []
    print(f"[agent] {len(msgs)} messages in trace", flush=True)
    parsed = extract_explanation(msgs)
    _maybe_publish_trace(req, user_prompt, msgs, parsed, started)
    return parsed


def _maybe_publish_trace(req, user_prompt, messages, parsed, started) -> None:
    """Submit an anonymized trace to the Hub publisher, if capture is on.

    Per-request opt-out: ``req.share_trace=False`` skips that row. The
    global kill switch is ``FABELLA_SHARE_TRACES=0`` (handled inside the
    publisher). Failures here must never affect the parent-facing flow.
    """
    if getattr(req, "share_trace", True) is False:
        return
    try:
        from trace import build_trace_record, publisher

        latency_ms = int((time.monotonic() - started) * 1000)
        # The judge verdict lives in the ``ToolMessage`` that came back
        # from ``validate_explanation``; ``trace.py`` already knows how to
        # extract and anonymize it, so we just pass the messages through
        # and let the publisher do the work.
        record = build_trace_record(
            req=req,
            user_prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
            messages=messages,
            final_draft=parsed,
            judge_verdict=None,  # trace.py extracts it from messages
            latency_ms=latency_ms,
        )
        publisher.submit(record)
    except Exception as e:
        print(f"[agent] trace publish skipped: {type(e).__name__}: {e}", flush=True)


def extract_explanation(messages) -> dict:
    """Pull the latest validated draft from the agent's tool-call trace.

    Order of preference:
    1. The `draft` argument of the last validate_explanation tool call
       whose paired ToolMessage returned "OK".
    2. The content of the last AI message (the model's free-form final).
    3. The last validate_explanation tool-call draft, even if the judge
       returned issues. This is the best available draft when middleware
       ends the loop after `max_tool_calls`.
    """
    tool_results = _tool_results_by_id(messages)

    # 1. Last validated tool-call draft
    for msg in reversed(messages):
        if getattr(msg, "type", "") == "ai" and getattr(msg, "tool_calls", None):
            for tc in reversed(msg.tool_calls):
                call_id = tc.get("id") if isinstance(tc, dict) else None
                if (tool_results.get(call_id) or "").strip() != "OK":
                    continue
                args = tc.get("args") if isinstance(tc, dict) else {}
                draft = (args or {}).get("draft") if isinstance(args, dict) else None
                if draft:
                    sections = _parse_sections(draft)
                    return {**sections, "raw": json.dumps(draft) if not isinstance(draft, str) else draft}

    # 2. Last AI message with content (the model wrote a final answer
    #    after the validate_explanation call, without re-emitting the tool args)
    for msg in reversed(messages):
        if getattr(msg, "type", "") == "ai" and msg.content:
            text = msg.content if isinstance(msg.content, str) else str(msg.content)
            if text.strip():
                sections = _parse_sections(text)
                return {**sections, "raw": text}

    # 3. Last attempted validation draft, regardless of judge result. This
    # prevents a blank UI when the second validation still reports issues and
    # middleware jumps to end before the model writes a final answer.
    for msg in reversed(messages):
        if getattr(msg, "type", "") == "ai" and getattr(msg, "tool_calls", None):
            for tc in reversed(msg.tool_calls):
                if not isinstance(tc, dict) or tc.get("name") != "validate_explanation":
                    continue
                args = tc.get("args") or {}
                draft = args.get("draft") if isinstance(args, dict) else None
                if draft:
                    sections = _parse_sections(draft)
                    return {**sections, "raw": json.dumps(draft) if not isinstance(draft, str) else draft}

    return {"opener": "", "body": "", "closer": "", "followup": "", "raw": ""}


def _tool_results_by_id(messages) -> dict:
    results = {}
    for m in messages:
        if getattr(m, "type", "") == "tool":
            call_id = getattr(m, "tool_call_id", None)
            if call_id:
                content = m.content if isinstance(m.content, str) else str(m.content)
                results[call_id] = content
    return results
