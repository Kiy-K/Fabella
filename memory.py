"""Fabella's production memory layer.

Memory is a single JSON document per parent, persisted in the HF Bucket
at `/data/fabella-data/user-<owner_key>.memory.json` (HF OAuth username
or anonymous browser session id). It survives restarts, Space restarts,
and frontend reloads.

The schema is intentionally simple so the drafter can read it on every
turn without expensive retrieval:

    {
      "version": 1,
      "owner_key": "hf:<username>" | "anon:<session_id>",
      "created_at": <iso>,
      "updated_at": <iso>,
      "preferences": {
        "child_name": "...",
        "child_age": 7,
        "preferred_tone": "gentle",
        "preferred_voice": "...",
        "language": "en"
      },
      "facts": [
        # Stable, durable things the parent has told us. Newest last.
        {"key": "family_event", "text": "Grandma had surgery in March 2026.", "added_at": <iso>},
        ...
      ],
      "summary": "Parent has been working through explaining grandma's hospitalization to a 7-year-old.",
      "threads": [
        # Recent parent/Fabella turns (full text, capped at 12).
        {"role": "parent", "content": "...", "created_at": <iso>},
        {"role": "fabella", "content": "...", "created_at": <iso>},
        ...
      ]
    }

Reading: `read_memory(owner_key)` returns the document, or a fresh one.
Writing: `append_turn(owner_key, ...)` is atomic (write-temp-then-rename)
and uses an in-process lock. It runs an LLM extraction pass to update
`summary`, `facts`, and `preferences` in one transaction. If the
extraction call fails, it falls back to a deterministic update.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


SCHEMA_VERSION = 1
DATA_DIR = Path(os.environ.get("FABELLA_DATA_DIR", "/data/fabella-data"))
MEMORY_LOCK = Lock()
_SAFE_KEY = re.compile(r"[^a-zA-Z0-9._-]+")
MAX_THREADS = 12
MAX_FACTS = 24
EXTRACT_TIMEOUT_S = 12


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_slug(value: str) -> str:
    cleaned = _SAFE_KEY.sub("-", value).strip("-")
    return (cleaned or "anon")[:80]


def memory_path(owner_key: str) -> Path:
    return DATA_DIR / f"user-{_safe_slug(owner_key)}.memory.json"


def empty_memory(owner_key: str) -> dict[str, Any]:
    return {
        "version": SCHEMA_VERSION,
        "owner_key": owner_key,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "preferences": {
            "child_name": "",
            "child_age": None,
            "preferred_tone": "gentle",
            "preferred_voice": "",
            "language": "en",
        },
        "facts": [],
        "summary": "",
        "threads": [],
    }


def _read_unlocked(owner_key: str) -> dict[str, Any]:
    path = memory_path(owner_key)
    if not path.exists():
        return empty_memory(owner_key)
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception as e:
        print(f"[memory] read failed for {owner_key}: {type(e).__name__}: {e}", flush=True)
        return empty_memory(owner_key)
    if not isinstance(data, dict):
        return empty_memory(owner_key)
    if data.get("version") != SCHEMA_VERSION:
        return empty_memory(owner_key)
    data.setdefault("preferences", empty_memory(owner_key)["preferences"])
    data.setdefault("facts", [])
    data.setdefault("summary", "")
    data.setdefault("threads", [])
    return data


def read_memory(owner_key: str) -> dict[str, Any]:
    with MEMORY_LOCK:
        return _read_unlocked(owner_key)


def _write_unlocked(owner_key: str, data: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = memory_path(owner_key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


@dataclass
class Turn:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {
            "role": self.role,
            "content": self.content,
            "created_at": _now_iso(),
        }


@dataclass
class AppendResult:
    memory: dict[str, Any]
    extraction: str  # "llm" | "fallback"


def _openai_call(judge_url: str, system: str, user: str) -> dict | None:
    """Best-effort structured call to the judge OpenAI-compatible endpoint."""
    if not judge_url:
        return None
    body = {
        "model": "nemotron-3-4b",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 600,
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        judge_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=EXTRACT_TIMEOUT_S) as res:
            payload = json.loads(res.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as e:
        print(f"[memory] extract LLM call failed: {type(e).__name__}: {e}", flush=True)
        return None
    text = (((payload.get("choices") or [{}])[0]).get("message") or {}).get("content") or ""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception as e:
        print(f"[memory] extract JSON parse failed: {type(e).__name__}: {e}", flush=True)
        return None


def _llm_extract(turns_to_summarize: list[dict[str, str]], prior_summary: str, prior_facts: list[dict[str, str]]) -> dict | None:
    judge_url = os.environ.get("MODAL_JUDGE_URL", "").strip()
    if not judge_url:
        return None
    sys_prompt = (
        "You extract long-term memory for a parent-facing assistant. "
        "Given recent turns and the parent's prior memory, return ONLY a JSON object with three keys: "
        '"new_facts" (array of objects {key, text}), '
        '"summary" (one short sentence, <= 200 chars, that captures the rolling thread), '
        '"preferences" (object with optional child_name, child_age, preferred_tone, language). '
        "Only include facts that are stable, durable, and likely to be referenced again. "
        "Do not include transient detail. Address the child as 'you'; never invent a name."
    )
    serialized_turns = "\n".join(
        f"{(t.get('role') or 'parent').title()}: {(t.get('content') or '').strip()[:600]}"
        for t in turns_to_summarize
    )
    user = (
        f"Prior summary: {prior_summary or '(none)'}\n\n"
        f"Prior facts: {json.dumps(prior_facts, ensure_ascii=False)}\n\n"
        f"Recent turns to fold into memory:\n{serialized_turns}\n\n"
        "Return JSON only."
    )
    return _openai_call(judge_url, sys_prompt, user)


def _deterministic_facts(turn: dict[str, str]) -> list[dict[str, str]]:
    """Cheap fallback: keep one fact per parent turn, capped to the most recent."""
    text = (turn.get("content") or "").strip()
    if not text or turn.get("role") != "parent":
        return []
    return [{"key": "recent_note", "text": text[:240], "added_at": _now_iso()}]


def _deterministic_summary(prior: str, new_turns: list[dict[str, str]]) -> str:
    parts = [prior.strip()] if prior else []
    for t in new_turns[-2:]:
        role = (t.get("role") or "").title() or "Parent"
        content = (t.get("content") or "").strip()
        if content:
            parts.append(f"{role}: {content[:120]}")
    summary = " | ".join(p for p in parts if p)
    return summary[:280]


def append_turn(
    owner_key: str,
    parent_text: str,
    fabella_text: str,
    preferences: dict[str, Any] | None = None,
) -> AppendResult:
    """Atomically append a parent/fabella turn and update memory.

    The function tries the LLM extractor first (Nemotron-3 judge) for
    rolling summary and fact extraction. If that fails, it falls back
    to a deterministic update so the user still gets continuity.
    """
    parent_text = (parent_text or "").strip()
    fabella_text = (fabella_text or "").strip()
    if not parent_text and not fabella_text:
        return AppendResult(memory=read_memory(owner_key), extraction="fallback")

    new_turns: list[dict[str, str]] = []
    if parent_text:
        new_turns.append(Turn(role="parent", content=parent_text).to_dict())
    if fabella_text:
        new_turns.append(Turn(role="fabella", content=fabella_text).to_dict())

    with MEMORY_LOCK:
        data = _read_unlocked(owner_key)
        prior_summary = data.get("summary", "")
        prior_facts = list(data.get("facts", []))
        summary = ""
        extraction = "fallback"

        extracted = _llm_extract(new_turns, prior_summary, prior_facts)
        if extracted:
            try:
                new_facts = extracted.get("new_facts") or []
                for f in new_facts:
                    if not isinstance(f, dict):
                        continue
                    text = (f.get("text") or "").strip()
                    if not text:
                        continue
                    prior_facts.append({
                        "key": (f.get("key") or "note")[:40],
                        "text": text[:240],
                        "added_at": _now_iso(),
                    })
                extracted_summary = (extracted.get("summary") or "").strip()[:280]
                if extracted_summary:
                    summary = extracted_summary
                pref_obj = extracted.get("preferences") or {}
                if isinstance(pref_obj, dict):
                    prefs = data.get("preferences") or {}
                    for k in ("child_name", "preferred_tone", "language"):
                        v = pref_obj.get(k)
                        if v:
                            prefs[k] = str(v)[:40]
                    age = pref_obj.get("child_age")
                    if isinstance(age, int) and 3 <= age <= 18:
                        prefs["child_age"] = age
                    data["preferences"] = prefs
                extraction = "llm"
            except Exception as e:
                print(f"[memory] extract LLM payload invalid: {type(e).__name__}: {e}", flush=True)

        if not summary:
            for t in new_turns:
                if t.get("role") == "parent":
                    for f in _deterministic_facts(t):
                        prior_facts.append(f)
            summary = _deterministic_summary(prior_summary, new_turns)

        if preferences:
            prefs = data.get("preferences") or {}
            for k, v in preferences.items():
                if v in (None, ""):
                    continue
                prefs[k] = v
            data["preferences"] = prefs

        data["summary"] = summary
        data["facts"] = prior_facts[-MAX_FACTS:]
        data["threads"] = (data.get("threads", []) + new_turns)[-MAX_THREADS:]
        data["updated_at"] = _now_iso()
        _write_unlocked(owner_key, data)

    return AppendResult(memory=data, extraction=extraction)


def clear_memory(owner_key: str) -> None:
    with MEMORY_LOCK:
        path = memory_path(owner_key)
        if path.exists():
            try:
                path.unlink()
            except Exception as e:
                print(f"[memory] clear failed for {owner_key}: {type(e).__name__}: {e}", flush=True)


def public_view(memory: dict[str, Any]) -> dict[str, Any]:
    """Strip internal fields before sending to the client."""
    return {
        "version": memory.get("version"),
        "owner_key": memory.get("owner_key"),
        "created_at": memory.get("created_at"),
        "updated_at": memory.get("updated_at"),
        "preferences": memory.get("preferences", {}),
        "facts": list(memory.get("facts", []))[-MAX_FACTS:],
        "summary": memory.get("summary", ""),
        "threads": list(memory.get("threads", []))[-MAX_THREADS:],
    }


def memory_context_block(memory: dict[str, Any], max_chars: int = 1600) -> str:
    """Format memory as a context block for the drafter prompt."""
    if not memory:
        return ""
    parts: list[str] = []
    summary = (memory.get("summary") or "").strip()
    if summary:
        parts.append(f"Summary so far: {summary[:240]}")
    prefs = memory.get("preferences") or {}
    if isinstance(prefs, dict):
        pn = (prefs.get("child_name") or "").strip()
        pa = prefs.get("child_age")
        pt = (prefs.get("preferred_tone") or "").strip()
        if pn or pa or pt:
            parts.append(
                "Parent preferences: "
                + ", ".join(
                    f for f in [
                        f"child_name={pn}" if pn else "",
                        f"child_age={pa}" if pa else "",
                        f"tone={pt}" if pt else "",
                    ] if f
                )
            )
    facts = memory.get("facts") or []
    if facts:
        recent = facts[-6:]
        parts.append("Durable facts about this family:\n" + "\n".join(f"- {(f.get('text') or '').strip()[:200]}" for f in recent if f.get("text")))
    block = "\n".join(parts).strip()
    return block[:max_chars]
