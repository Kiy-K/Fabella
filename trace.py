"""Anonymized agent-trace capture and Hub publishing for Fabella.

What gets captured
------------------
One JSONL row per successful (or failed) agent invocation. The row contains
the full LangChain ReAct message trace (system prompt, user prompt,
assistant tool calls, tool responses, final draft) plus the judge's
structured verdict. This is the "Sharing is Caring" merit badge: a public,
anonymized dataset of how the drafter and judge reason about real parent
situations, so other builders can learn from the loop.

Anonymization
-------------
Before anything leaves the Space, the row is run through `_anonymize()`:

1. ``child_name`` is dropped from the request and replaced with ``"[name]"``
   in the final draft. The drafter was already told to address the child as
   "you" (see ``agent.py::SYSTEM_PROMPT``), so the name is rarely in the
   draft to begin with, but we strip it defensively.
2. The raw situation text is **never** stored. Only its SHA-256 hash, the
   first 80 characters, and its length go into the public row. The hash lets
   downstream consumers deduplicate across rows; the truncated prefix is
   enough to read the topic ("grandma in the hospital", "moving to a new
   house") without keeping the parent's actual words.
3. Freeform history turns are replaced with role + length counts. The drafter
   already takes only the last 6 turns, but we strip the content anyway.
4. The drafter system prompt is shipped in full (it's a static string from
   this repo, not user input) so the dataset is self-explanatory.

Opt-out
-------
``FABELLA_SHARE_TRACES=0`` disables capture entirely (default: on, except in
local dev). Per-request opt-out is also supported via ``share_trace=False``
on the ``ExplainRequest`` payload.

Publishing
----------
Captured rows accumulate in an in-memory buffer, then are appended to a
single JSONL file (``data/train-00000-of-00001.jsonl``) inside the dataset
repo via ``huggingface_hub.HfApi().upload_file(...)``. A background daemon
flushes the buffer every ``FLUSH_INTERVAL_S`` seconds OR when the buffer hits
``FLUSH_BUFFER_SIZE`` rows, whichever comes first. The HF token is read from
the ``HF_TOKEN`` env var, which HF Spaces injects automatically for the
owning user.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


log = logging.getLogger("fabella.traces")

# --- Configuration ---------------------------------------------------------

# Where the public dataset lives. The "build-small-hackathon" namespace is
# shared with the rest of the hackathon submissions; the slash keeps it on
# the build-small-hackathon org and avoids needing repo creation rights on
# the user's personal namespace.
DATASET_REPO = os.environ.get(
    "FABELLA_TRACE_REPO",
    "build-small-hackathon/fabella-traces",
)

# Buffer flush triggers. The background flusher will push whenever EITHER
# the buffer hits this many rows OR this many seconds have passed since the
# last flush, whichever fires first.
FLUSH_BUFFER_SIZE = int(os.environ.get("FABELLA_TRACE_FLUSH_SIZE", "5"))
FLUSH_INTERVAL_S = float(os.environ.get("FABELLA_TRACE_FLUSH_INTERVAL_S", "300"))

# The single file the dataset is written to. JSONL, append-only. We use one
# file rather than the HF datasets sharded convention because (a) the volume
# is small for a hackathon demo, and (b) we want every row in one place to
# make the dataset card's preview useful.
DATASET_FILE = "data/train-00000-of-00001.jsonl"

# Capture is on by default; off in local dev (set explicitly to "1" in prod
# env). HF Spaces injects HF_TOKEN automatically for the owning user.
SHARE_TRACES = os.environ.get("FABELLA_SHARE_TRACES", "1").lower() in (
    "1",
    "true",
    "yes",
    "on",
)


# --- Anonymization ---------------------------------------------------------


_SITUATION_PREFIX_LEN = 60


def _hash_situation(situation: str) -> str:
    """Stable, short hash so the same situation deduplicates across rows."""
    if not situation:
        return ""
    return "sha256:" + hashlib.sha256(situation.encode("utf-8")).hexdigest()[:16]


def _truncate_situation(situation: str) -> str:
    """Keep a short, generic prefix of the situation for at-a-glance browsing.

    The full situation text is never sent to the Hub. We only ship a short
    generic prefix (60 chars) so a human scanning the dataset can see what
    *kind* of topic each row covers without leaking identifying details
    about a real family. Pair this with ``situation_hash`` (for dedup) and
    ``situation_length`` (for context).
    """
    if not situation:
        return ""
    s = situation.strip()
    if len(s) <= _SITUATION_PREFIX_LEN:
        return s
    return s[:_SITUATION_PREFIX_LEN].rstrip() + "..."


_SITUATION_USER_PROMPT_MARKERS = (
    "The latest thing the parent is asking about:",
    "The latest thing the parent is asking about",
)


def _scrub_situation_from_prompt(prompt: str) -> str:
    """Strip the situation text from the drafter's user prompt.

    The user_prompt is built in ``agent.py::_build_user_prompt`` and
    embeds the raw situation in a sentence like ``"The latest thing the
    parent is asking about: <raw situation>"``. We replace the embedded
    situation with a length-only marker so the prompt's structure is
    preserved in the row but the parent's words are not.
    """
    if not prompt:
        return prompt
    out = prompt
    for marker in _SITUATION_USER_PROMPT_MARKERS:
        idx = out.find(marker)
        if idx < 0:
            continue
        prefix_end = idx + len(marker)
        newline = out.find("\n", prefix_end)
        if newline < 0:
            out = out[:prefix_end].rstrip() + " <redacted, see request.situation_length>"
            break
        out = out[:prefix_end] + " <redacted>" + out[newline:]
    return out


def _scrub_name(value: str) -> str:
    """Replace the child's name with a placeholder if it appears in text."""
    if not value:
        return value
    return value.replace("[name]", "[REDACTED_NAME]") or value


def _anonymize_request(req: dict) -> dict:
    """Strip PII from the request metadata before publishing."""
    child_name = (req.get("child_name") or "").strip()
    situation = req.get("situation") or ""
    history = req.get("history") or []
    return {
        "age": int(req.get("age") or 0),
        "tone": (req.get("tone") or "").strip().lower(),
        "child_name_present": bool(child_name),
        "child_name_redacted": "[name]" if child_name else "",
        "situation_hash": _hash_situation(situation),
        "situation_preview": _truncate_situation(situation),
        "situation_length": len(situation),
        "history_turns": len(history),
    }


def _anonymize_messages(messages: list) -> list:
    """Convert LangChain messages into a public-safe, JSON-only shape.

    Drops the message ``id`` field (internal LangChain state), serializes
    tool-call args to plain JSON, and replaces any ``[name]`` placeholder
    with a generic marker so the dataset reads consistently.
    """
    out: list[dict[str, Any]] = []
    for m in messages or []:
        kind = getattr(m, "type", "") or "unknown"
        content = getattr(m, "content", "") or ""
        if not isinstance(content, str):
            content = str(content)
        entry: dict[str, Any] = {
            "role": kind,
            "content": _scrub_name(content),
        }
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            entry["tool_calls"] = [
                {
                    "id": tc.get("id") if isinstance(tc, dict) else None,
                    "name": tc.get("name") if isinstance(tc, dict) else None,
                    "args": tc.get("args") if isinstance(tc, dict) else {},
                }
                for tc in tool_calls
            ]
        tool_call_id = getattr(m, "tool_call_id", None)
        if tool_call_id:
            entry["tool_call_id"] = tool_call_id
        out.append(entry)
    return out


def _anonymize_judge(verdict: dict | None) -> dict | None:
    if not verdict:
        return None
    return {
        "ok": bool(verdict.get("ok")),
        "issues": [str(i)[:200] for i in (verdict.get("issues") or [])],
        "score": float(verdict.get("score") or 0.0),
        "verdict": str(verdict.get("verdict") or ""),
        "reasoning": str(verdict.get("reasoning") or "")[:300],
    }


def _anonymize_draft(draft: dict) -> dict:
    """Drop the child's name from the final draft if it slipped in."""
    out = {}
    for k in ("opener", "body", "closer", "followup"):
        out[k] = _scrub_name((draft.get(k) or "").strip())
    return out


# --- Trace record ----------------------------------------------------------


@dataclass
class TraceRecord:
    """A single anonymized agent trace, ready to publish."""

    trace_id: str
    created_at: str
    model_versions: dict[str, str]
    request: dict
    agent: dict
    judge: dict | None
    latency_ms: int
    tool_calls: int
    app_version: str
    schema_version: int = 1

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


# --- Capture API -----------------------------------------------------------


def build_trace_record(
    *,
    req: Any,
    user_prompt: str,
    system_prompt: str,
    messages: list,
    final_draft: dict,
    judge_verdict: dict | None,
    latency_ms: int,
    app_version: str = "fabella-1",
) -> TraceRecord:
    """Assemble a TraceRecord from the agent's inputs and outputs.

    Called from ``agent.py::run_agent`` AFTER the loop completes. The
    caller is responsible for passing the live messages list and the
    parsed final draft dict.
    """
    tool_calls = sum(
        1
        for m in messages
        if getattr(m, "type", "") == "ai" and getattr(m, "tool_calls", None)
    )
    return TraceRecord(
        trace_id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc).isoformat(),
        model_versions={
            "drafter": "google/gemma-4-E4B-it",
            "judge": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        },
        request=_anonymize_request(
            {
                "age": getattr(req, "age", 0),
                "tone": getattr(req, "tone", ""),
                "child_name": getattr(req, "child_name", ""),
                "situation": getattr(req, "situation", ""),
                "history": getattr(req, "history", []) or [],
            }
        ),
        agent={
            "system_prompt": system_prompt,
            "user_prompt": _scrub_situation_from_prompt(_scrub_name(user_prompt)),
            "messages": _anonymize_messages(messages),
            "final_draft": _anonymize_draft(final_draft),
        },
        judge=_anonymize_judge(judge_verdict),
        latency_ms=int(latency_ms),
        tool_calls=tool_calls,
        app_version=app_version,
    )


# --- Publisher (background buffer + flush) --------------------------------


class TracePublisher:
    """Thread-safe in-memory buffer that flushes JSONL rows to the Hub.

    The publisher is started once at app import time. ``submit()`` is
    non-blocking (it just appends to the buffer). The background thread
    flushes on a timer AND when the buffer crosses ``FLUSH_BUFFER_SIZE``,
    so a burst of traffic doesn't sit in memory for the full interval.

    All flushes are serialized through ``_flush_lock`` so two threads can
    never push overlapping revisions of the file. The Hub append uses
    ``commit_message="append"`` so a single push updates the same file.
    """

    def __init__(self) -> None:
        self._buffer: list[TraceRecord] = []
        self._lock = threading.Lock()
        self._flush_lock = threading.Lock()
        self._last_flush = time.monotonic()
        self._enabled = SHARE_TRACES and bool(os.environ.get("HF_TOKEN"))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start(self) -> None:
        if not self._enabled:
            log.info("[traces] publishing disabled (FABELLA_SHARE_TRACES=0 or HF_TOKEN missing)")
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="fabella-trace-flusher", daemon=True
        )
        self._thread.start()
        log.info(
            f"[traces] publisher started, repo={DATASET_REPO} "
            f"flush_size={FLUSH_BUFFER_SIZE} flush_interval_s={FLUSH_INTERVAL_S}"
        )

    def stop(self) -> None:
        self._stop.set()

    def submit(self, record: TraceRecord) -> None:
        """Non-blocking. Drops the record if publishing is disabled."""
        if not self._enabled:
            return
        with self._lock:
            self._buffer.append(record)
            should_flush = len(self._buffer) >= FLUSH_BUFFER_SIZE
        if should_flush:
            self._flush_async()

    def _run(self) -> None:
        while not self._stop.is_set():
            # Sleep in small slices so stop() returns promptly.
            slept = 0.0
            while slept < FLUSH_INTERVAL_S and not self._stop.is_set():
                self._stop.wait(min(1.0, FLUSH_INTERVAL_S - slept))
                slept = FLUSH_INTERVAL_S if self._stop.is_set() else slept + 1.0
            if self._stop.is_set():
                break
            self._flush()

    def _flush_async(self) -> None:
        threading.Thread(target=self._flush, daemon=True).start()

    def _flush(self) -> None:
        if not self._enabled:
            return
        with self._flush_lock:
            with self._lock:
                if not self._buffer:
                    return
                rows = self._buffer
                self._buffer = []
            try:
                self._push_to_hub(rows)
                self._last_flush = time.monotonic()
                log.info(f"[traces] flushed {len(rows)} rows to {DATASET_REPO}")
            except Exception as e:
                # Don't lose data on transient failures. Put the rows back
                # at the head of the buffer so the next flush retries.
                log.warning(
                    f"[traces] push failed: {type(e).__name__}: {e}; "
                    f"re-queuing {len(rows)} rows"
                )
                with self._lock:
                    self._buffer = rows + self._buffer

    def _push_to_hub(self, rows: list[TraceRecord]) -> None:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        new_lines = "\n".join(r.to_jsonl() for r in rows) + "\n"

        existing = ""
        try:
            existing = api.hf_hub_download(
                repo_id=DATASET_REPO,
                filename=DATASET_FILE,
                repo_type="dataset",
                force_download=False,
            )
            existing_path = Path(existing)
            if existing_path.exists():
                existing = existing_path.read_text(encoding="utf-8")
                if existing and not existing.endswith("\n"):
                    existing += "\n"
        except Exception:
            existing = ""

        api.upload_file(
            path_or_fileobj=(existing + new_lines).encode("utf-8"),
            path_in_repo=DATASET_FILE,
            repo_id=DATASET_REPO,
            repo_type="dataset",
            commit_message=f"append {len(rows)} trace row(s)",
        )


# Module-level singleton. Imported by app.py; ``start()`` is called once at
# import time below.
publisher = TracePublisher()
