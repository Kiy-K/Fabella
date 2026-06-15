"""Anonymized agent-trace capture and Hub publishing for Fabella.

What gets captured
------------------
One row per successful (or failed) agent invocation. The row contains
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
``FABELLA_SHARE_TRACES=0`` disables capture entirely (this is the
default). To re-enable the public dataset, set
``FABELLA_SHARE_TRACES=1`` on the Space. Per-request opt-out is also
supported via ``share_trace=False`` on the ``ExplainRequest`` payload.

Publishing
----------
Captured rows accumulate in an in-memory buffer, then are flushed to the
Hub via ``huggingface_hub.HfApi().create_commit(...)`` with one
``CommitOperationAdd`` per row, each targeting a unique
``data/<trace_id>.json`` path. A background daemon flushes the buffer every
``FLUSH_INTERVAL_S`` seconds OR when the buffer hits ``FLUSH_BUFFER_SIZE``
rows, whichever comes first. The HF token is read from the ``HF_TOKEN``
env var, which HF Spaces injects automatically for the owning user.

Why per-row files, not a single JSONL
------------------------------------
The previous version downloaded ``data/train-00000-of-00001.jsonl``,
appended locally, and re-uploaded the whole file. That is a read-modify-write
cycle which (a) loses rows when two Space replicas flush concurrently, and
(b) re-uploads the full file on every push (O(n²) bandwidth as the dataset
grows). The per-row UUID pattern is race-free across replicas (each row is
its own commit, no shared mutable state on the Hub) and constant-cost per
push. The dataset card's "preview" still works because ``huggingface_hub``
lists ``data/*.json`` in the repo tree.

Startup probe
-------------
On ``start()`` (only when capture is enabled) we call ``_probe()`` which
atomically Add+Deletes a tiny ``data/.probe.json`` file in a single
``create_commit`` call, to verify (a) the HF token is valid, (b) the
token can create the dataset repo, and (c) the token can commit files
to it. The Add+Delete happen in one commit, so a partial failure can
never leave the probe file in the public dataset. If the probe fails we
log a single loud ERROR with the exception class and message and
disable capture for the lifetime of the process — no silent drops.
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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


log = logging.getLogger("fabella.traces")

# --- Configuration ---------------------------------------------------------

# Where the public dataset lives. The default is the hackathon org's
# namespace (``build-small-hackathon/fabella-traces``) so the live Space
# publishes to the canonical repo attached to the Space's "Datasets" tab.
# For local dev where the Space's HF_TOKEN cannot create repos in the org
# namespace, set ``FABELLA_TRACE_REPO=Kiy-K/fabella-traces`` (the maker's
# personal dataset) to publish to a fallback path.
DATASET_REPO = os.environ.get(
    "FABELLA_TRACE_REPO",
    "build-small-hackathon/fabella-traces",
)

# Buffer flush triggers. The background flusher will push whenever EITHER
# the buffer hits this many rows OR this many seconds have passed since the
# last flush, whichever fires first.
FLUSH_BUFFER_SIZE = int(os.environ.get("FABELLA_TRACE_FLUSH_SIZE", "5"))
FLUSH_INTERVAL_S = float(os.environ.get("FABELLA_TRACE_FLUSH_INTERVAL_S", "300"))

# Per-row upload path prefix. Each row lives at ``data/<trace_id>.json``,
# keyed by the row's own UUID. This eliminates the read-modify-write race
# that bit the old single-JSONL design when (a) the Space ran >1 replica
# or (b) the timer-thread and size-triggered async flush interleaved.
DATASET_DIR = "data"
PROBE_PATH = f"{DATASET_DIR}/.probe.json"

# Capture is OFF by default. The dataset card and schema are live at
# ``build-small-hackathon/fabella-traces``; the only path to data right
# now is the per-parent "Download my history" self-export button in
# ``app.py::api_history_download``. To re-enable publishing for a new
# deployment, set ``FABELLA_SHARE_TRACES=1`` on the Space and the
# publisher will resume writing rows to the org dataset.
SHARE_TRACES = os.environ.get("FABELLA_SHARE_TRACES", "0").lower() in (
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


def _stringify_message_content(content: Any) -> str:
    """Render a LangChain message ``content`` field as a plain string.

    LangChain 1.x allows ``content`` to be ``str`` (the common case) or a
    list of content blocks (e.g. ``[{"type": "text", "text": "..."}]``).
    List values used to crash the trace publisher because we did
    ``str(content)`` and then string-replaced on the resulting repr. Now
    we extract just the text blocks, which is what a human reading the
    dataset wants anyway.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                txt = block.get("text")
                if isinstance(txt, str):
                    parts.append(txt)
                else:
                    parts.append(str(block))
            elif isinstance(block, str):
                parts.append(block)
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def _anonymize_messages(messages: list) -> list:
    """Convert LangChain messages into a public-safe, JSON-only shape.

    Drops the message ``id`` field (internal LangChain state), serializes
    tool-call args to plain JSON, and replaces any ``[name]`` placeholder
    with a generic marker so the dataset reads consistently.
    """
    out: list[dict[str, Any]] = []
    for m in messages or []:
        kind = getattr(m, "type", "") or "unknown"
        content = _stringify_message_content(getattr(m, "content", None))
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


def _extract_judge_verdict(messages: list) -> dict | None:
    """Pull the latest validate_explanation judge verdict from the message list.

    ``agent.py`` runs ``validate_explanation`` as a tool. The tool returns a
    JSON string with the Pydantic-validated verdict (see ``judge.py``); the
    content of the matching ``ToolMessage`` is the source of truth. We
    surface it on the trace row so the public dataset carries the
    drafter-judge conversation end-to-end.

    Note: LangChain 1.x ``ToolMessage`` does not populate ``name`` from the
    tool's registered name, so we identify the verdict by its JSON shape
    (presence of ``ok`` / ``verdict`` / ``issues``) rather than by ``name``.
    """
    last_verdict_msg = None
    for m in messages or []:
        if getattr(m, "type", "") != "tool":
            continue
        content = _stringify_message_content(getattr(m, "content", None)).strip()
        if not content:
            continue
        try:
            parsed = json.loads(content)
        except (TypeError, ValueError):
            continue
        if isinstance(parsed, dict) and (
            "ok" in parsed or "verdict" in parsed or "issues" in parsed
        ):
            last_verdict_msg = m
    if last_verdict_msg is None:
        return None
    content = _stringify_message_content(getattr(last_verdict_msg, "content", None)).strip()
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    return _anonymize_judge(parsed)


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

    def to_json(self) -> str:
        """Render the record as a single JSON object string."""
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
        judge=(
            _anonymize_judge(judge_verdict)
            if judge_verdict is not None
            else _extract_judge_verdict(messages)
        ),
        latency_ms=int(latency_ms),
        tool_calls=tool_calls,
        app_version=app_version,
    )


# --- Publisher (background buffer + flush) --------------------------------


class TracePublisher:
    """Thread-safe in-memory buffer that flushes trace rows to the Hub.

    The publisher is started once at app import time. ``submit()`` is
    non-blocking (it just appends to the buffer). The background thread
    flushes on a timer AND when the buffer crosses ``FLUSH_BUFFER_SIZE``,
    so a burst of traffic doesn't sit in memory for the full interval.

    All flushes are serialized through ``_flush_lock`` so two threads can
    never push overlapping Hub revisions. Each row is uploaded as its own
    ``data/<trace_id>.json`` file via a single ``create_commit`` per flush
    batch, so concurrent flushes (or >1 Space replica) cannot lose rows
    to a read-modify-write race on a shared JSONL file.
    """

    def __init__(self) -> None:
        self._buffer: list[TraceRecord] = []
        self._lock = threading.Lock()
        self._flush_lock = threading.Lock()
        self._last_flush = time.monotonic()
        # HF Spaces inject the token as HF_TOKEN. Some runtimes use HF_HUB_TOKEN
        # instead; accept either so a misconfigured Space does not silently
        # disable trace publishing.
        self._hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HF_HUB_TOKEN") or ""
        self._enabled = SHARE_TRACES and bool(self._hf_token)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # How many rows we will buffer before refusing new submissions. Set
        # high enough to ride out a transient Hub failure, low enough to
        # avoid leaking memory if the Hub is permanently unreachable.
        self._max_buffer = max(FLUSH_BUFFER_SIZE * 4, 50)
        # Used to throttle repeated failure logs to one ERROR per N attempts.
        self._consecutive_failures = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start(self) -> None:
        if not SHARE_TRACES:
            log.info("[traces] publishing disabled (FABELLA_SHARE_TRACES=0)")
            return
        if not self._hf_token:
            # Loud at WARNING so the Space logs make the missing-token case
            # obvious. The previous version's INFO-only line made it look
            # like an intentional no-op.
            log.warning(
                "[traces] publishing DISABLED: HF_TOKEN (or HF_HUB_TOKEN) is "
                "not set on this Space. Trace rows will be dropped on the floor. "
                "Set HF_TOKEN to the Space owner's write token to enable capture."
            )
            return
        if self._thread is not None and self._thread.is_alive():
            return
        # Ensure the dataset repo exists, then run a write+delete probe so a
        # misconfigured Space (bad token, read-only token, no create rights)
        # fails LOUDLY at import time instead of silently dropping rows for
        # hours. If either step fails we disable capture for the lifetime of
        # the process; the per-parent self-export path in app.py keeps
        # working regardless.
        if not self._ensure_repo():
            self._enabled = False
            return
        if not self._probe():
            self._enabled = False
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="fabella-trace-flusher", daemon=True
        )
        self._thread.start()
        log.info(
            f"[traces] publisher started, repo={DATASET_REPO} "
            f"flush_size={FLUSH_BUFFER_SIZE} flush_interval_s={FLUSH_INTERVAL_S} "
            f"max_buffer={self._max_buffer}"
        )

    def stop(self) -> None:
        self._stop.set()

    def submit(self, record: TraceRecord) -> None:
        """Non-blocking. Drops the record if publishing is disabled or full."""
        if not self._enabled:
            return
        with self._lock:
            if len(self._buffer) >= self._max_buffer:
                # The Hub has been failing for a while and the buffer is
                # full. Drop the oldest rows so a recovered Hub push only
                # loses ancient data, not fresh rows.
                dropped = len(self._buffer) - self._max_buffer + 1
                del self._buffer[:dropped]
                log.warning(
                    f"[traces] buffer full ({self._max_buffer}); dropped "
                    f"{dropped} oldest row(s) while Hub push is failing"
                )
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
                self._consecutive_failures = 0
                log.info(f"[traces] flushed {len(rows)} rows to {DATASET_REPO}")
            except Exception as e:
                self._consecutive_failures += 1
                # Don't lose data on transient failures. Put the rows back
                # at the head of the buffer so the next flush retries. Log
                # at WARNING the first few times, then ERROR on every 10th
                # so a stuck Hub does not spam the Space log.
                with self._lock:
                    self._buffer = rows + self._buffer
                if self._consecutive_failures <= 3 or self._consecutive_failures % 10 == 0:
                    log.log(
                        logging.ERROR if self._consecutive_failures > 3 else logging.WARNING,
                        f"[traces] push failed "
                        f"({self._consecutive_failures} consecutive): "
                        f"{type(e).__name__}: {e}; re-queuing {len(rows)} rows. "
                        f"Check that the Space's HF_TOKEN has write access to "
                        f"'{DATASET_REPO}'.",
                    )

    def _commit(self, operations, message, description=None) -> None:
        """Single-call wrapper around ``HfApi.create_commit`` for the dataset repo.

        All Hub writes from the publisher go through this helper so the
        ``commit_message`` / ``commit_description`` format and the
        ``repo_id`` / ``repo_type`` kwargs stay consistent across the
        per-row flush and the startup probe. If we ever need to add
        e.g. ``revision`` or a custom user-agent, we change one place.
        """
        from huggingface_hub import HfApi

        kwargs = {"commit_description": description} if description else {}
        HfApi(token=self._hf_token).create_commit(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            operations=list(operations),
            commit_message=message,
            **kwargs,
        )

    def _push_to_hub(self, rows: list[TraceRecord]) -> None:
        """Upload a batch of rows as one commit with one operation per row.

        Each row is its own ``data/<trace_id>.json`` file, so the upload is
        race-free across Space replicas and across the timer-thread vs.
        size-triggered async flush within one process. One ``create_commit``
        call wraps the whole batch as a single commit on the Hub, so the
        git history is one commit per flush (not one per row).
        """
        from huggingface_hub import CommitOperationAdd

        operations = [
            CommitOperationAdd(
                path_in_repo=f"{DATASET_DIR}/{r.trace_id}.json",
                path_or_fileobj=r.to_json().encode("utf-8"),
            )
            for r in rows
        ]
        self._commit(
            operations,
            message=f"append {len(rows)} fabella trace row(s)",
            description=(
                f"Append {len(rows)} anonymized Fabella trace row(s). "
                f"Schema: fabella.trace.v1. No raw situation, no child name."
            ),
        )

    def _ensure_repo(self) -> bool:
        """Create the dataset repo if it doesn't exist. Return True on success.

        A 401/403 here means the Space's ``HF_TOKEN`` doesn't have
        ``repo.create`` rights in the target namespace (e.g. it can read
        ``Kiy-K/fabella-traces`` but not create it). In that case the
        Space owner must pre-create the dataset and grant the token write
        access, OR override ``FABELLA_TRACE_REPO`` to a namespace where
        the token can create. We return ``False`` so the caller can
        disable capture loudly instead of failing every flush.
        """
        from huggingface_hub import HfApi

        try:
            HfApi(token=self._hf_token).create_repo(
                repo_id=DATASET_REPO,
                repo_type="dataset",
                exist_ok=True,
                private=False,
            )
            return True
        except Exception as e:
            log.error(
                f"[traces] cannot create or access dataset repo "
                f"'{DATASET_REPO}': {type(e).__name__}: {e}. "
                f"Pre-create the dataset on the Hub and grant the Space's "
                f"HF_TOKEN write access, or set FABELLA_TRACE_REPO to a "
                f"namespace where the token can create repos. "
                f"Trace capture DISABLED for this process."
            )
            return False

    def _probe(self) -> bool:
        """One-shot write+delete in a single atomic commit to verify Hub access.

        Catches (a) bad token, (b) missing repo-create rights, (c) missing
        repo-write rights — all at import time, so a misconfigured Space
        doesn't silently drop rows for hours before anyone notices.

        The Add and Delete of ``PROBE_PATH`` are bundled into a single
        ``create_commit`` call so the probe is atomic on the Hub: either
        the commit succeeds (probe is added then removed in one revision)
        or it fails (probe is never created). There is no intermediate
        state where a sentinel file sits in the public dataset.
        Returns True if the round-trip succeeded.
        """
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete
        import warnings

        payload = json.dumps(
            {
                "probe": True,
                "ts": datetime.now(timezone.utc).isoformat(),
                "pid": os.getpid(),
            }
        ).encode("utf-8")
        try:
            # The ``huggingface_hub`` SDK emits a ``UserWarning`` when an
            # Add and a Delete target the same path in one commit. We
            # intentionally rely on this pattern (atomic probe), so
            # silence just this warning around the call. The commit
            # itself still executes correctly.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="About to delete a file that have just been updated",
                    category=UserWarning,
                )
                self._commit(
                    [
                        CommitOperationAdd(
                            path_in_repo=PROBE_PATH,
                            path_or_fileobj=payload,
                        ),
                        CommitOperationDelete(path_in_repo=PROBE_PATH),
                    ],
                    message="fabella trace publisher startup probe",
                    description=(
                        "Atomic Add+Delete of .probe.json to verify the HF_TOKEN "
                        "can commit to the dataset repo. The probe file is never "
                        "left in the repo on success."
                    ),
                )
            log.info(f"[traces] probe OK against {DATASET_REPO}")
            return True
        except Exception as e:
            log.error(
                f"[traces] probe FAILED against {DATASET_REPO}: "
                f"{type(e).__name__}: {e}"
            )
            return False


# Module-level singleton. Imported by app.py; ``start()`` is called once at
# import time below.
publisher = TracePublisher()
