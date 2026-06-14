"""Fabella - small words for big questions.

A Gradio Server (FastAPI subclass) serves a custom HTML+CSS+JS page. The
parent describes a hard-to-explain situation; Fabella drafts a short,
kind, age-appropriate explanation, validated by a second small model.

Architecture (see modal_app.py for the server side):
  Gemma 4 E4B (drafter, A10G)  - writes the explanation
  Nemotron-3 Nano 4B (judge, A10G) - multi-criteria review
"""

import os
import sys
import asyncio
import traceback
import base64
import json
import urllib.error
import urllib.request
import re
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("HF_HUB_DISABLE_EXPERIMENTAL_WARNING", "1")
warnings.filterwarnings(
    "ignore",
    message=".*HTTP_422_UNPROCESSABLE_ENTITY.*",
)
warnings.filterwarnings(
    "ignore",
    message="OAuth is not supported outside of a Space environment.*",
)


def _silence_asyncio_invalid_fd_warning() -> None:
    import asyncio

    original_del = asyncio.BaseEventLoop.__del__

    def safe_del(self):
        try:
            original_del(self)
        except ValueError as exc:
            if "Invalid file descriptor" not in str(exc):
                raise

    asyncio.BaseEventLoop.__del__ = safe_del


_silence_asyncio_invalid_fd_warning()

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse

from agent import run_agent
from schema import ExplainRequest
import memory as memory_layer
from safety import (
    has_profanity,
    sanitize_name,
    sanitize_situation,
)

MODAL_DRAFTER_URL = os.environ.get(
    "MODAL_DRAFTER_URL",
    "https://khoitruong071510--fabella-serve-drafter.modal.run",
)
MODAL_JUDGE_URL = os.environ.get(
    "MODAL_JUDGE_URL",
    "https://khoitruong071510--fabella-serve-judge.modal.run",
)
MODAL_TTS_URL = os.environ.get(
    "MODAL_TTS_URL",
    "https://khoitruong071510--fabella-serve-tts.modal.run",
)

# HF Spaces bucket mount. The current Space has a writable bucket mounted at
# /models. We avoid a separate database cloud API by storing one minimal JSON
# file per parent (HF user or anonymous browser session) inside that bucket.
# This keeps persistence tied to HF infrastructure only.
DATA_DIR = Path(os.environ.get("FABELLA_DATA_DIR", "/data/fabella-data"))
_history_lock = Lock()
_MAX_MESSAGES = 80
_SAFE_KEY = re.compile(r"[^a-zA-Z0-9._-]+")

try:
    from huggingface_hub import attach_huggingface_oauth, parse_huggingface_oauth
except Exception:
    attach_huggingface_oauth = None
    parse_huggingface_oauth = None

try:
    import spaces

    @spaces.GPU(duration=1)
    def _gpu_placeholder() -> str:
        return "ok"
except ImportError:
    pass

from gradio import Server

app = Server()
demo = app
if attach_huggingface_oauth is not None:
    try:
        attach_huggingface_oauth(app)
    except Exception as e:
        print(f"[auth] OAuth endpoints disabled: {type(e).__name__}: {e}", flush=True)


# Start the trace publisher. It no-ops on local dev (no HF_TOKEN) and is
# safe to call when HF_TOKEN is missing. See trace.py for the schema,
# anonymization rules, and opt-out semantics.
try:
    import trace as _trace

    _trace.publisher.start()
except Exception as e:
    print(f"[traces] publisher failed to start: {type(e).__name__}: {e}", flush=True)





def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _oauth_user(request: Request) -> dict:
    if parse_huggingface_oauth is None:
        return {"signed_in": False}
    try:
        info = parse_huggingface_oauth(request)
    except Exception as e:
        print(f"[auth] parse failed: {type(e).__name__}: {e}", flush=True)
        return {"signed_in": False}
    if info is None:
        return {"signed_in": False}
    user = getattr(info, "user_info", None)
    username = getattr(user, "preferred_username", None) or getattr(user, "name", None) or getattr(user, "sub", None)
    if not username:
        return {"signed_in": False}
    return {
        "signed_in": True,
        "username": str(username),
        "display_name": str(getattr(user, "name", "") or username),
        "avatar_url": str(getattr(user, "picture", "") or ""),
    }


def _owner_from_request(request: Request, session_id: str) -> tuple[str, str | None, str]:
    user = _oauth_user(request)
    clean_session = (session_id or "").strip()[:80]
    if not clean_session:
        clean_session = str(uuid.uuid4())
    if user.get("signed_in"):
        return f"hf:{user['username']}", user["username"], clean_session
    return f"anon:{clean_session}", None, clean_session


def _safe_slug(value: str) -> str:
    cleaned = _SAFE_KEY.sub("-", value).strip("-")
    return (cleaned or "anon")[:80]


def _history_path(owner_key: str) -> Path:
    return DATA_DIR / f"user-{_safe_slug(owner_key)}.json"


def _empty_history() -> dict:
    return {"messages": [], "profile": None, "updated_at": None}


def _read_history(owner_key: str) -> dict:
    path = _history_path(owner_key)
    if not path.exists():
        return _empty_history()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[history] read failed for {owner_key}: {type(e).__name__}: {e}", flush=True)
        return _empty_history()
    if not isinstance(data, dict):
        return _empty_history()
    data.setdefault("messages", [])
    data.setdefault("profile", None)
    return data


def _write_history(owner_key: str, data: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = _history_path(owner_key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _public_messages(messages: list) -> list:
    return [
        {
            "role": m.get("role"),
            "content": m.get("content", ""),
            "age": m.get("age"),
            "child_name": m.get("child_name") or "",
            "tone": m.get("tone") or "",
            "created_at": m.get("created_at", ""),
        }
        for m in messages[-_MAX_MESSAGES:]
    ]


def _sectioned_to_text(sectioned: str) -> str:
    parts = (sectioned or "").split(SECTION_SEP)
    labels = ["Opener", "Body", "Closer", "If they ask more"]
    lines = []
    for label, part in zip(labels, parts):
        text = (part or "").strip()
        if text:
            lines.append(f"{label}: {text}")
    return "\n\n".join(lines)

def _make_drafter(seed: int = 0):
    from llm import FabellaVLLM
    return FabellaVLLM(base_url=MODAL_DRAFTER_URL, model_name="gemma-4", seed=seed)


def _make_judge(seed: int = 0):
    from llm import FabellaVLLM
    return FabellaVLLM(
        base_url=MODAL_JUDGE_URL,
        model_name="nemotron-3-4b",
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
        seed=seed,
    )


# Sections joined with U+001F (Unit Separator) so the frontend can split
# reliably on a character that never appears in natural text.
SECTION_SEP = "\x1f"


def _make_explanation_sync(situation: str, age: int, child_name: str, tone: str, seed: int, history: list | None = None, owner_key: str = "", session_id: str = "", share_trace: bool = True) -> str:
    clean_situation = sanitize_situation(situation)
    clean_name = sanitize_name(child_name)
    clean_tone = (tone or "gentle").strip().lower()
    if clean_tone not in ("gentle", "matter-of-fact", "playful"):
        clean_tone = "gentle"

    if has_profanity(clean_situation) or has_profanity(clean_name):
        return SECTION_SEP.join([
            "Fabella (safety)",
            "Some of the words used aren't allowed. Please try different words.",
            "", "",
        ])

    if not clean_situation:
        return SECTION_SEP.join([
            "Fabella (empty)",
            "Tell me about the situation first - what's going on, in a sentence or two?",
            "", "",
        ])

    if not (5 <= int(age) <= 12):
        age = 7  # default for out-of-range

    clean_history = []
    if isinstance(history, list):
        for item in history[-6:]:
            if not isinstance(item, dict):
                continue
            role = (item.get("role") or "").strip().lower()
            content = (item.get("content") or "").strip()
            if role in ("parent", "fabella") and content:
                clean_history.append({"role": role, "content": content[:1200]})

    # Merge in durable memory from the bucket so the drafter can use it.
    if not owner_key:
        owner_key = f"anon:{(session_id or 'anon')[:80]}"
    mem = memory_layer.read_memory(owner_key)
    mem_block = memory_layer.memory_context_block(mem)
    if mem_block:
        clean_history = [{"role": "memory", "content": mem_block}] + clean_history

    req = ExplainRequest(
        situation=clean_situation,
        age=int(age),
        child_name=clean_name,
        tone=clean_tone,
        seed=int(seed or 0),
        history=clean_history,
        share_trace=bool(share_trace),
    )
    try:
        print(
            f"[app] request: age={req.age} tone={req.tone} name='{req.child_name}' situation='{req.situation[:60]}…'",
            flush=True,
        )
        drafter = _make_drafter(seed=req.seed)
        judge = _make_judge(seed=req.seed)
        result = run_agent(drafter, req, judge_llm=judge)
        return SECTION_SEP.join([
            result.get("opener", "") or "",
            result.get("body", "") or "",
            result.get("closer", "") or "",
            result.get("followup", "") or "",
        ])
    except Exception as e:
        print(f"[app] handler error: {type(e).__name__}: {e}", flush=True)
        return SECTION_SEP.join([
            "Fabella (error)",
            f"_Generation failed: {type(e).__name__}: {e}_",
            "", "",
        ])


@app.api(name="make_explanation")
def make_explanation(
    situation: str,
    age: int,
    child_name: str,
    tone: str,
    seed: int,
    history: list | None = None,
    owner_key: str = "",
    session_id: str = "",
    share_trace: bool = True,
) -> str:
    """Draft a short, kind, age-appropriate explanation.

    Returns four sections joined by U+001F:
      0: opener (one sentence the parent can say to start)
      1: body (1-3 short paragraphs)
      2: closer (one sentence the parent can say to land)
      3: follow-up (optional one sentence if the child has another question)

    `history` is a list of recent parent/Fabella turns so follow-up questions
    in the same conversation can be answered with the previous explanation
    in context. The drafter sees the last 6 turns. Long-term memory from
    the bucket (summary, preferences, durable facts) is folded in
    automatically when `owner_key` is provided.

    `share_trace` (default True) opts the request into the public,
    anonymized agent-trace dataset (see trace.py). Pass False to skip
    publishing this request. The global kill switch is FABELLA_SHARE_TRACES=0.
    """
    return _make_explanation_sync(
        situation, age, child_name, tone, seed, history, owner_key, session_id, share_trace
    )


def _clean_audio_text(text: str) -> str:
    clean = " ".join((text or "").split())
    if len(clean) > 1600:
        clean = clean[:1600].rsplit(" ", 1)[0].strip()
    return clean


def _narration_from_sections(opener: str, body: str, closer: str, followup: str) -> str:
    """Join the four sections into one plain conversational narration.

    The TTS must never speak the structural labels ("Opener", "Body", "Closer",
    "If they ask more"). It must never include a child's name. We address the
    listener as "you" only.
    """
    parts = []
    for piece in (opener, body, closer, followup):
        piece = (piece or "").strip()
        if piece:
            parts.append(piece)
    return " ".join(parts)


def _strip_labeled_draft(text: str) -> str:
    """Fallback: if a client sends the raw labeled draft, strip the labels."""
    if not text:
        return ""
    out = []
    for line in text.splitlines():
        stripped = re.sub(
            r"^\s*(Opener|Body|Closer|If they ask more|If they ask another question)\s*:\s*",
            "",
            line,
            flags=re.IGNORECASE,
        )
        if stripped.strip():
            out.append(stripped.strip())
    return " ".join(out).strip()


def _make_audio_sync(text: str, tone: str, opener: str = "", body: str = "", closer: str = "", followup: str = "") -> str:
    if opener or body or closer or followup:
        clean_text = _narration_from_sections(opener, body, closer, followup)
    else:
        clean_text = _strip_labeled_draft(text)
    clean_text = _clean_audio_text(clean_text)
    if not clean_text:
        return "ERROR: Nothing to read aloud yet."

    clean_tone = (tone or "gentle").strip().lower()
    voice_by_tone = {
        "gentle": "A calm adult woman with a soft, warm, reassuring voice, speaking slowly and clearly for a child.",
        "matter-of-fact": "A calm adult woman with a clear, steady, practical voice, speaking plainly and kindly for a child.",
        "playful": "A friendly adult woman with a lightly playful, bright, reassuring voice, speaking clearly for a child.",
    }
    payload = {
        "text": clean_text,
        "voice_description": voice_by_tone.get(clean_tone, voice_by_tone["gentle"]),
        "cfg_value": 2.0,
        "inference_timesteps": 10,
        "normalize": True,
        "denoise": True,
    }
    req = urllib.request.Request(
        MODAL_TTS_URL.rstrip("/") + "/synthesize",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "audio/wav"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as res:
            audio = res.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:300]
        print(f"[app] tts HTTP error: {e.code}: {detail}", flush=True)
        return f"ERROR: Read-aloud failed: HTTP {e.code}"
    except Exception as e:
        print(f"[app] tts error: {type(e).__name__}: {e}", flush=True)
        return f"ERROR: Read-aloud failed: {type(e).__name__}: {e}"

    if not audio:
        return "ERROR: Read-aloud returned no audio."
    return "data:audio/wav;base64," + base64.b64encode(audio).decode("ascii")


@app.api(name="make_audio")
def make_audio(text: str, tone: str, opener: str = "", body: str = "", closer: str = "", followup: str = "") -> str:
    """Synthesize a Fabella explanation as a base64 WAV data URL.

    The frontend should pass the four sections explicitly so the server can
    join them into a clean conversational narration. `text` is accepted as a
    legacy fallback and its labels are stripped before TTS.
    """
    return _make_audio_sync(text, tone, opener, body, closer, followup)



@app.get("/api/me")
async def api_me(request: Request):
    user = _oauth_user(request)
    return JSONResponse(user)


@app.get("/api/history")
async def api_history(request: Request, session_id: str = ""):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    owner_key, _, session_id = _owner_from_request(request, session_id)
    with _history_lock:
        data = _read_history(owner_key)
    return JSONResponse({
        "session_id": session_id,
        "profile": data.get("profile"),
        "messages": _public_messages(data.get("messages", [])),
    })


@app.post("/api/history/append")
async def api_history_append(request: Request):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = await request.json()
    session_id = str(payload.get("session_id") or "")
    owner_key, _, session_id = _owner_from_request(request, session_id)
    parent = sanitize_situation(payload.get("parent") or "")
    fabella = _clean_audio_text(payload.get("fabella") or "")
    if not parent and not fabella:
        return JSONResponse({"ok": False, "error": "nothing to store"}, status_code=400)
    try:
        age = int(payload.get("age") or 0) or None
    except Exception:
        age = None
    child_name = sanitize_name(payload.get("child_name") or "")
    tone = (payload.get("tone") or "gentle").strip().lower()
    if tone not in ("gentle", "matter-of-fact", "playful"):
        tone = "gentle"
    now = _now_iso()
    with _history_lock:
        data = _read_history(owner_key)
        if parent:
            data["messages"].append({
                "role": "parent",
                "content": parent,
                "age": age,
                "child_name": child_name,
                "tone": tone,
                "created_at": now,
            })
        if fabella:
            data["messages"].append({
                "role": "fabella",
                "content": fabella,
                "age": age,
                "child_name": child_name,
                "tone": tone,
                "created_at": now,
            })
        data["messages"] = data["messages"][-_MAX_MESSAGES:]
        profile = data.get("profile") or {}
        profile.update({
            "child_name": child_name or profile.get("child_name", ""),
            "child_age": age or profile.get("child_age"),
            "preferred_tone": tone or profile.get("preferred_tone", "gentle"),
            "updated_at": now,
        })
        data["profile"] = profile
        data["updated_at"] = now
        _write_history(owner_key, data)
    return JSONResponse({"ok": True, "session_id": session_id})


@app.post("/api/history/clear")
async def api_history_clear(request: Request):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = await request.json()
    session_id = str(payload.get("session_id") or "")
    owner_key, _, session_id = _owner_from_request(request, session_id)
    with _history_lock:
        path = _history_path(owner_key)
        if path.exists():
            try:
                path.unlink()
            except Exception as e:
                print(f"[history] clear failed for {owner_key}: {type(e).__name__}: {e}", flush=True)
    return JSONResponse({"ok": True, "session_id": session_id})


# --- Memory endpoints -------------------------------------------------------


@app.get("/api/memory")
async def api_memory(request: Request, session_id: str = ""):
    owner_key, _, session_id = _owner_from_request(request, session_id)
    mem = memory_layer.read_memory(owner_key)
    return JSONResponse({
        "session_id": session_id,
        "memory": memory_layer.public_view(mem),
    })


@app.post("/api/memory/append")
async def api_memory_append(request: Request):
    payload = await request.json()
    session_id = str(payload.get("session_id") or "")
    owner_key, _, session_id = _owner_from_request(request, session_id)
    parent = sanitize_situation(payload.get("parent") or "")
    fabella = _clean_audio_text(payload.get("fabella") or "")
    preferences = {}
    if "child_name" in payload:
        preferences["child_name"] = sanitize_name(payload.get("child_name") or "")
    if "child_age" in payload:
        try:
            age = int(payload.get("child_age") or 0)
            if 3 <= age <= 18:
                preferences["child_age"] = age
        except Exception:
            pass
    if "preferred_tone" in payload:
        tone = (payload.get("preferred_tone") or "").strip().lower()
        if tone in ("gentle", "matter-of-fact", "playful"):
            preferences["preferred_tone"] = tone
    if not parent and not fabella:
        return JSONResponse({"ok": False, "error": "nothing to store"}, status_code=400)
    res = memory_layer.append_turn(owner_key, parent, fabella, preferences=preferences)
    return JSONResponse({
        "ok": True,
        "session_id": session_id,
        "extraction": res.extraction,
        "memory": memory_layer.public_view(res.memory),
    })


@app.post("/api/memory/clear")
async def api_memory_clear(request: Request):
    payload = await request.json()
    session_id = str(payload.get("session_id") or "")
    owner_key, _, session_id = _owner_from_request(request, session_id)
    memory_layer.clear_memory(owner_key)
    return JSONResponse({"ok": True, "session_id": session_id})


# --- HTML page --------------------------------------------------------------

TONE_CHOICES = [
    ("gentle", "Gentle"),
    ("matter-of-fact", "Matter-of-fact"),
    ("playful", "Playful"),
]

EXAMPLE_SITUATIONS = [
    ("My 7-year-old's grandma is in the hospital for surgery. She asked why grandma is there."),
    ("We're moving to a new house in 3 weeks. My kid is worried about leaving her friends."),
    ("My child's dog died yesterday. She keeps asking when the dog is coming back."),
    ("It's time to start sharing toys at preschool. My son refuses and has started hitting."),
    ("My 9-year-old wants a phone. All her friends have one and I said no."),
    ("There's a new baby coming in 4 months. My first grader is acting out and being mean to me."),
]

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
<title>Fabella - small words for big questions</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
:root {
  --bg: #f7f4ec;
  --bg-2: #efeae0;
  --surface: #ffffff;
  --surface-2: #f3efe6;
  --line: #e2dccd;
  --line-soft: #ebe6d8;
  --text: #1f2330;
  --text-soft: #3a3f4a;
  --text-muted: #6b6f78;
  --accent: #4f7a4a;
  --accent-strong: #2e5a36;
  --accent-soft: #dceadc;
  --bubble-parent: #4f7a4a;
  --bubble-parent-text: #ffffff;
  --bubble-fabella: #ffffff;
  --bubble-fabella-text: #1f2330;
  --danger: #b04a3a;
  --radius: 18px;
  --radius-sm: 12px;
  --pad-x: clamp(16px, 4vw, 40px);
  --shadow: 0 1px 0 rgba(0,0,0,0.02), 0 12px 30px -22px rgba(20,30,20,0.18);
  --ease: cubic-bezier(.2,.7,.2,1);
  --font-sans: "Outfit", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  --font-mono: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14181d;
    --bg-2: #1a1f25;
    --surface: #1d242b;
    --surface-2: #232a32;
    --line: #2d343d;
    --line-soft: #262c34;
    --text: #ece7d8;
    --text-soft: #cfd2cc;
    --text-muted: #8d9089;
    --accent: #8fbf83;
    --accent-strong: #b5d3a9;
    --accent-soft: #2a3a30;
    --bubble-parent: #2e5a36;
    --bubble-parent-text: #f1f1ec;
    --bubble-fabella: #232a32;
    --bubble-fabella-text: #ece7d8;
  }
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: var(--bg); color: var(--text); font-family: var(--font-sans); -webkit-font-smoothing: antialiased; }
body { min-height: 100vh; }
a { color: var(--accent-strong); }

.appbar {
  position: sticky;
  top: 0;
  z-index: 5;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px var(--pad-x);
  background: color-mix(in srgb, var(--bg) 85%, transparent);
  backdrop-filter: saturate(140%) blur(8px);
  border-bottom: 1px solid var(--line-soft);
}
.brand { display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 18px; letter-spacing: -0.01em; }
.brand-dot { width: 10px; height: 10px; border-radius: 50%; background: var(--accent); }
.brand small { color: var(--text-muted); font-weight: 500; margin-left: 6px; }
.appbar-actions { display: flex; align-items: center; gap: 8px; }
.chip {
  display: inline-flex; align-items: center; gap: 6px;
  font: 500 12px/1 var(--font-sans);
  padding: 8px 12px;
  border-radius: 999px;
  border: 1px solid var(--line);
  background: var(--surface);
  color: var(--text-soft);
  cursor: pointer;
}
.chip.is-active { border-color: var(--accent); color: var(--accent-strong); background: var(--accent-soft); }
.chip-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--text-muted); }
.chip-dot.is-on { background: var(--accent); }
.btn-ghost {
  border: 1px solid var(--line);
  background: var(--surface);
  color: var(--text-soft);
  border-radius: 999px;
  padding: 8px 14px;
  font: 500 13px var(--font-sans);
  cursor: pointer;
}
.btn-ghost:hover { color: var(--text); border-color: var(--accent); }

.thread {
  max-width: 760px;
  margin: 0 auto;
  padding: clamp(28px, 4vw, 48px) var(--pad-x) 200px;
  display: flex;
  flex-direction: column;
  gap: 22px;
}
.welcome { margin-top: 6vh; }
.welcome h1 {
  font-size: clamp(32px, 5vw, 44px);
  line-height: 1.05;
  letter-spacing: -0.02em;
  margin: 0 0 10px;
  font-weight: 700;
}
.welcome h1 em { font-style: normal; color: var(--accent-strong); }
.welcome p { color: var(--text-soft); margin: 0 0 22px; max-width: 56ch; }
.examples { display: flex; flex-direction: column; gap: 8px; max-width: 56ch; }
.example-btn {
  text-align: left;
  background: var(--surface);
  border: 1px solid var(--line);
  color: var(--text);
  border-radius: var(--radius-sm);
  padding: 12px 14px;
  font: 500 14px var(--font-sans);
  cursor: pointer;
}
.example-btn:hover { border-color: var(--accent); background: color-mix(in srgb, var(--accent-soft) 40%, var(--surface)); }

.turn { display: flex; gap: 12px; align-items: flex-start; }
.turn.user { flex-direction: row-reverse; }
.avatar {
  width: 32px; height: 32px;
  flex: 0 0 32px;
  border-radius: 50%;
  background: var(--surface-2);
  display: flex; align-items: center; justify-content: center;
  font: 600 12px var(--font-sans);
  color: var(--text-muted);
  border: 1px solid var(--line-soft);
}
.avatar.fabella { background: var(--accent-soft); color: var(--accent-strong); border-color: transparent; }
.avatar.user { background: var(--bubble-parent); color: var(--bubble-parent-text); border-color: transparent; }
.bubble {
  max-width: min(72ch, calc(100% - 56px));
  padding: 14px 16px;
  border-radius: 18px;
  background: var(--bubble-fabella);
  color: var(--bubble-fabella-text);
  border: 1px solid var(--line);
  line-height: 1.5;
  font-size: 15.5px;
  box-shadow: var(--shadow);
}
.turn.user .bubble {
  background: var(--bubble-parent);
  color: var(--bubble-parent-text);
  border-color: transparent;
  border-bottom-right-radius: 6px;
}
.turn.fabella .bubble { border-bottom-left-radius: 6px; }
.bubble p { margin: 0 0 8px; }
.bubble p:last-child { margin-bottom: 0; }

.section-heading {
  margin: 14px 0 4px;
  font: 600 12.5px var(--font-sans);
  color: var(--text-muted);
  letter-spacing: 0.02em;
  text-transform: none;
}
.section-heading:first-child { margin-top: 0; }
.section-text { white-space: pre-wrap; }

.turn-actions {
  display: flex;
  gap: 8px;
  margin-top: 10px;
  flex-wrap: wrap;
}
.btn-read {
  border: 1px solid var(--accent);
  background: var(--surface);
  color: var(--accent-strong);
  border-radius: 999px;
  padding: 6px 12px;
  font: 500 12.5px var(--font-sans);
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.btn-read:hover { background: var(--accent-soft); }
.btn-read[disabled] { opacity: 0.6; cursor: progress; }
.btn-read .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); }
.audio-inline { width: 100%; margin-top: 8px; display: none; }
.audio-inline.is-on { display: block; }
.audio-status { font: 500 11px var(--font-mono); color: var(--text-muted); margin-top: 4px; letter-spacing: 0.04em; }

.composer {
  position: fixed;
  left: 0; right: 0; bottom: 0;
  padding: 14px var(--pad-x) 22px;
  background: linear-gradient(180deg, transparent, var(--bg) 24px);
  z-index: 4;
}
.composer-inner {
  max-width: 760px;
  margin: 0 auto;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 22px;
  box-shadow: 0 24px 60px -28px rgba(20,30,20,0.18), 0 2px 0 rgba(0,0,0,0.02);
  padding: 10px 12px 10px 16px;
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 10px;
  align-items: end;
}
.composer textarea {
  width: 100%;
  resize: none;
  border: 0;
  outline: 0;
  background: transparent;
  font: 400 15px var(--font-sans);
  color: var(--text);
  padding: 8px 0;
  min-height: 28px;
  max-height: 200px;
}
.composer textarea::placeholder { color: var(--text-muted); }
.composer-controls { display: flex; align-items: center; gap: 8px; }
.btn-send {
  border: 0;
  background: var(--accent);
  color: #ffffff;
  border-radius: 999px;
  padding: 10px 18px;
  font: 600 14px var(--font-sans);
  cursor: pointer;
  display: inline-flex; align-items: center; gap: 8px;
}
.btn-send[disabled] { opacity: 0.55; cursor: not-allowed; }
.btn-send:hover:not([disabled]) { background: var(--accent-strong); }
.composer-hint {
  font: 500 11px var(--font-mono);
  color: var(--text-muted);
  letter-spacing: 0.06em;
  text-align: center;
  margin-top: 8px;
}

.typing {
  display: inline-flex; align-items: center; gap: 6px;
  font: 500 13px var(--font-sans); color: var(--text-muted);
}
.typing-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--text-muted); animation: bounce 1.2s infinite; }
.typing-dot:nth-child(2) { animation-delay: 0.15s; }
.typing-dot:nth-child(3) { animation-delay: 0.3s; }
@keyframes bounce {
  0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
  40% { transform: translateY(-4px); opacity: 1; }
}
.error {
  border: 1px solid color-mix(in srgb, var(--danger) 40%, transparent);
  color: var(--danger);
  background: color-mix(in srgb, var(--danger) 8%, var(--surface));
  border-radius: var(--radius-sm);
  padding: 10px 12px;
  font: 500 13px var(--font-sans);
}

.foot {
  text-align: center;
  font: 500 11px var(--font-mono);
  color: var(--text-muted);
  letter-spacing: 0.08em;
  padding: 0 0 8px;
  text-transform: uppercase;
}
.foot a { color: var(--text-muted); text-decoration: none; border-bottom: 1px dotted var(--line); }
.foot a:hover { color: var(--text); }
</style>
</head>
<body>

<header class="appbar" role="banner">
  <div class="brand">
    <span class="brand-dot" aria-hidden="true"></span>
    <span>Fabella <small>small words for big questions</small></span>
  </div>
  <div class="appbar-actions">
    <button type="button" class="chip" id="age-chip"><span class="chip-dot" aria-hidden="true"></span>Age <span id="age-chip-val">7</span></button>
    <button type="button" class="chip" id="tone-chip"><span class="chip-dot" aria-hidden="true"></span>Tone <span id="tone-chip-val">gentle</span></button>
    <button type="button" class="btn-ghost" id="open-settings">Settings</button>
    <button type="button" class="btn-ghost" id="clear-history">Clear</button>
  </div>
</header>

<main id="thread" class="thread" aria-live="polite"></main>

<section class="composer" aria-label="Composer">
  <form id="composer" class="composer-inner" novalidate>
    <textarea id="input" rows="1" placeholder="Describe the hard situation. A sentence or two is enough." maxlength="800" required></textarea>
    <div class="composer-controls">
      <button type="submit" class="btn-send" id="send-btn"><span id="send-label">Draft</span><span aria-hidden="true">&rarr;</span></button>
    </div>
  </form>
  <div class="composer-hint">Fabella checks every draft against a six-criterion rubric. Read-aloud uses VoxCPM2 on demand.</div>
</section>

<dialog id="settings" style="border:1px solid var(--line); border-radius: var(--radius); padding: 0; max-width: 480px; width: calc(100% - 32px); background: var(--surface); color: var(--text);">
  <form method="dialog" style="padding: 22px;">
    <h2 style="margin: 0 0 12px; font-size: 18px;">Settings</h2>
    <label style="display:block; font: 500 12px var(--font-mono); color: var(--text-muted); margin: 12px 0 6px; letter-spacing: 0.06em; text-transform: uppercase;">Child's age</label>
    <input id="age-range" type="range" min="5" max="12" step="1" value="7" style="width:100%;">
    <div style="font: 500 12px var(--font-sans); color: var(--text-soft); margin-top: 4px;"><span id="age-readout">7</span> years</div>
    <label style="display:block; font: 500 12px var(--font-mono); color: var(--text-muted); margin: 16px 0 6px; letter-spacing: 0.06em; text-transform: uppercase;">Child's name (optional)</label>
    <input id="child-name" type="text" maxlength="30" placeholder="leave empty to address the parent" style="width:100%; padding: 10px 12px; border:1px solid var(--line); border-radius: 10px; background: var(--bg); color: var(--text); font: 400 14px var(--font-sans);">
    <label style="display:block; font: 500 12px var(--font-mono); color: var(--text-muted); margin: 16px 0 6px; letter-spacing: 0.06em; text-transform: uppercase;">Tone</label>
    <div id="tone-row" style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap: 6px;"></div>
    <div style="display:flex; gap: 8px; justify-content: flex-end; margin-top: 18px;">
      <button type="button" class="btn-ghost" id="settings-cancel">Close</button>
      <button type="submit" class="btn-send" style="padding: 8px 14px;">Save</button>
    </div>
  </form>
</dialog>

<footer class="foot">
  Built for the <a href="https://huggingface.co/spaces/build-small-hackathon/README" target="_blank" rel="noopener">Build Small Hackathon</a>
</footer>

<script>
(function () {
  "use strict";

  var TONE_CHOICES = __TONE_CHOICES__;
  var EXAMPLES = __EXAMPLES__;
  var SECTION_SEP = "\x1f";
  var SESSION_KEY = "fabella_session_id";

  var sessionId = (function () {
    try {
      var existing = localStorage.getItem(SESSION_KEY);
      if (!existing) {
        existing = (crypto && crypto.randomUUID) ? crypto.randomUUID() : String(Date.now()) + "-" + Math.random().toString(16).slice(2);
        localStorage.setItem(SESSION_KEY, existing);
      }
      return existing;
    } catch (_) {
      return String(Date.now()) + "-" + Math.random().toString(16).slice(2);
    }
  })();

  var thread = document.getElementById("thread");
  var input = document.getElementById("input");
  var sendBtn = document.getElementById("send-btn");
  var sendLabel = document.getElementById("send-label");
  var ageChip = document.getElementById("age-chip");
  var ageChipVal = document.getElementById("age-chip-val");
  var toneChip = document.getElementById("tone-chip");
  var toneChipVal = document.getElementById("tone-chip-val");
  var clearBtn = document.getElementById("clear-history");

  var dlg = document.getElementById("settings");
  var ageRange = document.getElementById("age-range");
  var ageReadout = document.getElementById("age-readout");
  var childName = document.getElementById("child-name");
  var toneRow = document.getElementById("tone-row");
  var settingsCancel = document.getElementById("settings-cancel");
  var openSettings = document.getElementById("open-settings");

  var currentAge = 7;
  var currentTone = "gentle";
  var childNameValue = "";

  function escapeHTML(s) {
    return String(s).replace(/[&<>"']/g, function (c) { return ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[c]; });
  }

  function autosize() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 200) + "px";
  }
  input.addEventListener("input", autosize);

  function syncChips() {
    ageChipVal.textContent = String(currentAge);
    toneChipVal.textContent = currentTone;
  }

  function buildTones() {
    toneRow.innerHTML = "";
    TONE_CHOICES.forEach(function (pair) {
      var val = pair[0];
      var label = pair[1];
      var id = "tone-" + val;
      var wrap = document.createElement("label");
      wrap.htmlFor = id;
      wrap.style.cssText = "display:flex; align-items:center; justify-content:center; padding:10px; border:1px solid var(--line); border-radius: 10px; cursor:pointer; font: 500 13px var(--font-sans); color: var(--text-soft); background: var(--bg);";
      wrap.dataset.tone = val;
      var r = document.createElement("input");
      r.type = "radio"; r.name = "tone"; r.id = id; r.value = val; r.style.display = "none";
      r.checked = (val === currentTone);
      if (r.checked) wrap.style.cssText += " border-color: var(--accent); color: var(--accent-strong); background: var(--accent-soft);";
      r.addEventListener("change", function () {
        Array.from(toneRow.children).forEach(function (c) { c.style.cssText = "display:flex; align-items:center; justify-content:center; padding:10px; border:1px solid var(--line); border-radius: 10px; cursor:pointer; font: 500 13px var(--font-sans); color: var(--text-soft); background: var(--bg);"; });
        wrap.style.cssText += " border-color: var(--accent); color: var(--accent-strong); background: var(--accent-soft);";
        currentTone = val;
        syncChips();
      });
      wrap.appendChild(r);
      wrap.appendChild(document.createTextNode(label));
      toneRow.appendChild(wrap);
    });
  }

  function welcomeOrThread(messages) {
    thread.innerHTML = "";
    if (!messages || !messages.length) {
      var w = document.createElement("section");
      w.className = "welcome";
      w.innerHTML =
        '<h1>Tell me the <em>hard thing</em>.</h1>' +
        '<p>Fabella drafts a short, kind, age-appropriate explanation. A second small model checks it. You can read it aloud with one tap.</p>' +
        '<div class="examples" id="examples"></div>';
      thread.appendChild(w);
      var ex = document.getElementById("examples");
      EXAMPLES.forEach(function (s) {
        var b = document.createElement("button");
        b.type = "button";
        b.className = "example-btn";
        b.textContent = s;
        b.addEventListener("click", function () {
          input.value = s;
          autosize();
          input.focus();
        });
        ex.appendChild(b);
      });
      return;
    }
    messages.forEach(function (m) {
      if (m.role === "parent") {
        thread.appendChild(renderUserTurn(m.content));
      } else {
        thread.appendChild(renderFabellaTurn({ body: m.content }, null));
      }
    });
  }

  function renderUserTurn(text) {
    var wrap = document.createElement("article");
    wrap.className = "turn user";
    wrap.innerHTML =
      '<div class="avatar user" aria-hidden="true">You</div>' +
      '<div class="bubble"><div class="text">' + escapeHTML(text).replace(/\n/g, "<br>") + '</div></div>';
    return wrap;
  }

  function parseFabellaText(text) {
    var out = [];
    var re = /(Opener|Body|Closer|If they ask more):\s*([\s\S]*?)(?=(?:\n\s*\n)(?:Opener|Body|Closer|If they ask more):|$)/g;
    var m;
    while ((m = re.exec(text)) !== null) {
      out.push({ label: m[1], text: m[2].trim() });
    }
    return out;
  }
  function renderFabellaTurn(sections, audioUrl) {
    var wrap = document.createElement("article");
    wrap.className = "turn fabella";
    var opener = (sections && sections.opener) || "";
    var body = (sections && sections.body) || "";
    var closer = (sections && sections.closer) || "";
    var followup = (sections && sections.followup) || "";
    function para(text) {
      return '<p>' + escapeHTML(text).replace(/\n/g, "<br>") + '</p>';
    }
    var sectionsHTML =
      (opener ? '<h3 class="section-heading">Where to begin</h3>' + para(opener) : "") +
      (body ? '<h3 class="section-heading">The explanation</h3>' + body.split(/\n\s*\n/).filter(Boolean).map(function (p) { return para(p.trim()); }).join("") : "") +
      (closer ? '<h3 class="section-heading">How to land it</h3>' + para(closer) : "") +
      (followup ? '<h3 class="section-heading">If they ask another question</h3>' + para(followup) : "");
    if (!sectionsHTML) {
      sectionsHTML = '<div class="section-text">' + escapeHTML(String(sections || "")) + '</div>';
    }
    wrap.innerHTML =
      '<div class="avatar fabella" aria-hidden="true">F</div>' +
      '<div class="bubble">' + sectionsHTML +
        '<div class="turn-actions">' +
          '<button type="button" class="btn-read" data-action="read"><span class="dot"></span>Read aloud</button>' +
        '</div>' +
        '<audio class="audio-inline" controls></audio>' +
        '<div class="audio-status" data-status></div>' +
      '</div>';
    var readBtn = wrap.querySelector('[data-action="read"]');
    var audio = wrap.querySelector("audio.audio-inline");
    var status = wrap.querySelector("[data-status]");
    if (audioUrl) {
      audio.src = audioUrl;
      audio.classList.add("is-on");
    }
    readBtn.addEventListener("click", async function () {
      if (readBtn.dataset.busy === "1") return;
      readBtn.dataset.busy = "1";
      readBtn.disabled = true;
      status.textContent = "Warming VoxCPM2 and preparing narration...";
      try {
        var url = await readGradioString("make_audio", [
          "",
          currentTone,
          opener,
          body,
          closer,
          followup,
        ]);
        if (url.startsWith("ERROR:")) throw new Error(url.slice(6).trim());
        audio.src = url;
        audio.classList.add("is-on");
        status.textContent = "Ready. Press play.";
        await audio.play().catch(function () {});
      } catch (err) {
        status.textContent = String(err.message || err);
      } finally {
        readBtn.disabled = false;
        readBtn.dataset.busy = "0";
      }
    });
    wrap.dataset.sections = JSON.stringify({ opener: opener, body: body, closer: closer, followup: followup });
    return wrap;
  }

  function renderTyping() {
    var wrap = document.createElement("article");
    wrap.className = "turn fabella";
    wrap.dataset.role = "typing";
    wrap.innerHTML =
      '<div class="avatar fabella" aria-hidden="true">F</div>' +
      '<div class="bubble"><div class="typing"><span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span><span style="margin-left:6px;">Fabella is drafting, then checking.</span></div></div>';
    return wrap;
  }

  function renderError(text) {
    var wrap = document.createElement("article");
    wrap.className = "turn fabella";
    wrap.innerHTML =
      '<div class="avatar fabella" aria-hidden="true">F</div>' +
      '<div class="bubble"><div class="error">' + escapeHTML(text) + '</div></div>';
    return wrap;
  }

  function scrollToEnd() {
    window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
  }

  function setBusy(busy) {
    sendBtn.disabled = busy;
    input.disabled = busy;
    sendLabel.textContent = busy ? "Drafting..." : "Draft";
  }

  function _extractSseText(obj) {
    if (obj == null) return null;
    if (Array.isArray(obj)) return obj.find(function (v) { return typeof v === "string"; }) || null;
    if (obj.msg === "process_completed" && obj.success) {
      var out = obj.output && obj.output.data;
      if (Array.isArray(out)) return out.find(function (v) { return typeof v === "string"; }) || null;
      if (typeof out === "string") return out;
    }
    return null;
  }

  async function readGradioString(apiName, data) {
    var res = await fetch("/gradio_api/call/" + apiName, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data: data }),
    });
    if (!res.ok) {
      var t = await res.text();
      throw new Error("HTTP " + res.status + ": " + t.slice(0, 200));
    }
    var evt0 = await res.json();
    var eventId = evt0.event_id;
    var evt = await fetch("/gradio_api/call/" + apiName + "/" + eventId, { headers: { Accept: "text/event-stream" } });
    if (!evt.ok || !evt.body) throw new Error("SSE open failed");
    var reader = evt.body.getReader();
    var dec = new TextDecoder();
    var buf = "";
    var result = null;
    var err = null;
    while (true) {
      var r = await reader.read();
      if (r.done) break;
      buf += dec.decode(r.value, { stream: true });
      var idx;
      while ((idx = buf.indexOf("\n\n")) !== -1) {
        var frame = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        var lines = frame.split("\n");
        var line = null;
        for (var li = 0; li < lines.length; li++) { if (lines[li].startsWith("data: ")) { line = lines[li]; break; } }
        if (!line) continue;
        var payload = line.slice(6).trim();
        if (payload === "null" || payload === "") continue;
        try {
          var obj = JSON.parse(payload);
          var text = _extractSseText(obj);
          if (text) { result = text; break; }
          if (obj && obj.msg === "process_completed" && obj.success === false) {
            err = (obj.output && obj.output.error) || "Request failed";
            break;
          }
        } catch (_) {}
      }
      if (result !== null || err) break;
    }
    if (err) throw new Error(err);
    if (!result) throw new Error("No result");
    return result;
  }

  var conversation = [];

  async function callMakeExplanation(seed, situationText) {
    var res = await fetch("/gradio_api/call/make_explanation", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data: [situationText, currentAge, childNameValue, currentTone, seed, conversation.slice(-6), "", sessionId, true] }),
    });
    if (!res.ok) {
      var t = await res.text();
      throw new Error("HTTP " + res.status + ": " + t.slice(0, 200));
    }
    var evt0 = await res.json();
    var eventId = evt0.event_id;
    var evt = await fetch("/gradio_api/call/make_explanation/" + eventId, { headers: { Accept: "text/event-stream" } });
    if (!evt.ok || !evt.body) throw new Error("SSE open failed");
    var reader = evt.body.getReader();
    var dec = new TextDecoder();
    var buf = "";
    var result = null;
    var err = null;
    while (true) {
      var r = await reader.read();
      if (r.done) break;
      buf += dec.decode(r.value, { stream: true });
      var idx;
      while ((idx = buf.indexOf("\n\n")) !== -1) {
        var frame = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        var lines = frame.split("\n");
        var line = null;
        for (var li = 0; li < lines.length; li++) { if (lines[li].startsWith("data: ")) { line = lines[li]; break; } }
        if (!line) continue;
        var payload = line.slice(6).trim();
        if (payload === "null" || payload === "") continue;
        try {
          var obj = JSON.parse(payload);
          var text = _extractSseText(obj);
          if (text) {
            result = text.split(SECTION_SEP);
            while (result.length < 4) result.push("");
            break;
          }
          if (obj && obj.msg === "process_completed" && obj.success === false) {
            err = (obj.output && obj.output.error) || "Generation failed";
            break;
          }
        } catch (_) {}
      }
      if (result !== null || err) break;
    }
    if (err) throw new Error(err);
    if (!result) throw new Error("No result");
    return result;
  }

  function narrationFromSections(sectionByLabel) {
    var parts = [];
    ["Opener", "Body", "Closer", "If they ask more"].forEach(function (k) {
      var v = (sectionByLabel && sectionByLabel[k]) ? sectionByLabel[k].trim() : "";
      if (v) parts.push(v);
    });
    return parts.join(" ");
  }

  function narrationFromSections(s) {
    var parts = [];
    ["opener", "body", "closer", "followup"].forEach(function (k) {
      var v = (s && s[k]) ? String(s[k]).trim() : "";
      if (v) parts.push(v);
    });
    return parts.join(" ");
  }

  async function saveTurn(parentText, structured) {
    var fabellaText = narrationFromSections(structured);
    try {
      await fetch("/api/history/append", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          parent: parentText,
          fabella: fabellaText,
          age: currentAge,
          child_name: childNameValue,
          tone: currentTone,
        }),
      });
    } catch (_) {}
  }

  async function saveMemory(parentText, fabellaText) {
    try {
      await fetch("/api/memory/append", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          parent: parentText,
          fabella: fabellaText,
          child_name: childNameValue,
          child_age: currentAge,
          preferred_tone: currentTone,
        }),
      });
    } catch (_) {}
  }

  async function loadHistory() {
    try {
      var r = await fetch("/api/history?session_id=" + encodeURIComponent(sessionId));
      if (!r.ok) return null;
      return await r.json();
    } catch (_) { return null; }
  }

  async function init() {
    buildTones();
    syncChips();
    var data = await loadHistory();
    if (data && data.profile) {
      if (data.profile.child_name) childNameValue = data.profile.child_name;
      if (data.profile.child_age) currentAge = data.profile.child_age;
      if (data.profile.preferred_tone) currentTone = data.profile.preferred_tone;
    }
    childName.value = childNameValue;
    ageRange.value = String(currentAge);
    ageReadout.textContent = String(currentAge);
    syncChips();
    welcomeOrThread(data && data.messages ? data.messages : []);
  }

  ageRange.addEventListener("input", function () {
    ageReadout.textContent = ageRange.value;
  });

  dlg.addEventListener("close", function () {
    currentAge = parseInt(ageRange.value, 10) || 7;
    childNameValue = childName.value.trim();
    var checked = dlg.querySelector("input[name='tone']:checked");
    if (checked) currentTone = checked.value;
    syncChips();
  });

  openSettings.addEventListener("click", function () { dlg.showModal(); });
  settingsCancel.addEventListener("click", function () { dlg.close(); });

  clearBtn.addEventListener("click", async function () {
    if (!confirm("Clear this conversation and long-term memory? This cannot be undone.")) return;
    try {
      await fetch("/api/history/clear", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      });
      await fetch("/api/memory/clear", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      });
    } catch (_) {}
    conversation = [];
    welcomeOrThread([]);
  });

  ageChip.addEventListener("click", function () { dlg.showModal(); });
  toneChip.addEventListener("click", function () { dlg.showModal(); });

  var seed = 0;
  var form = document.getElementById("composer");
  form.addEventListener("submit", async function (e) {
    e.preventDefault();
    var text = input.value.trim();
    if (!text) { input.focus(); return; }
    if (sendBtn.disabled) return;
    setBusy(true);
    thread.appendChild(renderUserTurn(text));
    var typing = renderTyping();
    thread.appendChild(typing);
    scrollToEnd();
    var parentText = text;
    input.value = "";
    autosize();
    try {
      var sections = await callMakeExplanation(seed, text);
      seed += 1;
      typing.remove();
      var structured = {
        opener: (sections[0] || "").trim(),
        body: (sections[1] || "").trim(),
        closer: (sections[2] || "").trim(),
        followup: (sections[3] || "").trim(),
      };
      thread.appendChild(renderFabellaTurn(structured, null));
      scrollToEnd();
      conversation.push({ role: "parent", content: parentText });
      var narrationText = narrationFromSections(structured);
      conversation.push({ role: "fabella", content: narrationText });
      if (conversation.length > 12) conversation = conversation.slice(-12);
      saveTurn(parentText, structured);
      saveMemory(parentText, narrationText);
    } catch (err) {
      typing.remove();
      thread.appendChild(renderError(String(err.message || err)));
      scrollToEnd();
    } finally {
      setBusy(false);
    }
  });

  init();
})();
</script>
</body>
</html>
"""
INDEX_HTML = (
    INDEX_HTML  # rebuilt for Phase 3 chat UI

    .replace("__TONE_CHOICES__", "[" + ",".join('["' + v + '","' + l + '"]' for v, l in TONE_CHOICES) + "]")
    .replace("__EXAMPLES__", "[" + ",".join('"' + s.replace('"', '\\"') + '"' for s in EXAMPLE_SITUATIONS) + "]")
)


@app.get("/", response_class=HTMLResponse)
async def homepage():
    return HTMLResponse(content=INDEX_HTML, status_code=200)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
