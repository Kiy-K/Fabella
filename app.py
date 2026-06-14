"""Fabella — small words for big questions.

A Gradio Server (FastAPI subclass) serves a custom HTML+CSS+JS page. The
parent describes a hard-to-explain situation; Fabella drafts a short,
kind, age-appropriate explanation, validated by a second small model.

Architecture (see modal_app.py for the server side):
  Gemma 4 E4B (drafter, A10G)  — writes the explanation
  Nemotron-3 Nano 4B (judge, A10G) — multi-criteria review
"""

import os
import sys
import asyncio
import traceback
import base64
import json
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


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

from fastapi.responses import HTMLResponse

from agent import run_agent
from schema import ExplainRequest
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

try:
    import spaces

    @spaces.GPU(duration=1)
    def _gpu_placeholder() -> str:
        return "ok"
except ImportError:
    pass

from gradio import Server

app = Server()


def _make_drafter(seed: int = 0):
    from llm import FabellaVLLM
    return FabellaVLLM(base_url=MODAL_DRAFTER_URL, model_name="gemma-4", seed=seed)


def _make_judge(seed: int = 0):
    from llm import FabellaVLLM
    return FabellaVLLM(base_url=MODAL_JUDGE_URL, model_name="nemotron-3-4b", seed=seed)


# Sections joined with U+001F (Unit Separator) so the frontend can split
# reliably on a character that never appears in natural text.
SECTION_SEP = "\x1f"


def _make_explanation_sync(situation: str, age: int, child_name: str, tone: str, seed: int) -> str:
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
            "Tell me about the situation first — what's going on, in a sentence or two?",
            "", "",
        ])

    if not (5 <= int(age) <= 12):
        age = 7  # default for out-of-range

    req = ExplainRequest(
        situation=clean_situation,
        age=int(age),
        child_name=clean_name,
        tone=clean_tone,
        seed=int(seed or 0),
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
def make_explanation(situation: str, age: int, child_name: str, tone: str, seed: int) -> str:
    """Draft a short, kind, age-appropriate explanation.

    Returns four sections joined by U+001F:
      0: opener (one sentence the parent can say to start)
      1: body (1-3 short paragraphs)
      2: closer (one sentence the parent can say to land)
      3: follow-up (optional one sentence if the child has another question)
    """
    return _make_explanation_sync(situation, age, child_name, tone, seed)


def _clean_audio_text(text: str) -> str:
    clean = " ".join((text or "").split())
    if len(clean) > 1600:
        clean = clean[:1600].rsplit(" ", 1)[0].strip()
    return clean


def _make_audio_sync(text: str, tone: str) -> str:
    clean_text = _clean_audio_text(text)
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
def make_audio(text: str, tone: str) -> str:
    """Synthesize a Fabella explanation as a base64 WAV data URL."""
    return _make_audio_sync(text, tone)


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
<title>Fabella — small words for big questions</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght,SOFT,WONK@9..144,300..900,0..100,0..1&family=Literata:ital,opsz,wght@0,7..72,400..800;1,7..72,400..800&family=Fragment+Mono&display=swap">
<style>
/* =========================================================================
   FABELLA — small words for big questions
   Cool palette, NOT the banned warm-cream+brass+espresso default.
   Same storybook-modernist language as before.
   ========================================================================= */

:root {
  --bone:       #ece9e0;
  --bone-deep:  #e3dfd3;
  --paper:      #f4f1e7;
  --paper-2:    #faf7ed;
  --ink:        #1a1d1f;
  --ink-soft:   #3d4144;
  --ink-mute:   #6b6e6f;
  --rule:       #c8c2b1;
  --rule-soft:  #d9d4c2;
  --forest:     #2d4a2b;
  --forest-ink: #1a2f18;
  --wax:        #a8341f;
  --wax-ink:    #7a2415;

  --serif: "Source Serif 4", "Iowan Old Style", Georgia, "Times New Roman", serif;
  --mono:  "JetBrains Mono", ui-monospace, "SF Mono", Menlo, monospace;

  --pad-x: clamp(20px, 4.5vw, 56px);
  --shadow-leaf: 0 1px 0 rgba(26,31,27,0.04), 0 12px 28px -18px rgba(26,31,27,0.18);
  --ease: cubic-bezier(0.16, 1, 0.3, 1);
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
html { background: var(--bone); }
body {
  color: var(--ink);
  font-family: var(--serif);
  font-size: 17px;
  line-height: 1.55;
  background: var(--bone);
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-rendering: optimizeLegibility;
  font-feature-settings: "kern", "liga", "onum";
  min-height: 100dvh;
  overflow-x: hidden;
}
body::before {
  content: "";
  position: fixed; inset: 0;
  pointer-events: none;
  z-index: 60;
  opacity: 0.5;
  mix-blend-mode: multiply;
  background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' seed='4'/><feColorMatrix values='0 0 0 0 0.55  0 0 0 0 0.50  0 0 0 0 0.40  0 0 0 0.07 0'/></filter><rect width='100%25' height='100%25' filter='url(%23n)'/></svg>");
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}

/* ---- header strip ---- */
.imprint {
  padding: 14px var(--pad-x);
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 20px;
  border-bottom: 1px solid var(--rule);
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--ink-soft);
  background: var(--bone);
}
.imprint .wordmark {
  font-family: var(--serif);
  font-style: italic;
  font-size: 19px;
  letter-spacing: 0.005em;
  text-transform: none;
  color: var(--ink);
  font-weight: 400;
}
.imprint .wordmark b { font-style: normal; font-weight: 700; color: var(--forest-ink); }
.imprint .meta { display: flex; gap: 22px; }
.imprint .meta span b { color: var(--ink); font-weight: 700; }

/* ---- main split ---- */
.stage {
  display: grid;
  grid-template-columns: minmax(0, 1.1fr) minmax(0, 1fr);
  gap: clamp(24px, 4vw, 64px);
  padding: clamp(28px, 5vw, 64px) var(--pad-x) clamp(40px, 6vw, 88px);
  max-width: 1280px;
  margin: 0 auto;
  align-items: start;
}
@media (max-width: 880px) { .stage { grid-template-columns: 1fr; } }

/* ---- left: form ---- */
.col-form { position: sticky; top: 28px; }
@media (max-width: 880px) { .col-form { position: static; } }
.title-block { margin-bottom: 28px; }
.title-block h1 {
  font-family: var(--serif);
  font-size: clamp(40px, 6.4vw, 64px);
  line-height: 0.98;
  letter-spacing: -0.015em;
  font-weight: 700;
  margin: 0 0 14px 0;
  color: var(--ink);
  text-wrap: balance;
}
.title-block h1 em { font-style: italic; font-weight: 400; color: var(--forest-ink); }
.title-block .lede {
  font-size: 17px;
  line-height: 1.55;
  color: var(--ink-soft);
  max-width: 40ch;
  margin: 0;
}
form { display: flex; flex-direction: column; gap: 22px; margin-top: 8px; }
.field { display: flex; flex-direction: column; gap: 8px; }
.label {
  font-family: var(--mono);
  font-size: 10.5px;
  font-weight: 700;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--ink-mute);
}
.label small { text-transform: none; letter-spacing: 0; font-weight: 400; font-family: var(--serif); font-style: italic; font-size: 12px; color: var(--ink-mute); margin-left: 6px; }
.hint {
  font-family: var(--serif);
  font-style: italic;
  font-size: 13px;
  color: var(--ink-mute);
  line-height: 1.5;
  margin-top: 2px;
}

input[type="text"], textarea {
  font-family: var(--serif);
  font-size: 17px;
  line-height: 1.45;
  color: var(--ink);
  background: var(--paper);
  border: 1px solid var(--rule);
  border-radius: 0;
  padding: 12px 14px;
  outline: none;
  transition: border-color 0.18s var(--ease), background 0.18s var(--ease);
  width: 100%;
  font-feature-settings: "kern", "liga";
}
textarea { min-height: 110px; resize: vertical; line-height: 1.55; }
input[type="text"]::placeholder, textarea::placeholder { color: var(--ink-mute); font-style: italic; }
input[type="text"]:focus, textarea:focus {
  border-color: var(--forest);
  background: var(--paper-2);
}

.age { display: grid; grid-template-columns: 1fr auto; align-items: baseline; gap: 14px; }
.age .readout {
  font-family: var(--serif);
  font-size: 30px;
  line-height: 1;
  color: var(--ink);
  font-feature-settings: "tnum";
  font-weight: 600;
}
.age .readout small { font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.18em; text-transform: uppercase; color: var(--ink-mute); margin-left: 6px; vertical-align: middle; }
.age input[type="range"] {
  grid-column: 1 / -1;
  -webkit-appearance: none; appearance: none;
  width: 100%; height: 4px;
  background: var(--rule);
  border-radius: 999px;
  outline: none;
}
.age input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none; appearance: none;
  width: 22px; height: 22px;
  border-radius: 50%;
  background: var(--forest);
  cursor: grab;
  border: 3px solid var(--paper);
  box-shadow: 0 0 0 1px var(--forest);
  transition: transform 0.15s var(--ease);
}
.age input[type="range"]::-webkit-slider-thumb:active { transform: scale(0.94); cursor: grabbing; }
.age input[type="range"]::-moz-range-thumb {
  width: 22px; height: 22px; border-radius: 50%;
  background: var(--forest); border: 3px solid var(--paper);
  box-shadow: 0 0 0 1px var(--forest);
}

/* tone radio as 3 segmented buttons */
.tone-row {
  display: grid; grid-template-columns: 1fr 1fr 1fr;
  border: 1px solid var(--rule);
  border-radius: 0;
  overflow: hidden;
  background: var(--paper);
}
.tone-row label {
  text-align: center;
  padding: 11px 8px;
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--ink-soft);
  cursor: pointer;
  border-right: 1px solid var(--rule);
  transition: background 0.18s var(--ease), color 0.18s var(--ease);
  font-weight: 500;
  user-select: none;
}
.tone-row label:last-child { border-right: none; }
.tone-row input { display: none; }
.tone-row label.is-on { background: var(--ink); color: var(--bone); }

/* example chips */
.examples { display: flex; flex-direction: column; gap: 6px; }
.ex-chip {
  font-family: var(--serif);
  font-size: 14px;
  line-height: 1.4;
  color: var(--ink-soft);
  background: var(--paper);
  border: 1px solid var(--rule-soft);
  padding: 8px 12px;
  cursor: pointer;
  text-align: left;
  border-radius: 0;
  transition: border-color 0.18s var(--ease), color 0.18s var(--ease), background 0.18s var(--ease);
}
.ex-chip:hover { border-color: var(--ink-soft); color: var(--ink); background: var(--paper-2); }
.ex-chip small { display: block; font-family: var(--mono); font-size: 9.5px; letter-spacing: 0.16em; text-transform: uppercase; color: var(--ink-mute); margin-bottom: 2px; }

/* submit */
.actions { display: flex; align-items: center; gap: 14px; margin-top: 6px; flex-wrap: wrap; }
.btn-primary {
  font-family: var(--serif);
  font-size: 17px;
  font-weight: 600;
  letter-spacing: 0.005em;
  color: var(--bone);
  background: var(--wax);
  border: 1px solid var(--wax-ink);
  padding: 13px 22px;
  border-radius: 0;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 10px;
  box-shadow: 0 1px 0 rgba(0,0,0,0.04), 0 4px 0 -1px var(--wax-ink);
  transition: transform 0.12s var(--ease), box-shadow 0.12s var(--ease), background 0.18s var(--ease);
  white-space: nowrap;
}
.btn-primary:hover { background: var(--wax-ink); }
.btn-primary:active { transform: translateY(2px); box-shadow: 0 1px 0 rgba(0,0,0,0.04), 0 2px 0 -1px var(--wax-ink); }
.btn-primary:disabled { background: var(--ink-mute); border-color: var(--ink-mute); box-shadow: 0 1px 0 rgba(0,0,0,0.04); cursor: progress; transform: none; }
.btn-primary .arrow { display: inline-block; transition: transform 0.18s var(--ease); }
.btn-primary:hover:not(:disabled) .arrow { transform: translateX(3px); }
.btn-ghost {
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--ink-soft);
  background: transparent;
  border: none;
  cursor: pointer;
  padding: 8px 4px;
  border-bottom: 1px dashed var(--ink-mute);
  transition: color 0.18s var(--ease), border-color 0.18s var(--ease);
}
.btn-ghost:hover { color: var(--ink); border-color: var(--ink); }
.btn-ghost:disabled { opacity: 0.4; cursor: not-allowed; }
.btn-read {
  font-family: var(--mono);
  font-size: 10.5px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--forest-ink);
  background: transparent;
  border: 1px solid var(--rule);
  padding: 9px 12px;
  cursor: pointer;
  transition: background 0.18s var(--ease), color 0.18s var(--ease), border-color 0.18s var(--ease);
}
.btn-read:hover { background: var(--paper-2); border-color: var(--forest); color: var(--ink); }
.btn-read:disabled { opacity: 0.45; cursor: progress; }
.seed-pill {
  font-family: var(--mono);
  font-size: 10.5px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--ink-mute);
  padding-left: 4px;
}

/* ---- right: book page ---- */
.col-story { position: relative; min-height: 60vh; }
.book {
  background: var(--paper);
  border: 1px solid var(--rule);
  box-shadow: var(--shadow-leaf);
  padding: clamp(28px, 4.5vw, 56px) clamp(24px, 4vw, 52px);
  position: relative;
}
.book::before {
  content: "";
  position: absolute; left: 18px; top: 18px; bottom: 18px; right: 18px;
  border: 1px solid var(--rule-soft);
  pointer-events: none;
}
.book .corner {
  position: absolute;
  width: 22px; height: 22px;
  pointer-events: none;
  color: var(--ink-mute);
}
.book .corner.tl { top: 10px; left: 10px; }
.book .corner.tr { top: 10px; right: 10px; transform: scaleX(-1); }
.book .corner.bl { bottom: 10px; left: 10px; transform: scaleY(-1); }
.book .corner.br { bottom: 10px; right: 10px; transform: scale(-1,-1); }
.book .inner { position: relative; z-index: 1; }

.cover { display: flex; flex-direction: column; gap: 18px; min-height: 380px; justify-content: center; }
.cover .folio { font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.22em; text-transform: uppercase; color: var(--ink-mute); }
.cover h2 {
  font-family: var(--serif);
  font-size: clamp(34px, 4.5vw, 50px);
  line-height: 1.02;
  letter-spacing: -0.012em;
  font-weight: 600;
  margin: 0;
  color: var(--ink);
  font-style: italic;
  font-weight: 400;
}
.cover h2 b { font-style: normal; font-weight: 700; color: var(--forest-ink); }
.cover p { font-size: 16px; line-height: 1.6; color: var(--ink-soft); max-width: 40ch; margin: 0; }

.writing { display: flex; flex-direction: column; gap: 14px; min-height: 380px; justify-content: center; }
.writing .penline {
  font-family: var(--mono);
  font-size: 10.5px;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--ink-mute);
  display: inline-flex; align-items: center; gap: 10px;
}
.writing .quill {
  display: inline-block; width: 14px; height: 14px; background: var(--ink);
  -webkit-mask: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><path d='M3 21l3.5-1 11-11a2.83 2.83 0 0 0-4-4l-11 11L3 21z' fill='currentColor'/></svg>") center/contain no-repeat;
          mask: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><path d='M3 21l3.5-1 1 1-11a2.83 2.83 0 0 0-4-4l-11 11L3 21z' fill='currentColor'/></svg>") center/contain no-repeat;
  transform-origin: 50% 80%;
  animation: nib 1.6s var(--ease) infinite;
}
@keyframes nib { 0%, 100% { transform: rotate(-8deg) translateX(0); } 50% { transform: rotate(14deg) translateX(2px); } }
@media (prefers-reduced-motion: reduce) { .writing .quill { animation: none; } }
.writing .progress {
  height: 1px;
  background: linear-gradient(90deg, var(--forest) 0%, var(--forest) var(--p,40%), var(--rule) var(--p,40%), var(--rule) 100%);
  width: 100%;
  max-width: 320px;
  transition: background 0.4s var(--ease);
}
.writing p { font-size: 16px; line-height: 1.6; color: var(--ink-soft); max-width: 40ch; margin: 0; }

/* the actual explanation page */
.expl { animation: arrive 0.55s var(--ease) both; }
@keyframes arrive { from { opacity: 0; transform: translateY(8px) rotate(-0.2deg); } to { opacity: 1; transform: translateY(0) rotate(0); } }
@media (prefers-reduced-motion: reduce) { .expl { animation: none; } }

.expl .byline {
  font-family: var(--mono);
  font-size: 10.5px;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--ink-mute);
  margin-bottom: 18px;
  display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
}
.expl .byline .dot { width: 4px; height: 4px; border-radius: 50%; background: var(--ink-mute); display: inline-block; }
.expl .byline .label-tiny { color: var(--ink-soft); font-weight: 700; }

.expl h2.title {
  font-family: var(--serif);
  font-size: clamp(28px, 3.6vw, 36px);
  line-height: 1.1;
  letter-spacing: -0.012em;
  font-weight: 700;
  margin: 0 0 22px 0;
  color: var(--ink);
  text-wrap: balance;
}
.expl h2.title small {
  display: block;
  font-family: var(--mono);
  font-size: 10.5px;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--ink-mute);
  font-weight: 400;
  margin-top: 8px;
}

/* the four sections, in order */
.section { margin-bottom: 18px; }
.section:last-child { margin-bottom: 0; }
.section .tag {
  font-family: var(--mono);
  font-size: 10px;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--forest-ink);
  font-weight: 700;
  margin-bottom: 6px;
  display: block;
}
.section .text {
  font-family: var(--serif);
  font-size: 18px;
  line-height: 1.55;
  color: var(--ink);
  margin: 0;
}
.section.opener .text {
  font-style: italic;
  color: var(--ink-soft);
  font-size: 17px;
  border-left: 2px solid var(--forest);
  padding: 2px 0 2px 14px;
}
.section.closer .text {
  font-weight: 600;
}
.section.body p { margin: 0 0 0.85em 0; }
.section.body p:last-child { margin-bottom: 0; }
.section.followup .text {
  font-size: 15px;
  color: var(--ink-mute);
  font-style: italic;
}
.section .text:empty { display: none; }
.section:has(.text:empty) { display: none; }

.expl .signoff {
  margin-top: 32px;
  padding-top: 18px;
  border-top: 1px solid var(--rule-soft);
  display: flex; align-items: baseline; justify-content: space-between;
  gap: 12px; flex-wrap: wrap;
  font-family: var(--mono);
  font-size: 10.5px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--ink-mute);
}
.expl .signoff .regen {
  color: var(--ink-soft);
  border-bottom: 1px dashed var(--ink-mute);
  cursor: pointer;
  padding: 0 0 1px 0;
  background: transparent;
  border-left: 0; border-right: 0; border-top: 0;
  font: inherit; letter-spacing: inherit; text-transform: inherit;
}
.expl .signoff .regen:hover { color: var(--ink); border-color: var(--ink); }
.expl .signoff .regen:disabled { opacity: 0.4; cursor: progress; }

.audio-panel {
  margin-top: 18px;
  padding-top: 18px;
  border-top: 1px solid var(--rule-soft);
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}
.audio-panel .audio-status {
  font-family: var(--serif);
  font-style: italic;
  font-size: 14px;
  color: var(--ink-mute);
}
.audio-panel audio {
  width: 100%;
  min-width: 220px;
  margin-top: 2px;
}

.banner {
  font-family: var(--serif);
  font-style: italic;
  font-size: 16px;
  line-height: 1.5;
  color: var(--wax-ink);
  border-left: 3px double var(--wax);
  padding: 4px 0 4px 14px;
}

.colophon {
  border-top: 1px solid var(--rule);
  padding: 22px var(--pad-x) 28px;
  font-family: var(--mono);
  font-size: 10.5px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--ink-mute);
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 20px;
  flex-wrap: wrap;
  max-width: 1280px;
  margin: 0 auto;
}
.colophon a { color: var(--ink-soft); text-decoration: none; border-bottom: 1px dotted var(--ink-mute); }
.colophon a:hover { color: var(--ink); border-color: var(--ink); }

/* =========================================================================
   REDESIGN — night observatory field guide
   A parent is not asking for a dashboard; they are trying to find a careful
   sentence in the dark. The interface should feel like a quiet instrument:
   ink, brass, star maps, and one illuminated page.
   ========================================================================= */

:root {
  --night:      #081019;
  --night-2:    #0d1924;
  --night-3:    #132333;
  --mist:       #dbe3dd;
  --mist-dim:   #aebbb7;
  --vellum:     #f1ead7;
  --vellum-2:   #fbf5e8;
  --vellum-3:   #e4d8bd;
  --ink:        #13202a;
  --ink-soft:   #314150;
  --ink-mute:   #67747d;
  --rule:       rgba(210, 188, 139, 0.42);
  --rule-soft:  rgba(210, 188, 139, 0.22);
  --forest:     #5f8e79;
  --forest-ink: #1d5d4f;
  --wax:        #d46a45;
  --wax-ink:    #8f381f;
  --gold:       #d8b56a;
  --bluefire:   #8cc7d8;
  --serif:      "Literata", "Iowan Old Style", Georgia, serif;
  --display:    "Fraunces", "Literata", Georgia, serif;
  --mono:       "Fragment Mono", "JetBrains Mono", ui-monospace, monospace;
  --pad-x:      clamp(18px, 4vw, 64px);
  --ease:       cubic-bezier(0.16, 1, 0.3, 1);
  --shadow-leaf: 0 34px 90px -48px rgba(0, 0, 0, 0.82), 0 0 0 1px rgba(216,181,106,0.14);
}

html { background: var(--night); }
body {
  color: var(--mist);
  background:
    radial-gradient(circle at 14% 16%, rgba(140,199,216,0.18) 0 12%, transparent 30%),
    radial-gradient(circle at 86% 8%, rgba(212,106,69,0.16) 0 10%, transparent 28%),
    radial-gradient(circle at 72% 86%, rgba(216,181,106,0.11) 0 11%, transparent 26%),
    linear-gradient(135deg, #060b12 0%, var(--night) 38%, #101b26 100%);
  isolation: isolate;
}
body::before {
  opacity: 0.38;
  mix-blend-mode: screen;
  background-image:
    radial-gradient(circle at 20px 24px, rgba(255,255,255,0.72) 0 1px, transparent 1.4px),
    radial-gradient(circle at 82px 68px, rgba(216,181,106,0.56) 0 1px, transparent 1.4px),
    url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='180' height='180'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.78' numOctaves='3' seed='11'/><feColorMatrix values='0 0 0 0 0.80 0 0 0 0 0.78 0 0 0 0 0.68 0 0 0 0.18 0'/></filter><rect width='100%25' height='100%25' filter='url(%23n)'/></svg>");
  background-size: 118px 118px, 172px 172px, 180px 180px;
}
body::after {
  content: "";
  position: fixed;
  inset: auto -8vw -18vh auto;
  width: min(62vw, 720px);
  height: min(62vw, 720px);
  pointer-events: none;
  z-index: -1;
  opacity: 0.34;
  border: 1px solid rgba(216,181,106,0.28);
  border-radius: 50%;
  background:
    linear-gradient(90deg, transparent 49.8%, rgba(216,181,106,0.24) 50%, transparent 50.2%),
    linear-gradient(0deg, transparent 49.8%, rgba(216,181,106,0.24) 50%, transparent 50.2%),
    radial-gradient(circle, transparent 0 44%, rgba(216,181,106,0.2) 44.2% 44.6%, transparent 45%),
    radial-gradient(circle, transparent 0 64%, rgba(216,181,106,0.16) 64.2% 64.6%, transparent 65%);
  transform: rotate(-11deg);
}

.imprint {
  position: relative;
  padding: 18px var(--pad-x);
  border-bottom: 1px solid rgba(216,181,106,0.2);
  background: rgba(8,16,25,0.72);
  color: var(--mist-dim);
  backdrop-filter: blur(18px);
}
.imprint::after {
  content: "";
  position: absolute;
  left: var(--pad-x);
  right: var(--pad-x);
  bottom: -1px;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--gold), transparent);
  opacity: 0.56;
}
.imprint .wordmark {
  color: var(--vellum-2);
  font-family: var(--display);
  font-size: clamp(20px, 2.2vw, 31px);
  font-style: normal;
  font-variation-settings: "SOFT" 75, "WONK" 1;
  letter-spacing: -0.025em;
}
.imprint .wordmark b { color: var(--gold); font-weight: 760; }
.imprint .meta { color: var(--mist-dim); opacity: 0.92; }
.imprint .meta span b { color: var(--bluefire); }

.stage {
  position: relative;
  max-width: 1380px;
  grid-template-columns: minmax(320px, 0.92fr) minmax(420px, 1.08fr);
  gap: clamp(28px, 5vw, 86px);
  padding-top: clamp(34px, 6vw, 84px);
}
.stage::before {
  content: "";
  position: absolute;
  top: 54px;
  left: calc(var(--pad-x) + 17px);
  width: 120px;
  height: 120px;
  opacity: 0.34;
  pointer-events: none;
  border-left: 1px solid var(--gold);
  border-top: 1px solid var(--gold);
  transform: rotate(-7deg);
}

.col-form {
  top: 34px;
  padding: clamp(22px, 3vw, 34px);
  background: linear-gradient(180deg, rgba(13,25,36,0.82), rgba(8,16,25,0.58));
  border: 1px solid rgba(216,181,106,0.22);
  box-shadow: 0 28px 78px -58px rgba(0,0,0,0.86);
  backdrop-filter: blur(16px);
}
.col-form::before {
  content: "Parent console / private draft";
  display: block;
  margin-bottom: 18px;
  font-family: var(--mono);
  font-size: 10px;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--gold);
}
.title-block { margin-bottom: 30px; }
.title-block h1 {
  color: var(--vellum-2);
  font-family: var(--display);
  font-size: clamp(48px, 6.8vw, 88px);
  line-height: 0.86;
  letter-spacing: -0.055em;
  font-weight: 820;
  font-variation-settings: "SOFT" 64, "WONK" 1;
  max-width: 8.4ch;
}
.title-block h1 em {
  color: var(--bluefire);
  font-style: italic;
  font-weight: 430;
}
.title-block .lede {
  color: var(--mist-dim);
  font-size: 16px;
  max-width: 45ch;
}
form { gap: 20px; }
.label {
  color: rgba(216,181,106,0.9);
  font-size: 9.5px;
}
.label small, .hint { color: rgba(219,227,221,0.56); }
input[type="text"], textarea {
  color: var(--vellum-2);
  background: rgba(5,10,16,0.42);
  border: 1px solid rgba(216,181,106,0.28);
  box-shadow: inset 0 0 0 1px rgba(255,255,255,0.025);
  border-radius: 18px 18px 18px 4px;
  padding: 14px 16px;
}
textarea { min-height: 138px; }
input[type="text"]::placeholder, textarea::placeholder { color: rgba(219,227,221,0.42); }
input[type="text"]:focus, textarea:focus {
  border-color: var(--bluefire);
  background: rgba(12,28,40,0.62);
  box-shadow: 0 0 0 4px rgba(140,199,216,0.1), inset 0 0 0 1px rgba(255,255,255,0.04);
}
.age .readout { color: var(--vellum-2); font-family: var(--display); font-size: 38px; }
.age .readout small { color: var(--mist-dim); }
.age input[type="range"] { height: 2px; background: rgba(216,181,106,0.34); }
.age input[type="range"]::-webkit-slider-thumb {
  background: var(--bluefire);
  border-color: var(--night);
  box-shadow: 0 0 0 1px var(--bluefire), 0 0 24px rgba(140,199,216,0.52);
}
.age input[type="range"]::-moz-range-thumb {
  background: var(--bluefire);
  border-color: var(--night);
  box-shadow: 0 0 0 1px var(--bluefire), 0 0 24px rgba(140,199,216,0.52);
}
.tone-row {
  border-color: rgba(216,181,106,0.28);
  border-radius: 999px;
  padding: 4px;
  gap: 4px;
  background: rgba(5,10,16,0.46);
}
.tone-row label {
  border: 0;
  border-radius: 999px;
  color: var(--mist-dim);
  padding: 10px 8px;
}
.tone-row label.is-on {
  background: var(--vellum);
  color: var(--night);
  box-shadow: 0 7px 24px -14px rgba(251,245,232,0.85);
}
.examples { gap: 8px; }
.ex-chip {
  position: relative;
  overflow: hidden;
  color: rgba(241,234,215,0.86);
  background: linear-gradient(90deg, rgba(19,35,51,0.7), rgba(8,16,25,0.34));
  border-color: rgba(216,181,106,0.16);
  border-radius: 16px 16px 16px 4px;
  padding: 10px 13px 10px 16px;
}
.ex-chip::before {
  content: "";
  position: absolute;
  left: 0;
  top: 12px;
  bottom: 12px;
  width: 2px;
  background: var(--gold);
  opacity: 0.6;
}
.ex-chip:hover {
  color: var(--vellum-2);
  background: rgba(19,35,51,0.92);
  border-color: rgba(140,199,216,0.42);
  transform: translateX(2px);
}
.ex-chip small { color: var(--bluefire); }

.btn-primary {
  color: #10161d;
  background: linear-gradient(135deg, var(--gold), #f0d99c 48%, #d37a52);
  border: 0;
  border-radius: 999px;
  padding: 14px 22px;
  box-shadow: 0 15px 36px -22px rgba(216,181,106,0.98), inset 0 1px 0 rgba(255,255,255,0.52);
  font-family: var(--display);
  font-weight: 760;
}
.btn-primary:hover { background: linear-gradient(135deg, #f1cf77, #fff0bd 48%, #e07b51); }
.btn-primary:disabled { background: rgba(174,187,183,0.42); color: rgba(8,16,25,0.72); }
.btn-ghost, .seed-pill { color: var(--mist-dim); }
.btn-ghost:hover { color: var(--bluefire); border-color: var(--bluefire); }

.col-story { min-height: 66vh; }
.book {
  color: var(--ink);
  background:
    linear-gradient(115deg, rgba(255,255,255,0.5), transparent 34%),
    radial-gradient(circle at 92% 8%, rgba(216,181,106,0.2), transparent 26%),
    linear-gradient(180deg, var(--vellum-2), var(--vellum));
  border: 1px solid rgba(255,255,255,0.58);
  border-radius: 34px 34px 34px 8px;
  box-shadow: var(--shadow-leaf);
  padding: clamp(34px, 5vw, 68px) clamp(26px, 4.8vw, 64px);
  transform: rotate(0.6deg);
}
.book::before {
  left: 20px;
  top: 20px;
  right: 20px;
  bottom: 20px;
  border: 1px solid rgba(143,56,31,0.13);
  border-radius: 24px 24px 24px 5px;
}
.book::after {
  content: "";
  position: absolute;
  inset: 0;
  pointer-events: none;
  border-radius: inherit;
  opacity: 0.28;
  background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='130' height='130'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' seed='8'/><feColorMatrix values='0 0 0 0 0.44 0 0 0 0 0.32 0 0 0 0 0.18 0 0 0 0.12 0'/></filter><rect width='100%25' height='100%25' filter='url(%23n)'/></svg>");
}
.book .corner { color: rgba(143,56,31,0.46); width: 26px; height: 26px; }
.book .corner.tl { top: 14px; left: 14px; }
.book .corner.tr { top: 14px; right: 14px; }
.book .corner.bl { bottom: 14px; left: 14px; }
.book .corner.br { bottom: 14px; right: 14px; }
.cover { min-height: 430px; }
.cover .folio, .expl .byline, .section .tag, .expl .signoff { color: rgba(19,32,42,0.58); }
.cover h2 {
  font-family: var(--display);
  color: var(--ink);
  font-size: clamp(42px, 5.1vw, 72px);
  line-height: 0.9;
  letter-spacing: -0.05em;
  font-style: normal;
  font-weight: 780;
  font-variation-settings: "SOFT" 76, "WONK" 1;
}
.cover h2 b { color: var(--wax-ink); }
.cover p { color: var(--ink-soft); font-size: 17px; max-width: 47ch; }
.writing { min-height: 430px; }
.writing .penline { color: var(--wax-ink); }
.writing .quill { background: var(--wax-ink); }
.writing .progress {
  height: 5px;
  border-radius: 999px;
  background: linear-gradient(90deg, var(--wax) 0%, var(--gold) var(--p,40%), rgba(19,32,42,0.12) var(--p,40%), rgba(19,32,42,0.12) 100%);
}
.writing p { color: var(--ink-soft); }
.expl { animation: arrive 0.7s var(--ease) both; }
@keyframes arrive { from { opacity: 0; transform: translateY(12px) rotate(-0.8deg); filter: blur(6px); } to { opacity: 1; transform: translateY(0) rotate(0); filter: blur(0); } }
.expl h2.title {
  color: var(--ink);
  font-family: var(--display);
  font-size: clamp(36px, 4.4vw, 58px);
  line-height: 0.94;
  letter-spacing: -0.045em;
  font-weight: 780;
  font-variation-settings: "SOFT" 70, "WONK" 1;
}
.expl h2.title small { color: rgba(19,32,42,0.52); }
.section { margin-bottom: 22px; }
.section .tag { color: var(--wax-ink); }
.section .text { color: var(--ink); font-size: 18.5px; }
.section.opener .text {
  color: #243545;
  background: rgba(255,255,255,0.34);
  border-left: 0;
  border-radius: 18px 18px 18px 4px;
  padding: 14px 16px;
  box-shadow: inset 0 0 0 1px rgba(143,56,31,0.09);
}
.section.closer .text { color: var(--wax-ink); }
.section.followup .text { color: rgba(19,32,42,0.62); }
.audio-panel {
  border-top-color: rgba(143,56,31,0.14);
  background: rgba(255,255,255,0.25);
  border-radius: 20px 20px 20px 6px;
  padding: 16px;
}
.btn-read {
  color: var(--vellum-2);
  background: var(--night-2);
  border: 1px solid rgba(19,32,42,0.88);
  border-radius: 999px;
  padding: 10px 14px;
}
.btn-read:hover { background: var(--wax-ink); color: var(--vellum-2); border-color: var(--wax-ink); }
.audio-panel .audio-status { color: rgba(19,32,42,0.62); }
.audio-panel audio { filter: sepia(0.18) saturate(0.8); }
.banner { color: var(--wax-ink); border-left-color: var(--wax); }
.colophon {
  border-top-color: rgba(216,181,106,0.16);
  color: rgba(219,227,221,0.54);
}
.colophon b { color: var(--gold) !important; }
.colophon a { color: var(--bluefire); border-bottom-color: rgba(140,199,216,0.42); }
.colophon a:hover { color: var(--vellum-2); border-color: var(--vellum-2); }

@media (max-width: 980px) {
  .imprint { align-items: flex-start; flex-direction: column; }
  .imprint .meta { flex-wrap: wrap; gap: 10px 18px; }
  .stage { grid-template-columns: 1fr; }
  .col-form { position: static; }
  .book { transform: none; }
}

@media (max-width: 560px) {
  .col-form { padding: 20px; }
  .title-block h1 { max-width: 9ch; }
  .tone-row { grid-template-columns: 1fr; border-radius: 22px; }
  .book { border-radius: 24px 24px 24px 6px; }
}
</style>
</head>
<body>

<header class="imprint" role="banner">
  <div class="wordmark"><b>Fabella</b> &mdash; a quiet instrument for hard questions</div>
  <div class="meta">
    <span><b>Track I</b> &middot; Backyard AI</span>
    <span>Gemma <b>4B</b> &middot; Nemotron <b>4B</b> &middot; VoxCPM2 &middot; Modal</span>
  </div>
</header>

<main class="stage" role="main">

  <section class="col-form" aria-label="The situation">
    <div class="title-block">
      <h1>What's the <em>hard thing</em>?</h1>
      <p class="lede">Tell Fabella the situation in a sentence or two. She drafts a small, careful script, has another model check it, then can read it back in a calm voice.</p>
    </div>

    <form id="explain-form" novalidate>
      <div class="field">
        <label class="label" for="situation">The situation</label>
        <textarea id="situation" name="situation" placeholder="e.g. My 7-year-old's grandma is in the hospital for surgery. She keeps asking why grandma won't come home." maxlength="600" required></textarea>
        <div class="hint">A sentence or two is enough. The more concrete, the better the explanation.</div>
      </div>

      <div class="field">
        <span class="label">Or start from an example</span>
        <div class="examples" id="examples"></div>
      </div>

      <div class="field">
        <label class="label" for="age-range">The child's age</label>
        <div class="age">
          <input id="age-range" name="age" type="range" min="5" max="12" step="1" value="7" />
          <div class="readout"><span id="age-readout">7</span><small>years</small></div>
        </div>
      </div>

      <div class="field">
        <label class="label" for="child-name">Child's name <small>(optional)</small></label>
        <input id="child-name" name="child_name" type="text" placeholder="leave empty to address the parent" maxlength="30" autocomplete="off" />
      </div>

      <div class="field">
        <span class="label">Tone</span>
        <div class="tone-row" id="tone-row" role="radiogroup" aria-label="Tone"></div>
      </div>

      <div class="actions">
        <button type="submit" id="submit-btn" class="btn-primary">
          <span id="submit-label">Draft an explanation</span>
          <span class="arrow" aria-hidden="true">&rarr;</span>
        </button>
        <button type="button" id="regen-btn" class="btn-ghost" disabled>New version</button>
        <span class="seed-pill" id="seed-pill">N&deg;&nbsp;0</span>
      </div>
    </form>
  </section>

  <section class="col-story" aria-label="The explanation">
    <article class="book" id="book">
      <svg class="corner tl" viewBox="0 0 22 22" fill="none" stroke="currentColor" stroke-width="1"><path d="M2 8 L2 2 L8 2 M2 4 Q10 4 10 10"/></svg>
      <svg class="corner tr" viewBox="0 0 22 22" fill="none" stroke="currentColor" stroke-width="1"><path d="M2 8 L2 2 L8 2 M2 4 Q10 4 10 10"/></svg>
      <svg class="corner bl" viewBox="0 0 22 22" fill="none" stroke="currentColor" stroke-width="1"><path d="M2 8 L2 2 L8 2 M2 4 Q10 4 10 10"/></svg>
      <svg class="corner br" viewBox="0 0 22 22" fill="none" stroke="currentColor" stroke-width="1"><path d="M2 8 L2 2 L8 2 M2 4 Q10 4 10 10"/></svg>

      <div class="inner" id="page">
        <div class="cover" id="cover">
          <div class="folio">Night folio &middot; private draft</div>
          <h2>Put the hard thing on the table.<br/><b>Leave with words.</b></h2>
          <p>Fill in the situation on the left. Fabella writes the first careful version, checks it against a child-language rubric, and keeps VoxCPM2 ready if you want to hear it aloud.</p>
        </div>
      </div>
    </article>
  </section>

</main>

<footer class="colophon">
  <span>Set in <b>Fraunces</b> and <b>Literata</b> &middot; Spoken by VoxCPM2 on demand</span>
  <span>Built for the <a href="https://huggingface.co/spaces/build-small-hackathon/README" target="_blank" rel="noopener">Build Small Hackathon</a> &middot; 2026</span>
</footer>

<script>
(function () {
  "use strict";

  const TONE_CHOICES = __TONE_CHOICES__;
  const EXAMPLES = __EXAMPLES__;
  const SECTION_SEP = "\x1f";

  // example chips
  const exEl = document.getElementById("examples");
  EXAMPLES.forEach((s, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "ex-chip";
    b.innerHTML = "<small>Example " + (i + 1) + "</small>" + escapeHTML(s);
    b.addEventListener("click", () => {
      document.getElementById("situation").value = s;
      document.getElementById("situation").focus();
    });
    exEl.appendChild(b);
  });

  // tone
  const toneRow = document.getElementById("tone-row");
  TONE_CHOICES.forEach(([val, label], i) => {
    const lbl = document.createElement("label");
    lbl.textContent = label;
    const input = document.createElement("input");
    input.type = "radio"; input.name = "tone"; input.value = val; input.checked = (i === 0);
    lbl.appendChild(input);
    lbl.className = input.checked ? "is-on" : "";
    lbl.addEventListener("click", () => {
      toneRow.querySelectorAll("label").forEach(x => x.classList.remove("is-on"));
      lbl.classList.add("is-on");
      input.checked = true;
    });
    toneRow.appendChild(lbl);
  });

  // age
  const ageRange = document.getElementById("age-range");
  const ageReadout = document.getElementById("age-readout");
  ageRange.addEventListener("input", () => { ageReadout.textContent = ageRange.value; });

  // form
  const form = document.getElementById("explain-form");
  const submitBtn = document.getElementById("submit-btn");
  const submitLabel = document.getElementById("submit-label");
  const regenBtn = document.getElementById("regen-btn");
  const page = document.getElementById("page");
  const seedPill = document.getElementById("seed-pill");
  let seed = 0;
  let lastResult = null;
  let lastAudioText = "";

  function setBusy(busy) {
    submitBtn.disabled = busy;
    regenBtn.disabled = busy || !lastResult;
    submitLabel.textContent = busy
      ? "Composing…"
      : (lastResult ? "Draft another" : "Draft an explanation");
  }

  function renderWriting() {
    page.innerHTML =
      '<div class="writing">' +
        '<div class="penline"><span class="quill" aria-hidden="true"></span><span>Drafting, with care</span></div>' +
        '<div class="progress" id="progress" style="--p:8%"></div>' +
        '<p>Fabella is drafting, then a second small model is checking the explanation against a six-criterion rubric (clarity, age, warmth, no moralizing, no scary content, concrete).</p>' +
      '</div>';
    let p = 8;
    const tick = setInterval(() => {
      p = Math.min(p + 4 + Math.random() * 6, 88);
      const el = document.getElementById("progress");
      if (el) el.style.setProperty("--p", p + "%");
      else clearInterval(tick);
    }, 320);
    return () => clearInterval(tick);
  }

  function escapeHTML(s) {
    return String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  }

  function renderExplanation(sections) {
    const opener = sections[0] || "";
    const body = sections[1] || "";
    const closer = sections[2] || "";
    const followup = sections[3] || "";
    lastAudioText = [opener, body, closer, followup].filter(Boolean).join("\n\n");
    const childName = (document.getElementById("child-name").value || "").trim();
    const toneLabel = (document.querySelector('#tone-row input:checked') || {}).value || "gentle";
    const ageStr = ageRange.value;
    const dateStr = new Date().toLocaleDateString(undefined, {year:"numeric",month:"short",day:"numeric"});

    const openerHTML = opener
      ? '<section class="section opener"><span class="tag">Opener — say this first</span><p class="text">' + escapeHTML(opener) + '</p></section>'
      : "";
    const bodyHTML = body
      ? '<section class="section body"><span class="tag">The explanation — read this aloud</span><div class="text">' +
          body.split(/\n\s*\n/).filter(Boolean).map(p => '<p>' + escapeHTML(p.trim()).replace(/\n/g, "<br/>") + '</p>').join("") +
        '</div></section>'
      : "";
    const closerHTML = closer
      ? '<section class="section closer"><span class="tag">Closer — say this to land it</span><p class="text">' + escapeHTML(closer) + '</p></section>'
      : "";
    const followupHTML = followup
      ? '<section class="section followup"><span class="tag">If they ask another question</span><p class="text">' + escapeHTML(followup) + '</p></section>'
      : "";

    page.innerHTML =
      '<div class="expl">' +
        '<div class="byline">' +
          '<span>Folio I</span><span class="dot" aria-hidden="true"></span>' +
          '<span>For a ' + escapeHTML(ageStr) + '-year-old' + (childName ? ' named ' + escapeHTML(childName) : '') + '</span>' +
          '<span class="dot" aria-hidden="true"></span>' +
          '<span>' + escapeHTML(toneLabel) + '</span>' +
        '</div>' +
        '<h2 class="title">A short explanation<small>Read aloud &middot; revise if you want</small></h2>' +
        openerHTML + bodyHTML + closerHTML + followupHTML +
        '<div class="audio-panel" id="audio-panel">' +
          '<button type="button" class="btn-read" id="read-aloud">Read aloud</button>' +
          '<span class="audio-status" id="audio-status">VoxCPM2 narration runs only when you ask for it.</span>' +
          '<audio id="audio-player" controls hidden></audio>' +
        '</div>' +
        '<div class="signoff">' +
          '<span>End of folio &middot; ' + dateStr + '</span>' +
          '<button type="button" class="regen" id="regen-inline">New version &rarr;</button>' +
        '</div>' +
      '</div>';
    const ri = document.getElementById("regen-inline");
    if (ri) ri.addEventListener("click", () => regenBtn.click());
    const readBtn = document.getElementById("read-aloud");
    if (readBtn) readBtn.addEventListener("click", readAloud);
  }

  function renderError(msg) {
    page.innerHTML =
      '<div class="expl">' +
        '<div class="byline"><span>Folio I</span><span class="dot" aria-hidden="true"></span><span>Hold on a moment</span></div>' +
        '<div class="banner">' + escapeHTML(msg) + '</div>' +
      '</div>';
  }

  async function callMakeExplanation(useSeed) {
    const data = {
      situation: document.getElementById("situation").value,
      age: parseInt(ageRange.value, 10),
      child_name: document.getElementById("child-name").value,
      tone: (document.querySelector('#tone-row input:checked') || {}).value || "gentle",
      seed: useSeed,
    };
    const res = await fetch("/gradio_api/call/make_explanation", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data: [data.situation, data.age, data.child_name, data.tone, data.seed] }),
    });
    if (!res.ok) {
      const t = await res.text();
      throw new Error("HTTP " + res.status + ": " + t.slice(0, 200));
    }
    const evt0 = await res.json();
    const eventId = evt0.event_id;
    const evt = await fetch("/gradio_api/call/make_explanation/" + eventId, { headers: { Accept: "text/event-stream" } });
    if (!evt.ok || !evt.body) throw new Error("SSE open failed");
    const reader = evt.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    let result = null;
    let err = null;
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) !== -1) {
        const frame = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        const line = frame.split("\n").find(l => l.startsWith("data: "));
        if (!line) continue;
        const payload = line.slice(6).trim();
        if (payload === "null" || payload === "") continue;
        try {
          const obj = JSON.parse(payload);
          if (obj.msg === "process_completed") {
            if (obj.success) {
              const out = obj.output && obj.output.data;
              let text = null;
              if (Array.isArray(out)) text = out.find(v => typeof v === "string");
              else if (typeof out === "string") text = out;
              if (text) {
                // 4 sections joined by U+001F
                result = text.split(SECTION_SEP);
                if (result.length < 4) {
                  while (result.length < 4) result.push("");
                }
              }
            } else {
              err = (obj.output && obj.output.error) || "Generation failed";
            }
          }
        } catch (_) {}
      }
    }
    if (err) throw new Error(err);
    if (!result) throw new Error("No result");
    return result;
  }

  async function readGradioString(apiName, data) {
    const res = await fetch("/gradio_api/call/" + apiName, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data: data }),
    });
    if (!res.ok) {
      const t = await res.text();
      throw new Error("HTTP " + res.status + ": " + t.slice(0, 200));
    }
    const evt0 = await res.json();
    const eventId = evt0.event_id;
    const evt = await fetch("/gradio_api/call/" + apiName + "/" + eventId, { headers: { Accept: "text/event-stream" } });
    if (!evt.ok || !evt.body) throw new Error("SSE open failed");
    const reader = evt.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    let result = null;
    let err = null;
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) !== -1) {
        const frame = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        const line = frame.split("\n").find(l => l.startsWith("data: "));
        if (!line) continue;
        const payload = line.slice(6).trim();
        if (payload === "null" || payload === "") continue;
        try {
          const obj = JSON.parse(payload);
          if (obj.msg === "process_completed") {
            if (obj.success) {
              const out = obj.output && obj.output.data;
              if (Array.isArray(out)) result = out.find(v => typeof v === "string") || null;
              else if (typeof out === "string") result = out;
            } else {
              err = (obj.output && obj.output.error) || "Request failed";
            }
          }
        } catch (_) {}
      }
    }
    if (err) throw new Error(err);
    if (!result) throw new Error("No result");
    return result;
  }

  async function readAloud() {
    const btn = document.getElementById("read-aloud");
    const status = document.getElementById("audio-status");
    const player = document.getElementById("audio-player");
    if (!btn || !status || !player || !lastAudioText) return;
    btn.disabled = true;
    status.textContent = "Warming VoxCPM2 and preparing narration…";
    player.hidden = true;
    player.removeAttribute("src");
    try {
      const tone = (document.querySelector('#tone-row input:checked') || {}).value || "gentle";
      const audioUrl = await readGradioString("make_audio", [lastAudioText, tone]);
      if (audioUrl.startsWith("ERROR:")) throw new Error(audioUrl.slice(6).trim());
      player.src = audioUrl;
      player.hidden = false;
      status.textContent = "Ready. Press play when you want to listen.";
      await player.play().catch(() => {});
    } catch (err) {
      status.textContent = String(err.message || err);
    } finally {
      btn.disabled = false;
    }
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const sitEl = document.getElementById("situation");
    if (!sitEl.value.trim()) {
      sitEl.focus();
      sitEl.style.borderColor = "var(--wax)";
      setTimeout(() => { sitEl.style.borderColor = ""; }, 1400);
      return;
    }
    setBusy(true);
    const stopProgress = renderWriting();
    try {
      const sections = await callMakeExplanation(seed);
      lastResult = sections;
      stopProgress();
      renderExplanation(sections);
    } catch (err) {
      stopProgress();
      renderError(String(err.message || err));
    } finally {
      setBusy(false);
    }
  });

  regenBtn.addEventListener("click", async () => {
    if (!document.getElementById("situation").value.trim()) return;
    seed += 1;
    seedPill.innerHTML = "N&deg;&nbsp;" + seed;
    setBusy(true);
    const stopProgress = renderWriting();
    try {
      const sections = await callMakeExplanation(seed);
      lastResult = sections;
      stopProgress();
      renderExplanation(sections);
    } catch (err) {
      stopProgress();
      renderError(String(err.message || err));
    } finally {
      setBusy(false);
    }
  });
})();
</script>
</body>
</html>
"""

INDEX_HTML = (
    INDEX_HTML
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
