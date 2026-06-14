---
title: Fabella
emoji: 📖
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.18.0
app_file: app.py
pinned: true
hf_oauth: true
license: apache-2.0
short_description: Small words for big questions.
tags:
  - track:backyard
  - sponsor:openbmb
  - sponsor:openai
  - sponsor:nvidia
  - sponsor:modal
  - achievement:offbrand
  - achievement:sharing
  - achievement:fieldnotes
---

# Fabella

**Small words for big questions.** Tell Fabella what's going on in a sentence or two. She drafts a short, kind, age-appropriate explanation you can read aloud — a second small model checks it against a six-criterion rubric before you see it.

**Submission for the [Build Small Hackathon](https://huggingface.co/spaces/build-small-hackathon/README) · Track I · Backyard AI.**

[Live demo](https://build-small-hackathon-fabella.hf.space) · [Public GitHub repo](https://github.com/Kiy-K/Fabella) · [HF Space repo](https://huggingface.co/spaces/build-small-hackathon/Fabella) · [Modal app](https://modal.com/apps/khoitruong071510/main/deployed/fabella)

## Demo video

[Watch on YouTube](https://youtu.be/dAoy1GRbEV8)

The 90-second walkthrough shows the parent flow (situation → age → tone → validated draft → read aloud), the 3-model pipeline (Gemma 4 E4B drafter · Nemotron 3 Nano judge · VoxCPM2 read-aloud), the HF Bucket memory layer, and the anonymized trace dataset. Narration is ElevenLabs (`eleven_multilingual_v2`, voice `Roger`); caption timings are derived from a Whisper `small.en` pass over the synthesized audio.

Source code: [`Kiy-K/Fabella`](https://github.com/Kiy-K/Fabella)

---

## The neighbor next door

This is the person I built Fabella for: a parent I know, at 9 p.m., trying to explain to a 6-year-old that the family dog was not coming back. She had already had a hard day. She did not have the words she wanted, and she did not have the bandwidth to draft them. She needed a second pair of eyes that could read what she was about to say and tell her whether it would land.

**Backyard AI** is exactly that brief: solve a real problem for someone you actually know. Fabella solves it for a parent in the moment they need help most — translating a hard adult situation into language a small child can hear, then having a second model double-check the draft before a human reads it.

---

## What it does

A parent types one or two sentences about the situation: a parent's hospitalization, a house move, a pet dying, a refusal to buy a phone. They pick the child's age, the child's name, and a tone (gentle, matter-of-fact, playful). The app drafts an explanation in the shape **Opener → Body → Closer → optional "if they ask more"**, then a second small model judges the draft against a rubric. The parent reads it, clicks **New version** if it isn't right, or clicks **Read aloud** for VoxCPM2 narration.

The rubric the judge scores against (six checks, all hard-coded in `judge.py`):

1. All three primary sections (opener, body, closer) are present and non-empty
2. Body length is appropriate (1–3 short paragraphs, not a wall of text)
3. Vocabulary matches the child's age
4. No moralizing, no lecturing, no "you should feel..."
5. No scary or violent content beyond what the situation requires
6. No invented facts — only what the parent actually said

The parent sees the validated draft, not a raw model output. If the judge rejects, the drafter gets one revision pass. If it still fails, the rule-based fallback runs in `agent.py` so the parent always gets *something* usable.

---

## The two-model pipeline

| Layer | Model | Size | Runtime | Why this model | Why this execution |
|---|---|---|---|---|---|
| **Drafter** | `google/gemma-4-E4B-it` | 4B | Modal A10G · vLLM | Apache 2.0, fast on short empathetic text, native tool calling | **LangGraph ReAct** — needs the state machine (draft → validate → revise → end) with tool calls and middleware-driven early exit |
| **Judge** | `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` | 4B | Modal A10G · vLLM | Follows structured-output instructions reliably | **Pydantic v2** + one LLM call + one repair retry — task is bounded, no agent loop needed |
| **Read aloud** | `openbmb/VoxCPM2` | ~2B | Modal L4 · FastAPI | Apache 2.0, 48 kHz, voice-description control | Separate FastAPI server; only called when the user clicks **Read aloud** |

The split is deliberate. The drafter needs agentic machinery (state machine, tool calls, conditional edges, jump-to-end). The judge doesn't — its job is "receive rubric + draft, return a structured verdict." Pydantic gives disciplined output, type safety, and a one-shot repair retry. Two layers, two files, two execution models: `agent.py` for the loop, `judge.py` for the verdict.

All three models sit comfortably under the **32B cap** — Fabella uses **10B of parameters total** for inference, with the largest single model at 4B. That makes Fabella a candidate for the **Tiny Titan** special award (≤4B).

---

## Sponsor prize notes

- **OpenAI / Codex** — Codex was used as a coding assistant for early boilerplate and scaffolding. This sponsor-track note is about development assistance, not runtime inference: Fabella's model pipeline uses Gemma, Nemotron, and VoxCPM2.
- **NVIDIA** — `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` is the second model in the pipeline and acts as the structured-output judge in `judge.py`.
- **Modal** — Modal runs all three inference services: the Gemma drafter, the Nemotron judge, and the VoxCPM2 TTS service.
- **OpenBMB** — `openbmb/VoxCPM2` powers the optional **Read aloud** feature.

---

## Stack

- **HF Space (CPU)** — custom HTML + CSS + JS frontend served by `gradio.Server` (FastAPI subclass). Chat-style, parent-friendly UI: welcome screen with example situations, alternating parent / Fabella turns, per-turn Read-aloud button, no default Gradio chrome.
- **HF OAuth** — enabled for personalization; unsigned users fall back to browser-local anonymous sessions.
- **HF Bucket per-user JSON** — minimal chat history and parent preferences persist at `/data/fabella-data/user-<owner_key>.json` (signed-in users keyed by HF username, anonymous users keyed by a `localStorage` session ID).
- **Modal** — one app, three web servers, all `min_containers=0` with a 2-minute `scaledown_window` so they cold-start on demand (3-day demo budget):
  - **Drafter** (A10G) — vLLM with `--language-model-only --enable-auto-tool-choice --tool-call-parser gemma4 --enforce-eager`
  - **Judge** (A10G) — vLLM with `--enforce-eager` (no tool-calling flags; Nemotron's tool-call dialect isn't a vLLM built-in)
  - **TTS** (L4) — VoxCPM2 wrapped in a tiny FastAPI app on the smallest GPU that fits
  - **`--enforce-eager`** on both vLLM servers skips CUDA-graph capture. Saves 20–40s of cold start at a small per-token throughput cost; the right tradeoff for a demo where first-token latency matters more than tokens/sec.
  - **Cold-start warmup ping** on Space import: `app.py::_warm_modal_endpoints` fires a non-blocking `/health` request to each endpoint from a daemon thread. The cold start happens while the parent is reading the welcome screen; the first real request lands on a warm container.
- **LangChain 1.x** ReAct loop with a custom middleware (`FabellaAgentMiddleware`) that jumps to `end` after a successful validation or after a hard cap of two tool calls. The `@hook_config(can_jump_to=["end"])` is required — without it the early-exit silently does nothing.
- **Pydantic v2** for the judge's structured output. `JudgeVerdict` has five fields (`ok`, `issues`, `score`, `verdict`, `reasoning`); cross-field consistency (`ok` ⇔ `verdict`) is enforced in code, not in the prompt.

---

## Output shape

Every result is a four-section explanation in the parent's chosen tone:

- **Opener** — one sentence the parent can say to start the conversation
- **Body** — 1–3 short paragraphs, second-person, age-appropriate, no moralizing
- **Closer** — one sentence to land the conversation
- **If they ask another question** — an optional follow-up the parent can use

Example (situation: *"My 7-year-old's grandma is in the hospital for surgery. She keeps asking when grandma is coming home."*, tone: *gentle*):

> **Opener:** I want to talk to you about Grandma.
> **Body:** Grandma is in the hospital right now. She is having a little surgery. The doctors are taking good care of her. It is a part of getting her better. The doctors are very kind and they know just what to do.
> **Closer:** We are all hoping she comes home very soon.
> **If they ask more:** We can wait together and find out what the doctors say.

---

## Merit badges this submission is stacking

Three claimed, three skipped. Fabella's honest inventory:

| Badge | Status | Why |
|---|---|---|
| **Off-Brand** 🎨 | Claimed | Custom HTML+CSS+JS frontend served by `gradio.Server` — zero default Gradio chrome. |
| **Sharing is Caring** 📡 | Claimed | Anonymized ReAct traces published to [`build-small-hackathon/fabella-traces`](https://huggingface.co/datasets/build-small-hackathon/fabella-traces). See `trace.py` for the schema and anonymization rules. |
| **Field Notes** 📓 | Claimed | Blog/report on what was built and learned, by the maker. |
| **Off the Grid** 🔌 | Skipped | Drafter, judge, and TTS all run on Modal — a cloud GPU platform, not "in front of you." |
| **Well-Tuned** 🎯 | Skipped | No fine-tuning; Gemma 4 E4B-IT and Nemotron Nano 4B are used stock, no PEFT/LoRA, no published checkpoint on the Hub. |
| **Llama Champion** 🦙 | Skipped | Gemma 4 + Nemotron + VoxCPM2. No llama.cpp in the stack. |

## Off-Brand 🎨 — what to look for

The hackathon's Off-Brand badge points at `gr.Server`. Fabella uses it. Concretely:

- **`app.py:97`** — `app = Server()` from `gradio`, not `gr.Blocks` or `gr.ChatInterface`. The Space has zero default Gradio chrome.
- **`app.py:602`–`1453`** — `INDEX_HTML = r"""<!doctype html>..."""`, ~850 lines of hand-written HTML, CSS, and vanilla JS. Custom welcome screen, example-situation chips, alternating parent/Fabella chat bubbles, per-turn Read-aloud buttons, settings dialog, history pane.
- **`app.py:1454`** — `@app.get("/", response_class=HTMLResponse) def index(): return INDEX_HTML` — the only thing at `/` is the hand-coded page.
- **No `gr.Blocks`, `gr.ChatInterface`, `gr.Tabs`, `gr.Interface`, or `with gr.` anywhere in `app.py`.** All UI state, all event handlers, and all the styling are in `INDEX_HTML` and the JS that lives inside it.
- The demo video shows the running Space: the parent types a situation, the custom chat UI streams the four sections, and the per-turn Read-aloud button speaks the result. None of that ships with default Gradio.

In other words: the canvas is `gradio.Server`'s FastAPI subclass, but the page is a hand-rolled SPA on top of it. Judges can verify by opening the Space, then running `grep -nE "gr\.Blocks|gr\.ChatInterface|gr\.Tabs" app.py` in the Space repo — empty result.

## Agent trace dataset

Every Fabella generation that opts in (the default) appends an anonymized row to [`build-small-hackathon/fabella-traces`](https://huggingface.co/datasets/build-small-hackathon/fabella-traces). One JSONL row per request, capturing the full ReAct loop:

- `agent.system_prompt` — the drafter prompt (static, in-repo)
- `agent.user_prompt` — the built user message, with the raw situation text replaced by `<redacted>`
- `agent.messages` — assistant tool calls + tool responses for `validate_explanation`
- `agent.final_draft` — opener / body / closer / followup
- `judge` — the Nemotron verdict (`ok`, `issues`, `score`, `verdict`, `reasoning`)
- `request` — `age`, `tone`, `situation_hash` (sha256, for dedup), `situation_preview` (60-char truncated topic), `situation_length`, `history_turns`

**Anonymization** (in order, all applied before the row ever leaves the Space): the raw situation is never written — only its hash, a 60-char topic prefix, and its length. The child's name is dropped from the request and redacted in the draft. The drafter's system prompt ships in full because it's a static string from this repo, not user input.

**Capture is on by default.** Set `FABELLA_SHARE_TRACES=0` on the Space to kill the publisher, or pass `share_trace=False` on `make_explanation` to opt out per-request.

---

## Files

- `app.py` — `gradio.Server` app, custom HTML+CSS+JS, `@app.api()` endpoint, HF OAuth-aware history APIs, no-op `@spaces.GPU` placeholder for HF runtime
- `agent.py` — LangChain ReAct drafter, `validate_explanation` tool, `FabellaAgentMiddleware`
- `judge.py` — Pydantic-validated judge with one repair retry, cross-field consistency enforcement
- `schema.py` — `ExplainRequest` dataclass + `JudgeVerdict` Pydantic model + `JudgeFailed` exception
- `llm.py` — `FabellaVLLM` BaseChatModel wrapping vLLM's OpenAI-compatible API; `bind_tools` builds the OpenAI-spec `tools=[...]` payload
- `modal_app.py` — Modal deployment (drafter + judge on A10G, VoxCPM2 TTS on L4)
- `memory.py` — bucket-backed parent memory and preference summaries for follow-up continuity
- `safety.py` — input sanitization, profanity block, `explain_to_words(tone)`
- `trace.py` — anonymized ReAct-trace capture and Hub publishing for the [fabella-traces](https://huggingface.co/datasets/build-small-hackathon/fabella-traces) dataset

---

## Run locally

Local dev uses `uv`, not `pip`. The frontend runs on CPU; the three Modal inference containers must be live.

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
export MODAL_DRAFTER_URL=https://khoitruong071510--fabella-serve-drafter.modal.run
export MODAL_JUDGE_URL=https://khoitruong071510--fabella-serve-judge.modal.run
export MODAL_TTS_URL=https://khoitruong071510--fabella-serve-tts.modal.run
python app.py
```

Runtime notes:

- `app.py` exposes `demo = app` for Gradio hot reload, while still launching the `gradio.Server` instance directly in normal runs.
- The custom frontend calls `/gradio_api/call/make_explanation` with all nine API inputs, including `share_trace`, so Gradio's queue input validation matches the Python handler signature.
- Known Hugging Face OAuth and Gradio/Starlette deprecation warnings are filtered at startup; they do not affect Space behavior.

---

## Constraints honored

- **≤ 32B params** · both LLMs are 4B; total inference is 10B
- **Gradio app** · hosted as an HF Space, custom UI served by `gradio.Server`
- **No API key needed for the models** · all open weights on Modal credits
- **Show, don't tell** · demo video + social post in submission
