---
title: Fabella
emoji: 📖
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.18.0
app_file: app.py
pinned: true
license: apache-2.0
short_description: Small words for big questions.
---

# Fabella

**Small words for big questions.** Tell Fabella what's going on in a sentence or two. She drafts a short, kind, age-appropriate explanation you can read aloud &mdash; a second small model checks it against a six-criterion rubric before you see it.

Built for the [Build Small Hackathon](https://huggingface.co/spaces/build-small-hackathon/README) &middot; **Track I &middot; Backyard AI.**

[Live demo](https://build-small-hackathon-fabella.hf.space) &middot; [Modal app](https://modal.com/apps/khoitruong071510/main/deployed/fabella) &middot; [HF Space repo](https://huggingface.co/spaces/build-small-hackathon/Fabella)

## What it solves

Parents have to explain hard things &mdash; a parent's hospitalization, a house move, a pet dying, a refusal to buy a phone &mdash; to kids who don't have the vocabulary for what's happening. Most of the time, we end up improvising at 9pm and getting it half-right.

Fabella is a second pair of eyes. You describe the situation in a sentence or two; the app drafts an explanation in the **Opener &rarr; Body &rarr; Closer &rarr; optional "if they ask more"** shape, then a second small model checks it for: opener/body/closer present, body length, age-appropriate vocabulary, no moralizing, no scary content, no invented facts.

You read it. If it's good, you can read it yourself or click **Read aloud** for VoxCPM2 narration. If it's not, you click **New version** for a fresh draft.

## The two-model pipeline

| Layer | Model | Why this model | Why this execution |
|---|---|---|---|
| **Drafter** | `google/gemma-4-E4B-it` (4B) on Modal A10G | Fast, smart enough for short empathetic text, Apache 2.0 | **LangGraph ReAct** &mdash; needs the state machine (draft &rarr; validate &rarr; revise &rarr; end) with tool calling and middleware-driven early exit |
| **Judge** | `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` (4B) on Modal A10G | Fast, follows structured-output instructions reliably | **Pydantic** + single LLM call + one repair retry &mdash; the task is bounded, no agent loop needed |
| **Read-aloud** | `openbmb/VoxCPM2` (~2B) on Modal A10G | Apache 2.0, 48 kHz speech, good voice-description control | Separate FastAPI server, only called after the user clicks **Read aloud** |

The split is deliberate. The drafter needs agentic machinery (tool calls, conditional edges, jump-to-end). The judge doesn't &mdash; it's "receive a rubric + a draft, return a structured verdict." Pydantic gives us disciplined output, type safety, and a one-shot repair retry. The two layers stay in their own files: `agent.py` for the LangGraph loop, `judge.py` for the Pydantic validator.

## Output shape

Every result is a four-section explanation:

- **Opener** &mdash; one sentence the parent can say to start the conversation
- **Body** &mdash; 1-3 short paragraphs, written in the second person, age-appropriate, no moralizing
- **Closer** &mdash; one sentence to land the conversation
- **If they ask another question** &mdash; an optional follow-up the parent can use

For example, given the situation *"My 7-year-old's grandma is in the hospital for surgery. She keeps asking when grandma is coming home."* and tone *gentle*, the app returns something like:

> **Opener:** I want to talk to you about Grandma.
> **Body:** Grandma is in the hospital right now. She is having a little surgery. The doctors are taking good care of her. It is a part of getting her better. The doctors are very kind and they know just what to do.
> **Closer:** We are all hoping she comes home very soon.
> **If they ask more:** We can wait together and find out what the doctors say.

## The stack

- **HF Space** &mdash; custom HTML+CSS+JS frontend served by `gradio.Server` (FastAPI subclass). Storybook-modernist design, no default Gradio chrome.
- **Modal** &mdash; three containers in one app. Drafter (Gemma 4 E4B-IT) runs with `--enable-auto-tool-choice --tool-call-parser gemma4 --language-model-only` (text-only today; the model itself is multimodal and could take audio input if we add a voice-memo feature later). Judge runs with no tool-calling flags. TTS (VoxCPM2) is a small FastAPI wrapper around the official `voxcpm` library. All three A10G, 10-min scaledown.
- **LangChain 1.x** ReAct loop with a custom middleware that jumps to `end` after a successful validation or after a hard cap of two tool calls.
- **Pydantic v2** for the judge's structured output.

## Files

- `app.py` &mdash; `gradio.Server` app, custom HTML+CSS+JS, `@app.api()` endpoint, no-op `@spaces.GPU` placeholder for HF runtime
- `agent.py` &mdash; LangChain ReAct drafter, `validate_explanation` tool, middleware
- `judge.py` &mdash; Pydantic-validated judge with one repair retry
- `schema.py` &mdash; `ExplainRequest` dataclass + `JudgeVerdict` Pydantic model + `JudgeFailed` exception
- `llm.py` &mdash; `FabellaVLLM` BaseChatModel wrapping vLLM's OpenAI-compatible API
- `modal_app.py` &mdash; Modal deployment (drafter + judge + VoxCPM2 TTS on separate A10Gs)
- `safety.py` &mdash; input sanitization, profanity block, `explain_to_words(tone)`

## Run locally

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
export MODAL_DRAFTER_URL=https://khoitruong071510--fabella-serve-drafter.modal.run
export MODAL_JUDGE_URL=https://khoitruong071510--fabella-serve-judge.modal.run
export MODAL_TTS_URL=https://khoitruong071510--fabella-serve-tts.modal.run
python app.py
```

The frontend runs on CPU locally. The two LLM Modal containers cold-start in ~2 min each on the first request of a new session; the TTS container cold-starts separately only when **Read aloud** is clicked.

## Constraints honored

- ≤ 32B params &middot; both models are 4B
- Gradio app &middot; hosted as a HF Space
- No API key needed for the models &middot; Modal credits only
- No database, no auth, no orchestration

## Why "small"

Two small models, one tool, one page, one rubric. The whole product is a form on the left, a book page on the right, and a few seconds of waiting. The point of Fabella isn't to replace the parent's voice &mdash; it's to give the parent a second opinion at the moment they need one, in language the kid can hear.
