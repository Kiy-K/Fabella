---
title: Fabella
emoji: 📖
colorFrom: pink
colorTo: yellow
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Fabella

A personalized storytelling companion for children. Tell Fabella your child's name, age, favorite themes, and a moral — Fabella writes a warm, age-appropriate story just for them.

## What it does

- Generates a short, personalized children's story
- Matches reading level to the child's age (6–10)
- Weaves in their favorite themes
- Resolves a clear moral at the end
- Works on first run with **no API key** (mock mode)

## Run locally

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python app.py
```

Open the URL printed in the terminal (default `http://localhost:7860`).

## Run on Hugging Face Spaces

This repo is structured to push directly to a HF Space:

1. Create a new Space → **Gradio** SDK
2. Push these files
3. The Space builds and serves the app

No secrets are required for the first run. The mock generator is fully active by default.

## Modes

- **Mock mode** (default): template-based, deterministic, zero GPU. Perfect for demos and offline use.
- **Real model mode**: set `USE_MOCK=false` in the Space's environment variables to route to [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it) on ZeroGPU. Open (Apache 2.0), no API key required. Override the model with `FABELLA_MODEL_ID`.

## Phase 1 scope

This is the first vertical slice. It includes:

- Gradio Blocks UI with form, regenerate button, mode badge
- Template-based mock story generator
- Input sanitization and profanity guard
- Real-model path wired (`@spaces.GPU`) but deferred for polish

Out of scope for Phase 1: PDF export, user accounts, story history, audio narration, image generation.

## Build Small

Fabella is intentionally tiny: a single Gradio app, one mock generator, one optional real-model call, no database, no auth, no orchestration. The goal is a child who finishes a story and asks for another.
