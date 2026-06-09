# AGENTS.md

Quick orientation for future OpenCode sessions working on Fabella.

## What this is

A Gradio app that generates personalized children's stories. Phase 1 ships with a template-based mock generator (default) and a wired-but-unpolished real-model path on ZeroGPU. Single HF Space, no DB, no auth.

## Run / verify

Local dev uses `uv`, not `pip`:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python app.py   # http://localhost:7860
```

`app.py` is the only entrypoint. No test suite exists yet (was deferred from Phase 1 plan).

## File map

- `app.py` — Gradio Blocks UI, form handler, launch
- `generator.py` — dispatcher; reads `USE_MOCK` at import time, routes to `mock.py` or `real.py`
- `mock.py` — template-based generator. One `THEMES: dict[theme][bucket] = (primary, alt)` table (8 themes × 3 buckets × 2 variants = 48 templates). `_select(theme, age, seed)` picks primary on even seed, alt on odd. Adding a new theme = one new key in `THEMES` + an entry in `THEME_CHOICES` in `app.py`.
- `real.py` — real-model call. `@spaces.GPU` is applied **lazily** via `_get_run_gpu()` so the module imports cleanly even when `spaces` is not installed (mock path stays the default).
- `prompts.py` — system + user prompt builders (used only by `real.py`)
- `safety.py` — input sanitization, profanity block, length/age helpers
- `schema.py` — `StoryRequest` dataclass
- `README.md` — has HF Space YAML frontmatter; the Space builder reads `sdk_version: 4.44.0` from there, not from local install

## Non-obvious gotchas

- **`sys.path` hack in every module.** Each file does `sys.path.insert(0, os.path.dirname(...))` so imports work when run as `python app.py` from the package root. Don't refactor to relative imports — they break the entrypoint.
- **`USE_MOCK` is module-level.** Read once at import. Toggling the env var requires a process restart.
- **Gradio 6 moved `theme=` from `Blocks(...)` to `launch(...)`.** If you upgrade Gradio and see a `UserWarning`, check `app.py:139`.
- **`FABELLA_MODEL_ID` env var** overrides the default `google/gemma-4-E4B-it` slug in `real.py`. **Verified open (Apache 2.0, not gated).** Gemma 4 uses `AutoProcessor` (not `AutoTokenizer`) and supports `enable_thinking=False` for direct, non-reasoning output — keep it disabled for children's stories. Recommended sampling: `temperature=1.0, top_p=0.95, top_k=64`. Do NOT swap to `gemma-3-4b-it` (it's gated and would break the no-API-key rule).
- **Real-model path is best-effort.** On any failure (no GPU, missing dep, model 404), it returns a user-visible "real model unavailable" message — it does NOT silently fall back to mock.
- **No test directory.** The Phase 1 plan included `tests/` but it was not built. Add one if/when a verification pass is needed.
- **HF Space metadata in README.** `sdk_version: 4.44.0` is the version the Space runtime builds against, not your local `pip show gradio` (currently 6.17.3). They can diverge safely.

## When editing

- Adding a new theme → add one key to `THEMES` in `mock.py` (with all three buckets) + an entry in `THEME_CHOICES` in `app.py`.
- Adding a moral preset → `MORAL_PRESETS` in `app.py`.
- Changing age rules → `age_bucket()` in `safety.py` and the vocab rules in `prompts.py` (real model path) must stay in sync.
- PDF export, accounts, history, audio, images are **out of scope** unless the user reopens Phase 2.

## Deployment

Push to a HF Space with Gradio SDK; no secrets required for first boot (mock mode). To activate the real model, set `USE_MOCK=false` in the Space's Variables & Secrets tab.
