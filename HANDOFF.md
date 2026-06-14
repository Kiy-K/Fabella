# Fabella Handoff — Explanation + Read-Aloud Pipeline on Modal

## Context

Fabella is a Gradio children's-storytelling app in `/home/khoi/fabella`,
now **pivoted to Track I · Backyard AI** ("useful for someone the maker
actually knows"). It solves a specific real problem parents face: **how
do I explain a hard thing to my kid in their own language?**

The parent describes a situation in a sentence or two. Fabella drafts a
short, kind, age-appropriate explanation in an Opener / Body / Closer /
follow-up shape. A second small model checks the draft against a
6-criterion rubric before the parent sees it. After generation, the
parent can optionally click **Read aloud** to synthesize the explanation
with VoxCPM2.

**Live URLs:**

- HF Space: https://build-small-hackathon-fabella.hf.space
- Modal drafter (Gemma 4 E4B-IT): https://khoitruong071510--fabella-serve-drafter.modal.run
- Modal judge (Nemotron-3 Nano 4B): https://khoitruong071510--fabella-serve-judge.modal.run
- Modal TTS (VoxCPM2): https://khoitruong071510--fabella-serve-tts.modal.run
- Modal app: https://modal.com/apps/khoitruong071510/main/deployed/fabella
- HF Space repo: https://huggingface.co/spaces/build-small-hackathon/Fabella

## Current Architecture

```
                    HF Space (CPU, custom HTML+CSS+JS)
                                   |
                                   |  POST /gradio_api/call/make_explanation
                                   |  -> SSE stream with 4-section string
                                   v
                         app.py (gradio.Server / FastAPI)
                                   |
              +--------------------+--------------------+--------------------+
              |                                         |                    |
              v                                         v                    v
   Modal drafter (A10G, Gemma 4 E4B-IT)     Modal judge (A10G, Nemotron-3)  Modal TTS (A10G, VoxCPM2)
   --tool-call-parser gemma4                (no tool-calling flags)         FastAPI /synthesize
   ReAct via LangChain                      Pydantic JSON verdict          audio/wav when requested
   validate_explanation tool                one invoke + one repair retry
   middleware jumps to "end" on OK
```

**Key design decisions:**

- **Drafter on LangGraph.** The drafter is a `create_agent` ReAct loop
  with one tool (`validate_explanation`) and a custom middleware that
  jumps to `end` after a successful validation or after a hard cap of
  two tool calls. State machine, conditional edges, tool-call plumbing
  — that's LangGraph's job, and it works.
- **Judge on Pydantic.** The judge task is bounded — one rubric, one
  draft, one structured verdict — so it doesn't need an agent loop.
  Single `llm.invoke()` + `JudgeVerdict.model_validate_json()` + one
  repair retry. Cross-field consistency (`ok` ⇔ `verdict`) is enforced
  in code, not in the prompt.
- **Three separate Modal web_servers, three A10Gs.** Drafter, judge,
  and TTS don't share a GPU. Independent scaling, independent
  `scaledown_window`.
- **TTS only runs on demand.** VoxCPM2 is a separate FastAPI wrapper,
  not vLLM. The HF Space `make_audio` API posts explanation text to
  `/synthesize`, receives `audio/wav`, and returns a base64 data URL to
  the browser.
- **The judge has NO tool-calling flags on the server side.** Its
  prompt asks for raw JSON in `content`; the Pydantic parser does the
  rest. This dodges Nemotron-3-Nano's chat-template tool-dialect
  (`<tool_call>...</tool_call>` markers that vLLM's `hermes` parser
  doesn't recognize) entirely.
- **The drafter DOES use tool calling.** vLLM is launched with
  `--enable-auto-tool-choice --tool-call-parser gemma4`; the server
  parses Gemma 4's `<|tool_call|>...<tool_call|>` markers into
  OpenAI-spec `tool_calls` JSON, which the client reads off
  `response.choices[0].message.tool_calls` directly.
- **HF Space runs a custom `gradio.Server` (FastAPI subclass).** No
  default Gradio chrome. Storybook design, single hand-coded page.
- **API contract for the frontend:** the @app.api endpoint returns one
  string with sections joined by U+001F (Unit Separator) — Opener,
  Body, Closer, Follow-up. The frontend splits on that. (Gradio Server
  `@app.api` has no output components, so tuples get dropped — single
  string is the simplest workaround.)

## File Map

| File | Purpose |
|------|---------|
| `app.py` | `gradio.Server` (FastAPI subclass) app, custom HTML+CSS+JS, `make_explanation` API, `make_audio` TTS proxy, no-op `@spaces.GPU` placeholder for HF runtime |
| `agent.py` | LangChain ReAct agent. `build_agent(llm, req, judge_llm=None)` returns `(agent, user_prompt)`. `make_validate_tool` builds a closure that calls `judge_explanation()` if a judge is given, else falls back to a rule check. `FabellaAgentMiddleware.before_model` jumps to `end` once validation passes or after `max_tool_calls=2`. `extract_explanation(messages)` parses Opener/Body/Closer/follow-up sections from the validated tool-call draft. |
| `judge.py` | Pydantic-validated judge. `judge_explanation(llm, draft, req_age, req_tone, child_name, situation) -> JudgeVerdict`. Two attempts (original + repair prompt) before raising `JudgeFailed` for fallback. Tolerant of markdown fences and pretty-printed JSON. |
| `schema.py` | `ExplainRequest` dataclass + `JudgeVerdict` Pydantic model + `JudgeFailed` exception. |
| `safety.py` | Input sanitization, profanity block, `sanitize_situation`, `explain_to_words(tone)`, `age_bucket(age)`. |
| `llm.py` | `FabellaVLLM` BaseChatModel wrapping vLLM's OpenAI-compatible API. `bind_tools` builds OpenAI-spec `tools=[...]`, `_generate` passes it and reads `message.tool_calls` from the response. Replay of prior `AIMessage.tool_calls` and `ToolMessage` results into next-turn messages uses the OpenAI chat-completions shape. |
| `modal_app.py` | Modal deployment: `download_drafter` + `download_judge` + `download_tts`; `serve_drafter` (port 8000), `serve_judge` (port 8001), `serve_tts` (port 8002). One A10G each. |
| `modal_app_gemma.py` | Legacy: a previous-session single-model Modal deploy, kept for reference. Not the live deploy. |

## What Changed This Session

The most recent session (pivot to Backyard AI) changed:

### New files
- `judge.py` — Pydantic-validated judge with repair retry
- `modal_app_gemma.py` — kept as reference for the prior single-model deploy

### Substantially rewritten
- `agent.py` — story-generation agent replaced with explanation-generation agent. `make_validate_tool` now optionally takes `judge_llm` and routes through `judge_explanation()`. The drafter's output format changed from "Title: / body" to "Opener: / Body: / Closer: / (optional) If they ask more:". `extract_explanation` parses these four sections.
- `schema.py` — `StoryRequest` replaced with `ExplainRequest` (situation, age, child_name, tone, seed). Added `JudgeVerdict` (Pydantic) and `JudgeFailed`.
- `safety.py` — `sanitize_situation`, `explain_to_words(tone)`. Legacy theme/moral/length functions kept for compat.
- `app.py` — frontend redesigned for the "explain a hard thing" use case. New form fields: situation textarea, age slider (5-12), child_name (optional), tone segmented control (gentle / matter-of-fact / playful), example chips. Output is the four sections in a book-page layout with the new "Opener" / "The explanation" / "Closer" / "If they ask another question" tags. Added **Read aloud** with `make_audio` proxy to VoxCPM2.
- `modal_app.py` — three web_servers in one Modal app. Drafter uses `--tool-call-parser gemma4`; judge uses no tool flags; TTS runs VoxCPM2 behind FastAPI `/synthesize`.
- `llm.py` — defaults updated to point at the drafter endpoint (`gemma-4` model name).

### Removed earlier
- `multi_agent.py` — earlier multi-agent design (3 parallel drafters + judge) was reverted
- `nemotron3_tool_parser.py` — custom XML tool parser for the (also removed) 30B Nemotron path
- `prompts.py`, `generator.py`, `mock.py`, `real.py` — legacy files

## Non-obvious gotchas

- **No-op `@spaces.GPU` in `app.py`.** HF Spaces runtime scans for at
  least one `@spaces.GPU` function at import and raises
  `RUNTIME_ERROR: No @spaces.GPU function detected` if none exists.
  The placeholder is a 1-second no-op. Do not delete it.
- **`sys.path` hack in every module.** Each file does
  `sys.path.insert(0, os.path.dirname(...))` so imports work when run
  as `python app.py` from the package root. Don't refactor to relative
  imports.
- **Pydantic disallows `_`-prefixed fields.** In `llm.py`, the
  runtime-mutable state (OpenAI client, tools, call counter) is
  declared with `PrivateAttr`, not `Field`.
- **Three Modal endpoints, three env vars.** HF Space reads
  `MODAL_DRAFTER_URL`, `MODAL_JUDGE_URL`, and `MODAL_TTS_URL`. The old
  `MODAL_VLLM_URL` is dead — delete it if it's still there.
- **The drafter uses native tool calling via vLLM.** vLLM is started
  with `--enable-auto-tool-choice --tool-call-parser gemma4`; the
  server parses Gemma 4's native `<|tool_call|>call:name{args}<tool_call|>`
  markers into OpenAI-spec `tool_calls` JSON. The client passes real
  `tools=[{type:"function", function:{name, description,
  parameters:JSON-schema}}]` on each request and reads
  `response.choices[0].message.tool_calls` directly. If the model
  emits no tool call, `content` is returned as the final answer.
- **The judge does NOT use tool calling.** The chat template for
  Nemotron-3-Nano-4B emits tool calls in a custom XML dialect inside
  `<tool_call>...</tool_call>` markers that vLLM's built-in parsers
  don't recognize. The judge server runs with no tool-calling flags;
  the judge prompt asks for raw JSON in `content`, and `judge.py`
  parses that with Pydantic.
- **Pydantic judge schema in `schema.py`.** `JudgeVerdict` has five
  fields: `ok` (bool), `issues` (list[str], each capped at 200 chars),
  `score` (float in [0, 1]), `verdict` (Literal["approve", "revise"]),
  `reasoning` (str, capped at 300 chars). Cross-field consistency
  (`ok` ⇔ `verdict`) is enforced in `judge_explanation()` — the model
  is asked to agree, and the code normalizes if it doesn't.
- **Judge retry-on-failure.** If the first response isn't parseable
  JSON, `judge_explanation()` retries once with a `REPAIR_PROMPT` that
  shows the previous bad response. If both fail, `JudgeFailed` is
  raised and the validate tool falls back to the deterministic rule
  check.
- **Middleware `@hook_config(can_jump_to=["end"])` is required.**
  Without it, LangGraph never creates the conditional edge and the
  early-exit silently does nothing.
- **Modal uses CUDA devel image.** The `nvidia/cuda:12.9.0-devel-ubuntu22.04`
  base provides nvcc, which vLLM/FlashInfer need. `debian_slim` crashes
  during vLLM startup.
- **Drafter flag `--language-model-only` is required.** Gemma 4's
  multimodal processor pulls heavy deps and crashes the vLLM server on
  text-only requests. This flag tells vLLM to skip processor init.
  The judge (Nemotron-Nano-4B) is text-only and does NOT need this
  flag.
- **Gemma 4 E4B is multimodal — it can take audio input.** This
  matters in two ways:
  1. `--language-model-only` is correct **today** because Fabella's
     drafter only ever receives text. If you later add a feature
     where the parent records a 30s voice memo and the drafter
     transcribes it (Whisper-style), the vLLM flag will need to
     change to support audio inputs. The model supports it natively.
  2. The audio side of Gemma 4 is a *separate* path from the
     VoxCPM2 TTS endpoint. They are independent: VoxCPM2 reads
     text and produces 48 kHz audio; Gemma 4 could (if enabled)
     read audio and produce text. Don't conflate them when
     debugging.
- **All A10Gs are independent.** Modal scaledown_window is 10 minutes
  on each. The first request after idle triggers a ~2 min cold start per
  LLM container. TTS cold-starts only after **Read aloud** is clicked.
- **VoxCPM2 TTS is not vLLM.** `serve_tts` writes a generated FastAPI
  server into the container and runs `uvicorn --app-dir /root`. It
  returns `audio/wav` from `/synthesize`; `app.py::make_audio` converts
  that to a base64 data URL for the browser.
- **`section_sep` is U+001F (Unit Separator).** The `@app.api`
  endpoint returns Opener, Body, Closer, Follow-up joined by `\x1f`.
  The frontend splits on it. Don't use `\n` — body text can contain
  newlines legitimately.
- **`FABELLA_MODEL_PATH` env var** is no longer consulted on the
  deployed path. Modal's `download_drafter` hardcodes
  `google/gemma-4-E4B-it` (Apache 2.0, not gated). Do not swap to
  `gemma-3-4b-it` (gated — would break the no-API-key rule).

## Deployment Commands

```bash
# Modal: download weights (run once per model)
.venv/bin/modal run modal_app.py::download_drafter
.venv/bin/modal run modal_app.py::download_judge
.venv/bin/modal run modal_app.py::download_tts

# Modal: deploy (rebuilds the image, rolls out all web_servers)
.venv/bin/modal deploy modal_app.py

# HF Space: env vars
hf spaces variables add build-small-hackathon/Fabella \
    --env MODAL_DRAFTER_URL=https://khoitruong071510--fabella-serve-drafter.modal.run
hf spaces variables add build-small-hackathon/Fabella \
    --env MODAL_JUDGE_URL=https://khoitruong071510--fabella-serve-judge.modal.run
hf spaces variables add build-small-hackathon/Fabella \
    --env MODAL_TTS_URL=https://khoitruong071510--fabella-serve-tts.modal.run

# HF Space: upload code
hf upload build-small-hackathon/Fabella app.py    --type space
hf upload build-small-hackathon/Fabella agent.py  --type space
hf upload build-small-hackathon/Fabella judge.py  --type space
hf upload build-small-hackathon/Fabella llm.py    --type space
hf upload build-small-hackathon/Fabella schema.py --type space
hf upload build-small-hackathon/Fabella safety.py --type space
hf upload build-small-hackathon/Fabella requirements.txt --type space

# HF Space: restart to pick up new code
hf spaces restart build-small-hackathon/Fabella
```

## Cost

- **Drafter**: 1× A10G, $0.80/hr, 10-min scaledown
- **Judge**: 1× A10G, $0.80/hr, 10-min scaledown
- **TTS**: 1× A10G, $0.80/hr, 10-min scaledown, only after **Read aloud**
- **At idle**: $0/hr (scaledown)
- **Typical demo session**: a few minutes warm = ~$0.03-0.05

## Known Issues / Open Questions

1. **Cold start latency** — First request after 10 min idle triggers
   vLLM cold start (~2 min per container for model load + torch.compile
   + CUDA graph capture). Both containers cold-start in sequence on
   the first request of a new session. Could add `min_containers=1` to
   each Modal serve() to keep warm (costs ~$1.60/hr idle).
2. **No test suite** — No automated tests exist. Manual smoke-tests
   are in this handoff (search "Live test" or "Smoke-test").
3. **Judge occasionally emits unparseable thinking-traces.** The
   `judge_explanation()` repair prompt fixes this most of the time.
   When both attempts fail, the validate tool falls back to the
   rule check, so the system never hard-errors. The model is a
   reasoning model; a `--default-chat-template-kwargs '{"enable_thinking":
   false}'` flag could be added to the judge server to make outputs
   shorter, but the retry handles it well enough.
4. **Drafter at temperature 0.9** — produces creative variety but the
   judge sometimes rejects a perfectly good draft on style grounds. The
   `seed` UI control lets parents re-roll for variety.

## Suggested Skills

- `hf-cli` — Manage HF Space: variables, logs, uploads, restarts
- `find-docs` — For Modal, vLLM, Gradio, LangChain, Pydantic API
  questions (use ctx7 CLI)
- `diagnose` — If runtime errors occur (vLLM startup, agent failures,
  judge parsing)
- `agent-browser` — For end-to-end testing of the live HF Space
- `handoff` — If handing off again after further work

## Next Steps (if continuing)

1. Add a `min_containers=1` warmup to both Modal serves for zero
   cold-start latency
2. Add basic test suite: judge parsing (valid / repair / fallback),
   validate tool, explanation extraction, end-to-end agent with stubs
3. Stream the explanation token-by-token as the drafter writes it
   (the API contract would change from one-shot to SSE)
4. Cache common patterns (the same situation often comes up — "moving",
   "new baby", "death of grandparent") so warm requests skip the LLM
5. Polish the HF Space card README to match the new Backyard AI
   framing before the hackathon submission deadline
