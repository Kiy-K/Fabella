# AGENTS.md

Quick orientation for future OpenCode sessions working on Fabella.

## What this is

A Gradio app for **parents who need to explain hard things to their child
in kid-appropriate language**. The parent types a sentence or two about
the situation; Fabella drafts a short explanation (Opener / Body /
Closer / optional follow-up) that's reviewed by a second small model
before the parent sees it. After generation, the parent can optionally
click **Read aloud** to synthesize the explanation with VoxCPM2.

Built for the [Build Small Hackathon](https://huggingface.co/spaces/build-small-hackathon/README) · **Track I · Backyard AI** ("useful for someone the maker actually knows").

The product design has two distinct execution layers, each tuned to its job:

- **Drafter on LangGraph.** The drafter is a `create_agent` ReAct loop
  with one tool (`validate_explanation`) and a custom middleware. State
  machine, conditional edges, tool-call plumbing — that's LangGraph's
  job, and it works.
- **Judge on direct OpenAI + Pydantic.** The judge task is bounded — one
  rubric, one draft, one structured verdict. No LangGraph loop, no
  LangChain agent, no state machine. `judge.py` calls the judge endpoint
  through the OpenAI-compatible client, validates with
  `JudgeVerdict.model_validate_json()`, and has one direct JSON repair
  retry. Cross-field consistency is enforced in code.

## Architecture

```
                    HF Space (CPU, custom HTML+CSS+JS)
                                   |
                                   |  POST /gradio_api/call/make_explanation
                                   |  <-  SSE stream with 4-section string
                                   v
                         app.py (gradio.Server / FastAPI)
                                   |
              +--------------------+--------------------+--------------------+
              |                                         |                    |
              v                                         v                    v
   Modal drafter (A10G, Gemma 4 E4B-IT)     Modal judge (A10G, Nemotron-3)  Modal TTS (A10G, VoxCPM2)
   --tool-call-parser gemma4                (no tool-calling flags)         FastAPI /synthesize
   ReAct via LangChain                      Direct Pydantic verdict       audio/wav when requested
   validate_explanation tool                one invoke + one repair retry
   middleware jumps to "end" on OK
```

- HF Space env vars: `MODAL_DRAFTER_URL`, `MODAL_JUDGE_URL`, and `MODAL_TTS_URL`.
- HF OAuth is enabled (`hf_oauth: true`) for personalization; unsigned users fall back to browser-local anonymous sessions.
- The mounted HF bucket stores minimal SQLite history at `/models/fabella-data/history.sqlite3`. This is not PostgreSQL; use external Postgres later if multi-replica writes are needed.
- Modal uses the `nvidia/cuda:12.9.0-devel-ubuntu22.04` base image (provides nvcc for FlashInfer).
- Drafter vLLM flags: `--language-model-only --enable-auto-tool-choice --tool-call-parser gemma4`.
- Judge vLLM flags: none (the judge emits raw JSON in `content`; Pydantic parses).
- TTS is separate from drafter/judge and only called from `make_audio` when the user clicks **Read aloud**.
- Drafter and judge use `min_containers=1` while budget allows, so the critical-path LLMs stay warm for demos.
- TTS also uses `min_containers=1` and runs on `L4` instead of A10G because VoxCPM2 is small enough for a cheaper/newer GPU class.

## Live URLs

- HF Space: https://build-small-hackathon-fabella.hf.space
- Modal drafter: https://khoitruong071510--fabella-serve-drafter.modal.run
- Modal judge: https://khoitruong071510--fabella-serve-judge.modal.run
- Modal TTS: https://khoitruong071510--fabella-serve-tts.modal.run
- Modal app: https://modal.com/apps/khoitruong071510/main/deployed/fabella
- HF Space repo: https://huggingface.co/spaces/build-small-hackathon/Fabella

## Run / verify

Local dev uses `uv`, not `pip`:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python app.py   # http://localhost:7860
```

The custom frontend runs on CPU locally. For story generation to work,
the `MODAL_DRAFTER_URL`, `MODAL_JUDGE_URL`, and `MODAL_TTS_URL` env vars
must point to live Modal deploys. Use the deployed HF Space for
end-to-end testing.

`app.py` is the only entrypoint. No test suite exists yet.

## File map

- `app.py` — `gradio.Server` (FastAPI subclass) app. Imports `FabellaVLLM` from `llm.py` and calls `run_agent` from `agent.py`. Serves a hand-coded HTML+CSS+JS page. `@app.api()` endpoint `make_explanation` returns Opener/Body/Closer/Follow-up joined by U+001F. `make_audio` proxies VoxCPM2 and returns a base64 WAV data URL. `/api/me`, `/api/history`, `/api/history/append`, and `/api/history/clear` store minimal chat history/preferences in bucket-backed SQLite with HF OAuth identity when available. Contains a no-op `@spaces.GPU` placeholder function (HF Spaces runtime scans for at least one during import; all inference runs on Modal).
- `agent.py` — LangChain ReAct agent. `build_agent(llm, req, judge_llm=None)` returns `(agent, user_prompt)`. One tool: `validate_explanation` (calls the Pydantic judge if `judge_llm` is given, else falls back to a rule check). `FabellaAgentMiddleware.before_model` jumps to `end` once validation passes or after `max_tool_calls=2`. `extract_explanation(messages)` parses the four sections from the validated tool-call draft.
- `judge.py` — Direct OpenAI-compatible + Pydantic-validated judge. `judge_explanation(llm, draft, req_age, req_tone, child_name, situation) -> JudgeVerdict`. First tries vLLM/OpenAI `response_format` with `JudgeVerdict.model_json_schema()`, then falls back to prompt-only JSON plus one repair retry before raising `JudgeFailed`. Tolerant of markdown fences and pretty-printed JSON.
- `schema.py` — `ExplainRequest` dataclass (situation, age, child_name, tone, seed), `JudgeVerdict` Pydantic model (ok, issues, score, verdict, reasoning), `JudgeFailed` exception.
- `safety.py` — input sanitization (`sanitize_situation`, `sanitize_name`, `has_profanity`), `explain_to_words(tone)`, `age_bucket(age)`.
- `llm.py` — `FabellaVLLM`, a `BaseChatModel` subclass wrapping vLLM's OpenAI-compatible API. `bind_tools` builds an OpenAI-spec `tools=[...]` payload, `_generate` passes it on the request and reads `response.choices[0].message.tool_calls` from the response. Replay of prior `AIMessage.tool_calls` and `ToolMessage` results into next-turn messages uses the OpenAI chat-completions shape.
- `modal_app.py` — Modal deployment. `download_drafter`, `download_judge`, and `download_tts` write weights to the `fabella-models` Volume. `serve_drafter` runs vLLM with `--language-model-only --enable-auto-tool-choice --tool-call-parser gemma4` on port 8000 (A10G). `serve_judge` runs vLLM with no tool-calling flags on port 8001 (A10G). `serve_tts` runs a tiny VoxCPM2 FastAPI app on port 8002 (L4). One Modal app, three web_server functions.
- `modal_app_gemma.py` — Legacy single-model Modal deploy from the previous session. Not the live deploy. Reference only.

## Non-obvious gotchas

- **No-op `@spaces.GPU` in `app.py`.** The HF Spaces runtime scans for
  at least one `@spaces.GPU` function at module import and raises
  `RUNTIME_ERROR: No @spaces.GPU function detected` if none exists.
  The placeholder is a 1-second no-op. Do not delete it.
- **`sys.path` hack in every module.** Each file does
  `sys.path.insert(0, os.path.dirname(...))` so imports work when run
  as `python app.py` from the package root. Don't refactor to relative
  imports.
- **Pydantic disallows `_`-prefixed fields.** In `llm.py`, the
  runtime-mutable state (OpenAI client, tools, call counter) is
  declared with `PrivateAttr`, not `Field`.

- **Bucket per-user JSON files, not PostgreSQL or a separate database cloud API.** The mounted HF bucket is file/object storage, so Fabella stores one minimal JSON file per parent at `/data/fabella-data/user-<owner_key>.json` (signed-in users keyed by HF username, anonymous users keyed by a `localStorage` session id). The file holds the last ~80 messages plus a small profile (child name/age, preferred tone). No external database cloud API is used. If the Space is scaled to multiple replicas or needs analytics, move this to external Postgres via `DATABASE_URL`.
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
- **The judge does NOT use tool calling.** Nemotron-3-Nano-4B's chat
  template emits tool calls in a custom XML dialect inside
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
  raised and the validate tool falls back to the rule check. The judge
  path intentionally bypasses LangChain message invocation on the
  deployed path; LangGraph stays only in the drafter loop.
- **`@app.api` returns a single string.** Gradio Server's `@app.api`
  has no output components, so tuples get dropped silently. The
  handler concatenates the four sections with `\x1f` (Unit Separator)
  and the frontend splits. Don't use `\n` as a separator — body text
  can contain newlines legitimately.
- **Middleware `@hook_config(can_jump_to=["end"])` is required.**
  Without it, LangGraph never creates the conditional edge and the
  early-exit silently does nothing.
- **Modal uses CUDA devel image.** The `nvidia/cuda:12.9.0-devel-ubuntu22.04`
  base provides nvcc, which vLLM/FlashInfer need. `debian_slim` crashes
  during vLLM startup.
- **Drafter flag `--language-model-only` is required.** Gemma 4's
  multimodal processor pulls heavy deps and crashes the vLLM server
  on text-only requests. This flag tells vLLM to skip processor init.
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
- **Critical-path LLMs are kept warm.** Drafter and judge use
  `min_containers=1`, which removes most demo cold-start latency but
  bills continuously while deployed. TTS also uses `min_containers=1` on L4
  so **Read aloud** is responsive during demos.
- **TTS runs on L4.** VoxCPM2 is ~2B and fits smaller GPUs, so
  `serve_tts` uses `gpu="L4"` plus `min_containers=1` instead of A10G.
  If L4 availability or latency is bad, switch back to A10G or try Modal
  GPU fallbacks.
- **VoxCPM2 TTS is not vLLM.** `serve_tts` writes a generated FastAPI
  server into the container and runs `uvicorn`. It returns `audio/wav`
  from `/synthesize`; `app.py::make_audio` converts that to a base64
  data URL for the browser.
- **Do not switch to `nanovllm-voxcpm` for this demo.** It is faster,
  but it needs `flash-attn`, changes the API (`target_text`, streamed
  MP3), and is not worth the integration risk with one day left and a
  tight GPU budget. Keep the stable official VoxCPM2 server.
- **`FABELLA_MODEL_PATH` env var** is no longer consulted on the
  deployed path. Modal's `download_drafter` hardcodes
  `google/gemma-4-E4B-it` (Apache 2.0, not gated). Do not swap to
  `gemma-3-4b-it` (gated — would break the no-API-key rule).

## When editing

- Adding a new tone preset → add to `TONE_CHOICES` in `app.py`
  (gentle / matter-of-fact / playful is the current set).
- Adding an example situation → add to `EXAMPLE_SITUATIONS` in
  `app.py`. They appear as one-click chips on the left column.
- Changing the drafter's tool set → edit `make_validate_tool` in
  `agent.py` (it builds the tool closure per request).
- Changing the judge's rubric → edit `judge.py::_build_rubric` and the
  `SYSTEM_PROMPT` in the same file. The output schema is in
  `schema.py::JudgeVerdict` — change both.
- Changing the agent's max tool calls → pass
  `FabellaAgentMiddleware(max_tool_calls=N)` to `create_agent` in
  `agent.py::build_agent`.
- Adding a new example chip → add to `EXAMPLE_SITUATIONS` in
  `app.py`.
- History, accounts, image upload are **out of scope** unless reopened.

## Deployment

```bash
# Modal: download weights (run once per model)
.venv/bin/modal run modal_app.py::download_drafter
.venv/bin/modal run modal_app.py::download_judge
.venv/bin/modal run modal_app.py::download_tts

# Modal: deploy (rebuilds image, rolls out both web_servers)
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
