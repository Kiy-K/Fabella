"""Real model path. Wired with @spaces.GPU; polish deferred to Phase 2."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompts import system_prompt, user_prompt
from schema import StoryRequest

MODEL_ID = os.environ.get("FABELLA_MODEL_ID", "google/gemma-4-E4B-it")

_run_gpu = None  # lazily wrapped by spaces.GPU on first real call


def _parse_title_and_body(text: str) -> tuple[str, str]:
    text = (text or "").strip()
    if not text:
        return "A Small Story", ""
    lines = text.splitlines()
    title = "A Small Story"
    body_lines = list(lines)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.lower().startswith("title:"):
            title = stripped.split(":", 1)[1].strip() or title
            body_lines = lines[i + 1:]
            break
    body = "\n".join(body_lines).strip()
    return (title, body) if body else (title, text)


def _run_with_spaces_gpu(req: StoryRequest) -> tuple[str, str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    messages = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": user_prompt(req.name, req.age, req.themes, req.moral, req.length)},
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = processor(text=text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=900,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            do_sample=True,
        )
    raw = processor.decode(output[0][input_len:], skip_special_tokens=True)
    return _parse_title_and_body(raw)


def _get_run_gpu():
    """Wrap _run_with_spaces_gpu with @spaces.GPU(duration=60) on first call."""
    global _run_gpu
    if _run_gpu is None:
        import spaces

        _run_gpu = spaces.GPU(duration=60)(_run_with_spaces_gpu)
    return _run_gpu


def generate_real(req: StoryRequest) -> tuple[str, str]:
    """Real-model call. Best-effort; surfaces a user-visible message on any failure."""
    try:
        return _get_run_gpu()(req)
    except Exception as e:
        return "Fabella (real model error)", f"_Real model call failed: {e}._"
