"""Fabella inference servers on Modal.

Three independent web_servers in one app, each on its own A10G:

  serve_drafter (port 8000) — Gemma 4 E4B-IT (4B). Generates explanations.
  serve_judge   (port 8001) — Nemotron-3 Nano 4B. Scores the draft against
                                the request and returns a structured verdict.
  serve_tts     (port 8002) — VoxCPM2. Synthesizes read-aloud WAV audio.

The judge runs after the drafter; if the verdict is "revise", the
drafter is re-invoked. This is the cheapest way to get model-driven
quality control without a parallel-multi-agent setup.

All models live on the same Modal Volume (fabella-models) with distinct
sub-directories so we only pay for one download per model.
"""

import os
import subprocess
from pathlib import Path

import modal

app = modal.App("fabella")

model_volume = modal.Volume.from_name("fabella-models", create_if_missing=True)
vllm_cache_volume = modal.Volume.from_name("fabella-vllm-cache", create_if_missing=True)

MODEL_PATH = "/models"

DRAFTER_REPO = "google/gemma-4-E4B-it"
DRAFTER_DIR = "gemma-4-E4B-it"
DRAFTER_SERVED_NAME = "gemma-4"
DRAFTER_PORT = 8000

JUDGE_REPO = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
JUDGE_DIR = "NVIDIA-Nemotron-3-Nano-4B-BF16"
JUDGE_SERVED_NAME = "nemotron-3-4b"
JUDGE_PORT = 8001

TTS_REPO = "openbmb/VoxCPM2"
TTS_DIR = "VoxCPM2"
TTS_PORT = 8002

# --- Images ---------------------------------------------------------------

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub[hf_xet]>=0.24")
    .env({"HF_HUB_CACHE": MODEL_PATH})
)

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .pip_install("vllm>=0.22", "huggingface_hub[hf_xet]>=0.24")
    .env({"HF_HUB_CACHE": MODEL_PATH})
)

# VoxCPM2 is a tokenizer-free diffusion-autoregressive TTS model (MiniCPM-4
# backbone + AudioVAE V2). It's served by the official `voxcpm` Python
# library, NOT vLLM. The image is therefore a separate CUDA image with
# torch + voxcpm installed and a tiny FastAPI wrapper that calls
# VoxCPM.from_pretrained(...).generate(...) and returns audio/wav bytes.
tts_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .pip_install(
        # VoxCPM2 + its torch/torchaudio deps. We pin a major range
        # compatible with the README's "torch>=2.5.0, CUDA>=12.0" claim.
        "voxcpm>=1.0",
        "torch>=2.5.0",
        "torchaudio>=2.5.0",
        "soundfile",
        "fastapi>=0.110",
        "uvicorn[standard]>=0.27",
    )
    .env({"HF_HUB_CACHE": MODEL_PATH})
)


# --- Model download (one entry per model) --------------------------------


@app.function(image=download_image, volumes={MODEL_PATH: model_volume}, timeout=60 * 60)
def download_drafter(force: bool = False):
    """Pull Gemma 4 E4B-IT weights to the Volume (run once)."""
    from huggingface_hub import snapshot_download
    target = Path(MODEL_PATH) / DRAFTER_DIR
    if target.exists() and any(target.iterdir()) and not force:
        print(f"Drafter model already at {target}; skipping")
        return
    print(f"Downloading {DRAFTER_REPO} to {target}...")
    snapshot_download(
        repo_id=DRAFTER_REPO,
        local_dir=str(target),
        allow_patterns=[
            "config.json", "generation_config.json", "chat_template.jinja",
            "tokenizer.json", "tokenizer_config.json",
            "preprocessor_config.json", "processor_config.json",
            "model*.safetensors", "*.py",
        ],
    )
    model_volume.commit()
    print("Drafter download complete")


@app.function(image=download_image, volumes={MODEL_PATH: model_volume}, timeout=60 * 60)
def download_judge(force: bool = False):
    """Pull Nemotron-Nano-9B-v2 weights to the Volume (run once)."""
    from huggingface_hub import snapshot_download
    target = Path(MODEL_PATH) / JUDGE_DIR
    if target.exists() and any(target.iterdir()) and not force:
        print(f"Judge model already at {target}; skipping")
        return
    print(f"Downloading {JUDGE_REPO} to {target}...")
    snapshot_download(
        repo_id=JUDGE_REPO,
        local_dir=str(target),
        allow_patterns=[
            "config.json", "generation_config.json", "chat_template.jinja",
            "tokenizer.json", "tokenizer_config.json",
            "preprocessor_config.json", "processor_config.json",
            "model*.safetensors", "*.py",
        ],
    )
    model_volume.commit()
    print("Judge download complete")


# --- vLLM servers --------------------------------------------------------

MINUTES = 60

# Demo latency policy:
# - The two 4B LLM endpoints sit on the critical path for every explanation, so
#   keep one warm replica while budget allows.
# - TTS is also kept warm while budget allows; VoxCPM2 is small enough
#   for a cheaper/newer L4 instead of A10G.
LLM_MIN_CONTAINERS = 1
TTS_MIN_CONTAINERS = 1
TTS_GPU = "L4"


def _vllm_cmd(model_dir: Path, served_name: str, port: int, extra: list[str]) -> list[str]:
    return [
        "vllm", "serve",
        str(model_dir),
        "--host", "0.0.0.0",
        "--port", str(port),
        "--served-model-name", served_name,
        "--uvicorn-log-level", "info",
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.90",
        *extra,
    ]


@app.function(
    image=vllm_image,
    gpu="A10G",
    min_containers=LLM_MIN_CONTAINERS,
    scaledown_window=10 * MINUTES,
    timeout=10 * MINUTES,
    volumes={MODEL_PATH: model_volume, "/root/.cache/vllm": vllm_cache_volume},
)
@modal.concurrent(max_inputs=10)
@modal.web_server(port=DRAFTER_PORT, startup_timeout=10 * MINUTES)
def serve_drafter():
    """Gemma 4 E4B-IT — the story drafter.

    Tool-calling is native via vLLM's gemma4 parser (the model's chat
    template uses <|tool_call|>...<tool_call|> markers).
    """
    model_dir = Path(MODEL_PATH) / DRAFTER_DIR
    cmd = _vllm_cmd(model_dir, DRAFTER_SERVED_NAME, DRAFTER_PORT, extra=[
        "--language-model-only",          # skip multimodal processor
        "--enable-auto-tool-choice",
        "--tool-call-parser", "gemma4",
    ])
    print(f"Starting drafter vLLM: {' '.join(cmd)}", flush=True)
    subprocess.Popen(cmd)


@app.function(
    image=vllm_image,
    gpu="A10G",
    min_containers=LLM_MIN_CONTAINERS,
    scaledown_window=10 * MINUTES,
    timeout=10 * MINUTES,
    volumes={MODEL_PATH: model_volume, "/root/.cache/vllm": vllm_cache_volume},
)
@modal.concurrent(max_inputs=10)
@modal.web_server(port=JUDGE_PORT, startup_timeout=10 * MINUTES)
def serve_judge():
    """Nemotron-3-Nano-4B-BF16 — the multi-criteria story judge.

    No tool-calling flags on the server side: the judge prompt in
    llm.py asks for plain JSON in `content` and the client parses it.
    This dodges the chat-template tool-dialect dance entirely.
    """
    model_dir = Path(MODEL_PATH) / JUDGE_DIR
    cmd = _vllm_cmd(model_dir, JUDGE_SERVED_NAME, JUDGE_PORT, extra=[])
    print(f"Starting judge vLLM: {' '.join(cmd)}", flush=True)
    subprocess.Popen(cmd)


# --- VoxCPM2 TTS ----------------------------------------------------------

TTS_SERVER_PY = '''
"""VoxCPM2 TTS server for Fabella.

Wraps the official `voxcpm` library in a tiny FastAPI app that exposes
POST /synthesize. Accepts JSON {text, voice_description, cfg_value,
inference_timesteps} and returns audio/wav bytes. The model is loaded
once on import (Modal keeps the container warm while traffic is hot).
"""

import io
import os
import sys
import traceback

# Pin HF cache before voxcpm / torch import so model weights land in
# the shared Modal Volume, not the container overlay.
os.environ.setdefault("HF_HUB_CACHE", "/models")
MODEL_DIR = "/models/VoxCPM2"

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Response


print("[tts] importing voxcpm", flush=True)
try:
    from voxcpm import VoxCPM
except Exception as e:
    print(f"[tts] voxcpm import failed: {type(e).__name__}: {e}", flush=True)
    raise

print(f"[tts] loading VoxCPM2 from {MODEL_DIR}", flush=True)
_model = VoxCPM.from_pretrained(MODEL_DIR, load_denoiser=False)
print(f"[tts] loaded; sample_rate = {_model.tts_model.sample_rate}", flush=True)

app = FastAPI()


@app.get("/health")
async def health():
    return {"status": "ok", "sample_rate": int(_model.tts_model.sample_rate)}


@app.post("/synthesize")
async def synthesize(payload: dict):
    text = (payload.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    voice_description = (payload.get("voice_description") or "").strip() or None
    cfg_value = float(payload.get("cfg_value") or 2.0)
    inference_timesteps = int(payload.get("inference_timesteps") or 10)
    normalize = bool(payload.get("normalize", True))
    denoise = bool(payload.get("denoise", True))

    # VoxCPM2 voice-design convention: put the description in parens at
    # the start of `text` when no reference audio is provided.
    if voice_description and not payload.get("reference_wav_path"):
        text = f"({voice_description}){text}"

    try:
        wav = _model.generate(
            text=text,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            normalize=normalize,
            denoise=denoise,
            prompt_wav_path=payload.get("prompt_wav_path") or None,
            prompt_text=payload.get("prompt_text") or None,
            reference_wav_path=payload.get("reference_wav_path") or None,
        )
    except Exception as e:
        print(f"[tts] generate failed: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"generate failed: {e}")

    # wav is a 1-D numpy array at model.tts_model.sample_rate
    sr = int(_model.tts_model.sample_rate)
    buf = io.BytesIO()
    sf.write(buf, np.asarray(wav, dtype=np.float32), sr, format="WAV", subtype="PCM_16")
    return Response(content=buf.getvalue(), media_type="audio/wav")
'''


@app.function(image=download_image, volumes={MODEL_PATH: model_volume}, timeout=60 * 60)
def download_tts(force: bool = False):
    """Pull VoxCPM2 weights to the Volume (run once)."""
    from huggingface_hub import snapshot_download
    target = Path(MODEL_PATH) / TTS_DIR
    if target.exists() and any(target.iterdir()) and not force:
        print(f"TTS model already at {target}; skipping")
        return
    print(f"Downloading {TTS_REPO} to {target}...")
    snapshot_download(
        repo_id=TTS_REPO,
        local_dir=str(target),
        allow_patterns=[
            "config.json", "configuration_*.py", "modeling_*.py",
            "generation_config.json", "chat_template.jinja",
            "tokenizer.json", "tokenizer_config.json",
            "preprocessor_config.json", "processor_config.json",
            "audio_vae_config.json", "audiovae_*", "audiovae.pth", "audiovae.safetensors",
            "model*.safetensors", "*.py",
            "*.json",
        ],
    )
    model_volume.commit()
    print("TTS download complete")


@app.function(
    image=tts_image,
    gpu=TTS_GPU,
    min_containers=TTS_MIN_CONTAINERS,
    scaledown_window=10 * MINUTES,
    timeout=10 * MINUTES,
    volumes={MODEL_PATH: model_volume},
)
@modal.concurrent(max_inputs=10)
@modal.web_server(port=TTS_PORT, startup_timeout=10 * MINUTES)
def serve_tts():
    """VoxCPM2 — read-aloud narration for Fabella explanations.

    Wrapped in a small FastAPI app. The drafter's text is sent to
    `/synthesize` and the result is a `audio/wav` blob. The HF Space
    frontend renders the audio inline via a standard `<audio>` element.
    """
    # Write the FastAPI server source into the container and run it.
    server_path = "/root/voxcpm_server.py"
    with open(server_path, "w") as f:
        f.write(TTS_SERVER_PY)
    print(f"[tts] wrote server to {server_path}", flush=True)
    cmd = [
        "uvicorn", "voxcpm_server:app",
        "--app-dir", "/root",
        "--host", "0.0.0.0",
        "--port", str(TTS_PORT),
        "--log-level", "info",
    ]
    print(f"Starting TTS: {' '.join(cmd)}", flush=True)
    subprocess.Popen(cmd)
