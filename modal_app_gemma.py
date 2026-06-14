"""Fabella vLLM server on Modal.

Serves gemma-4-E4B-it via OpenAI-compatible API.
Model weights cached in Modal Volume for fast cold starts.
"""

import os
import subprocess
from pathlib import Path

import modal

# --- App & volumes ---
app = modal.App("fabella")

model_volume = modal.Volume.from_name("fabella-models", create_if_missing=True)
vllm_cache_volume = modal.Volume.from_name("fabella-vllm-cache", create_if_missing=True)

MODEL_PATH = "/models"
MODEL_NAME = "google/gemma-4-E4B-it"
VLLM_PORT = 8000

# --- Images ---
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


# --- Model download ---
@app.function(
    image=download_image,
    volumes={MODEL_PATH: model_volume},
    timeout=60 * 60,
)
def download_model(force: bool = False):
    """Download gemma-4-E4B-it to the Modal Volume. Run once."""
    from huggingface_hub import snapshot_download

    target = Path(MODEL_PATH) / "gemma-4-E4B-it"
    if target.exists() and any(target.iterdir()) and not force:
        print(f"Model already exists at {target}, skipping download")
        print("Run with --force to re-download")
        return

    print(f"Downloading {MODEL_NAME} to {target}...")
    snapshot_download(
        repo_id=MODEL_NAME,
        local_dir=str(target),
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "chat_template.jinja",
            "tokenizer.json",
            "tokenizer_config.json",
            "preprocessor_config.json",
            "processor_config.json",
            "model*.safetensors",
            "*.py",
        ],
    )
    model_volume.commit()
    print("Download complete")


# --- vLLM server ---
MINUTES = 60


@app.function(
    image=vllm_image,
    gpu="A10G",
    scaledown_window=10 * MINUTES,
    timeout=10 * MINUTES,
    volumes={
        MODEL_PATH: model_volume,
        "/root/.cache/vllm": vllm_cache_volume,
    },
)
@modal.concurrent(max_inputs=10)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    """vLLM OpenAI-compatible server for gemma-4-E4B-it."""
    model_dir = Path(MODEL_PATH) / "gemma-4-E4B-it"

    cmd = [
        "vllm", "serve",
        str(model_dir),
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--served-model-name", "gemma-4",
        "--uvicorn-log-level", "info",
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.90",
        "--language-model-only",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "gemma4",
    ]

    print(f"Starting vLLM: {' '.join(cmd)}", flush=True)
    subprocess.Popen(cmd)
