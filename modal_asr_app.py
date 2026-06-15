"""Isolated Nemotron 3.5 ASR experiment on Modal T4.

This is deliberately separate from ``modal_app.py`` so ASR can be tested
without adding a fourth endpoint to the production Fabella app. Deploy/run this
first, call ``/transcribe`` manually, then wire the HF Space only if it works.
"""

import base64
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import modal


app = modal.App("fabella-asr-experiment")

MODEL_PATH = "/models"
ASR_REPO = "nvidia/nemotron-3.5-asr-streaming-0.6b"
ASR_DIR = "nemotron-3.5-asr-streaming-0.6b"
ASR_NEMO = "nemotron-3.5-asr-streaming-0.6b.nemo"
ASR_PORT = 8003
MINUTES = 60

model_volume = modal.Volume.from_name("fabella-asr-models", create_if_missing=True)
cache_volume = modal.Volume.from_name("fabella-asr-cache", create_if_missing=True)


download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub[hf_xet]>=0.24")
    .env({"HF_HUB_CACHE": MODEL_PATH})
)


def _download_asr(force: bool = False):
    from huggingface_hub import snapshot_download

    target = Path(MODEL_PATH) / ASR_DIR
    if target.exists() and (target / ASR_NEMO).exists() and not force:
        print(f"ASR checkpoint already at {target}; skipping", flush=True)
        return
    print(f"Downloading {ASR_REPO} to {target}...", flush=True)
    snapshot_download(
        repo_id=ASR_REPO,
        local_dir=str(target),
        allow_patterns=[ASR_NEMO],
    )
    model_volume.commit()
    print("ASR download complete", flush=True)


@app.function(image=download_image, volumes={MODEL_PATH: model_volume}, timeout=60 * MINUTES)
def download_asr(force: bool = False):
    """Pull the .nemo checkpoint once before deploying the web endpoint."""
    return _download_asr(force=force)


asr_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("ffmpeg", "git", "libsndfile1")
    .pip_install(
        "Cython",
        "packaging",
        "torch>=2.5.0",
        "torchaudio>=2.5.0",
        "soundfile",
        "fastapi>=0.110",
        "uvicorn[standard]>=0.27",
        "huggingface_hub[hf_xet]>=0.24",
        "git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]",
    )
    .env(
        {
            "HF_HUB_CACHE": MODEL_PATH,
            "TORCH_HOME": "/cache/torch",
            "NUMBA_CACHE_DIR": "/cache/numba",
        }
    )
    .run_function(_download_asr, volumes={MODEL_PATH: model_volume}, force_build=False)
)


ASR_SERVER_PY = r'''
import base64
import json
import os
import re
import subprocess
import tempfile
import traceback
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException

os.environ.setdefault("HF_HUB_CACHE", "/models")
os.environ.setdefault("TORCH_HOME", "/cache/torch")
os.environ.setdefault("NUMBA_CACHE_DIR", "/cache/numba")

MODEL_PATH = "/models/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo"

print("[asr] importing torch/nemo", flush=True)
import torch
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

print(f"[asr] restoring {MODEL_PATH}", flush=True)
_model = nemo_asr.models.ASRModel.restore_from(MODEL_PATH, map_location="cuda" if torch.cuda.is_available() else "cpu")
_model.eval()
if torch.cuda.is_available():
    _model = _model.cuda()
print(f"[asr] loaded on {'cuda' if torch.cuda.is_available() else 'cpu'}", flush=True)
_model_lock = Lock()

app = FastAPI()


def _suffix(header: str) -> str:
    header = header.lower()
    if "wav" in header:
        return ".wav"
    if "mp4" in header or "m4a" in header:
        return ".m4a"
    if "ogg" in header:
        return ".ogg"
    return ".webm"


def _decode_data_url(data_url: str) -> tuple[bytes, str]:
    if not data_url.startswith("data:audio/") or "," not in data_url:
        raise HTTPException(status_code=400, detail="expected audio data URL")
    header, payload = data_url.split(",", 1)
    if ";base64" not in header:
        raise HTTPException(status_code=400, detail="expected base64 audio data URL")
    audio = base64.b64decode(payload, validate=True)
    if len(audio) > 8 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="audio too large")
    if len(audio) < 256:
        raise HTTPException(status_code=400, detail="audio was empty")
    return audio, _suffix(header)


def _normalize_result(result) -> str:
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, list):
        result = result[0] if result else ""
    text = getattr(result, "text", result)
    if isinstance(text, list):
        text = text[0] if text else ""
    return re.sub(r"\s+<[^>]+>\s*$", "", str(text or "")).strip()


def _extract_transcriptions(hyps):
    if not hyps:
        return []
    if isinstance(hyps[0], Hypothesis):
        return [hyp.text for hyp in hyps]
    return hyps


def _transcribe_wav(wav_path: str, target_lang: str) -> str:
    # Follow NeMo's documented cache-aware streaming path for
    # prompt-conditioned models. Plain ASRModel.transcribe() does not populate
    # the per-stream prompt correctly for this checkpoint.
    with _model_lock:
        if hasattr(_model, "set_inference_prompt"):
            _model.set_inference_prompt(target_lang)
            if hasattr(_model, "decoding") and hasattr(_model.decoding, "set_strip_lang_tags"):
                _model.decoding.set_strip_lang_tags(True)
        if hasattr(_model.encoder, "set_default_att_context_size"):
            _model.encoder.set_default_att_context_size(att_context_size=[56, 13])

        device = next(_model.parameters()).device
        compute_dtype = torch.float32
        streaming_buffer = CacheAwareStreamingAudioBuffer(
            model=_model,
            online_normalization=False,
            pad_and_drop_preencoded=False,
        )
        streaming_buffer.append_audio_file(wav_path, stream_id=-1)
        batch_size = len(streaming_buffer.streams_length)
        cache_last_channel, cache_last_time, cache_last_channel_len = _model.encoder.get_initial_cache_state(
            batch_size=batch_size
        )
        previous_hypotheses = None
        pred_out_stream = None
        transcribed_texts = []
        for step_num, (chunk_audio, chunk_lengths) in enumerate(iter(streaming_buffer)):
            chunk_audio = chunk_audio.to(device=device, dtype=compute_dtype)
            chunk_lengths = chunk_lengths.to(device=device)
            (
                pred_out_stream,
                transcribed_texts,
                cache_last_channel,
                cache_last_time,
                cache_last_channel_len,
                previous_hypotheses,
            ) = _model.conformer_stream_step(
                processed_signal=chunk_audio,
                processed_signal_length=chunk_lengths,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
                keep_all_outputs=streaming_buffer.is_buffer_empty(),
                previous_hypotheses=previous_hypotheses,
                previous_pred_out=pred_out_stream,
                drop_extra_pre_encoded=0 if step_num == 0 else _model.encoder.streaming_cfg.drop_extra_pre_encoded,
                return_transcription=True,
            )
        texts = _extract_transcriptions(transcribed_texts)
    return _normalize_result(texts[0] if texts else "")


@app.get("/health")
async def health():
    return {"ok": True, "model": "nvidia/nemotron-3.5-asr-streaming-0.6b", "cuda": bool(torch.cuda.is_available())}


@app.post("/transcribe")
async def transcribe(payload: dict):
    data_url = str(payload.get("audio_data_url") or "")
    target_lang = str(payload.get("target_lang") or "auto").strip()
    if target_lang != "auto" and not re.match(r"^[a-z]{2}-[A-Z]{2}$", target_lang):
        target_lang = "auto"
    audio, suffix = _decode_data_url(data_url)
    try:
        with tempfile.TemporaryDirectory(prefix="fabella-asr-") as tmp:
            src = Path(tmp) / f"input{suffix}"
            wav = Path(tmp) / "converted.wav"
            src.write_bytes(audio)
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-ac", "1", "-ar", "16000", str(wav)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            text = _transcribe_wav(str(wav), target_lang)
    except subprocess.CalledProcessError as e:
        detail = (e.stderr or b"").decode("utf-8", errors="replace")[:300]
        raise HTTPException(status_code=400, detail=f"ffmpeg failed: {detail}")
    except Exception as e:
        print(f"[asr] transcribe failed: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"transcribe failed: {e}")
    return {"ok": True, "text": text, "target_lang": target_lang}


@app.post("/warmup")
async def warmup():
    return {"ok": True, "message": "model loaded"}
'''


@app.function(
    image=asr_image,
    gpu="T4",
    min_containers=0,
    scaledown_window=2 * MINUTES,
    timeout=10 * MINUTES,
    volumes={MODEL_PATH: model_volume, "/cache": cache_volume},
)
@modal.concurrent(max_inputs=2)
@modal.web_server(port=ASR_PORT, startup_timeout=15 * MINUTES)
def serve_asr():
    server_path = "/root/asr_server.py"
    with open(server_path, "w") as f:
        f.write(ASR_SERVER_PY)
    cmd = [
        "uvicorn",
        "asr_server:app",
        "--app-dir",
        "/root",
        "--host",
        "0.0.0.0",
        "--port",
        str(ASR_PORT),
        "--log-level",
        "info",
    ]
    print(f"Starting ASR experiment: {' '.join(cmd)}", flush=True)
    subprocess.Popen(cmd)


@app.local_entrypoint()
def main():
    print("Deploy with: .venv/bin/modal deploy modal_asr_app.py")
    print("Download only: .venv/bin/modal run modal_asr_app.py::download_asr")
