"""Download Gemma 4 12B-IT and sync to the Fabella-models bucket.

Runs as a HF Job — all downloading and uploading happens on HF infrastructure.

Gemma 4 is multimodal; the AutoProcessor requires both `processor_config.json`
and `preprocessor_config.json` even for text-only usage.
"""

# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub"]
# ///

import os
import subprocess

from huggingface_hub import snapshot_download

BUCKET = "hf://buckets/build-small-hackathon/Fabella-models/gemma-4-12B-it"
SRC = "google/gemma-4-12B-it"

print(f"Downloading {SRC} (text-only files)...", flush=True)
local_path = snapshot_download(
    repo_id=SRC,
    allow_patterns=[
        "config.json",
        "generation_config.json",
        "chat_template.jinja",
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "model.safetensors",
    ],
    cache_dir="/tmp/hf-cache",
    max_workers=4,
)
print(f"Downloaded to {local_path}", flush=True)

print(f"Syncing to {BUCKET}...", flush=True)
result = subprocess.run(
    ["hf", "buckets", "sync", local_path, BUCKET],
    check=True,
)
print(f"Done. Exit: {result.returncode}", flush=True)
