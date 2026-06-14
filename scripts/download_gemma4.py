"""Download Gemma 4 E4B-IT and sync to the Fabella-models bucket.

Runs as a HF Job — all downloading and uploading happens on HF infrastructure.
"""

# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub"]
# ///

import os
import subprocess
import sys

from huggingface_hub import snapshot_download

BUCKET = "hf://buckets/build-small-hackathon/Fabella-models/gemma-4-E4B-it"
SRC = "google/gemma-4-E4B-it"

# Files we need for text-only inference. We deliberately skip the multimodal
# Gemma4Processor — its dependencies pull in torchvision, which we don't need
# for chat-templated text generation.
ALLOW = [
    "config.json",
    "generation_config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors",
]

print(f"Downloading {SRC} (text-only files)...", flush=True)
local_path = snapshot_download(
    repo_id=SRC,
    allow_patterns=ALLOW,
    cache_dir="/tmp/hf-cache",
)
print(f"Downloaded to {local_path}", flush=True)

print(f"Syncing to {BUCKET}...", flush=True)
result = subprocess.run(
    ["hf", "buckets", "sync", local_path, BUCKET],
    check=True,
)
print(f"Done. Exit: {result.returncode}", flush=True)
