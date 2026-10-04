"""Pin and cache only the evaluated pretrained checkpoint for the A100 runs."""

import os
from pathlib import Path

root = Path('/workspace/voxelflight_a100_20260928')
os.environ['HF_HOME'] = str(root / 'cache/huggingface')
from huggingface_hub import snapshot_download
location = snapshot_download(
    'facebook/map-anything-apache',
    revision='00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a',
    allow_patterns=['config.json', 'model.safetensors'],
)
print(location, flush=True)
