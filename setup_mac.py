"""One-time online model preparation. Inference itself does not download anything."""
import gc
import os
from pathlib import Path
from huggingface_hub import snapshot_download
import torch

root=Path(__file__).resolve().parent
torch.hub.set_dir(str(root/'cache-mac/torch/hub'))
snapshot_download('facebook/map-anything-apache',revision='00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a',cache_dir=str(root/'cache-mac/huggingface/hub'),allow_patterns=['config.json','model.safetensors'])
if not (root/'cache-mac/torch/hub/facebookresearch_dinov2_main/hubconf.py').is_file():
    # Small untrained architecture only, to cache official encoder code. No extra weights.
    encoder=torch.hub.load('facebookresearch/dinov2','dinov2_vits14',pretrained=False)
    del encoder;gc.collect()
print('Weights and encoder code are cached. Apple GPU available:',torch.backends.mps.is_available())
