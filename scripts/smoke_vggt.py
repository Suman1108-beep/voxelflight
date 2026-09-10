#!/usr/bin/env python3
"""Load the pretrained geometry model and run a minimal GPU inference."""

from __future__ import annotations

import json
import time

import torch

from vggt.models.vggt import VGGT


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke test")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    started = time.perf_counter()
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    loaded_s = time.perf_counter() - started

    # Two deterministic, shifted gradients exercise the complete prediction path.
    torch.manual_seed(7)
    base = torch.linspace(0, 1, 518, device=device)
    grid_x = base[None, :].expand(518, 518)
    grid_y = base[:, None].expand(518, 518)
    first = torch.stack((grid_x, grid_y, (grid_x + grid_y) / 2))
    second = torch.roll(first, shifts=12, dims=2)
    images = torch.stack((first, second), dim=0)

    infer_started = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        output = model(images)
    torch.cuda.synchronize()
    inference_s = time.perf_counter() - infer_started

    summary: dict[str, object] = {
        "model_load_seconds": round(loaded_s, 3),
        "inference_seconds": round(inference_s, 3),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "outputs": {},
    }
    for key, value in output.items():
        if torch.is_tensor(value):
            summary["outputs"][key] = list(value.shape)
        else:
            summary["outputs"][key] = type(value).__name__
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
