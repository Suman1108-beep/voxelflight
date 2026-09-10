#!/usr/bin/env python3
"""Fail fast if the SIH environment cannot use the reconstruction GPU."""

from __future__ import annotations

import platform
import sys

import torch


def main() -> int:
    print(f"Python: {platform.python_version()}")
    print(f"Executable: {sys.executable}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("ERROR: CUDA is unavailable in this environment.", file=sys.stderr)
        return 1

    props = torch.cuda.get_device_properties(0)
    total_gib = props.total_memory / 1024**3
    print(f"GPU: {props.name}")
    print(f"VRAM: {total_gib:.1f} GiB")
    print(f"Compute capability: {props.major}.{props.minor}")

    x = torch.randn((2048, 2048), device="cuda", dtype=torch.float16)
    y = x @ x
    torch.cuda.synchronize()
    print(f"CUDA smoke test: PASS ({float(y[0, 0]):.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

