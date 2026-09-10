#!/usr/bin/env python3
"""Run gsplat 1.5.3 trainer with its modern official-PyCOLMAP parser."""

from __future__ import annotations

import runpy
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F


def torch_fused_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    padding: str = "valid",
    **_: object,
) -> torch.Tensor:
    """Differentiable SSIM fallback for hosts without the optional CUDA add-on."""
    window = 11
    pad = window // 2 if padding == "same" else 0
    mean_prediction = F.avg_pool2d(prediction, window, 1, pad)
    mean_target = F.avg_pool2d(target, window, 1, pad)
    prediction_variance = (
        F.avg_pool2d(prediction.square(), window, 1, pad)
        - mean_prediction.square()
    )
    target_variance = (
        F.avg_pool2d(target.square(), window, 1, pad) - mean_target.square()
    )
    covariance = (
        F.avg_pool2d(prediction * target, window, 1, pad)
        - mean_prediction * mean_target
    )
    c1, c2 = 0.01**2, 0.03**2
    score = (
        (2 * mean_prediction * mean_target + c1) * (2 * covariance + c2)
        / (
            (mean_prediction.square() + mean_target.square() + c1)
            * (prediction_variance + target_variance + c2)
        )
    )
    return score.mean()


def main() -> None:
    stable_root = Path("/mnt/road/sih3d/models/gsplat")
    modern_root = Path("/mnt/road/sih3d/models/gsplat-main")
    stable_examples = stable_root / "examples"
    modern_examples = modern_root / "examples"

    sys.path.insert(0, str(modern_examples))
    import datasets  # type: ignore[import-not-found]
    import datasets.colmap  # noqa: F401  # type: ignore[import-not-found]

    datasets.__path__.insert(0, str(stable_examples / "datasets"))
    fused_ssim = types.ModuleType("fused_ssim")
    fused_ssim.fused_ssim = torch_fused_ssim
    sys.modules.setdefault("fused_ssim", fused_ssim)
    sys.path.remove(str(modern_examples))
    sys.path.insert(0, str(stable_examples))
    runpy.run_path(str(stable_examples / "simple_trainer.py"), run_name="__main__")


if __name__ == "__main__":
    main()
