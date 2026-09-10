"""CUDA-extension-free compatibility implementation of ``fused_ssim``."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def _gaussian_kernel(
    channels: int, *, device: torch.device, dtype: torch.dtype, size: int = 11
) -> torch.Tensor:
    coordinates = torch.arange(size, device=device, dtype=dtype) - size // 2
    weights = torch.exp(-(coordinates**2) / (2 * 1.5**2))
    weights /= weights.sum()
    kernel = weights[:, None] @ weights[None, :]
    return kernel.expand(channels, 1, size, size).contiguous()


def fused_ssim(
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    padding: str = "same",
    train: bool = True,
) -> torch.Tensor:
    """Compute differentiable mean SSIM with the fused extension's interface."""
    del train
    if image_a.shape != image_b.shape or image_a.ndim != 4:
        raise ValueError("Expected equal BCHW image tensors")
    channels = image_a.shape[1]
    kernel = _gaussian_kernel(channels, device=image_a.device, dtype=image_a.dtype)
    pad = kernel.shape[-1] // 2 if padding == "same" else 0

    mean_a = functional.conv2d(image_a, kernel, padding=pad, groups=channels)
    mean_b = functional.conv2d(image_b, kernel, padding=pad, groups=channels)
    mean_a_sq, mean_b_sq = mean_a.square(), mean_b.square()
    mean_ab = mean_a * mean_b
    variance_a = functional.conv2d(image_a.square(), kernel, padding=pad, groups=channels) - mean_a_sq
    variance_b = functional.conv2d(image_b.square(), kernel, padding=pad, groups=channels) - mean_b_sq
    covariance = functional.conv2d(image_a * image_b, kernel, padding=pad, groups=channels) - mean_ab

    constant_1, constant_2 = 0.01**2, 0.03**2
    numerator = (2 * mean_ab + constant_1) * (2 * covariance + constant_2)
    denominator = (mean_a_sq + mean_b_sq + constant_1) * (
        variance_a + variance_b + constant_2
    )
    return (numerator / denominator.clamp_min(torch.finfo(image_a.dtype).eps)).mean()


__all__ = ["fused_ssim"]
