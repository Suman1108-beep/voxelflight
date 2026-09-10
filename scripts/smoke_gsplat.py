#!/usr/bin/env python3
"""Compile and exercise gsplat's CUDA forward/backward rasterizer."""

from __future__ import annotations

import json
import time

import torch
import torch.nn.functional as functional
from gsplat.rendering import rasterization


def main() -> int:
    device = torch.device("cuda")
    torch.manual_seed(11)
    count = 512
    means = torch.randn(count, 3, device=device, requires_grad=True)
    with torch.no_grad():
        means[:, :2].mul_(0.6)
        means[:, 2].mul_(0.2).add_(3.0)
    quats = functional.normalize(torch.randn(count, 4, device=device), dim=-1)
    scales = torch.full((count, 3), 0.04, device=device)
    opacities = torch.full((count,), 0.75, device=device)
    colors = torch.rand(count, 3, device=device)
    viewmats = torch.eye(4, device=device)[None]
    intrinsics = torch.tensor(
        [[[150.0, 0.0, 64.0], [0.0, 150.0, 64.0], [0.0, 0.0, 1.0]]],
        device=device,
    )

    started = time.perf_counter()
    rendered, alpha, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=intrinsics,
        width=128,
        height=128,
    )
    loss = rendered.mean() + alpha.mean()
    loss.backward()
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "render_shape": list(rendered.shape),
                "alpha_shape": list(alpha.shape),
                "loss": float(loss.detach()),
                "gradient_finite": bool(torch.isfinite(means.grad).all()),
                "compile_and_run_seconds": round(time.perf_counter() - started, 3),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
