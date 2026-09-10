#!/usr/bin/env python3
"""Render orthographic inspection views of a large colored GLB scene."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("glb", type=Path)
    parser.add_argument("reference_image", type=Path)
    parser.add_argument("output_png", type=Path)
    parser.add_argument("--max-points", type=int, default=240_000)
    args = parser.parse_args()

    scene = trimesh.load(args.glb, process=False)
    vertices: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node_name]
        geometry = scene.geometry[geometry_name]
        points = trimesh.transform_points(np.asarray(geometry.vertices), transform)
        vertex_colors = np.asarray(geometry.visual.vertex_colors)[:, :3]
        vertices.append(points)
        colors.append(vertex_colors)
    points = np.concatenate(vertices)
    rgb = np.concatenate(colors) / 255.0
    total_points = len(points)
    finite = np.isfinite(points).all(axis=1)
    points, rgb = points[finite], rgb[finite]
    rng = np.random.default_rng(17)
    if len(points) > args.max_points:
        chosen = rng.choice(len(points), args.max_points, replace=False)
        points, rgb = points[chosen], rgb[chosen]

    low, high = np.percentile(points, [1, 99], axis=0)
    within = ((points >= low) & (points <= high)).all(axis=1)
    points, rgb = points[within], rgb[within]
    center = np.median(points, axis=0)
    points = points - center

    figure = plt.figure(figsize=(16, 9), constrained_layout=True, facecolor="#0b1020")
    grid = figure.add_gridspec(2, 3)
    ax_image = figure.add_subplot(grid[:, 0])
    ax_3d = figure.add_subplot(grid[:, 1], projection="3d")
    ax_top = figure.add_subplot(grid[0, 2])
    ax_side = figure.add_subplot(grid[1, 2])

    ax_image.imshow(Image.open(args.reference_image).convert("RGB"))
    ax_image.set_title("Observed frame", color="white", fontsize=14)
    ax_image.axis("off")
    ax_3d.scatter(points[:, 0], points[:, 1], points[:, 2], c=rgb, s=0.12, alpha=0.8, depthshade=False, rasterized=True)
    ax_3d.view_init(elev=25, azim=-60)
    ax_3d.set_box_aspect(np.maximum(np.ptp(points, axis=0), 1e-3))
    ax_3d.set_title("Perspective", color="white", fontsize=14)
    ax_top.scatter(points[:, 0], points[:, 2], c=rgb, s=0.14, alpha=0.8, rasterized=True)
    ax_top.set_title("Top/plan view", color="white", fontsize=14)
    ax_top.set_aspect("equal", adjustable="datalim")
    ax_side.scatter(points[:, 0], points[:, 1], c=rgb, s=0.14, alpha=0.8, rasterized=True)
    ax_side.set_title("Elevation view", color="white", fontsize=14)
    ax_side.set_aspect("equal", adjustable="datalim")
    for axis in (ax_3d, ax_top, ax_side):
        axis.set_facecolor("#0b1020")
        axis.tick_params(colors="white")
        axis.grid(alpha=0.2)
    figure.suptitle(
        f"MapAnything Metric Reconstruction · {total_points / 1_000_000:.1f}M source points",
        color="white",
        fontsize=19,
        fontweight="bold",
    )
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_png, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)
    print(args.output_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
