#!/usr/bin/env python3
"""Render an inspection sheet for a metric point cloud and camera trajectory."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("point_cloud", type=Path)
    parser.add_argument("trajectory_csv", type=Path)
    parser.add_argument("reference_image", type=Path)
    parser.add_argument("output_png", type=Path)
    parser.add_argument("--max-points", type=int, default=120_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cloud = trimesh.load(args.point_cloud, process=False)
    points = np.asarray(cloud.vertices)
    colors = np.asarray(cloud.colors)[:, :3] / 255.0
    total_points = len(points)
    rng = np.random.default_rng(17)
    if len(points) > args.max_points:
        chosen = rng.choice(len(points), args.max_points, replace=False)
        points, colors = points[chosen], colors[chosen]

    with args.trajectory_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    fieldnames = set(rows[0]) if rows else set()

    def columns(prefix: str) -> np.ndarray | None:
        if not {f"{prefix}_{axis}" for axis in "xyz"} <= fieldnames:
            return None
        return np.asarray(
            [[float(row[f"{prefix}_{axis}"]) for axis in "xyz"] for row in rows]
        )

    prediction = columns("pred")
    if prediction is None:
        prediction = columns("camera")
    if prediction is None:
        raise KeyError("Trajectory needs pred_x/y/z or camera_x/y/z columns")
    truth, gps = columns("gt"), columns("gps")
    center = np.median(points, axis=0)
    local_points = points - center
    local_prediction = prediction - center
    local_truth = None if truth is None else truth - center
    local_gps = None if gps is None else gps - center

    figure = plt.figure(figsize=(16, 9), constrained_layout=True, facecolor="#0b1020")
    grid = figure.add_gridspec(2, 2, width_ratios=(1.05, 1.45))
    ax_image = figure.add_subplot(grid[0, 0])
    ax_top = figure.add_subplot(grid[1, 0])
    ax_cloud = figure.add_subplot(grid[:, 1], projection="3d")

    ax_image.imshow(Image.open(args.reference_image).convert("RGB"))
    ax_image.set_title("Observed drone frame", color="white", fontsize=14)
    ax_image.axis("off")

    if local_truth is not None:
        ax_top.plot(local_truth[:, 0], local_truth[:, 1], color="#35e6a7", linewidth=3, label="evaluation reference")
    ax_top.plot(local_prediction[:, 0], local_prediction[:, 1], "--", color="#ffd166", linewidth=2, label="visual + GNSS/IMU fusion")
    if local_gps is not None:
        ax_top.plot(local_gps[:, 0], local_gps[:, 1], ":", color="#56b4ff", linewidth=2, label="raw onboard GPS")
    ax_top.set_title("Top-down georeferenced camera trajectory", color="white", fontsize=14)
    ax_top.set_xlabel("local Easting (m)", color="white")
    ax_top.set_ylabel("local Northing (m)", color="white")
    ax_top.set_aspect("equal", adjustable="datalim")
    ax_top.grid(alpha=0.2)
    ax_top.legend(facecolor="#11182b", labelcolor="white", fontsize=9)

    ax_cloud.scatter(
        local_points[:, 0], local_points[:, 1], local_points[:, 2],
        c=colors, s=0.12, alpha=0.75, depthshade=False, rasterized=True,
    )
    ax_cloud.plot(
        local_prediction[:, 0], local_prediction[:, 1], local_prediction[:, 2],
        color="#ffd166", linewidth=2.5,
    )
    ax_cloud.set_title(
        f"{total_points / 1_000_000:.1f}M-point metric reconstruction "
        f"({len(points) / 1_000:.0f}k shown)",
        color="white",
        fontsize=15,
    )
    ax_cloud.set_xlabel("E (m)")
    ax_cloud.set_ylabel("N (m)")
    ax_cloud.set_zlabel("Z (m)")
    ax_cloud.view_init(elev=35, azim=-65)
    ax_cloud.set_box_aspect(np.ptp(local_points, axis=0))

    for axis in (ax_top, ax_cloud):
        axis.set_facecolor("#0b1020")
        axis.tick_params(colors="white")
        for spine in getattr(axis, "spines", {}).values():
            spine.set_color("#54617a")
    figure.suptitle(
        "SIH Single-Pass Drone Video → Metric 3D Reconstruction",
        color="white", fontsize=19, fontweight="bold",
    )
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_png, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)
    print(args.output_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
