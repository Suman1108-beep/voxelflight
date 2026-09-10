#!/usr/bin/env python3
"""Join overlapping VGGT windows into one model coordinate system."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pycolmap
import trimesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("window_dirs", nargs="+", type=Path)
    parser.add_argument("--max-points", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    source_mean, target_mean = source.mean(0), target.mean(0)
    source_centered, target_centered = source - source_mean, target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    left, singular_values, right_t = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(left @ right_t) < 0:
        sign[-1] = -1
    rotation = left @ np.diag(sign) @ right_t
    variance = np.mean(np.sum(source_centered**2, axis=1))
    scale = float(np.sum(singular_values * sign) / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def transform(points: np.ndarray, similarity: tuple[float, np.ndarray, np.ndarray]) -> np.ndarray:
    scale, rotation, translation = similarity
    return scale * (points @ rotation.T) + translation


def load_window(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    reconstruction = pycolmap.Reconstruction(path / "sparse")
    centers = {
        image.name: np.asarray(image.projection_center())
        for image in reconstruction.images.values()
    }
    cloud = trimesh.load(path / "points_dense.ply", process=False)
    return centers, np.asarray(cloud.vertices), np.asarray(cloud.colors)


def main() -> int:
    args = parse_args()
    if len(args.window_dirs) < 2:
        raise ValueError("At least two overlapping windows are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    global_centers: dict[str, list[np.ndarray]] = {}
    merged_points: list[np.ndarray] = []
    merged_colors: list[np.ndarray] = []
    window_reports: list[dict[str, object]] = []

    for window_index, window_dir in enumerate(args.window_dirs):
        centers, points, colors = load_window(window_dir)
        if window_index == 0:
            similarity = (1.0, np.eye(3), np.zeros(3))
            overlap_names: list[str] = []
            overlap_rmse = 0.0
        else:
            overlap_names = sorted(set(centers) & set(global_centers))
            if len(overlap_names) < 3:
                raise RuntimeError(
                    f"{window_dir} has only {len(overlap_names)} shared cameras; at least 3 required"
                )
            source = np.asarray([centers[name] for name in overlap_names])
            target = np.asarray(
                [np.mean(global_centers[name], axis=0) for name in overlap_names]
            )
            similarity = umeyama(source, target)
            residual = transform(source, similarity) - target
            overlap_rmse = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))

        transformed_centers = {
            name: transform(center[None], similarity)[0] for name, center in centers.items()
        }
        for name, center in transformed_centers.items():
            global_centers.setdefault(name, []).append(center)
        merged_points.append(transform(points, similarity))
        merged_colors.append(colors)
        window_reports.append(
            {
                "window": str(window_dir),
                "cameras": len(centers),
                "shared_cameras": len(overlap_names),
                "overlap_rmse_model_units": overlap_rmse,
                "scale_to_global": similarity[0],
            }
        )

    all_points = np.concatenate(merged_points)
    all_colors = np.concatenate(merged_colors)
    if len(all_points) > args.max_points:
        chosen = rng.choice(len(all_points), args.max_points, replace=False)
        all_points, all_colors = all_points[chosen], all_colors[chosen]
    trimesh.PointCloud(all_points, colors=all_colors).export(
        args.output_dir / "points_dense_model.ply"
    )

    with (args.output_dir / "camera_trajectory_model.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "x", "y", "z", "window_observations"])
        for name in sorted(global_centers):
            center = np.mean(global_centers[name], axis=0)
            writer.writerow([name, *center, len(global_centers[name])])

    report = {
        "windows": window_reports,
        "unique_cameras": len(global_centers),
        "merged_points": len(all_points),
    }
    (args.output_dir / "merge_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
