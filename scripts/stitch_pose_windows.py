#!/usr/bin/env python3
"""Globally stitch COLMAP windows using both positions and camera orientations.

Using camera orientation removes the unobservable rotation around an almost
straight flight trajectory that makes centre-only Sim(3) registration unstable.
Ground truth is never used to estimate a window transform; it is an optional
post-hoc benchmark input.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap
import trimesh

from align_metric import (
    apply_similarity,
    error_summary,
    interpolate_reference,
    load_manifest,
    load_reference,
    segment_scale_error,
    umeyama,
)


@dataclass(frozen=True)
class Window:
    path: Path
    names: list[str]
    centres: np.ndarray
    camera_to_world_rotations: np.ndarray
    points: np.ndarray
    colors: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("keyframe_manifest", type=Path)
    parser.add_argument("window_dirs", nargs="+", type=Path)
    parser.add_argument("--source-image-offset", type=int, required=True)
    parser.add_argument("--ground-truth-csv", type=Path)
    parser.add_argument("--minimum-overlap", type=int, default=6)
    parser.add_argument("--max-points", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def load_window(path: Path) -> Window:
    reconstruction = pycolmap.Reconstruction(path / "sparse")
    images = sorted(reconstruction.images.values(), key=lambda image: image.name)
    cloud = trimesh.load(path / "points_dense.ply", process=False)
    return Window(
        path=path,
        names=[image.name for image in images],
        centres=np.asarray([image.projection_center() for image in images]),
        camera_to_world_rotations=np.asarray(
            [image.cam_from_world.rotation.matrix().T for image in images]
        ),
        points=np.asarray(cloud.vertices),
        colors=np.asarray(cloud.colors),
    )


def mean_rotation(rotations: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    if weights is None:
        weights = np.ones(len(rotations))
    matrix = np.sum(rotations * weights[:, None, None], axis=0)
    left, _, right_t = np.linalg.svd(matrix)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(left @ right_t))
    return left @ correction @ right_t


def rotation_angle_degrees(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def fit_orientation_constrained_similarity(
    source_centres: np.ndarray,
    source_orientations: np.ndarray,
    target_centres: np.ndarray,
    target_orientations: np.ndarray,
    iterations: int = 8,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    candidate_rotations = target_orientations @ np.transpose(
        source_orientations, (0, 2, 1)
    )
    weights = np.ones(len(source_centres))
    for _ in range(iterations):
        rotation = mean_rotation(candidate_rotations, weights)
        rotated = source_centres @ rotation.T
        source_mean = np.average(rotated, axis=0, weights=weights)
        target_mean = np.average(target_centres, axis=0, weights=weights)
        source_delta = rotated - source_mean
        target_delta = target_centres - target_mean
        numerator = np.sum(weights[:, None] * source_delta * target_delta)
        denominator = np.sum(weights[:, None] * source_delta**2)
        scale = max(float(numerator / denominator), 1e-6)
        translation = target_mean - scale * source_mean
        prediction = scale * rotated + translation
        position_residuals = np.linalg.norm(prediction - target_centres, axis=1)
        angular_residuals = np.asarray(
            [
                rotation_angle_degrees(candidate @ rotation.T)
                for candidate in candidate_rotations
            ]
        )
        median = np.median(position_residuals)
        mad = np.median(np.abs(position_residuals - median))
        position_delta = max(1.4826 * mad * 1.5, 0.03)
        position_weights = np.minimum(
            1.0, position_delta / np.maximum(position_residuals, 1e-9)
        )
        angular_weights = np.minimum(
            1.0, 2.0 / np.maximum(angular_residuals, 1e-9)
        )
        weights = position_weights * angular_weights
    return scale, rotation, translation, position_residuals, angular_residuals


def edge_weight(index: int, length: int) -> float:
    if length <= 1:
        return 1.0
    normalized = abs((2.0 * index / (length - 1)) - 1.0)
    return float(0.25 + 0.75 * np.cos(0.5 * np.pi * normalized) ** 2)


def rigid_alignment(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_mean, target_mean = source.mean(0), target.mean(0)
    covariance = (target - target_mean).T @ (source - source_mean)
    left, _, right_t = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(left @ right_t))
    rotation = left @ correction @ right_t
    return rotation, target_mean - rotation @ source_mean


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    windows = [load_window(path) for path in args.window_dirs]
    manifest = load_manifest(args.keyframe_manifest)
    rng = np.random.default_rng(args.seed)

    centre_observations: dict[str, list[tuple[np.ndarray, float]]] = {}
    orientation_observations: dict[str, list[tuple[np.ndarray, float]]] = {}
    transformed_points: list[np.ndarray] = []
    point_colors: list[np.ndarray] = []
    window_reports: list[dict[str, object]] = []

    for window_index, window in enumerate(windows):
        if window_index == 0:
            scale, rotation, translation = 1.0, np.eye(3), np.zeros(3)
            position_residuals = np.zeros(0)
            angular_residuals = np.zeros(0)
            overlap_names: list[str] = []
        else:
            name_to_index = {name: index for index, name in enumerate(window.names)}
            overlap_names = sorted(set(name_to_index) & set(centre_observations))
            if len(overlap_names) < args.minimum_overlap:
                raise RuntimeError(
                    f"{window.path} has only {len(overlap_names)} placed frames; "
                    f"need {args.minimum_overlap}"
                )
            local_indices = [name_to_index[name] for name in overlap_names]
            source_centres = window.centres[local_indices]
            source_orientations = window.camera_to_world_rotations[local_indices]
            target_centres = np.asarray(
                [
                    np.average(
                        np.asarray([value for value, _ in centre_observations[name]]),
                        axis=0,
                        weights=np.asarray([weight for _, weight in centre_observations[name]]),
                    )
                    for name in overlap_names
                ]
            )
            target_orientations = np.asarray(
                [
                    mean_rotation(
                        np.asarray([value for value, _ in orientation_observations[name]]),
                        np.asarray([weight for _, weight in orientation_observations[name]]),
                    )
                    for name in overlap_names
                ]
            )
            (
                scale,
                rotation,
                translation,
                position_residuals,
                angular_residuals,
            ) = fit_orientation_constrained_similarity(
                source_centres,
                source_orientations,
                target_centres,
                target_orientations,
            )

        global_centres = apply_similarity(
            window.centres, scale, rotation, translation
        )
        global_orientations = rotation[None] @ window.camera_to_world_rotations
        for local_index, (name, centre, orientation) in enumerate(
            zip(window.names, global_centres, global_orientations)
        ):
            weight = edge_weight(local_index, len(window.names))
            centre_observations.setdefault(name, []).append((centre, weight))
            orientation_observations.setdefault(name, []).append((orientation, weight))
        transformed_points.append(
            apply_similarity(window.points, scale, rotation, translation)
        )
        point_colors.append(window.colors)
        window_reports.append(
            {
                "window": str(window.path),
                "frames": len(window.names),
                "overlap_frames": len(overlap_names),
                "scale_to_global": scale,
                "overlap_position_rmse": (
                    float(np.sqrt(np.mean(position_residuals**2)))
                    if len(position_residuals)
                    else 0.0
                ),
                "overlap_orientation_median_degrees": (
                    float(np.median(angular_residuals))
                    if len(angular_residuals)
                    else 0.0
                ),
                "rotation": rotation.tolist(),
                "translation": translation.tolist(),
            }
        )

    ordered_names = sorted(centre_observations, key=lambda name: manifest[name])
    stitched = np.asarray(
        [
            np.average(
                np.asarray([value for value, _ in centre_observations[name]]),
                axis=0,
                weights=np.asarray([weight for _, weight in centre_observations[name]]),
            )
            for name in ordered_names
        ]
    )
    image_ids = np.asarray(
        [manifest[name] + args.source_image_offset for name in ordered_names]
    )
    report: dict[str, object] = {
        "method": "orientation-constrained robust Sim(3) window stitching",
        "frames": len(ordered_names),
        "windows": window_reports,
        "benchmark_uses_ground_truth_only_after_stitching": True,
    }

    benchmark_columns: list[np.ndarray] = []
    if args.ground_truth_csv:
        sample_ids, ground_truth_samples, gps_samples = load_reference(
            args.ground_truth_csv
        )
        ground_truth = interpolate_reference(image_ids, sample_ids, ground_truth_samples)
        gps = interpolate_reference(image_ids, sample_ids, gps_samples)
        rigid_rotation, rigid_translation = rigid_alignment(stitched, ground_truth)
        rigid_aligned = stitched @ rigid_rotation.T + rigid_translation
        sim_scale, sim_rotation, sim_translation = umeyama(stitched, ground_truth)
        sim_aligned = apply_similarity(stitched, sim_scale, sim_rotation, sim_translation)
        gps_rotation, gps_translation = rigid_alignment(stitched, gps)
        gps_aligned = stitched @ gps_rotation.T + gps_translation
        report["evaluation"] = {
            "se3_metric_error": error_summary(rigid_aligned, ground_truth),
            "sim3_shape_error": error_summary(sim_aligned, ground_truth),
            "global_scale_correction_needed": sim_scale,
            "segment_length_error": segment_scale_error(rigid_aligned, ground_truth),
            "gps_anchored_absolute_error": error_summary(gps_aligned, ground_truth),
            "raw_gps_absolute_error": error_summary(gps, ground_truth),
        }
        benchmark_columns = [ground_truth, gps, rigid_aligned, sim_aligned]

    with (args.output_dir / "camera_trajectory_model.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.writer(handle)
        header = ["image", "x", "y", "z", "window_observations", "image_id"]
        if benchmark_columns:
            header += [
                "gt_x",
                "gt_y",
                "gt_z",
                "gps_x",
                "gps_y",
                "gps_z",
                "se3_x",
                "se3_y",
                "se3_z",
                "sim3_x",
                "sim3_y",
                "sim3_z",
            ]
        writer.writerow(header)
        for row_index, (name, image_id, centre) in enumerate(
            zip(ordered_names, image_ids, stitched)
        ):
            row: list[object] = [
                name,
                *centre,
                len(centre_observations[name]),
                image_id,
            ]
            for values in benchmark_columns:
                row.extend(values[row_index])
            writer.writerow(row)

    points, colors = np.concatenate(transformed_points), np.concatenate(point_colors)
    if len(points) > args.max_points:
        selected = rng.choice(len(points), args.max_points, replace=False)
        points, colors = points[selected], colors[selected]
    trimesh.PointCloud(points, colors=colors).export(
        args.output_dir / "points_dense_model.ply"
    )
    report["merged_points"] = len(points)
    (args.output_dir / "stitch_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
