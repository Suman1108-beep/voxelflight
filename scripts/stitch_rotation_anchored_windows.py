#!/usr/bin/env python3
"""Fuse local VGGT translations under globally averaged feature rotations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pycolmap

from align_metric import (
    apply_similarity,
    error_summary,
    interpolate_reference,
    load_manifest,
    load_reference,
    segment_scale_error,
    umeyama,
)
from evaluate_global_sfm import rigid_alignment, robust_metric_scale
from stitch_depth_windows import Window, edge_weight, load_window
from stitch_pose_windows import mean_rotation, rotation_angle_degrees


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("rotation_anchor_model", type=Path)
    parser.add_argument("keyframe_manifest", type=Path)
    parser.add_argument("ground_truth_csv", type=Path)
    parser.add_argument("window_npz", nargs="+", type=Path)
    parser.add_argument("--source-image-offset", type=int, required=True)
    parser.add_argument("--metric-window", type=Path, action="append", default=[])
    return parser.parse_args()


def robust_mean_rotation(candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights = np.ones(len(candidates))
    for _ in range(10):
        rotation = mean_rotation(candidates, weights)
        residuals = np.asarray(
            [rotation_angle_degrees(candidate @ rotation.T) for candidate in candidates]
        )
        median = np.median(residuals)
        mad = max(1.4826 * np.median(np.abs(residuals - median)), 0.05)
        cutoff = median + 2.5 * mad
        weights = np.minimum(1.0, cutoff / np.maximum(residuals, 1e-6))
    return rotation, residuals


def fit_fixed_rotation_similarity(
    centres: np.ndarray,
    rotation: np.ndarray,
    target: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    rotated = centres @ rotation.T
    weights = np.ones(len(rotated))
    for _ in range(10):
        source_mean = np.average(rotated, axis=0, weights=weights)
        target_mean = np.average(target, axis=0, weights=weights)
        source_delta = rotated - source_mean
        target_delta = target - target_mean
        denominator = np.sum(weights[:, None] * source_delta**2)
        scale = float(
            np.sum(weights[:, None] * source_delta * target_delta) / denominator
        )
        scale = max(scale, 1e-6)
        translation = target_mean - scale * source_mean
        residuals = np.linalg.norm(scale * rotated + translation - target, axis=1)
        median = np.median(residuals)
        mad = max(1.4826 * np.median(np.abs(residuals - median)), 1e-5)
        cutoff = median + 2.5 * mad
        weights = np.minimum(1.0, cutoff / np.maximum(residuals, 1e-8))
    return scale, translation, residuals


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    anchor = pycolmap.Reconstruction(args.rotation_anchor_model)
    anchor_orientations = {
        image.name: image.cam_from_world.rotation.matrix().T
        for image in anchor.images.values()
    }
    windows: list[Window] = [load_window(path) for path in args.window_npz]
    observations: dict[str, list[tuple[np.ndarray, float]]] = {}
    reports: list[dict[str, object]] = []

    for window_index, window in enumerate(windows):
        local_orientations = np.transpose(
            window.extrinsics[:, :3, :3], (0, 2, 1)
        )
        usable = [i for i, name in enumerate(window.names) if name in anchor_orientations]
        candidates = np.asarray(
            [
                anchor_orientations[window.names[i]] @ local_orientations[i].T
                for i in usable
            ]
        )
        rotation, angular_residuals = robust_mean_rotation(candidates)

        if window_index == 0:
            scale = 1.0
            translation = -(rotation @ window.centres[0])
            position_residuals = np.zeros(0)
            overlap_names: list[str] = []
        else:
            overlap_names = [name for name in window.names if name in observations]
            local_indices = [window.names.index(name) for name in overlap_names]
            targets = np.asarray(
                [
                    np.average(
                        np.asarray([point for point, _ in observations[name]]),
                        axis=0,
                        weights=np.asarray([weight for _, weight in observations[name]]),
                    )
                    for name in overlap_names
                ]
            )
            scale, translation, position_residuals = fit_fixed_rotation_similarity(
                window.centres[local_indices], rotation, targets
            )

        transformed = apply_similarity(window.centres, scale, rotation, translation)
        for local_index, (name, point) in enumerate(zip(window.names, transformed)):
            observations.setdefault(name, []).append(
                (point, edge_weight(local_index, len(window.names)))
            )
        reports.append(
            {
                "window": str(window.path),
                "frames": len(window.names),
                "anchor_orientation_median_deg": float(np.median(angular_residuals)),
                "anchor_orientation_p95_deg": float(np.percentile(angular_residuals, 95)),
                "overlap_frames": len(overlap_names),
                "overlap_position_rmse": (
                    float(np.sqrt(np.mean(position_residuals**2)))
                    if len(position_residuals)
                    else 0.0
                ),
                "scale_to_global": scale,
                "rotation": rotation.tolist(),
                "translation": translation.tolist(),
            }
        )

    manifest = load_manifest(args.keyframe_manifest)
    names = np.asarray(sorted(observations, key=lambda name: manifest[name]))
    stitched = np.asarray(
        [
            np.average(
                np.asarray([point for point, _ in observations[name]]),
                axis=0,
                weights=np.asarray([weight for _, weight in observations[name]]),
            )
            for name in names
        ]
    )
    image_ids = np.asarray(
        [manifest[name] + args.source_image_offset for name in names]
    )
    sample_ids, gt_samples, gps_samples = load_reference(args.ground_truth_csv)
    ground_truth = interpolate_reference(image_ids, sample_ids, gt_samples)
    gps = interpolate_reference(image_ids, sample_ids, gps_samples)

    shape_scale, shape_rotation, shape_translation = umeyama(stitched, ground_truth)
    shape_aligned = apply_similarity(
        stitched, shape_scale, shape_rotation, shape_translation
    )
    metric_scale, metric_estimates = robust_metric_scale(
        names, stitched, args.metric_window
    )
    metric = stitched * metric_scale
    metric_rotation, metric_translation = rigid_alignment(metric, ground_truth)
    metric_aligned = metric @ metric_rotation.T + metric_translation
    gps_rotation, gps_translation = rigid_alignment(metric, gps)
    georeferenced = metric @ gps_rotation.T + gps_translation

    result = {
        "protocol": {
            "ground_truth_used_in_fusion": False,
            "rotation_source": str(args.rotation_anchor_model),
            "translation_source": "overlapping VGGT windows",
            "metric_scale_source": "MapAnything windows",
            "ground_truth_use": "post-hoc benchmark scoring only",
        },
        "frames": len(names),
        "windows": reports,
        "sim3_shape_error": error_summary(shape_aligned, ground_truth),
        "metric_scale": {
            "estimated_m_per_model_unit": metric_scale,
            "independent_window_estimates": metric_estimates,
            "se3_error": error_summary(metric_aligned, ground_truth),
            "segment_length_error": segment_scale_error(metric_aligned, ground_truth),
        },
        "gps_georeferenced_absolute_error": error_summary(
            georeferenced, ground_truth
        ),
    }
    np.savez_compressed(
        args.output_dir / "trajectory.npz",
        image_names=names,
        image_ids=image_ids,
        camera_centers=stitched,
        metric_camera_centers=metric,
    )
    with (args.output_dir / "trajectory.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["image", "image_id", "x", "y", "z", "shape_x", "shape_y", "shape_z", "gt_x", "gt_y", "gt_z"]
        )
        for name, image_id, point, shape, truth in zip(
            names, image_ids, stitched, shape_aligned, ground_truth
        ):
            writer.writerow([name, image_id, *point, *shape, *truth])
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
