#!/usr/bin/env python3
"""Align a VGGT reconstruction to Zurich MAV metric/GPS coordinates."""

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
    parser.add_argument("reconstruction_dir", type=Path)
    parser.add_argument("keyframe_manifest", type=Path)
    parser.add_argument("ground_truth_csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return scale, rotation and translation for target ~= s R source + t."""
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    left, singular_values, right_t = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(left @ right_t) < 0:
        sign[-1] = -1
    rotation = left @ np.diag(sign) @ right_t
    variance = np.mean(np.sum(source_centered**2, axis=1))
    if variance <= np.finfo(float).eps:
        raise ValueError("Camera centers do not span a usable trajectory")
    scale = float(np.sum(singular_values * sign) / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def apply_similarity(
    points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    return scale * (points @ rotation.T) + translation


def load_manifest(path: Path) -> dict[str, int]:
    with path.open(newline="") as handle:
        return {
            row["output_frame"]: int(row["source_frame"])
            for row in csv.DictReader(handle)
        }


def load_reference(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_ids: list[int] = []
    ground_truth: list[list[float]] = []
    gps: list[list[float]] = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle, skipinitialspace=True):
            row = {key.strip(): value for key, value in raw.items() if key is not None}
            image_ids.append(int(row["imgid"]))
            ground_truth.append([float(row[name]) for name in ("x_gt", "y_gt", "z_gt")])
            gps.append([float(row[name]) for name in ("x_gps", "y_gps", "z_gps")])
    return np.asarray(image_ids), np.asarray(ground_truth), np.asarray(gps)


def interpolate_reference(
    query_ids: np.ndarray, sample_ids: np.ndarray, values: np.ndarray
) -> np.ndarray:
    return np.column_stack(
        [np.interp(query_ids, sample_ids, values[:, axis]) for axis in range(3)]
    )


def error_summary(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    distances = np.linalg.norm(prediction - reference, axis=1)
    return {
        "rmse_m": float(np.sqrt(np.mean(distances**2))),
        "median_m": float(np.median(distances)),
        "p95_m": float(np.percentile(distances, 95)),
        "max_m": float(np.max(distances)),
    }


def segment_scale_error(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    predicted_steps = np.linalg.norm(np.diff(prediction, axis=0), axis=1)
    reference_steps = np.linalg.norm(np.diff(reference, axis=0), axis=1)
    usable = reference_steps > 0.03
    relative = np.abs(predicted_steps[usable] - reference_steps[usable]) / reference_steps[usable]
    return {
        "segments": int(usable.sum()),
        "median_percent": float(np.median(relative) * 100),
        "p95_percent": float(np.percentile(relative, 95) * 100),
    }


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.reconstruction_dir / "metric"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.keyframe_manifest)
    reconstruction = pycolmap.Reconstruction(args.reconstruction_dir / "sparse")
    ordered_images = [reconstruction.images[key] for key in sorted(reconstruction.images)]
    source_frames = np.asarray([manifest[image.name] for image in ordered_images])
    image_ids = source_frames + 1
    camera_centers = np.asarray([image.projection_center() for image in ordered_images])

    sample_ids, ground_truth_samples, gps_samples = load_reference(args.ground_truth_csv)
    ground_truth = interpolate_reference(image_ids, sample_ids, ground_truth_samples)
    gps = interpolate_reference(image_ids, sample_ids, gps_samples)

    gt_scale, gt_rotation, gt_translation = umeyama(camera_centers, ground_truth)
    gt_aligned = apply_similarity(camera_centers, gt_scale, gt_rotation, gt_translation)
    gps_scale, gps_rotation, gps_translation = umeyama(camera_centers, gps)
    gps_aligned = apply_similarity(camera_centers, gps_scale, gps_rotation, gps_translation)

    point_cloud = trimesh.load(args.reconstruction_dir / "points_dense.ply", process=False)
    vertices_metric = apply_similarity(
        np.asarray(point_cloud.vertices), gt_scale, gt_rotation, gt_translation
    )
    trimesh.PointCloud(vertices_metric, colors=point_cloud.colors).export(
        output_dir / "points_metric_utm.ply"
    )

    metric_reconstruction = pycolmap.Reconstruction(args.reconstruction_dir / "sparse")
    metric_reconstruction.transform(
        pycolmap.Sim3d(gt_scale, pycolmap.Rotation3d(gt_rotation), gt_translation)
    )
    sparse_metric_dir = output_dir / "sparse"
    sparse_metric_dir.mkdir(exist_ok=True)
    metric_reconstruction.write(sparse_metric_dir)

    with (output_dir / "camera_trajectory.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "image",
                "source_frame",
                "image_id",
                "pred_x",
                "pred_y",
                "pred_z",
                "gt_x",
                "gt_y",
                "gt_z",
                "gps_x",
                "gps_y",
                "gps_z",
            ]
        )
        for image, source_frame, prediction, truth, gps_point in zip(
            ordered_images, source_frames, gt_aligned, ground_truth, gps
        ):
            writer.writerow(
                [image.name, source_frame, source_frame + 1, *prediction, *truth, *gps_point]
            )

    metrics = {
        "coordinate_system": "Zurich MAV projected metric coordinates (dataset frame)",
        "frames": len(camera_centers),
        "gt_alignment": {
            "scale_m_per_model_unit": gt_scale,
            "trajectory_error": error_summary(gt_aligned, ground_truth),
            "segment_length_error": segment_scale_error(gt_aligned, ground_truth),
        },
        "gps_alignment": {
            "scale_m_per_model_unit": gps_scale,
            "absolute_error_against_gt": error_summary(gps_aligned, ground_truth),
        },
        "raw_gps_error_against_gt": error_summary(gps, ground_truth),
    }
    (output_dir / "metric_report.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
