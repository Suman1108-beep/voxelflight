#!/usr/bin/env python3
"""Evaluate global SfM without using reference poses to construct the solution.

The reference trajectory is used only after reconstruction to report benchmark
errors. Metric scale is estimated independently from feed-forward MapAnything
windows; georeferencing is estimated independently from the supplied GPS.
"""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("sparse_model", type=Path)
    parser.add_argument("keyframe_manifest", type=Path)
    parser.add_argument("ground_truth_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--source-image-offset", type=int, required=True)
    parser.add_argument(
        "--metric-window",
        type=Path,
        action="append",
        default=[],
        help="MapAnything raw_prediction.npz; may be repeated",
    )
    return parser.parse_args()


def rigid_alignment(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return R,t for target ~= R source + t, with scale fixed to one."""
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    covariance = (target - target_mean).T @ (source - source_mean)
    left, _, right_t = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(left @ right_t))
    rotation = left @ correction @ right_t
    translation = target_mean - rotation @ source_mean
    return rotation, translation


def robust_metric_scale(
    names: np.ndarray,
    centers: np.ndarray,
    window_paths: list[Path],
) -> tuple[float, list[dict[str, float | int | str]]]:
    """Estimate model-units-to-metres scale from independent metric predictions."""
    center_by_name = dict(zip(names.tolist(), centers))
    estimates: list[dict[str, float | int | str]] = []
    for path in window_paths:
        prediction = np.load(path)
        common = [
            str(name)
            for name in prediction["image_names"]
            if str(name) in center_by_name
        ]
        if len(common) < 3:
            continue
        source = np.asarray([center_by_name[name] for name in common])
        window_lookup = {
            str(name): point
            for name, point in zip(
                prediction["image_names"], prediction["camera_centers"]
            )
        }
        target = np.asarray([window_lookup[name] for name in common])
        scale, rotation, translation = umeyama(source, target)
        fitted = apply_similarity(source, scale, rotation, translation)
        residual = error_summary(fitted, target)
        estimates.append(
            {
                "path": str(path),
                "frames": len(common),
                "scale_m_per_model_unit": float(scale),
                "cross_model_rmse_m": residual["rmse_m"],
            }
        )
    if not estimates:
        raise ValueError("No metric window has at least three matching frame names")

    # A log median is positive, robust to one bad window, and does not consult GT.
    log_scales = np.log([float(item["scale_m_per_model_unit"]) for item in estimates])
    return float(np.exp(np.median(log_scales))), estimates


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reconstruction = pycolmap.Reconstruction(args.sparse_model)
    images = sorted(reconstruction.images.values(), key=lambda image: image.name)
    names = np.asarray([image.name for image in images])
    centers = np.asarray([image.projection_center() for image in images])

    manifest = load_manifest(args.keyframe_manifest)
    source_frames = np.asarray([manifest[name] for name in names])
    image_ids = source_frames + args.source_image_offset
    sample_ids, gt_samples, gps_samples = load_reference(args.ground_truth_csv)
    ground_truth = interpolate_reference(image_ids, sample_ids, gt_samples)
    gps = interpolate_reference(image_ids, sample_ids, gps_samples)

    # Standard monocular ATE: GT supplies only the evaluation-time Sim(3).
    shape_scale, shape_rotation, shape_translation = umeyama(centers, ground_truth)
    shape_aligned = apply_similarity(
        centers, shape_scale, shape_rotation, shape_translation
    )

    # Honest metric test: scale comes from predictions, while GT supplies only the
    # benchmark coordinate-frame SE(3), which cannot change scale.
    metric_scale, scale_estimates = robust_metric_scale(
        names, centers, args.metric_window
    )
    metric_centers = centers * metric_scale
    metric_rotation, metric_translation = rigid_alignment(metric_centers, ground_truth)
    metric_aligned = metric_centers @ metric_rotation.T + metric_translation

    # Deployable absolute georeferencing: both orientation and translation come
    # from noisy onboard GPS, while metric scale remains fixed by MapAnything.
    gps_rotation, gps_translation = rigid_alignment(metric_centers, gps)
    georeferenced = metric_centers @ gps_rotation.T + gps_translation

    report = {
        "protocol": {
            "ground_truth_used_in_reconstruction": False,
            "ground_truth_used_for_metric_scale": False,
            "ground_truth_used_for_georeferencing": False,
            "ground_truth_use": "post-hoc benchmark scoring only",
        },
        "registered_frames": len(images),
        "points3D": reconstruction.num_points3D(),
        "mean_reprojection_error_px": float(
            reconstruction.compute_mean_reprojection_error()
        ),
        "sim3_shape": {
            "evaluation_scale_m_per_model_unit": float(shape_scale),
            "trajectory_error": error_summary(shape_aligned, ground_truth),
        },
        "metric_scale": {
            "estimated_m_per_model_unit": metric_scale,
            "independent_window_estimates": scale_estimates,
            "se3_trajectory_error": error_summary(metric_aligned, ground_truth),
            "segment_length_error": segment_scale_error(metric_aligned, ground_truth),
        },
        "gps_georeferenced": {
            "absolute_error_against_gt": error_summary(georeferenced, ground_truth),
        },
        "raw_gps": {
            "absolute_error_against_gt": error_summary(gps, ground_truth),
        },
    }

    (args.output_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    with (args.output_dir / "trajectory.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "image",
                "image_id",
                "model_x",
                "model_y",
                "model_z",
                "shape_x",
                "shape_y",
                "shape_z",
                "metric_x",
                "metric_y",
                "metric_z",
                "georef_x",
                "georef_y",
                "georef_z",
                "gt_x",
                "gt_y",
                "gt_z",
                "gps_x",
                "gps_y",
                "gps_z",
            ]
        )
        for row in zip(
            names,
            image_ids,
            centers,
            shape_aligned,
            metric_aligned,
            georeferenced,
            ground_truth,
            gps,
        ):
            name, image_id, model, shape, metric, georef, truth, gps_point = row
            writer.writerow(
                [name, image_id, *model, *shape, *metric, *georef, *truth, *gps_point]
            )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
