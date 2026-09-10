#!/usr/bin/env python3
"""Evaluate MapAnything metric camera poses against Zurich ground truth."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from align_metric import (
    apply_similarity,
    error_summary,
    interpolate_reference,
    load_manifest,
    load_reference,
    segment_scale_error,
    umeyama,
)
from mapanything.models import MapAnything
from mapanything.utils.image import load_images


def rigid_alignment(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_mean, target_mean = source.mean(0), target.mean(0)
    covariance = (target - target_mean).T @ (source - source_mean)
    left, _, right_t = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(left @ right_t))
    rotation = left @ correction @ right_t
    translation = target_mean - rotation @ source_mean
    return rotation, translation


def path_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("keyframe_manifest", type=Path)
    parser.add_argument("ground_truth_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--source-image-offset", type=int, required=True)
    parser.add_argument("--model", default="facebook/map-anything-apache")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
    names = sorted(
        path.name
        for path in args.image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in image_extensions
    )
    views = load_images(str(args.image_dir))
    if len(names) != len(views):
        raise RuntimeError("Image enumeration does not match MapAnything views")
    model = MapAnything.from_pretrained(args.model).to("cuda").eval()
    with torch.inference_mode():
        outputs = model.infer(
            views,
            memory_efficient_inference=True,
            minibatch_size=1,
            use_amp=True,
            amp_dtype="bf16",
            apply_mask=True,
            mask_edges=True,
            apply_confidence_mask=True,
            confidence_percentile=20,
            use_multiview_confidence=True,
        )
    camera_poses = np.asarray(
        [prediction["camera_poses"][0].float().cpu().numpy() for prediction in outputs]
    )
    camera_centers = camera_poses[:, :3, 3]
    metric_scales = np.asarray(
        [float(prediction["metric_scaling_factor"][0].cpu()) for prediction in outputs]
    )

    manifest = load_manifest(args.keyframe_manifest)
    image_ids = np.asarray(
        [manifest[name] + args.source_image_offset for name in names]
    )
    sample_ids, ground_truth_samples, _ = load_reference(args.ground_truth_csv)
    ground_truth = interpolate_reference(image_ids, sample_ids, ground_truth_samples)

    rotation, translation = rigid_alignment(camera_centers, ground_truth)
    rigid_aligned = camera_centers @ rotation.T + translation
    scale, sim_rotation, sim_translation = umeyama(camera_centers, ground_truth)
    similarity_aligned = apply_similarity(
        camera_centers, scale, sim_rotation, sim_translation
    )
    metrics = {
        "frames": len(names),
        "metric_scale_correction_needed": scale,
        "predicted_path_length_m": path_length(camera_centers),
        "ground_truth_path_length_m": path_length(ground_truth),
        "se3_metric_trajectory_error": error_summary(rigid_aligned, ground_truth),
        "sim3_shape_trajectory_error": error_summary(similarity_aligned, ground_truth),
        "segment_length_error": segment_scale_error(rigid_aligned, ground_truth),
        "model_metric_scale_factor_median": float(np.median(metric_scales)),
    }
    np.savez_compressed(
        args.output_dir / "raw_prediction.npz",
        image_names=np.asarray(names),
        image_ids=image_ids,
        camera_centers=camera_centers,
        camera_poses=camera_poses,
        metric_scaling_factors=metric_scales,
    )
    with (args.output_dir / "camera_trajectory_raw.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "image_id", "model_x", "model_y", "model_z", "metric_scale_factor"])
        for name, image_id, prediction, metric_scale in zip(
            names, image_ids, camera_centers, metric_scales
        ):
            writer.writerow([name, image_id, *prediction, metric_scale])
    (args.output_dir / "evaluation.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with (args.output_dir / "camera_trajectory.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["image", "image_id", "pred_x", "pred_y", "pred_z", "gt_x", "gt_y", "gt_z"]
        )
        for name, image_id, prediction, truth in zip(
            names, image_ids, rigid_aligned, ground_truth
        ):
            writer.writerow([name, image_id, *prediction, *truth])
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
