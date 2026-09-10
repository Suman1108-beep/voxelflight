#!/usr/bin/env python3
"""Metric-align a merged windowed trajectory and point cloud."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("merged_dir", type=Path)
    parser.add_argument("keyframe_manifest", type=Path)
    parser.add_argument("ground_truth_csv", type=Path)
    parser.add_argument("--source-image-offset", type=int, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.merged_dir / "metric"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.keyframe_manifest)

    names: list[str] = []
    centers: list[list[float]] = []
    with (args.merged_dir / "camera_trajectory_model.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            names.append(row["image"])
            centers.append([float(row[axis]) for axis in "xyz"])
    camera_centers = np.asarray(centers)
    source_frames = np.asarray([manifest[name] for name in names])
    image_ids = source_frames + args.source_image_offset

    sample_ids, ground_truth_samples, gps_samples = load_reference(args.ground_truth_csv)
    ground_truth = interpolate_reference(image_ids, sample_ids, ground_truth_samples)
    gps = interpolate_reference(image_ids, sample_ids, gps_samples)
    scale, rotation, translation = umeyama(camera_centers, ground_truth)
    aligned = apply_similarity(camera_centers, scale, rotation, translation)
    gps_scale, gps_rotation, gps_translation = umeyama(camera_centers, gps)
    gps_aligned = apply_similarity(camera_centers, gps_scale, gps_rotation, gps_translation)

    cloud = trimesh.load(args.merged_dir / "points_dense_model.ply", process=False)
    metric_points = apply_similarity(np.asarray(cloud.vertices), scale, rotation, translation)
    trimesh.PointCloud(metric_points, colors=cloud.colors).export(
        output_dir / "points_metric_utm.ply"
    )

    with (output_dir / "camera_trajectory.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["image", "source_frame", "image_id", "pred_x", "pred_y", "pred_z", "gt_x", "gt_y", "gt_z", "gps_x", "gps_y", "gps_z"]
        )
        for name, source_frame, prediction, truth, gps_point in zip(
            names, source_frames, aligned, ground_truth, gps
        ):
            writer.writerow(
                [name, source_frame, source_frame + args.source_image_offset, *prediction, *truth, *gps_point]
            )

    metrics = {
        "frames": len(camera_centers),
        "source_image_offset": args.source_image_offset,
        "gt_alignment": {
            "scale_m_per_model_unit": scale,
            "trajectory_error": error_summary(aligned, ground_truth),
            "segment_length_error": segment_scale_error(aligned, ground_truth),
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
