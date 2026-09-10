#!/usr/bin/env python3
"""Evaluate a VGGT-SLAM trajectory against timestamped metric ground truth."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from align_metric import (
    apply_similarity,
    error_summary,
    interpolate_reference,
    load_reference,
    segment_scale_error,
    umeyama,
)


def path_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory_csv", type=Path)
    parser.add_argument("keyframe_manifest", type=Path)
    parser.add_argument("ground_truth_csv", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--source-image-offset", type=int, default=0)
    args = parser.parse_args()

    trajectory_rows = list(csv.DictReader(args.trajectory_csv.open(newline="")))
    manifest_rows = list(csv.DictReader(args.keyframe_manifest.open(newline="")))
    source_by_output = {
        index: int(row["source_frame"]) + args.source_image_offset
        for index, row in enumerate(manifest_rows)
    }
    frame_ids = np.asarray([int(round(float(row["frame_id"]))) for row in trajectory_rows])
    missing = sorted(set(frame_ids) - source_by_output.keys())
    if missing:
        raise KeyError(f"Trajectory contains unknown output frame IDs: {missing[:10]}")
    source_ids = np.asarray([source_by_output[frame_id] for frame_id in frame_ids])
    prediction = np.asarray(
        [
            [float(row["camera_x"]), float(row["camera_y"]), float(row["camera_z"])]
            for row in trajectory_rows
        ]
    )
    reference_ids, reference_points, _ = load_reference(args.ground_truth_csv)
    ground_truth = interpolate_reference(source_ids, reference_ids, reference_points)

    scale, rotation, translation = umeyama(prediction, ground_truth)
    aligned = apply_similarity(prediction, scale, rotation, translation)
    residual = np.linalg.norm(aligned - ground_truth, axis=1)
    official_limit_m = 1.0
    report = {
        "frames": len(frame_ids),
        "predicted_path_length_model_units": path_length(prediction),
        "ground_truth_path_length_m": path_length(ground_truth),
        "scale_m_per_model_unit": scale,
        "trajectory_shape_error_after_sim3": error_summary(aligned, ground_truth),
        "segment_length_error_after_sim3": segment_scale_error(aligned, ground_truth),
        "frames_within_1m_after_sim3": float(np.mean(residual <= official_limit_m)),
        "official_spatial_limit_m": official_limit_m,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

