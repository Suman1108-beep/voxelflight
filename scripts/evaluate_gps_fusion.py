#!/usr/bin/env python3
"""Evaluate GPS-fused camera centers against Zurich metric ground truth."""

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory_csv", type=Path)
    parser.add_argument("ground_truth_csv", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument(
        "--image-id-offset",
        type=int,
        default=0,
        help="Add this source-video frame offset before reference interpolation.",
    )
    args = parser.parse_args()

    rows = list(csv.DictReader(args.trajectory_csv.open(newline="")))
    image_ids = np.asarray(
        [int(row["image_id"]) + args.image_id_offset for row in rows]
    )
    prediction = np.asarray(
        [[float(row[f"pred_{axis}"]) for axis in "xyz"] for row in rows]
    )
    gps = np.asarray([[float(row[f"gps_{axis}"]) for axis in "xyz"] for row in rows])
    sample_ids, ground_truth_samples, _ = load_reference(args.ground_truth_csv)
    ground_truth = interpolate_reference(image_ids, sample_ids, ground_truth_samples)

    scale, rotation, translation = umeyama(prediction, ground_truth)
    shape_aligned = apply_similarity(prediction, scale, rotation, translation)
    gps_scale, gps_rotation, gps_translation = umeyama(gps, ground_truth)
    gps_shape_aligned = apply_similarity(gps, gps_scale, gps_rotation, gps_translation)
    metrics = {
        "evaluation_only": True,
        "construction_uses_ground_truth": False,
        "image_id_offset": args.image_id_offset,
        "frames": len(rows),
        "absolute_georeferencing_error": error_summary(prediction, ground_truth),
        "trajectory_shape_error_after_global_sim3": error_summary(shape_aligned, ground_truth),
        "segment_length_error": segment_scale_error(shape_aligned, ground_truth),
        "raw_gps_absolute_error": error_summary(gps, ground_truth),
        "raw_gps_shape_limit_after_global_sim3": error_summary(gps_shape_aligned, ground_truth),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(metrics, indent=2) + "\n")
    trajectory_output = args.output_json.with_name("camera_trajectory_evaluation.csv")
    with trajectory_output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["image", "image_id", "pred_x", "pred_y", "pred_z", "gt_x", "gt_y", "gt_z", "gps_x", "gps_y", "gps_z"]
        )
        for row, image_id, prediction_point, truth, gps_point in zip(
            rows, image_ids, prediction, ground_truth, gps
        ):
            writer.writerow(
                [row["image"], image_id, *prediction_point, *truth, *gps_point]
            )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
