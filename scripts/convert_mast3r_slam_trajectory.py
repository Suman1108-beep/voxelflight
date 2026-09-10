#!/usr/bin/env python3
"""Convert a MASt3R-SLAM TUM trajectory to the SIH evaluation CSV schema."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_trajectory", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--keyframe-manifest", type=Path)
    parser.add_argument("--output-prediction", type=Path)
    parser.add_argument("--source-image-offset", type=int, default=0)
    parser.add_argument(
        "--timestamp-fps",
        type=float,
        default=30.0,
        help="FPS used by the RGB-files loader when constructing timestamps.",
    )
    args = parser.parse_args()

    if args.timestamp_fps <= 0:
        raise ValueError("--timestamp-fps must be positive")

    rows: list[list[float]] = []
    for line_number, line in enumerate(
        args.input_trajectory.read_text().splitlines(), start=1
    ):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        values = [float(value) for value in line.split()]
        if len(values) != 8:
            raise ValueError(
                f"Expected 8 TUM values on line {line_number}; got {len(values)}"
            )
        rows.append(values)

    if not rows:
        raise ValueError(f"No trajectory poses found in {args.input_trajectory}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "frame_id",
                "camera_x",
                "camera_y",
                "camera_z",
                "qx",
                "qy",
                "qz",
                "qw",
                "timestamp_s",
            ]
        )
        for timestamp, x, y, z, qx, qy, qz, qw in rows:
            frame_id = int(round(timestamp * args.timestamp_fps))
            writer.writerow([frame_id, x, y, z, qx, qy, qz, qw, timestamp])

    if (args.keyframe_manifest is None) != (args.output_prediction is None):
        raise ValueError(
            "--keyframe-manifest and --output-prediction must be supplied together"
        )
    if args.output_prediction is not None:
        manifest = list(csv.DictReader(args.keyframe_manifest.open(newline="")))
        frame_ids = np.asarray(
            [int(round(row[0] * args.timestamp_fps)) for row in rows], dtype=int
        )
        if np.any(frame_ids < 0) or np.any(frame_ids >= len(manifest)):
            raise IndexError("MASt3R-SLAM frame IDs fall outside the manifest")
        image_ids = np.asarray(
            [
                int(manifest[index]["source_frame"]) + args.source_image_offset
                for index in frame_ids
            ],
            dtype=int,
        )
        image_names = np.asarray(
            [manifest[index]["output_frame"] for index in frame_ids]
        )
        camera_centers = np.asarray([row[1:4] for row in rows], dtype=np.float64)
        args.output_prediction.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output_prediction,
            image_ids=image_ids,
            image_names=image_names,
            camera_centers=camera_centers,
            construction_uses_ground_truth=np.asarray(False),
            source=np.asarray("MASt3R-SLAM calibrated global anchors"),
        )
        print(
            f"Wrote {len(rows)} global anchors to {args.output_prediction}"
        )

    print(f"Converted {len(rows)} poses to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
