#!/usr/bin/env python3
"""Create evenly sampled keyframes and overlapping windows from image frames."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("frame_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--keyframes", type=int, default=180)
    parser.add_argument("--window-size", type=int, default=72)
    parser.add_argument("--window-overlap", type=int, default=18)
    args = parser.parse_args()
    if args.end_frame <= args.start_frame:
        raise ValueError("--end-frame must be greater than --start-frame")
    if args.keyframes < 2:
        raise ValueError("--keyframes must be at least two")
    step = args.window_size - args.window_overlap
    if step <= 0:
        raise ValueError("Window overlap must be smaller than window size")

    frame_ids = np.linspace(
        args.start_frame, args.end_frame, args.keyframes, dtype=int
    )
    if len(np.unique(frame_ids)) != len(frame_ids):
        raise ValueError("Frame range is too short for the requested keyframes")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for output_index, frame_id in enumerate(frame_ids):
        source = args.frame_dir / f"{frame_id:05d}.jpg"
        if not source.is_file():
            raise FileNotFoundError(source)
        name = f"frame_{output_index:05d}.jpg"
        destination = args.output_dir / name
        if not destination.exists():
            destination.symlink_to(source.resolve())
        rows.append(
            {
                "output_frame": name,
                "source_frame": frame_id,
                "timestamp_s": "",
                "sharpness": "",
                "brightness": "",
                "exposure_score": "",
                "quality_score": "",
            }
        )
    with (args.output_dir / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    maximum_start = max(0, args.keyframes - args.window_size)
    starts = list(range(0, maximum_start + 1, step))
    if starts[-1] != maximum_start:
        starts.append(maximum_start)
    windows: list[dict[str, int]] = []
    for start in starts:
        end = min(start + args.window_size, args.keyframes)
        window = args.output_dir / "windows" / f"w{start:03d}"
        window.mkdir(parents=True, exist_ok=True)
        for index in range(start, end):
            name = f"frame_{index:05d}.jpg"
            destination = window / name
            if not destination.exists():
                destination.symlink_to((args.output_dir / name).resolve())
        windows.append({"start": start, "end": end, "frames": end - start})

    report = {
        "source": str(args.frame_dir.resolve()),
        "start_frame": int(frame_ids[0]),
        "end_frame": int(frame_ids[-1]),
        "keyframes": len(frame_ids),
        "windows": windows,
    }
    (args.output_dir / "sequence.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
