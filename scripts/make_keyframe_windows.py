#!/usr/bin/env python3
"""Create overlapping inference windows from an extracted keyframe directory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("keyframe_dir", type=Path)
    parser.add_argument("--window-size", type=int, default=72)
    parser.add_argument("--window-overlap", type=int, default=18)
    args = parser.parse_args()
    step = args.window_size - args.window_overlap
    if args.window_size < 2 or step <= 0:
        raise ValueError("Window size must exceed overlap and be at least two")

    manifest_path = args.keyframe_dir / "manifest.csv"
    rows = list(csv.DictReader(manifest_path.open(newline="")))
    if len(rows) < 2:
        raise ValueError("Keyframe manifest needs at least two frames")
    window_size = min(args.window_size, len(rows))
    maximum_start = max(0, len(rows) - window_size)
    starts = list(range(0, maximum_start + 1, step))
    if starts[-1] != maximum_start:
        starts.append(maximum_start)

    report = []
    for start in starts:
        end = min(start + window_size, len(rows))
        directory = args.keyframe_dir / "windows" / f"w{start:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        for row in rows[start:end]:
            source = (args.keyframe_dir / row["output_frame"]).resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            destination = directory / row["output_frame"]
            if not destination.exists():
                destination.symlink_to(source)
        report.append(
            {
                "name": directory.name,
                "start": start,
                "end": end,
                "frames": end - start,
                "path": str(directory),
            }
        )
    (args.keyframe_dir / "windows.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
