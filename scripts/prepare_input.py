#!/usr/bin/env python3
"""Prepare an unseen SIH evaluation delivery for offline reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sih3d.ingest import prepare_scene


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Auto-discover and normalize drone video plus GPS telemetry."
    )
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--telemetry", type=Path)
    parser.add_argument("--calibration", type=Path)
    args = parser.parse_args()
    manifest = prepare_scene(
        args.input_root,
        args.output_root,
        video_path=args.video,
        telemetry_path=args.telemetry,
        calibration_path=args.calibration,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

