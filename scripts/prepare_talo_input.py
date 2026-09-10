#!/usr/bin/env python3
"""Create TALO's single-camera input layout without duplicating image data."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--camera", default="cam0")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    if not source_dir.is_dir():
        raise NotADirectoryError(source_dir)
    images = sorted(
        path for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        images = images[: args.limit]
    if not images:
        raise FileNotFoundError(f"No images found in {source_dir}")

    camera_dir = args.output_dir.resolve() / "image" / args.camera
    camera_dir.mkdir(parents=True, exist_ok=True)
    existing = list(camera_dir.iterdir())
    if existing:
        raise FileExistsError(
            f"Refusing to mix inputs in non-empty directory: {camera_dir}"
        )

    digits = max(3, len(str(len(images) - 1)))
    for index, source in enumerate(images):
        destination = camera_dir / f"{index:0{digits}d}.jpg"
        os.symlink(source, destination)

    print(f"Prepared {len(images)} ordered frames in {camera_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
