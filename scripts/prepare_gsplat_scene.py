#!/usr/bin/env python3
"""Create the COLMAP/image layout used by the official gsplat trainer."""

from __future__ import annotations

import argparse
from pathlib import Path

import pycolmap
from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("sparse_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--factor", type=int, default=4)
    args = parser.parse_args()
    if args.factor < 1:
        raise ValueError("factor must be positive")

    reconstruction = pycolmap.Reconstruction(args.sparse_dir)
    names = sorted(image.name for image in reconstruction.images.values())
    images_dir = args.output_dir / "images"
    resized_dir = args.output_dir / f"images_{args.factor}"
    sparse_output = args.output_dir / "sparse" / "0"
    images_dir.mkdir(parents=True, exist_ok=True)
    resized_dir.mkdir(parents=True, exist_ok=True)
    sparse_output.mkdir(parents=True, exist_ok=True)

    for name in names:
        source = (args.image_dir / name).resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        link = images_dir / name
        if not link.exists():
            link.symlink_to(source)
        output = resized_dir / f"{Path(name).stem}.png"
        if not output.exists():
            with Image.open(source) as image:
                size = (round(image.width / args.factor), round(image.height / args.factor))
                image.convert("RGB").resize(size, Image.Resampling.LANCZOS).save(output)

    for source in args.sparse_dir.iterdir():
        if source.is_file():
            destination = sparse_output / source.name
            if not destination.exists():
                destination.symlink_to(source.resolve())
    print(
        f"Prepared {len(names)} images at 1/{args.factor} resolution in {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
