#!/usr/bin/env python3
"""Build a labeled contact sheet from selected image paths."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--width", type=int, default=480)
    args = parser.parse_args()

    label_height = 34
    opened = [Image.open(path).convert("RGB") for path in args.images]
    ratio = opened[0].height / opened[0].width
    tile_height = round(args.width * ratio)
    rows = math.ceil(len(opened) / args.columns)
    sheet = Image.new(
        "RGB", (args.columns * args.width, rows * (tile_height + label_height)), "#0b1020"
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=22)
    for index, (path, image) in enumerate(zip(args.images, opened)):
        row, column = divmod(index, args.columns)
        x, y = column * args.width, row * (tile_height + label_height)
        sheet.paste(image.resize((args.width, tile_height)), (x, y))
        draw.text((x + 8, y + tile_height + 5), path.name, fill="white", font=font)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, quality=92)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
