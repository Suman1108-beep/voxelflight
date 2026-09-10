#!/usr/bin/env python3
"""Extract sharp, evenly distributed frames from a single-pass drone video."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Candidate:
    source_frame: int
    timestamp_s: float
    sharpness: float
    brightness: float
    exposure_score: float
    quality_score: float
    jpeg: bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Divide a video into temporal bins and retain the highest-quality "
            "frame in each bin. This preserves flight coverage while rejecting blur."
        )
    )
    parser.add_argument("video", type=Path, help="Input MP4/MOV drone video")
    parser.add_argument("output", type=Path, help="Output keyframe directory")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--candidates-per-bin", type=int, default=6)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--minimum-exposure-score",
        type=float,
        default=0.15,
        help="Reject almost-black/white candidates when a better one exists",
    )
    return parser.parse_args()


def score_frame(frame: np.ndarray) -> tuple[float, float, float, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thumb_scale = min(1.0, 960.0 / max(gray.shape))
    if thumb_scale < 1.0:
        gray = cv2.resize(
            gray,
            dsize=None,
            fx=thumb_scale,
            fy=thumb_scale,
            interpolation=cv2.INTER_AREA,
        )
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    exposure_score = max(0.0, 1.0 - abs(brightness - 127.5) / 127.5)
    quality_score = math.log1p(sharpness) * (0.35 + 0.65 * exposure_score)
    return sharpness, brightness, exposure_score, quality_score


def main() -> int:
    args = parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if args.max_frames < 2 or args.candidates_per_bin < 1:
        raise ValueError("max-frames must be >= 2 and candidates-per-bin >= 1")

    args.output.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open: {args.video}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total <= 0 or not np.isfinite(fps) or fps <= 0:
        raise RuntimeError("Video has invalid frame-count or FPS metadata")

    bin_count = min(args.max_frames, total)
    stride = max(1, total // (bin_count * args.candidates_per_bin))
    selected: list[Candidate | None] = [None] * bin_count
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]

    frame_index = 0
    decoded = 0
    while frame_index < total:
        if not cap.grab():
            break
        if frame_index % stride == 0:
            ok, frame = cap.retrieve()
            if ok and frame is not None:
                decoded += 1
                bin_index = min(bin_count - 1, frame_index * bin_count // total)
                sharpness, brightness, exposure, quality = score_frame(frame)
                current = selected[bin_index]
                eligible = exposure >= args.minimum_exposure_score
                current_eligible = (
                    current is not None
                    and current.exposure_score >= args.minimum_exposure_score
                )
                should_replace = current is None or (
                    (eligible and not current_eligible)
                    or (eligible == current_eligible and quality > current.quality_score)
                )
                if should_replace:
                    encoded_ok, encoded = cv2.imencode(".jpg", frame, encode_params)
                    if encoded_ok:
                        selected[bin_index] = Candidate(
                            source_frame=frame_index,
                            timestamp_s=frame_index / fps,
                            sharpness=sharpness,
                            brightness=brightness,
                            exposure_score=exposure,
                            quality_score=quality,
                            jpeg=encoded.tobytes(),
                        )
        frame_index += 1
    cap.release()

    rows: list[dict[str, object]] = []
    for output_index, candidate in enumerate(x for x in selected if x is not None):
        name = f"frame_{output_index:05d}.jpg"
        (args.output / name).write_bytes(candidate.jpeg)
        rows.append(
            {
                "output_frame": name,
                "source_frame": candidate.source_frame,
                "timestamp_s": f"{candidate.timestamp_s:.6f}",
                "sharpness": f"{candidate.sharpness:.4f}",
                "brightness": f"{candidate.brightness:.4f}",
                "exposure_score": f"{candidate.exposure_score:.6f}",
                "quality_score": f"{candidate.quality_score:.6f}",
            }
        )

    if not rows:
        raise RuntimeError("No frames could be decoded from the video")

    with (args.output / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "input_video": str(args.video.resolve()),
        "source_frame_count": total,
        "source_fps": fps,
        "source_width": width,
        "source_height": height,
        "duration_s": total / fps,
        "candidate_stride": stride,
        "candidates_decoded": decoded,
        "keyframes_written": len(rows),
    }
    (args.output / "video_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

