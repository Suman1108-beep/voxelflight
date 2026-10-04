#!/usr/bin/env python3
"""Convert the Zurich Urban MAV release into the SIH canonical input layout."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--video", action="store_true", help="Also encode MP4")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_dir = args.dataset_root / "MAV Images"
    log_dir = args.dataset_root / "Log Files"
    gps_path = log_dir / "OnboardGPS.csv"
    calibration_path = args.dataset_root / "calibration_data.npz"
    images = sorted(image_dir.glob("*.jpg"))
    if not images or not gps_path.is_file() or not calibration_path.is_file():
        raise FileNotFoundError("Expected MAV Images, GPS log, and calibration data")
    if args.fps<=0:raise ValueError('Frame rate must be positive.')
    if args.video and (args.output_root/'flight.mp4').exists():
        raise FileExistsError('Existing video will not be overwritten; choose a new output folder.')

    args.output_root.mkdir(parents=True, exist_ok=True)
    canonical_frames = args.output_root / "frames"
    if not canonical_frames.exists():
        canonical_frames.symlink_to(image_dir.resolve(), target_is_directory=True)

    with gps_path.open(newline="", errors="replace") as handle:
        gps_rows = list(csv.DictReader(handle))
    gps_by_id = {
        int(row[" imgid"].strip() if " imgid" in row else row["imgid"].strip()): row
        for row in gps_rows
        if (row.get(" imgid") or row.get("imgid") or "").strip()
    }

    manifest_rows = []
    for image in images:
        image_id = int(image.stem)
        row = gps_by_id.get(image_id)
        if row is None:
            continue
        cleaned = {key.strip(): value.strip() for key, value in row.items() if key}
        manifest_rows.append(
            {
                "image": image.name,
                "image_id": image_id,
                "timestamp_us": int(cleaned["Timpstemp"]),
                "latitude": float(cleaned["lat"]),
                "longitude": float(cleaned["lon"]),
                "altitude_m": float(cleaned["alt"]),
                "gps_fix_type": int(cleaned["fix_type"]),
                "gps_eph_m": float(cleaned["eph_m"]),
                "satellites": int(cleaned["num_sat"]),
            }
        )

    if not manifest_rows:
        raise RuntimeError("No image IDs could be joined to the GPS log")
    with (args.output_root / "frame_telemetry.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    calibration = np.load(calibration_path)
    camera = {
        "model": "OPENCV",
        "width": 1920,
        "height": 1080,
        "intrinsic_matrix": calibration["intrinsic_matrix"].tolist(),
        "distortion_coefficients": calibration["distCoeff"].reshape(-1).tolist(),
        "rolling_shutter_readout_ms": 30.0,
        "source": str(calibration_path.resolve()),
    }
    (args.output_root / "camera.json").write_text(json.dumps(camera, indent=2) + "\n")

    copied_logs = args.output_root / "telemetry"
    copied_logs.mkdir(exist_ok=True)
    input_logs = {
        "OnboardGPS.csv", "OnboardPose.csv", "BarometricPressure.csv",
        "RawAccel.csv", "RawGyro.csv",
    }
    for source in log_dir.glob("*.csv"):
        # Keep only onboard sensor logs. GroundTruth* and StreetViewGPS are
        # evaluation/reference data, not reconstruction inputs.
        if source.name not in input_logs:
            continue
        target = copied_logs / source.name
        if not target.exists():
            shutil.copy2(source, target)

    if args.video:
        output_video = args.output_root / "flight.mp4"
        image_numbers=[int(image.stem) for image in images]
        if any(b-a!=1 for a,b in zip(image_numbers,image_numbers[1:])):
            raise ValueError('Video encoding requires a contiguous original frame sequence.')
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "warning",
                "-n",
                "-framerate",
                str(args.fps),
                "-start_number",
                str(image_numbers[0]),
                "-i",
                str(image_dir / "%05d.jpg"),
                "-frames:v",
                str(len(images)),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "21",
                "-threads",
                "4",
                "-movflags",
                "+faststart",
                "-pix_fmt",
                "yuv420p",
                str(output_video),
            ],
            check=True,
        )
        # Explicit source-frame mapping for this derived CFR video. Preserve the
        # original sensor timestamps alongside video-relative frame timestamps.
        with (args.output_root/'video_telemetry.csv').open('w',newline='') as stream:
            writer=csv.writer(stream)
            writer.writerow(['source_frame','image','timestamp_s','original_timestamp_us','latitude','longitude','altitude_m'])
            for row in manifest_rows:
                number=row['image_id']-image_numbers[0]
                writer.writerow([number,row['image'],number/args.fps,row['timestamp_us'],row['latitude'],row['longitude'],row['altitude_m']])

    summary = {
        "frames": len(images),
        "frames_with_gps": len(manifest_rows),
        "first_timestamp_us": manifest_rows[0]["timestamp_us"],
        "last_timestamp_us": manifest_rows[-1]["timestamp_us"],
        "duration_s": (
            manifest_rows[-1]["timestamp_us"] - manifest_rows[0]["timestamp_us"]
        )
        / 1_000_000,
        "video_created": args.video,
        "video_provenance": "Derived CFR H.264 from original sequential JPEGs, not an original camera MP4" if args.video else None,
    }
    (args.output_root / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
