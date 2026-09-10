#!/usr/bin/env python3
"""Interpolate canonical GPS telemetry onto selected keyframes for fusion."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from pyproj import Transformer


def choose_utm_epsg(latitude: float, longitude: float) -> int:
    zone = min(60, max(1, int((longitude + 180.0) // 6.0) + 1))
    return (32600 if latitude >= 0 else 32700) + zone


def numeric(rows: list[dict[str, str]], name: str) -> np.ndarray:
    values = []
    for row in rows:
        text = (row.get(name) or "").strip()
        values.append(float(text) if text else np.nan)
    return np.asarray(values, dtype=np.float64)


def interpolate_finite(x: np.ndarray, values: np.ndarray, target: np.ndarray) -> np.ndarray:
    finite = np.isfinite(x) & np.isfinite(values)
    if np.count_nonzero(finite) == 0:
        return np.zeros_like(target, dtype=np.float64)
    if np.count_nonzero(finite) == 1:
        return np.full_like(target, values[finite][0], dtype=np.float64)
    order = np.argsort(x[finite])
    source_x = x[finite][order]
    source_values = values[finite][order]
    unique_x, unique_indices = np.unique(source_x, return_index=True)
    return np.interp(target, unique_x, source_values[unique_indices])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("canonical_telemetry", type=Path)
    parser.add_argument("keyframe_manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--utm-epsg", type=int)
    parser.add_argument("--source-image-offset", type=int, default=0)
    args = parser.parse_args()

    telemetry = list(csv.DictReader(args.canonical_telemetry.open(newline="")))
    manifest = list(csv.DictReader(args.keyframe_manifest.open(newline="")))
    if len(telemetry) < 2 or len(manifest) < 2:
        raise ValueError("Telemetry and manifest each need at least two rows")

    telemetry_fields = set(telemetry[0])
    source_field = "source_frame" if "source_frame" in telemetry_fields else "image_id"
    if source_field not in telemetry_fields:
        raise KeyError("Telemetry needs source_frame or image_id")
    source_frames = numeric(telemetry, source_field)
    target_frames = np.asarray(
        [int(row["source_frame"]) for row in manifest], dtype=np.float64
    )
    latitude = interpolate_finite(source_frames, numeric(telemetry, "latitude"), target_frames)
    longitude = interpolate_finite(source_frames, numeric(telemetry, "longitude"), target_frames)
    altitude = interpolate_finite(source_frames, numeric(telemetry, "altitude_m"), target_frames)
    if "timestamp_s" in telemetry_fields:
        source_time = numeric(telemetry, "timestamp_s")
        timestamp_field = "timestamp_s"
    elif "timestamp_us" in telemetry_fields:
        source_time = numeric(telemetry, "timestamp_us") * 1e-6
        timestamp_field = "timestamp_us"
    else:
        raise KeyError("Telemetry needs timestamp_s or timestamp_us")
    timestamp_s = interpolate_finite(source_frames, source_time, target_frames)

    epsg = args.utm_epsg or choose_utm_epsg(
        float(np.median(latitude)), float(np.median(longitude))
    )
    transformer = Transformer.from_crs(4326, epsg, always_xy=True)
    easting, northing = transformer.transform(longitude, latitude)
    position = np.column_stack((easting, northing, altitude))
    if np.any(np.diff(timestamp_s) <= 0):
        raise ValueError("Interpolated keyframe timestamps must increase")
    velocity = np.gradient(position, timestamp_s, axis=0, edge_order=1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_path = args.output_dir / "frame_telemetry.csv"
    with frame_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "image", "image_id", "timestamp_us", "latitude", "longitude",
                "altitude_m", "gps_fix_type", "gps_eph_m", "satellites",
            ]
        )
        for row, frame, timestamp, lat, lon, alt in zip(
            manifest, target_frames, timestamp_s, latitude, longitude, altitude
        ):
            image_id = int(frame) + args.source_image_offset
            writer.writerow(
                [
                    row["output_frame"], image_id, int(round(timestamp * 1e6)),
                    lat, lon, alt, "", "", "",
                ]
            )

    velocity_path = args.output_dir / "OnboardGPS.csv"
    with velocity_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["imgid", "vel_e_m_s", "vel_n_m_s", "vel_d_m_s"])
        for frame, vector in zip(target_frames, velocity):
            writer.writerow(
                [int(frame) + args.source_image_offset, vector[0], vector[1], -vector[2]]
            )

    barometer_path = args.output_dir / "BarometricPressure.csv"
    with barometer_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Timpstemp", "Altitude"])
        for timestamp, alt in zip(timestamp_s, altitude):
            writer.writerow([int(round(timestamp * 1e6)), alt])

    report = {
        "construction_uses_ground_truth": False,
        "coordinate_reference_system": f"EPSG:{epsg}",
        "utm_epsg": epsg,
        "telemetry_samples": len(telemetry),
        "keyframes": len(manifest),
        "source_image_offset": args.source_image_offset,
        "source_frame_field": source_field,
        "timestamp_field": timestamp_field,
        "altitude_available": bool(np.isfinite(numeric(telemetry, "altitude_m")).any()),
        "outputs": {
            "frame_telemetry": str(frame_path),
            "onboard_velocity": str(velocity_path),
            "barometric_altitude": str(barometer_path),
        },
    }
    (args.output_dir / "telemetry_adapter_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
