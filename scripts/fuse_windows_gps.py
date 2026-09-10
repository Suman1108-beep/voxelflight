#!/usr/bin/env python3
"""Fuse overlapping VGGT windows in GPS/UTM coordinates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pycolmap
import trimesh
from pyproj import Transformer

from align_metric import apply_similarity, load_manifest, umeyama


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("keyframe_manifest", type=Path)
    parser.add_argument("frame_telemetry", type=Path)
    parser.add_argument("window_dirs", nargs="+", type=Path)
    parser.add_argument("--source-image-offset", type=int, required=True)
    parser.add_argument("--utm-epsg", type=int, default=32632)
    parser.add_argument("--max-points", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def load_gps(path: Path, epsg: int) -> dict[int, np.ndarray]:
    transformer = Transformer.from_crs(4326, epsg, always_xy=True)
    result: dict[int, np.ndarray] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            longitude, latitude = float(row["longitude"]), float(row["latitude"])
            easting, northing = transformer.transform(longitude, latitude)
            result[int(row["image_id"])] = np.asarray(
                [easting, northing, float(row["altitude_m"])]
            )
    return result


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.keyframe_manifest)
    gps_by_image = load_gps(args.frame_telemetry, args.utm_epsg)
    rng = np.random.default_rng(args.seed)
    camera_observations: dict[str, list[np.ndarray]] = {}
    point_sets: list[np.ndarray] = []
    color_sets: list[np.ndarray] = []
    reports: list[dict[str, object]] = []

    for window_dir in args.window_dirs:
        reconstruction = pycolmap.Reconstruction(window_dir / "sparse")
        images = [reconstruction.images[key] for key in sorted(reconstruction.images)]
        names = [image.name for image in images]
        centers = np.asarray([image.projection_center() for image in images])
        image_ids = [manifest[name] + args.source_image_offset for name in names]
        gps = np.asarray([gps_by_image[image_id] for image_id in image_ids])
        scale, rotation, translation = umeyama(centers, gps)
        metric_centers = apply_similarity(centers, scale, rotation, translation)
        residual = np.linalg.norm(metric_centers - gps, axis=1)
        for name, center in zip(names, metric_centers):
            camera_observations.setdefault(name, []).append(center)

        cloud = trimesh.load(window_dir / "points_dense.ply", process=False)
        point_sets.append(
            apply_similarity(np.asarray(cloud.vertices), scale, rotation, translation)
        )
        color_sets.append(np.asarray(cloud.colors))
        reports.append(
            {
                "window": str(window_dir),
                "cameras": len(names),
                "scale_m_per_model_unit": scale,
                "gps_fit_rmse_m": float(np.sqrt(np.mean(residual**2))),
                "gps_fit_median_m": float(np.median(residual)),
            }
        )

    points, colors = np.concatenate(point_sets), np.concatenate(color_sets)
    if len(points) > args.max_points:
        chosen = rng.choice(len(points), args.max_points, replace=False)
        points, colors = points[chosen], colors[chosen]
    trimesh.PointCloud(points, colors=colors).export(args.output_dir / "points_gps_utm.ply")

    with (args.output_dir / "camera_trajectory_gps_utm.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["image", "image_id", "pred_x", "pred_y", "pred_z", "gps_x", "gps_y", "gps_z", "window_observations"]
        )
        for name in sorted(camera_observations):
            image_id = manifest[name] + args.source_image_offset
            center = np.mean(camera_observations[name], axis=0)
            writer.writerow(
                [name, image_id, *center, *gps_by_image[image_id], len(camera_observations[name])]
            )

    report = {
        "coordinate_reference_system": f"EPSG:{args.utm_epsg}",
        "windows": reports,
        "unique_cameras": len(camera_observations),
        "fused_points": len(points),
    }
    (args.output_dir / "gps_fusion_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
