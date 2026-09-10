#!/usr/bin/env python3
"""Fuse metric visual windows with GNSS velocity and barometric altitude.

This is a construction-stage tool: it never reads ground truth.  It builds a
smooth ENU trajectory by integrating onboard GNSS velocity, uses barometric
altitude for the vertical component, registers each local visual window to that
sensor trajectory, and robustly blends overlapping visual observations.  Raw
GPS positions supply only the final absolute UTM translation.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import trimesh
from pyproj import Transformer
from scipy.linalg import solve as solve_positive_definite
from scipy.signal import savgol_filter

from align_metric import apply_similarity, umeyama


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("keyframe_manifest", type=Path)
    parser.add_argument("frame_telemetry", type=Path)
    parser.add_argument("onboard_gps", type=Path)
    parser.add_argument("barometric_pressure", type=Path)
    parser.add_argument("window_predictions", nargs="+", type=Path)
    parser.add_argument(
        "--global-anchor-prediction",
        type=Path,
        help=(
            "Sparse globally consistent visual trajectory. Local windows with "
            "at least three shared frames are registered to these anchors."
        ),
    )
    parser.add_argument("--source-image-offset", type=int, required=True)
    parser.add_argument("--utm-epsg", type=int, default=32632)
    parser.add_argument(
        "--velocity-smoothing-seconds",
        type=float,
        default=15.0,
        help="Savitzky-Golay smoothing duration for GNSS velocity.",
    )
    parser.add_argument(
        "--sensor-weight",
        type=float,
        default=0.5,
        help="Sensor weight used by the legacy pointwise fusion mode.",
    )
    parser.add_argument(
        "--fusion-mode",
        choices=("factor-graph", "pointwise"),
        default="factor-graph",
    )
    parser.add_argument("--sensor-position-weight", type=float, default=0.03)
    parser.add_argument("--sensor-delta-weight", type=float, default=3.0)
    parser.add_argument("--visual-position-weight", type=float, default=0.03)
    parser.add_argument("--visual-delta-weight", type=float, default=3.0)
    parser.add_argument(
        "--visual-delta-lags",
        type=int,
        nargs="+",
        default=(1, 3),
        help="Within-window frame separations used as relative-motion factors.",
    )
    parser.add_argument("--max-points", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def clean_row(row: dict[str, str]) -> dict[str, str]:
    return {key.strip(): value.strip() for key, value in row.items() if key}


def load_manifest(path: Path, offset: int) -> tuple[list[str], np.ndarray]:
    rows = list(csv.DictReader(path.open(newline="")))
    names = [row["output_frame"] for row in rows]
    image_ids = np.asarray([int(row["source_frame"]) + offset for row in rows])
    return names, image_ids


def load_frame_samples(
    path: Path, image_ids: np.ndarray, utm_epsg: int
) -> tuple[np.ndarray, np.ndarray]:
    rows = {
        int(row["image_id"]): row
        for row in csv.DictReader(path.open(newline=""))
    }
    missing = sorted(set(image_ids) - rows.keys())
    if missing:
        raise KeyError(f"Missing frame telemetry for image IDs {missing[:10]}")
    selected = [rows[int(image_id)] for image_id in image_ids]
    timestamps_s = np.asarray(
        [float(row["timestamp_us"]) * 1e-6 for row in selected]
    )
    transformer = Transformer.from_crs(4326, utm_epsg, always_xy=True)
    xy = np.asarray(
        [
            transformer.transform(float(row["longitude"]), float(row["latitude"]))
            for row in selected
        ]
    )
    gps_utm = np.column_stack(
        (xy, [float(row["altitude_m"]) for row in selected])
    )
    return timestamps_s, gps_utm


def load_velocities(path: Path, image_ids: np.ndarray) -> np.ndarray:
    rows = {
        int(row["imgid"]): row
        for raw in csv.DictReader(path.open(newline=""))
        for row in [clean_row(raw)]
    }
    missing = sorted(set(image_ids) - rows.keys())
    if missing:
        raise KeyError(f"Missing onboard velocity for image IDs {missing[:10]}")
    return np.asarray(
        [
            [
                float(rows[int(image_id)]["vel_e_m_s"]),
                float(rows[int(image_id)]["vel_n_m_s"]),
                -float(rows[int(image_id)]["vel_d_m_s"]),
            ]
            for image_id in image_ids
        ]
    )


def interpolate_baro(path: Path, timestamps_s: np.ndarray) -> np.ndarray:
    rows = [clean_row(row) for row in csv.DictReader(path.open(newline=""))]
    baro_timestamps = np.asarray(
        [float(row["Timpstemp"]) * 1e-6 for row in rows]
    )
    altitude = np.asarray([float(row["Altitude"]) for row in rows])
    return np.interp(timestamps_s, baro_timestamps, altitude)


def smoothing_window(
    timestamps_s: np.ndarray, duration_s: float, sample_count: int
) -> int:
    if sample_count < 3 or duration_s <= 0:
        return 1
    median_step = float(np.median(np.diff(timestamps_s)))
    requested = max(int(round(duration_s / median_step)), 3)
    if requested % 2 == 0:
        requested += 1
    maximum = sample_count if sample_count % 2 else sample_count - 1
    return min(requested, maximum)


def integrate_sensor_trajectory(
    timestamps_s: np.ndarray,
    velocity_enu: np.ndarray,
    barometric_altitude: np.ndarray,
    smoothing_duration_s: float,
) -> tuple[np.ndarray, int]:
    window = smoothing_window(
        timestamps_s, smoothing_duration_s, len(timestamps_s)
    )
    smoothed = (
        savgol_filter(velocity_enu, window, 2, axis=0)
        if window >= 5
        else velocity_enu
    )
    delta_t = np.diff(timestamps_s)
    integrated = np.zeros_like(smoothed)
    integrated[1:] = np.cumsum(
        0.5 * (smoothed[:-1] + smoothed[1:]) * delta_t[:, None], axis=0
    )
    # Barometric altitude is markedly less noisy than onboard GPS altitude.
    integrated[:, 2] = barometric_altitude - barometric_altitude[0]
    return integrated, window


def edge_weight(index: int, length: int) -> float:
    if length <= 1:
        return 1.0
    normalized = abs((2.0 * index / (length - 1)) - 1.0)
    return float(0.25 + 0.75 * np.cos(0.5 * np.pi * normalized) ** 2)


def path_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def factor_graph_fusion(
    sensor: np.ndarray,
    registered_windows: list[tuple[np.ndarray, np.ndarray]],
    *,
    sensor_position_weight: float,
    sensor_delta_weight: float,
    visual_position_weight: float,
    visual_delta_weight: float,
    visual_delta_lags: tuple[int, ...],
) -> np.ndarray:
    """Jointly solve absolute and relative visual/inertial pose constraints."""
    count = len(sensor)
    normal = np.zeros((count, count), dtype=np.float64)
    right = np.zeros((count, 3), dtype=np.float64)

    def absolute(index: int, target: np.ndarray, weight: float) -> None:
        normal[index, index] += weight
        right[index] += weight * target

    def relative(
        first: int, second: int, delta: np.ndarray, weight: float
    ) -> None:
        normal[first, first] += weight
        normal[second, second] += weight
        normal[first, second] -= weight
        normal[second, first] -= weight
        right[first] -= weight * delta
        right[second] += weight * delta

    # The first pose fixes the otherwise free global translation gauge.
    absolute(0, sensor[0], 1_000.0)
    for index, target in enumerate(sensor):
        absolute(index, target, sensor_position_weight)
    for index in range(count - 1):
        relative(
            index,
            index + 1,
            sensor[index + 1] - sensor[index],
            sensor_delta_weight,
        )

    for indices, visual in registered_windows:
        window_length = len(indices)
        for local_index, global_index in enumerate(indices):
            absolute(
                int(global_index),
                visual[local_index],
                visual_position_weight * edge_weight(local_index, window_length),
            )
        for lag in visual_delta_lags:
            if lag <= 0 or lag >= window_length:
                continue
            for local_index in range(window_length - lag):
                first = int(indices[local_index])
                second = int(indices[local_index + lag])
                confidence = min(
                    edge_weight(local_index, window_length),
                    edge_weight(local_index + lag, window_length),
                )
                relative(
                    first,
                    second,
                    visual[local_index + lag] - visual[local_index],
                    visual_delta_weight * confidence / lag,
                )

    normal.flat[:: count + 1] += 1e-9
    return solve_positive_definite(
        normal, right, assume_a="pos", check_finite=False
    )


def main() -> int:
    start_time = time.perf_counter()
    args = parse_args()
    if args.sensor_weight < 0:
        raise ValueError("--sensor-weight cannot be negative")
    factor_weights = (
        args.sensor_position_weight,
        args.sensor_delta_weight,
        args.visual_position_weight,
        args.visual_delta_weight,
    )
    if any(weight < 0 for weight in factor_weights):
        raise ValueError("Factor-graph weights cannot be negative")
    if any(lag <= 0 for lag in args.visual_delta_lags):
        raise ValueError("Visual delta lags must be positive")
    if args.max_points <= 0:
        raise ValueError("--max-points must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    names, image_ids = load_manifest(
        args.keyframe_manifest, args.source_image_offset
    )
    index_by_image_id = {
        int(image_id): index for index, image_id in enumerate(image_ids)
    }
    timestamps_s, raw_gps_utm = load_frame_samples(
        args.frame_telemetry, image_ids, args.utm_epsg
    )
    velocity_enu = load_velocities(args.onboard_gps, image_ids)
    barometric_altitude = interpolate_baro(
        args.barometric_pressure, timestamps_s
    )
    sensor_relative, filter_window = integrate_sensor_trajectory(
        timestamps_s,
        velocity_enu,
        barometric_altitude,
        args.velocity_smoothing_seconds,
    )

    visual_observations: dict[int, list[tuple[np.ndarray, float]]] = {
        int(image_id): [] for image_id in image_ids
    }
    dense_cloud_sources: list[tuple[Path, np.ndarray, np.ndarray]] = []
    registered_windows: list[tuple[np.ndarray, np.ndarray]] = []
    window_reports: list[dict[str, object]] = []
    global_anchor_lookup: dict[int, np.ndarray] = {}

    if args.global_anchor_prediction:
        path = args.global_anchor_prediction
        with np.load(path) as prediction:
            anchor_ids = prediction["image_ids"].astype(int)
            anchor_centres = prediction["camera_centers"].astype(np.float64)
        missing = sorted(set(anchor_ids) - index_by_image_id.keys())
        if missing:
            raise KeyError(f"{path} contains unknown image IDs {missing[:10]}")
        anchor_indices = np.asarray(
            [index_by_image_id[int(value)] for value in anchor_ids], dtype=int
        )
        anchor_target = sensor_relative[anchor_indices]
        scale, rotation, translation = umeyama(anchor_centres, anchor_target)
        registered = apply_similarity(
            anchor_centres, scale, rotation, translation
        )
        registered_windows.append((anchor_indices, registered))
        global_anchor_lookup = {
            int(image_id): point
            for image_id, point in zip(anchor_ids, registered)
        }
        residuals = np.linalg.norm(registered - anchor_target, axis=1)
        for local_index, (image_id, point) in enumerate(
            zip(anchor_ids, registered)
        ):
            visual_observations[int(image_id)].append(
                (point, edge_weight(local_index, len(anchor_ids)))
            )
        window_reports.append(
            {
                "path": str(path),
                "role": "global visual anchors",
                "registration_target": "sensor trajectory",
                "frames": len(anchor_ids),
                "scale_to_sensor": scale,
                "sensor_fit_rmse_m": float(np.sqrt(np.mean(residuals**2))),
                "sensor_fit_median_m": float(np.median(residuals)),
                "rotation": rotation.tolist(),
                "translation": translation.tolist(),
            }
        )
        cloud_path = path.parent / "points_dense.ply"
        if cloud_path.exists() and cloud_path.stat().st_size > 1_024:
            dense_cloud_sources.append(
                (cloud_path, anchor_centres, anchor_indices)
            )

    for path in sorted(args.window_predictions):
        with np.load(path) as prediction:
            window_ids = prediction["image_ids"].astype(int)
            centres = prediction["camera_centers"].astype(np.float64)
        missing = sorted(set(window_ids) - index_by_image_id.keys())
        if missing:
            raise KeyError(f"{path} contains unknown image IDs {missing[:10]}")
        target_indices = [index_by_image_id[int(value)] for value in window_ids]
        target = sensor_relative[target_indices]
        shared_local_indices = np.asarray(
            [
                index
                for index, image_id in enumerate(window_ids)
                if int(image_id) in global_anchor_lookup
            ],
            dtype=int,
        )
        if len(shared_local_indices) >= 3:
            fit_source = centres[shared_local_indices]
            fit_target = np.asarray(
                [
                    global_anchor_lookup[int(window_ids[index])]
                    for index in shared_local_indices
                ]
            )
            registration_target = "global visual anchors"
        else:
            fit_source = centres
            fit_target = target
            registration_target = "sensor trajectory"
        scale, rotation, translation = umeyama(fit_source, fit_target)
        registered = apply_similarity(
            centres, scale, rotation, translation
        )
        registered_windows.append((np.asarray(target_indices), registered))
        residuals = np.linalg.norm(registered - target, axis=1)
        for local_index, (image_id, point) in enumerate(
            zip(window_ids, registered)
        ):
            visual_observations[int(image_id)].append(
                (point, edge_weight(local_index, len(window_ids)))
            )
        window_reports.append(
            {
                "path": str(path),
                "role": "local metric visual window",
                "registration_target": registration_target,
                "shared_global_anchor_frames": len(shared_local_indices),
                "frames": len(window_ids),
                "scale_to_sensor": scale,
                "sensor_fit_rmse_m": float(np.sqrt(np.mean(residuals**2))),
                "sensor_fit_median_m": float(np.median(residuals)),
                "rotation": rotation.tolist(),
                "translation": translation.tolist(),
            }
        )
        cloud_path = path.parent / "points_dense.ply"
        if cloud_path.exists() and cloud_path.stat().st_size > 1_024:
            dense_cloud_sources.append(
                (cloud_path, centres, np.asarray(target_indices, dtype=int))
            )

    observation_counts: list[int] = []
    pointwise_fused: list[np.ndarray] = []
    for index, image_id in enumerate(image_ids):
        observations = visual_observations[int(image_id)]
        values = [sensor_relative[index], *[value for value, _ in observations]]
        weights = [args.sensor_weight, *[weight for _, weight in observations]]
        if sum(weights) <= 0:
            raise RuntimeError(f"Frame {image_id} has no usable observation")
        pointwise_fused.append(np.average(values, axis=0, weights=weights))
        observation_counts.append(len(observations))
    if args.fusion_mode == "factor-graph":
        fused_relative_array = factor_graph_fusion(
            sensor_relative,
            registered_windows,
            sensor_position_weight=args.sensor_position_weight,
            sensor_delta_weight=args.sensor_delta_weight,
            visual_position_weight=args.visual_position_weight,
            visual_delta_weight=args.visual_delta_weight,
            visual_delta_lags=tuple(args.visual_delta_lags),
        )
    else:
        fused_relative_array = np.asarray(pointwise_fused)

    # Place dense geometry against the final solved trajectory, not against the
    # preliminary sensor path.  This is what removes window seams when global
    # visual anchors improve the pose graph after each window was registered.
    transformed_clouds: list[np.ndarray] = []
    transformed_colors: list[np.ndarray] = []
    dense_alignment_reports: list[dict[str, object]] = []
    for cloud_path, centres, target_indices in dense_cloud_sources:
        scale, rotation, translation = umeyama(
            centres, fused_relative_array[target_indices]
        )
        aligned_centres = apply_similarity(
            centres, scale, rotation, translation
        )
        centre_residuals = np.linalg.norm(
            aligned_centres - fused_relative_array[target_indices], axis=1
        )
        cloud = trimesh.load(cloud_path, process=False)
        points = np.asarray(cloud.vertices)
        if not len(points):
            continue
        transformed_clouds.append(
            apply_similarity(points, scale, rotation, translation)
        )
        transformed_colors.append(np.asarray(cloud.colors)[:, :3])
        dense_alignment_reports.append(
            {
                "path": str(cloud_path),
                "frames": len(target_indices),
                "scale_to_final_trajectory": float(scale),
                "centre_fit_rmse_m": float(
                    np.sqrt(np.mean(centre_residuals**2))
                ),
                "centre_fit_median_m": float(np.median(centre_residuals)),
            }
        )

    # Velocity and barometer already establish ENU orientation and metric scale.
    # A robust translation is the only operation needed for absolute UTM output.
    utm_translation = np.median(
        raw_gps_utm - fused_relative_array, axis=0
    )
    fused_utm = fused_relative_array + utm_translation

    relative_csv = args.output_dir / "trajectory.csv"
    with relative_csv.open("w", newline="") as stream:
        writer = csv.writer(stream)
        pose_columns = [
            f"pose_{row}{column}" for row in range(4) for column in range(4)
        ]
        writer.writerow(
            ["frame_id", "camera_x", "camera_y", "camera_z", *pose_columns]
        )
        for frame_id, point in enumerate(fused_relative_array):
            pose = np.eye(4)
            pose[:3, 3] = point
            writer.writerow([frame_id, *point, *pose.reshape(-1)])

    absolute_csv = args.output_dir / "camera_trajectory_gps_utm.csv"
    with absolute_csv.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "image", "image_id", "pred_x", "pred_y", "pred_z",
                "gps_x", "gps_y", "gps_z", "visual_window_observations",
            ]
        )
        for values in zip(
            names,
            image_ids,
            fused_utm,
            raw_gps_utm,
            observation_counts,
        ):
            name, image_id, prediction, gps, count = values
            writer.writerow([name, image_id, *prediction, *gps, count])

    np.savez_compressed(
        args.output_dir / "trajectory.npz",
        image_names=np.asarray(names),
        image_ids=image_ids,
        timestamps_s=timestamps_s,
        sensor_relative=sensor_relative,
        fused_relative=fused_relative_array,
        fused_utm=fused_utm,
        raw_gps_utm=raw_gps_utm,
    )
    cloud_report: dict[str, object] = {"available": False}
    if transformed_clouds:
        points = np.concatenate(transformed_clouds)
        colors = np.concatenate(transformed_colors)
        source_points = len(points)
        if source_points > args.max_points:
            rng = np.random.default_rng(args.seed)
            selected = rng.choice(source_points, args.max_points, replace=False)
            points, colors = points[selected], colors[selected]
        relative_cloud = trimesh.PointCloud(points, colors=colors)
        relative_cloud.export(args.output_dir / "reconstruction_relative.ply")
        trimesh.PointCloud(points + utm_translation, colors=colors).export(
            args.output_dir / "reconstruction_utm.ply"
        )
        cloud_report = {
            "available": True,
            "source_points": source_points,
            "exported_points": len(points),
            "relative_path": str(args.output_dir / "reconstruction_relative.ply"),
            "utm_path": str(args.output_dir / "reconstruction_utm.ply"),
        }
    report = {
        "method": "metric visual windows + smoothed GNSS velocity + barometric altitude",
        "construction_uses_ground_truth": False,
        "construction_uses_oracle_window_placement": False,
        "coordinate_reference_system": f"EPSG:{args.utm_epsg}",
        "frames": len(image_ids),
        "velocity_smoothing_seconds": args.velocity_smoothing_seconds,
        "velocity_filter_window_samples": filter_window,
        "fusion_mode": args.fusion_mode,
        "sensor_weight": args.sensor_weight,
        "factor_graph": {
            "sensor_position_weight": args.sensor_position_weight,
            "sensor_delta_weight": args.sensor_delta_weight,
            "visual_position_weight": args.visual_position_weight,
            "visual_delta_weight": args.visual_delta_weight,
            "visual_delta_lags": args.visual_delta_lags,
        },
        "sensor_path_length_m": path_length(sensor_relative),
        "fused_path_length_m": path_length(fused_relative_array),
        "utm_translation": utm_translation.tolist(),
        "windows": window_reports,
        "global_anchor_prediction": (
            str(args.global_anchor_prediction)
            if args.global_anchor_prediction
            else None
        ),
        "dense_geometry_alignment": dense_alignment_reports,
        "point_cloud": cloud_report,
        "runtime_seconds": time.perf_counter() - start_time,
    }
    (args.output_dir / "construction_protocol.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
