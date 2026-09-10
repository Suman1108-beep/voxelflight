#!/usr/bin/env python3
"""Export TALO poses and its fused TPS geometry into SIH3D artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


POSE_COLUMNS = [f"pose_{row}{col}" for row in range(4) for col in range(4)]


def load_pose(path: Path) -> np.ndarray:
    with np.load(path) as data:
        pose = np.asarray(data["pose"], dtype=np.float64)
    if pose.shape == (3, 4):
        pose = np.vstack((pose, np.array([[0.0, 0.0, 0.0, 1.0]])))
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"Invalid pose in {path}: shape={pose.shape}")
    return pose


def write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    vertices = np.empty(
        len(points),
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
               ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["red"], vertices["green"], vertices["blue"] = colors.T
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        vertices.tofile(stream)


def export_cloud(talo_dir: Path, output_dir: Path, max_points: int) -> dict:
    source = talo_dir / "tps_fitted_points.npz"
    if not source.exists():
        return {"available": False}
    with np.load(source) as data:
        points = np.asarray(data["world_points_tps"], dtype=np.float32).reshape(-1, 3)
        colors = np.asarray(data["colors"]).reshape(-1, 3)
    valid = np.isfinite(points).all(axis=1) & np.isfinite(colors).all(axis=1)
    points, colors = points[valid], colors[valid]
    original_count = len(points)
    if original_count > max_points:
        # Deterministic uniform sampling keeps the entire flown path represented.
        indices = np.linspace(0, original_count - 1, max_points, dtype=np.int64)
        points, colors = points[indices], colors[indices]
    if colors.dtype.kind == "f" and colors.size and float(np.nanmax(colors)) <= 1.0:
        colors = colors * 255.0
    colors = np.clip(colors, 0, 255).astype(np.uint8)
    destination = output_dir / "reconstruction.ply"
    write_binary_ply(destination, points, colors)
    return {
        "available": True,
        "source_points": original_count,
        "exported_points": len(points),
        "path": str(destination),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("talo_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--camera", default="cam0")
    parser.add_argument("--method", default="TALO TPS + MapAnything")
    parser.add_argument("--expected-frames", type=int)
    parser.add_argument("--max-points", type=int, default=2_000_000)
    args = parser.parse_args()

    if args.max_points <= 0:
        raise ValueError("--max-points must be positive")
    pose_dir = args.talo_dir / args.camera
    pose_files = sorted(
        (path for path in pose_dir.glob("*.npz") if path.stem.isdigit()),
        key=lambda path: int(path.stem),
    )
    if not pose_files:
        raise FileNotFoundError(f"No TALO poses found in {pose_dir}")
    frame_ids = [int(path.stem) for path in pose_files]
    expected_ids = list(range(len(frame_ids)))
    if frame_ids != expected_ids:
        raise ValueError(
            f"TALO pose IDs are not contiguous: got {frame_ids[:5]}...{frame_ids[-5:]}"
        )
    if args.expected_frames is not None and len(frame_ids) != args.expected_frames:
        raise ValueError(
            f"Expected {args.expected_frames} poses but found {len(frame_ids)}"
        )

    poses = np.stack([load_pose(path) for path in pose_files])
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_dir / "trajectory.csv"
    with trajectory_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame_id", "camera_x", "camera_y", "camera_z", *POSE_COLUMNS])
        for frame_id, pose in zip(frame_ids, poses):
            writer.writerow([frame_id, *pose[:3, 3], *pose.reshape(-1)])
    np.savez(output_dir / "trajectory.npz", frame_ids=frame_ids, poses=poses)

    cloud = export_cloud(args.talo_dir, output_dir, args.max_points)
    report = {
        "method": args.method,
        "frames": len(frame_ids),
        "construction_uses_ground_truth": False,
        "construction_uses_oracle_window_placement": False,
        "trajectory_path": str(trajectory_path),
        "point_cloud": cloud,
    }
    (output_dir / "construction_protocol.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
