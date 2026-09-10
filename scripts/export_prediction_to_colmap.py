#!/usr/bin/env python3
"""Export predicted camera poses and geometry as a COLMAP text model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_npz", type=Path)
    parser.add_argument("point_cloud", type=Path)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--max-points", type=int, default=80_000)
    parser.add_argument(
        "--prediction-size",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        help="Network image size for predictions that predate stored image_sizes.",
    )
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    prediction = np.load(args.prediction_npz)
    names = [str(value) for value in prediction["image_names"]]
    poses = np.asarray(prediction["camera_poses"], dtype=np.float64)
    intrinsics = np.asarray(prediction["intrinsics"], dtype=np.float64)
    if "image_sizes" in prediction:
        prediction_sizes = np.asarray(prediction["image_sizes"], dtype=np.float64)
    elif args.prediction_size:
        prediction_sizes = np.tile(
            np.asarray(args.prediction_size[::-1], dtype=np.float64), (len(names), 1)
        )
    else:
        raise ValueError(
            "Prediction does not contain image_sizes; pass --prediction-size WIDTH HEIGHT"
        )
    if not (len(names) == len(poses) == len(intrinsics)):
        raise ValueError("Image, pose and intrinsic counts do not match")

    model = trimesh.load(args.point_cloud, process=False)
    points = np.asarray(model.vertices, dtype=np.float64)
    colors = np.asarray(model.colors[:, :3], dtype=np.uint8)
    finite = np.isfinite(points).all(axis=1)
    points, colors = points[finite], colors[finite]
    if len(points) > args.max_points:
        selected = np.random.default_rng(args.seed).choice(
            len(points), args.max_points, replace=False
        )
        points, colors = points[selected], colors[selected]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    camera_lines = ["# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]"]
    image_lines = [
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "# POINTS2D[] as (X, Y, POINT3D_ID)",
    ]
    for index, (name, camera_to_world, intrinsic, prediction_size) in enumerate(
        zip(names, poses, intrinsics, prediction_sizes), start=1
    ):
        path = args.image_dir / name
        if not path.exists():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            width, height = image.size
        intrinsic = intrinsic.copy()
        prediction_height, prediction_width = prediction_size
        intrinsic[0, :] *= width / prediction_width
        intrinsic[1, :] *= height / prediction_height
        camera_lines.append(
            f"{index} PINHOLE {width} {height} "
            f"{intrinsic[0, 0]:.12g} {intrinsic[1, 1]:.12g} "
            f"{intrinsic[0, 2]:.12g} {intrinsic[1, 2]:.12g}"
        )
        world_to_camera = np.linalg.inv(camera_to_world)
        qx, qy, qz, qw = Rotation.from_matrix(
            world_to_camera[:3, :3]
        ).as_quat()
        tx, ty, tz = world_to_camera[:3, 3]
        image_lines.extend(
            [
                f"{index} {qw:.12g} {qx:.12g} {qy:.12g} {qz:.12g} "
                f"{tx:.12g} {ty:.12g} {tz:.12g} {index} {name}",
                "",
            ]
        )

    point_lines = ["# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]"]
    for index, (point, color) in enumerate(zip(points, colors), start=1):
        point_lines.append(
            f"{index} {point[0]:.9g} {point[1]:.9g} {point[2]:.9g} "
            f"{int(color[0])} {int(color[1])} {int(color[2])} 0"
        )

    (args.output_dir / "cameras.txt").write_text("\n".join(camera_lines) + "\n")
    (args.output_dir / "images.txt").write_text("\n".join(image_lines) + "\n")
    (args.output_dir / "points3D.txt").write_text("\n".join(point_lines) + "\n")
    report = {
        "construction_uses_ground_truth": False,
        "frames": len(names),
        "initial_gaussians": len(points),
        "pose_source": str(args.prediction_npz),
        "geometry_source": str(args.point_cloud),
    }
    (args.output_dir / "export_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
