#!/usr/bin/env python3
"""Apply the frozen onboard-GPS transform to a fused observed-surface mesh.

Cached predictions intentionally stay in the model's visual frame. This tool
recovers the exact already-used transform from pre-GPS and saved post-GPS camera
poses, verifies it on every frame, and exports the fused mesh in the same local
UTM-offset coordinates as its source run. No ground-truth reference is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from mac_geometry import transform_points
from mac_reconstruct import export_gis


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recover_transform(pre: np.ndarray, post: np.ndarray):
    """Recover model→GPS-local similarity from paired camera poses, then audit it."""
    if pre.shape != post.shape or pre.ndim != 3 or pre.shape[1:] != (4, 4) or len(pre) < 3:
        raise ValueError("Need at least three paired 4×4 camera poses")
    if not np.isfinite(pre).all() or not np.isfinite(post).all():
        raise ValueError("Non-finite camera pose")
    rotations = post[:, :3, :3] @ np.swapaxes(pre[:, :3, :3], 1, 2)
    rotation = rotations[0]
    if not np.allclose(rotations, rotation, atol=2e-4):
        raise ValueError("Pre/post camera orientations do not share one global rotation")
    x = pre[:, :3, 3]
    y = post[:, :3, 3]
    xc = x - x.mean(axis=0)
    yc = y - y.mean(axis=0)
    denominator = np.sum(xc * xc)
    if denominator <= 1e-8:
        raise ValueError("Camera translation baseline is too small")
    scale = float(np.sum(yc * (xc @ rotation.T)) / denominator)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid recovered metric scale")
    translation = y.mean(axis=0) - scale * (x.mean(axis=0) @ rotation.T)
    prediction = transform_points(x, scale, rotation, translation)
    residual = np.linalg.norm(prediction - y, axis=1)
    if residual.max() > 2e-3:
        raise ValueError(f"Saved GPS transform is inconsistent: {residual.max():.4f} m")
    return scale, rotation, translation, float(residual.max())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path,
                        help="Original video run with cached predictions and onboard GPS")
    parser.add_argument("--fusion", required=True, type=Path,
                        help="Fused mesh in the original prediction-local frame")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    start = time.monotonic()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    source_report = json.loads((args.source / "report.json").read_text())
    fusion_report = json.loads((args.fusion / "report.json").read_text())
    if source_report.get("ground_truth_used") is not False or not source_report.get("georeferenced"):
        raise ValueError("Source must be an onboard-GPS-only georeferenced run")
    if fusion_report.get("ground_truth_used") is not False or fusion_report.get("georeferenced"):
        raise ValueError("Fusion must be an un-georeferenced, non-GT geometry run")
    if Path(fusion_report["input"]).resolve() != args.source.resolve():
        raise ValueError("Fusion source does not match the supplied video run")
    for name, expected in source_report["checksums"].items():
        if sha(args.source / name) != expected:
            raise ValueError(f"Source artifact changed: {name}")
    for name, expected in fusion_report["artifact_sha256"].items():
        if sha(args.fusion / name) != expected:
            raise ValueError(f"Fused geometry changed: {name}")

    caches = sorted((args.source / "predictions").glob("frame_*.npz"))
    if len(caches) != source_report["keyframes"]:
        raise ValueError("Missing visual-frame predictions")
    pre = []
    for path in caches:
        with np.load(path) as cached:
            pre.append(cached["pose"])
    with np.load(args.source / "camera_poses.npz") as saved:
        post = saved["camera_to_world"]
    scale, rotation, translation, residual = recover_transform(np.asarray(pre), post)
    match = re.search(r"EPSG:(\d+)", source_report["coordinate_system"])
    if not match:
        raise ValueError("Source coordinate reference is missing its EPSG code")
    geo = dict(origin=np.asarray(source_report["utm_origin"], dtype=float),
               epsg=int(match.group(1)))
    if geo["origin"].shape != (3,):
        raise ValueError("Invalid UTM origin")

    import trimesh
    scene = trimesh.load(args.fusion / "reconstruction_mesh.glb", force="scene", process=False)
    scene.apply_transform(np.block([
        [scale * rotation, translation[:, None]],
        [np.zeros((1, 3)), np.ones((1, 1))],
    ]))
    args.output.mkdir(parents=True, exist_ok=True)
    scene.export(args.output / "reconstruction_mesh.glb")
    cloud = trimesh.load(args.fusion / "pointcloud.ply", process=False)
    points = transform_points(np.asarray(cloud.vertices), scale, rotation, translation)
    colors = np.asarray(cloud.colors)[:, :3].astype(np.uint8)
    trimesh.PointCloud(points, colors=colors).export(args.output / "pointcloud.ply")
    gis = export_gis(args.output, points, colors, geo)
    artifacts = ["reconstruction_mesh.glb", "pointcloud.ply", *gis]
    result = dict(
        schema="voxelflight.georeferenced-refusion.v1",
        source=str(args.source.resolve()), fusion=str(args.fusion.resolve()),
        method="Exact frozen onboard-GPS camera transform applied to fused mesh",
        scale=scale, rotation=rotation.tolist(), translation=translation.tolist(),
        pose_transform_max_residual_m=residual,
        coordinate_system=source_report["coordinate_system"],
        utm_origin=geo["origin"].tolist(), horizontal_epsg=geo["epsg"],
        georeferenced=True, ground_truth_used=False,
        independent_surface_accuracy_verified=False,
        absolute_position_accuracy_verified=False,
        runtime_s=time.monotonic()-start,
        artifacts=artifacts,
        source_report_sha256=sha(args.source / "report.json"),
        fusion_report_sha256=sha(args.fusion / "report.json"),
        artifact_sha256={name:sha(args.output / name) for name in artifacts},
        limitations=[
            "This transfers the original noisy-GPS transform; it cannot improve absolute accuracy.",
            "Fused surfaces remain partial and may contain detached fragments.",
            "A coordinate label and GPS fit are not independent surface validation.",
        ],
    )
    (args.output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in (
        "runtime_s", "pose_transform_max_residual_m", "horizontal_epsg", "artifacts")}, indent=2))


if __name__ == "__main__":
    main()
