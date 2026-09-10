#!/usr/bin/env python3
"""Create a VGGT camera/depth reconstruction and COLMAP-compatible export."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pycolmap
import torch
import torch.nn.functional as functional
import trimesh

from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap_wo_track
from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--selection", choices=("even", "contiguous"), default="even"
    )
    parser.add_argument("--confidence-percentile", type=float, default=40.0)
    parser.add_argument("--max-ply-points", type=int, default=1_000_000)
    parser.add_argument("--max-colmap-points", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def evenly_select(paths: list[Path], maximum: int) -> list[Path]:
    if len(paths) <= maximum:
        return paths
    indices = np.linspace(0, len(paths) - 1, maximum).round().astype(int)
    return [paths[index] for index in indices]


def valid_content_mask(coords: np.ndarray, height: int, width: int) -> np.ndarray:
    mask = np.zeros((len(coords), height, width), dtype=bool)
    for index, (x1, y1, x2, y2, _, _) in enumerate(coords):
        left = max(0, int(np.floor(x1)))
        top = max(0, int(np.floor(y1)))
        right = min(width, int(np.ceil(x2)))
        bottom = min(height, int(np.ceil(y2)))
        mask[index, top:bottom, left:right] = True
    return mask


def rescale_reconstruction(
    reconstruction: pycolmap.Reconstruction,
    image_paths: list[Path],
    original_coords: np.ndarray,
    model_size: int,
) -> None:
    for image_id in reconstruction.images:
        image = reconstruction.images[image_id]
        camera = reconstruction.cameras[image.camera_id]
        image.name = image_paths[image_id - 1].name
        real_width, real_height = original_coords[image_id - 1, -2:]
        resize_ratio = max(real_width, real_height) / model_size
        parameters = camera.params.copy() * resize_ratio
        parameters[-2:] = np.array([real_width, real_height]) / 2
        camera.params = parameters
        camera.width = int(real_width)
        camera.height = int(real_height)
        top_left = original_coords[image_id - 1, :2]
        for point in image.points2D:
            point.xy = (point.xy - top_left) * resize_ratio


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not 0 <= args.confidence_percentile < 100:
        raise ValueError("confidence-percentile must be in [0, 100)")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    paths = sorted(
        path
        for path in args.image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if args.selection == "contiguous":
        paths = paths[args.start_index : args.start_index + args.max_frames]
    else:
        if args.start_index:
            raise ValueError("--start-index is only valid with --selection contiguous")
        paths = evenly_select(paths, args.max_frames)
    if len(paths) < 2:
        raise RuntimeError("At least two input images are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    dtype = torch.bfloat16
    model_size = 518
    images, original_coords_tensor = load_and_preprocess_images_square(
        [str(path) for path in paths], target_size=model_size
    )
    images = images.to(device)
    original_coords = original_coords_tensor.numpy()

    model_started = time.perf_counter()
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    load_seconds = time.perf_counter() - model_started
    inference_started = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        batched = images[None]
        aggregated_tokens, patch_start_index = model.aggregator(batched)
        pose_encoding = model.camera_head(aggregated_tokens)[-1]
        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            pose_encoding, batched.shape[-2:]
        )
        depth, confidence = model.depth_head(
            aggregated_tokens, batched, patch_start_index
        )
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started

    extrinsics_np = extrinsics.squeeze(0).float().cpu().numpy()
    intrinsics_np = intrinsics.squeeze(0).float().cpu().numpy()
    depth_np = depth.squeeze(0).float().cpu().numpy()
    confidence_np = confidence.squeeze(0).float().cpu().numpy()
    points = unproject_depth_map_to_point_map(
        depth_np, extrinsics_np, intrinsics_np
    )
    colors = (
        functional.interpolate(
            images,
            size=(model_size, model_size),
            mode="bilinear",
            align_corners=False,
        )
        .mul(255)
        .byte()
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )

    valid = valid_content_mask(original_coords, model_size, model_size)
    valid &= np.isfinite(points).all(axis=-1)
    valid &= np.isfinite(depth_np[..., 0]) & (depth_np[..., 0] > 0)
    threshold = float(np.percentile(confidence_np[valid], args.confidence_percentile))
    confident = valid & (confidence_np >= threshold)

    dense_mask = randomly_limit_trues(confident.copy(), args.max_ply_points)
    dense_points = points[dense_mask]
    dense_colors = colors[dense_mask]
    trimesh.PointCloud(dense_points, colors=dense_colors).export(
        args.output_dir / "points_dense.ply"
    )

    sparse_mask = randomly_limit_trues(confident.copy(), args.max_colmap_points)
    pixel_coordinates = create_pixel_coordinate_grid(
        len(paths), model_size, model_size
    )
    reconstruction = batch_np_matrix_to_pycolmap_wo_track(
        points[sparse_mask],
        pixel_coordinates[sparse_mask],
        colors[sparse_mask],
        extrinsics_np,
        intrinsics_np,
        np.array([model_size, model_size]),
        shared_camera=False,
        camera_type="PINHOLE",
    )
    rescale_reconstruction(reconstruction, paths, original_coords, model_size)
    sparse_dir = args.output_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)
    reconstruction.write(sparse_dir)
    trimesh.PointCloud(
        points[sparse_mask], colors=colors[sparse_mask]
    ).export(sparse_dir / "points.ply")

    np.savez_compressed(
        args.output_dir / "geometry.npz",
        extrinsics=extrinsics_np,
        intrinsics=intrinsics_np,
        depth=depth_np,
        confidence=confidence_np,
        original_coords=original_coords,
        image_names=np.array([path.name for path in paths]),
    )
    report = {
        "frames": len(paths),
        "selection": args.selection,
        "start_index": args.start_index,
        "input_first": paths[0].name,
        "input_last": paths[-1].name,
        "model_load_seconds": round(load_seconds, 3),
        "inference_seconds": round(inference_seconds, 3),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "confidence_percentile": args.confidence_percentile,
        "confidence_threshold": threshold,
        "dense_points": int(dense_mask.sum()),
        "colmap_points": int(sparse_mask.sum()),
    }
    (args.output_dir / "reconstruction_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
